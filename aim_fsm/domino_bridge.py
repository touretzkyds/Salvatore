"""Bridge between camera-based domino perception and the Block Game rules engine.

Architecture (do not blur these layers):

    camera -> DominoObj in the world map -> DominoBridge -> DominoBlockGameState -> LLM text

`DominoBridge` only *reports* what it sees. Every judgement about whether a move
is legal comes from :class:`~aim_fsm.domino_game.DominoBlockGameState`; the
bridge never decides that on its own, and neither does the language model.

Two jitter defenses are built in, because a single camera frame is not
trustworthy:

* the bridge reads persistent world-map objects (which already survive several
  frames of data association) rather than raw detections, and
* pip readings are pooled per world-map object and resolved by majority vote
  over a sliding window, so one bad frame cannot flip a tile's identity.
"""

from math import cos, sin, hypot
from collections import Counter, deque, defaultdict

from .worldmap import DominoObj, KNOWN_LENGTH_MM
from .domino_game import Domino

# Two tiles count as joined when an end of one lands within this distance of an
# end of the other. A tile is 48 mm long, so this is a generous but not absurd
# tolerance for hand-placed tiles.
JOIN_TOLERANCE_MM = 30.0

# Tiles further apart than this cannot belong to the same board, whatever the
# pips say. Guards against a stray detection across the table joining the chain.
MAX_JOIN_DISTANCE_MM = 60.0

# How many recent frames feed the pip majority vote, and how many of them must
# agree before the bridge is willing to name a tile.
PIP_VOTE_WINDOW = 7
PIP_VOTE_MIN_AGREEMENT = 3


class TileReading:
    """One domino the camera currently believes is on the table."""

    def __init__(self, obj):
        self.id = obj.id
        self.x = float(obj.pose.x)
        self.y = float(obj.pose.y)
        self.theta = float(obj.pose.theta or 0.0)
        self.is_fallen = bool(getattr(obj, 'is_fallen', False))
        self.is_visible = bool(getattr(obj, 'is_visible', False))
        self.confidence = getattr(obj, 'confidence', None)
        self.face_confidence = getattr(obj, 'face_confidence', None)
        self.pips = None       # (a, b) after voting, or None if not yet certain
        self.domino = None     # Domino built from self.pips, or None
        self.oriented = None   # Domino turned to read left-to-right along the board
        self.length = float(getattr(obj, 'length', KNOWN_LENGTH_MM))

    @property
    def ends(self):
        "World coordinates of the tile's two short ends."
        half = self.length / 2.0
        dx, dy = cos(self.theta) * half, sin(self.theta) * half
        return ((self.x - dx, self.y - dy), (self.x + dx, self.y + dy))

    def end_values(self):
        """Pip value at each end, matching the order of :attr:`ends`.

        The detector reads half faces along the tile's axis, so half_counts[0]
        belongs to the negative-theta end.
        """
        if self.pips is None:
            return (None, None)
        return (self.pips[0], self.pips[1])

    def __repr__(self):
        face = f'{self.pips[0]}-{self.pips[1]}' if self.pips else '?-?'
        state = 'fallen' if self.is_fallen else 'standing'
        return f'<Tile {self.id} {face} {state} at ({self.x:.0f}, {self.y:.0f})>'


class TableReading:
    """Everything the bridge could work out about the table this instant."""

    def __init__(self, board, near_tiles, far_tiles, unreadable):
        self.board = board            # TileReadings ordered along the layout
        self.near_tiles = near_tiles  # loose tiles on the robot's side
        self.far_tiles = far_tiles    # loose tiles on the far (player's) side
        self.unreadable = unreadable  # seen, but pips not yet established

    @property
    def board_dominoes(self):
        "The board as Dominoes, oriented left-to-right for the rules engine."
        return [tile.oriented for tile in self.board if tile.oriented is not None]

    def hand_dominoes(self, side):
        tiles = self.near_tiles if side == 'near' else self.far_tiles
        return [tile.domino for tile in tiles if tile.domino is not None]

    def __repr__(self):
        return (f'<TableReading board={len(self.board)} near={len(self.near_tiles)} '
                f'far={len(self.far_tiles)} unreadable={len(self.unreadable)}>')


class DominoBridge:
    """Turns world-map DominoObjs into rules-engine and LLM-ready descriptions.

    `hand_side_for_player` says which side of the table the human sits on.
    'far' is the normal setup: the robot looks across the table at the player.

    `play_area` is an optional (x, y, radius) in world millimetres marking where
    tiles get played. It is only needed to recognise the *opening* tile: with an
    empty board, a lone tile on the table is geometrically indistinguishable
    from a tile someone set down beside their hand. Once two tiles are joined,
    the layout speaks for itself and the play area is not consulted.
    """

    def __init__(self, robot, hand_side_for_player='far',
                 play_area=None,
                 vote_window=PIP_VOTE_WINDOW,
                 min_agreement=PIP_VOTE_MIN_AGREEMENT):
        self.robot = robot
        self.hand_side_for_player = hand_side_for_player
        self.play_area = play_area
        self.vote_window = int(vote_window)
        self.min_agreement = int(min_agreement)
        self._pip_votes = defaultdict(lambda: deque(maxlen=self.vote_window))
        self._voted_frame = {}
        self.last_reading = None
        # After the rules engine refuses a camera-seen placement, remember the
        # camera layout so WatchDominoTable does not report that same illegal
        # scene forever. Cleared as soon as the camera board changes.
        self._suppressed_play = None  # (rules_board_key, camera_board_key) or None
        # The FSM parks the live game here so anything that only has a robot
        # handle (the world map prompt, for one) can still describe the game.
        self.game_state = None

    # ---------------------------------------------------------------- sensing

    def available(self):
        "True when camera-based domino perception is switched on."
        return getattr(self.robot, 'domino_detector', None) is not None

    def reset(self):
        "Forget accumulated pip votes, e.g. when starting a new game."
        self._pip_votes.clear()
        self._voted_frame.clear()
        self.last_reading = None
        self._suppressed_play = None

    def observe(self):
        """Read the table once and return a :class:`TableReading`.

        Safe to call as often as you like. Votes are counted at most once per
        camera frame, so polling faster than the camera cannot manufacture
        agreement out of a single frame seen several times.
        """
        tiles = []
        for obj in self._domino_objects():
            tile = TileReading(obj)
            self._record_vote(obj)
            tile.pips = self._voted_pips(obj.id)
            if tile.pips is not None:
                tile.domino = Domino(*tile.pips)
            tiles.append(tile)

        readable = [t for t in tiles if t.domino is not None]
        unreadable = [t for t in tiles if t.domino is None]

        board = self._build_board_chain(readable)
        if not board:
            board = self._lone_opening_tile(readable)
        loose = [t for t in readable if t not in board]
        near, far = self._split_by_side(loose, board)

        self.last_reading = TableReading(board, near, far, unreadable)
        return self.last_reading

    def _domino_objects(self):
        world_map = getattr(self.robot, 'world_map', None)
        if world_map is None:
            return []
        snapshot = world_map.snapshot_objects()
        return [obj for obj in snapshot.values()
                if isinstance(obj, DominoObj) and not obj.is_missing]

    def _record_vote(self, obj):
        # The world map stamps each tile with the camera frame it was last read
        # from, so a tile that has not been looked at again does not get a
        # second vote no matter how often the bridge is asked to observe.
        frame = getattr(obj, 'observed_frame', None)
        if frame is None:
            frame = getattr(self.robot, 'frame_count', None)
        if frame is not None and self._voted_frame.get(obj.id) == frame:
            return
        try:
            pips = obj.pips
        except (TypeError, ValueError):
            return
        if pips is None or len(pips) != 2:
            return
        if not all(0 <= p <= 6 for p in pips):
            return
        self._voted_frame[obj.id] = frame
        self._pip_votes[obj.id].append(tuple(pips))

    def _voted_pips(self, obj_id):
        """Majority pip reading for this tile, or None if it is not yet settled.

        A tile stays unnamed until `min_agreement` frames agree on it. Reporting
        a tile the camera has only glimpsed once would defeat the whole point of
        voting, so there is deliberately no fallback for the impatient caller.
        """
        votes = self._pip_votes.get(obj_id)
        if not votes:
            return None
        pips, count = Counter(votes).most_common(1)[0]
        if count < self.min_agreement:
            return None
        return pips

    # ------------------------------------------------------- board geometry

    def _build_board_chain(self, tiles):
        """Find the played layout: the longest run of tiles that both touch
        end-to-end and agree on pips at every joint.

        Requiring a pip match as well as proximity is what separates the board
        from a hand laid out in a neat row, where neighboring tiles touch but
        their facing ends rarely match.
        """
        if len(tiles) < 2:
            return []

        neighbors = defaultdict(list)
        for i, a in enumerate(tiles):
            for b in tiles[i + 1:]:
                joint = self._joint_between(a, b)
                if joint is not None:
                    neighbors[a.id].append(b)
                    neighbors[b.id].append(a)

        best = []
        for start in tiles:
            path = self._walk_chain(start, neighbors)
            if len(path) > len(best):
                best = path
        if len(best) < 2:
            return []
        self._orient_chain(best)
        return best

    def _orient_chain(self, chain):
        """Set tile.oriented so each tile's right end feeds the next tile's left.

        The rules engine reads the board as a left-to-right sequence, so a tile
        whose axis happens to point the other way has to be flipped before the
        board makes sense to it.
        """
        joint = self._joint_between(chain[0], chain[1])
        first = chain[0].domino
        chain[0].oriented = first if first.right == joint else first.flipped()
        for prev, tile in zip(chain, chain[1:]):
            entry = prev.oriented.right
            chain_domino = tile.domino
            tile.oriented = (chain_domino if chain_domino.left == entry
                             else chain_domino.flipped())

    def _lone_opening_tile(self, tiles):
        """The single tile that opens a game, as a one-tile board, or [].

        Requires a configured play area. Without one there is no way to tell an
        opening tile from a tile lying beside somebody's hand, and guessing
        would silently invent a board out of a hand.
        """
        if not tiles or self.play_area is None:
            return []
        inside = [t for t in tiles if self._in_play_area(t)]
        if len(inside) != 1:
            return []
        inside[0].oriented = inside[0].domino
        return inside

    def _in_play_area(self, tile):
        px, py, radius = self.play_area
        return hypot(tile.x - px, tile.y - py) <= radius

    def _joint_between(self, a, b):
        """If tiles a and b are joined end-to-end with matching pips, return
        the shared pip value; otherwise None.
        """
        a_ends, b_ends = a.ends, b.ends
        a_vals, b_vals = a.end_values(), b.end_values()
        best = None
        for i, ap in enumerate(a_ends):
            for j, bp in enumerate(b_ends):
                gap = hypot(ap[0] - bp[0], ap[1] - bp[1])
                if gap > JOIN_TOLERANCE_MM:
                    continue
                if a_vals[i] is None or b_vals[j] is None:
                    continue
                if a_vals[i] != b_vals[j]:
                    continue
                if hypot(a.x - b.x, a.y - b.y) > MAX_JOIN_DISTANCE_MM + a.length:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, a_vals[i])
        return None if best is None else best[1]

    def _walk_chain(self, start, neighbors):
        "Greedily extend a simple path from start in both directions."
        forward = self._extend(start, neighbors, {start.id})
        backward = self._extend(start, neighbors, {start.id} | {t.id for t in forward})
        return list(reversed(backward)) + [start] + forward

    def _extend(self, tile, neighbors, visited):
        chain = []
        current = tile
        while True:
            nxt = next((t for t in neighbors[current.id] if t.id not in visited), None)
            if nxt is None:
                return chain
            visited.add(nxt.id)
            chain.append(nxt)
            current = nxt

    def _split_by_side(self, tiles, board):
        """Split loose tiles into the robot's side and the far side.

        The dividing line runs through the board (or, on an empty table, through
        the midpoint of the tiles) perpendicular to the robot's line of sight.
        """
        if not tiles:
            return [], []
        pose = getattr(self.robot, 'pose', None)
        rx = float(getattr(pose, 'x', 0.0) or 0.0)
        ry = float(getattr(pose, 'y', 0.0) or 0.0)
        rtheta = float(getattr(pose, 'theta', 0.0) or 0.0)
        forward = (cos(rtheta), sin(rtheta))

        def depth(t):
            "Distance from the robot along its forward axis."
            return (t.x - rx) * forward[0] + (t.y - ry) * forward[1]

        reference = board or tiles
        split = sum(depth(t) for t in reference) / len(reference)

        near = [t for t in tiles if depth(t) < split]
        far = [t for t in tiles if depth(t) >= split]
        return near, far

    # ------------------------------------------------- rules-engine interface

    def hand_for(self, reading, who):
        """Dominoes the camera attributes to 'player' or 'opponent'.

        'opponent' is Salvatore, who sits on the robot's own side of the table.
        """
        player_side = self.hand_side_for_player
        robot_side = 'near' if player_side == 'far' else 'far'
        side = player_side if who == 'player' else robot_side
        return reading.hand_dominoes(side)

    def detect_played_tile(self, reading, state):
        """Which tile the camera says was just added to the board.

        Returns (domino, anchor_domino) suitable for
        :meth:`DominoBlockGameState.play_domino`, or None if the board looks
        unchanged or the change is not a single tile at one end.

        This reports a *candidate*. The caller must pass it to `play_domino`,
        which is the only thing entitled to declare it legal or illegal. If the
        rules engine refuses, call :meth:`note_rejected_play` so the same
        placement is not reported again until the table changes.
        """
        known = list(state.board)
        seen = reading.board_dominoes
        camera_key = self._board_key(seen)

        # Drop a stale suppression as soon as the camera no longer shows the
        # refused layout (player took the tile away, or placed a different one).
        if self._suppressed_play is not None:
            rules_key, refused_camera_key = self._suppressed_play
            if (rules_key == self._board_key(known)
                    and refused_camera_key == camera_key):
                return None
            self._suppressed_play = None

        if len(seen) != len(known) + 1:
            return None
        if not known:
            return (seen[0], None) if len(seen) == 1 else None

        # The camera has no way to know which end of the layout the rules engine
        # calls "left", so try reading its chain both ways round.
        for chain in (seen, list(reversed(seen))):
            # A new tile can only have appeared at one end or the other.
            if self._same_layout(chain[1:], known):
                return (chain[0], known[0])
            if self._same_layout(chain[:-1], known):
                return (chain[-1], known[-1])
        return None

    def note_rejected_play(self, domino, state, reading=None):
        """Suppress re-reporting a placement the rules engine just refused.

        Without this, an illegal tile left on the table would be detected on
        every poll and the teaching loop would never move on. `domino` is kept
        in the signature for callers; suppression is keyed on the camera layout.
        """
        reading = reading or self.last_reading
        camera_key = (self._board_key(reading.board_dominoes)
                      if reading is not None else ())
        self._suppressed_play = (self._board_key(state.board), camera_key)

    @staticmethod
    def _domino_key(domino):
        "Orientation-independent identity for a Domino (or None)."
        if domino is None:
            return None
        a, b = int(domino.left), int(domino.right)
        return (a, b) if a <= b else (b, a)

    @classmethod
    def _board_key(cls, board):
        "Orientation-independent identity for a whole board layout."
        return tuple(cls._domino_key(d) for d in board)

    @staticmethod
    def _same_layout(a, b):
        "Compare two runs of tiles, ignoring which way round each tile reads."
        return len(a) == len(b) and all(x == y for x, y in zip(a, b))

    def sync_hands(self, state, reading=None, sides=('player', 'opponent')):
        """Overwrite the given hands in `state` from what the camera sees.

        Returns the sides that were actually updated. A side is left untouched
        when the camera sees nothing there, so a momentary occlusion cannot
        silently empty somebody's hand.
        """
        reading = reading or self.last_reading
        if reading is None:
            return ()
        updated = []
        for who in sides:
            hand = self.hand_for(reading, who)
            if not hand:
                continue
            if who == 'player':
                state.player_hand = list(hand)
            else:
                state.opponent_hand = list(hand)
            updated.append(who)
        return tuple(updated)

    # ------------------------------------------------------------ LLM output

    def describe(self, reading=None, state=None):
        """A plain-language account of the table for the language model.

        Board and hand wording comes from the rules engine's own formatters
        whenever a game state exists, so the model never sees a board that
        disagrees with the referee.
        """
        reading = reading or self.last_reading
        state = state if state is not None else self.game_state
        if reading is None:
            return 'The camera has not looked at the table yet.'

        lines = []
        if state is not None and state.board:
            lines.append(f'Board: {state.format_board()}')
            left, right = state.board_ends()
            lines.append(f'Open ends: {left} and {right}')
        elif reading.board:
            lines.append('Board: ' + ' - '.join(self._fmt(t) for t in reading.board))
        else:
            lines.append('Board: (empty board)')

        if state is not None:
            lines.append(f"Learner's hand: {state.format_hand('player')}")
            lines.append(f'Your hand: {state.format_hand("opponent")}')
        else:
            lines.append('Tiles on the far side: ' + self._fmt_group(reading.far_tiles))
            lines.append('Tiles on your side: ' + self._fmt_group(reading.near_tiles))

        n = len(reading.unreadable)
        if n:
            lines.append(f'You can see {"another tile" if n == 1 else f"{n} more tiles"} '
                         'but cannot make out the pips yet.')
        fallen = sum(1 for t in reading.board + reading.near_tiles + reading.far_tiles
                     if t.is_fallen)
        if fallen:
            lines.append('One tile has fallen over.' if fallen == 1
                         else f'{fallen} tiles have fallen over.')
        return '\n'.join(lines)

    @staticmethod
    def _fmt(tile):
        face = tile.oriented or tile.domino
        return f'[{face.left}|{face.right}]' if face else '[?|?]'

    def _fmt_group(self, tiles):
        if not tiles:
            return '(none)'
        return ', '.join(self._fmt(t) for t in tiles)


def attach_bridge(robot, **kwargs):
    """Return robot.domino_bridge, creating it on first use."""
    bridge = getattr(robot, 'domino_bridge', None)
    if bridge is None:
        bridge = DominoBridge(robot, **kwargs)
        robot.domino_bridge = bridge
    return bridge
