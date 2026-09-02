"""Tests for the vision -> rules-engine bridge.

These run without a robot, a camera, or any model weights: a fake world map
serves hand-placed DominoObj instances, so the geometry, the pip voting, and
the handoff to the rules engine can all be checked on a laptop.
"""

import unittest
from math import pi, cos, sin

from .worldmap import DominoObj, KNOWN_LENGTH_MM, reverse_half_ordering
from .domino_game import DominoBlockGameState, Domino
from .domino_bridge import DominoBridge


class FakePose:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x, self.y, self.theta = x, y, theta


class FakeWorldMap:
    """Stands in for the real world map, including its observation stamping."""

    def __init__(self, robot=None):
        self.objects = {}
        self.robot = robot

    def snapshot_objects(self):
        return dict(self.objects)

    def observe_frame(self):
        """Mimic a camera frame reaching the world map.

        The real world map stamps every tile it re-reads with the current frame
        number; the bridge relies on that to count one vote per observation.
        """
        self.robot.frame_count += 1
        for obj in self.objects.values():
            obj.observed_frame = self.robot.frame_count


class FakeRobot:
    """Just enough robot for the bridge: a pose, a world map, a detector flag."""

    def __init__(self, x=0.0, y=0.0, theta=pi/2):
        self.pose = FakePose(x, y, theta)
        self.frame_count = 0
        self.world_map = FakeWorldMap(self)
        self.domino_detector = object()


def place(world_map, name, pips, x, y, theta=0.0, is_fallen=True):
    "Put a domino on the fake table with its pips already read."
    obj = DominoObj(id=name, x=x, y=y, theta=theta, is_fallen=is_fallen)
    obj.half_counts = tuple(pips)
    obj.face_label = f'{pips[0]}-{pips[1]}'
    obj.is_visible = True
    obj.is_missing = False
    world_map.objects[name] = obj
    return obj


def chain_x(index):
    "Centre x of the index-th tile in a row laid end to end along the x axis."
    return index * KNOWN_LENGTH_MM


class BridgeTestCase(unittest.TestCase):
    """Shared setup: a fake robot whose camera can be advanced a frame."""

    def setUp(self):
        self.robot = FakeRobot()
        self.bridge = DominoBridge(self.robot)

    def observe_frame(self):
        "Advance the camera one frame and let the bridge look at the table."
        self.robot.world_map.observe_frame()
        return self.bridge.observe()

    def settle(self, frames=5):
        "Observe over several frames so the pip votes reach agreement."
        for _ in range(frames):
            reading = self.observe_frame()
        return reading


class BridgeGeometryTests(BridgeTestCase):
    """The bridge should recover the played layout from tile positions."""

    def test_matching_row_becomes_the_board(self):
        wm = self.robot.world_map
        # [3|5] - [5|2] - [2|6] laid end to end, board at y=200.
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)
        place(wm, 'd3', (2, 6), chain_x(2), 200)

        reading = self.settle()
        self.assertEqual(len(reading.board), 3)
        self.assertEqual([str(d) for d in reading.board_dominoes],
                         ['[3|5]', '[5|2]', '[2|6]'])
        self.assertEqual(reading.unreadable, [])

    def test_flipped_tile_is_reoriented_for_the_rules_engine(self):
        wm = self.robot.world_map
        # The middle tile physically points the other way: its axis reads 2-5,
        # so the chain only makes sense once the bridge flips it to 5-2.
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (2, 5), chain_x(1), 200, theta=pi)
        place(wm, 'd3', (2, 6), chain_x(2), 200)

        reading = self.settle()
        self.assertEqual([str(d) for d in reading.board_dominoes],
                         ['[3|5]', '[5|2]', '[2|6]'])

    def test_touching_tiles_that_do_not_match_are_not_a_board(self):
        wm = self.robot.world_map
        # A hand laid out in a tidy row: the tiles touch but the ends disagree.
        place(wm, 'h1', (1, 2), chain_x(0), 200)
        place(wm, 'h2', (4, 6), chain_x(1), 200)
        place(wm, 'h3', (0, 3), chain_x(2), 200)

        reading = self.settle()
        self.assertEqual(reading.board, [])
        self.assertEqual(len(reading.near_tiles) + len(reading.far_tiles), 3)

    def test_distant_tile_does_not_join_the_chain(self):
        wm = self.robot.world_map
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)
        # Pips would match at the 2 end, but this tile is across the table.
        place(wm, 'far', (2, 6), chain_x(1) + 400, 200)

        reading = self.settle()
        self.assertEqual(len(reading.board), 2)

    def test_hands_split_by_side_of_the_board(self):
        wm = self.robot.world_map
        # Robot sits at the origin looking along +y; board sits at y=200.
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)
        place(wm, 'mine', (6, 6), 0, 90)      # near the robot
        place(wm, 'theirs', (1, 1), 0, 320)   # beyond the board

        reading = self.settle()
        self.assertEqual([t.id for t in reading.near_tiles], ['mine'])
        self.assertEqual([t.id for t in reading.far_tiles], ['theirs'])
        self.assertEqual([str(d) for d in self.bridge.hand_for(reading, 'player')],
                         ['[1|1]'])
        self.assertEqual([str(d) for d in self.bridge.hand_for(reading, 'opponent')],
                         ['[6|6]'])

    def test_diagonal_chain(self):
        wm = self.robot.world_map
        theta = pi / 4
        step = KNOWN_LENGTH_MM
        for i, pips in enumerate([(3, 5), (5, 2), (2, 6)]):
            place(wm, f'd{i}', pips,
                  200 + i * step * cos(theta), 200 + i * step * sin(theta),
                  theta=theta)

        reading = self.settle()
        self.assertEqual([str(d) for d in reading.board_dominoes],
                         ['[3|5]', '[5|2]', '[2|6]'])


class PipVotingTests(BridgeTestCase):
    """One bad frame must not be allowed to rename a tile."""

    def setUp(self):
        super().setUp()
        self.bridge.min_agreement = 3
        self.tile = place(self.robot.world_map, 'd1', (3, 5), 0, 200)

    def test_a_single_frame_is_not_enough(self):
        reading = self.observe_frame()
        self.assertEqual(reading.board, [])
        self.assertEqual(len(reading.unreadable), 1)

    def test_agreement_settles_the_reading(self):
        reading = self.settle(3)
        self.assertEqual(len(reading.unreadable), 0)
        self.assertEqual(str(reading.far_tiles[0].domino), '[3|5]')

    def test_outlier_frame_is_outvoted(self):
        self.settle(4)
        self.tile.half_counts = (3, 1)   # one misread frame
        reading = self.observe_frame()
        self.assertEqual(str(reading.far_tiles[0].domino), '[3|5]')

    def test_reset_clears_the_votes(self):
        self.settle(4)
        self.bridge.reset()
        reading = self.observe_frame()
        self.assertEqual(len(reading.unreadable), 1)

    def test_polling_between_camera_frames_does_not_build_agreement(self):
        """Looking at the same frame repeatedly is still only one frame."""
        self.robot.world_map.observe_frame()
        for _ in range(10):
            reading = self.bridge.observe()
        self.assertEqual(len(reading.unreadable), 1)


class RulesHandoffTests(BridgeTestCase):
    """The bridge proposes; only DominoBlockGameState disposes."""

    def test_new_tile_at_the_right_end_is_reported_with_its_anchor(self):
        state = DominoBlockGameState(player_hand=[(2, 6)], opponent_hand=[],
                                     board=[(3, 5), (5, 2)])
        wm = self.robot.world_map
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)
        place(wm, 'd3', (2, 6), chain_x(2), 200)

        played = self.bridge.detect_played_tile(self.settle(), state)
        self.assertIsNotNone(played)
        domino, anchor = played
        self.assertEqual(domino, Domino(2, 6))
        self.assertEqual(anchor, Domino(5, 2))

        # The rules engine, not the bridge, is what accepts the move.
        state.play_domino(domino, anchor, player='player')
        self.assertEqual(state.format_board(), '[3|5] - [5|2] - [2|6]')

    def test_new_tile_at_the_left_end(self):
        state = DominoBlockGameState(player_hand=[(1, 3)], opponent_hand=[],
                                     board=[(3, 5), (5, 2)])
        wm = self.robot.world_map
        place(wm, 'new', (1, 3), chain_x(-1), 200)
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)

        domino, anchor = self.bridge.detect_played_tile(self.settle(), state)
        self.assertEqual(domino, Domino(1, 3))
        self.assertEqual(anchor, Domino(3, 5))
        state.play_domino(domino, anchor, player='player')
        self.assertEqual(state.format_board(), '[1|3] - [3|5] - [5|2]')

    def test_camera_chain_read_backwards_still_matches(self):
        state = DominoBlockGameState(player_hand=[(2, 6)], opponent_hand=[],
                                     board=[(3, 5), (5, 2)])
        wm = self.robot.world_map
        # Same layout, but the tiles are placed so the chain walks the other way.
        place(wm, 'd3', (6, 2), chain_x(0), 200)
        place(wm, 'd2', (2, 5), chain_x(1), 200)
        place(wm, 'd1', (5, 3), chain_x(2), 200)

        domino, anchor = self.bridge.detect_played_tile(self.settle(), state)
        self.assertEqual(domino, Domino(2, 6))
        self.assertEqual(anchor, Domino(5, 2))

    def test_unchanged_board_reports_nothing(self):
        state = DominoBlockGameState(player_hand=[], opponent_hand=[],
                                     board=[(3, 5), (5, 2)])
        wm = self.robot.world_map
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)
        self.assertIsNone(self.bridge.detect_played_tile(self.settle(), state))

    def test_first_tile_on_an_empty_board_needs_a_play_area(self):
        """A lone tile only counts as an opening play inside the play area."""
        state = DominoBlockGameState(player_hand=[(6, 6)], opponent_hand=[])
        place(self.robot.world_map, 'd1', (6, 6), 0, 200)

        # Without a play area the bridge refuses to guess.
        self.assertIsNone(self.bridge.detect_played_tile(self.settle(), state))

        self.bridge.play_area = (0, 200, 80)
        domino, anchor = self.bridge.detect_played_tile(self.settle(), state)
        self.assertEqual(domino, Domino(6, 6))
        self.assertIsNone(anchor)

    def test_tile_outside_the_play_area_is_not_an_opening_play(self):
        state = DominoBlockGameState(player_hand=[(6, 6)], opponent_hand=[])
        self.bridge.play_area = (0, 200, 80)
        place(self.robot.world_map, 'd1', (6, 6), 0, 600)  # beside a hand
        self.assertIsNone(self.bridge.detect_played_tile(self.settle(), state))

    def test_bridge_does_not_judge_legality(self):
        """An impossible layout is still only ever a proposal.

        The bridge hands the tile over and the rules engine refuses it; the
        bridge itself never contains that verdict.
        """
        state = DominoBlockGameState(player_hand=[(4, 4)], opponent_hand=[],
                                     board=[(3, 5)])
        with self.assertRaises(ValueError):
            state.play_domino(Domino(4, 4), Domino(3, 5), player='player')

    def test_rejected_play_is_not_reported_again_until_table_changes(self):
        """An illegal tile left on the table must not re-fire every poll."""
        state = DominoBlockGameState(player_hand=[(1, 1)], opponent_hand=[],
                                     board=[(3, 5), (5, 2)])
        wm = self.robot.world_map
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)
        place(wm, 'd3', (2, 6), chain_x(2), 200)
        reading = self.settle()

        played = self.bridge.detect_played_tile(reading, state)
        self.assertIsNotNone(played)
        domino, anchor = played
        with self.assertRaises(ValueError):
            state.play_domino(domino, anchor, player='player')
        self.bridge.note_rejected_play(domino, state)

        # Same illegal tile, still on the table: stay quiet.
        self.assertIsNone(self.bridge.detect_played_tile(reading, state))

        # Player takes the bad tile away: suppression lifts with the board.
        del wm.objects['d3']
        self.assertIsNone(self.bridge.detect_played_tile(self.settle(), state))

        # A different legal tile can still be proposed afterward.
        place(wm, 'good', (2, 6), chain_x(2), 200)
        state.player_hand = [Domino(2, 6)]
        played = self.bridge.detect_played_tile(self.settle(), state)
        self.assertIsNotNone(played)
        self.assertEqual(played[0], Domino(2, 6))

    def test_sync_hands_leaves_an_unseen_hand_alone(self):
        state = DominoBlockGameState(player_hand=[(0, 0)], opponent_hand=[(1, 1)])
        # Only the far side (the learner) has anything visible.
        place(self.robot.world_map, 'theirs', (4, 5), 0, 400)
        updated = self.bridge.sync_hands(state, self.settle())
        self.assertEqual(updated, ('player',))
        self.assertEqual(state.format_hand('player'), '[4|5]')
        self.assertEqual(state.format_hand('opponent'), '[1|1]')


class HalfOrderingTests(unittest.TestCase):
    """A tile seen from the far side reports its halves back to front.

    The world map folds such an observation's heading 180 degrees onto the
    tracked heading; the halves have to be swapped to match, or the pip counts
    end up describing the opposite ends from the ones pose.theta points at.
    """

    def test_counts_and_label_are_swapped_together(self):
        self.assertEqual(reverse_half_ordering('half_counts', (3, 5)), (5, 3))
        self.assertEqual(reverse_half_ordering('face_label', '3-5'), '5-3')

    def test_positions_are_swapped(self):
        centers = ((10.0, 20.0), (30.0, 40.0))
        self.assertEqual(reverse_half_ordering('half_image_centers', centers),
                         ((30.0, 40.0), (10.0, 20.0)))

    def test_unordered_attributes_pass_through(self):
        self.assertEqual(reverse_half_ordering('confidence', 0.9), 0.9)
        self.assertIsNone(reverse_half_ordering('half_counts', None))
        # A doubled tile reads the same either way round.
        self.assertEqual(reverse_half_ordering('face_label', '4-4'), '4-4')


class DescriptionTests(BridgeTestCase):
    """Text handed to the language model must agree with the rules engine."""

    def test_description_before_looking(self):
        self.assertIn('not looked', self.bridge.describe())

    def test_description_defers_to_the_game_state(self):
        state = DominoBlockGameState(player_hand=[(2, 6)], opponent_hand=[(0, 4)],
                                     board=[(3, 5), (5, 2)])
        wm = self.robot.world_map
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)

        text = self.bridge.describe(self.settle(), state)
        self.assertIn('Board: [3|5] - [5|2]', text)
        self.assertIn('Open ends: 3 and 2', text)
        self.assertIn("Learner's hand: [2|6]", text)
        self.assertIn('Your hand: [0|4]', text)

    def test_description_uses_the_game_state_parked_on_the_bridge(self):
        """Callers holding only a robot get the same text as the FSM does."""
        self.bridge.game_state = DominoBlockGameState(
            player_hand=[(2, 6)], opponent_hand=[(0, 4)], board=[(3, 5)])
        place(self.robot.world_map, 'd1', (3, 5), chain_x(0), 200)
        text = self.bridge.describe(self.settle())
        self.assertIn('Board: [3|5]', text)
        self.assertIn("Learner's hand: [2|6]", text)

    def test_description_mentions_unread_tiles(self):
        place(self.robot.world_map, 'd1', (3, 5), 0, 200)
        text = self.bridge.describe(self.observe_frame())
        self.assertIn('cannot make out the pips', text)
        self.assertIn('(empty board)', text)


if __name__ == '__main__':
    unittest.main()
