"""Tests for the FSM node that watches the table during the player's turn.

`WatchDominoTable` is where perception finally meets the game, so these tests
pin down the property that matters most: the node never rules on a move itself.
It hands what the camera saw to DominoBlockGameState and reports whichever way
that goes, so a misread tile becomes a normal "that's illegal" conversation
rather than a corrupted board.

The node is exercised directly with fakes standing in for the event router and
the enclosing state machine, so no robot or camera is needed.
"""

import unittest

from .domino_game import DominoBlockGameState
from .domino_bridge import DominoBridge
from .test_domino_bridge import FakeRobot, place, chain_x

import Salvatore


class FakeParent:
    """Stands in for the Salvatore state machine around the node."""

    def __init__(self, state):
        self.domino_state = state
        self.domino_last_move = None
        self.domino_last_mover = None
        self.domino_illegal_move_prompt = ""

    def describe_domino(self, domino):
        return f'{domino.left}-{domino.right}'

    def build_domino_illegal_move_prompt(self, attempted, error):
        return f'{attempted}: {error}'


class RecordingWatcher(Salvatore.Salvatore.WatchDominoTable):
    """The real node, with its two outputs captured instead of dispatched."""

    def __init__(self, parent, bridge):
        self.parent = parent
        self.bridge = bridge
        self.watching = True
        self.posted = []
        self.polling_interval = None

    def post_data(self, value):
        self.posted.append(value)

    def set_polling_interval(self, interval):
        self.polling_interval = interval


class WatchDominoTableTests(unittest.TestCase):

    def setUp(self):
        self.robot = FakeRobot()
        self.bridge = DominoBridge(self.robot)

    def settle(self, frames=5):
        for _ in range(frames):
            self.robot.world_map.observe_frame()
            self.bridge.observe()

    def watcher_for(self, state):
        self.parent = FakeParent(state)
        return RecordingWatcher(self.parent, self.bridge)

    def test_legal_placement_is_applied_and_reported(self):
        state = DominoBlockGameState(player_hand=[(2, 6)], opponent_hand=[(0, 1)],
                                     board=[(3, 5), (5, 2)])
        wm = self.robot.world_map
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)
        place(wm, 'd3', (2, 6), chain_x(2), 200)
        self.settle()

        watcher = self.watcher_for(state)
        watcher.poll()

        self.assertEqual(watcher.posted, ['resolved'])
        self.assertEqual(state.format_board(), '[3|5] - [5|2] - [2|6]')
        self.assertEqual(state.format_hand('player'), '(empty hand)')
        self.assertEqual(self.parent.domino_last_mover, 'player')
        self.assertEqual(state.current_player, 'opponent')
        # Polling stops once the move has been dealt with.
        self.assertIsNone(watcher.polling_interval)
        self.assertFalse(watcher.watching)

    def test_placement_the_rules_reject_is_reported_as_illegal(self):
        """A tile the player does not hold must not slip onto the board."""
        state = DominoBlockGameState(player_hand=[(1, 1)], opponent_hand=[(0, 1)],
                                     board=[(3, 5), (5, 2)])
        wm = self.robot.world_map
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)
        place(wm, 'd3', (2, 6), chain_x(2), 200)
        self.settle()

        watcher = self.watcher_for(state)
        watcher.poll()

        self.assertEqual(watcher.posted, ['illegal'])
        self.assertEqual(state.format_board(), '[3|5] - [5|2]')
        self.assertIn('2-6', self.parent.domino_illegal_move_prompt)
        self.assertEqual(state.current_player, 'player')
        # The bad tile is still on the table, but must not re-fire forever.
        self.assertIsNotNone(self.bridge._suppressed_play)
        watcher.watching = True
        watcher.poll()
        self.assertEqual(watcher.posted, ['illegal'])

    def test_nothing_is_posted_while_the_board_is_unchanged(self):
        state = DominoBlockGameState(player_hand=[(2, 6)], opponent_hand=[],
                                     board=[(3, 5), (5, 2)])
        wm = self.robot.world_map
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)
        self.settle()

        watcher = self.watcher_for(state)
        for _ in range(3):
            watcher.poll()

        self.assertEqual(watcher.posted, [])
        self.assertTrue(watcher.watching)

    def test_the_robots_own_turn_is_left_alone(self):
        """Salvatore places his own tiles; the watcher must not race him."""
        state = DominoBlockGameState(player_hand=[], opponent_hand=[(2, 6)],
                                     board=[(3, 5), (5, 2)],
                                     current_player='opponent')
        wm = self.robot.world_map
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)
        place(wm, 'd3', (2, 6), chain_x(2), 200)
        self.settle()

        watcher = self.watcher_for(state)
        watcher.poll()

        self.assertEqual(watcher.posted, [])
        self.assertEqual(state.format_board(), '[3|5] - [5|2]')

    def test_a_half_read_tile_is_waited_out_rather_than_guessed(self):
        """Before the pip vote settles, the node reports nothing at all."""
        state = DominoBlockGameState(player_hand=[(2, 6)], opponent_hand=[],
                                     board=[(3, 5), (5, 2)])
        wm = self.robot.world_map
        place(wm, 'd1', (3, 5), chain_x(0), 200)
        place(wm, 'd2', (5, 2), chain_x(1), 200)
        place(wm, 'd3', (2, 6), chain_x(2), 200)

        watcher = self.watcher_for(state)
        self.settle(1)
        watcher.poll()
        self.assertEqual(watcher.posted, [])

        self.settle(4)
        watcher.poll()
        self.assertEqual(watcher.posted, ['resolved'])

    def test_a_broken_bridge_stops_the_watch_instead_of_crashing(self):
        state = DominoBlockGameState(player_hand=[(2, 6)], opponent_hand=[],
                                     board=[(3, 5)])
        watcher = self.watcher_for(state)

        class ExplodingBridge:
            def observe(self):
                raise RuntimeError('camera on fire')

            def detect_played_tile(self, reading, state):
                raise AssertionError('should never be reached')

        watcher.bridge = ExplodingBridge()
        watcher.poll()

        self.assertEqual(watcher.posted, [])
        self.assertFalse(watcher.watching)
        self.assertIsNone(watcher.polling_interval)


if __name__ == '__main__':
    unittest.main()
