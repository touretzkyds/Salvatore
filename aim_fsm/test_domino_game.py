import unittest

try:
    from .domino_game import DominoBlockGameState, Domino
except ImportError:  # running this file directly from inside aim_fsm/
    from domino_game import DominoBlockGameState, Domino


class DominoGameSimulationTests(unittest.TestCase):
    def test_partial_game(self) -> None:
        player_hand = [
            Domino(6, 6),
            Domino(6, 5),
            Domino(5, 4),
            Domino(4, 4),
            Domino(4, 1),
            Domino(1, 0),
            Domino(2, 0),
        ]
        opponent_hand = [
            Domino(6, 3),
            Domino(3, 3),
            Domino(3, 1),
            Domino(1, 1),
            Domino(2, 1),
            Domino(2, 2),
            Domino(5, 0),
        ]

        state = DominoBlockGameState(
            player_hand=player_hand,
            opponent_hand=opponent_hand,
            current_player="player",
        )

        # check who goes first
        self.assertEqual(state.who_goes_first(), "player")
        state.current_player = state.who_goes_first()

        self.assertEqual(state.format_board(), "(empty board)")
        self.assertEqual(len(state.player_hand), 7)
        self.assertEqual(len(state.opponent_hand), 7)

        state.play_domino(Domino(6, 6))
        self.assertEqual(state.format_board(), "[6|6]")
        self.assertEqual(len(state.player_hand), 6)
        self.assertEqual(len(state.opponent_hand), 7)

        state.play_domino(Domino(6, 3), Domino(6, 6))
        self.assertEqual(state.format_board(), "[6|6] - [6|3]")
        self.assertEqual(state.board_ends(), (6, 3))
        self.assertEqual(len(state.player_hand), 6)
        self.assertEqual(len(state.opponent_hand), 6)

        state.play_domino(Domino(6, 5), Domino(6, 6))
        self.assertEqual(state.format_board(), "[5|6] - [6|6] - [6|3]")
        self.assertEqual(state.board_ends(), (5, 3))
        self.assertEqual(len(state.player_hand), 5)
        self.assertEqual(len(state.opponent_hand), 6)

        state.play_domino(Domino(3, 1), Domino(6, 3))
        self.assertEqual(state.format_board(), "[5|6] - [6|6] - [6|3] - [3|1]")
        self.assertEqual(state.board_ends(), (5, 1))
        self.assertEqual(len(state.player_hand), 5)
        self.assertEqual(len(state.opponent_hand), 5)

    def test_domino_quality(self):
        self.assertEqual(Domino(6, 3), Domino(3, 6))
        self.assertNotEqual(Domino(6, 3), Domino(6, 4))
        self.assertEqual(Domino(6, 6), Domino(6, 6))

    def test_who_goes_first_prefers_higher_double_over_rank(self):
        state = DominoBlockGameState(
            player_hand=[Domino(3, 3), Domino(1, 0), Domino(2, 1)],
            opponent_hand=[Domino(2, 2), Domino(6, 5), Domino(4, 0)],
            current_player="player",
        )
        self.assertEqual(state.who_goes_first(), "player")


if __name__ == "__main__":
    unittest.main()
