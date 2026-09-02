from __future__ import annotations

class Domino:
    """Domino Class"""
    def __init__(self, left, right):
        if left > 6 or right > 6:
            raise ValueError("Domino ends must be <= 6 for a double-6 set")
        self.left = left
        self.right = right

    def flipped(self):
        return Domino(self.right, self.left)

    def matches(self, value):
        return self.left == value or self.right == value

    def __repr__(self):
        return f"[{self.left}|{self.right}]"

    def __str__(self):
        return f"[{self.left}|{self.right}]"

    def __eq__(self, other):
        if not isinstance(other, Domino):
            return False
        return (
            (self.left == other.left and self.right == other.right)
            or (self.left == other.right and self.right == other.left)
        )

class Move:
    """Move Class"""
    def __init__(self, domino, anchor_domino, flipped, anchor_value=None):
        self.domino = domino
        self.anchor_domino = anchor_domino
        self.anchor_value = anchor_value
        self.flipped = flipped

    def oriented(self):
        """Makes sure the domino is flipped properly for appending to the list."""
        return self.domino.flipped() if self.flipped else self.domino


class DominoBlockGameState:
    """Tracks the board state of the Block Game to help Salvatore handle legal moves
    and board state. Will also format board into a text/readable format."""

    def __init__(
        self,
        player_hand,
        opponent_hand,
        board = None,
        current_player = "player",
        teaching_mode = False
    ):
        self.player_hand = [self.convert_to_domino(d) for d in player_hand]
        self.opponent_hand = [self.convert_to_domino(d) for d in opponent_hand]

        # the board layout will be represented as a straight line from left to right in the array
        self.board = [self.convert_to_domino(d) for d in (board or [])]
        self.current_player = current_player

        self.teaching_mode = teaching_mode

    def legal_moves(self, player = None):
        player = player or self.current_player
        hand = self._hand(player)
        if not self.board:
            return [Move(domino, None, False) for domino in hand]

        left_end, right_end = self.board_ends()
        moves = []
        for domino in hand:
            if domino.matches(left_end):
                moves.append(
                    Move(domino, self.board[0], flipped=(domino.left == left_end), anchor_value=left_end)
                )
            if domino.matches(right_end):
                moves.append(
                    Move(domino, self.board[-1], flipped=(domino.right == right_end), anchor_value=right_end)
                )
        return moves

    def play_domino(
        self,
        domino,
        anchor_domino=None,
        anchor_value=None,
        player = None,
    ):
        player = player or self.current_player
        domino = self.convert_to_domino(domino)
        move, side = self._resolve_move(domino, anchor_domino, anchor_value)
        if move is None or side is None:
            if anchor_domino is not None and self.board:
                anchor_domino = self.convert_to_domino(anchor_domino)
                left_end, right_end = self.board_ends()
                if anchor_domino == self.board[0]:
                    open_end = left_end
                elif anchor_domino == self.board[-1]:
                    open_end = right_end
                else:
                    open_end = None
                if open_end is not None:
                    raise ValueError(
                        f"Illegal move: {domino} cannot connect to the open {open_end} end of {anchor_domino}"
                    )
            raise ValueError(f"Illegal move: {domino} has no matching open end on the board")

        self.remove_from_hand(player, domino)
        oriented = move.oriented()
        if not self.board:
            self.board.append(oriented)
        elif side == "left":
            self.board = [oriented] + self.board
        else: # insert on the right side
            self.board.append(oriented)

        self.current_player = self.switch_players(player)
        return move

    def pass_turn(self, player):
        """Happens if blocked"""
        self.current_player = self.switch_players(player)

    def who_goes_first(self) -> str:
        """Highest double, then highest rank"""
        player_double = self.get_highest_double(self.player_hand)
        opponent_double = self.get_highest_double(self.opponent_hand)
        if player_double is not None or opponent_double is not None:
            if player_double is None:
                self.current_player = "opponent"
                return self.current_player
            if opponent_double is None:
                self.current_player = "player"
                return self.current_player
            if player_double > opponent_double:
                self.current_player = "player"
                return self.current_player
            elif opponent_double > player_double:
                self.current_player = "opponent"
                return self.current_player

        player_rank = self.get_highest_rank(self.player_hand)
        opponent_rank = self.get_highest_rank(self.opponent_hand)
        if player_rank >= opponent_rank: # TODO: add more tiebreakers here
            self.current_player = "player"
        else:
            self.current_player = "opponent"
        return self.current_player

    def board_ends(self):
        """Returns the ends of the board."""
        return self.board[0].left, self.board[-1].right

    def format_board(self):
        if not self.board:
            return "(empty board)"
        return " - ".join(self.format_domino(d) for d in self.board)

    def format_hand(self, player):
        hand = self._hand(player or self.current_player)
        if not hand:
            return "(empty hand)"
        return ", ".join(self.format_domino(d) for d in hand)

    def __str__(self):
        # TODO: in teaching mode, also see opponent's hand
        return f"Board: {self.format_board()}\nYour hand: {self.format_hand(player='player')}"

    def _resolve_move(self, domino, anchor_domino=None, anchor_value=None):
        if not self.board: # nothing in board right now
            return Move(domino, None, False), "right"

        left_domino = self.board[0]
        right_domino = self.board[-1]
        left_end, right_end = self.board_ends()

        def resolve_left():
            if domino.right == left_end:
                return Move(domino, left_domino, False, anchor_value=left_end), "left"
            if domino.left == left_end:
                return Move(domino, left_domino, True, anchor_value=left_end), "left"
            return (None, None)

        def resolve_right():
            if domino.left == right_end:
                return Move(domino, right_domino, False, anchor_value=right_end), "right"
            if domino.right == right_end:
                return Move(domino, right_domino, True, anchor_value=right_end), "right"
            return (None, None)

        if anchor_domino is not None:
            anchor_domino = self.convert_to_domino(anchor_domino)
            match_left = anchor_domino == left_domino
            match_right = anchor_domino == right_domino
            if match_left and not match_right:
                return resolve_left()
            if match_right and not match_left:
                return resolve_right()
            if match_left and match_right:
                # Single-tile board; prefer the end value if provided.
                if anchor_value is not None:
                    if anchor_value == left_end and anchor_value != right_end:
                        return resolve_left()
                    if anchor_value == right_end and anchor_value != left_end:
                        return resolve_right()
                # If only one end value matches, choose it.
                match_left_val = domino.matches(left_end)
                match_right_val = domino.matches(right_end)
                if match_left_val and not match_right_val:
                    return resolve_left()
                if match_right_val and not match_left_val:
                    return resolve_right()
                # Ends equal (double) or fully ambiguous; default to right.
                if left_end == right_end:
                    return resolve_right()
                return (None, None)
            return (None, None)

        match_left_val = domino.matches(left_end)
        match_right_val = domino.matches(right_end)
        if match_left_val and not match_right_val:
            return resolve_left()
        if match_right_val and not match_left_val:
            return resolve_right()
        if left_end == right_end and match_left_val and match_right_val:
            return resolve_right()
        return (None, None)

    def _hand(self, player):
        if player == "player":
            return self.player_hand

        # opponent
        return self.opponent_hand

    def remove_from_hand(self, player, domino):
        hand = self._hand(player)
        for idx, item in enumerate(hand):
            if item == domino:
                hand.pop(idx)
                return
        raise ValueError(f"{self.format_domino(domino)} not in {player} hand")

    @staticmethod
    def switch_players(player):
        return "opponent" if player == "player" else "player"

    @staticmethod
    def convert_to_domino(value):
        """Convert to domino class is needed."""
        if isinstance(value, Domino):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            return Domino(int(value[0]), int(value[1]))
        raise ValueError("Unable to convert to a valid domino")

    @staticmethod
    def format_domino(domino):
        return f"[{domino.left}|{domino.right}]"

    @staticmethod
    def get_highest_double(hand):
        doubles = [domino.left for domino in hand if domino.left == domino.right]
        if not doubles:
            return None
        return max(doubles)

    @staticmethod
    def get_highest_rank(hand):
        return max(domino.left + domino.right for domino in hand)

    @staticmethod
    def get_highest_end(domino):
        """Only used for tiebreaking if no doubles and rank is the same."""
        return max(domino.left, domino.right)
