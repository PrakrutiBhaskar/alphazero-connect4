"""
Connect Four environment for AlphaZero self-play.

Board convention:
  - 6 rows x 7 columns
  - board[row, col]: 0 = empty, 1 = player 1, -1 = player 2
  - row 0 is the TOP of the board, row 5 is the BOTTOM (gravity fills from bottom up)
  - `to_play` tracks whose turn it is (1 or -1); the network always sees the
    board from the CURRENT player's perspective (canonical form) so it only
    ever has to learn one side.
"""

from __future__ import annotations
import numpy as np

ROWS = 6
COLS = 7
N_IN_ROW = 4


class ConnectFour:
    def __init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.to_play = 1
        self.last_move = None
        self.move_count = 0
        return self.board.copy()

    def clone(self) -> "ConnectFour":
        c = ConnectFour.__new__(ConnectFour)
        c.board = self.board.copy()
        c.to_play = self.to_play
        c.last_move = self.last_move
        c.move_count = self.move_count
        return c

    def legal_moves(self) -> list[int]:
        """Columns that are not full."""
        return [c for c in range(COLS) if self.board[0, c] == 0]

    def legal_moves_mask(self) -> np.ndarray:
        mask = np.zeros(COLS, dtype=np.float32)
        for c in self.legal_moves():
            mask[c] = 1.0
        return mask

    def _drop_row(self, col: int) -> int:
        """Return the row a piece dropped in `col` would land on."""
        for r in range(ROWS - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        raise ValueError(f"Column {col} is full")

    def step(self, col: int) -> tuple[np.ndarray, bool, int]:
        """
        Apply a move for the current player.
        Returns (board, done, winner) where winner in {1, -1, 0}
        (0 = draw or game not yet over).
        """
        if col not in self.legal_moves():
            raise ValueError(f"Illegal move: column {col}")
        row = self._drop_row(col)
        self.board[row, col] = self.to_play
        self.last_move = (row, col)
        self.move_count += 1

        done, winner = self._check_terminal(row, col)
        self.to_play *= -1
        return self.board.copy(), done, winner

    def _check_terminal(self, row: int, col: int) -> tuple[bool, int]:
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= N_IN_ROW:
                return True, player
        if self.move_count >= ROWS * COLS:
            return True, 0  # draw
        return False, 0

    def canonical_board(self) -> np.ndarray:
        """Board from the perspective of the player about to move."""
        return (self.board * self.to_play).astype(np.float32)

    def render(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        rows = []
        for r in range(ROWS):
            rows.append(" ".join(symbols[v] for v in self.board[r]))
        rows.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(rows)
