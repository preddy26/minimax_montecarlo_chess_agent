from __future__ import annotations

import random

import chess

from .base_agent import ChessAgent


class RandomAgent(ChessAgent):
    def __init__(self, seed: int | None = None, name: str = "Random"):
        super().__init__(name)
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available.")
        return self._rng.choice(legal_moves)
