from __future__ import annotations

from abc import ABC, abstractmethod

import chess


class ChessAgent(ABC):
    """Interface for all chess agents."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_move(self, board: chess.Board) -> chess.Move:
        """Return a legal move for the current board state."""
