from __future__ import annotations

import random

import chess

from .base_agent import ChessAgent


class MCTSAgent(ChessAgent):
    """
    Monte Carlo Tree Search agent.
    """

    def __init__(self, simulations: int = 500, seed: int | None = None, name: str | None = None):
        super().__init__(name or f"MCTS(sim={simulations})")
        self.simulations = simulations
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        # Placeholder strategy until full MCTS is implemented 
        return self._rng.choice(legal_moves)
