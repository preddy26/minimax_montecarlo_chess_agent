from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess

from .base_agent import ChessAgent



PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}


@dataclass
class SearchStats:
    #nodes visited in the most recent search
    nodes: int = 0


class MinimaxAgent(ChessAgent):
    """
    Depth-limited minimax chess agent with alpha-beta pruning.

    Evaluation is from White's perspective; maximizing player is whichever side
    is to move at the root.

    High-level flow in `select_move()`:
    1) Generate legal root moves.
    2) For each move, search the resulting position up to `depth`.
    3) Pick the move with best minimax value.

    Minimax intuition:
    - One side tries to maximize score.
    - The opponent tries to minimize score.
    - The chosen move assumes both players respond optimally.

    Alpha-beta pruning intuition:
    - `alpha` = best guaranteed score for maximizing side so far.
    - `beta`  = best guaranteed score for minimizing side so far.
    - If `beta <= alpha`, further children cannot affect final decision,
      so that branch is skipped (pruned).
    """

    def __init__(self, depth: int = 3, name: Optional[str] = None):
        #`epth is measured in half-moves, not full turns
        super().__init__(name or f"Minimax(d={depth})")
        self.depth = depth
        self.stats = SearchStats()

    def select_move(self, board: chess.Board) -> chess.Move:
        """Choose the best legal move from the current position."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        self.stats = SearchStats()
        #If White to move, search as max player else min.
        maximizing = board.turn == chess.WHITE
        best_move = legal_moves[0]
        best_value = float("-inf") if maximizing else float("inf")

        ordered_moves = self._order_moves(board, legal_moves)
        for move in ordered_moves:
            #Have to apply candidate move, evaluate subtree, then undo move
            board.push(move)
            value = self._minimax(
                board=board,
                depth=self.depth - 1,
                alpha=float("-inf"),
                beta=float("inf"),
                maximizing_player=not maximizing,
            )
            board.pop()

            if maximizing and value > best_value:
                best_value = value
                best_move = move
            elif not maximizing and value < best_value:
                best_value = value
                best_move = move

        return best_move

    def _minimax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
    ) -> float:
        """
        Recursive minimax search with alpha-beta pruning.

        Returns a scalar score where:
        - larger is better for White
        - smaller is better for Black
        """
        self.stats.nodes += 1

        if board.is_checkmate():
            # Side to move has no legal moves and is mated.
            # If White to move and checkmated = -inf
            # If Black to move and checkmated = + inf
            return float("-inf") if board.turn == chess.WHITE else float("inf")

        if board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_seventyfive_moves():
            return 0.0

        #Depth cutoff
        if depth == 0:
            return self._evaluate_board(board)
        moves = self._order_moves(board, list(board.legal_moves))

        if maximizing_player:
            value = float("-inf")
            for move in moves:
                board.push(move)
                value = max(value, self._minimax(board, depth - 1, alpha, beta, False))
                board.pop()

                # Raise alpha
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = float("inf")
            for move in moves:
                board.push(move)
                value = min(value, self._minimax(board, depth - 1, alpha, beta, True))
                board.pop()

                # Lower beta
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def _evaluate_board(self, board: chess.Board) -> float:
        """
        Static position evaluator (no further search).

        Positive score favors White; negative score favors Black.

        Components:
        - Material: weighted piece counts.
        - Mobility: small bonus for side to move having many legal options.
        """
        material_score = 0
        for piece_type, value in PIECE_VALUES.items():
            material_score += len(board.pieces(piece_type, chess.WHITE)) * value
            material_score -= len(board.pieces(piece_type, chess.BLACK)) * value

        # Sign is flipped when Black is to move so the score remains
        mobility_score = 0.1 * len(list(board.legal_moves))
        if board.turn == chess.BLACK:
            mobility_score *= -1

        return material_score + mobility_score

    @staticmethod
    def _order_moves(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
        """
        Light move-ordering to improve alpha-beta pruning.
        Prioritize captures, promotions, and checks.

        Better move ordering does not change minimax correctness;
        it only changes how quickly pruning happens.
        """

        def move_priority(m: chess.Move) -> int:
            score = 0
            if board.is_capture(m):
                score += 10
            if m.promotion is not None:
                score += 8

            board.push(m)
            if board.is_check():
                score += 6
            board.pop()
            return score

        return sorted(moves, key=move_priority, reverse=True)
