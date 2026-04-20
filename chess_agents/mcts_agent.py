from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable

import chess

from .base_agent import ChessAgent


ValueEvaluator = Callable[[chess.Board], float]


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def result_value(board: chess.Board, perspective: chess.Color) -> float:
    """Return terminal value from `perspective`: win=1, draw=0, loss=-1."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == perspective else -1.0


def material_value(board: chess.Board, perspective: chess.Color) -> float:
    """Small fallback evaluator used when a rollout reaches the depth limit."""
    opponent = not perspective
    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        score += value * len(board.pieces(piece_type, perspective))
        score -= value * len(board.pieces(piece_type, opponent))
    return math.tanh(score / 10.0)


@dataclass
class SearchStats:
    simulations: int = 0
    nodes: int = 0
    root_visits: int = 0


@dataclass
class MCTSNode:
    board: chess.Board
    parent: MCTSNode | None = None
    move: chess.Move | None = None
    untried_moves: list[chess.Move] = field(init=False)
    children: dict[chess.Move, MCTSNode] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    def __post_init__(self) -> None:
        self.untried_moves = list(self.board.legal_moves)

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    @property
    def fully_expanded(self) -> bool:
        return not self.untried_moves


class MCTSAgent(ChessAgent):
    """
    Monte Carlo Tree Search agent.

    Values are stored from the perspective of the side to move at each node.
    When a parent compares children, each child value is negated because the
    child board is the opponent's turn after the parent makes a move.
    """

    def __init__(
        self,
        simulations: int = 500,
        exploration_weight: float = 1.4,
        rollout_depth: int = 40,
        seed: int | None = None,
        value_evaluator: ValueEvaluator | None = None,
        name: str | None = None,
    ):
        if simulations < 1:
            raise ValueError("MCTS requires at least one simulation.")

        super().__init__(name or f"MCTS(sim={simulations})")
        self.simulations = simulations
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth
        self.value_evaluator = value_evaluator
        self._rng = random.Random(seed)
        self.stats = SearchStats()

    def select_move(self, board: chess.Board) -> chess.Move:
        move, _ = self.choose_move(board, temperature=0.0)
        return move

    def choose_move(self, board: chess.Board, temperature: float = 0.0) -> tuple[chess.Move, dict[str, float]]:
        root = self.search(board)
        children = list(root.children.values())
        if not children:
            raise ValueError("No legal moves available.")

        visits = [child.visits for child in children]
        if temperature <= 1e-8:
            best_visit_count = max(visits)
            best_children = [child for child in children if child.visits == best_visit_count]
            best_child = max(best_children, key=lambda child: -child.mean_value)
            policy = {child.move.uci(): 1.0 if child is best_child else 0.0 for child in children}
            return best_child.move, policy

        weights = [visit ** (1.0 / temperature) for visit in visits]
        if sum(weights) == 0:
            weights = [1.0 for _ in children]
        total_weight = sum(weights)
        selected = self._rng.choices(children, weights=weights, k=1)[0]
        policy = {
            child.move.uci(): weight / total_weight
            for child, weight in zip(children, weights)
        }
        return selected.move, policy

    def search(self, board: chess.Board) -> MCTSNode:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        root = MCTSNode(board.copy(stack=True))
        nodes_created = 1

        for _ in range(self.simulations):
            node = root
            path = [node]

            while (
                not node.board.is_game_over(claim_draw=True)
                and node.fully_expanded
                and node.children
            ):
                node = self._select_child(node)
                path.append(node)

            if not node.board.is_game_over(claim_draw=True) and node.untried_moves:
                node = self._expand(node)
                nodes_created += 1
                path.append(node)

            value = self._evaluate(node.board)
            self._backpropagate(path, value)

        self.stats = SearchStats(
            simulations=self.simulations,
            nodes=nodes_created,
            root_visits=root.visits,
        )
        return root

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        log_parent_visits = math.log(max(node.visits, 1))

        def uct_score(child: MCTSNode) -> float:
            if child.visits == 0:
                return float("inf")
            exploitation = -child.mean_value
            exploration = self.exploration_weight * math.sqrt(log_parent_visits / child.visits)
            return exploitation + exploration

        return max(node.children.values(), key=uct_score)

    def _expand(self, node: MCTSNode) -> MCTSNode:
        move = self._rng.choice(node.untried_moves)
        node.untried_moves.remove(move)

        child_board = node.board.copy(stack=True)
        child_board.push(move)
        child = MCTSNode(child_board, parent=node, move=move)
        node.children[move] = child
        return child

    def _evaluate(self, board: chess.Board) -> float:
        if board.is_game_over(claim_draw=True):
            return result_value(board, board.turn)

        if self.value_evaluator is not None:
            value = self.value_evaluator(board)
            return max(-1.0, min(1.0, float(value)))

        return self._rollout(board)

    def _rollout(self, board: chess.Board) -> float:
        perspective = board.turn
        simulation = board.copy(stack=True)

        for _ in range(self.rollout_depth):
            if simulation.is_game_over(claim_draw=True):
                return result_value(simulation, perspective)
            legal_moves = list(simulation.legal_moves)
            if not legal_moves:
                break
            simulation.push(self._rng.choice(legal_moves))

        if simulation.is_game_over(claim_draw=True):
            return result_value(simulation, perspective)
        return material_value(simulation, perspective)

    @staticmethod
    def _backpropagate(path: list[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value
