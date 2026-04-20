from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import time
from typing import Iterable

import chess

from .base_agent import ChessAgent
from .mcts_agent import MCTSAgent
from .minimax_agent import MinimaxAgent
from .random_agent import RandomAgent


@dataclass
class SideStats:
    # stats for a single game.
    agent_name: str
    moves: int
    total_move_time_sec: float
    total_legal_options: int
    captures: int
    checks_given: int
    promotions: int

    @property
    def avg_move_time_ms(self) -> float:
        if self.moves == 0:
            return 0.0
        return (self.total_move_time_sec / self.moves) * 1000.0

    @property
    def avg_legal_options(self) -> float:
        # average branching factor seen by this agent
        if self.moves == 0:
            return 0.0
        return self.total_legal_options / self.moves


@dataclass
class GameResult:
    # This denotes the full record for one completed game
    white_agent: str
    black_agent: str
    result: str
    plies: int
    termination: str
    white_stats: SideStats
    black_stats: SideStats


@dataclass
class AgentMetrics:
    #All metrics across all games for one agent
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: float = 0.0

    games_as_white: int = 0
    games_as_black: int = 0
    wins_as_white: int = 0
    wins_as_black: int = 0

    total_moves: int = 0
    total_move_time_sec: float = 0.0
    total_legal_options: int = 0
    captures: int = 0
    checks_given: int = 0
    promotions: int = 0

    checkmates_delivered: int = 0
    checkmates_suffered: int = 0

    @property
    def win_rate(self) -> float:
        # Win rate
        if self.games == 0:
            return 0.0
        return self.wins / self.games

    @property
    def avg_move_time_ms(self) -> float:
        # Avg
        if self.total_moves == 0:
            return 0.0
        return (self.total_move_time_sec / self.total_moves) * 1000.0

    @property
    def avg_legal_options(self) -> float:
        # Avg number of legal choices on each turn.
        if self.total_moves == 0:
            return 0.0
        return self.total_legal_options / self.total_moves

    @property
    def captures_per_game(self) -> float:
        # Avg captures made per game
        if self.games == 0:
            return 0.0
        return self.captures / self.games

    @property
    def checks_per_game(self) -> float:
        # Avg chekcs made per game
        if self.games == 0:
            return 0.0
        return self.checks_given / self.games


def play_game(
    white_agent: ChessAgent,
    black_agent: ChessAgent,
    max_plies: int = 300,
) -> GameResult:
    board = chess.Board()
    plies = 0

    # This rack separate stats for each side in this game
    white_moves = 0
    black_moves = 0
    white_move_time = 0.0
    black_move_time = 0.0
    white_legal_options = 0
    black_legal_options = 0
    white_captures = 0
    black_captures = 0
    white_checks = 0
    black_checks = 0
    white_promotions = 0
    black_promotions = 0

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        current_agent = white_agent if board.turn == chess.WHITE else black_agent
        legal_moves = list(board.legal_moves)
        num_options = len(legal_moves)

        # Measure how long the agent takes to make move
        start_time = time.perf_counter()
        move = current_agent.select_move(board)
        elapsed = time.perf_counter() - start_time

        if move not in legal_moves:
            raise ValueError(f"Agent {current_agent.name} returned illegal move: {move}")

        is_white_turn = board.turn == chess.WHITE
        is_capture = board.is_capture(move)
        is_promotion = move.promotion is not None

        board.push(move)

        # If check
        if board.is_check():
            if is_white_turn:
                white_checks += 1
            else:
                black_checks += 1

        if is_white_turn:
            white_moves += 1
            white_move_time += elapsed
            white_legal_options += num_options
            if is_capture:
                white_captures += 1
            if is_promotion:
                white_promotions += 1
        else:
            black_moves += 1
            black_move_time += elapsed
            black_legal_options += num_options
            if is_capture:
                black_captures += 1
            if is_promotion:
                black_promotions += 1

        plies += 1

    # Denotes why the game ended
    if plies >= max_plies and not board.is_game_over(claim_draw=True):
        result = "1/2-1/2"
        termination = "max_plies"
    else:
        result = board.result(claim_draw=True)
        if board.is_checkmate():
            termination = "checkmate"
        elif board.is_stalemate():
            termination = "stalemate"
        elif board.is_insufficient_material():
            termination = "insufficient_material"
        elif board.is_seventyfive_moves():
            termination = "seventyfive_moves"
        elif board.is_fivefold_repetition():
            termination = "fivefold_repetition"
        else:
            termination = "draw_claim_or_other"

    return GameResult(
        white_agent=white_agent.name,
        black_agent=black_agent.name,
        result=result,
        plies=plies,
        termination=termination,
        white_stats=SideStats(
            agent_name=white_agent.name,
            moves=white_moves,
            total_move_time_sec=white_move_time,
            total_legal_options=white_legal_options,
            captures=white_captures,
            checks_given=white_checks,
            promotions=white_promotions,
        ),
        black_stats=SideStats(
            agent_name=black_agent.name,
            moves=black_moves,
            total_move_time_sec=black_move_time,
            total_legal_options=black_legal_options,
            captures=black_captures,
            checks_given=black_checks,
            promotions=black_promotions,
        ),
    )


def run_round_robin(agents: Iterable[ChessAgent], games_per_pair: int = 2) -> tuple[list[GameResult], dict[str, float]]:
    agent_list = list(agents)
    scores = {agent.name: 0.0 for agent in agent_list}
    game_results: list[GameResult] = []

    for a1, a2 in combinations(agent_list, 2):
        # Alternate colors so each agent gets a fair White/Black split
        for game_index in range(games_per_pair):
            if game_index % 2 == 0:
                white, black = a1, a2
            else:
                white, black = a2, a1

            result = play_game(white, black)
            game_results.append(result)

            if result.result == "1-0":
                scores[white.name] += 1.0
            elif result.result == "0-1":
                scores[black.name] += 1.0
            else:
                scores[white.name] += 0.5
                scores[black.name] += 0.5

    return game_results, scores


def build_agent_metrics(game_results: Iterable[GameResult]) -> dict[str, AgentMetrics]:
    # Combine game and agent stats
    metrics: dict[str, AgentMetrics] = {}

    for result in game_results:
        white = metrics.setdefault(result.white_agent, AgentMetrics())
        black = metrics.setdefault(result.black_agent, AgentMetrics())

        white.games += 1
        black.games += 1
        white.games_as_white += 1
        black.games_as_black += 1

        white.total_moves += result.white_stats.moves
        black.total_moves += result.black_stats.moves
        white.total_move_time_sec += result.white_stats.total_move_time_sec
        black.total_move_time_sec += result.black_stats.total_move_time_sec
        white.total_legal_options += result.white_stats.total_legal_options
        black.total_legal_options += result.black_stats.total_legal_options
        white.captures += result.white_stats.captures
        black.captures += result.black_stats.captures
        white.checks_given += result.white_stats.checks_given
        black.checks_given += result.black_stats.checks_given
        white.promotions += result.white_stats.promotions
        black.promotions += result.black_stats.promotions

        if result.result == "1-0":
            # White win
            white.wins += 1
            black.losses += 1
            white.wins_as_white += 1
            white.points += 1.0
        elif result.result == "0-1":
            # Black win
            black.wins += 1
            white.losses += 1
            black.wins_as_black += 1
            black.points += 1.0
        else:
            # Draw
            white.draws += 1
            black.draws += 1
            white.points += 0.5
            black.points += 0.5

        if result.termination == "checkmate":
            if result.result == "1-0":
                white.checkmates_delivered += 1
                black.checkmates_suffered += 1
            elif result.result == "0-1":
                black.checkmates_delivered += 1
                white.checkmates_suffered += 1

    return metrics


def print_metrics_table(metrics: dict[str, AgentMetrics]) -> None:
    print("\n=== Agent Comparison Metrics ===")
    header = (
        f"{'Agent':<18} {'Pts':>5} {'W-D-L':>9} {'Win%':>7} {'AsW':>4} {'AsB':>4} "
        f"{'AvgMove(ms)':>12} {'AvgBranch':>10} {'Cap/G':>7} {'Chk/G':>7} {'CM+':>4} {'CM-':>4}"
    )
    print(header)
    print("-" * len(header))

    for agent_name, m in sorted(metrics.items(), key=lambda item: item[1].points, reverse=True):
        # Format one row per agent
        wdl = f"{m.wins}-{m.draws}-{m.losses}"
        asw = f"{m.wins_as_white}/{m.games_as_white}"
        asb = f"{m.wins_as_black}/{m.games_as_black}"
        print(
            f"{agent_name:<18} {m.points:>5.1f} {wdl:>9} {100.0 * m.win_rate:>6.1f}% "
            f"{asw:>4} {asb:>4} {m.avg_move_time_ms:>12.2f} {m.avg_legal_options:>10.2f} "
            f"{m.captures_per_game:>7.2f} {m.checks_per_game:>7.2f} {m.checkmates_delivered:>4} {m.checkmates_suffered:>4}"
        )


def main() -> None:
    # Runs a small tournament between a few baseline agents
    agents: list[ChessAgent] = [
        MinimaxAgent(depth=2),
        MinimaxAgent(depth=3),
        MCTSAgent(simulations=80, rollout_depth=18, seed=7),
        RandomAgent(seed=42),
    ]

    checkpoint = Path("value_net.pt")
    if checkpoint.exists():
        from .rl_value_agent import RLMCTSAgent

        agents.append(RLMCTSAgent(checkpoint=checkpoint, simulations=80, rollout_depth=18, seed=23))

    results, scores = run_round_robin(agents, games_per_pair=2)

    print("=== Game Results ===")
    for r in results:
        print(
            f"{r.white_agent} vs {r.black_agent}: {r.result} "
            f"({r.plies} plies, end={r.termination}, "
            f"W avg {r.white_stats.avg_move_time_ms:.2f}ms, "
            f"B avg {r.black_stats.avg_move_time_ms:.2f}ms)"
        )

    print("\n=== Leaderboard ===")
    # Rank by points
    for agent_name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{agent_name}: {score:.1f}")

    # full comparison metrics table
    metrics = build_agent_metrics(results)
    print_metrics_table(metrics)


if __name__ == "__main__":
    main()
