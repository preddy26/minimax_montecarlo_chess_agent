from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .mcts_agent import MCTSAgent, result_value


PIECE_PLANES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    planes = np.zeros((13, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        offset = 0 if piece.color == board.turn else 6
        planes[offset + PIECE_PLANES[piece.piece_type], row, col] = 1.0
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return planes


class ValueNet(nn.Module):
    """Small convolutional value network for board evaluation."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(13, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def value_from_outcome(outcome: chess.Outcome | None, perspective: chess.Color) -> float:
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == perspective else -1.0


def evaluate_with_model(model: ValueNet, board: chess.Board, device: str = "cpu") -> float:
    if board.is_game_over(claim_draw=True):
        return result_value(board, board.turn)

    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(model(x).item())


def load_value_net(path: str | Path, device: str = "cpu") -> ValueNet:
    payload = torch.load(path, map_location=device)
    channels = int(payload.get("channels", 64))
    model = ValueNet(channels=channels).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


class RLMCTSAgent(MCTSAgent):
    """MCTS agent whose leaf evaluations come from a trained value network."""

    def __init__(
        self,
        checkpoint: str | Path,
        simulations: int = 500,
        exploration_weight: float = 1.4,
        rollout_depth: int = 40,
        seed: int | None = None,
        device: str = "cpu",
        name: str | None = None,
    ):
        self.device = device
        self.value_net = load_value_net(checkpoint, device=device)

        def evaluator(board: chess.Board) -> float:
            return evaluate_with_model(self.value_net, board, device=self.device)

        super().__init__(
            simulations=simulations,
            exploration_weight=exploration_weight,
            rollout_depth=rollout_depth,
            seed=seed,
            value_evaluator=evaluator,
            name=name or f"RL-MCTS(sim={simulations})",
        )


@dataclass
class TrainConfig:
    games: int = 20
    simulations: int = 60
    rollout_depth: int = 24
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 1e-3
    checkpoint: str = "value_net.pt"
    channels: int = 64
    max_plies: int = 200
    seed: int | None = 7
    device: str = "cpu"


def self_play_examples(config: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    agent = MCTSAgent(
        simulations=config.simulations,
        rollout_depth=config.rollout_depth,
        seed=config.seed,
    )
    states: list[np.ndarray] = []
    values: list[float] = []

    for game_index in range(config.games):
        board = chess.Board()
        history: list[tuple[np.ndarray, chess.Color]] = []
        ply = 0

        while not board.is_game_over(claim_draw=True) and ply < config.max_plies:
            history.append((board_to_tensor(board), board.turn))
            temperature = 1.0 if ply < 12 else 0.0
            move, _ = agent.choose_move(board, temperature=temperature)
            board.push(move)
            ply += 1

        outcome = board.outcome(claim_draw=True)
        for state, turn in history:
            states.append(state)
            values.append(value_from_outcome(outcome, turn))

        result_text = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
        print(f"self-play game {game_index + 1}/{config.games}: {result_text} in {ply} plies")

    return np.stack(states), np.array(values, dtype=np.float32)


def train_value_net(config: TrainConfig) -> Path:
    x_train, y_train = self_play_examples(config)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = ValueNet(channels=config.channels).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(config.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(config.device)
            yb = yb.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(loader), 1)
        print(f"epoch {epoch + 1}/{config.epochs}: loss={avg_loss:.4f}")

    checkpoint_path = Path(config.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "channels": config.channels,
            "training_games": config.games,
            "training_simulations": config.simulations,
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def arena(
    checkpoint: str | Path,
    games: int = 4,
    baseline_sims: int = 40,
    rl_sims: int = 40,
    rollout_depth: int = 24,
    max_plies: int = 200,
    device: str = "cpu",
) -> None:
    baseline = MCTSAgent(simulations=baseline_sims, rollout_depth=rollout_depth, seed=11)
    rl_agent = RLMCTSAgent(
        checkpoint=checkpoint,
        simulations=rl_sims,
        rollout_depth=rollout_depth,
        seed=23,
        device=device,
    )
    rl_score = 0.0

    for game_index in range(games):
        board = chess.Board()
        rl_is_white = game_index % 2 == 0
        plies = 0

        while not board.is_game_over(claim_draw=True) and plies < max_plies:
            if board.turn == rl_is_white:
                move = rl_agent.select_move(board)
            else:
                move = baseline.select_move(board)
            board.push(move)
            plies += 1

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            score = 0.0
        elif outcome.winner == rl_is_white:
            score = 1.0
        else:
            score = -1.0
        rl_score += score

        result_text = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
        print(f"game {game_index + 1}/{games}: {result_text}, rl_score={score:+.1f}, plies={plies}")

    print(f"average RL-MCTS score: {rl_score / games:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate an RL-guided MCTS chess agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the value network from MCTS self-play.")
    train.add_argument("--games", type=int, default=20)
    train.add_argument("--simulations", type=int, default=60)
    train.add_argument("--rollout-depth", type=int, default=24)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--checkpoint", default="value_net.pt")
    train.add_argument("--channels", type=int, default=64)
    train.add_argument("--max-plies", type=int, default=200)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default="cpu")

    arena_cmd = subparsers.add_parser("arena", help="Compare baseline MCTS with RL-guided MCTS.")
    arena_cmd.add_argument("--checkpoint", default="value_net.pt")
    arena_cmd.add_argument("--games", type=int, default=4)
    arena_cmd.add_argument("--baseline-sims", type=int, default=40)
    arena_cmd.add_argument("--rl-sims", type=int, default=40)
    arena_cmd.add_argument("--rollout-depth", type=int, default=24)
    arena_cmd.add_argument("--max-plies", type=int, default=200)
    arena_cmd.add_argument("--device", default="cpu")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        train_value_net(
            TrainConfig(
                games=args.games,
                simulations=args.simulations,
                rollout_depth=args.rollout_depth,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                checkpoint=args.checkpoint,
                channels=args.channels,
                max_plies=args.max_plies,
                seed=args.seed,
                device=args.device,
            )
        )
    elif args.command == "arena":
        arena(
            checkpoint=args.checkpoint,
            games=args.games,
            baseline_sims=args.baseline_sims,
            rl_sims=args.rl_sims,
            rollout_depth=args.rollout_depth,
            max_plies=args.max_plies,
            device=args.device,
        )


if __name__ == "__main__":
    main()
