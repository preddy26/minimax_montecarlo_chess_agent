# minimax_montecarlo_chess_agent

Chess AI group project implementing several game-playing agents:

- A depth-limited Minimax + alpha-beta pruning agent
- A full Monte Carlo Tree Search (MCTS) agent
- A reinforcement learning extension that trains a value network from self-play
- A competition runner for comparing agents with tournament metrics

## Project Structure

- [chess_agents/base_agent.py](chess_agents/base_agent.py): shared `ChessAgent` interface
- [chess_agents/minimax_agent.py](chess_agents/minimax_agent.py): depth-limited minimax implementation
- [chess_agents/mcts_agent.py](chess_agents/mcts_agent.py): full MCTS implementation with UCT, expansion, rollout, and backpropagation
- [chess_agents/rl_value_agent.py](chess_agents/rl_value_agent.py): RL value network, self-play training, and RL-guided MCTS
- [chess_agents/random_agent.py](chess_agents/random_agent.py): baseline random policy
- [chess_agents/competition.py](chess_agents/competition.py): tournament runner and leaderboard metrics

## Contribution Map

The minimax agent, random baseline, and tournament metrics are the core comparison framework. The MCTS/RL contribution adds:

- baseline MCTS search as a real agent instead of placeholder random behavior
- UCT selection using visit counts and value estimates
- random rollouts with material fallback evaluation
- backpropagation of win/draw/loss values through the search tree
- a PyTorch value network trained from self-play outcomes
- an RL-guided MCTS agent that uses the learned value network at search leaves

## Setup

Activate a virtual environment, then install dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not want to activate the environment, use the virtual environment Python directly.

## Run Tournament

```bash
python -m chess_agents.competition
```

This runs a small round-robin and prints:

- per-game result
- final scoreboard
- comparison metrics table for each agent

If `value_net.pt` exists in the repo root, the tournament also includes the RL-guided MCTS agent.

## Metrics for Comparing Agents

The tournament runner reports:

- `Pts`: total tournament points
- `W-D-L`: wins / draws / losses
- `Win%`: win rate
- `AsW`, `AsB`: wins as White / games as White, and wins as Black / games as Black
- `AvgMove(ms)`: average decision time per move
- `AvgBranch`: average number of legal options seen on each move
- `Cap/G`: captures per game
- `Chk/G`: checks given per game
- `CM+`, `CM-`: checkmates delivered / checkmates suffered

These metrics compare both playing strength and behavior, not just final score.

## Depth-Limited Minimax Agent

Implemented in [chess_agents/minimax_agent.py](chess_agents/minimax_agent.py):

- depth-limited minimax search
- alpha-beta pruning
- light move ordering for captures, promotions, and checks
- evaluation based on material and mobility

Tuneable knobs:

- `depth` in `MinimaxAgent(depth=...)`
- `_evaluate_board()` weights

## Monte Carlo Tree Search Agent

Implemented in [chess_agents/mcts_agent.py](chess_agents/mcts_agent.py):

- selection with UCT-style exploration/exploitation
- expansion by adding an unvisited legal move
- rollout simulation until terminal state or depth cutoff
- material-based fallback when a rollout reaches the cutoff
- backpropagation of values through the tree
- final move choice by root child visit count

Example:

```python
import chess
from chess_agents.mcts_agent import MCTSAgent

board = chess.Board()
agent = MCTSAgent(simulations=100, rollout_depth=18, seed=7)
move = agent.select_move(board)
print(move)
```

## Reinforcement Learning Extension

Implemented in [chess_agents/rl_value_agent.py](chess_agents/rl_value_agent.py):

- encodes board positions into 13 planes
- generates training examples from MCTS self-play
- trains a small convolutional value network
- saves a checkpoint such as `value_net.pt`
- loads the checkpoint into `RLMCTSAgent`
- compares baseline MCTS against RL-guided MCTS

Train the value network:

```bash
python -m chess_agents.rl_value_agent train --games 10 --simulations 40 --epochs 3 --checkpoint value_net.pt
```

Compare baseline MCTS against RL-guided MCTS:

```bash
python -m chess_agents.rl_value_agent arena --checkpoint value_net.pt --games 4 --baseline-sims 40 --rl-sims 40
```

The RL component is intentionally lightweight. It is not a full AlphaZero implementation because it learns only a value function, not a policy network. It still satisfies the reinforcement learning extension because the agent improves its evaluation function from self-play experience and then uses that learned evaluation inside MCTS.
