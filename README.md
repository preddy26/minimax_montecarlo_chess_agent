# minimax_montecarlo_chess_agent

## Overview
This group project builds and compares multiple AI agents for chess. The repository includes a depth-limited **Minimax** agent with alpha-beta pruning, a full **Monte Carlo Tree Search (MCTS)** agent, a lightweight **reinforcement learning** extension, and a **competition runner** for evaluating agent performance in tournaments.

## Project Structure
- `base_agent.py` defines the shared `ChessAgent` interface.
- `minimax_agent.py` implements depth-limited minimax with alpha-beta pruning, move ordering, and a material/mobility evaluation function.
- `mcts_agent.py` implements UCT-based Monte Carlo Tree Search with selection, expansion, rollout simulation, and backpropagation.
- `rl_value_agent.py` adds a self-play value-network extension and an RL-guided MCTS agent.
- `random_agent.py` provides a baseline random agent.
- `competition.py` runs round-robin tournaments and reports leaderboard statistics.

## Features
The Minimax agent provides a strong classical search baseline. The MCTS agent explores moves through repeated simulations and visit-based selection. The RL extension trains a value network from self-play and uses the learned evaluation inside MCTS to improve leaf evaluation. While this is not a full AlphaZero-style implementation, it demonstrates how reinforcement learning can enhance search quality.

## Evaluation
The competition framework reports not only results, but also agent behavior. Metrics include total points, win rate, wins as White and Black, average move time, average branching factor, captures per game, checks per game, and checkmates delivered or suffered.

## Running the Project
Install the required dependencies, then run:

`python -m chess_agents.competition`

If `value_net.pt` is available, the RL-guided MCTS agent is included automatically.

## Slide Presentation
https://docs.google.com/presentation/d/1qkFwfgyt430IC3tnFxYn2KuTjF0blVq9l67VDSMj0ZA/edit?usp=sharing
