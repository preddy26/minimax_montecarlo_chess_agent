# minimax_montecarlo_chess_agent

Chess AI project with:
- A depth-limited Minimax + alpha-beta pruning agent
- Monte Carlo Tree Search (MCTS) agent
- A competition runner tournaments across multiple agents

## Project Structure

- [chess_agents/base_agent.py](chess_agents/base_agent.py): shared interface (`ChessAgent`)
- [chess_agents/minimax_agent.py](chess_agents/minimax_agent.py): depth-limited minimax implementation
- [chess_agents/mcts_agent.py](chess_agents/mcts_agent.py): MCTS scaffold (placeholder behavior)
- [chess_agents/random_agent.py](chess_agents/random_agent.py): baseline random policy
- [chess_agents/competition.py](chess_agents/competition.py): run games + leaderboard

## Setup

Activate virtual environment, then install dependencies:

source .venv/bin/activate
pip install -r requirements.txt

If you do not want to activate the environment, use the venv Python directly.

## Run Tournament



python -m chess_agents.competition


This runs a small round-robin and prints:
- per-game result
- final scoreboard 
- comparison metrics table for each agent

## Metrics for Comparing Agents

The tournament runner reports a metrics table with:

- `Pts`: total tournament points
- `W-D-L`: wins / draws / losses
- `Win%`: win rate
- `AsW`, `AsB`: wins as White / games as White, and wins as Black / games as Black
- `AvgMove(ms)`: average decision time per move
- `AvgBranch`: average number of legal options seen on each move (search context)
- `Cap/G`: captures per game
- `Chk/G`: checks given per game
- `CM+`, `CM-`: checkmates delivered / checkmates suffered

These metrics make it easier to compare style and efficiency, not just final score.

## Depth-Limited Minimax Agent

Implemented in [chess_agents/minimax_agent.py](chess_agents/minimax_agent.py):
- depth-limited search
- alpha-beta pruning
- light move ordering (captures/promotions/checks)
- evaluation based on material + mobility

Tuneable knobs:
- `depth` in `MinimaxAgent(depth=...)`
- `_evaluate_board()` weights

## Monte Carlo Tree Search (MCTS) Agent

Use [chess_agents/mcts_agent.py](chess_agents/mcts_agent.py) to add:
- UCT selection: maximize $Q(s, a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}}$
- expansion on first unvisited legal move
- rollout policy (random or heuristic)
- backpropagate win/draw/loss value

Then register that agent in [chess_agents/competition.py](chess_agents/competition.py) `main()`.
