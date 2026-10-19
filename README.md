# Chess Reinforcement Learning

An experimental standard 8x8 chess project exploring reinforcement
learning, neural board evaluation, and AlphaZero-style Monte Carlo
Tree Search (MCTS).

## Implementations

### TD(0)
A value-based agent trained using one-step temporal-difference learning.

### AlphaZero-style MCTS
A neural policy/value model combined with Monte Carlo Tree Search.

## Project Goals

- Train and evaluate a TD(0) chess agent.
- Compare agent performance through controlled tournaments.
- Investigate neural evaluation and MCTS.
- Track training progress and reproducible experiments.

## Current Status

- TD(0): training checkpoint saved through iteration 8 by the project owner.
- MCTS: implementation available; a trained baseline checkpoint has not
  yet been confirmed.
- Tournament evaluation: preliminary results require further testing.

## Reproducibility

Training configuration, dependencies, evaluation procedures, and
experiment results will be documented as the project progresses.

Model checkpoints and replay buffers are excluded from Git by default.
