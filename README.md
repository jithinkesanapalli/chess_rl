# Chess Reinforcement Learning ♟️

An experimental standard 8×8 chess AI project exploring value-based
reinforcement learning and AlphaZero-style neural Monte Carlo Tree Search.

**Repository:** https://github.com/jithinkesanapalli/chess_rl

## Project Overview

This project contains two separate chess AI implementations.

### 1. TD(0) Reinforcement Learning

The TD(0) agent learns a state-value function using one-step temporal-difference
updates. Its value network estimates the expected outcome from the perspective
of the side to move.

Implementation: [`td_learning/`](td_learning/)

### 2. AlphaZero-Style MCTS

The MCTS implementation combines a neural policy/value network with
Monte Carlo Tree Search (MCTS). The policy head guides move exploration,
while the value head estimates positions.

Implementation: [`alphazero_mcts/`](alphazero_mcts/)

This is an experimental implementation and should not be assumed to match
the playing strength of a mature chess engine.

## Repository Structure

```text
chess_rl/
├── README.md
├── .gitignore
├── td_learning/
│   ├── td_learning.py
│   ├── model.py
│   ├── config.py
│   ├── chess_utils.py
│   ├── play.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── requirements.txt
└── alphazero_mcts/
    ├── train.py
    ├── model.py
    ├── config.py
    ├── mcts.py
    ├── chess_utils.py
    ├── replay.py
    ├── play.py
    ├── evaluate.py
    ├── visualize.py
    └── requirements.txt
```

## Requirements

- Python 3
- PyTorch
- python-chess
- NumPy
- Matplotlib

The individual implementation directories contain their own
`requirements.txt` files.

## Installation

Clone the repository:

```bash
git clone https://github.com/jithinkesanapalli/chess_rl.git
cd chess_rl
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the TD(0) dependencies:

```bash
pip install -r td_learning/requirements.txt
```

For the MCTS implementation, install its dependencies as well:

```bash
pip install -r alphazero_mcts/requirements.txt
```

## Training the TD(0) Agent

From the TD(0) directory:

```bash
cd td_learning
python td_learning.py
```

To resume training from an existing `checkpoints/latest.pt` checkpoint
and continue toward iteration 50, with one self-play game and five TD
updates per iteration:

```bash
python td_learning.py --resume --iterations 50 --games 1 --updates 5
```

**Important:** `--resume` requires a compatible checkpoint at the configured
checkpoint path. Keep the checkpoint and replay buffer together when resuming.
The iteration target is 50; this command does not guarantee that the agent
will become stronger.

## Training the MCTS Agent

From the MCTS directory:

```bash
cd alphazero_mcts
python train.py
```

Use the training configuration and script options in that directory
to configure further experiments. A trained MCTS checkpoint is required
to evaluate a trained neural MCTS agent; a randomly initialized network
is not a trained baseline.

## Playing and Evaluation

Each implementation includes play, evaluation, and visualization scripts.
Run them from their respective directories and consult their script options
before launching an experiment.

Evaluation should record wins, draws, losses, score percentage, and game
termination reasons. Compare agents using consistent rules and settings,
and distinguish genuine draws from games stopped at a move limit.

## Checkpoints and Reproducibility

Model checkpoints and replay buffers are excluded from Git by default to
avoid accidentally committing large training artifacts.

For reproducible experiments, record:

- Checkpoint and training iteration
- Configuration and random seed
- Training loss and replay-buffer size
- Evaluation opponents and game count
- Wins, draws, losses, and termination reasons
- Software and hardware environment

A trained model will be published or linked with instructions for downloading
and loading it once a suitable checkpoint has been selected.

## Current Status

- Source code for both implementations is in the repository.
- The TD(0) project has been trained in iterations; a saved iteration 8
  checkpoint was previously reported by the project owner.
- The next training milestone is iteration 50.
- Tournament evaluation is preliminary; stronger playing ability has not
  yet been established through sufficient testing.
- The first selected trained checkpoint and its loading instructions
  will be added after training and validation.

## Roadmap

- [x] Create the GitHub repository and upload the source code.
- [x] Add initial project documentation.
- [ ] Resume TD(0) training and reach iteration 50.
- [ ] Preserve and validate the selected trained checkpoint.
- [ ] Publish the trained agent and inference instructions.
- [ ] Run a reproducible tournament and document the results.
- [ ] Continue improving the agents through tracked commits.

## License

No license has been specified yet. Until a license is added, do not assume
that the repository grants permission to reuse or redistribute its code.
