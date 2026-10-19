# Standard 8×8 Chess RL Starter — Train, Save, Resume, Play, Visualize

This is a compact AlphaZero-style learning project for **standard chess**. It includes:
- a PyTorch residual policy/value network;
- legal-move-aware PUCT MCTS;
- self-play with root exploration noise;
- replay buffer and policy/value training;
- resumable checkpoints (model + optimizer + training history);
- terminal play against the trained agent;
- board and training-curve visualization;
- a notebook with copy/paste setup and commands.

It is a **starter implementation**, not a reproduction of DeepMind's full AlphaZero or a claim of strong playing strength. Training strength depends on game count, search budget, network size, and compute. The compact policy head maps a move to `from_square * 64 + to_square`; underpromotions share the same policy index, so this is a simplification.

## 1. Kaggle setup (recommended if you want a GPU)
1. Create a new Kaggle Notebook.
2. In Notebook settings, select **Accelerator: GPU** if available.
3. Upload `chess_rl_8x8.zip` as a notebook input, or upload/extract the files.
4. In a notebook cell, run:
   ```python
   !unzip -o /kaggle/input/YOUR_DATASET/chess_rl_8x8.zip -d /kaggle/working/chess_rl_8x8
   %cd /kaggle/working/chess_rl_8x8
   !pip install -r requirements.txt
   ```
   Replace `YOUR_DATASET` with the actual Kaggle dataset folder name. If you upload the individual files instead, put them together in one directory and `cd` into it.

## 2. Verify setup
```bash
python -c "import torch, chess; print('torch', torch.__version__); print('CUDA', torch.cuda.is_available()); print('python-chess OK')"
python visualize.py
```

## 3. Run a tiny smoke test first
This creates checkpoints and output plots. It may take a while even with small settings:
```bash
python train.py --iterations 1 --games 1 --simulations 8 --steps 2
```
Then:
```bash
python evaluate.py --checkpoint checkpoints/latest.pt --simulations 16
python visualize.py
```

## 4. Start a real run
Edit `config.py` or pass arguments. A cautious starting point:
```bash
python train.py --iterations 20 --games 4 --simulations 64 --steps 100
```
For a quick GPU smoke test, use smaller values. Increase simulations and games only after the basic run works. A first run is not expected to produce a strong chess engine.

## 5. Resume after a session disconnects
The latest model is in `checkpoints/latest.pt`; replay samples are in `checkpoints/replay.pkl`.
```bash
python train.py --resume --iterations 40 --games 4 --simulations 64 --steps 100
```
`--iterations 40` means train through iteration 40, not add 40 new iterations. Download or copy `checkpoints/` out of the temporary runtime periodically.

## 6. Play against it
```bash
python play.py --checkpoint checkpoints/latest.pt --simulations 128 --human white
```
Type moves in UCI (`e2e4`) or SAN (`Nf3`). Use `--human black` to let the engine play White.

## 7. Visualize
```bash
python visualize.py
```
Outputs:
- `outputs/board.png`: starting position;
- `outputs/training_curves.png`: loss and progress plots once history exists;
- `outputs/history.json`: per-iteration metrics.

Render a custom FEN:
```bash
python visualize.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

## 8. Important limitations and next improvements
- This is a learning scaffold, not a production-strength engine.
- It uses a compact 4096-action policy; underpromotion types share a policy index.
- The simple terminal rule treats games that hit `max_game_plies` as draws.
- Self-play-only training may take a long time to become strong; evaluate against a fixed baseline and alternate colors.
- For a fair baseline comparison, use the same time budget, enough games, and report wins/draws/losses. Do not infer strength from training loss alone.
- Keep checkpoint backups: notebook runtimes can be temporary.
