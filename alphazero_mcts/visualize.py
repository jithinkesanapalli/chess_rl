"""Plot training history and render a chess position."""
import argparse, json, os
import chess
import matplotlib.pyplot as plt
from chess_utils import plot_board

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", default="outputs/history.json")
    p.add_argument("--fen", default=chess.STARTING_FEN)
    p.add_argument("--board-out", default="outputs/board.png")
    p.add_argument("--plot-out", default="outputs/training_curves.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.board_out) or ".", exist_ok=True)
    board = chess.Board(args.fen)
    fig, _ = plot_board(board, title="Standard 8×8 chess position", save_path=args.board_out)
    plt.close(fig)
    print("Saved board:", args.board_out)
    if os.path.exists(args.history):
        with open(args.history) as f: history = json.load(f)
        if history:
            xs = [r["iteration"] for r in history]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for key in ("loss", "policy_loss", "value_loss"):
                ys = [r.get(key) for r in history]
                axes[0].plot(xs, ys, marker="o", label=key)
            axes[0].set_title("Training losses"); axes[0].set_xlabel("Iteration"); axes[0].legend(); axes[0].grid(True, alpha=.3)
            axes[1].plot(xs, [r.get("replay_size",0) for r in history], marker="o", label="Replay positions")
            axes[1].plot(xs, [r.get("avg_plies",0) for r in history], marker="s", label="Average game plies")
            axes[1].set_title("Training progress"); axes[1].set_xlabel("Iteration"); axes[1].legend(); axes[1].grid(True, alpha=.3)
            fig.tight_layout(); fig.savefig(args.plot_out, dpi=160); plt.close(fig)
            print("Saved training curves:", args.plot_out)
        else: print("History file is empty; train first.")
    else: print("No history yet. Train first to create", args.history)

if __name__ == "__main__":
    main()
