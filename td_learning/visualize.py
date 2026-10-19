"""Plot TD loss/history and a board image."""
import argparse,json,os
import matplotlib.pyplot as plt
import chess
from chess_utils import plot_board

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--history",default="outputs/history.json")
    p.add_argument("--fen",default=chess.STARTING_FEN)
    p.add_argument("--board-out",default="outputs/board.png")
    p.add_argument("--plot-out",default="outputs/td_training_curves.png")
    args=p.parse_args()
    os.makedirs(os.path.dirname(args.board_out) or ".",exist_ok=True)
    fig,_=plot_board(chess.Board(args.fen),title="Standard 8×8 chess",save_path=args.board_out)
    plt.close(fig); print("Saved",args.board_out)
    if os.path.exists(args.history):
        h=json.load(open(args.history))
        if h:
            xs=[r["iteration"] for r in h]
            fig,ax=plt.subplots(1,2,figsize=(12,4))
            ax[0].plot(xs,[r.get("td_loss") for r in h],marker="o"); ax[0].set_title("TD(0) loss"); ax[0].set_xlabel("Iteration"); ax[0].grid(True,alpha=.3)
            ax[1].plot(xs,[r.get("replay_size",0) for r in h],marker="o",label="Replay transitions")
            ax[1].plot(xs,[r.get("avg_plies",0) for r in h],marker="s",label="Average plies")
            ax[1].set_title("Training progress"); ax[1].legend(); ax[1].grid(True,alpha=.3)
            fig.tight_layout(); fig.savefig(args.plot_out,dpi=160); plt.close(fig); print("Saved",args.plot_out)
        else: print("History empty; train first.")
    else: print("No history file yet.")

if __name__=="__main__": main()
