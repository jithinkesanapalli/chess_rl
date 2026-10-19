"""Quick sanity checks and a simple match between two checkpoint copies is not implied."""
import argparse, torch, chess
from model import ChessNet
from mcts import MCTS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--simulations", type=int, default=32)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = ChessNet(cfg.get("channels",18), cfg.get("hidden_channels",96), cfg.get("residual_blocks",4)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    board = chess.Board()
    mcts = MCTS(model, device, args.simulations, cfg.get("c_puct",1.5))
    root = mcts.search(board)
    print("Checkpoint iteration:", ckpt.get("iteration"))
    print("Legal moves at start:", board.legal_moves.count())
    print("Search edges:", len(root.edges))
    print("Top 10 moves by visits:")
    for edge in sorted(root.edges, key=lambda e:e.visit_count, reverse=True)[:10]:
        print(f"  {board.san(edge.move):6s} visits={edge.visit_count:4d} prior={edge.prior:.4f} Q={edge.q:+.3f}")
    move = MCTS.select_move(root, temperature=0)
    print("Selected move:", board.san(move) if move else "(none)")

if __name__ == "__main__":
    main()
