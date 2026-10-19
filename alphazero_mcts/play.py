"""Play against the trained network/MCTS agent in the terminal."""
import argparse, torch, chess
from model import ChessNet
from mcts import MCTS
from chess_utils import plot_board

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--simulations", type=int, default=128)
    p.add_argument("--human", choices=["white", "black"], default="white")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = ChessNet(cfg.get("channels",18), cfg.get("hidden_channels",96), cfg.get("residual_blocks",4)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    board = chess.Board()
    human_color = chess.WHITE if args.human == "white" else chess.BLACK
    print("Enter moves in UCI (e2e4) or SAN (Nf3). Type 'quit' to stop.")
    while not board.is_game_over(claim_draw=True):
        print(board, "\n", board.fen(), "\n")
        if board.turn == human_color:
            raw = input("Your move: ").strip()
            if raw.lower() in {"quit", "exit"}: break
            try:
                try: move = chess.Move.from_uci(raw)
                except ValueError: move = board.parse_san(raw)
                if move not in board.legal_moves: raise ValueError("illegal move")
            except Exception as e:
                print(f"Invalid move: {e}"); continue
        else:
            search = MCTS(model, device, args.simulations, float(cfg.get("c_puct",1.5)))
            root = search.search(board)
            move = MCTS.select_move(root, temperature=0)
            if move is None: break
            print("Engine:", board.san(move))
        board.push(move)
    print(board)
    print("Result:", board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "game stopped")

if __name__ == "__main__":
    main()
