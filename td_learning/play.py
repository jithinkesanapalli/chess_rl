"""Play a terminal game against the TD-trained value network."""
import argparse, torch, chess
from model import ValueNet
from chess_utils import encode_board

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/latest.pt")
    p.add_argument("--human",choices=["white","black"],default="white")
    p.add_argument("--device",default="auto")
    args=p.parse_args()
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device))
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    cfg=ckpt.get("config",{})
    model=ValueNet(cfg.get("channels",18),cfg.get("hidden_channels",96),cfg.get("residual_blocks",4)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    board=chess.Board(); human=chess.WHITE if args.human=="white" else chess.BLACK
    print("Moves: UCI e2e4 or SAN Nf3; type quit to stop.")
    while not board.is_game_over(claim_draw=True):
        print(board,"\n",board.fen(),"\n")
        if board.turn==human:
            raw=input("Your move: ").strip()
            if raw.lower() in ("quit","exit"): break
            try:
                try: move=chess.Move.from_uci(raw)
                except ValueError: move=board.parse_san(raw)
                if move not in board.legal_moves: raise ValueError("illegal move")
            except Exception as e: print("Invalid move:",e); continue
        else:
            candidates=[]
            for m in board.legal_moves:
                b=board.copy(); b.push(m)
                with torch.no_grad():
                    x=torch.from_numpy(encode_board(b)).unsqueeze(0).to(device)
                    v=float(model(x)[0].item())
                # The value is from the opponent's side-to-move perspective.
                candidates.append((-v,m))
            move=max(candidates,key=lambda x:x[0])[1]
            print("Engine:",board.san(move))
        board.push(move)
    print(board); print("Result:",board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "stopped")

if __name__=="__main__": main()
