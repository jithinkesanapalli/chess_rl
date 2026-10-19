"""Inspect value estimates on the starting position and candidate moves."""
import argparse,torch,chess
from model import ValueNet
from chess_utils import encode_board

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="checkpoints/latest.pt")
    p.add_argument("--device",default="auto")
    args=p.parse_args()
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device))
    ck=torch.load(args.checkpoint,map_location=device,weights_only=False); cfg=ck.get("config",{})
    net=ValueNet(cfg.get("channels",18),cfg.get("hidden_channels",96),cfg.get("residual_blocks",4)).to(device)
    net.load_state_dict(ck["model"]); net.eval(); b=chess.Board()
    scored=[]
    for m in b.legal_moves:
        b2=b.copy(); b2.push(m)
        with torch.no_grad(): v=float(net(torch.from_numpy(encode_board(b2)).unsqueeze(0).to(device))[0])
        scored.append((-v,m))
    print("Iteration:",ck.get("iteration"))
    print("Value estimates are from current side-to-move perspective; move score is -V(next).")
    for score,m in sorted(scored,reverse=True)[:10]: print(f"{b.san(m):6s} score={score:+.4f}")
if __name__=="__main__": main()
