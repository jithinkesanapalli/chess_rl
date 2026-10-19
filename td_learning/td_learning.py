"""
One-step TD(0) learning for standard 8x8 chess.

Value convention:
  V(s) is the expected outcome from the perspective of the side to move in s.
  After a legal move, side to move switches, so the bootstrap target is
  reward - gamma * V(s_next) for the previous player, or equivalently
  -gamma * V(s_next) for zero-reward nonterminal chess moves.
Terminal outcome is returned from the perspective of the side to move in
the terminal board (the player who has no legal move).
"""
import argparse, os, json, random, time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import chess
from config import Config
from model import ValueNet
from chess_utils import encode_board

def choose_device(name):
    if name != "auto": return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def terminal_value_for_side_to_move(board):
    """Return terminal value from current side-to-move perspective."""
    result = board.result(claim_draw=True)
    if result == "1/2-1/2": return 0.0
    winner = chess.WHITE if result == "1-0" else chess.BLACK
    return 1.0 if board.turn == winner else -1.0

def play_self_game(model, device, cfg):
    """Generate transitions using epsilon-random legal moves and value-guided one-ply selection."""
    board = chess.Board()
    transitions = []
    ply = 0
    model.eval()
    while not board.is_game_over(claim_draw=True) and ply < cfg.max_game_plies:
        legal = list(board.legal_moves)
        if not legal: break
        # Explore sometimes. Otherwise choose the move whose successor has best value
        # from the current player's perspective. This is intentionally simple starter play.
        if random.random() < 0.20:
            move = random.choice(legal)
        else:
            candidates = []
            for move0 in legal:
                b2 = board.copy()
                b2.push(move0)
                if b2.is_game_over(claim_draw=True):
                    # A terminal successor's current side-to-move value is negated to
                    # convert it back to the player who just moved.
                    v_next = terminal_value_for_side_to_move(b2)
                else:
                    with torch.no_grad():
                        x = torch.from_numpy(encode_board(b2)).unsqueeze(0).to(device)
                        v_next = float(model(x)[0].item())
                current_player_value = -v_next
                candidates.append(current_player_value)
            best = max(candidates)
            best_ids = [i for i, v in enumerate(candidates) if abs(v-best) < 1e-8]
            move = legal[random.choice(best_ids)]
        s = encode_board(board)
        player = board.turn
        board.push(move)
        s_next = encode_board(board)
        done = board.is_game_over(claim_draw=True) or ply + 1 >= cfg.max_game_plies
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            white_result = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
            # Outcome from perspective of player who made this move.
            outcome = white_result if player == chess.WHITE else -white_result
            reward = outcome
        else:
            reward = 0.0
        transitions.append((s, reward, s_next, done))
        ply += 1
    # The terminal transition gets the game result as reward; max-ply truncation is a draw.
    if transitions and not board.is_game_over(claim_draw=True):
        s, _, sn, _ = transitions[-1]
        transitions[-1] = (s, 0.0, sn, True)
    return transitions, board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "truncated_draw", ply

def update_td0(model, optimizer, batch, device, gamma):
    states = torch.from_numpy(np.stack([t[0] for t in batch])).to(device)
    rewards = torch.tensor([t[1] for t in batch], dtype=torch.float32, device=device)
    next_states = torch.from_numpy(np.stack([t[2] for t in batch])).to(device)
    dones = torch.tensor([t[3] for t in batch], dtype=torch.float32, device=device)
    # Stop gradients through the bootstrap target. Values are from side-to-move perspective;
    # every legal move flips the player, hence the minus sign.
    model.train()
    current_values = model(states)
    with torch.no_grad():
        next_values = model(next_states)
        targets = rewards - gamma * (1.0 - dones) * next_values
    loss = F.mse_loss(current_values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--games", type=int, default=None)
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    cfg = Config()
    if args.iterations is not None: cfg.iterations = args.iterations
    if args.games is not None: cfg.self_play_games_per_iteration = args.games
    if args.updates is not None: cfg.td_updates_per_iteration = args.updates
    device = choose_device(args.device)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True); os.makedirs(cfg.output_dir, exist_ok=True)
    model = ValueNet(cfg.channels, cfg.hidden_channels, cfg.residual_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    replay = deque(maxlen=cfg.replay_capacity)
    history, start = [], 1
    ckpt_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
    replay_path = os.path.join(cfg.checkpoint_dir, "td_replay.npz")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        history = ckpt.get("history", []); start = int(ckpt.get("iteration",0))+1
        if os.path.exists(replay_path):
            data = np.load(replay_path, allow_pickle=False)
            for i in range(len(data["rewards"])):
                replay.append((data["states"][i], float(data["rewards"][i]), data["next_states"][i], bool(data["dones"][i])))
        print(f"Resumed at iteration {start}; replay transitions={len(replay)}")
    print(f"Device={device}; parameters={sum(p.numel() for p in model.parameters()):,}")
    for iteration in range(start, cfg.iterations+1):
        t0 = time.time()
        new, results, lengths = [], [], []
        for gi in range(cfg.self_play_games_per_iteration):
            trans, result, length = play_self_game(model, device, cfg)
            new.extend(trans); results.append(result); lengths.append(length)
            print(f"  game {gi+1}/{cfg.self_play_games_per_iteration}: transitions={len(trans)} result={result} plies={length}")
        replay.extend(new)
        losses = []
        if len(replay) >= 2:
            for _ in range(cfg.td_updates_per_iteration):
                batch = random.sample(list(replay), min(cfg.batch_size, len(replay)))
                losses.append(update_td0(model, optimizer, batch, device, cfg.gamma))
        record = {
            "iteration": iteration, "new_transitions": len(new), "replay_size": len(replay),
            "td_loss": float(np.mean(losses)) if losses else None,
            "avg_plies": float(np.mean(lengths)) if lengths else None,
            "results": {r: results.count(r) for r in sorted(set(results))},
            "seconds": time.time()-t0
        }
        history.append(record)
        # Save replay as fixed-shape arrays for portability.
        if replay:
            np.savez_compressed(replay_path,
                states=np.stack([x[0] for x in replay]),
                rewards=np.asarray([x[1] for x in replay], dtype=np.float32),
                next_states=np.stack([x[2] for x in replay]),
                dones=np.asarray([x[3] for x in replay], dtype=np.bool_))
        temp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "iteration": iteration, "config": cfg.__dict__, "history": history}, temp)
        os.replace(temp, ckpt_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "iteration": iteration, "config": cfg.__dict__, "history": history},
                   os.path.join(cfg.checkpoint_dir, f"iteration_{iteration:04d}.pt"))
        with open(os.path.join(cfg.output_dir, "history.json"), "w") as f: json.dump(history, f, indent=2)
        print("  saved:", json.dumps(record))
    print("Done. Checkpoint:", ckpt_path)

if __name__ == "__main__": main()
