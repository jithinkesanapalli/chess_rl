"""Train an AlphaZero-style policy/value model by self-play on standard 8x8 chess."""
import argparse, os, random, json, time
import numpy as np
import torch
import torch.nn.functional as F
import chess
from config import Config
from model import ChessNet
from mcts import MCTS
from chess_utils import encode_board
from replay import ReplayBuffer

def choose_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def self_play_game(model, device, cfg):
    board = chess.Board()
    samples = []
    mcts = MCTS(model, device, cfg.mcts_simulations, cfg.c_puct)
    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < cfg.max_game_plies:
        root = mcts.search(board, add_root_noise=True)
        temp = 1.0 if ply < cfg.temperature_moves else 0.0
        policy = MCTS.visit_policy(root, temperature=temp)
        samples.append((encode_board(board), policy, board.turn))
        move = MCTS.select_move(root, temperature=temp)
        if move is None:
            break
        board.push(move)
        ply += 1
    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
        white_result = 1.0 if result == "1-0" else (-1.0 if result == "0-1" else 0.0)
    else:
        # Treat max-ply truncation as a draw to prevent runaway self-play games.
        white_result = 0.0
    finished = []
    for position, policy, turn in samples:
        value = white_result if turn == chess.WHITE else -white_result
        finished.append((position, policy, np.float32(value)))
    return finished, white_result, ply

def train_batch(model, optimizer, batch, device):
    states = torch.from_numpy(np.stack([x[0] for x in batch])).to(device)
    target_policies = torch.from_numpy(np.stack([x[1] for x in batch])).to(device)
    target_values = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=device)
    logits, values = model(states)
    policy_loss = -(target_policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(values, target_values)
    loss = policy_loss + value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())

def save_checkpoint(path, model, optimizer, iteration, cfg, history, replay_path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "config": cfg.__dict__,
        "history": history,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "replay_path": replay_path,
    }, temp)
    os.replace(temp, path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--games", type=int, default=None)
    p.add_argument("--simulations", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    cfg = Config()
    if args.iterations is not None: cfg.iterations = args.iterations
    if args.games is not None: cfg.self_play_games_per_iteration = args.games
    if args.simulations is not None: cfg.mcts_simulations = args.simulations
    if args.steps is not None: cfg.training_steps_per_iteration = args.steps
    device = choose_device(args.device if args.device else cfg.device)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True); os.makedirs(cfg.output_dir, exist_ok=True)

    model = ChessNet(cfg.channels, cfg.hidden_channels, cfg.residual_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    replay = ReplayBuffer(cfg.replay_capacity)
    history = []
    start_iteration = 1
    checkpoint_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
    replay_path = os.path.join(cfg.checkpoint_dir, "replay.pkl")

    if args.resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        history = ckpt.get("history", [])
        start_iteration = int(ckpt.get("iteration", 0)) + 1
        if os.path.exists(replay_path): replay.load(replay_path)
        print(f"Resumed from iteration {start_iteration-1}; replay positions={len(replay)}")
    print(f"Device: {device}; CUDA available: {torch.cuda.is_available()}")
    print(f"Starting at iteration {start_iteration}; model parameters={sum(p.numel() for p in model.parameters()):,}")

    for iteration in range(start_iteration, cfg.iterations + 1):
        t0 = time.time()
        model.eval()
        new_samples, outcomes, lengths = [], [], []
        for game_i in range(cfg.self_play_games_per_iteration):
            samples, outcome, length = self_play_game(model, device, cfg)
            new_samples.extend(samples); outcomes.append(outcome); lengths.append(length)
            print(f"  self-play {game_i+1}/{cfg.self_play_games_per_iteration}: positions={len(samples)}, result={outcome:+.0f}, plies={length}")
        replay.add_many(new_samples)
        model.train()
        losses, policy_losses, value_losses = [], [], []
        if len(replay) > 0:
            for _ in range(cfg.training_steps_per_iteration):
                batch = replay.sample(cfg.batch_size)
                l, pl, vl = train_batch(model, optimizer, batch, device)
                losses.append(l); policy_losses.append(pl); value_losses.append(vl)
        record = {
            "iteration": iteration,
            "positions_added": len(new_samples),
            "replay_size": len(replay),
            "loss": float(np.mean(losses)) if losses else None,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else None,
            "value_loss": float(np.mean(value_losses)) if value_losses else None,
            "white_score": float(np.mean([(x+1)/2 for x in outcomes])) if outcomes else None,
            "avg_plies": float(np.mean(lengths)) if lengths else None,
            "seconds": time.time()-t0,
        }
        history.append(record)
        replay.save(replay_path)
        save_checkpoint(checkpoint_path, model, optimizer, iteration, cfg, history, replay_path)
        save_checkpoint(os.path.join(cfg.checkpoint_dir, f"iteration_{iteration:04d}.pt"), model, optimizer, iteration, cfg, history, replay_path)
        with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print("  saved checkpoint + replay; " + json.dumps(record))
    print(f"Training complete. Latest checkpoint: {checkpoint_path}")
    print("Plot results with: python visualize.py --history outputs/history.json")

if __name__ == "__main__":
    main()
