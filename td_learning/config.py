"""Defaults for a TD-learning chess experiment."""
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 7
    channels: int = 18
    hidden_channels: int = 96
    residual_blocks: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 50_000
    self_play_games_per_iteration: int = 8
    simulations: int = 64
    c_puct: float = 1.5
    max_game_plies: int = 240
    td_updates_per_iteration: int = 200
    iterations: int = 20
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    device: str = "auto"
