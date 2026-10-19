"""Policy/value network for standard 8x8 chess."""
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))

class ChessNet(nn.Module):
    """
    Input: 18x8x8 planes.
    Outputs: 4096 from-square/to-square policy logits and a scalar value in [-1,1].
    Promotion choices share the same from-to policy index in this compact starter model.
    """
    def __init__(self, channels=18, hidden_channels=96, residual_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(residual_blocks)])
        self.policy = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, 4096),
        )
        self.value = nn.Sequential(
            nn.Conv2d(hidden_channels, 8, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.body(self.stem(x))
        return self.policy(z), self.value(z).squeeze(-1)
