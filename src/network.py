"""
Policy-value network for AlphaZero on Connect Four.

Scaled down from the paper (which used ~20-40 residual blocks for Go/Chess)
to something a T4 can train in a reasonable number of self-play iterations:
a handful of residual blocks over a small channel width is enough to learn
strong Connect Four play, since the state/action space is far smaller than
Go or Chess. Block count and width are both config-driven so a width/depth
ablation is a one-line change.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import ROWS, COLS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class PolicyValueNet(nn.Module):
    """
    Input:  (N, 2, ROWS, COLS) -- two binary planes:
              plane 0: current player's pieces
              plane 1: opponent's pieces
            (Two-plane input, rather than a single signed plane, matches
            the AlphaZero/AlphaGo Zero input convention and makes it
            trivial to add a player-color/history plane later.)
    Output: policy logits (N, COLS), value (N, 1) in [-1, 1]
    """

    def __init__(self, channels: int = 64, num_res_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * ROWS * COLS, COLS)

        # Value head
        self.value_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * ROWS * COLS, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @staticmethod
    def board_to_planes(canonical_board) -> torch.Tensor:
        """canonical_board: (ROWS, COLS) array with values in {-1, 0, 1},
        already from the current player's perspective (1 = me, -1 = opp)."""
        import numpy as np
        b = np.asarray(canonical_board)
        planes = np.stack([(b == 1).astype(np.float32),
                            (b == -1).astype(np.float32)])
        return torch.from_numpy(planes)
