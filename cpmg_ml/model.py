from __future__ import annotations

import torch
import torch.nn as nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.film = nn.Linear(cond_dim, channels * 4)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, gamma2, beta2 = self.film(cond).chunk(4, dim=1)
        gamma1 = gamma1.unsqueeze(-1)
        beta1 = beta1.unsqueeze(-1)
        gamma2 = gamma2.unsqueeze(-1)
        beta2 = beta2.unsqueeze(-1)

        h = self.conv1(x)
        h = self.norm1(h)
        h = h * (1.0 + gamma1) + beta1
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = h * (1.0 + gamma2) + beta2
        return self.act(x + h)


class ConditionalCPMGDeJNet(nn.Module):
    """Conditional 1D residual ConvNet for CPMG profile de-J-coupling."""

    def __init__(
        self,
        input_channels: int = 2,
        metadata_dim: int = 13,
        hidden_channels: int = 64,
        metadata_hidden: int = 128,
        num_blocks: int = 6,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, metadata_hidden),
            nn.GELU(),
            nn.Linear(metadata_hidden, metadata_hidden),
            nn.GELU(),
        )
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                FiLMResidualBlock(hidden_channels, metadata_hidden, kernel_size=kernel_size)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        cond = self.metadata_encoder(metadata)
        h = self.stem(x)
        for block in self.blocks:
            h = block(h, cond)
        delta = self.head(h).squeeze(1)
        return x[:, 0, :] + delta
