"""Mix-FFN: Feed-forward with depthwise convolution for positional encoding.

Replaces standard FFN + positional encoding. The 3x3 DWConv injects
spatial locality, removing need for explicit positional embeddings.
"""

import torch
import torch.nn as nn


class MixFFN(nn.Module):
    """Linear → DWConv3x3 → GELU → Linear."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(
            hidden, hidden, kernel_size=3, padding=1, groups=hidden
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            H, W: spatial dims (N = H * W)
        Returns:
            (B, N, D)
        """
        B, N, D = x.shape
        x = self.fc1(x)  # (B, N, hidden)
        # Reshape to spatial for DWConv
        x = x.transpose(1, 2).reshape(B, -1, H, W)  # (B, hidden, H, W)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden)
        x = self.fc2(x)  # (B, N, D)
        return x
