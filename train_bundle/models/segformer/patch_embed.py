"""Overlapping Patch Embedding for SegFormer.

Stage 1: patch_size=4, stride=4 (non-overlapping)
Stages 2-4: patch_size=3, stride=2 (overlapping, for downsampling)
"""

import torch
import torch.nn as nn


class OverlapPatchEmbed(nn.Module):
    """Project image patches to embedding space via Conv2d + LayerNorm."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 4,
        stride: int = 4,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, N, D) where N = H' * W'
            H': output height
            W': output width
        """
        x = self.proj(x)  # (B, D, H', W')
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x, H, W
