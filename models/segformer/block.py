"""Transformer Block: selects attention variant via use_mla flag.

LayerNorm → Attention → residual → LayerNorm → MixFFN → residual
"""

import torch
import torch.nn as nn

from .attention import EfficientSelfAttention, MLASelfAttention
from .mix_ffn import MixFFN


class TransformerBlock(nn.Module):
    """Single transformer block with configurable attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int = 1,
        mlp_ratio: int = 4,
        use_mla: bool = False,
        rank_divisor: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        if use_mla:
            self.attn = MLASelfAttention(
                dim, num_heads, sr_ratio, rank_divisor=rank_divisor
            )
        else:
            self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, expansion=mlp_ratio)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            H, W: spatial dims
        Returns:
            (B, N, D)
        """
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x
