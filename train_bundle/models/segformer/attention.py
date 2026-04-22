"""Two attention implementations for SegFormer comparison.

EfficientSelfAttention — Standard ViT with spatial reduction.
MLASelfAttention — MLA-style low-rank KV compression + spatial reduction.

Both share the same forward(x, H, W) signature for interchangeability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientSelfAttention(nn.Module):
    """Standard efficient self-attention with spatial reduction on K, V.

    Params per layer: 4 * D * D (Q, K, V, out projections).
    """

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

        # Spatial reduction for K, V
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) where N = H * W
            H, W: spatial dimensions
        Returns:
            (B, N, D)
        """
        B, N, D = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # q: (B, heads, N, head_dim)

        if self.sr_ratio > 1:
            # Reshape to spatial, apply SR, flatten back
            x_sr = x.permute(0, 2, 1).reshape(B, D, H, W)  # (B, D, H, W)
            x_sr = self.sr(x_sr).flatten(2).transpose(1, 2)  # (B, N_sr, D)
            x_sr = self.sr_norm(x_sr)
        else:
            x_sr = x

        k = self.k(x_sr).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x_sr).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # k, v: (B, heads, N_sr, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N_sr)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        return self.out(out)


class MLASelfAttention(nn.Module):
    """MLA-style attention with low-rank KV compression + spatial reduction.

    Instead of separate W_K (D→D) and W_V (D→D), uses:
        W_down: D → d_c  (shared compression)
        W_K_up: d_c → D  (key up-projection)
        W_V_up: d_c → D  (value up-projection)

    Where d_c = D // rank_divisor (default 4).

    Params per layer: D*D + D*d_c + 2*d_c*D + D*D
    Fewer than standard when d_c < D.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int = 1,
        rank_divisor: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.d_c = dim // rank_divisor  # compressed dimension

        self.q = nn.Linear(dim, dim)
        # Low-rank KV: shared down-projection, separate up-projections
        self.kv_down = nn.Linear(dim, self.d_c)
        self.k_up = nn.Linear(self.d_c, dim)
        self.v_up = nn.Linear(self.d_c, dim)
        self.out = nn.Linear(dim, dim)

        # Spatial reduction
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            H, W: spatial dimensions
        Returns:
            (B, N, D)
        """
        B, N, D = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_sr = x.permute(0, 2, 1).reshape(B, D, H, W)
            x_sr = self.sr(x_sr).flatten(2).transpose(1, 2)
            x_sr = self.sr_norm(x_sr)
        else:
            x_sr = x

        # Low-rank KV path
        compressed = self.kv_down(x_sr)  # (B, N_sr, d_c)
        k = self.k_up(compressed).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_up(compressed).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)
