"""Full SegFormer model: encoder + decode head.

Two configurations via use_mla flag:
    SegFormer(use_mla=False) → Standard ViT (EfficientSelfAttention)
    SegFormer(use_mla=True)  → MLA ViT (MLASelfAttention with low-rank KV)
"""

import torch
import torch.nn as nn

from .segformer.encoder import MixTransformerEncoder
from .segformer.decode_head import MLPDecodeHead


class SegFormer(nn.Module):
    """SegFormer-B0 scale segmentation model."""

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 2,
        embed_dims: list[int] = (32, 64, 160, 256),
        num_heads: list[int] = (1, 2, 5, 8),
        sr_ratios: list[int] = (8, 4, 2, 1),
        num_blocks: list[int] = (2, 2, 2, 2),
        mlp_ratios: list[int] = (4, 4, 4, 4),
        patch_sizes: list[int] = (4, 3, 3, 3),
        strides: list[int] = (4, 2, 2, 2),
        decoder_dim: int = 256,
        use_mla: bool = False,
        rank_divisor: int = 4,
    ):
        super().__init__()
        self.use_mla = use_mla

        self.encoder = MixTransformerEncoder(
            in_channels=in_channels,
            embed_dims=list(embed_dims),
            num_heads=list(num_heads),
            sr_ratios=list(sr_ratios),
            num_blocks=list(num_blocks),
            mlp_ratios=list(mlp_ratios),
            patch_sizes=list(patch_sizes),
            strides=list(strides),
            use_mla=use_mla,
            rank_divisor=rank_divisor,
        )

        self.decode_head = MLPDecodeHead(
            embed_dims=list(embed_dims),
            decoder_dim=decoder_dim,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — input SAR image
        Returns:
            logits: (B, num_classes, H, W)
        """
        target_size = (x.shape[2], x.shape[3])
        features = self.encoder(x)
        logits = self.decode_head(features, target_size)
        return logits

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_summary(self) -> str:
        """Human-readable parameter count."""
        total = self.count_parameters()
        variant = "MLA" if self.use_mla else "Standard"
        if total >= 1_000_000:
            return f"SegFormer ({variant}): {total / 1e6:.2f}M params"
        return f"SegFormer ({variant}): {total / 1e3:.1f}K params"
