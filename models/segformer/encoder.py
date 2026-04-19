"""4-stage hierarchical Mix Transformer encoder.

Each stage: OverlapPatchEmbed → N x TransformerBlock → LayerNorm
Outputs 4 feature maps at 1/4, 1/8, 1/16, 1/32 resolution.
"""

import torch
import torch.nn as nn

from .block import TransformerBlock
from .patch_embed import OverlapPatchEmbed


class MixTransformerEncoder(nn.Module):
    """Hierarchical SegFormer encoder with 4 stages."""

    def __init__(
        self,
        in_channels: int = 2,
        embed_dims: list[int] = (32, 64, 160, 256),
        num_heads: list[int] = (1, 2, 5, 8),
        sr_ratios: list[int] = (8, 4, 2, 1),
        num_blocks: list[int] = (2, 2, 2, 2),
        mlp_ratios: list[int] = (4, 4, 4, 4),
        patch_sizes: list[int] = (4, 3, 3, 3),
        strides: list[int] = (4, 2, 2, 2),
        use_mla: bool = False,
        rank_divisor: int = 4,
    ):
        super().__init__()
        self.num_stages = 4

        # Build stages
        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_stages):
            in_ch = in_channels if i == 0 else embed_dims[i - 1]
            self.patch_embeds.append(
                OverlapPatchEmbed(in_ch, embed_dims[i], patch_sizes[i], strides[i])
            )
            stage_blocks = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    sr_ratio=sr_ratios[i],
                    mlp_ratio=mlp_ratios[i],
                    use_mla=use_mla,
                    rank_divisor=rank_divisor,
                )
                for _ in range(num_blocks[i])
            ])
            self.blocks.append(stage_blocks)
            self.norms.append(nn.LayerNorm(embed_dims[i]))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) — input image
        Returns:
            List of 4 feature maps: [(B, D_i, H_i, W_i) for i in 0..3]
        """
        outputs = []

        for i in range(self.num_stages):
            x, H, W = self.patch_embeds[i](x)  # (B, N, D)

            for block in self.blocks[i]:
                x = block(x, H, W)

            x = self.norms[i](x)
            # Reshape back to spatial
            x = x.permute(0, 2, 1).reshape(-1, x.shape[2], H, W)  # (B, D, H, W)
            outputs.append(x)

        return outputs
