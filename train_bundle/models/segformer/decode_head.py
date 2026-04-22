"""MLP Decode Head: fuses multiscale encoder features into segmentation logits.

For each of 4 feature maps:
    Linear(C_i, decoder_dim) → upsample to 1/4 resolution
Concatenate → Linear(4*decoder_dim, decoder_dim) → Linear(decoder_dim, num_classes)
Final 4x upsample to original resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecodeHead(nn.Module):
    """Lightweight all-MLP decoder for SegFormer."""

    def __init__(
        self,
        embed_dims: list[int] = (32, 64, 160, 256),
        decoder_dim: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Per-stage linear projection to decoder_dim
        self.linear_projections = nn.ModuleList([
            nn.Linear(dim, decoder_dim) for dim in embed_dims
        ])

        # Fuse concatenated features
        self.fuse = nn.Sequential(
            nn.Linear(decoder_dim * 4, decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(decoder_dim, num_classes)

    def forward(
        self, features: list[torch.Tensor], target_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
            features: List of 4 tensors [(B, D_i, H_i, W_i)]
            target_size: (H, W) of original input image
        Returns:
            logits: (B, num_classes, H, W)
        """
        # Target: 1/4 of original (same as first stage output)
        h_quarter = target_size[0] // 4
        w_quarter = target_size[1] // 4

        projected = []
        for i, feat in enumerate(features):
            B, D, H, W = feat.shape
            # Flatten spatial → apply linear → reshape
            x = feat.flatten(2).transpose(1, 2)  # (B, H*W, D)
            x = self.linear_projections[i](x)  # (B, H*W, decoder_dim)
            x = x.transpose(1, 2).reshape(B, -1, H, W)  # (B, decoder_dim, H, W)
            # Upsample to 1/4 resolution
            x = F.interpolate(x, size=(h_quarter, w_quarter), mode="bilinear", align_corners=False)
            projected.append(x)

        # Concatenate along channel dim
        x = torch.cat(projected, dim=1)  # (B, 4*decoder_dim, H/4, W/4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, 4*decoder_dim)
        x = self.fuse(x)  # (B, N, decoder_dim)
        x = self.classifier(x)  # (B, N, num_classes)
        x = x.transpose(1, 2).reshape(B, self.num_classes, H, W)  # (B, C, H/4, W/4)

        # Final upsample to original resolution
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x
