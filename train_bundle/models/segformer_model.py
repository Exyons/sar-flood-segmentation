"""SegFormer model dispatcher.

Three kinds:
    kind="scratch" → from-scratch SegFormer (EfficientSelfAttention)
    kind="mla"     → from-scratch SegFormer with MLA (low-rank KV) attention
    kind="hf"      → HuggingFace SegformerForSemanticSegmentation (pretrained)

The scratch / mla variants are kept as backup. The HF path is the primary
route for pre-train + fine-tune.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer.encoder import MixTransformerEncoder
from .segformer.decode_head import MLPDecodeHead


class SegFormer(nn.Module):
    """From-scratch SegFormer-B0 scale. Used for kind=scratch and kind=mla."""

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
        target_size = (x.shape[2], x.shape[3])
        features = self.encoder(x)
        return self.decode_head(features, target_size)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_summary(self) -> str:
        total = self.count_parameters()
        variant = "MLA" if self.use_mla else "Standard"
        if total >= 1_000_000:
            return f"SegFormer ({variant}): {total / 1e6:.2f}M params"
        return f"SegFormer ({variant}): {total / 1e3:.1f}K params"


class HFSegFormer(nn.Module):
    """Wrapper around HuggingFace SegformerForSemanticSegmentation.

    HF outputs logits at H/4 x W/4. We upsample to input resolution so the
    rest of the pipeline (loss, metrics, saving) can treat it like any other
    segmentation model with full-res logits.
    """

    def __init__(self, pretrained_id: str = "nvidia/mit-b2", num_labels: int = 2):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.pretrained_id = pretrained_id
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        out = self.model(pixel_values=x)
        logits = out.logits  # (B, num_labels, H/4, W/4)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_summary(self) -> str:
        total = self.count_parameters()
        return f"SegFormer (HF {self.pretrained_id}): {total / 1e6:.2f}M params"


def build(
    kind: str = "hf",
    num_labels: int = 2,
    pretrained_id: str = "nvidia/mit-b2",
    **kwargs,
) -> nn.Module:
    """Build a SegFormer by kind.

    Args:
        kind:          "scratch" | "mla" | "hf"
        num_labels:    number of output classes (binary flood → 2)
        pretrained_id: HF hub id for kind="hf"
        **kwargs:      forwarded to SegFormer for scratch/mla (in_channels, embed_dims, etc.)
    """
    if kind == "scratch":
        kwargs.pop("use_mla", None)
        return SegFormer(num_classes=num_labels, use_mla=False, **kwargs)
    if kind == "mla":
        kwargs.pop("use_mla", None)
        return SegFormer(num_classes=num_labels, use_mla=True, **kwargs)
    if kind == "hf":
        return HFSegFormer(pretrained_id=pretrained_id, num_labels=num_labels)
    raise ValueError(f"Unknown model kind: {kind!r} (expected scratch|mla|hf)")
