"""Quick visualizer for a GEE-exported S1 + Label tile pair.

Usage:
    uv run python -m gee.python.viz_tile \
        --s1    data/assam2022_x000_y000_S1.tif \
        --label data/assam2022_x000_y000_Label.tif \
        --out   data/assam2022_x000_y000_viz.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def _clip_minmax(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)


def visualize(s1_path: str, label_path: str, out_path: str) -> None:
    with rasterio.open(s1_path) as src:
        sar = src.read().astype(np.float32)  # (2, H, W)
        s1_bounds = src.bounds
        s1_crs = src.crs
    vv, vh = sar[0], sar[1]

    with rasterio.open(label_path) as src:
        label = src.read(1)
        lbl_bounds = src.bounds

    # Stats
    print(f"S1 bounds:    {s1_bounds}  CRS {s1_crs}")
    print(f"Label bounds: {lbl_bounds}")
    print(f"VV dB        : min {np.nanmin(vv):.2f}  max {np.nanmax(vv):.2f}  mean {np.nanmean(vv):.2f}")
    print(f"VH dB        : min {np.nanmin(vh):.2f}  max {np.nanmax(vh):.2f}  mean {np.nanmean(vh):.2f}")

    uniq, counts = np.unique(label, return_counts=True)
    total = label.size
    print("Label distribution:")
    for v, c in zip(uniq, counts):
        print(f"  {int(v):>3d}: {c:>10d}  ({100 * c / total:5.2f}%)")

    # Normalized views for display
    vv_n = _clip_minmax(vv, -23.0, 0.0)
    vh_n = _clip_minmax(vh, -28.0, -5.0)
    diff_n = _clip_minmax(vv - vh, -15.0, 15.0)

    # Flood overlay on VV
    flood_mask = (label == 1)
    dry_mask = (label == 0)
    ignore_mask = (label == -1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(vv_n, cmap="gray")
    axes[0, 0].set_title("VV (clipped -23..0 dB)")
    axes[0, 1].imshow(vh_n, cmap="gray")
    axes[0, 1].set_title("VH (clipped -28..-5 dB)")
    axes[0, 2].imshow(diff_n, cmap="gray")
    axes[0, 2].set_title("VV - VH (clipped -15..15)")

    axes[1, 0].imshow(label, cmap="viridis", vmin=-1, vmax=1)
    axes[1, 0].set_title(f"Label (flood={flood_mask.sum()}, dry={dry_mask.sum()}, ignore={ignore_mask.sum()})")

    # Flood highlighted on VV
    rgb = np.stack([vv_n, vv_n, vv_n], axis=-1)
    rgb[flood_mask] = [1.0, 0.2, 0.2]
    rgb[ignore_mask] = [0.5, 0.5, 0.0]
    axes[1, 1].imshow(rgb)
    axes[1, 1].set_title("VV + flood (red) + ignore (yellow)")

    # Histogram
    axes[1, 2].hist(vv.ravel(), bins=100, alpha=0.5, label="VV", color="C0")
    axes[1, 2].hist(vh.ravel(), bins=100, alpha=0.5, label="VH", color="C1")
    axes[1, 2].set_title("SAR dB histogram")
    axes[1, 2].set_xlabel("dB")
    axes[1, 2].legend()

    for ax in axes.flat[:5]:
        ax.axis("off")

    fig.suptitle(f"{Path(s1_path).stem} / {Path(label_path).stem}", fontsize=12)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", default="data/tile_viz.png")
    args = parser.parse_args()
    visualize(args.s1, args.label, args.out)


if __name__ == "__main__":
    main()
