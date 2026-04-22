"""Empirical effective receptive field (ERF) for scratch vs MLA SegFormer.

Implements Luo et al. 2016 ("Understanding the Effective Receptive Field
in Deep CNNs"): backprop a unit gradient from the center output pixel's
flood-class logit to the input, take ``|grad|.mean(channels)``, average
over M val tiles, normalize. Larger ERF means aggregation over a broader
context. Low-rank KV (MLA) is expected to compress this.

Outputs (``reports/figures/xai/`` by default):
    erf_scratch.png    heatmap + 50/25/10 % radius contours
    erf_mla.png        same
    erf_compare.png    side-by-side + ratio text
    erf_radial.png     1-D angular average, log-y
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import Sen1FloodsDataset  # noqa: E402
from evaluate import _build_from_ckpt  # noqa: E402


DPI = 150


# ---------------------------------------------------------------------------
# ERF computation
# ---------------------------------------------------------------------------


def erf_for_model(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    device: torch.device,
    flood_class: int = 1,
) -> np.ndarray:
    """Accumulate |grad| at the image-center output pixel over ``images``.

    Returns ``(H, W)`` float32 array normalized to ``[0, 1]``.
    """
    model.train(False)
    for p in model.parameters():
        p.requires_grad_(False)

    H, W = images[0].shape[-2:]
    cy, cx = H // 2, W // 2
    acc = torch.zeros(H, W, device=device)

    for img in images:
        x = img.unsqueeze(0).to(device).detach().float().clone()
        x.requires_grad_(True)
        logits = model(x)                                 # (1, C, H, W)
        grad_out = torch.zeros_like(logits)
        grad_out[0, flood_class, cy, cx] = 1.0
        grad = torch.autograd.grad(
            outputs=logits, inputs=x, grad_outputs=grad_out,
            retain_graph=False, create_graph=False,
        )[0]
        acc += grad[0].abs().mean(dim=0)

    erf = acc.detach().cpu().numpy()
    m = float(erf.max())
    if m > 0:
        erf = erf / m
    return erf.astype(np.float32)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def radial_profile(erf: np.ndarray, n_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(radii, mean_erf_at_radius)`` — angular average around center."""
    H, W = erf.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.indices((H, W))
    r = np.hypot(yy - cy, xx - cx).astype(np.float32)
    r_max = r.max()
    edges = np.linspace(0, r_max, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    prof = np.zeros(n_bins, dtype=np.float32)
    for b in range(n_bins):
        sel = (r >= edges[b]) & (r < edges[b + 1])
        if sel.any():
            prof[b] = erf[sel].mean()
    return mids, prof


def contour_radius(erf: np.ndarray, level: float) -> float:
    """Approx radius of the level-set contour: sqrt(area(erf >= level) / pi)."""
    mask = erf >= level
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    return float(np.sqrt(area / np.pi))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


LEVELS = (0.5, 0.25, 0.1)
LEVEL_COLORS = ("#f7d046", "#ef7e4a", "#d6355b")


def _imshow_erf(ax, erf: np.ndarray, title: str) -> None:
    ax.imshow(erf, cmap="magma", vmin=0, vmax=1)
    H, W = erf.shape
    yy, xx = np.indices((H, W))
    for lev, col in zip(LEVELS, LEVEL_COLORS):
        ax.contour(xx, yy, erf, levels=[lev], colors=col, linewidths=1.2)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    radii = [contour_radius(erf, lev) for lev in LEVELS]
    txt = "\n".join(f"r@{int(lev * 100)}% = {r:5.1f}px"
                    for lev, r in zip(LEVELS, radii))
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes, fontsize=9,
        va="top", ha="left", color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.55),
    )


def plot_single(erf: np.ndarray, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _imshow_erf(ax, erf, title)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_compare(
    erf_s: np.ndarray, erf_m: np.ndarray, out: Path,
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    _imshow_erf(ax[0], erf_s, "scratch")
    _imshow_erf(ax[1], erf_m, "MLA")
    r_s = contour_radius(erf_s, 0.1)
    r_m = contour_radius(erf_m, 0.1)
    ratio = (r_s / r_m) if r_m > 0 else float("inf")
    fig.suptitle(
        f"Effective receptive field  (r@10% ratio scratch/MLA = {ratio:.2f})",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_radial(
    erf_s: np.ndarray, erf_m: np.ndarray, out: Path,
) -> None:
    r_s, p_s = radial_profile(erf_s)
    r_m, p_m = radial_profile(erf_m)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    eps = 1e-6
    ax.semilogy(r_s, p_s + eps, label="scratch", linewidth=2)
    ax.semilogy(r_m, p_m + eps, label="MLA",     linewidth=2)
    ax.set_xlabel("radius (px from center)")
    ax.set_ylabel("mean |grad| (normalized, log)")
    ax.set_title("ERF radial profile")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-ckpt", type=Path, required=True)
    ap.add_argument("--mla-ckpt",     type=Path, required=True)
    ap.add_argument("--val-csv",      type=Path, required=True)
    ap.add_argument("--data-root",    type=Path, required=True)
    ap.add_argument("--out-dir",      type=Path, default=Path("reports/figures/xai"))
    ap.add_argument("--n-samples",    type=int, default=32)
    ap.add_argument("--seed",         type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading scratch from {args.scratch_ckpt}")
    m_s, s_cfg = _build_from_ckpt(args.scratch_ckpt, device)
    tile_size = (s_cfg or {}).get("data", {}).get("tile_size")

    ds = Sen1FloodsDataset(
        csv_path=args.val_csv, data_root=args.data_root,
        label_key=None, augment=False, tile_size=tile_size,
    )
    n = min(args.n_samples, len(ds))
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(ds)), n)
    print(f"Val tiles: {len(ds)} | sampling {n} (seed={args.seed})")

    images = [ds[i]["image"] for i in idx]
    print("Computing ERF — scratch")
    erf_s = erf_for_model(m_s, images, device)
    del m_s
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Loading MLA     from {args.mla_ckpt}")
    m_m, _ = _build_from_ckpt(args.mla_ckpt, device)
    print("Computing ERF — MLA")
    erf_m = erf_for_model(m_m, images, device)
    del m_m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    plot_single(erf_s, "scratch — ERF",  args.out_dir / "erf_scratch.png")
    plot_single(erf_m, "MLA — ERF",      args.out_dir / "erf_mla.png")
    plot_compare(erf_s, erf_m,           args.out_dir / "erf_compare.png")
    plot_radial(erf_s, erf_m,            args.out_dir / "erf_radial.png")

    r_s = contour_radius(erf_s, 0.1)
    r_m = contour_radius(erf_m, 0.1)
    print(f"\nr@10%  scratch = {r_s:.1f}px   MLA = {r_m:.1f}px   "
          f"ratio = {(r_s / r_m) if r_m > 0 else float('inf'):.2f}")
    print(f"Done. Figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
