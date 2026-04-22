"""Attention-map visualization for scratch vs MLA SegFormer.

Both attention classes compute their softmax tensor internally; to capture
it without touching ``models/segformer/attention.py`` we monkey-patch each
attention module's ``forward`` inside a context manager. The wrapper
reimplements the stock forward body (short + stable) and stashes the attn
matrix on ``module.last_attn`` before returning.

For each val tile and each variant we then:
  * forward under the capture context
  * at every stage, take the last block's attn (B, heads, N_q, N_kv)
  * average over heads; sum over queries predicted as flood at that stage's
    spatial resolution -> per-key mass
  * reshape (N_kv,) back to the sr-reduced grid, upsample to input size,
    min-max normalize
  * overlay on the SAR VV channel

Outputs (reports/figures/xai/ by default):
    attn_tile_<name>.png          per-tile grid: SAR | s1 | s2 | s3 | s4 | pred
    attn_mean_stages.png          mean map per stage, scratch vs MLA
"""

from __future__ import annotations

import argparse
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import Sen1FloodsDataset  # noqa: E402
from evaluate import _build_from_ckpt  # noqa: E402
from models.segformer.attention import (  # noqa: E402
    EfficientSelfAttention, MLASelfAttention,
)


DPI = 150


# ---------------------------------------------------------------------------
# Capture context — monkey-patch attention.forward to stash softmax tensor.
# ---------------------------------------------------------------------------


def _make_eff_forward(m: EfficientSelfAttention):
    def forward(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, D = x.shape
        q = m.q(x).reshape(B, N, m.num_heads, m.head_dim).permute(0, 2, 1, 3)
        if m.sr_ratio > 1:
            x_sr = x.permute(0, 2, 1).reshape(B, D, H, W)
            x_sr = m.sr(x_sr).flatten(2).transpose(1, 2)
            x_sr = m.sr_norm(x_sr)
            H_sr, W_sr = H // m.sr_ratio, W // m.sr_ratio
        else:
            x_sr = x
            H_sr, W_sr = H, W
        k = m.k(x_sr).reshape(B, -1, m.num_heads, m.head_dim).permute(0, 2, 1, 3)
        v = m.v(x_sr).reshape(B, -1, m.num_heads, m.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * m.scale
        attn = F.softmax(attn, dim=-1)
        m.last_attn = attn.detach()
        m.last_hw = (H, W)
        m.last_sr_hw = (H_sr, W_sr)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return m.out(out)
    return forward


def _make_mla_forward(m: MLASelfAttention):
    def forward(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, D = x.shape
        q = m.q(x).reshape(B, N, m.num_heads, m.head_dim).permute(0, 2, 1, 3)
        if m.sr_ratio > 1:
            x_sr = x.permute(0, 2, 1).reshape(B, D, H, W)
            x_sr = m.sr(x_sr).flatten(2).transpose(1, 2)
            x_sr = m.sr_norm(x_sr)
            H_sr, W_sr = H // m.sr_ratio, W // m.sr_ratio
        else:
            x_sr = x
            H_sr, W_sr = H, W
        compressed = m.kv_down(x_sr)
        k = m.k_up(compressed).reshape(B, -1, m.num_heads, m.head_dim).permute(0, 2, 1, 3)
        v = m.v_up(compressed).reshape(B, -1, m.num_heads, m.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * m.scale
        attn = F.softmax(attn, dim=-1)
        m.last_attn = attn.detach()
        m.last_hw = (H, W)
        m.last_sr_hw = (H_sr, W_sr)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return m.out(out)
    return forward


@contextmanager
def capture_attention(model: torch.nn.Module):
    originals: list[tuple[torch.nn.Module, object]] = []
    for mod in model.modules():
        if isinstance(mod, EfficientSelfAttention):
            originals.append((mod, mod.forward))
            mod.forward = _make_eff_forward(mod)
        elif isinstance(mod, MLASelfAttention):
            originals.append((mod, mod.forward))
            mod.forward = _make_mla_forward(mod)
    try:
        yield
    finally:
        for mod, fn in originals:
            mod.forward = fn


# ---------------------------------------------------------------------------
# Stage attention map computation
# ---------------------------------------------------------------------------


def per_stage_attention(
    model: torch.nn.Module,
    pred_full: torch.Tensor,  # (H, W) long
    target_hw: tuple[int, int],
) -> list[np.ndarray]:
    """Return a list of 4 upsampled attention maps (H, W), one per stage.

    Uses the **last block** of each stage. Aggregates attention by summing
    over queries whose predicted class is flood at that stage's spatial
    resolution, then reshapes key axis back to (H_sr, W_sr) and resamples
    to ``target_hw``.
    """
    encoder = model.encoder
    H_out, W_out = target_hw
    maps: list[np.ndarray] = []
    for stage_idx in range(encoder.num_stages):
        last_block = encoder.blocks[stage_idx][-1]
        attn = last_block.attn.last_attn          # (1, heads, N_q, N_kv)
        H, W = last_block.attn.last_hw
        H_sr, W_sr = last_block.attn.last_sr_hw
        attn = attn[0].mean(0)                    # (N_q, N_kv)

        pred_stage = F.interpolate(
            pred_full.float().view(1, 1, *pred_full.shape),
            size=(H, W), mode="nearest",
        ).view(H * W).long()
        flood_mask = pred_stage == 1
        if flood_mask.any():
            mass = attn[flood_mask].mean(0)       # (N_kv,)
        else:
            mass = attn.mean(0)                   # fallback: mean over all queries
        heat = mass.view(H_sr, W_sr).cpu().numpy()

        heat_t = torch.from_numpy(heat).float().view(1, 1, H_sr, W_sr)
        heat_up = F.interpolate(heat_t, size=(H_out, W_out),
                                mode="bilinear", align_corners=False)
        heat_up = heat_up[0, 0].numpy()
        lo, hi = heat_up.min(), heat_up.max()
        if hi > lo:
            heat_up = (heat_up - lo) / (hi - lo)
        maps.append(heat_up)
    return maps


# ---------------------------------------------------------------------------
# Dataset sampling
# ---------------------------------------------------------------------------


def _flood_frac(label: torch.Tensor) -> float:
    valid = label != -1
    if not valid.any():
        return 0.0
    return float((label[valid] == 1).float().mean().item())


def _sample_tiles(ds, n: int, min_frac: float, seed: int) -> list[int]:
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    picked: list[int] = []
    for i in order:
        if _flood_frac(ds[i]["label"]) >= min_frac:
            picked.append(i)
            if len(picked) >= n:
                break
    if len(picked) < n:
        for i in order:
            if i not in picked:
                picked.append(i)
                if len(picked) >= n:
                    break
    return picked[:n]


def _sar_display(vv_01: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(vv_01, (2, 98))
    if hi <= lo:
        return np.zeros_like(vv_01)
    return np.clip((vv_01 - lo) / (hi - lo), 0, 1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_tile_grid(
    name: str, sar_vv: np.ndarray,
    pred_s: np.ndarray, pred_m: np.ndarray,
    maps_s: list[np.ndarray], maps_m: list[np.ndarray],
    out: Path,
) -> None:
    fig, ax = plt.subplots(2, 6, figsize=(18, 6))
    cmap_attn = plt.get_cmap("magma")
    cmap_lab = plt.get_cmap("Blues")
    for row, (label, pred, maps) in enumerate([
        ("scratch", pred_s, maps_s),
        ("MLA",     pred_m, maps_m),
    ]):
        ax[row, 0].imshow(sar_vv, cmap="gray")
        ax[row, 0].set_ylabel(label, fontsize=11)
        for s in range(4):
            ax[row, 0].set_title("SAR (VV)" if row == 0 else "")
            ax[row, s + 1].imshow(sar_vv, cmap="gray", alpha=0.6)
            ax[row, s + 1].imshow(maps[s], cmap=cmap_attn, alpha=0.55,
                                  vmin=0, vmax=1)
            if row == 0:
                ax[row, s + 1].set_title(f"stage {s + 1} attn")
        ax[row, 5].imshow(pred, cmap=cmap_lab, vmin=0, vmax=1)
        if row == 0:
            ax[row, 5].set_title("prediction")
        for c in range(6):
            ax[row, c].set_xticks([]); ax[row, c].set_yticks([])
    fig.suptitle(f"Attention — {name}", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_mean_stages(
    mean_s: list[np.ndarray], mean_m: list[np.ndarray], out: Path,
) -> None:
    fig, ax = plt.subplots(2, 4, figsize=(14, 6.5))
    for row, (label, maps) in enumerate([("scratch", mean_s), ("MLA", mean_m)]):
        for s in range(4):
            ax[row, s].imshow(maps[s], cmap="magma", vmin=0, vmax=1)
            ax[row, s].set_xticks([]); ax[row, s].set_yticks([])
            if row == 0:
                ax[row, s].set_title(f"stage {s + 1}")
        ax[row, 0].set_ylabel(label, fontsize=11)
    fig.suptitle("Mean attention per stage (averaged over sampled tiles)")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_variant(
    model: torch.nn.Module, image: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """Forward + per-stage maps. Returns (pred_full, maps)."""
    model.train(False)
    x = image.unsqueeze(0).to(device)
    with capture_attention(model):
        logits = model(x)
    pred = logits.argmax(dim=1)[0]
    maps = per_stage_attention(model, pred, target_hw=x.shape[-2:])
    return pred.cpu(), maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-ckpt", type=Path, required=True)
    ap.add_argument("--mla-ckpt",     type=Path, required=True)
    ap.add_argument("--val-csv",      type=Path, required=True)
    ap.add_argument("--data-root",    type=Path, required=True)
    ap.add_argument("--out-dir",      type=Path, default=Path("reports/figures/xai"))
    ap.add_argument("--n-tiles",      type=int, default=6)
    ap.add_argument("--flood-frac-min", type=float, default=0.05)
    ap.add_argument("--seed",         type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = Sen1FloodsDataset(
        csv_path=args.val_csv, data_root=args.data_root,
        label_key=None, augment=False,
    )
    print(f"Val tiles: {len(ds)} | sampling {args.n_tiles} "
          f"with flood-frac >= {args.flood_frac_min}")
    idx = _sample_tiles(ds, args.n_tiles, args.flood_frac_min, args.seed)

    print(f"Loading scratch  from {args.scratch_ckpt}")
    m_s, _ = _build_from_ckpt(args.scratch_ckpt, device)
    print(f"Loading MLA      from {args.mla_ckpt}")
    m_m, _ = _build_from_ckpt(args.mla_ckpt, device)

    sum_s = [np.zeros((1,)) for _ in range(4)]
    sum_m = [np.zeros((1,)) for _ in range(4)]
    n_acc = 0

    for i in idx:
        sample = ds[i]
        image = sample["image"]
        name = sample["tile_name"]
        print(f"  tile {name}")
        pred_s, maps_s = _run_variant(m_s, image, device)
        pred_m, maps_m = _run_variant(m_m, image, device)

        sar_vv = _sar_display(image[0].numpy())
        plot_tile_grid(
            name, sar_vv,
            pred_s.numpy(), pred_m.numpy(),
            maps_s, maps_m,
            args.out_dir / f"attn_tile_{name}.png",
        )

        if n_acc == 0:
            sum_s = [m.copy() for m in maps_s]
            sum_m = [m.copy() for m in maps_m]
        else:
            for s in range(4):
                sum_s[s] += maps_s[s]
                sum_m[s] += maps_m[s]
        n_acc += 1

    if n_acc > 0:
        mean_s = [m / n_acc for m in sum_s]
        mean_m = [m / n_acc for m in sum_m]
        plot_mean_stages(mean_s, mean_m, args.out_dir / "attn_mean_stages.png")

    print(f"\nDone. Figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
