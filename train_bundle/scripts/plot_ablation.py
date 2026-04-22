"""Ablation plots — scratch vs MLA SegFormer on Sen1Floods11 / India GEE.

Reads per-run ``history.csv`` + ``best.pt`` and writes headline figures:

    curves_scratch.png     loss / iou / f1 (train + val) for the scratch run
    curves_mla.png         same for the MLA run
    curves_compare.png     val-only, both runs on shared axes (headline figure)
    bar_metrics.png        IoU / F1 / Prec / Recall, scratch vs MLA
    bar_per_region.png     per-region / per-event IoU (from evaluate.eval_ckpt_on_split)
    pred_grid.png          [SAR VV, GT, scratch pred, MLA pred] for N random val tiles
    param_count.txt        param counts + median epoch time

Usage:
    uv run python scripts/plot_ablation.py \
        --scratch-dir checkpoints/segformer_scratch_sen1floods \
        --mla-dir     checkpoints/segformer_mla_sen1floods \
        --val-csv     data/sen1floods11-splits/val.csv \
        --data-root   data \
        --out-dir     reports/figures
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Ensure repo root on path so `evaluate` + `data` + `models` import cleanly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import Sen1FloodsDataset  # noqa: E402
from evaluate import eval_ckpt_on_split, _build_from_ckpt  # noqa: E402


DPI = 150
CURVE_METRICS = [
    ("loss", "train_loss", "val_loss", "Loss"),
    ("iou",  "train_iou",  "val_iou",  "IoU"),
    ("f1",   "train_f1",   "val_f1",   "F1"),
]


def _load_history(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "history.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing {p} — train must complete first")
    return pd.read_csv(p)


def plot_single_run(hist: pd.DataFrame, title: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (_, tcol, vcol, label) in zip(axes, CURVE_METRICS):
        ax.plot(hist["epoch"], hist[tcol], label=f"train {label}", color="#1f77b4")
        ax.plot(hist["epoch"], hist[vcol], label=f"val {label}",   color="#d62728")
        ax.set_xlabel("epoch"); ax.set_ylabel(label); ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_compare(scratch: pd.DataFrame, mla: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (_, _, vcol, label) in zip(axes, CURVE_METRICS):
        ax.plot(scratch["epoch"], scratch[vcol], label="scratch", color="#1f77b4")
        ax.plot(mla["epoch"],     mla[vcol],     label="MLA",     color="#2ca02c")
        if label in ("IoU", "F1"):
            best_s = scratch[vcol].idxmax()
            best_m = mla[vcol].idxmax()
            ax.axvline(scratch.loc[best_s, "epoch"], color="#1f77b4", ls=":", alpha=0.5)
            ax.axvline(mla.loc[best_m, "epoch"],     color="#2ca02c", ls=":", alpha=0.5)
        ax.set_xlabel("epoch"); ax.set_ylabel(f"val {label}"); ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Scratch vs MLA — validation curves")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_bar_metrics(scratch_res: dict, mla_res: dict, out: Path) -> None:
    keys = ["IoU", "F1", "Precision", "Recall"]
    x = np.arange(len(keys)); w = 0.38
    s = [scratch_res["overall"][k] for k in keys]
    m = [mla_res["overall"][k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, s, w, label="scratch", color="#1f77b4")
    ax.bar(x + w/2, m, w, label="MLA",     color="#2ca02c")
    for xi, v in zip(x - w/2, s):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + w/2, m):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_ylim(0, max(max(s), max(m)) * 1.15 + 0.01)
    ax.set_ylabel("score"); ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Scratch vs MLA — validation metrics")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_bar_per_region(scratch_res: dict, mla_res: dict, out: Path) -> None:
    groups = sorted(set(scratch_res["per_group"]) | set(mla_res["per_group"]))
    if not groups:
        return
    x = np.arange(len(groups)); w = 0.38
    s = [scratch_res["per_group"].get(g, {}).get("IoU", 0.0) for g in groups]
    m = [mla_res["per_group"].get(g, {}).get("IoU", 0.0) for g in groups]
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 4.5))
    ax.bar(x - w/2, s, w, label="scratch", color="#1f77b4")
    ax.bar(x + w/2, m, w, label="MLA",     color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("val IoU"); ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Per-region / per-event IoU")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _sar_display(vv_01: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(vv_01, (2, 98))
    if hi <= lo:
        return np.zeros_like(vv_01)
    return np.clip((vv_01 - lo) / (hi - lo), 0, 1)


@torch.no_grad()
def _predict(model, image_chw: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(image_chw).unsqueeze(0).to(device)
    logits = model(x)
    return logits.argmax(dim=1)[0].cpu().numpy()


def plot_pred_grid(
    scratch_ckpt: Path, mla_ckpt: Path,
    val_csv: Path, data_root: Path, out: Path,
    n_tiles: int = 6, seed: int = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_model, s_cfg = _build_from_ckpt(scratch_ckpt, device)
    tile_size = (s_cfg or {}).get("data", {}).get("tile_size")
    ds = Sen1FloodsDataset(csv_path=val_csv, data_root=data_root,
                           label_key=None, augment=False, tile_size=tile_size)
    if len(ds) == 0:
        print("  pred_grid: empty val set, skipping")
        return
    n = min(n_tiles, len(ds))
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), n)

    m_model, _ = _build_from_ckpt(mla_ckpt, device)

    fig, axes = plt.subplots(n, 4, figsize=(4 * 3, n * 3))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["SAR (VV)", "Ground truth", "scratch pred", "MLA pred"]
    cmap_lab = plt.get_cmap("Blues")

    for row, i in enumerate(idx):
        sample = ds[i]
        img = sample["image"].numpy()
        gt = sample["label"].numpy()
        name = sample["tile_name"]

        s_pred = _predict(s_model, img, device)
        m_pred = _predict(m_model, img, device)

        axes[row, 0].imshow(_sar_display(img[0]), cmap="gray")
        gt_disp = np.where(gt == -1, np.nan, gt.astype(float))
        axes[row, 1].imshow(gt_disp, cmap=cmap_lab, vmin=0, vmax=1)
        axes[row, 2].imshow(s_pred, cmap=cmap_lab, vmin=0, vmax=1)
        axes[row, 3].imshow(m_pred, cmap=cmap_lab, vmin=0, vmax=1)
        axes[row, 0].set_ylabel(name, fontsize=8)
        for c in range(4):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
            if row == 0:
                axes[row, c].set_title(col_titles[c])

    fig.suptitle(f"Predictions on {n} random val tiles", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def write_param_count(scratch_ckpt: Path, mla_ckpt: Path,
                      scratch_hist: pd.DataFrame, mla_hist: pd.DataFrame,
                      out: Path) -> None:
    device = torch.device("cpu")
    s_model, _ = _build_from_ckpt(scratch_ckpt, device)
    m_model, _ = _build_from_ckpt(mla_ckpt, device)
    s_params = sum(p.numel() for p in s_model.parameters())
    m_params = sum(p.numel() for p in m_model.parameters())
    s_time = float(scratch_hist["elapsed_s"].median()) if len(scratch_hist) else 0.0
    m_time = float(mla_hist["elapsed_s"].median()) if len(mla_hist) else 0.0
    with open(out, "w") as f:
        f.write("variant   params(M)   median_epoch_s\n")
        f.write(f"scratch   {s_params/1e6:>8.3f}   {s_time:>12.2f}\n")
        f.write(f"mla       {m_params/1e6:>8.3f}   {m_time:>12.2f}\n")
        if s_params:
            f.write(f"\nmla / scratch params: {m_params / s_params:.3f}\n")
        if s_time:
            f.write(f"mla / scratch epoch time: {m_time / s_time:.3f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-dir", type=Path, required=True)
    ap.add_argument("--mla-dir",     type=Path, required=True)
    ap.add_argument("--val-csv",     type=Path, required=True)
    ap.add_argument("--data-root",   type=Path, required=True)
    ap.add_argument("--out-dir",     type=Path, default=Path("reports/figures"))
    ap.add_argument("--n-tiles",     type=int,  default=6)
    ap.add_argument("--batch-size",  type=int,  default=4)
    ap.add_argument("--num-workers", type=int,  default=2)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading histories...")
    scratch_hist = _load_history(args.scratch_dir)
    mla_hist     = _load_history(args.mla_dir)

    print("Plotting per-run curves...")
    plot_single_run(scratch_hist, "scratch SegFormer", args.out_dir / "curves_scratch.png")
    plot_single_run(mla_hist,     "MLA SegFormer",     args.out_dir / "curves_mla.png")

    print("Plotting compare curves...")
    plot_compare(scratch_hist, mla_hist, args.out_dir / "curves_compare.png")

    scratch_ckpt = args.scratch_dir / "best.pt"
    mla_ckpt     = args.mla_dir / "best.pt"
    if not scratch_ckpt.exists() or not mla_ckpt.exists():
        raise SystemExit(f"missing best.pt — {scratch_ckpt} / {mla_ckpt}")

    print("Evaluating scratch on val split...")
    scratch_res = eval_ckpt_on_split(
        ckpt_path=scratch_ckpt, split_csv=args.val_csv,
        data_root=args.data_root, label_key=None,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print("Evaluating MLA on val split...")
    mla_res = eval_ckpt_on_split(
        ckpt_path=mla_ckpt, split_csv=args.val_csv,
        data_root=args.data_root, label_key=None,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    print("Plotting metric bars...")
    plot_bar_metrics(scratch_res, mla_res, args.out_dir / "bar_metrics.png")
    plot_bar_per_region(scratch_res, mla_res, args.out_dir / "bar_per_region.png")

    print("Plotting prediction grid...")
    plot_pred_grid(
        scratch_ckpt, mla_ckpt, args.val_csv, args.data_root,
        args.out_dir / "pred_grid.png", n_tiles=args.n_tiles,
    )

    print("Writing param_count.txt...")
    write_param_count(scratch_ckpt, mla_ckpt, scratch_hist, mla_hist,
                      args.out_dir / "param_count.txt")

    print(f"\nDone. Figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
