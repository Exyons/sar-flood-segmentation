"""Model comparison and metrics computation.

Two modes:

1. Comparison of all 6 classical / ViT / fusion variants on the legacy
   FloodDataset (RF / XGBoost + from-scratch SegFormer):

       uv run python evaluate.py --config configs/default.yaml --compare

2. Checkpoint evaluation on a CSV split with per-region (or per-event)
   breakdown — used for HF pre-train / India fine-tune:

       uv run python evaluate.py \
         --ckpt checkpoints/segformer_hf_india_ft/best.pt \
         --split data/india_floods/splits/val.csv \
         --data-root data/india_floods

Group key is extracted from each tile's filename (first ``_``-separated
token of ``Path(s1_rel).stem``) — ``India`` / ``USA`` for Sen1Floods11,
``assam2022`` / ``kerala2018`` / … for India GEE exports.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from data.dataset import Sen1FloodsDataset, get_dataloaders
from models.segformer_model import SegFormer, build as build_model
from models.rf_model import RFFloodModel
from models.xgb_model import XGBFloodModel
from fusion.fuse import weighted_average_fusion


class Metrics:
    """Accumulate and compute segmentation metrics."""

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred: np.ndarray, target: np.ndarray):
        """Update with batch predictions. Both (N,) binary arrays."""
        pred = pred.flatten().astype(bool)
        target = target.flatten().astype(bool)
        self.tp += (pred & target).sum()
        self.fp += (pred & ~target).sum()
        self.fn += (~pred & target).sum()
        self.tn += (~pred & ~target).sum()

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp) / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def summary(self) -> dict:
        return {
            "IoU": self.iou,
            "F1": self.f1,
            "Precision": self.precision,
            "Recall": self.recall,
            "Accuracy": self.accuracy,
        }


def load_segformer(cfg: dict, use_mla: bool, device: torch.device) -> SegFormer:
    """Load trained SegFormer from checkpoint."""
    seg_cfg = cfg["segformer"]
    model = SegFormer(
        in_channels=seg_cfg["in_channels"],
        num_classes=seg_cfg["num_classes"],
        embed_dims=seg_cfg["embed_dims"],
        num_heads=seg_cfg["num_heads"],
        sr_ratios=seg_cfg["sr_ratios"],
        num_blocks=seg_cfg["num_blocks"],
        mlp_ratios=seg_cfg["mlp_ratios"],
        patch_sizes=seg_cfg["patch_sizes"],
        strides=seg_cfg["strides"],
        decoder_dim=seg_cfg["decoder_dim"],
        use_mla=use_mla,
        rank_divisor=seg_cfg["mla_rank_divisor"],
    ).to(device)

    variant = "mla" if use_mla else "standard"
    ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / f"segformer_{variant}" / "best.pt"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {variant} SegFormer from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"WARNING: No checkpoint at {ckpt_path}, using random weights")

    model.train(False)
    return model


@torch.no_grad()
def run_segformer_metrics(model: SegFormer, test_loader, device: torch.device) -> Metrics:
    """Run SegFormer on test set, return metrics."""
    metrics = Metrics()
    for batch in tqdm(test_loader, desc="  SegFormer"):
        sar = batch["sar"].to(device)
        mask = batch["mask"].numpy()
        logits = model(sar)
        pred = logits.argmax(dim=1).cpu().numpy()
        metrics.update(pred, mask)
    return metrics


@torch.no_grad()
def get_segformer_probs(model: SegFormer, test_loader, device: torch.device) -> list[tuple]:
    """Get per-tile flood probabilities from SegFormer."""
    results = []
    for batch in test_loader:
        sar = batch["sar"].to(device)
        mask = batch["mask"].numpy()
        geo = batch["geo"].numpy()
        logits = model(sar)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # flood prob
        for i in range(probs.shape[0]):
            results.append((probs[i], mask[i], geo[i]))
    return results


def run_ml_model_metrics(model, test_data: list[tuple], model_name: str) -> Metrics:
    """Run RF or XGBoost on test tiles."""
    metrics = Metrics()
    for _, mask, geo in tqdm(test_data, desc=f"  {model_name}"):
        pred_probs = model.predict_tile(geo)
        pred = (pred_probs >= 0.5).astype(np.int64)
        metrics.update(pred, mask)
    return metrics


def run_fusion_metrics(
    vit_data: list[tuple],
    ml_model,
    vit_weight: float,
    ml_weight: float,
    name: str,
) -> Metrics:
    """Run fused predictions."""
    metrics = Metrics()
    for vit_probs, mask, geo in tqdm(vit_data, desc=f"  {name}"):
        ml_probs = ml_model.predict_tile(geo)
        _, fused_pred = weighted_average_fusion(vit_probs, ml_probs, vit_weight, ml_weight)
        metrics.update(fused_pred, mask)
    return metrics


def print_comparison_table(results: dict[str, dict]):
    """Print formatted comparison table."""
    print("\n" + "=" * 78)
    header = f"{'Model':<25} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Params':>12}"
    print(header)
    print("-" * 78)
    for name, data in results.items():
        m = data["metrics"]
        params = data.get("params", "—")
        print(
            f"{name:<25} {m['IoU']:>8.4f} {m['F1']:>8.4f} "
            f"{m['Precision']:>8.4f} {m['Recall']:>8.4f} {params:>12}"
        )
    print("=" * 78)


def _group_key(tile_name: str) -> str:
    """Extract the grouping key (region / event) from a tile filename stem."""
    return tile_name.split("_", 1)[0] if "_" in tile_name else tile_name


def _build_from_ckpt(ckpt_path: Path, device: torch.device):
    """Rebuild a model from a training checkpoint and load its weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None or "model" not in cfg:
        raise ValueError(
            f"Checkpoint {ckpt_path} has no ``config.model`` block — "
            "this evaluator expects the new-style YAML configs."
        )
    model_cfg = dict(cfg["model"])
    model_cfg.pop("init_from", None)
    model = build_model(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.train(False)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint {ckpt_path} (epoch {epoch})")
    return model, cfg


@torch.no_grad()
def eval_ckpt_on_split(
    ckpt_path: str | Path,
    split_csv: str | Path,
    data_root: str | Path,
    label_key: str | None = "LabelHand",
    batch_size: int = 4,
    num_workers: int = 4,
    device: torch.device | None = None,
) -> dict:
    """Run a checkpoint over a CSV split. Return overall + per-group metrics."""
    from torch.utils.data import DataLoader

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _build_from_ckpt(Path(ckpt_path), device)

    ds = Sen1FloodsDataset(
        csv_path=split_csv,
        data_root=data_root,
        label_key=label_key,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    overall = Metrics()
    per_group: dict[str, Metrics] = defaultdict(Metrics)

    for batch in tqdm(loader, desc=f"  eval {Path(ckpt_path).name}"):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].numpy()  # (B, H, W)
        logits = model(image)
        pred = logits.argmax(dim=1).cpu().numpy()

        valid = label != -1
        names = batch["tile_name"]

        for i in range(pred.shape[0]):
            v = valid[i]
            p = pred[i][v]
            t = label[i][v]
            overall.update(p, t)
            per_group[_group_key(names[i])].update(p, t)

    return {
        "overall": overall.summary(),
        "per_group": {k: m.summary() for k, m in per_group.items()},
        "n_tiles": len(ds),
    }


def print_grouped_results(result: dict, title: str = "Evaluation") -> None:
    overall = result["overall"]
    groups = result["per_group"]
    print(f"\n{title}  ({result['n_tiles']} tiles)")
    print("=" * 78)
    print(f"{'Group':<20} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Acc':>8}")
    print("-" * 78)
    for k in sorted(groups):
        m = groups[k]
        print(f"{k:<20} {m['IoU']:>8.4f} {m['F1']:>8.4f} "
              f"{m['Precision']:>8.4f} {m['Recall']:>8.4f} {m['Accuracy']:>8.4f}")
    print("-" * 78)
    print(f"{'OVERALL':<20} {overall['IoU']:>8.4f} {overall['F1']:>8.4f} "
          f"{overall['Precision']:>8.4f} {overall['Recall']:>8.4f} "
          f"{overall['Accuracy']:>8.4f}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Flood model evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--use_mla", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare all 6 legacy models")

    # Checkpoint + CSV split mode (per-region / per-event breakdown)
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint to evaluate (HF pre-train / India fine-tune).")
    parser.add_argument("--split", default=None,
                        help="CSV split to evaluate on.")
    parser.add_argument("--data-root", default=None,
                        help="Root the CSV paths are relative to. "
                             "Defaults inferred from the split path.")
    parser.add_argument("--label-key", default="LabelHand",
                        help="Label-key override; use 'none' for CSV-driven label paths.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if args.ckpt and args.split:
        split_path = Path(args.split)
        if args.data_root:
            data_root = args.data_root
        elif "india_floods" in split_path.parts:
            data_root = "data/india_floods"
        else:
            data_root = "data/sen1floods11/data"
        label_key = None if args.label_key.lower() == "none" else args.label_key
        result = eval_ckpt_on_split(
            ckpt_path=args.ckpt,
            split_csv=args.split,
            data_root=data_root,
            label_key=label_key,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print_grouped_results(result, title=f"{Path(args.ckpt).stem} on {split_path.name}")
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(cfg)
    print(f"Test set: {len(test_loader.dataset)} tiles")

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    fusion_cfg = cfg["fusion"]

    if args.compare:
        results = {}

        # Standard ViT
        print("\n--- Standard ViT ---")
        std_model = load_segformer(cfg, use_mla=False, device=device)
        std_data = get_segformer_probs(std_model, test_loader, device)
        std_metrics = Metrics()
        for probs, mask, _ in std_data:
            std_metrics.update((probs >= 0.5).astype(int), mask)
        params_std = f"{std_model.count_parameters() / 1e6:.2f}M"
        results["Standard ViT"] = {"metrics": std_metrics.summary(), "params": params_std}

        # MLA ViT
        print("\n--- MLA ViT ---")
        mla_model = load_segformer(cfg, use_mla=True, device=device)
        mla_data = get_segformer_probs(mla_model, test_loader, device)
        mla_metrics = Metrics()
        for probs, mask, _ in mla_data:
            mla_metrics.update((probs >= 0.5).astype(int), mask)
        params_mla = f"{mla_model.count_parameters() / 1e6:.2f}M"
        results["MLA ViT"] = {"metrics": mla_metrics.summary(), "params": params_mla}

        # RF
        print("\n--- Random Forest ---")
        rf = RFFloodModel()
        rf_path = ckpt_dir / "rf_model.joblib"
        if rf_path.exists():
            rf.load(str(rf_path))
            rf_metrics = run_ml_model_metrics(rf, std_data, "RF")
            results["RF (geo only)"] = {"metrics": rf_metrics.summary()}
        else:
            print(f"  No RF model at {rf_path}")

        # XGBoost
        print("\n--- XGBoost ---")
        xgb = XGBFloodModel()
        xgb_path = ckpt_dir / "xgb_model.json"
        if xgb_path.exists():
            xgb.load(str(xgb_path))
            xgb_metrics = run_ml_model_metrics(xgb, std_data, "XGBoost")
            results["XGBoost (geo only)"] = {"metrics": xgb_metrics.summary()}
        else:
            print(f"  No XGBoost model at {xgb_path}")

        # Fusion: Standard ViT + RF
        if rf_path.exists():
            print("\n--- Standard ViT + RF Fusion ---")
            fuse_std_metrics = run_fusion_metrics(
                std_data, rf, fusion_cfg["vit_weight"], fusion_cfg["ml_weight"],
                "Std+RF Fusion",
            )
            results["Standard ViT + RF"] = {"metrics": fuse_std_metrics.summary()}

        # Fusion: MLA ViT + RF
        if rf_path.exists():
            print("\n--- MLA ViT + RF Fusion ---")
            fuse_mla_metrics = run_fusion_metrics(
                mla_data, rf, fusion_cfg["vit_weight"], fusion_cfg["ml_weight"],
                "MLA+RF Fusion",
            )
            results["MLA ViT + RF"] = {"metrics": fuse_mla_metrics.summary()}

        print_comparison_table(results)

    else:
        # Single model
        model = load_segformer(cfg, use_mla=args.use_mla, device=device)
        metrics = run_segformer_metrics(model, test_loader, device)
        variant = "MLA" if args.use_mla else "Standard"
        print(f"\n{variant} SegFormer Results:")
        for k, v in metrics.summary().items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
