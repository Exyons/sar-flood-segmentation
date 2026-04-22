"""Unified SegFormer training loop — pre-train on Sen1Floods11, fine-tune on India floods.

Usage:
    # Stage 1 — pre-train HF SegFormer on USA+Pakistan+Sri-Lanka (India held out)
    uv run python train_segformer.py --config configs/pretrain_hf.yaml

    # Stage 2 — fine-tune on GEE-exported India events, init from pretrain
    uv run python train_segformer.py --config configs/finetune_hf.yaml

    # Sanity check: 2 batches on CPU
    uv run python train_segformer.py --config configs/pretrain_hf.yaml --smoke

Config shape (YAML):
    model:
      kind: hf | scratch | mla
      num_labels: 2
      pretrained_id: nvidia/mit-b2    # hf only
      init_from: path/to/best.pt      # optional warm-start
      # scratch / mla extras: in_channels, embed_dims, num_heads, ...

    data:
      root:      data/sen1floods11/data
      train_csv: data/sen1floods11/splits/pretrain_train.csv
      val_csv:   data/sen1floods11/splits/pretrain_val.csv
      label_key: LabelHand

    training:
      epochs: 60
      lr: 6.0e-5
      weight_decay: 0.01
      batch_size: 4
      num_workers: 4
      class_weights: [1.0, 5.0]   # optional
      poly_power: 0.9
      warmup_epochs: 5
      use_amp: true
      seed: 42

    ckpt_dir: checkpoints/segformer_hf_pretrain
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

from data.dataset import Sen1FloodsDataset
from torch.utils.data import DataLoader
from models.segformer_model import build as build_model


# ---------------------------------------------------------------------------
# Metrics (ignore_index aware)
# ---------------------------------------------------------------------------


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2,
                ignore_index: int = -1) -> float:
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    if pred.numel() == 0:
        return 0.0
    ious = []
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def compute_f1(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = -1) -> float:
    valid = target != ignore_index
    pred = pred[valid] == 1
    target = target[valid] == 1
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def get_polynomial_lr_scheduler(optimizer, epochs: int, power: float = 0.9,
                                warmup_epochs: int = 5):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return (1.0 - progress) ** power
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp,
                    ignore_index: int = -1, max_steps: int | None = None):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    n = 0

    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(image)
            loss = criterion(logits, label)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pred = logits.argmax(dim=1)
        total_loss += loss.item()
        total_iou += compute_iou(pred, label, ignore_index=ignore_index)
        total_f1 += compute_f1(pred, label, ignore_index=ignore_index)
        n += 1

        if max_steps is not None and step + 1 >= max_steps:
            break

    return total_loss / max(1, n), total_iou / max(1, n), total_f1 / max(1, n)


@torch.no_grad()
def run_validation(model, loader, criterion, device, use_amp,
                   ignore_index: int = -1, max_steps: int | None = None):
    model.train(False)
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    n = 0

    for step, batch in enumerate(tqdm(loader, desc="  Val", leave=False)):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(image)
            loss = criterion(logits, label)

        pred = logits.argmax(dim=1)
        total_loss += loss.item()
        total_iou += compute_iou(pred, label, ignore_index=ignore_index)
        total_f1 += compute_f1(pred, label, ignore_index=ignore_index)
        n += 1

        if max_steps is not None and step + 1 >= max_steps:
            break

    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, total_iou / n, total_f1 / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_init_from(model: nn.Module, path: str | Path) -> None:
    """Load weights from a training checkpoint, tolerating key mismatches."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model.init_from points at missing file: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  init_from: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  init_from: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    print(f"  init_from: loaded {path}")


def build_loaders(cfg: dict, pin_memory: bool = True) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    tr_cfg = cfg["training"]
    root = data_cfg["root"]
    label_key = data_cfg.get("label_key", "LabelHand")

    train_ds = Sen1FloodsDataset(
        csv_path=data_cfg["train_csv"],
        data_root=root,
        label_key=label_key,
        augment=True,
    )
    val_ds = Sen1FloodsDataset(
        csv_path=data_cfg["val_csv"],
        data_root=root,
        label_key=label_key,
        augment=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        num_workers=tr_cfg.get("num_workers", 4),
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        num_workers=tr_cfg.get("num_workers", 4),
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


HISTORY_COLS = [
    "epoch", "lr",
    "train_loss", "train_iou", "train_f1",
    "val_loss", "val_iou", "val_f1",
    "elapsed_s",
]


def _append_history(history_path: Path, row: dict) -> None:
    new_file = not history_path.exists()
    with open(history_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in HISTORY_COLS})


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer (HF or from-scratch)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 2 train + 2 val steps on CPU for shape sanity")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tr_cfg = cfg["training"]
    seed = tr_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu" if args.smoke else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Build model
    model_cfg = dict(cfg["model"])
    init_from = model_cfg.pop("init_from", None)
    model = build_model(**model_cfg).to(device)
    print(model.param_summary() if hasattr(model, "param_summary") else
          f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    if init_from:
        load_init_from(model, init_from)

    # Loaders
    train_loader, val_loader = build_loaders(cfg, pin_memory=device.type == "cuda")
    print(f"Train: {len(train_loader.dataset)} tiles | Val: {len(val_loader.dataset)} tiles")

    # Loss
    class_weights = tr_cfg.get("class_weights")
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Optimizer / scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg.get("weight_decay", 0.01),
    )
    epochs = 1 if args.smoke else tr_cfg["epochs"]
    scheduler = get_polynomial_lr_scheduler(
        optimizer, epochs,
        power=tr_cfg.get("poly_power", 0.9),
        warmup_epochs=tr_cfg.get("warmup_epochs", 5),
    )

    use_amp = tr_cfg.get("use_amp", True) and device.type == "cuda" and not args.smoke
    scaler = GradScaler(device=device.type, enabled=use_amp)

    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = ckpt_dir / "history.csv"

    if args.smoke:
        print("\n[SMOKE] 2 train + 2 val steps on CPU")
        print("-" * 70)
        tl, ti, tf = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scaler, device, use_amp, max_steps=2)
        vl, vi, vf = run_validation(model, val_loader, criterion, device, use_amp,
                                    max_steps=2)
        print(f"train  loss={tl:.4f} iou={ti:.4f} f1={tf:.4f}")
        print(f"val    loss={vl:.4f} iou={vi:.4f} f1={vf:.4f}")
        assert np.isfinite(tl) and np.isfinite(vl), "non-finite loss"
        print("[SMOKE] OK")
        return

    best_iou = 0.0
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_iou, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp,
        )
        val_loss, val_iou, val_f1 = run_validation(
            model, val_loader, criterion, device, use_amp,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train loss {train_loss:.4f} iou {train_iou:.4f} f1 {train_f1:.4f} | "
            f"Val loss {val_loss:.4f} iou {val_iou:.4f} f1 {val_f1:.4f} | "
            f"LR {lr:.2e} | {elapsed:.1f}s"
        )

        _append_history(history_path, {
            "epoch": epoch, "lr": f"{lr:.6e}",
            "train_loss": f"{train_loss:.6f}",
            "train_iou":  f"{train_iou:.6f}",
            "train_f1":   f"{train_f1:.6f}",
            "val_loss":   f"{val_loss:.6f}",
            "val_iou":    f"{val_iou:.6f}",
            "val_f1":     f"{val_f1:.6f}",
            "elapsed_s":  f"{elapsed:.2f}",
        })

        if val_iou > best_iou:
            best_iou = val_iou
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_f1": val_f1,
                "config": cfg,
            }, ckpt_path)
            print(f"  -> New best val IoU: {val_iou:.4f} (saved to {ckpt_path})")

    # Save final
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, ckpt_dir / "final.pt")

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
