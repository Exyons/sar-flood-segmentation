"""Training loop for SegFormer (Standard ViT or MLA ViT).

Usage:
    uv run python train_segformer.py --config configs/default.yaml              # Standard ViT
    uv run python train_segformer.py --config configs/default.yaml --use_mla    # MLA ViT
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

from data.dataset import get_dataloaders
from models.segformer_model import SegFormer


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    """Compute mean IoU."""
    pred = pred.flatten()
    target = target.flatten()
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def compute_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute F1 for flood class (class 1)."""
    pred = (pred.flatten() == 1)
    target = (target.flatten() == 1)
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def get_polynomial_lr_scheduler(optimizer, epochs: int, power: float = 0.9, warmup_epochs: int = 5):
    """Polynomial LR decay with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return (1.0 - progress) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, float]:
    """Train for one epoch. Returns (loss, iou, f1)."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        sar = batch["sar"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(sar)  # (B, 2, H, W)
            loss = criterion(logits, mask)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pred = logits.argmax(dim=1)  # (B, H, W)
        total_loss += loss.item()
        total_iou += compute_iou(pred, mask)
        total_f1 += compute_f1(pred, mask)
        n_batches += 1

    return total_loss / n_batches, total_iou / n_batches, total_f1 / n_batches


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, float]:
    """Run validation pass. Returns (loss, iou, f1)."""
    model.train(False)
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="  Val", leave=False):
        sar = batch["sar"].to(device)
        mask = batch["mask"].to(device)

        with autocast(enabled=use_amp):
            logits = model(sar)
            loss = criterion(logits, mask)

        pred = logits.argmax(dim=1)
        total_loss += loss.item()
        total_iou += compute_iou(pred, mask)
        total_f1 += compute_f1(pred, mask)
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n_batches, total_iou / n_batches, total_f1 / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--use_mla", action="store_true", help="Use MLA attention variant")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seed
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model
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
        use_mla=args.use_mla,
        rank_divisor=seg_cfg["mla_rank_divisor"],
    ).to(device)

    variant = "MLA" if args.use_mla else "Standard"
    print(f"\n{model.param_summary()}")
    print(f"Variant: {variant}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(cfg)
    print(f"Train: {len(train_loader.dataset)} tiles, Val: {len(val_loader.dataset)} tiles")

    # Loss with class weights
    class_weights = torch.tensor(cfg["training"]["class_weights"], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler
    scheduler = get_polynomial_lr_scheduler(
        optimizer, train_cfg["epochs"], train_cfg["poly_power"], train_cfg["warmup_epochs"]
    )

    # AMP
    use_amp = train_cfg["use_amp"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Checkpoint dir
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"]) / f"segformer_{'mla' if args.use_mla else 'standard'}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_iou = 0.0
    epochs = train_cfg["epochs"]

    print(f"\nTraining {variant} SegFormer for {epochs} epochs...")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_iou, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_iou, val_f1 = run_validation(
            model, val_loader, criterion, device, use_amp
        )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f} F1: {val_f1:.4f} | "
            f"LR: {lr:.2e} | {elapsed:.1f}s"
        )

        # Save best
        if val_iou > best_iou:
            best_iou = val_iou
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_f1": val_f1,
                "use_mla": args.use_mla,
                "config": cfg,
            }, ckpt_path)
            print(f"  -> New best IoU: {val_iou:.4f} (saved to {ckpt_path})")

    # Save final
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "use_mla": args.use_mla,
        "config": cfg,
    }, ckpt_dir / "final.pt")

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
