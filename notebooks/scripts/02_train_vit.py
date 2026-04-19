# %% [markdown]
# # 02 — SegFormer Training: Standard ViT vs MLA ViT
#
# Train both attention variants side by side.
# Compare loss curves, IoU, F1, and parameter counts.

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
import time

from data.dataset import get_dataloaders
from models.segformer_model import SegFormer
from train_segformer import (
    train_one_epoch, run_validation,
    get_polynomial_lr_scheduler, compute_iou, compute_f1,
)
from torch.cuda.amp import GradScaler

with open("../configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %% [markdown]
# ## Model Parameter Comparison

# %%
seg_cfg = cfg["segformer"]

model_std = SegFormer(
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
    use_mla=False,
)

model_mla = SegFormer(
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
    use_mla=True,
    rank_divisor=seg_cfg["mla_rank_divisor"],
)

print(model_std.param_summary())
print(model_mla.param_summary())

param_reduction = (1 - model_mla.count_parameters() / model_std.count_parameters()) * 100
print(f"MLA parameter reduction: {param_reduction:.1f}%")

# %% [markdown]
# ## Shape Verification

# %%
x = torch.randn(2, 2, 512, 512)
with torch.no_grad():
    out_std = model_std(x)
    out_mla = model_mla(x)
print(f"Input:  {x.shape}")
print(f"Standard output: {out_std.shape}")
print(f"MLA output:      {out_mla.shape}")
assert out_std.shape == (2, 2, 512, 512), "Standard ViT output shape mismatch"
assert out_mla.shape == (2, 2, 512, 512), "MLA ViT output shape mismatch"
print("Shape verification passed!")

# %% [markdown]
# ## Training Loop (Both Variants)

# %%
train_loader, val_loader, _ = get_dataloaders(cfg)
print(f"Train: {len(train_loader.dataset)} tiles")
print(f"Val:   {len(val_loader.dataset)} tiles")

# %%
EPOCHS = cfg["training"]["epochs"]
train_cfg = cfg["training"]
class_weights = torch.tensor(train_cfg["class_weights"], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

history = {"standard": {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []},
           "mla":      {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}}

for variant_name, use_mla in [("standard", False), ("mla", True)]:
    print(f"\n{'='*50}")
    print(f"Training {variant_name.upper()} SegFormer")
    print(f"{'='*50}")

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = get_polynomial_lr_scheduler(
        optimizer, EPOCHS, train_cfg["poly_power"], train_cfg["warmup_epochs"]
    )
    use_amp = train_cfg["use_amp"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_iou = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, ti, tf = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        vl, vi, vf = run_validation(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        history[variant_name]["train_loss"].append(tl)
        history[variant_name]["val_loss"].append(vl)
        history[variant_name]["val_iou"].append(vi)
        history[variant_name]["val_f1"].append(vf)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{EPOCHS} | TLoss:{tl:.4f} | VLoss:{vl:.4f} | "
              f"VIoU:{vi:.4f} | VF1:{vf:.4f} | {elapsed:.1f}s")

        if vi > best_iou:
            best_iou = vi

    print(f"Best val IoU: {best_iou:.4f}")

# %% [markdown]
# ## Training Curves Comparison

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
for name in ["standard", "mla"]:
    axes[0].plot(history[name]["train_loss"], label=f"{name} train", linestyle="--")
    axes[0].plot(history[name]["val_loss"], label=f"{name} val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

# IoU
for name in ["standard", "mla"]:
    axes[1].plot(history[name]["val_iou"], label=f"{name}")
axes[1].set_title("Validation IoU")
axes[1].set_xlabel("Epoch")
axes[1].legend()

# F1
for name in ["standard", "mla"]:
    axes[2].plot(history[name]["val_f1"], label=f"{name}")
axes[2].set_title("Validation F1 (flood class)")
axes[2].set_xlabel("Epoch")
axes[2].legend()

plt.tight_layout()
plt.show()
