# %% [markdown]
# # 04 — Fusion, Evaluation & Model Comparison
#
# Compare all 6 models:
# 1. Standard ViT
# 2. MLA ViT
# 3. Random Forest (geo only)
# 4. XGBoost (geo only)
# 5. Standard ViT + RF (fused)
# 6. MLA ViT + RF (fused)
#
# Includes SHAP analysis and per-event breakdown.

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from tqdm import tqdm

from data.dataset import get_dataloaders
from models.segformer_model import SegFormer
from models.rf_model import RFFloodModel
from models.xgb_model import XGBFloodModel
from fusion.fuse import weighted_average_fusion
from evaluate import Metrics, load_segformer, get_segformer_probs

with open("../configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, test_loader = get_dataloaders(cfg)
ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
fusion_cfg = cfg["fusion"]

print(f"Test set: {len(test_loader.dataset)} tiles")

# %% [markdown]
# ## Load All Models

# %%
# Standard ViT
std_model = load_segformer(cfg, use_mla=False, device=device)
std_data = get_segformer_probs(std_model, test_loader, device)

# MLA ViT
mla_model = load_segformer(cfg, use_mla=True, device=device)
mla_data = get_segformer_probs(mla_model, test_loader, device)

# RF
rf = RFFloodModel()
rf.load(str(ckpt_dir / "rf_model.joblib"))

# XGBoost
xgb = XGBFloodModel()
xgb.load(str(ckpt_dir / "xgb_model.json"))

# %% [markdown]
# ## Full 6-Model Comparison

# %%
results = {}

# Standard ViT
m = Metrics()
for probs, mask, _ in std_data:
    m.update((probs >= 0.5).astype(int), mask)
results["Standard ViT"] = {**m.summary(), "params": f"{std_model.count_parameters()/1e6:.2f}M"}

# MLA ViT
m = Metrics()
for probs, mask, _ in mla_data:
    m.update((probs >= 0.5).astype(int), mask)
results["MLA ViT"] = {**m.summary(), "params": f"{mla_model.count_parameters()/1e6:.2f}M"}

# RF
m = Metrics()
for _, mask, geo in std_data:
    rf_probs = rf.predict_tile(geo)
    m.update((rf_probs >= 0.5).astype(int), mask)
results["RF (geo only)"] = {**m.summary(), "params": "—"}

# XGBoost
m = Metrics()
for _, mask, geo in std_data:
    xgb_probs = xgb.predict_tile(geo)
    m.update((xgb_probs >= 0.5).astype(int), mask)
results["XGBoost (geo only)"] = {**m.summary(), "params": "—"}

# Standard ViT + RF
m = Metrics()
for vit_probs, mask, geo in std_data:
    ml_probs = rf.predict_tile(geo)
    _, fused = weighted_average_fusion(vit_probs, ml_probs,
                                        fusion_cfg["vit_weight"], fusion_cfg["ml_weight"])
    m.update(fused, mask)
results["Standard ViT + RF"] = {**m.summary(), "params": "—"}

# MLA ViT + RF
m = Metrics()
for vit_probs, mask, geo in mla_data:
    ml_probs = rf.predict_tile(geo)
    _, fused = weighted_average_fusion(vit_probs, ml_probs,
                                        fusion_cfg["vit_weight"], fusion_cfg["ml_weight"])
    m.update(fused, mask)
results["MLA ViT + RF"] = {**m.summary(), "params": "—"}

# Print table
print(f"\n{'Model':<25} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Params':>10}")
print("-" * 75)
for name, data in results.items():
    print(f"{name:<25} {data['IoU']:>8.4f} {data['F1']:>8.4f} "
          f"{data['Precision']:>8.4f} {data['Recall']:>8.4f} {data['params']:>10}")

# %% [markdown]
# ## Comparison Bar Charts

# %%
model_names = list(results.keys())
metrics_to_plot = ["IoU", "F1", "Precision", "Recall"]

fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 5))
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

for ax, metric in zip(axes, metrics_to_plot):
    values = [results[n][metric] for n in model_names]
    bars = ax.bar(range(len(model_names)), values, color=colors)
    ax.set_title(metric)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Side-by-Side Prediction Maps

# %%
# Visualize first 3 test tiles
for idx in range(min(3, len(std_data))):
    vit_probs_std, mask, geo = std_data[idx]
    vit_probs_mla = mla_data[idx][0]
    rf_probs = rf.predict_tile(geo)
    _, fused_std = weighted_average_fusion(vit_probs_std, rf_probs,
                                            fusion_cfg["vit_weight"], fusion_cfg["ml_weight"])
    _, fused_mla = weighted_average_fusion(vit_probs_mla, rf_probs,
                                            fusion_cfg["vit_weight"], fusion_cfg["ml_weight"])

    panels = {
        "Ground Truth": mask,
        "Standard ViT": (vit_probs_std >= 0.5).astype(int),
        "MLA ViT": (vit_probs_mla >= 0.5).astype(int),
        "RF": (rf_probs >= 0.5).astype(int),
        "Fused (Std)": fused_std,
        "Fused (MLA)": fused_mla,
    }

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    for ax, (title, data) in zip(axes, panels.items()):
        ax.imshow(data, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.suptitle(f"Test Tile {idx + 1}", fontsize=14)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## SHAP Analysis (XGBoost)

# %%
import shap

# Sample data for SHAP
from train_rf_xgb import load_pixel_data, get_tile_names

train_tiles, test_tiles = get_tile_names(cfg)
sen1_dir = Path(cfg["paths"]["sen1floods11_dir"])
mask_dir = str(sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand")

X_test_shap, y_test_shap = load_pixel_data(
    cfg["paths"]["geo_features_dir"], mask_dir, test_tiles[:5], max_pixels=10000
)

feature_names = cfg["geo_features"]["features"]
explainer = shap.TreeExplainer(xgb.model)
shap_values = explainer.shap_values(X_test_shap)

# %%
shap.summary_plot(shap_values, X_test_shap, feature_names=feature_names, show=True)

# %%
shap.summary_plot(shap_values, X_test_shap, feature_names=feature_names,
                  plot_type="bar", show=True)

# %% [markdown]
# ## Fusion Weight Sensitivity

# %%
vit_weights = np.arange(0.0, 1.05, 0.1)
ious_std = []
ious_mla = []

for w in vit_weights:
    m_std = Metrics()
    m_mla = Metrics()
    for i in range(len(std_data)):
        vit_std, mask, geo = std_data[i]
        vit_mla = mla_data[i][0]
        ml = rf.predict_tile(geo)
        _, pred_std = weighted_average_fusion(vit_std, ml, w, 1 - w)
        _, pred_mla = weighted_average_fusion(vit_mla, ml, w, 1 - w)
        m_std.update(pred_std, mask)
        m_mla.update(pred_mla, mask)
    ious_std.append(m_std.iou)
    ious_mla.append(m_mla.iou)

plt.figure(figsize=(8, 5))
plt.plot(vit_weights, ious_std, "o-", label="Standard ViT + RF")
plt.plot(vit_weights, ious_mla, "s-", label="MLA ViT + RF")
plt.xlabel("ViT Weight (ML Weight = 1 - ViT Weight)")
plt.ylabel("IoU")
plt.title("Fusion Weight Sensitivity")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Best Standard ViT weight: {vit_weights[np.argmax(ious_std)]:.1f} (IoU={max(ious_std):.4f})")
print(f"Best MLA ViT weight: {vit_weights[np.argmax(ious_mla)]:.1f} (IoU={max(ious_mla):.4f})")

# %% [markdown]
# ## Per-Event Breakdown
#
# Sen1Floods11 covers 11 flood events. Analyze performance per event.

# %%
# Group test tiles by event (first part of tile name before "_")
event_tiles = {}
for i, batch in enumerate(test_loader):
    for name in batch["tile_name"]:
        event = name.rsplit("_", 1)[0]  # e.g., "Bolivia" from "Bolivia_7"
        if event not in event_tiles:
            event_tiles[event] = []
        event_tiles[event].append(name)

print(f"Events in test set: {list(event_tiles.keys())}")
print(f"Tiles per event: {[(k, len(v)) for k, v in event_tiles.items()]}")
