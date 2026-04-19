# %% [markdown]
# # 01 — Exploratory Data Analysis
# ## Sen1Floods11 SAR + Flood Masks
#
# Visualize SAR imagery (VV/VH), flood masks, class distributions,
# and geospatial feature histograms.

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import yaml

with open("../configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

sen1_dir = Path(cfg["paths"]["sen1floods11_dir"])
geo_dir = Path(cfg["paths"]["geo_features_dir"])

# %% [markdown]
# ## Load a Sample Tile

# %%
sar_dir = sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
mask_dir = sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"

sar_tiles = sorted(sar_dir.glob("*.tif"))
print(f"Found {len(sar_tiles)} SAR tiles")

# Pick first tile
tile_path = sar_tiles[0]
tile_name = tile_path.stem.replace("_S1Hand", "")
print(f"Sample tile: {tile_name}")

with rasterio.open(tile_path) as src:
    sar = src.read().astype(np.float32)
    print(f"SAR shape: {sar.shape}, dtype: {sar.dtype}")
    print(f"CRS: {src.crs}, Bounds: {src.bounds}")

mask_path = mask_dir / f"{tile_name}_LabelHand.tif"
with rasterio.open(mask_path) as src:
    mask = src.read(1)
    print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")

# %% [markdown]
# ## Visualize SAR Bands + Mask

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(sar[0], cmap="gray", vmin=np.percentile(sar[0], 2), vmax=np.percentile(sar[0], 98))
axes[0].set_title(f"VV — {tile_name}")
axes[0].axis("off")

if sar.shape[0] > 1:
    axes[1].imshow(sar[1], cmap="gray", vmin=np.percentile(sar[1], 2), vmax=np.percentile(sar[1], 98))
    axes[1].set_title("VH")
else:
    axes[1].set_title("VH (not available)")
axes[1].axis("off")

axes[2].imshow(mask, cmap="Blues", vmin=0, vmax=1)
axes[2].set_title("Flood Mask")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## VV/VH Histograms

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(sar[0].flatten(), bins=100, alpha=0.7, label="VV", color="steelblue")
axes[0].set_title("VV Backscatter Distribution")
axes[0].set_xlabel("dB")
axes[0].legend()

if sar.shape[0] > 1:
    axes[1].hist(sar[1].flatten(), bins=100, alpha=0.7, label="VH", color="coral")
    axes[1].set_title("VH Backscatter Distribution")
    axes[1].set_xlabel("dB")
    axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Class Distribution Across Dataset

# %%
flood_counts = []
total_counts = []

for mask_path in sorted(mask_dir.glob("*.tif"))[:50]:  # sample 50 tiles
    with rasterio.open(mask_path) as src:
        m = src.read(1)
        m = np.clip(m, 0, 1)
        flood_counts.append(m.sum())
        total_counts.append(m.size)

flood_ratio = np.array(flood_counts) / np.array(total_counts) * 100
print(f"Mean flood coverage: {flood_ratio.mean():.2f}%")
print(f"Median flood coverage: {np.median(flood_ratio):.2f}%")
print(f"Max flood coverage: {flood_ratio.max():.2f}%")

plt.figure(figsize=(10, 4))
plt.bar(range(len(flood_ratio)), sorted(flood_ratio, reverse=True), color="steelblue")
plt.xlabel("Tile index (sorted)")
plt.ylabel("Flood coverage (%)")
plt.title("Flood Coverage per Tile")
plt.show()

# %% [markdown]
# ## Geospatial Feature Visualization

# %%
geo_files = sorted(Path(geo_dir).glob("*_geo.npy"))
print(f"Found {len(geo_files)} geo feature files")

if geo_files:
    geo = np.load(geo_files[0])
    feature_names = cfg["geo_features"]["features"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (ax, name) in enumerate(zip(axes.flatten(), feature_names)):
        if i < geo.shape[0]:
            im = ax.imshow(geo[i], cmap="viridis")
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(name)
        ax.axis("off")
    plt.suptitle(f"Geospatial Features — {geo_files[0].stem}", fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("No geo features extracted yet. Run: uv run python data/extract_geo_features.py")
