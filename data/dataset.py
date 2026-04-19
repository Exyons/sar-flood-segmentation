"""PyTorch Dataset for Sen1Floods11 + geospatial features.

Returns per sample:
    sar_image:     (2, 512, 512) — VV + VH
    geo_features:  (6, 512, 512) — DEM, slope, TWI, NDVI, dist_water, landcover
    mask:          (512, 512)    — binary flood label (0=no-flood, 1=flood)
"""

import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

import yaml


class FloodDataset(Dataset):
    """Sen1Floods11 dataset with geospatial feature augmentation."""

    # Per-dataset statistics for z-score normalization (VV, VH)
    # Computed from Sen1Floods11 training split — update after first pass
    SAR_MEAN = np.array([-12.0, -19.0], dtype=np.float32)
    SAR_STD = np.array([5.0, 6.0], dtype=np.float32)

    def __init__(
        self,
        sar_dir: str,
        mask_dir: str,
        geo_dir: str | None = None,
        split_file: str | None = None,
        tile_names: list[str] | None = None,
        augment: bool = False,
    ):
        """
        Args:
            sar_dir:    Directory with *_S1Hand.tif SAR tiles
            mask_dir:   Directory with *_LabelHand.tif mask tiles
            geo_dir:    Directory with *_geo.npy geo feature files
            split_file: Path to text file listing tile basenames
            tile_names: Explicit list of tile basenames (alternative to split_file)
            augment:    Enable geospatially-safe augmentation
        """
        self.sar_dir = Path(sar_dir)
        self.mask_dir = Path(mask_dir)
        self.geo_dir = Path(geo_dir) if geo_dir else None
        self.augment = augment

        # Resolve tile list
        if tile_names is not None:
            self.tiles = tile_names
        elif split_file and Path(split_file).exists():
            with open(split_file) as f:
                self.tiles = [line.strip() for line in f if line.strip()]
        else:
            # Auto-discover from SAR directory
            self.tiles = sorted(
                p.stem.replace("_S1Hand", "")
                for p in self.sar_dir.glob("*_S1Hand.tif")
            )

        if not self.tiles:
            # Try without suffix pattern
            self.tiles = sorted(
                p.stem for p in self.sar_dir.glob("*.tif")
            )

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict:
        tile_name = self.tiles[idx]

        # Load SAR (VV + VH)
        sar_path = self.sar_dir / f"{tile_name}_S1Hand.tif"
        if not sar_path.exists():
            sar_path = self.sar_dir / f"{tile_name}.tif"
        with rasterio.open(sar_path) as src:
            sar = src.read().astype(np.float32)  # (C, H, W)

        # Ensure 2 channels
        if sar.shape[0] == 1:
            sar = np.concatenate([sar, sar], axis=0)
        elif sar.shape[0] > 2:
            sar = sar[:2]

        # Z-score normalize SAR
        for c in range(2):
            sar[c] = (sar[c] - self.SAR_MEAN[c]) / (self.SAR_STD[c] + 1e-8)

        # Load mask
        mask_path = self.mask_dir / f"{tile_name}_LabelHand.tif"
        if not mask_path.exists():
            mask_path = self.mask_dir / f"{tile_name}.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        # Binarize: -1 or 255 = ignore, 0 = no flood, 1 = flood
        mask = np.where(mask > 1, 0, mask)  # clamp invalid to 0
        mask = np.clip(mask, 0, 1)

        # Load geo features
        if self.geo_dir:
            geo_path = self.geo_dir / f"{tile_name}_S1Hand_geo.npy"
            if not geo_path.exists():
                geo_path = self.geo_dir / f"{tile_name}_geo.npy"
            if geo_path.exists():
                geo = np.load(geo_path).astype(np.float32)  # (6, H, W)
            else:
                geo = np.zeros((6, sar.shape[1], sar.shape[2]), dtype=np.float32)
        else:
            geo = np.zeros((6, sar.shape[1], sar.shape[2]), dtype=np.float32)

        # Min-max normalize geo features per channel
        for c in range(geo.shape[0]):
            cmin, cmax = geo[c].min(), geo[c].max()
            if cmax - cmin > 1e-8:
                geo[c] = (geo[c] - cmin) / (cmax - cmin)

        # Augmentation (geospatially safe)
        if self.augment:
            sar, geo, mask = self._augment(sar, geo, mask)

        return {
            "sar": torch.from_numpy(sar),
            "geo": torch.from_numpy(geo),
            "mask": torch.from_numpy(mask),
            "tile_name": tile_name,
        }

    def _augment(
        self, sar: np.ndarray, geo: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply geospatially-safe augmentations."""
        # Random horizontal flip
        if random.random() > 0.5:
            sar = np.flip(sar, axis=2).copy()
            geo = np.flip(geo, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random vertical flip
        if random.random() > 0.5:
            sar = np.flip(sar, axis=1).copy()
            geo = np.flip(geo, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            sar = np.rot90(sar, k, axes=(1, 2)).copy()
            geo = np.rot90(geo, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        return sar, geo, mask


def get_dataloaders(
    cfg: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from config."""
    sen1_dir = Path(cfg["paths"]["sen1floods11_dir"])
    geo_dir = Path(cfg["paths"]["geo_features_dir"])

    sar_dir = sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
    mask_dir = sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"

    # Discover all tiles
    all_tiles = sorted(
        p.stem.replace("_S1Hand", "")
        for p in sar_dir.glob("*_S1Hand.tif")
    )

    if not all_tiles:
        # Fallback: try without suffix
        all_tiles = sorted(p.stem for p in sar_dir.glob("*.tif"))

    # Train/val/test split
    # Use Sen1Floods11 official split files if available
    split_dir = sen1_dir / "v1.1" / "splits" / "flood_handlabeled"
    train_file = split_dir / "train.txt"
    test_file = split_dir / "test.txt"

    if train_file.exists() and test_file.exists():
        with open(train_file) as f:
            train_tiles = [l.strip() for l in f if l.strip()]
        with open(test_file) as f:
            test_tiles = [l.strip() for l in f if l.strip()]
        # Split train into train/val
        val_ratio = cfg["splits"]["val_ratio"]
        n_val = max(1, int(len(train_tiles) * val_ratio))
        random.seed(cfg["training"]["seed"])
        random.shuffle(train_tiles)
        val_tiles = train_tiles[:n_val]
        train_tiles = train_tiles[n_val:]
    else:
        # Manual split
        random.seed(cfg["training"]["seed"])
        random.shuffle(all_tiles)
        n = len(all_tiles)
        n_test = max(1, int(n * 0.15))
        n_val = max(1, int(n * 0.15))
        test_tiles = all_tiles[:n_test]
        val_tiles = all_tiles[n_test : n_test + n_val]
        train_tiles = all_tiles[n_test + n_val :]

    geo_dir_str = str(geo_dir) if geo_dir.exists() else None

    train_ds = FloodDataset(
        str(sar_dir), str(mask_dir), geo_dir_str,
        tile_names=train_tiles, augment=True,
    )
    val_ds = FloodDataset(
        str(sar_dir), str(mask_dir), geo_dir_str,
        tile_names=val_tiles, augment=False,
    )
    test_ds = FloodDataset(
        str(sar_dir), str(mask_dir), geo_dir_str,
        tile_names=test_tiles, augment=False,
    )

    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    return train_loader, val_loader, test_loader
