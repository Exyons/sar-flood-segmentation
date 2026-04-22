"""PyTorch Datasets for Sen1Floods11 and GEE-exported India flood events.

Two dataset classes live here:

- ``Sen1FloodsDataset`` — CSV-driven, 3-channel SAR (VV, VH, VV−VH) stack,
  used by the HF SegFormer pre-train + India fine-tune pipeline.
- ``FloodDataset`` — the original per-directory, 2-channel SAR + 6-channel
  geospatial-feature dataset, kept as-is for the RF / XGBoost branch.
"""

import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset


class Sen1FloodsDataset(Dataset):
    """CSV-driven flood segmentation dataset.

    CSV format (one sample per row, two columns, paths relative to ``data_root``):

        <s1_relpath>,<label_relpath>

    Example Sen1Floods11:
        India/India_25540_S1Hand.tif,India/India_25540_LabelHand.tif

    Example India GEE export:
        assam2022/S1/chip_0003.tif,assam2022/Label/chip_0003.tif

    Each sample is returned as:
        {
          "image":      (3, H, W) float32 — VV, VH, VV-VH (clipped + min-max)
          "label":      (H, W)    int64   — {-1 ignore, 0 dry, 1 flood}
          "tile_name":  str
        }
    """

    # Per-channel clip ranges (dB) for Sentinel-1 GRD
    VV_CLIP = (-23.0, 0.0)
    VH_CLIP = (-28.0, -5.0)
    DIFF_CLIP = (-15.0, 15.0)  # VV - VH can swing both signs

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        label_key: str | None = "LabelHand",
        augment: bool = False,
        tile_size: int | None = None,
    ):
        """
        Args:
            csv_path:  Path to split CSV. Rows: ``<s1_rel>,<label_rel>``.
            data_root: Directory the CSV paths are relative to.
            label_key: If set, replaces ``_S1Hand.tif`` in the S1 filename
                       with ``_<label_key>.tif`` to derive the label path,
                       overriding the CSV label column. If the S1 filename
                       does not contain ``_S1Hand.tif``, falls back to the
                       CSV label column.
            augment:   Enable flip / 90° rotation augmentation.
        """
        self.data_root = Path(data_root)
        self.label_key = label_key
        self.augment = augment
        self.tile_size = tile_size   # None = keep native H/W (requires uniform tiles)

        self.samples: list[tuple[Path, Path, str]] = []
        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                s1_rel, label_rel = parts[0], parts[1]

                if label_key and "_S1Hand.tif" in s1_rel:
                    derived = s1_rel.replace("_S1Hand.tif", f"_{label_key}.tif")
                    label_rel = derived

                s1_path = self._resolve(s1_rel)
                label_path = self._resolve(label_rel)
                tile_name = Path(s1_rel).stem
                self.samples.append((s1_path, label_path, tile_name))

    def _resolve(self, rel: str) -> Path:
        """Resolve a CSV-relative path to an absolute path.

        Prefers ``data_root / rel``. If that doesn't exist and ``rel`` has no
        directory component, try ``data_root / <region>/<rel>`` where region is
        the filename's prefix before the first underscore (Sen1Floods11 convention).
        """
        candidate = self.data_root / rel
        if candidate.exists():
            return candidate
        if "/" not in rel and "_" in rel:
            region = rel.split("_", 1)[0]
            alt = self.data_root / region / rel
            if alt.exists():
                return alt
        return candidate  # let the open() fail loudly if truly missing

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s1_path, label_path, tile_name = self.samples[idx]

        with rasterio.open(s1_path) as src:
            sar = src.read().astype(np.float32)  # (C, H, W)

        if sar.shape[0] == 1:
            sar = np.concatenate([sar, sar], axis=0)
        elif sar.shape[0] > 2:
            sar = sar[:2]

        vv = sar[0]
        vh = sar[1]

        nan_mask = ~np.isfinite(vv) | ~np.isfinite(vh)
        vv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0)
        vh = np.nan_to_num(vh, nan=0.0, posinf=0.0, neginf=0.0)

        vv_n = self._clip_minmax(vv, *self.VV_CLIP)
        vh_n = self._clip_minmax(vh, *self.VH_CLIP)
        diff_n = self._clip_minmax(vv - vh, *self.DIFF_CLIP)

        image = np.stack([vv_n, vh_n, diff_n], axis=0)  # (3, H, W)

        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int64)

        # Sen1Floods11 convention: 255 → ignore. Map to -1.
        label = np.where(label == 255, -1, label)
        # Anything > 1 (and not already -1) is ambiguous → ignore.
        label = np.where((label > 1) | (label < -1), -1, label)

        if nan_mask.any():
            label[nan_mask] = -1

        if self.tile_size is not None:
            image = self._fit(image, self.tile_size, fill=0.0)
            label = self._fit(label, self.tile_size, fill=-1)

        if self.augment:
            image, label = self._augment(image, label)

        return {
            "image": torch.from_numpy(image.copy()),
            "label": torch.from_numpy(label.copy()),
            "tile_name": tile_name,
        }

    @staticmethod
    def _fit(arr: np.ndarray, size: int, fill: float) -> np.ndarray:
        """Center-crop or symmetrically pad the last two axes to ``size``."""
        h, w = arr.shape[-2:]
        y0 = max(0, (h - size) // 2)
        x0 = max(0, (w - size) // 2)
        arr = arr[..., y0:y0 + size, x0:x0 + size]
        h, w = arr.shape[-2:]
        if h == size and w == size:
            return arr
        pad_h, pad_w = size - h, size - w
        pads = [(0, 0)] * (arr.ndim - 2) + [
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        ]
        return np.pad(arr, pads, mode="constant", constant_values=fill)

    @staticmethod
    def _clip_minmax(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo + 1e-8)

    def _augment(self, image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() > 0.5:
            image = np.flip(image, axis=2)
            label = np.flip(label, axis=1)
        if random.random() > 0.5:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=0)
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(0, 1))
        return image, label


# ---------------------------------------------------------------------------
# Legacy dataset (kept for the RF / XGBoost branch which needs geo features)
# ---------------------------------------------------------------------------


class FloodDataset(Dataset):
    """Sen1Floods11 with geospatial feature augmentation for RF/XGB training."""

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
        self.sar_dir = Path(sar_dir)
        self.mask_dir = Path(mask_dir)
        self.geo_dir = Path(geo_dir) if geo_dir else None
        self.augment = augment

        if tile_names is not None:
            self.tiles = tile_names
        elif split_file and Path(split_file).exists():
            with open(split_file) as f:
                self.tiles = [line.strip() for line in f if line.strip()]
        else:
            self.tiles = sorted(
                p.stem.replace("_S1Hand", "")
                for p in self.sar_dir.glob("*_S1Hand.tif")
            )

        if not self.tiles:
            self.tiles = sorted(p.stem for p in self.sar_dir.glob("*.tif"))

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict:
        tile_name = self.tiles[idx]

        sar_path = self.sar_dir / f"{tile_name}_S1Hand.tif"
        if not sar_path.exists():
            sar_path = self.sar_dir / f"{tile_name}.tif"
        with rasterio.open(sar_path) as src:
            sar = src.read().astype(np.float32)

        if sar.shape[0] == 1:
            sar = np.concatenate([sar, sar], axis=0)
        elif sar.shape[0] > 2:
            sar = sar[:2]

        for c in range(2):
            sar[c] = (sar[c] - self.SAR_MEAN[c]) / (self.SAR_STD[c] + 1e-8)

        mask_path = self.mask_dir / f"{tile_name}_LabelHand.tif"
        if not mask_path.exists():
            mask_path = self.mask_dir / f"{tile_name}.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        mask = np.where(mask > 1, 0, mask)
        mask = np.clip(mask, 0, 1)

        if self.geo_dir:
            geo_path = self.geo_dir / f"{tile_name}_S1Hand_geo.npy"
            if not geo_path.exists():
                geo_path = self.geo_dir / f"{tile_name}_geo.npy"
            if geo_path.exists():
                geo = np.load(geo_path).astype(np.float32)
            else:
                geo = np.zeros((6, sar.shape[1], sar.shape[2]), dtype=np.float32)
        else:
            geo = np.zeros((6, sar.shape[1], sar.shape[2]), dtype=np.float32)

        for c in range(geo.shape[0]):
            cmin, cmax = geo[c].min(), geo[c].max()
            if cmax - cmin > 1e-8:
                geo[c] = (geo[c] - cmin) / (cmax - cmin)

        if self.augment:
            sar, geo, mask = self._augment(sar, geo, mask)

        return {
            "sar": torch.from_numpy(sar),
            "geo": torch.from_numpy(geo),
            "mask": torch.from_numpy(mask),
            "tile_name": tile_name,
        }

    def _augment(self, sar, geo, mask):
        if random.random() > 0.5:
            sar = np.flip(sar, axis=2).copy()
            geo = np.flip(geo, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() > 0.5:
            sar = np.flip(sar, axis=1).copy()
            geo = np.flip(geo, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        k = random.randint(0, 3)
        if k > 0:
            sar = np.rot90(sar, k, axes=(1, 2)).copy()
            geo = np.rot90(geo, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return sar, geo, mask


def get_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Legacy per-dir train/val/test loaders for the RF/XGB flow."""
    sen1_dir = Path(cfg["paths"]["sen1floods11_dir"])
    geo_dir = Path(cfg["paths"]["geo_features_dir"])

    sar_dir = sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
    mask_dir = sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"

    all_tiles = sorted(
        p.stem.replace("_S1Hand", "")
        for p in sar_dir.glob("*_S1Hand.tif")
    )
    if not all_tiles:
        all_tiles = sorted(p.stem for p in sar_dir.glob("*.tif"))

    split_dir = sen1_dir / "v1.1" / "splits" / "flood_handlabeled"
    train_file = split_dir / "train.txt"
    test_file = split_dir / "test.txt"

    if train_file.exists() and test_file.exists():
        with open(train_file) as f:
            train_tiles = [l.strip() for l in f if l.strip()]
        with open(test_file) as f:
            test_tiles = [l.strip() for l in f if l.strip()]
        val_ratio = cfg["splits"]["val_ratio"]
        n_val = max(1, int(len(train_tiles) * val_ratio))
        random.seed(cfg["training"]["seed"])
        random.shuffle(train_tiles)
        val_tiles = train_tiles[:n_val]
        train_tiles = train_tiles[n_val:]
    else:
        random.seed(cfg["training"]["seed"])
        random.shuffle(all_tiles)
        n = len(all_tiles)
        n_test = max(1, int(n * 0.15))
        n_val = max(1, int(n * 0.15))
        test_tiles = all_tiles[:n_test]
        val_tiles = all_tiles[n_test : n_test + n_val]
        train_tiles = all_tiles[n_test + n_val :]

    geo_dir_str = str(geo_dir) if geo_dir.exists() else None

    train_ds = FloodDataset(str(sar_dir), str(mask_dir), geo_dir_str,
                            tile_names=train_tiles, augment=True)
    val_ds = FloodDataset(str(sar_dir), str(mask_dir), geo_dir_str,
                          tile_names=val_tiles, augment=False)
    test_ds = FloodDataset(str(sar_dir), str(mask_dir), geo_dir_str,
                           tile_names=test_tiles, augment=False)

    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
    )


def get_sen1floods_loader(
    csv_path: str | Path,
    data_root: str | Path,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = False,
    augment: bool = False,
    label_key: str | None = "LabelHand",
) -> DataLoader:
    """Thin helper to build a DataLoader for ``Sen1FloodsDataset``."""
    ds = Sen1FloodsDataset(csv_path, data_root, label_key=label_key, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
