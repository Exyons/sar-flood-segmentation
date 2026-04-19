"""Training script for Random Forest and XGBoost classifiers.

Loads geo feature stacks + masks, subsamples pixels, trains both models.

Usage:
    uv run python train_rf_xgb.py --config configs/default.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from models.rf_model import RFFloodModel
from models.xgb_model import XGBFloodModel


def load_pixel_data(
    geo_dir: str,
    mask_dir: str,
    tile_names: list[str],
    max_pixels: int = 1_000_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and flatten geo features + masks into pixel arrays.

    Args:
        geo_dir: Directory with *_geo.npy files
        mask_dir: Directory with *_LabelHand.tif mask files
        tile_names: List of tile basenames
        max_pixels: Max pixels to sample (for memory)
        seed: Random seed

    Returns:
        X: (N_pixels, N_features)
        y: (N_pixels,) binary labels
    """
    import rasterio

    geo_path = Path(geo_dir)
    mask_path = Path(mask_dir)

    all_X = []
    all_y = []

    for tile_name in tqdm(tile_names, desc="Loading tiles"):
        # Load geo features
        geo_file = geo_path / f"{tile_name}_S1Hand_geo.npy"
        if not geo_file.exists():
            geo_file = geo_path / f"{tile_name}_geo.npy"
        if not geo_file.exists():
            continue

        geo = np.load(geo_file).astype(np.float32)  # (6, H, W)

        # Load mask
        mask_file = mask_path / f"{tile_name}_LabelHand.tif"
        if not mask_file.exists():
            mask_file = mask_path / f"{tile_name}.tif"
        if not mask_file.exists():
            continue

        with rasterio.open(mask_file) as src:
            mask = src.read(1).astype(np.int64)

        # Binarize mask
        mask = np.clip(mask, 0, 1)
        valid = (mask >= 0) & (mask <= 1)

        # Flatten
        C, H, W = geo.shape
        X_flat = geo.reshape(C, -1).T  # (H*W, C)
        y_flat = mask.flatten()
        valid_flat = valid.flatten()

        all_X.append(X_flat[valid_flat])
        all_y.append(y_flat[valid_flat])

    if not all_X:
        raise ValueError("No valid tiles found. Check geo features and mask paths.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Subsample if too many pixels
    if len(y) > max_pixels:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y), max_pixels, replace=False)
        X = X[idx]
        y = y[idx]

    print(f"Loaded {len(y)} pixels ({y.sum()} flood, {len(y) - y.sum()} non-flood)")
    return X, y


def get_tile_names(cfg: dict) -> tuple[list[str], list[str]]:
    """Get train/test tile names from dataset."""
    sen1_dir = Path(cfg["paths"]["sen1floods11_dir"])
    sar_dir = sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"

    all_tiles = sorted(
        p.stem.replace("_S1Hand", "")
        for p in sar_dir.glob("*_S1Hand.tif")
    )

    if not all_tiles:
        all_tiles = sorted(p.stem for p in sar_dir.glob("*.tif"))

    # Split
    seed = cfg["training"]["seed"]
    random.seed(seed)
    random.shuffle(all_tiles)
    n = len(all_tiles)
    n_test = max(1, int(n * 0.15))
    test_tiles = all_tiles[:n_test]
    train_tiles = all_tiles[n_test:]

    return train_tiles, test_tiles


def main():
    parser = argparse.ArgumentParser(description="Train RF and XGBoost")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sen1_dir = Path(cfg["paths"]["sen1floods11_dir"])
    geo_dir = cfg["paths"]["geo_features_dir"]
    mask_dir = str(sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand")
    max_pixels = cfg["geo_features"]["pixel_subsample"]

    train_tiles, test_tiles = get_tile_names(cfg)
    print(f"Train tiles: {len(train_tiles)}, Test tiles: {len(test_tiles)}")

    # Load data
    print("\n--- Loading training data ---")
    X_train, y_train = load_pixel_data(geo_dir, mask_dir, train_tiles, max_pixels)

    print("\n--- Loading test data ---")
    X_test, y_test = load_pixel_data(geo_dir, mask_dir, test_tiles, max_pixels // 4)

    # Split some training data for XGB validation
    val_size = min(100_000, len(X_train) // 5)
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train_xgb, y_train_xgb = X_train[val_size:], y_train[val_size:]

    # --- Train Random Forest ---
    print("\n" + "=" * 50)
    print("Training Random Forest")
    print("=" * 50)

    rf_cfg = cfg["rf"]
    rf = RFFloodModel(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        n_jobs=rf_cfg["n_jobs"],
        random_state=rf_cfg["random_state"],
    )
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    print("\nRF Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=["no-flood", "flood"]))
    print(f"RF F1 (flood): {f1_score(y_test, rf_pred, pos_label=1):.4f}")

    rf_path = str(Path(cfg["paths"]["checkpoints_dir"]) / "rf_model.joblib")
    rf.save(rf_path)
    print(f"RF saved to {rf_path}")

    # --- Train XGBoost ---
    print("\n" + "=" * 50)
    print("Training XGBoost")
    print("=" * 50)

    xgb_cfg = cfg["xgb"]
    xgb = XGBFloodModel(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        tree_method=xgb_cfg["tree_method"],
        random_state=xgb_cfg["random_state"],
    )
    xgb.fit(X_train_xgb, y_train_xgb, X_val=X_val, y_val=y_val)

    xgb_pred = xgb.predict(X_test)
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, xgb_pred, target_names=["no-flood", "flood"]))
    print(f"XGB F1 (flood): {f1_score(y_test, xgb_pred, pos_label=1):.4f}")

    xgb_path = str(Path(cfg["paths"]["checkpoints_dir"]) / "xgb_model.json")
    xgb.save(xgb_path)
    print(f"XGBoost saved to {xgb_path}")

    # Feature importance
    feature_names = cfg["geo_features"]["features"]
    importances = xgb.feature_importance()
    print("\nXGBoost Feature Importance:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")


if __name__ == "__main__":
    main()
