"""Inference and visualization on individual tiles.

Usage:
    uv run python predict.py --config configs/default.yaml --tile Bolivia_7
    uv run python predict.py --config configs/default.yaml --tile Bolivia_7 --use_mla
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import yaml

from models.segformer_model import SegFormer
from models.rf_model import RFFloodModel
from fusion.fuse import weighted_average_fusion


def load_tile(tile_name: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SAR, geo features, and mask for a tile."""
    sen1_dir = Path(cfg["paths"]["sen1floods11_dir"])
    geo_dir = Path(cfg["paths"]["geo_features_dir"])

    # SAR
    sar_path = sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand" / f"{tile_name}_S1Hand.tif"
    with rasterio.open(sar_path) as src:
        sar = src.read().astype(np.float32)
    if sar.shape[0] > 2:
        sar = sar[:2]

    # Z-score normalize
    sar_mean = np.array([-12.0, -19.0])
    sar_std = np.array([5.0, 6.0])
    for c in range(2):
        sar[c] = (sar[c] - sar_mean[c]) / (sar_std[c] + 1e-8)

    # Mask
    mask_path = sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand" / f"{tile_name}_LabelHand.tif"
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.int64)
    mask = np.clip(mask, 0, 1)

    # Geo features
    geo_path = geo_dir / f"{tile_name}_S1Hand_geo.npy"
    if not geo_path.exists():
        geo_path = geo_dir / f"{tile_name}_geo.npy"
    if geo_path.exists():
        geo = np.load(geo_path).astype(np.float32)
    else:
        geo = np.zeros((6, sar.shape[1], sar.shape[2]), dtype=np.float32)

    return sar, geo, mask


def predict_tile(
    tile_name: str,
    cfg: dict,
    use_mla: bool = False,
) -> dict[str, np.ndarray]:
    """Run all models on a tile and return predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sar, geo, mask = load_tile(tile_name, cfg)

    results = {"ground_truth": mask, "sar_vv": sar[0]}
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])

    # SegFormer
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
    ckpt_path = ckpt_dir / f"segformer_{variant}" / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)

    with torch.no_grad():
        sar_tensor = torch.from_numpy(sar).unsqueeze(0).to(device)
        logits = model(sar_tensor)
        vit_probs = F.softmax(logits, dim=1)[0, 1].cpu().numpy()

    vit_name = f"{'MLA' if use_mla else 'Standard'} ViT"
    results[vit_name] = (vit_probs >= 0.5).astype(np.int64)

    # RF
    rf_path = ckpt_dir / "rf_model.joblib"
    if rf_path.exists():
        rf = RFFloodModel()
        rf.load(str(rf_path))
        rf_probs = rf.predict_tile(geo)
        results["RF"] = (rf_probs >= 0.5).astype(np.int64)

        # Fusion
        fused_probs, fused_pred = weighted_average_fusion(
            vit_probs, rf_probs,
            cfg["fusion"]["vit_weight"], cfg["fusion"]["ml_weight"],
        )
        results[f"Fused ({vit_name}+RF)"] = fused_pred

    return results


def visualize_predictions(results: dict[str, np.ndarray], tile_name: str, save_path: str | None = None):
    """Plot side-by-side prediction maps."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        if name == "sar_vv":
            ax.imshow(data, cmap="gray")
        else:
            ax.imshow(data, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    plt.suptitle(f"Flood Prediction — {tile_name}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predict and visualize")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tile", required=True, help="Tile basename (e.g., Bolivia_7)")
    parser.add_argument("--use_mla", action="store_true")
    parser.add_argument("--save", default=None, help="Save path for figure")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = predict_tile(args.tile, cfg, use_mla=args.use_mla)
    visualize_predictions(results, args.tile, save_path=args.save)


if __name__ == "__main__":
    main()
