"""Flood Inundation Mapping — SegFormer + RF/XGBoost Fusion.

Entry point for quick verification and model summary.
See individual scripts for training and evaluation:
    - train_segformer.py    Train SegFormer (Standard or MLA)
    - train_rf_xgb.py       Train RF + XGBoost
    - evaluate.py           Evaluate and compare all models
    - predict.py            Inference + visualization
"""

import torch
from models.segformer_model import SegFormer


def main():
    print("Flood Inundation Mapping — SegFormer + RF/XGBoost Fusion")
    print("=" * 55)

    # Model summaries
    model_std = SegFormer(in_channels=2, num_classes=2, use_mla=False)
    model_mla = SegFormer(in_channels=2, num_classes=2, use_mla=True, rank_divisor=4)

    print(f"\n{model_std.param_summary()}")
    print(model_mla.param_summary())

    reduction = (1 - model_mla.count_parameters() / model_std.count_parameters()) * 100
    print(f"MLA parameter reduction: {reduction:.1f}%")

    # Shape check
    x = torch.randn(1, 2, 512, 512)
    with torch.no_grad():
        out = model_std(x)
    print(f"\nInput:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")

    print("\nRun commands:")
    print("  uv run python train_segformer.py --config configs/default.yaml")
    print("  uv run python train_segformer.py --config configs/default.yaml --use_mla")
    print("  uv run python train_rf_xgb.py --config configs/default.yaml")
    print("  uv run python evaluate.py --config configs/default.yaml --compare")
    print("  uv run python notebooks/convert_to_nb.py")


if __name__ == "__main__":
    main()
