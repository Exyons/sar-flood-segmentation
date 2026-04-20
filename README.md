# Flood Inundation Mapping — Dual-Branch Pipeline

SAR-based flood segmentation with a hierarchical Vision Transformer (SegFormer from scratch) fused with classical machine learning on hand-engineered geospatial features. Built for **KSS605** coursework.

The project compares **six** model configurations on Sen1Floods11:

1. Standard SegFormer ViT
2. MLA SegFormer ViT (low-rank KV attention — ~6.5% parameter reduction vs. Standard)
3. Random Forest on geospatial features
4. XGBoost on geospatial features
5. Standard ViT + RF fusion
6. MLA ViT + RF fusion

## Architecture

```
           Sentinel-1 SAR (VV + VH)                DEM / slope / TWI / NDVI /
                     │                              distance-to-water / landcover
                     ▼                                         │
         ┌───────────────────────┐                             ▼
         │  SegFormer Encoder    │                   ┌───────────────────┐
         │  (Standard or MLA)    │                   │   RF / XGBoost    │
         │  4 hierarchical       │                   │   per-pixel       │
         │  stages + MLP head    │                   │   classification  │
         └──────────┬────────────┘                   └─────────┬─────────┘
                    │ ViT probs                                │ ML probs
                    └────────────────┬───────────────────────┘
                                     ▼
                           Weighted / Stacking
                                 Fusion
                                     │
                                     ▼
                              Flood mask (0 / 1)
```

### SegFormer (from scratch)

- **Overlap Patch Embedding** — 4×4 non-overlap (stage 1), 3×3 overlap (stages 2-4)
- **Mix Transformer encoder** — 4 stages, embed dims `[32, 64, 160, 256]`, spatial reduction ratios `[8, 4, 2, 1]`
- **Efficient self-attention** (Standard) **or** **MLA self-attention** (low-rank KV: `d_c = D // rank_divisor`)
- **Mix-FFN** — `Linear → DWConv3×3 → GELU → Linear` (no positional encoding)
- **MLP decode head** — unifies 4 stages via per-stage `Linear` + bilinear upsample

Verified parameters at B0 scale: **Standard 3.71M**, **MLA 3.47M**.

### Classical branch

Random Forest and XGBoost trained per-pixel on 6 hand-engineered features:

| Feature | Source |
|---|---|
| DEM | NASADEM / SRTM |
| Slope | `np.gradient` → arctan |
| TWI | `scipy` smoothed-DEM flow proxy |
| NDVI | Sentinel-2 red / NIR |
| Distance to water | `scipy.ndimage.distance_transform_edt` on JRC GSW |
| Land cover | ESA WorldCover |

### Fusion

- **Weighted average** — `α · ViT_prob + (1 − α) · ML_prob`, threshold at 0.5 (default α = 0.6)
- **Stacking** — `sklearn.LogisticRegression` meta-learner over `[vit_prob, rf_prob, xgb_prob]`

## Dataset

- **Sen1Floods11** — primary, 4,831 hand-labeled 512×512 chips at 10 m (Bolivia, Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA)
- **SEN12-FLOOD** — secondary (manual download from IEEE DataPort)

Expected layout under `data/sen1floods11/`:

```
v1.1/
├── data/flood_events/HandLabeled/
│   ├── S1Hand/          # *_S1Hand.tif  (VV + VH SAR)
│   └── LabelHand/       # *_LabelHand.tif  (0 = no flood, 1 = flood, −1/255 = ignore)
└── splits/flood_handlabeled/
    ├── train.txt
    └── test.txt
```

## Project layout

```
configs/default.yaml          # all hyperparameters
data/
  dataset.py                  # FloodDataset + get_dataloaders
  extract_geo_features.py     # DEM / slope / TWI / NDVI / distance / landcover
  download_sen1floods11.py
models/
  segformer/                  # patch_embed, attention, mix_ffn, block, encoder, decode_head
  segformer_model.py          # full SegFormer wrapper
  rf_model.py
  xgb_model.py
fusion/fuse.py                # weighted average + stacking
train_segformer.py            # ViT training (Standard or --use_mla)
train_rf_xgb.py               # classical branch training
evaluate.py                   # IoU / F1 / precision / recall (--compare runs all 6)
predict.py                    # single-tile side-by-side inference
notebooks/
  scripts/                    # scripts-first (# %% cell markers)
  convert_to_nb.py            # nbformat → .ipynb
```

## Setup

Python 3.13, managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Install `gsutil` separately to download Sen1Floods11.

## Training

### SegFormer ViT

```bash
# Standard attention
uv run python train_segformer.py --config configs/default.yaml

# MLA (low-rank KV) attention
uv run python train_segformer.py --config configs/default.yaml --use_mla
```

Checkpoints land in `checkpoints/segformer_{standard,mla}/best.pt` and `final.pt`.

### Random Forest + XGBoost

```bash
uv run python train_rf_xgb.py --config configs/default.yaml
```

Requires geospatial features at `data/geo_features/` — extract them first:

```bash
uv run python data/extract_geo_features.py --config configs/default.yaml
```

## Evaluation

Single model:

```bash
uv run python evaluate.py --config configs/default.yaml --model segformer --ckpt checkpoints/segformer_standard/best.pt
```

Full 6-model comparison table:

```bash
uv run python evaluate.py --config configs/default.yaml --compare
```

Single-tile visualization:

```bash
uv run python predict.py --tile <tile_name> --config configs/default.yaml
```

## Notebooks

Scripts-first: edit `notebooks/scripts/*.py`, then convert:

```bash
uv run python notebooks/convert_to_nb.py
```

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | sample tiles, SAR histograms, class distribution, geo features |
| `02_train_vit.ipynb` | parameter comparison, shape checks, train both ViT variants, curves |
| `03_train_rf_xgb.ipynb` | RF + XGB training, confusion matrices, feature importance |
| `04_fusion_eval.ipynb` | 6-model table, prediction panels, SHAP, fusion weight sweep |

## Configuration

All hyperparameters in `configs/default.yaml`:

- **SegFormer**: `embed_dims`, `num_heads`, `sr_ratios`, `num_blocks`, `mla_rank_divisor`
- **Training**: `batch_size`, `epochs`, `learning_rate`, `class_weights`, `poly_power`, `warmup_epochs`, `use_amp`
- **RF / XGB**: `n_estimators`, `max_depth`, `subsample`, `tree_method`
- **Fusion**: `vit_weight`, `ml_weight`

## Notes

- Augmentation is restricted to geospatially-safe ops (flip, 90° rotation) — no elastic or colour jitter.
- `richdem` and `pysheds` are **not** used (Python 3.13 incompatibility); TWI is approximated with `scipy.ndimage.uniform_filter`.
- XGBoost 3.x does not accept `use_label_encoder`; the wrapper omits it.

# How to run
Runbook:
## EE auth (once)
uv run python -c "import ee; ee.Authenticate()"

## Stage 1 pretrain
uv run python train_segformer.py --config configs/pretrain_hf.yaml

## Assam 2022 export (GCP project id)
uv run python -m gee.python.export_event --event assam2022 --project <gcp>
## wait for tasks on https://code.earthengine.google.com/tasks
uv run python -m gee.python.download_drive --event assam2022

## Stage 2 fine-tune
uv run python train_segformer.py --config configs/finetune_hf.yaml

## Per-event eval
uv run python evaluate.py \
  --ckpt checkpoints/segformer_hf_india_ft/best.pt \
  --split data/india_floods/splits/val.csv