# Plan: SegFormer pre-train on Sen1Floods11, fine-tune on GEE-extracted India floods

## Context

Two-stage training to get a SegFormer flood-water segmenter tuned for India:

1. **Pre-train** SegFormer (HF `nvidia/mit-b2`) on your Sen1Floods11 hand-labeled subset (USA + Pakistan + Sri-Lanka for training, India held out for val) to learn a general SAR→water prior.
2. **Fine-tune** on India-specific flood events (Assam, Bihar, Kerala, Chennai, …) whose labels we generate on Google Earth Engine following the Sen1Floods11 paper recipe: cloud-masked Sentinel-2 MNDWI + Otsu threshold, minus JRC permanent water, paired with Sentinel-1 VV+VH SAR.

We'll add an HF-Transformers SegFormer variant **alongside** your existing from-scratch and MLA implementations — those stay as backups. A `model.kind` config switch picks between `scratch | mla | hf` so training/eval code is shared.

## Your current tree (relevant pieces)

```
configs/default.yaml
data/
  dataset.py                 # existing Dataset — extend, don't replace
  download_sen1floods11.py
  extract_geo_features.py
  sen1floods11/
    data/<Region>/<Region>_<id>_{S1Hand,S2Hand,LabelHand,
                                 S1OtsuLabelHand,S1OtsuLabelWeak,
                                 S2IndexLabelWeak,S2Weak,JRCWaterHand}.tif
    splits/                  # existing CSVs — reuse if suitable
models/
  segformer_model.py         # existing entrypoint — make it dispatch on model.kind
  segformer/                 # from-scratch impl — KEEP as backup (kind=scratch)
                             # MLA variant — KEEP as backup (kind=mla)
train_segformer.py           # extend to handle pretrain + finetune via config
evaluate.py, predict.py, main.py
checkpoints/
  segformer_standard/        # from-scratch ckpts (existing use)
  segformer_mla/             # MLA ckpts (existing use)
  segformer_hf_pretrain/     # NEW: HF pre-train on Sen1Floods11
  segformer_hf_india_ft/     # NEW: HF fine-tune on India GEE data
```

## Pipeline shape (ASCII)

```
 ┌───────────────── STAGE 1: PRE-TRAIN on Sen1Floods11 ─────────────────┐
 │                                                                      │
 │  data/sen1floods11/data/{USA,Pakistan,Sri-Lanka}/*_S1Hand.tif        │
 │       │  (LabelHand as target, -1 ignored)                           │
 │       ▼                                                              │
 │  data/dataset.py::Sen1FloodsDataset                                  │
 │       │  (VV, VH, VV-VH  →  3×512×512)                               │
 │       ▼                                                              │
 │  models/segformer_model.py::build(kind="hf", n=2)  ← new HF path      │
 │       │  CE(ignore_index=-1), AdamW 6e-5, poly decay, AMP, 60 ep     │
 │       ▼                                                              │
 │  val = India split (held out) → mIoU                                 │
 │       ▼                                                              │
 │  checkpoints/segformer_hf_pretrain/best.pt                           │
 └──────────────────────────────┬───────────────────────────────────────┘
                                │ init_from
 ┌──────────────────────────────┼── STAGE 2: INDIA FINE-TUNE ───────────┐
 │                              │                                      │
 │  configs/events.yaml (Assam'22, Bihar'21, Kerala'18, Chennai'15)    │
 │        │                                                             │
 │        ▼                                                             │
 │  gee/python/export_event.py                                         │
 │   ├─ s1_preprocess   (Refined-Lee, dB, clip)                        │
 │   └─ weak_labels     (S2 MNDWI + Otsu − JRC permanent)              │
 │        │                                                             │
 │        ▼       Drive → local                                         │
 │  data/india_floods/<event>/{S1,Label}/*.tif + splits/{train,val}.csv│
 │        │                                                             │
 │        ▼                                                             │
 │  SAME Sen1FloodsDataset (different root) ─► SAME build_segformer    │
 │        │                                init from hf_pretrain/best.pt│
 │        │                                lr=1e-5, epochs=25           │
 │        ▼                                                             │
 │  checkpoints/segformer_hf_india_ft/best.pt                          │
 └──────────────────────────────────────────────────────────────────────┘
```

Same dataset class, same model builder, same train loop — only config paths + init_from change.

## What to create vs extend vs remove

### Extend
- **`data/dataset.py`** — make `Sen1FloodsDataset(csv_path, transforms, label_key="LabelHand")` work for both stages:
  1. `rasterio.open(s1_path).read()` → `(2,H,W)` float32 dB.
  2. Clip VV `[-23,0]`, VH `[-28,-5]`; min-max normalize per channel.
  3. Channel 3 = normalized `VV-VH` → `(3,H,W)` so pretrained MiT-B2 patch-embed loads unchanged.
  4. Read label `(H,W)` int64; keep `-1` (no-data) for `ignore_index`. Replace NaNs in S1 with 0 and mask to `-1` in label.
  5. Augs: hflip, vflip, 90° rotations only. No color jitter — SAR semantics.
  6. Parameter `label_key` selects which `*_{LabelHand,S1OtsuLabelWeak,…}.tif` to load, so the same class handles Sen1Floods11 hand labels and India GEE weak labels.

- **`models/segformer_model.py`** — turn it into a dispatcher so scratch, mla, and hf coexist:
  ```python
  def build(kind="hf", num_labels=2, pretrained_id="nvidia/mit-b2", **kw):
      if kind == "scratch":
          from models.segformer.encoder import SegFormer  # existing
          return SegFormer(num_labels=num_labels, **kw)
      if kind == "mla":
          from models.segformer.decode_head import SegFormerMLA  # existing
          return SegFormerMLA(num_labels=num_labels, **kw)
      if kind == "hf":
          from transformers import SegformerForSemanticSegmentation
          return SegformerForSemanticSegmentation.from_pretrained(
              pretrained_id, num_labels=num_labels, ignore_mismatched_sizes=True)
      raise ValueError(kind)
  ```
  For `kind="hf"`, upsample logits from H/4×W/4 to label res with `F.interpolate(mode="bilinear", align_corners=False)` before loss/metrics. Leave the scratch/MLA paths untouched — they're the backup.

- **`train_segformer.py`** — single entrypoint reading a YAML config:
  - Calls `models.segformer_model.build(**cfg.model)` so it works for all three kinds.
  - For `kind="hf"`, wraps forward to upsample logits; for scratch/mla, keep existing forward.
  - `nn.CrossEntropyLoss(ignore_index=-1)`, AdamW, poly LR, AMP.
  - Honor `model.init_from` (path to `.pt`) so fine-tune just loads pre-train weights.
  - Save best by val mIoU to `cfg.ckpt_dir`.
  - `--smoke` flag runs 2 batches on CPU for shape sanity.

- **`evaluate.py`** — add per-region / per-event IoU + F1 breakdown; accept a checkpoint and a split CSV.

- **`configs/default.yaml`** — keep as shared defaults; add `model.kind`, `model.pretrained_id`, `model.init_from` fields. Add two new overrides:
  - `configs/pretrain_hf.yaml`: `model.kind: hf`, `model.pretrained_id: nvidia/mit-b2`, data.root `data/sen1floods11/data`, splits `data/sen1floods11/splits/pretrain_{train,val}.csv`, lr 6e-5, epochs 60, `ckpt_dir: checkpoints/segformer_hf_pretrain`.
  - `configs/finetune_hf.yaml`: `model.kind: hf`, data.root `data/india_floods`, splits `data/india_floods/splits/{train,val}.csv`, lr 1e-5, epochs 25, `model.init_from: checkpoints/segformer_hf_pretrain/best.pt`, `ckpt_dir: checkpoints/segformer_hf_india_ft`.
  The existing scratch / MLA configs (whatever your current `default.yaml` drives) keep working unchanged — they just don't set `model.kind` or set it to `scratch`/`mla`.

- **`data/download_sen1floods11.py`** — if it doesn't already, add a tiny helper `build_splits()` that walks `data/sen1floods11/data/<Region>/` and writes:
  - `splits/pretrain_train.csv` = all chips from USA, Pakistan, Sri-Lanka,
  - `splits/pretrain_val.csv`   = all chips from India (held out baseline).
  (Verify on-disk spelling `Sri-Lanka` vs `SriLanka` before hard-coding.)
  If `data/sen1floods11/splits/` already contains canonical CSVs, prefer those and just filter rows by country.

### New (GEE pipeline — none of this exists)

```
configs/events.yaml                  # list of Indian flood events: name, dates, bbox
gee/
  __init__.py
  python/
    s1_preprocess.py                 # S1_GRD filter, Refined-Lee speckle, dB, clip
    weak_labels.py                   # MNDWI + server-side Otsu − JRC GSW permanent
    export_event.py                  # CLI --event assam2022 → exports to Drive
    download_drive.py                # PyDrive2: Drive folder → data/india_floods/<event>/
  js/
    qa_assam2022.js                  # paste into Code Editor for visual QA
```

Details:
- **`s1_preprocess.py`** — `COPERNICUS/S1_GRD`, IW, VV+VH, AOI+date filter. Refined-Lee kernel. Keep dB; clip `[-50,1]`. Optional SRTM terrain flatten behind a flag (default off).
- **`weak_labels.py`** — `COPERNICUS/S2_SR_HARMONIZED` ± window, `s2cloudless` cloud mask (`MSK_CLDPRB < 40`), `MNDWI = (B3 − B11)/(B3 + B11)`, server-side Otsu on MNDWI histogram, subtract `JRC/GSW1_4/GlobalSurfaceWater` `seasonality ≥ 10`. Encode `{1: flood-water, 0: dry, -1: cloud/nodata}`. Reproject to S1 10 m grid.
- **`export_event.py`** — tile AOI to 512×512 @ 10 m, export S1 (2-band) + Label (1-band) GeoTIFFs to `gee_exports/<event>/` on Drive.
- **`download_drive.py`** — PyDrive2 pulls into `data/india_floods/<event>/{S1,Label}/`, then runs a split builder to create event-disjoint splits (suggest **Kerala 2018** as val).
- **`qa_<event>.js`** — per event: S1 VV, MNDWI, Otsu mask, JRC permanent, final label layers.

### Keep as backup (no changes)
- `models/segformer/` (from-scratch: `attention.py`, `block.py`, `decode_head.py`, `encoder.py`, `mix_ffn.py`, `patch_embed.py`) — reachable via `model.kind: scratch`.
- MLA variant — reachable via `model.kind: mla`.
- `checkpoints/segformer_standard/` and `checkpoints/segformer_mla/` — continue to hold scratch/MLA artifacts.
- `.gitignore`: ensure `__pycache__/` is ignored.

## Files — implementation order

1. `pyproject.toml` — add `transformers`, `rasterio`, `earthengine-api`, `geemap`, `pydrive2` (confirm `torch`, `pyyaml`, `tqdm`, `scikit-learn` already present).
2. Make `models/segformer_model.py` a dispatcher on `model.kind` (scratch / mla / hf), preserving existing behavior.
3. Extend `data/dataset.py`: 3-channel stack (VV, VH, VV−VH), `label_key` param, NaN handling.
4. Splits helper in `data/download_sen1floods11.py` (or new `data/splits.py`) producing `pretrain_{train,val}.csv`. Reuse existing `data/sen1floods11/splits/*` if it already separates by country.
5. Refactor `train_segformer.py` to consume YAML configs + `model.init_from` + `--smoke`.
6. `configs/pretrain_hf.yaml` (and keep existing configs untouched for scratch/MLA).
7. Smoke-run HF pre-train, then real run → `checkpoints/segformer_hf_pretrain/best.pt`.
8. `gee/python/{s1_preprocess,weak_labels,export_event,download_drive}.py` + `configs/events.yaml`.
9. Export + download Assam 2022 first as the canary event.
10. `configs/finetune_hf.yaml`, run fine-tune → `checkpoints/segformer_hf_india_ft/best.pt`.
11. Extend `evaluate.py` for per-region / per-event IoU / F1 breakdown.
12. `gee/js/qa_assam2022.js` for visual QA.

## Verification

- **Shape smoke test** — `python train_segformer.py --config configs/pretrain_hf.yaml --smoke` must produce input `(B,3,512,512)`, label `(B,512,512)` with values in `{-1,0,1}`, finite loss, 2 steps on CPU.
- **Pre-train sanity** — val mIoU on held-out India hand-labels ≥ 0.55 within 60 epochs (Sen1Floods11 paper baseline range). Log per-region IoU.
- **GEE QA** — open `gee/js/qa_assam2022.js` in Code Editor, visually verify Otsu mask covers known inundated districts and excludes the Brahmaputra permanent channel (JRC subtraction working).
- **Fine-tune benefit** — `python evaluate.py --ckpt <path> --split data/india_floods/splits/val.csv` for both `segformer_hf_pretrain/best.pt` and `segformer_hf_india_ft/best.pt` on Kerala 2018. Fine-tune should beat pre-train zero-shot by ≥ 5 IoU points on the flood-water class — that's the acceptance bar for the whole pipeline.
- **End-to-end command log** —
  ```
  python -c "from data.download_sen1floods11 import build_splits; build_splits()"
  python train_segformer.py --config configs/pretrain_hf.yaml --smoke
  python train_segformer.py --config configs/pretrain_hf.yaml
  python -m gee.python.export_event   --event assam2022
  python -m gee.python.download_drive --event assam2022
  python train_segformer.py --config configs/finetune_hf.yaml
  python evaluate.py --ckpt checkpoints/segformer_hf_india_ft/best.pt \
                     --split data/india_floods/splits/val.csv
  ```