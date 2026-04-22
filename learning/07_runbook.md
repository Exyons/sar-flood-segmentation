# 07 — Runbook

End-to-end commands for reproducing the ablation on a fresh 2-GPU box.
Every step is a stage in `train_bundle/run.sh` — override with `ONLY=...`.

## 0. Environment

```bash
# local — develop with uv
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"

# server — conda env with the same pins
conda activate <env>
pip install -r requirements.txt   # generated from pyproject.toml
```

Dependencies that matter: `torch>=2.1`, `transformers`, `rasterio`,
`earthengine-api`, `geemap`, `pydrive2`, `rich`.

## 1. Sync bundle to server

`train_bundle/` is a self-contained copy — no import gymnastics, no
repo root assumptions.

```bash
rsync -av --delete train_bundle/ user@server:~/flood/
# Or scp -r train_bundle/ user@server:~/flood/
```

## 2. Stage 1 — Sen1Floods11 pretrain

### Data layout

Expected on the server:

```
data/sen1floods11/data/
  HandLabeled/S1Hand/{India,Bolivia,...}/*.tif
  HandLabeled/LabelHand/.../*.tif
  WeaklyLabeled/S1Weak/.../*.tif
  WeaklyLabeled/S2IndexLabelWeak/.../*.tif
data/sen1floods11/splits/
  pretrain_train_weak.csv         # USA + Pakistan + Sri-Lanka
  pretrain_val_hand.csv           # India only (geographic holdout)
```

Produced by the `sen1floods_colab` notebook; copy both splits + the
`data` tree once.

### Parallel training

```bash
cd ~/flood
PARALLEL=1 bash run.sh        # scratch -> cuda:0, MLA -> cuda:1
```

What happens:

- `ONLY` unset → all stages run.
- `reshape` is skipped (uses Colab splits directly; not the reshape
  script).
- `train` smokes both configs on CPU, then fires two concurrent
  training jobs, each pinned to its configured GPU. Logs interleave
  with `[scratch] ` / `[mla]    ` prefixes.
- `eval` runs per-region IoU/F1 against the India val.
- `plot` writes `reports/figures/curves_*.png`, `bar_*.png`,
  `pred_grid.png`, `param_count.txt`.

### Process management

- Launch backgrounded — `PARALLEL=1 bash run.sh > /tmp/flood.log 2>&1 &`
  (the rich ANSI survives — `Console(force_terminal=True)`).
- Terminate in-flight:
  ```bash
  pkill -f "train_segformer.py --config"
  ```
- Resume from a checkpoint: `init_from: <prev>/best.pt` in the config.

## 3. Stage 2 — Assam 2022 fine-tune

### GEE export (one-time, ~30 min of wall time on GEE side)

```bash
cd gee/python
python s1_preprocess.py --event assam2022      # optional: caches composites
python weak_labels.py --event assam2022        # MNDWI + Otsu masks
python export_event.py --event assam2022       # queues Drive exports
# wait for GEE tasks to finish in the code editor
python download_drive.py --event assam2022     # mirror to data/
```

QA before exporting: paste `gee/js/qa_assam2022.js` into the GEE code
editor and eyeball the per-sub-AOI masks.

### Build splits

```bash
python scripts/build_gee_splits.py \
    --s1-dir    data/gee_exports-assam2022-S1 \
    --label-dir data/gee_exports-assam2022-Label \
    --out-dir   data/assam2022-splits \
    --data-root data \
    --holdout-aoi assam2022_dibrugarh_tinsukia
```

### Fine-tune (parallel)

`configs/{scratch,mla}_assam2022.yaml` already carry:

- `init_from: checkpoints/segformer_{scratch,mla}_sen1floods/best.pt`
- `tile_size: 512` (GEE tiles have variable H/W)
- `device: cuda:{0,1}`
- `lr: 3.0e-5` (one order below the pretrain LR)
- `epochs: 40`, `batch_size: 2`

```bash
PARALLEL=1 \
  SCRATCH_CFG=configs/scratch_assam2022.yaml \
  MLA_CFG=configs/mla_assam2022.yaml \
  SCRATCH_DIR=checkpoints/segformer_scratch_assam2022 \
  MLA_DIR=checkpoints/segformer_mla_assam2022 \
  VAL_CSV=data/assam2022-splits/val.csv \
  EVAL_ROOT=data \
  ONLY=train \
  bash run.sh
```

## 4. Evaluation + plots

```bash
ONLY=eval bash run.sh     # prints comparison table
ONLY=plot bash run.sh     # writes reports/figures/*.png
```

Swap the `VAL_CSV` / `EVAL_ROOT` env vars to run the same Sen1Floods11
checkpoints on the Assam val CSV (cross-region zero-shot number) or the
fine-tuned checkpoints on the Sen1Floods11 val (catastrophic forgetting
check).

## 5. XAI

```bash
ONLY=xai bash run.sh      # writes reports/figures/xai/*.png
```

Produces:

- `attn_tile_<name>.png` × N (per-tile attention overlays per stage).
- `attn_mean_stages.png` (2×4 mean-over-tiles grid).
- `erf_scratch.png`, `erf_mla.png`, `erf_compare.png`, `erf_radial.png`.

All figures are deterministic given `--seed 0`.

## 6. Pull results back

```bash
rsync -av user@server:~/flood/reports/ reports/
rsync -av user@server:~/flood/checkpoints/ checkpoints/
```

Checkpoints are ~15 MB each for B0 — safe to keep.

## Common gotchas

- **DataLoader stack error** on GEE runs — missing `tile_size: 512` in
  the config.
- **GPU index out of range** — `cuda:1` on a single-GPU box. Set
  `PARALLEL=0` and override `--device cuda:0` on the MLA run, or edit
  the MLA config.
- **Rich colors stripped** in a pager — pipe to `less -R` (raw control
  chars) not `less`.
- **History file growing** across resumed runs — `history.csv` is
  append-only. Delete it if you restart from epoch 1, or plots will
  show a sawtooth.
- **S1 orbit mixing** — `events.yaml` should have one orbit direction
  per event. Otherwise the composite has orthogonal speckle patterns.
