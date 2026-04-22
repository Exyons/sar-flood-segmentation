# 00 — Project overview

## Goal

Pixel-wise SAR flood segmentation across India, with two angles:

1. **Scratch vs MLA ablation** — SegFormer-B0 trained from random init on
   Sen1Floods11, compared against the same backbone with **low-rank KV
   attention** (MLA). Apples-to-apples schedule; only the attention
   module differs.
2. **Transfer to real flood events** — fine-tune both variants on
   **GEE-exported** Sentinel-1 tiles from the Assam 2022 monsoon floods
   (weak labels from MNDWI-Otsu S2 composites).

Deliverables: IoU / F1 metrics per region, per-epoch curves, prediction
grids, attention maps, and effective receptive field (ERF) plots.

## End-to-end flow

```
[GEE]  events.yaml
         |
         v
  s1_preprocess.py       (border-noise mask, dB stack, Lee filter)
         |
         v
  weak_labels.py         (S2 MNDWI -> Otsu -> cloud/no-data=-1)
         |
         v
  export_event.py        (river-corridor hotspots, per-tile flood
                          filter, push S1+Label .tifs to Drive)
         |
         v
  download_drive.py      (rclone/pydrive pulls tiles to data/)
         |
         v
[Local] data/gee_exports-<event>-{S1,Label}/
         |
         v
  build_gee_splits.py    (pair S1<->Label by stem, write train/val CSVs)
         |
         v
  configs/<variant>_<event>.yaml  (model kind, cuda:N pin, tile_size)
         |
         v
  train_segformer.py     (yaml, AMP, poly LR, history.csv)
         |           \
         |            \ PARALLEL=1 -> cuda:0 scratch + cuda:1 MLA
         |           /
         v          /
  checkpoints/<run>/best.pt
         |
         +--> evaluate.py                 (per-region IoU/F1)
         +--> scripts/plot_ablation.py    (curves, bars, pred grid)
         +--> scripts/visualize_attention.py
         +--> scripts/visualize_receptive_field.py
```

## Repo map

```
configs/                YAML experiment configs
data/dataset.py         Sen1FloodsDataset (CSV-driven, 3ch SAR stack)
models/segformer/       Scratch encoder + attention variants
models/segformer_model.py  kind dispatcher (scratch | mla | hf)
gee/python/             GEE preprocess / weak-label / export / download
gee/js/                 QA + export-plan visualizers
scripts/                reshape, build_gee_splits, plot_ablation,
                        visualize_attention, visualize_receptive_field
train_segformer.py      training entrypoint (yaml-driven)
evaluate.py             per-region metrics
train_bundle/           self-contained scp-to-server copy of the above
learning/               these notes
```

## Key design choices (why the code looks the way it does)

- **CSV-driven dataset** — swap corpora by editing `train_csv` / `val_csv`,
  never by touching the dataset class.
- **Checkpoint embeds its config** — `evaluate.py`, XAI scripts, and plots
  rebuild the model from `ckpt["config"]["model"]` so there's one source
  of truth.
- **train_bundle/ mirror** — server gets everything via `scp`; no import
  gymnastics relative to a Python package root.
- **Per-variant device pinning in config** — `training.device: cuda:0` on
  scratch and `cuda:1` on MLA lets `PARALLEL=1` launch both concurrently
  without juggling `CUDA_VISIBLE_DEVICES`.

See each subsequent note for deeper treatment.
