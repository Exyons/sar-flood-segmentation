# 05 — Evaluation + metrics

`evaluate.py` is the ground truth for numbers in the report. Everything
else (`plot_ablation.py`, XAI scripts) reads through it or reuses its
helpers.

## Metrics class

```python
class Metrics:
    def update(self, pred: np.ndarray, target: np.ndarray):
        # masks ignore_index, accumulates TP/FP/FN/TN
    def result(self) -> dict:
        return {"IoU": ..., "F1": ..., "Precision": ..., "Recall": ...}
```

- **IoU (flood class)**: `tp / (tp + fp + fn)`.
- **F1**: `2·p·r / (p + r)` — rewards balance of precision and recall.
- **Precision**: `tp / (tp + fp)` — false-alarm discipline.
- **Recall**: `tp / (tp + fn)` — miss rate.

All binary, computed on the flood class only. Dry-class metrics are
implied (1 − flood) and carry no extra signal.

## Per-region / per-event breakdown

Every CSV row carries a `tile_name`. `_group_key(name)` takes the first
`_`-delimited token as the region/event:

```
India_25540_S1Hand      -> India
assam2022_dhubri_x0_y3  -> assam2022
```

`eval_ckpt_on_split` returns `{ "overall": {...}, "per_group": {...} }`.
The pretrain val is India-only, so the interesting breakdown is the
cross-corpus evaluation: run the Sen1Floods11 ckpt on the Assam val CSV
to see how much region generalization you get before any fine-tuning.

## Usage

```bash
python evaluate.py \
    --scratch-ckpt checkpoints/segformer_scratch_sen1floods/best.pt \
    --mla-ckpt     checkpoints/segformer_mla_sen1floods/best.pt \
    --val-csv      data/sen1floods11/splits/pretrain_val_hand.csv \
    --data-root    data/sen1floods11/data
```

Console output — the comparison table you screenshot into the report:

```
================================================================================
Model                          IoU       F1     Prec   Recall       Params
--------------------------------------------------------------------------------
scratch_sen1floods          0.7421   0.8232   0.8517   0.7965       3.6M
mla_sen1floods              0.7198   0.8054   0.8311   0.7812       2.8M
================================================================================
```

## Ablation plots — `scripts/plot_ablation.py`

Consumes `history.csv` (train curves) + `best.pt` (inference) + val CSV
(pred grid). Outputs under `reports/figures/`:

| file                   | content                                         |
| :--------------------- | :---------------------------------------------- |
| `curves_scratch.png`   | loss / iou / f1 (train + val), scratch run      |
| `curves_mla.png`       | same, MLA run                                   |
| `curves_compare.png`   | val-only both runs on shared axes (headline)    |
| `bar_metrics.png`      | IoU / F1 / Prec / Recall, side-by-side bars     |
| `bar_per_region.png`   | per-region IoU from `eval_ckpt_on_split`        |
| `pred_grid.png`        | [SAR VV, GT, scratch pred, MLA pred] × N tiles  |
| `param_count.txt`      | param counts + median epoch time per variant    |

The pred grid picks tiles with non-trivial flood fraction (≥ 5 %), seeded
so figures are deterministic between runs.

## What to look for in numbers

- **IoU gap < 2 %** — acceptable; MLA's ~30 % fewer attention params
  come nearly free. Headline talking point.
- **MLA's recall drops more than precision** — consistent with low-rank
  KV undercovering small fragmented flood patches (rice paddies,
  narrow channels). Shows up in `bar_per_region` for fragmented
  regions.
- **Cross-region degradation on `assam2022`** from a Sen1Floods11-only
  ckpt is the motivation for Stage-2 fine-tune.
- **After fine-tune**: Assam IoU should lift substantially on both
  variants; the scratch/MLA gap stays roughly the same — the ablation
  claim is architectural, not corpus-dependent.
