# 02 — Dataset + preprocessing

## Sen1FloodsDataset — `data/dataset.py`

CSV-driven so the same class serves Sen1Floods11 pretrain and GEE
fine-tune with zero code change.

### CSV format

```
<s1_relpath>,<label_relpath>
```

Paths are relative to `data_root`. One sample per line. The resolver
also handles the Sen1Floods11 convention of per-country subfolders
(`India/India_25540_S1Hand.tif`) by falling back to
`data_root/<region>/<file>` if the direct path misses.

### Per-sample read

Each `__getitem__` returns:

```python
{
  "image":     (3, H, W) float32,     # (VV, VH, VV-VH) clipped + min-max
  "label":     (H, W)    int64,       # {-1 ignore, 0 dry, 1 flood}
  "tile_name": str,
}
```

### SAR channel engineering

```
VV_CLIP   = (-23.0,  0.0)   # dB
VH_CLIP   = (-28.0, -5.0)
DIFF_CLIP = (-15.0, 15.0)
```

Clip each band, then min-max normalize to `[0, 1]`. The third channel
is `VV - VH` — cheap polarimetric contrast that lights up open water
(VV and VH both very low) vs bare soil (VV low, VH less low).

`NaN` / `inf` pixels get masked and the corresponding label written as
`-1` so they never contribute to loss or metrics.

### Label handling

- Sen1Floods11 uses `255 = ignore` — remapped to `-1`.
- Anything `> 1` or `< -1` is ambiguous, also mapped to `-1`.
- GEE weak labels already encode cloud/no-data as `-1` (see note 01).

### tile_size — the GEE variable-size fix

Sen1Floods11 tiles are uniformly 512x512. GEE exports are whatever size
the AOI / pixel scale demands, varying between AOIs. Default
`torch.utils.data.default_collate` can't stack tensors with different
H/W, so without a fit step it crashes with:

```
RuntimeError: Trying to resize storage that is not resizable
```

`Sen1FloodsDataset(..., tile_size=N)` center-crops bigger tiles and
**symmetrically zero-pads** smaller ones. SAR fill `0`, label fill `-1`
so padded pixels are ignored downstream.

Configs opt in: `data.tile_size: 512` on Assam configs; Sen1Floods11
configs don't need it.

### Augmentation

Flip H / flip V / 90°×k rotations. Applied jointly to image and label.
Only in `augment=True` datasets (train loader).

## Splits

### Sen1Floods11 — Colab notebook

`sen1floods_colab` produces:
- `data/sen1floods11/splits/pretrain_train_weak.csv`
  (USA + Pakistan + Sri-Lanka, WeaklyLabeled)
- `data/sen1floods11/splits/pretrain_val_hand.csv`
  (India only, HandLabeled)

Geographic holdout = no leakage between train and val countries.
Paths are relative to `data/sen1floods11/data` — so run.sh sets
`EVAL_ROOT=data/sen1floods11/data` for eval/plot/XAI.

### Sen1Floods11 — local reshape (alternative)

`scripts/reshape_sen1floods11_to_gee.py` walks the HandLabeled /
WeaklyLabeled tree and **symlinks** tiles into a flat GEE-style layout
(`sen1floods11_<Region>_<id>_S1.tif`, etc.), then writes
`data/sen1floods11-splits/{train,val}.csv`. Handy when you want the same
code path for Sen1Floods11 and GEE exports.

### GEE — `scripts/build_gee_splits.py`

Pairs `<stem>_S1.tif` and `<stem>_Label.tif` by stem, writes
`data/<event>-splits/{train,val}.csv`. Two modes:

- `--holdout-aoi assam2022_dibrugarh_tinsukia` — every tile whose stem
  starts with this becomes val. **Essential for tiny event sets** —
  random splits leak geography across train/val.
- `--val-frac 0.2 --seed 0` — shuffled random split. Only if your
  per-AOI sample count is large enough that geography isn't a confound.

## Loader knobs

```yaml
training:
  batch_size: 4        # 2 for tiny GEE sets (val must have >=1 batch)
  num_workers: 4       # drops for CPU smoke runs
```

`build_loaders(cfg, pin_memory=device.type == "cuda")` passes
`pin_memory=False` on CPU smoke so you don't eat RAM for nothing.
