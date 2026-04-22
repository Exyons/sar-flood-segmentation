# 06 — Explainable AI

Two tools, both read-only on the checkpoints — no retraining, no
changes to the encoder/attention code.

## 1. Attention maps — `scripts/visualize_attention.py`

### Capturing attention without touching the model

`models/segformer/attention.py` doesn't stash its softmax matrices —
it returns the value tensor and drops `attn`. Adding a hook via
`register_forward_hook` doesn't help either because `attn` is a local
variable inside `forward`.

Solution: a context manager that monkey-patches each attention module's
`forward`:

```python
@contextmanager
def capture_attention(model):
    originals = []
    for m in model.modules():
        if isinstance(m, (EfficientSelfAttention, MLASelfAttention)):
            originals.append((m, m.forward))
            m.forward = _patched_forward.__get__(m, type(m))
    try:
        yield
    finally:
        for m, fn in originals:
            m.forward = fn
```

`_patched_forward` is a ~20-line re-implementation of the original
body that stashes `attn` on `m.last_attn` before returning. Training +
eval code remain untouched; the patch exists only for the duration of
the viz forward pass.

### Per-tile figure

For each of N val tiles (picked with `--flood-frac-min 0.05` and
`--seed 0` for reproducibility):

1. Forward under `capture_attention`.
2. For each encoder stage, grab the last block's attention:
   `(B=1, heads, N_q, N_kv)` → mean over heads.
3. Restrict queries to pixels the model predicts as flood
   (`logits.argmax()` upsampled to stage spatial size).
4. Average the per-query softmax weights → one `(N_kv,)` vector of
   how much the flood predictions attend to each key position.
5. Reshape `N_kv` back to `(H_i / sr_ratio, W_i / sr_ratio)`, bilinear
   upsample to 512×512, min-max normalize.
6. Overlay on VV grayscale with alpha.

Layout per tile: rows = `[scratch, mla]`, columns = `[SAR, stage 1,
stage 2, stage 3, stage 4, prediction]`. Saved to
`reports/figures/xai/attn_tile_<tile>.png`.

### Summary figure

`attn_mean_stages.png` — 2×4 grid: mean attention map (across all
sampled tiles) per variant per stage. Individual tiles are noisy; the
mean reveals systematic differences.

### What to expect

- Stage 1 is fine-grained — sr_ratio=8 on B0, so early attention is
  coarse spatially but over many keys. Tends to highlight local
  low-backscatter patches.
- Stage 4 has `sr_ratio=1` and full spatial context but only 16×16
  resolution on a 512 input. Attention is globally diffuse.
- MLA's low-rank bottleneck (`d_c = D/4`) shows as smoother, more
  averaged attention — compression blurs sharp attenders.

## 2. Effective receptive field — `scripts/visualize_receptive_field.py`

Gradient-based ERF (Luo et al. 2016, "Understanding the Effective
Receptive Field in Deep Convolutional Neural Networks"). Measures how
far an output pixel actually sees — not the theoretical bound, the
empirical gradient magnitude.

### Algorithm

Per variant, per sampled tile:

```python
model.train(False)
image = tile.clone().requires_grad_(True)
logits = model(image)                     # (1, 2, H, W)

grad_out = torch.zeros_like(logits)
cy, cx = H // 2, W // 2
grad_out[0, 1, cy, cx] = 1.0              # flood-class logit at center

logits.backward(grad_out)
erf = image.grad[0].abs().mean(dim=0)     # (H, W)
```

Accumulate over M=32 tiles, normalize to `[0, 1]` with `erf /= erf.max()`.

### AMP must be off

`F.softmax` + autocast + backward occasionally returns NaN gradients on
the attention path. The script runs in fp32 throughout — 32 samples ×
2 models is seconds on GPU, ~1 min on CPU.

### Outputs

| file                 | content                                              |
| :------------------- | :--------------------------------------------------- |
| `erf_scratch.png`    | heatmap + 50 / 25 / 10 % contours, radius printed    |
| `erf_mla.png`        | same for MLA                                         |
| `erf_compare.png`    | side-by-side + overlaid contours + ratio text        |
| `erf_radial.png`     | 1-D angular-averaged profile, log-y, both variants   |

`erf_radial.png` is the decisive plot — a single line per variant makes
the ERF-tail difference obvious.

### What to expect

- Both variants should produce a roughly Gaussian ERF centered on
  `(cy, cx)`.
- Scratch attention aggregates from N_kv tokens per stage at full rank.
  MLA routes all K/V through a `d_c = D/4` bottleneck → strictly less
  KV expressivity → typically a **shorter ERF tail**. If the ratio goes
  the other way in measurement, that's itself reportable.
- `erf_radius_10%(scratch) / erf_radius_10%(mla)` is the number to quote
  in the report.

## Integrating with `run.sh`

`ONLY=xai bash run.sh` runs both scripts on whichever checkpoint dir the
config points at. No retraining needed.
