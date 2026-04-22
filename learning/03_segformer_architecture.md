# 03 — SegFormer architecture

Two-variant apples-to-apples ablation. Same encoder topology, same
decoder, same training schedule — only the attention module swaps.

## Variants

```python
models.segformer_model.build(kind=..., **cfg)
  kind="scratch"  -> MixTransformerEncoder(use_mla=False)
  kind="mla"      -> MixTransformerEncoder(use_mla=True, rank_divisor=4)
  kind="hf"       -> HF nvidia/mit-b* backbone (reference baseline)
```

All three return the same `(B, num_classes, H, W)` logit tensor, so the
training loop / eval / XAI scripts don't branch on `kind`.

## Encoder — MixTransformer (`models/segformer/encoder.py`)

4 hierarchical stages producing feature maps at `1/4, 1/8, 1/16, 1/32`.
Per stage:

```
x -> OverlapPatchEmbed(patch_size, stride)     # conv, stride downsample
  -> [TransformerBlock] * num_blocks[i]        # attn + MixFFN
  -> LayerNorm
  -> reshape (B, D, H_i, W_i)
```

B0 config (what we use):

| stage | embed_dim | heads | sr_ratio | blocks |
| :---: | :-------: | :---: | :------: | :----: |
|   1   |     32    |   1   |     8    |    2   |
|   2   |     64    |   2   |     4    |    2   |
|   3   |    160    |   5   |     2    |    2   |
|   4   |    256    |   8   |     1    |    2   |

`sr_ratio` is the spatial-reduction stride on K/V inside attention —
lets early stages attend over a manageable number of tokens. Only stage
4 is un-reduced.

## TransformerBlock

```python
y = x + Attn(LN(x), H, W)
y = y + MixFFN(LN(y), H, W)
```

`Attn` is either `EfficientSelfAttention` or `MLASelfAttention` based
on `use_mla`. `MixFFN` is the standard SegFormer `Linear → 3×3
DWConv → GELU → Linear` (positional info baked into the conv; no
learned PE).

## Attention variants (`models/segformer/attention.py`)

### EfficientSelfAttention (scratch)

Standard ViT attention with spatial reduction on K, V:

```
q = W_q x                                 # (B, heads, N, d_h)
x_sr = sr_conv(reshape(x))  if sr_ratio>1 # (B, N_sr, D)
k = W_k x_sr ; v = W_v x_sr               # (B, heads, N_sr, d_h)
attn = softmax(q kᵀ / sqrt(d_h))          # (B, heads, N, N_sr)
out  = attn v                             # (B, heads, N, d_h)
return W_o flatten(out)
```

Params/layer: `4 · D · D` (Q, K, V, out).

### MLASelfAttention (MLA variant)

Low-rank KV compression — `d_c = D / rank_divisor` (default 4):

```
q = W_q x
compressed = W_down x_sr               # (B, N_sr, d_c)
k = W_k_up compressed                  # (B, heads, N_sr, d_h)
v = W_v_up compressed                  # (B, heads, N_sr, d_h)
attn = softmax(q kᵀ / sqrt(d_h))
out  = W_o flatten(attn v)
```

One shared `W_down: D → d_c` feeds both `W_k_up` and `W_v_up`. At
`rank_divisor=4`:

```
Params/layer = D·D + D·(D/4) + 2·(D/4)·D + D·D
             = D² (1 + 1/4 + 1/2 + 1) = 2.75 · D²
```

vs. `4 · D²` for the scratch baseline — ~31 % fewer attention params.

The `forward` signature is identical, so everything downstream
(encoder, decoder, training, monkey-patched attention capture) doesn't
care which variant is instantiated.

## Decoder — MLPDecodeHead (`models/segformer/decode_head.py`)

All-MLP fusion head, no convolutions beyond the MixFFN DWConvs:

```
for i in 0..3:
    x_i = LinearProject_i(feat_i)          # D_i -> decoder_dim
    x_i = bilinear_upsample(x_i, 1/4 res)  # stage_i res -> H/4
concat      = torch.cat(x_0..3, dim=C)     # (B, 4·decoder_dim, H/4, W/4)
fused       = ReLU(Linear(concat))         # -> decoder_dim
logits      = Linear(fused)                # -> num_classes
return bilinear_upsample(logits, H, W)
```

Lightweight by design — SegFormer's thesis is that a strong hierarchical
encoder doesn't need a heavy decoder.

## Top-level model (`models/segformer_model.py`)

Dispatcher. Exposes `param_summary()` for console log:

```
SegFormer-B0 (kind=scratch, use_mla=False)
  encoder: 3.2M   decoder: 0.4M   total: 3.6M trainable
```

Checkpoints store `ckpt["config"]` so `evaluate._build_from_ckpt` can
reconstruct the exact architecture without the user re-specifying it.
