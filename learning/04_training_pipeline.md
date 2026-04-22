# 04 — Training pipeline

Entrypoint: `train_segformer.py`. YAML-driven, no CLI flags for
hyperparameters — every run is reproducible from its config.

## Config shape

```yaml
model:
  kind: scratch | mla | hf
  num_labels: 2
  in_channels: 3
  rank_divisor: 4                # mla only
  init_from: path/to/best.pt     # optional warm-start

data:
  root:      data/sen1floods11/data
  train_csv: data/.../train.csv
  val_csv:   data/.../val.csv
  label_key: null                # Sen1Floods11 hand tiles need "LabelHand"
  tile_size: 512                 # GEE runs; omit for Sen1Floods11

training:
  device: cuda:0                 # cuda:1 on MLA config -> PARALLEL=1 works
  epochs: 60
  lr: 6.0e-5
  weight_decay: 0.01
  batch_size: 4
  num_workers: 4
  class_weights: [1.0, 5.0]      # flood is the rare class
  poly_power: 0.9
  warmup_epochs: 5
  use_amp: true
  seed: 42
  log_every_steps: 50            # rich console prints
  val_log_every_steps: 0         # 0 = only end-of-epoch row

ckpt_dir: checkpoints/segformer_<kind>_<corpus>
```

## Loss + metrics

- `CrossEntropyLoss(weight=class_weights, ignore_index=-1)` — `-1`
  covers Sen1Floods11 `255` remap and GEE cloud/no-data.
- `compute_iou` and `compute_f1` both mask `ignore_index` before
  counting. IoU is mean over classes present in the valid pixels; F1 is
  binary (flood = positive).

## LR schedule

Polynomial decay with linear warmup:

```
epoch < warmup_epochs:   lr *= (epoch+1) / warmup_epochs
else:                    lr *= (1 - progress) ** poly_power
```

Checked per epoch, not per step — simpler and plenty for a 40–60 epoch
run.

## AMP

`torch.amp.autocast(device_type=device.type)` + `GradScaler`. Disabled
automatically on CPU smoke runs. Also disabled inside the XAI scripts —
`F.softmax` gradients are flaky under autocast.

## Device pinning + parallel training

`training.device: cuda:0` in `scratch_*.yaml` and `cuda:1` in `mla_*.yaml`
means a single run uses one GPU. The bundle launcher exploits that:

```bash
# run.sh
if [[ "$PARALLEL" == "1" ]]; then
  stdbuf -oL -eL $PY train_segformer.py --config "$SCRATCH_CFG" 2>&1 \
      | sed -u 's/^/[scratch] /' &
  stdbuf -oL -eL $PY train_segformer.py --config "$MLA_CFG" 2>&1 \
      | sed -u 's/^/[mla]     /' &
  wait
fi
```

- `stdbuf -oL -eL` line-buffers each child so lines interleave cleanly
  instead of flushing in multi-KB chunks.
- `sed -u 's/^/[variant] /'` prefixes each line so interleaved logs are
  readable.
- No redirect to log files — rich ANSI colors survive the sed pipe
  (`Console(force_terminal=True)`).
- `CUDA_VISIBLE_DEVICES` juggling is unnecessary; each process just
  calls `torch.cuda.set_device(device)` and the other GPU is free.

`PARALLEL=0` falls back to sequential: scratch run finishes, then MLA.

## Logging — rich, interval-based

Replaced tqdm (which spammed multi-line progress into non-TTY logs).
Now: one line per `log_every_steps` steps.

```
  ep 7 Train  250/844 | loss 0.2814 | iou 0.612 | f1 0.743 | 88 ms/it
```

`iou` / `f1` are running means since start of epoch. `ms/it` is a
rolling average over the last window. End-of-epoch row prints
epoch-level train + val metrics together plus the LR.

## Checkpoint contents

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "val_iou": ..., "val_f1": ...,
    "config": cfg,   # full YAML dict -> downstream rebuilds the model
}, ckpt_dir / "best.pt")
```

`best.pt` is overwritten whenever `val_iou` improves. `final.pt` is the
last epoch regardless. `history.csv` appends every epoch's row —
`plot_ablation.py` reads it straight.

## Warm-start behavior

`model.init_from: ...best.pt` calls `load_state_dict(strict=False)`:

- Missing keys (e.g., new `in_channels` conv) keep their random init.
- Unexpected keys (e.g., old head if `num_labels` changed) are ignored.

Useful for fine-tune: Assam 2022 configs load the Sen1Floods11 pretrain
and continue at `lr=3e-5`.

## Smoke test

```bash
python train_segformer.py --config configs/scratch_sen1floods.yaml --smoke
```

Forces CPU, 2 train + 2 val steps, asserts finite loss. Catches shape
mismatches and missing files in ~30 seconds without burning GPU time.
