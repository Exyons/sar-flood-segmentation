#!/usr/bin/env bash
# End-to-end pipeline: reshape -> smoke -> train scratch + MLA -> eval -> plots.
#
# Run from the train_bundle/ dir on the server. Activate your conda env first:
#     conda activate <your_env>
#     bash run.sh
#
# Override defaults via env vars:
#     SRC=/path/to/sen1floods11/data DST=data bash run.sh
#     SKIP_SMOKE=1 bash run.sh
#     ONLY=reshape bash run.sh
#     ONLY=train   bash run.sh
#     ONLY=plot    bash run.sh

set -euo pipefail

SRC="${SRC:-data/sen1floods11/data}"
DST="${DST:-data}"
# EVAL_ROOT = what CSV paths in VAL_CSV are relative to.
# - If using Colab splits (HandLabeled/..., WeaklyLabeled/...):  data/sen1floods11/data
# - If using reshape script output (sen1floods11-val-S1/...):    data
EVAL_ROOT="${EVAL_ROOT:-data/sen1floods11/data}"
SCRATCH_CFG="${SCRATCH_CFG:-configs/scratch_sen1floods.yaml}"
MLA_CFG="${MLA_CFG:-configs/mla_sen1floods.yaml}"
SCRATCH_DIR="${SCRATCH_DIR:-checkpoints/segformer_scratch_sen1floods}"
MLA_DIR="${MLA_DIR:-checkpoints/segformer_mla_sen1floods}"
VAL_CSV="${VAL_CSV:-data/sen1floods11/splits/val_hand.csv}"
OUT_DIR="${OUT_DIR:-reports/figures}"
ONLY="${ONLY:-}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
PARALLEL="${PARALLEL:-0}"        # 1 = launch scratch + MLA concurrently (2-GPU boxes)

PY=python

banner() { printf '\n============================================================\n%s\n============================================================\n' "$1"; }
run_step() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

if run_step reshape; then
  banner "1. Reshape Sen1Floods11 -> GEE layout  (src=$SRC dst=$DST)"
  $PY scripts/reshape_sen1floods11_to_gee.py --src "$SRC" --dst "$DST"

  banner "2. Sanity"
  wc -l "$DST/sen1floods11-splits/train.csv" "$DST/sen1floods11-splits/val.csv"
  ls "$DST/sen1floods11-weak-S1" | head -3
  ls "$DST/sen1floods11-val-S1"  | head -3
fi

if run_step train; then
  if [[ "$SKIP_SMOKE" != "1" ]]; then
    banner "3. Smoke (scratch, 2+2 steps CPU)"
    $PY train_segformer.py --config "$SCRATCH_CFG" --smoke
    banner "3. Smoke (MLA, 2+2 steps CPU)"
    $PY train_segformer.py --config "$MLA_CFG" --smoke
  fi

  if [[ "$PARALLEL" == "1" ]]; then
    banner "4+5. Train scratch + MLA in parallel (device pinning from config)"
    stdbuf -oL -eL $PY train_segformer.py --config "$SCRATCH_CFG" \
      2>&1 | sed -u 's/^/[scratch] /' &
    SCRATCH_PID=$!
    stdbuf -oL -eL $PY train_segformer.py --config "$MLA_CFG" \
      2>&1 | sed -u 's/^/[mla]     /' &
    MLA_PID=$!
    FAIL=0
    wait $SCRATCH_PID || { echo "scratch training failed"; FAIL=1; }
    wait $MLA_PID     || { echo "MLA training failed";     FAIL=1; }
    [[ $FAIL -eq 0 ]] || exit 1
  else
    banner "4. Train scratch"
    $PY train_segformer.py --config "$SCRATCH_CFG"

    banner "5. Train MLA"
    $PY train_segformer.py --config "$MLA_CFG"
  fi
fi

if run_step eval; then
  banner "6. Per-region eval (scratch)  [data-root=$EVAL_ROOT]"
  $PY evaluate.py --ckpt "$SCRATCH_DIR/best.pt" --split "$VAL_CSV" \
      --data-root "$EVAL_ROOT" --label-key none

  banner "7. Per-region eval (MLA)"
  $PY evaluate.py --ckpt "$MLA_DIR/best.pt" --split "$VAL_CSV" \
      --data-root "$EVAL_ROOT" --label-key none
fi

if run_step plot; then
  banner "8. Ablation plots -> $OUT_DIR  [data-root=$EVAL_ROOT]"
  $PY scripts/plot_ablation.py \
      --scratch-dir "$SCRATCH_DIR" \
      --mla-dir     "$MLA_DIR" \
      --val-csv     "$VAL_CSV" \
      --data-root   "$EVAL_ROOT" \
      --out-dir     "$OUT_DIR"
fi

if run_step xai; then
  banner "9. Attention maps -> $OUT_DIR/xai  [data-root=$EVAL_ROOT]"
  $PY scripts/visualize_attention.py \
      --scratch-ckpt "$SCRATCH_DIR/best.pt" \
      --mla-ckpt     "$MLA_DIR/best.pt" \
      --val-csv      "$VAL_CSV" \
      --data-root    "$EVAL_ROOT" \
      --out-dir      "$OUT_DIR/xai"

  banner "10. Effective receptive field -> $OUT_DIR/xai"
  $PY scripts/visualize_receptive_field.py \
      --scratch-ckpt "$SCRATCH_DIR/best.pt" \
      --mla-ckpt     "$MLA_DIR/best.pt" \
      --val-csv      "$VAL_CSV" \
      --data-root    "$EVAL_ROOT" \
      --out-dir      "$OUT_DIR/xai"
fi

banner "All done."
