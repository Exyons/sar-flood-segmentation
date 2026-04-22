"""Reshape Sen1Floods11 modality tree into GEE-export layout (symlinks).

Target (mirrors ``data/gee_exports-<event>-{S1,Label}/<event>_..._tif``):

    data/sen1floods11-weak-S1/sen1floods11_<Region>_<id>_S1.tif       -> S1Weak
    data/sen1floods11-weak-Label/sen1floods11_<Region>_<id>_Label.tif -> S1OtsuLabelWeak
    data/sen1floods11-val-S1/sen1floods11_India_<id>_S1.tif           -> S1Hand
    data/sen1floods11-val-Label/sen1floods11_India_<id>_Label.tif     -> LabelHand
    data/sen1floods11-splits/{train,val}.csv

Train = all S1Weak + S1OtsuLabelWeak regions **except India**.
Val   = India S1Hand + LabelHand.

Usage:
    uv run python scripts/reshape_sen1floods11_to_gee.py \
        --src data/sen1floods11/data --dst data
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from pathlib import Path

import rasterio


WEAK_S1_REL = "WeaklyLabeled/S1Weak"
WEAK_LABEL_REL = "WeaklyLabeled/S1OtsuLabelWeak"
HAND_S1_REL = "HandLabeled/S1Hand"
HAND_LABEL_REL = "HandLabeled/LabelHand"

WEAK_LABEL_STEM_SUFFIX = "_S1OtsuLabelWeak"
WEAK_S1_STEM_SUFFIX = "_S1Weak"
HAND_LABEL_STEM_SUFFIX = "_LabelHand"
HAND_S1_STEM_SUFFIX = "_S1Hand"


def _iter_pairs(s1_dir: Path, label_dir: Path, s1_suffix: str, label_suffix: str):
    """Yield (region, tile_id, s1_path, label_path) for every valid chip pair."""
    if not s1_dir.exists():
        return
    for s1_path in sorted(s1_dir.rglob("*.tif")):
        stem = s1_path.stem
        if not stem.endswith(s1_suffix):
            continue
        tile_base = stem[: -len(s1_suffix)]
        m = re.match(r"([A-Za-z]+)_(.+)", tile_base)
        if not m:
            continue
        region, tile_id = m.group(1), m.group(2)
        label_path = label_dir / f"{tile_base}{label_suffix}.tif"
        if not label_path.exists():
            continue
        yield region, tile_id, s1_path, label_path


def _symlink(target: Path, link: Path) -> bool:
    """Idempotent symlink. Returns True if created, False if already exists."""
    if link.is_symlink() or link.exists():
        return False
    link.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(target, link.parent)
    link.symlink_to(rel)
    return True


def reshape(src: Path, dst: Path, exclude_regions_train=("India",)) -> dict:
    weak_s1 = dst / "sen1floods11-weak-S1"
    weak_lb = dst / "sen1floods11-weak-Label"
    val_s1 = dst / "sen1floods11-val-S1"
    val_lb = dst / "sen1floods11-val-Label"
    splits = dst / "sen1floods11-splits"
    for d in (weak_s1, weak_lb, val_s1, val_lb, splits):
        d.mkdir(parents=True, exist_ok=True)

    train_rows: list[tuple[str, str]] = []
    val_rows: list[tuple[str, str]] = []
    per_region_train: dict[str, int] = {}
    per_region_val: dict[str, int] = {}

    for region, tile_id, s1, lb in _iter_pairs(
        src / WEAK_S1_REL, src / WEAK_LABEL_REL,
        WEAK_S1_STEM_SUFFIX, WEAK_LABEL_STEM_SUFFIX,
    ):
        if region in exclude_regions_train:
            continue
        link_s1 = weak_s1 / f"sen1floods11_{region}_{tile_id}_S1.tif"
        link_lb = weak_lb / f"sen1floods11_{region}_{tile_id}_Label.tif"
        _symlink(s1.resolve(), link_s1)
        _symlink(lb.resolve(), link_lb)
        train_rows.append((
            str(link_s1.relative_to(dst)),
            str(link_lb.relative_to(dst)),
        ))
        per_region_train[region] = per_region_train.get(region, 0) + 1

    for region, tile_id, s1, lb in _iter_pairs(
        src / HAND_S1_REL, src / HAND_LABEL_REL,
        HAND_S1_STEM_SUFFIX, HAND_LABEL_STEM_SUFFIX,
    ):
        if region != "India":
            continue
        link_s1 = val_s1 / f"sen1floods11_{region}_{tile_id}_S1.tif"
        link_lb = val_lb / f"sen1floods11_{region}_{tile_id}_Label.tif"
        _symlink(s1.resolve(), link_s1)
        _symlink(lb.resolve(), link_lb)
        val_rows.append((
            str(link_s1.relative_to(dst)),
            str(link_lb.relative_to(dst)),
        ))
        per_region_val[region] = per_region_val.get(region, 0) + 1

    train_rows.sort()
    val_rows.sort()

    with open(splits / "train.csv", "w", newline="") as f:
        csv.writer(f).writerows(train_rows)
    with open(splits / "val.csv", "w", newline="") as f:
        csv.writer(f).writerows(val_rows)

    return {
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "per_region_train": per_region_train,
        "per_region_val": per_region_val,
        "train_csv": splits / "train.csv",
        "val_csv": splits / "val.csv",
        "train_rows": train_rows,
        "val_rows": val_rows,
    }


def _spot_check(dst: Path, rows: list[tuple[str, str]], label: str) -> None:
    if not rows:
        print(f"  [{label}] empty — nothing to spot-check")
        return
    s1_rel, lb_rel = random.choice(rows)
    s1 = dst / s1_rel
    lb = dst / lb_rel
    with rasterio.open(s1) as src:
        s1_shape = (src.count, src.height, src.width)
        s1_dtype = src.dtypes[0]
    with rasterio.open(lb) as src:
        lb_shape = (src.count, src.height, src.width)
        lb_dtype = src.dtypes[0]
    print(f"  [{label}] sample {Path(s1_rel).name}")
    print(f"           S1    shape={s1_shape} dtype={s1_dtype}")
    print(f"           Label shape={lb_shape} dtype={lb_dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/sen1floods11/data"))
    ap.add_argument("--dst", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    if not args.src.exists():
        raise SystemExit(f"--src missing: {args.src}")

    print(f"src: {args.src.resolve()}")
    print(f"dst: {args.dst.resolve()}")

    out = reshape(args.src, args.dst)

    print(f"\nTrain tiles: {out['n_train']}")
    for r, n in sorted(out["per_region_train"].items()):
        print(f"  {r:<12} {n}")
    print(f"Val tiles:   {out['n_val']}")
    for r, n in sorted(out["per_region_val"].items()):
        print(f"  {r:<12} {n}")

    print(f"\nCSVs:")
    print(f"  {out['train_csv']}")
    print(f"  {out['val_csv']}")

    print("\nSpot-check:")
    _spot_check(args.dst, out["train_rows"], "train")
    _spot_check(args.dst, out["val_rows"], "val")


if __name__ == "__main__":
    main()
