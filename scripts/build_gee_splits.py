"""Build train / val CSVs from paired GEE export dirs.

Pairs S1 tiles to Label tiles by stripping the suffix (``_S1.tif`` vs
``_Label.tif``) and emits CSV rows the Sen1FloodsDataset understands —
``s1_rel,label_rel`` where both are relative to ``--data-root``.

Two split modes:
  * ``--holdout-aoi <name>``: every tile whose basename starts with
    ``<name>`` goes to val, the rest to train. Best for tiny GEE event
    sets where random splits leak geography across train/val.
  * ``--val-frac 0.2``: shuffled random split with ``--seed``.

Example:
    python scripts/build_gee_splits.py \\
        --s1-dir     data/gee_exports-assam2022-S1 \\
        --label-dir  data/gee_exports-assam2022-Label \\
        --data-root  data \\
        --holdout-aoi assam2022_dibrugarh_tinsukia \\
        --out-dir    data/assam2022-splits
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _pair_tiles(s1_dir: Path, label_dir: Path) -> list[tuple[Path, Path, str]]:
    """Return ``(s1_path, label_path, stem)`` for each S1 tile with a label match."""
    pairs: list[tuple[Path, Path, str]] = []
    missing: list[str] = []
    for s1 in sorted(s1_dir.glob("*_S1.tif")):
        stem = s1.name[: -len("_S1.tif")]
        lbl = label_dir / f"{stem}_Label.tif"
        if not lbl.exists():
            missing.append(stem)
            continue
        pairs.append((s1, lbl, stem))
    if missing:
        print(f"  warning: {len(missing)} S1 tiles have no label "
              f"(skipped): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return pairs


def _relpath(p: Path, root: Path) -> str:
    return str(p.resolve().relative_to(root.resolve()))


def _write_csv(out: Path, rows: list[tuple[str, str]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for s1_rel, lbl_rel in rows:
            f.write(f"{s1_rel},{lbl_rel}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-dir",     type=Path, required=True)
    ap.add_argument("--label-dir",  type=Path, required=True)
    ap.add_argument("--data-root",  type=Path, required=True,
                    help="CSV paths are written relative to this directory.")
    ap.add_argument("--out-dir",    type=Path, required=True,
                    help="Where to drop train.csv and val.csv.")
    ap.add_argument("--holdout-aoi", default=None,
                    help="Tiles whose stem starts with this string become val. "
                         "Overrides --val-frac if set.")
    ap.add_argument("--val-frac",   type=float, default=0.2)
    ap.add_argument("--seed",       type=int,   default=0)
    args = ap.parse_args()

    pairs = _pair_tiles(args.s1_dir, args.label_dir)
    if not pairs:
        raise SystemExit("No paired tiles found.")

    rows: list[tuple[str, str, str]] = [
        (_relpath(s1, args.data_root), _relpath(lbl, args.data_root), stem)
        for s1, lbl, stem in pairs
    ]

    if args.holdout_aoi:
        train = [(s, l) for s, l, stem in rows if not stem.startswith(args.holdout_aoi)]
        val   = [(s, l) for s, l, stem in rows if     stem.startswith(args.holdout_aoi)]
        if not val:
            raise SystemExit(
                f"--holdout-aoi {args.holdout_aoi!r} matched zero tiles. "
                f"Known prefixes: {sorted({r[2].rsplit('_x', 1)[0] for r in rows})}"
            )
    else:
        rng = random.Random(args.seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * args.val_frac)))
        val   = [(s, l) for s, l, _ in shuffled[:n_val]]
        train = [(s, l) for s, l, _ in shuffled[n_val:]]

    _write_csv(args.out_dir / "train.csv", train)
    _write_csv(args.out_dir / "val.csv",   val)
    print(f"Paired:  {len(rows)}  tiles")
    print(f"Train:   {len(train)}  -> {args.out_dir / 'train.csv'}")
    print(f"Val:     {len(val)}    -> {args.out_dir / 'val.csv'}")


if __name__ == "__main__":
    main()
