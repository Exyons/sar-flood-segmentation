"""Download and verify the Sen1Floods11 dataset.

Sen1Floods11: 4,831 chips of 512x512 @ 10m resolution.
Source: https://github.com/cloudtostreet/Sen1Floods11

Directory structure after download:
    data/sen1floods11/
    ├── v1.1/
    │   ├── catalog/              # CSV split files
    │   ├── data/
    │   │   ├── flood_events/     # per-event directories
    │   │   │   ├── HandLabeled/  # hand-labeled SAR + masks
    │   │   │   └── S2Hand/       # Sentinel-2 optical
    │   │   └── ...
    │   └── splits/
    └── ...
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def download_sen1floods11(output_dir: str) -> None:
    """Clone Sen1Floods11 repo and download data via GCS."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    catalog_dir = output_path / "v1.1" / "catalog"
    if catalog_dir.exists():
        print(f"Sen1Floods11 catalog already exists at {catalog_dir}, skipping clone.")
    else:
        print("Cloning Sen1Floods11 repository for catalog/split files...")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/cloudtostreet/Sen1Floods11.git",
                str(output_path / "repo"),
            ],
            check=True,
        )

    # Download actual GeoTIFF data from GCS (public bucket)
    data_dir = output_path / "v1.1" / "data" / "flood_events"
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"Data already exists at {data_dir}, skipping download.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    bucket = "gs://sen1floods11"

    print("Downloading Sen1Floods11 data from Google Cloud Storage...")
    print("This requires gsutil. Install via: pip install gsutil")
    print(f"Target: {data_dir}")

    # Download hand-labeled data (most important subset)
    for sub in ["HandLabeled", "S2Hand", "QCHand", "LabelHand"]:
        target = data_dir / sub
        target.mkdir(exist_ok=True)
        src = f"{bucket}/v1.1/data/flood_events/{sub}/"
        print(f"  Downloading {sub}...")
        try:
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", src, str(target)],
                check=True,
            )
        except FileNotFoundError:
            print(
                f"  gsutil not found. Download manually from:\n"
                f"    {src}\n"
                f"  Place files in: {target}"
            )
            break

    print("Sen1Floods11 download complete.")


def verify_dataset(data_dir: str) -> None:
    """Count and report downloaded tiles."""
    data_path = Path(data_dir) / "v1.1" / "data" / "flood_events"
    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        return

    tif_files = list(data_path.rglob("*.tif"))
    print(f"Found {len(tif_files)} GeoTIFF files in {data_path}")

    # Count by subdirectory
    subdirs = {}
    for f in tif_files:
        rel = f.relative_to(data_path)
        key = rel.parts[0] if rel.parts else "root"
        subdirs[key] = subdirs.get(key, 0) + 1

    for k, v in sorted(subdirs.items()):
        print(f"  {k}: {v} files")


DEFAULT_TRAIN_REGIONS = ("USA", "Pakistan", "Sri-Lanka")
DEFAULT_VAL_REGIONS = ("India",)


def build_splits(
    data_dir: str | Path = "data/sen1floods11",
    train_regions: tuple[str, ...] = DEFAULT_TRAIN_REGIONS,
    val_regions: tuple[str, ...] = DEFAULT_VAL_REGIONS,
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Walk ``data/sen1floods11/data/<Region>/*_S1Hand.tif`` and write pretrain CSVs.

    CSV rows are ``<rel_s1_path>,<rel_label_path>`` relative to
    ``data/sen1floods11/data`` — the same root passed to Sen1FloodsDataset.

    Returns a dict mapping split name → output CSV path.
    """
    data_root = Path(data_dir)
    tiles_root = data_root / "data"
    if not tiles_root.exists():
        raise FileNotFoundError(f"Sen1Floods11 tiles root missing: {tiles_root}")

    out_dir = Path(out_dir) if out_dir else data_root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    def gather(regions: tuple[str, ...]) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for region in regions:
            region_dir = tiles_root / region
            if not region_dir.exists():
                print(f"  WARN: region dir missing, skipping: {region_dir}")
                continue
            for s1 in sorted(region_dir.glob("*_S1Hand.tif")):
                label = s1.with_name(s1.name.replace("_S1Hand.tif", "_LabelHand.tif"))
                if not label.exists():
                    continue
                s1_rel = f"{region}/{s1.name}"
                label_rel = f"{region}/{label.name}"
                rows.append((s1_rel, label_rel))
        return rows

    train_rows = gather(train_regions)
    val_rows = gather(val_regions)

    outputs: dict[str, Path] = {}
    for name, rows in (("pretrain_train", train_rows), ("pretrain_val", val_rows)):
        out_path = out_dir / f"{name}.csv"
        with open(out_path, "w") as f:
            for s1_rel, label_rel in rows:
                f.write(f"{s1_rel},{label_rel}\n")
        outputs[name] = out_path
        print(f"  wrote {out_path} — {len(rows)} tiles")

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Download Sen1Floods11 dataset")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--build-splits",
        action="store_true",
        help="Walk on-disk regions and emit pretrain_train / pretrain_val CSVs",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["paths"]["sen1floods11_dir"]

    if args.build_splits:
        build_splits(data_dir)
        return
    if args.verify_only:
        verify_dataset(data_dir)
    else:
        download_sen1floods11(data_dir)
        verify_dataset(data_dir)


if __name__ == "__main__":
    main()
