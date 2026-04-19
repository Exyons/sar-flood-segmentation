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


def main():
    parser = argparse.ArgumentParser(description="Download Sen1Floods11 dataset")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["paths"]["sen1floods11_dir"]

    if args.verify_only:
        verify_dataset(data_dir)
    else:
        download_sen1floods11(data_dir)
        verify_dataset(data_dir)


if __name__ == "__main__":
    main()
