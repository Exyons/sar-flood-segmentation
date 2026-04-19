"""Download instructions and processing for SEN12-FLOOD dataset.

SEN12-FLOOD: 336 time series of co-registered SAR + optical images.
Source: https://ieee-dataport.org/open-access/sen12-flood-sar-and-multispectral-dataset-flood-detection

NOTE: IEEE DataPort requires manual authentication. This script provides
instructions and processes the downloaded data.

Expected directory structure after manual download:
    data/sen12flood/
    ├── train/         # 269 sequences
    │   ├── seq_0001/
    │   │   ├── S1/    # Sentinel-1 VV+VH
    │   │   └── S2/    # Sentinel-2 optical
    │   └── ...
    └── test/          # 68 sequences
        └── ...
"""

import argparse
from pathlib import Path

import yaml


def print_download_instructions(output_dir: str) -> None:
    """Print manual download instructions for SEN12-FLOOD."""
    print("=" * 70)
    print("SEN12-FLOOD Dataset — Manual Download Required")
    print("=" * 70)
    print()
    print("1. Go to: https://ieee-dataport.org/open-access/"
          "sen12-flood-sar-and-multispectral-dataset-flood-detection")
    print("2. Create a free IEEE DataPort account (or log in)")
    print("3. Download the dataset archive")
    print(f"4. Extract to: {output_dir}")
    print()
    print("Expected structure:")
    print(f"  {output_dir}/")
    print("  ├── train/   (269 sequences)")
    print("  └── test/    (68 sequences)")
    print()
    print("After extracting, run this script with --verify to check.")
    print("=" * 70)


def verify_dataset(data_dir: str) -> None:
    """Verify SEN12-FLOOD directory structure."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Directory not found: {data_path}")
        print("Run without --verify for download instructions.")
        return

    for split in ["train", "test"]:
        split_dir = data_path / split
        if not split_dir.exists():
            print(f"  Missing: {split_dir}")
            continue
        seqs = [d for d in split_dir.iterdir() if d.is_dir()]
        tif_count = len(list(split_dir.rglob("*.tif")))
        print(f"  {split}: {len(seqs)} sequences, {tif_count} GeoTIFF files")


def main():
    parser = argparse.ArgumentParser(description="SEN12-FLOOD dataset helper")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["paths"]["sen12flood_dir"]

    if args.verify:
        verify_dataset(data_dir)
    else:
        print_download_instructions(data_dir)


if __name__ == "__main__":
    main()
