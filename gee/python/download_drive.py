"""Pull GEE-exported flood-event GeoTIFFs from Drive into ``data/india_floods``.

After running ``export_event.py`` and waiting for the EE tasks to finish,
this pulls ``<drive_folder>/<event>/{S1,Label}/*.tif`` down to
``data/india_floods/<event>/{S1,Label}/`` and writes a split CSV.

CLI:
    uv run python -m gee.python.download_drive --event assam2022
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


def _auth() -> GoogleDrive:
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)


def _list_folder(drive: GoogleDrive, folder_id: str) -> list:
    q = f"'{folder_id}' in parents and trashed=false"
    return drive.ListFile({"q": q}).GetList()


def _find_folder_id(drive: GoogleDrive, name: str, parent_id: str = "root") -> str | None:
    q = (
        f"'{parent_id}' in parents and trashed=false "
        f"and mimeType='application/vnd.google-apps.folder' "
        f"and title='{name}'"
    )
    matches = drive.ListFile({"q": q}).GetList()
    if not matches:
        return None
    return matches[0]["id"]


def _download_folder(drive: GoogleDrive, folder_id: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _list_folder(drive, folder_id)
    n = 0
    for f in files:
        if f.get("mimeType") == "application/vnd.google-apps.folder":
            continue
        dest = out_dir / f["title"]
        if dest.exists():
            n += 1
            continue
        print(f"  -> {dest}")
        f.GetContentFile(str(dest))
        n += 1
    return n


def download_event(
    event_key: str,
    events_cfg_path: str | Path = "configs/events.yaml",
    local_root: str | Path = "data/india_floods",
    val_fraction: float = 0.15,
) -> None:
    with open(events_cfg_path) as f:
        cfg = yaml.safe_load(f)
    drive_folder = cfg.get("export", {}).get("drive_folder", "gee_exports")

    drive = _auth()

    root_id = _find_folder_id(drive, drive_folder)
    if root_id is None:
        raise FileNotFoundError(f"Drive folder '{drive_folder}' not found")

    event_id = _find_folder_id(drive, event_key, parent_id=root_id)
    if event_id is None:
        raise FileNotFoundError(f"Drive folder '{drive_folder}/{event_key}' not found")

    s1_id = _find_folder_id(drive, "S1", parent_id=event_id)
    label_id = _find_folder_id(drive, "Label", parent_id=event_id)
    if s1_id is None or label_id is None:
        raise FileNotFoundError("Expected S1/ and Label/ subfolders under the event")

    out = Path(local_root) / event_key
    s1_dir = out / "S1"
    label_dir = out / "Label"

    n_s1 = _download_folder(drive, s1_id, s1_dir)
    n_lb = _download_folder(drive, label_id, label_dir)
    print(f"Downloaded {n_s1} S1 tiles and {n_lb} label tiles to {out}")

    _write_splits(out, event_key, val_fraction=val_fraction)


def _write_splits(event_dir: Path, event_key: str, val_fraction: float = 0.15) -> None:
    """Write ``train.csv`` / ``val.csv`` under ``data/india_floods/splits/``.

    The splits are global across all downloaded events, so this appends rather
    than overwrites. Caller should delete the files to start fresh.
    """
    splits_dir = event_dir.parent / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    import random
    random.seed(42)

    s1_files = sorted((event_dir / "S1").glob("*.tif"))
    rows = []
    for s1 in s1_files:
        # Label filename mirrors S1 filename except suffix
        stem = s1.stem.replace("_S1", "_Label")
        label = event_dir / "Label" / f"{stem}.tif"
        if not label.exists():
            continue
        s1_rel = f"{event_key}/S1/{s1.name}"
        label_rel = f"{event_key}/Label/{label.name}"
        rows.append((s1_rel, label_rel))

    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    # Append (so multi-event runs accumulate)
    with open(train_csv, "a") as f:
        for s1_rel, label_rel in train_rows:
            f.write(f"{s1_rel},{label_rel}\n")
    with open(val_csv, "a") as f:
        for s1_rel, label_rel in val_rows:
            f.write(f"{s1_rel},{label_rel}\n")

    print(f"  appended {len(train_rows)} train rows -> {train_csv}")
    print(f"  appended {len(val_rows)} val rows   -> {val_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True)
    parser.add_argument("--events-config", default="configs/events.yaml")
    parser.add_argument("--local-root", default="data/india_floods")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    download_event(args.event, args.events_config, args.local_root, args.val_fraction)


if __name__ == "__main__":
    main()
