"""Export an Indian flood event (S1 + weak label) to Drive for later download.

CLI:
    uv run python -m gee.python.export_event --event assam2022
    uv run python -m gee.python.export_event --event bihar2021 --tiles-only

The event definition comes from ``configs/events.yaml``. Exports land in
``<drive_folder>/<event>/{S1,Label}/`` as 512x512 GeoTIFFs at the configured
scale. Tasks are kicked off in ``Earth Engine`` — monitor at
https://code.earthengine.google.com/tasks.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import ee
import yaml

from .s1_preprocess import load_s1
from .weak_labels import generate_weak_label


def _init_ee(project: str | None = None) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()


def _tile_bbox(
    bbox: tuple[float, float, float, float],
    scale_m: float,
    tile_px: int,
) -> list[tuple[float, float, float, float, int, int]]:
    """Split bbox into approx. ``tile_px * scale_m``-wide lon/lat tiles.

    Returns rows of (lon_min, lat_min, lon_max, lat_max, ix, iy).
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0

    # Rough meters-per-degree at the AOI's central latitude
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(mid_lat))

    tile_size_m = tile_px * scale_m
    step_lat = tile_size_m / meters_per_deg_lat
    step_lon = tile_size_m / meters_per_deg_lon

    tiles = []
    n_lat = int(math.ceil((max_lat - min_lat) / step_lat))
    n_lon = int(math.ceil((max_lon - min_lon) / step_lon))
    for iy in range(n_lat):
        for ix in range(n_lon):
            lo = min_lon + ix * step_lon
            la = min_lat + iy * step_lat
            hi_lo = min(lo + step_lon, max_lon)
            hi_la = min(la + step_lat, max_lat)
            tiles.append((lo, la, hi_lo, hi_la, ix, iy))
    return tiles


def export_event(
    event_key: str,
    events_cfg_path: str | Path = "configs/events.yaml",
    project: str | None = None,
    limit_tiles: int | None = None,
) -> None:
    events_cfg_path = Path(events_cfg_path)
    with open(events_cfg_path) as f:
        cfg = yaml.safe_load(f)

    if event_key not in cfg["events"]:
        raise KeyError(f"Event {event_key!r} not in {events_cfg_path}")

    event = cfg["events"][event_key]
    export_cfg = cfg.get("export", {})
    scale_m = export_cfg.get("scale_m", 10)
    tile_px = export_cfg.get("tile_size_px", 512)
    drive_folder = export_cfg.get("drive_folder", "gee_exports")
    crs = export_cfg.get("crs", "EPSG:4326")

    _init_ee(project)

    bbox = event["bbox"]
    s1 = load_s1(bbox, event["s1_start"], event["s1_end"])
    label = generate_weak_label(
        bbox,
        event["s2_start"],
        event["s2_end"],
        s1_reference=s1,
    )

    tiles = _tile_bbox(tuple(bbox), scale_m=scale_m, tile_px=tile_px)
    if limit_tiles:
        tiles = tiles[:limit_tiles]

    print(f"[{event_key}] {event['name']}")
    print(f"  bbox      = {bbox}")
    print(f"  tiles     = {len(tiles)} ({tile_px}x{tile_px} @ {scale_m} m)")
    print(f"  drive     = {drive_folder}/{event_key}/")

    for lon_lo, lat_lo, lon_hi, lat_hi, ix, iy in tiles:
        geom = ee.Geometry.Rectangle([lon_lo, lat_lo, lon_hi, lat_hi])
        tag = f"{event_key}_x{ix:03d}_y{iy:03d}"

        s1_task = ee.batch.Export.image.toDrive(
            image=s1.toFloat(),
            description=f"{tag}_S1",
            folder=f"{drive_folder}/{event_key}/S1",
            fileNamePrefix=f"{tag}_S1",
            region=geom,
            scale=scale_m,
            crs=crs,
            maxPixels=int(1e10),
            fileFormat="GeoTIFF",
        )
        label_task = ee.batch.Export.image.toDrive(
            image=label.toInt16(),
            description=f"{tag}_Label",
            folder=f"{drive_folder}/{event_key}/Label",
            fileNamePrefix=f"{tag}_Label",
            region=geom,
            scale=scale_m,
            crs=crs,
            maxPixels=int(1e10),
            fileFormat="GeoTIFF",
        )

        s1_task.start()
        label_task.start()

    print(f"  kicked off {2 * len(tiles)} tasks (watch at earthengine.google.com/tasks)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, help="Event key in configs/events.yaml")
    parser.add_argument("--events-config", default="configs/events.yaml")
    parser.add_argument("--project", default=None, help="GCP project for EE init")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tiles (for dry runs)")
    args = parser.parse_args()
    export_event(args.event, args.events_config, project=args.project, limit_tiles=args.limit)


if __name__ == "__main__":
    main()
