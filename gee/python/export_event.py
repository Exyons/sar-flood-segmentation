"""Export an Indian flood event (S1 + weak label) to Drive for later download.

CLI:
    uv run python -m gee.python.export_event --event assam2022
    uv run python -m gee.python.export_event --event bihar2021 --limit 5

Event config (``configs/events.yaml``) can use either a single ``bbox`` or a
list of ``bboxes``. Multi-bbox events tile each rectangle independently so
we can target river-corridor hotspots without wasting tiles on uplands.

Tiles are filtered by ``export.min_flood_px`` before export — set to 0 to
keep every tile. Drive layout:

    <drive_folder>/<event>/{S1,Label}/<event>_h<H>_x<IX>_y<IY>_{S1,Label}.tif
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
    """Split bbox into approx. ``tile_px * scale_m``-wide lon/lat tiles."""
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0

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


def _collect_tiles(
    bboxes: list[list[float]],
    scale_m: float,
    tile_px: int,
    event_key: str,
) -> list[dict]:
    """Build a tagged tile list across one or more bboxes."""
    out = []
    for h, bb in enumerate(bboxes):
        for lo, la, hi_lo, hi_la, ix, iy in _tile_bbox(tuple(bb), scale_m, tile_px):
            tag = f"{event_key}_h{h:02d}_x{ix:03d}_y{iy:03d}"
            out.append({
                "tag": tag,
                "h": h, "ix": ix, "iy": iy,
                "coords": (lo, la, hi_lo, hi_la),
            })
    return out


def _envelope(bboxes: list[list[float]]) -> list[float]:
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def _chunked(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _filter_by_flood(
    tiles: list[dict],
    label_image: ee.Image,
    scale_m: float,
    min_flood_px: int,
    chunk: int = 200,
) -> list[dict]:
    """Drop tiles with fewer than ``min_flood_px`` flood pixels.

    Uses ``reduceRegions`` for a single server round-trip per chunk instead
    of one ``getInfo`` per tile.
    """
    if min_flood_px <= 0:
        return tiles

    flood_bin = label_image.eq(1).rename("flood")
    kept: list[dict] = []
    total = len(tiles)
    print(f"  filtering {total} tiles by min_flood_px={min_flood_px} ...")
    for i, group in enumerate(_chunked(tiles, chunk)):
        feats = [
            ee.Feature(ee.Geometry.Rectangle(list(t["coords"])),
                       {"tag": t["tag"]})
            for t in group
        ]
        fc = ee.FeatureCollection(feats)
        stats = flood_bin.reduceRegions(
            collection=fc, reducer=ee.Reducer.sum(), scale=scale_m,
        ).getInfo()
        tag_to_sum = {
            f["properties"]["tag"]: (f["properties"].get("sum") or 0)
            for f in stats["features"]
        }
        for t in group:
            if tag_to_sum.get(t["tag"], 0) >= min_flood_px:
                kept.append(t)
        print(f"    chunk {i + 1}/{math.ceil(total / chunk)}: "
              f"{len(kept)} kept so far")
    print(f"  kept {len(kept)} / {total} tiles after flood filter")
    return kept


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
    scale_m       = export_cfg.get("scale_m", 10)
    tile_px       = export_cfg.get("tile_size_px", 512)
    drive_folder  = export_cfg.get("drive_folder", "gee_exports")
    crs           = export_cfg.get("crs", "EPSG:4326")
    min_flood_px  = int(export_cfg.get("min_flood_px", 0))

    # Accept either `bboxes: [[...], ...]` or legacy `bbox: [...]`
    bboxes = event.get("bboxes")
    if bboxes is None:
        bboxes = [event["bbox"]]

    _init_ee(project)

    envelope = _envelope(bboxes)
    s1 = load_s1(envelope, event["s1_start"], event["s1_end"])
    label = generate_weak_label(
        envelope,
        event["s2_start"],
        event["s2_end"],
        s1_reference=s1,
    )

    tiles = _collect_tiles(bboxes, scale_m, tile_px, event_key)
    if limit_tiles:
        tiles = tiles[:limit_tiles]

    per_hotspot = {h: 0 for h in range(len(bboxes))}
    for t in tiles:
        per_hotspot[t["h"]] += 1

    print(f"[{event_key}] {event['name']}")
    print(f"  hotspots  = {len(bboxes)}")
    for h, bb in enumerate(bboxes):
        print(f"    h{h:02d} bbox={bb}  tiles={per_hotspot[h]}")
    print(f"  envelope  = {envelope}")
    print(f"  tile size = {tile_px}x{tile_px} @ {scale_m} m "
          f"(~{tile_px * scale_m} m per side)")
    print(f"  candidate tiles = {len(tiles)}")
    print(f"  drive     = {drive_folder}/{event_key}/")

    # Envelope sanity check — if flood_px is tiny the S2 weak label didn't work
    # (most likely monsoon cloud cover). Use the viz JS to inspect further.
    env_geom = ee.Geometry.Rectangle(envelope)
    env_stats = (
        label.eq(1).rename("flood")
        .addBands(label.neq(-1).rename("valid"))
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=env_geom,
            scale=scale_m * 5,  # coarse sample to keep the roundtrip cheap
            bestEffort=True,
            maxPixels=int(1e10),
        )
        .getInfo()
    )
    f_px = int(env_stats.get("flood") or 0)
    v_px = int(env_stats.get("valid") or 0)
    frac = (100.0 * f_px / v_px) if v_px else 0.0
    print(f"  envelope label: flood={f_px}  valid={v_px}  "
          f"flood/valid={frac:.3f}%  (sampled @ {scale_m * 5} m)")
    if f_px == 0:
        print("  WARNING: 0 flood pixels in envelope — check:")
        print("    * S2 cloud coverage (run gee/js/viz_export_event.js)")
        print("    * S2 window may be fully cloudy — widen s2_start/s2_end")
        print("    * Consider an S1-only weak-label recipe instead")

    tiles = _filter_by_flood(tiles, label, scale_m, min_flood_px)

    if not tiles:
        print("  no tiles pass the flood filter — nothing exported")
        return

    for t in tiles:
        lo, la, hi_lo, hi_la = t["coords"]
        geom = ee.Geometry.Rectangle([lo, la, hi_lo, hi_la])
        tag = t["tag"]

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

    print(f"  kicked off {2 * len(tiles)} tasks "
          f"(watch at earthengine.google.com/tasks)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, help="Event key in configs/events.yaml")
    parser.add_argument("--events-config", default="configs/events.yaml")
    parser.add_argument("--project", default=None, help="GCP project for EE init")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit candidate tiles (for dry runs)")
    args = parser.parse_args()
    export_event(args.event, args.events_config, project=args.project,
                 limit_tiles=args.limit)


if __name__ == "__main__":
    main()
