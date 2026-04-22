# 01 — GEE data export

Pipeline to turn a real flood event (time window + AOI) into pairs of
`<stem>_S1.tif` + `<stem>_Label.tif` on Google Drive, which then download
locally.

## 1. Event definition — `configs/events.yaml`

Each event is a dict:

```yaml
assam2022:
  name: "Assam 2022 monsoon floods"
  bboxes:
    - {name: "dhubri_goalpara",    bbox: [89.80, 25.80, 91.00, 26.30]}
    - {name: "guwahati_morigaon",  bbox: [91.30, 26.00, 92.70, 26.55]}
    ...
  s1_start: "2022-06-15"
  s1_end:   "2022-07-20"
  s2_start: "2022-06-10"
  s2_end:   "2022-07-25"
```

Multi-bbox AOIs let us skip Himalayan foothills / plateaus — each sub-AOI
gets its own S1 composite, its own MNDWI-Otsu threshold, and tile names
namespaced `<event>_<sub>_x###_y###`. S2 window slightly brackets S1 so
cloud-free S2 scenes are easier to find.

## 2. S1 preprocessing — `gee/python/s1_preprocess.py`

- `ee.ImageCollection("COPERNICUS/S1_GRD")` filtered by:
  - `instrumentMode == "IW"`
  - polarization includes `VV` and `VH`
  - orbit properties (ASC or DESC, whichever has more scenes in window)
  - AOI intersection
- Border-noise mask: drop pixels with `angle < 30°` or `angle > 45°`.
- dB conversion already applied by GEE GRD product — just clamp.
- Optional Lee / refined Lee filter for speckle (small kernel so thin
  channels survive).
- Output: per-bbox **median composite** over the S1 window (two bands:
  VV, VH, dB).

## 3. Weak-label generation — `gee/python/weak_labels.py`

Goal: produce a binary flood mask without hand annotation. Source is
Sentinel-2 optical, used only for labeling (inference still uses SAR).

- Build an S2 SR composite (cloud-masked via `QA60` + `SCL`) over
  `s2_start..s2_end`.
- Compute `MNDWI = (Green - SWIR1) / (Green + SWIR1)`.
- **Otsu threshold** per sub-AOI (not global) — sun angle, turbidity,
  and seasonality differ across reaches.
- Post-process: morphological open to kill pepper noise.
- **Cloud / no-data encoding** — pixels where the S2 QA or SCL flagged
  cloud/shadow or where `MNDWI` is undefined get written as `-1`, not
  0. Downstream the dataset maps `-1 -> ignore` in loss and metrics.

Label codes: `-1 ignore | 0 dry | 1 flood`.

## 4. Export — `gee/python/export_event.py`

- For each sub-bbox, **river-corridor hotspot filter** picks interior
  tiles that overlap significant surface water (prevents 95 % of tiles
  being pure upland).
- Per-tile **flood-pixel filter**: drop tiles with < 0.5 % flood
  fraction in the weak label (wastes training signal).
- Writes to a Google Drive folder (`gee_exports-<event>-S1/` and
  `...-Label/`), one tile per rectangle, GeoTIFF at 10 m / px.
- `ee.batch.Export.image.toDrive()` is asynchronous — tasks show up in
  the GEE code editor. `export_event.py` logs task IDs so you can
  resume without re-queuing.

## 5. Download — `gee/python/download_drive.py`

- Uses `pydrive2` with an OAuth consent token (first-run only).
- Walks the two export folders, mirrors to
  `data/gee_exports-<event>-S1/` and `data/gee_exports-<event>-Label/`.
- Idempotent: skips tiles already on disk.

Manual alternative: `rclone copy gdrive:gee_exports-assam2022-S1 data/...`

## 6. QA — `gee/js/qa_assam2022.js`, `gee/js/viz_export_event.js`

Paste into the GEE Code Editor to see:

- `qa_assam2022.js` — overlays the weak-label flood mask on a true-color
  S2 composite for each sub-bbox. Lets you sanity-check the Otsu
  threshold before exporting 100 s of tiles.
- `viz_export_event.js` — draws the export tile grid so you can verify
  the hotspot filter didn't miss a major channel.

Local companion: `gee/python/viz_tile.py <tile.tif>` pops a matplotlib
window with VV / VH / mask side-by-side.

## Gotchas learned the hard way

- **Otsu is sensitive to composite noise.** A single bad scene in the S2
  window can shift the threshold by 0.2 and label half the tile as
  flood. Medoid-composite or simple median works better than mean.
- **Ascending and descending S1 orbits have different incidence
  geometries.** Mixing them in one composite adds orthogonal speckle
  patterns. Pick one orbit direction per event.
- **GEE task quotas.** Queueing > 3000 exports starts throttling. The
  per-bbox hotspot + flood filter gets Assam 2022 down to ~15 tiles,
  which is fine for fine-tune but very small for from-scratch training.
