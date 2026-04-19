"""Weak flood-water label generation on Earth Engine.

Recipe (mirrors Sen1Floods11 paper section 3.2):

    1. Sentinel-2 SR (harmonized) composite over AOI + date window.
    2. Cloud mask via MSK_CLDPRB < 40 (s2cloudless equivalent in GEE).
    3. MNDWI = (B3 - B11) / (B3 + B11).
    4. Server-side Otsu threshold on MNDWI histogram -> binary water mask.
    5. Subtract JRC GSW seasonality >= 10 (permanent / near-permanent water).
    6. Encode:  1 = flood water, 0 = dry, -1 = cloud / no data.
    7. Reproject to the Sentinel-1 10 m grid.
"""

from __future__ import annotations

import ee


# ---------------------------------------------------------------------------
# Cloud masking
# ---------------------------------------------------------------------------


def _s2_cloud_mask(img: ee.Image, cldprb_max: int = 40) -> ee.Image:
    """Mask S2 pixels where MSK_CLDPRB >= cldprb_max."""
    cldprb = img.select("MSK_CLDPRB")
    return img.updateMask(cldprb.lt(cldprb_max))


# ---------------------------------------------------------------------------
# Server-side Otsu on an ee.Image histogram
# ---------------------------------------------------------------------------


def _otsu(histogram) -> ee.Number:
    """Compute an Otsu threshold from an ee.Dictionary histogram."""
    counts = ee.Array(ee.Dictionary(histogram).get("histogram"))
    means = ee.Array(ee.Dictionary(histogram).get("bucketMeans"))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sum_ = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean_total = sum_.divide(total)

    indices = ee.List.sequence(1, size)

    def _bss(i):
        i = ee.Number(i)
        a_counts = counts.slice(0, 0, i)
        a_count = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
        a_means = means.slice(0, 0, i)
        a_mean = a_means.multiply(a_counts).reduce(ee.Reducer.sum(), [0]).get([0]) \
            .divide(a_count.max(1e-9))
        b_count = ee.Number(total).subtract(a_count)
        b_mean = ee.Number(sum_).subtract(a_mean.multiply(a_count)).divide(b_count.max(1e-9))
        return a_count.multiply(a_mean.subtract(mean_total).pow(2)).add(
            b_count.multiply(b_mean.subtract(mean_total).pow(2))
        )

    bss = indices.map(_bss)
    max_bss = ee.List(bss).reduce(ee.Reducer.max())
    # Threshold = bucket mean at the index that maximizes between-class variance
    idx = ee.List(bss).indexOf(max_bss)
    return ee.Number(means.get([idx]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_weak_label(
    bbox: list[float] | tuple[float, float, float, float],
    s2_start: str,
    s2_end: str,
    s1_reference: ee.Image | None = None,
    cloud_prob_max: int = 40,
    permanent_water_seasonality: int = 10,
) -> ee.Image:
    """Return a single-band weak flood-water label image.

    Values:
        1  -> flood water (MNDWI > Otsu, and not permanent water)
        0  -> dry
       -1  -> cloud / no data
    """
    aoi = ee.Geometry.Rectangle(list(bbox))

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(s2_start, s2_end)
        .map(lambda img: _s2_cloud_mask(img, cloud_prob_max))
    )

    s2_composite = s2.median().clip(aoi)

    mndwi = s2_composite.normalizedDifference(["B3", "B11"]).rename("MNDWI")

    # Otsu threshold over the AOI histogram
    histogram = mndwi.reduceRegion(
        reducer=ee.Reducer.histogram(maxBuckets=256, minBucketWidth=0.01),
        geometry=aoi,
        scale=30,
        bestEffort=True,
        maxPixels=1e10,
    ).get("MNDWI")

    threshold = _otsu(histogram)
    water = mndwi.gt(ee.Image.constant(threshold)).rename("water")

    # JRC permanent / near-permanent water subtraction
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("seasonality")
    permanent = jrc.gte(permanent_water_seasonality).unmask(0)

    flood = water.And(permanent.Not()).rename("flood")

    # Encode: -1 where S2 had no valid obs (cloud / gap), else 0/1
    valid = mndwi.mask()
    label = flood.toInt16().where(valid.Not(), -1)

    if s1_reference is not None:
        proj = s1_reference.projection()
        label = label.reproject(crs=proj.crs(), scale=proj.nominalScale())

    return label.rename("label").clip(aoi)
