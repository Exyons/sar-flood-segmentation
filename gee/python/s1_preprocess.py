"""Sentinel-1 GRD preprocessing on Google Earth Engine.

Pipeline:
    COPERNICUS/S1_GRD
    -> filter IW / VV+VH / AOI / date / orbit
    -> Refined-Lee speckle filter (server-side)
    -> clip dB to [-50, 1]
    -> two-band image (VV, VH)

S1 GRD values on GEE are already in decibels, so no linear→dB conversion
is needed.
"""

from __future__ import annotations

import ee


# ---------------------------------------------------------------------------
# Refined-Lee speckle filter (Guido Lemoine, public GEE community snippet)
# ---------------------------------------------------------------------------


def _refined_lee(image: ee.Image) -> ee.Image:
    """Server-side Refined-Lee filter on a single-band dB image."""
    # Convert dB -> power for variance math; output restored to dB.
    img_power = ee.Image(10.0).pow(image.divide(10.0))

    # 3x3 mean and variance
    kernel_3 = ee.Kernel.square(1)
    mean_3 = img_power.reduceNeighborhood(ee.Reducer.mean(), kernel_3)
    var_3 = img_power.reduceNeighborhood(ee.Reducer.variance(), kernel_3)

    # 7x7 means of each 3x3 mean / variance for directional gradient
    sample_weights = ee.List.repeat(ee.List.repeat(1, 7), 7)
    sample_kernel = ee.Kernel.fixed(7, 7, sample_weights, 3, 3, False)

    sample_mean = mean_3.neighborhoodToBands(sample_kernel)
    sample_var = var_3.neighborhoodToBands(sample_kernel)

    # Gradient magnitude across 4 directions
    grad = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
    grad = grad.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
    grad = grad.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
    grad = grad.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

    max_grad = grad.reduce(ee.Reducer.max())
    grad_mask = grad.eq(max_grad)

    # Pick variance along the dominant edge direction
    directional_var = sample_var.select(1).multiply(grad_mask.select(0))
    directional_var = directional_var.add(sample_var.select(6).multiply(grad_mask.select(1)))
    directional_var = directional_var.add(sample_var.select(3).multiply(grad_mask.select(2)))
    directional_var = directional_var.add(sample_var.select(0).multiply(grad_mask.select(3)))

    # Lee filter
    sigma_v_sq = 0.4472  # equivalent-number-of-looks 5 for GRD
    variance = directional_var
    signal_mean = mean_3
    signal_var = variance.subtract(signal_mean.pow(2).multiply(sigma_v_sq)) \
                         .divide(1.0 + sigma_v_sq)
    b = signal_var.divide(variance.max(1e-9))
    filtered_power = signal_mean.add(b.multiply(img_power.subtract(signal_mean)))

    filtered_db = ee.Image(10.0).multiply(filtered_power.max(1e-9).log10())
    return filtered_db.rename(image.bandNames())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_s1(
    bbox: list[float] | tuple[float, float, float, float],
    start: str,
    end: str,
    orbit: str = "DESCENDING",
    apply_refined_lee: bool = True,
) -> ee.Image:
    """Return a single VV+VH S1 GRD composite over the AOI / time window.

    Args:
        bbox:  [min_lon, min_lat, max_lon, max_lat]
        start: inclusive date "YYYY-MM-DD"
        end:   exclusive date "YYYY-MM-DD"
        orbit: "ASCENDING" or "DESCENDING" — keep one for consistent geometry
        apply_refined_lee: speckle filter on/off
    """
    aoi = ee.Geometry.Rectangle(list(bbox))
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
        .select(["VV", "VH"])
    )

    def _preprocess(img):
        if apply_refined_lee:
            vv = _refined_lee(img.select("VV"))
            vh = _refined_lee(img.select("VH"))
            img = ee.Image.cat([vv.rename("VV"), vh.rename("VH")])
        return img.clamp(-50, 1).copyProperties(img, ["system:time_start"])

    col = col.map(_preprocess)
    composite = col.median().clip(aoi)
    return composite.rename(["VV", "VH"])
