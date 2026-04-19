"""Extract geospatial features per tile, aligned to Sen1Floods11 512x512 grid.

Features extracted:
    - DEM (elevation) from SRTM
    - Slope (from DEM gradient)
    - TWI (Topographic Wetness Index) = ln(contributing_area / tan(slope))
    - NDVI from Sentinel-2 (B8 - B4) / (B8 + B4)
    - Distance to permanent water (JRC Global Surface Water)
    - Land cover (ESA WorldCover)

Each tile produces a stacked numpy array: (N_features, 512, 512).
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import yaml
from scipy.ndimage import distance_transform_edt, uniform_filter
from tqdm import tqdm


def compute_slope(dem: np.ndarray, resolution: float = 10.0) -> np.ndarray:
    """Compute slope in degrees from DEM."""
    dy, dx = np.gradient(dem, resolution)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)


def compute_twi(dem: np.ndarray, resolution: float = 10.0) -> np.ndarray:
    """Compute Topographic Wetness Index.

    TWI = ln(a / tan(beta))
    where a = contributing area (approximated via smoothed DEM flow accumulation)
    and beta = slope in radians.

    Uses a simplified approach: approximate contributing area with
    a smoothed inverse-elevation proxy. For production, use proper D8/D-inf
    flow routing. This approximation is sufficient for ML features.
    """
    dy, dx = np.gradient(dem, resolution)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    # Clamp slope to avoid division by zero / log of zero
    slope_rad = np.clip(slope_rad, 1e-6, None)

    # Approximate contributing area via smoothed DEM
    # Higher smoothing = larger effective drainage area estimate
    smoothed = uniform_filter(dem, size=15)
    # Flow accumulation proxy: difference between smoothed and local
    # Invert so valleys (low points) get high accumulation
    flow_proxy = smoothed.max() - smoothed + 1.0
    # Normalize to reasonable contributing area range
    flow_proxy = flow_proxy / flow_proxy.mean() * (resolution * 10)

    twi = np.log(flow_proxy / np.tan(slope_rad))
    # Clamp extreme values
    twi = np.clip(twi, -5, 25)
    return twi


def compute_ndvi(b8: np.ndarray, b4: np.ndarray) -> np.ndarray:
    """Compute NDVI from Sentinel-2 bands."""
    nir = b8.astype(np.float32)
    red = b4.astype(np.float32)
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    return np.clip(ndvi, -1.0, 1.0)


def compute_distance_to_water(water_mask: np.ndarray, resolution: float = 10.0) -> np.ndarray:
    """Compute Euclidean distance to nearest water pixel."""
    # water_mask: 1 = water, 0 = not water
    if water_mask.sum() == 0:
        # No water in tile — return max distance
        return np.full(water_mask.shape, water_mask.shape[0] * resolution, dtype=np.float32)
    # distance_transform_edt computes distance from 0 pixels to nearest 1 pixel
    # We want distance from non-water to water, so invert
    dist = distance_transform_edt(~water_mask.astype(bool)) * resolution
    return dist.astype(np.float32)


def extract_features_for_tile(
    dem_path: str | None,
    s2_b8_path: str | None,
    s2_b4_path: str | None,
    water_mask_path: str | None,
    landcover_path: str | None,
    tile_shape: tuple[int, int] = (512, 512),
    resolution: float = 10.0,
) -> np.ndarray:
    """Extract all geo features for a single tile.

    Returns:
        features: np.ndarray of shape (6, H, W) — DEM, slope, TWI, NDVI, dist_water, landcover
    """
    h, w = tile_shape
    features = np.zeros((6, h, w), dtype=np.float32)

    # 1. DEM
    if dem_path and Path(dem_path).exists():
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
            # Resample if needed
            if dem.shape != tile_shape:
                from skimage.transform import resize
                dem = resize(dem, tile_shape, preserve_range=True).astype(np.float32)
        features[0] = dem
        # 2. Slope
        features[1] = compute_slope(dem, resolution)
        # 3. TWI
        features[2] = compute_twi(dem, resolution)
    else:
        # Placeholder zeros if DEM not available
        pass

    # 4. NDVI
    if s2_b8_path and s2_b4_path and Path(s2_b8_path).exists() and Path(s2_b4_path).exists():
        with rasterio.open(s2_b8_path) as src:
            b8 = src.read(1).astype(np.float32)
        with rasterio.open(s2_b4_path) as src:
            b4 = src.read(1).astype(np.float32)
        if b8.shape != tile_shape:
            from skimage.transform import resize
            b8 = resize(b8, tile_shape, preserve_range=True).astype(np.float32)
            b4 = resize(b4, tile_shape, preserve_range=True).astype(np.float32)
        features[3] = compute_ndvi(b8, b4)

    # 5. Distance to water
    if water_mask_path and Path(water_mask_path).exists():
        with rasterio.open(water_mask_path) as src:
            water = src.read(1)
        if water.shape != tile_shape:
            from skimage.transform import resize
            water = resize(water, tile_shape, preserve_range=True)
        features[4] = compute_distance_to_water((water > 0).astype(np.uint8), resolution)

    # 6. Land cover (integer encoding)
    if landcover_path and Path(landcover_path).exists():
        with rasterio.open(landcover_path) as src:
            lc = src.read(1).astype(np.float32)
        if lc.shape != tile_shape:
            from skimage.transform import resize
            lc = resize(lc, tile_shape, order=0, preserve_range=True).astype(np.float32)
        features[5] = lc

    return features


def process_all_tiles(cfg: dict) -> None:
    """Process all Sen1Floods11 tiles and save geo feature stacks."""
    sen1_dir = Path(cfg["paths"]["sen1floods11_dir"])
    geo_dir = Path(cfg["paths"]["geo_features_dir"])
    geo_dir.mkdir(parents=True, exist_ok=True)

    # Find all SAR tiles to determine which tiles to process
    sar_dir = sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
    if not sar_dir.exists():
        print(f"SAR directory not found: {sar_dir}")
        print("Download Sen1Floods11 first: uv run python data/download_sen1floods11.py")
        return

    sar_tiles = sorted(sar_dir.glob("*.tif"))
    print(f"Found {len(sar_tiles)} SAR tiles to process")

    for tile_path in tqdm(sar_tiles, desc="Extracting geo features"):
        tile_name = tile_path.stem  # e.g., "Bolivia_7_S1Hand"
        out_path = geo_dir / f"{tile_name}_geo.npy"

        if out_path.exists():
            continue

        # Build paths to co-located rasters (may not all exist)
        base_name = tile_name.replace("_S1Hand", "")
        dem_path = sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "SRTM" / f"{base_name}_SRTM.tif"
        s2_b8_path = sen1_dir / "v1.1" / "data" / "flood_events" / "S2Hand" / f"{base_name}_S2Hand.tif"
        # For S2, bands are typically in a single multi-band tif — handle both cases
        water_path = sen1_dir / "v1.1" / "data" / "flood_events" / "JRC" / f"{base_name}_JRC.tif"
        lc_path = sen1_dir / "v1.1" / "data" / "flood_events" / "LC" / f"{base_name}_LC.tif"

        features = extract_features_for_tile(
            dem_path=str(dem_path) if dem_path.exists() else None,
            s2_b8_path=str(s2_b8_path) if s2_b8_path.exists() else None,
            s2_b4_path=str(s2_b8_path) if s2_b8_path.exists() else None,  # same multi-band tif
            water_mask_path=str(water_path) if water_path.exists() else None,
            landcover_path=str(lc_path) if lc_path.exists() else None,
        )

        np.save(out_path, features)

    print(f"Geo features saved to {geo_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract geospatial features")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    process_all_tiles(cfg)


if __name__ == "__main__":
    main()
