"""
data_fetch_gee.py — Google Earth Engine satellite data fetching.

All heavy computation happens SERVER-SIDE in GEE via reduceRegions.
We never download full-resolution raster tiles to the laptop.
Only small aggregated tables (one row per grid cell) are returned.

PERFORMANCE OPTIMIZATIONS (v2):
  - GEE calls (LST, LULC, NDVI, GHSL) run in parallel via ThreadPoolExecutor
  - FeatureCollection built via batch JSON (no Python loop per feature)
  - Multi-year LST trend fetched in parallel (not sequential)
  - GEE computation scale auto-tuned: larger for big cities to avoid timeout

Datasets used (all free):
  - Landsat 8/9 C02 L2   → Land Surface Temperature (LST)
  - ESA WorldCover v200   → Land use / land cover class fractions
  - Sentinel-2 SR         → NDVI (vegetation index)
  - JRC GHSL              → Population density
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# GEE initialized lazily on first call
_GEE_INITIALIZED = False
_GEE_AVAILABLE = False
_ee = None


def _init_gee(project_id: str) -> bool:
    """
    Initialize GEE with automatic auth mode detection.

    Priority:
    1. Streamlit secrets / env var GEE_SERVICE_ACCOUNT_JSON  → service account (cloud deploy)
    2. Streamlit secrets GEE_SERVICE_ACCOUNT + GEE_PRIVATE_KEY → service account (alt format)
    3. Default credentials (local earthengine authenticate) → user auth (local dev)
    """
    global _GEE_INITIALIZED, _GEE_AVAILABLE, _ee
    if _GEE_INITIALIZED:
        return _GEE_AVAILABLE
    try:
        import ee
        _ee = ee

        # ── Try service account auth first (Streamlit Cloud / production) ──
        service_account_json = None

        # Method 1: Full JSON blob in env/Streamlit secrets
        sa_json_str = os.environ.get("GEE_SERVICE_ACCOUNT_JSON", "")
        if not sa_json_str:
            try:
                import streamlit as st
                sa_json_str = st.secrets.get("GEE_SERVICE_ACCOUNT_JSON", "")
            except Exception:
                pass

        if sa_json_str:
            import json
            import tempfile
            sa_info = json.loads(sa_json_str)
            # Write to temp file (ee.ServiceAccountCredentials needs a file path)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(sa_info, f)
                tmp_path = f.name
            credentials = ee.ServiceAccountCredentials(
                email=sa_info['client_email'],
                key_file=tmp_path,
            )
            ee.Initialize(credentials=credentials, project=project_id)
            os.unlink(tmp_path)
            _GEE_AVAILABLE = True
            logger.info(f"GEE initialized via service account JSON (cloud mode), project: {project_id}")

        else:
            # Method 2: Fallback to local user credentials (earthengine authenticate)
            ee.Initialize(project=project_id)
            _GEE_AVAILABLE = True
            logger.info(f"GEE initialized via local user credentials, project: {project_id}")

    except Exception as e:
        logger.warning(f"GEE initialization failed: {e}. Will use fallback LST estimation.")
        _GEE_AVAILABLE = False

    _GEE_INITIALIZED = True
    return _GEE_AVAILABLE


def _gdf_to_ee_feature_collection(grid_gdf: gpd.GeoDataFrame):
    """
    Convert GeoDataFrame to EE FeatureCollection efficiently.
    Uses batch JSON construction instead of Python-level per-feature loop.
    """
    import ee
    # Build GeoJSON FeatureCollection directly — much faster than iterrows
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {"cell_id": int(row.cell_id)},
            }
            for row in grid_gdf.itertuples()
        ],
    }
    return ee.FeatureCollection(geojson)


def _get_scale_for_grid(grid_gdf: gpd.GeoDataFrame, base_scale: int = 100) -> int:
    """
    Auto-tune GEE reduceRegions scale based on grid size.
    Larger grids (more cells) use a coarser scale to avoid GEE memory limits.
    """
    n = len(grid_gdf)
    if n > 3000:
        return max(base_scale, 200)
    elif n > 1500:
        return max(base_scale, 150)
    return base_scale


def _get_optimal_date_range(months_back: int = 6) -> Tuple[str, str]:
    """
    Return a date range targeting recent cloud-free imagery.
    Uses the same-month window in previous year(s) for seasonal consistency.
    For summer LST analysis, we target Apr–Jun.
    """
    now = datetime.utcnow()
    # Target peak summer (Apr–Jun) for Indian UHI analysis
    year = now.year if now.month >= 4 else now.year - 1
    start = f"{year}-04-01"
    end = f"{year}-06-30"
    return start, end


def fetch_lst(
    grid_gdf: gpd.GeoDataFrame,
    project_id: str,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch median Land Surface Temperature (°C) per grid cell via Landsat 8/9.

    Returns DataFrame with columns: cell_id, lst_celsius, lst_count
    Returns None if GEE unavailable (caller should use fallback).
    """
    if not _init_gee(project_id):
        return None

    if date_start is None or date_end is None:
        date_start, date_end = _get_optimal_date_range()

    try:
        import ee
        ee_fc = _gdf_to_ee_feature_collection(grid_gdf)
        scale = _get_scale_for_grid(grid_gdf, 100)

        def apply_scale_factors(image):
            """Convert Landsat C02 L2 thermal band DN to Kelvin, then Celsius."""
            # ST_B10: scale=0.00341802, offset=149.0 (per Landsat C02 L2 spec)
            thermal = image.select('ST_B10').multiply(0.00341802).add(149.0)
            lst_celsius = thermal.subtract(273.15).rename('lst_celsius')
            # Also get optical bands for NDVI
            optical = image.select(['SR_B4', 'SR_B5']).multiply(0.0000275).add(-0.2)
            return image.addBands(lst_celsius).addBands(optical)

        def mask_landsat_clouds(image):
            qa = image.select('QA_PIXEL')
            cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)   # cloud bit
            shadow_mask = qa.bitwiseAnd(1 << 4).eq(0)  # cloud shadow
            return image.updateMask(cloud_mask).updateMask(shadow_mask)

        # Landsat 8 (OLI/TIRS)
        l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
              .filterDate(date_start, date_end)
              .filterBounds(ee_fc)
              .map(mask_landsat_clouds)
              .map(apply_scale_factors))

        # Landsat 9 (OLI-2/TIRS-2)
        l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
              .filterDate(date_start, date_end)
              .filterBounds(ee_fc)
              .map(mask_landsat_clouds)
              .map(apply_scale_factors))

        combined = l8.merge(l9)
        median_img = combined.select('lst_celsius').median()

        # Server-side reduceRegions — returns one value per cell
        reduced = median_img.reduceRegions(
            collection=ee_fc,
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.count(), sharedInputs=True
            ),
            scale=scale,
            crs='EPSG:4326',
        )

        features = reduced.getInfo()['features']
        rows = []
        for f in features:
            props = f['properties']
            rows.append({
                'cell_id': int(props.get('cell_id', -1)),
                'lst_celsius': props.get('mean', np.nan),
                'lst_pixel_count': props.get('count', 0),
            })

        df = pd.DataFrame(rows)
        # Sanity check: plausible Indian summer LST range
        valid = (df['lst_celsius'] > 15) & (df['lst_celsius'] < 70)
        invalid_count = (~valid & df['lst_celsius'].notna()).sum()
        if invalid_count > len(df) * 0.3:
            logger.warning(f"Many out-of-range LST values ({invalid_count}). Check date range or units.")
        df.loc[~valid, 'lst_celsius'] = np.nan

        logger.info(f"LST fetched: {len(df)} cells, mean={df['lst_celsius'].mean():.1f}°C, "
                    f"date range: {date_start}–{date_end}")
        return df

    except Exception as e:
        logger.error(f"GEE LST fetch failed: {e}")
        return None


def fetch_lulc(
    grid_gdf: gpd.GeoDataFrame,
    project_id: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch ESA WorldCover v200 (10 m) land use class fractions per grid cell.

    ESA WorldCover classes → mapped to UHI-relevant categories:
      10=Tree cover, 20=Shrubland, 30=Grassland, 40=Cropland,
      50=Built-up, 60=Bare/sparse, 70=Snow/ice, 80=Water,
      90=Herbaceous wetland, 95=Mangroves, 100=Moss/lichen

    Returns DataFrame with columns: cell_id, lulc_builtup_frac, lulc_green_frac,
      lulc_water_frac, lulc_bare_frac, lulc_dominant_class
    """
    if not _init_gee(project_id):
        return None
    try:
        import ee
        ee_fc = _gdf_to_ee_feature_collection(grid_gdf)
        scale = _get_scale_for_grid(grid_gdf, 30)

        worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()

        # Built-up (50)
        builtup = worldcover.eq(50)
        # Green = tree(10) + shrub(20) + grass(30) + crop(40)
        green = worldcover.eq(10).Or(worldcover.eq(20)).Or(
            worldcover.eq(30)).Or(worldcover.eq(40))
        # Water (80)
        water = worldcover.eq(80)
        # Bare (60)
        bare = worldcover.eq(60)

        # Combined image with all class fractions
        combined_img = (builtup.rename('builtup')
                        .addBands(green.rename('green'))
                        .addBands(water.rename('water'))
                        .addBands(bare.rename('bare')))

        reduced = combined_img.reduceRegions(
            collection=ee_fc,
            reducer=ee.Reducer.mean(),
            scale=scale,
            crs='EPSG:4326',
        )

        features = reduced.getInfo()['features']
        rows = []
        for f in features:
            props = f['properties']
            rows.append({
                'cell_id': int(props.get('cell_id', -1)),
                'lulc_builtup_frac': props.get('builtup', np.nan),
                'lulc_green_frac': props.get('green', np.nan),
                'lulc_water_frac': props.get('water', np.nan),
                'lulc_bare_frac': props.get('bare', np.nan),
            })

        df = pd.DataFrame(rows)
        # Normalize fractions (should sum to ~1)
        frac_cols = ['lulc_builtup_frac', 'lulc_green_frac', 'lulc_water_frac', 'lulc_bare_frac']
        df[frac_cols] = df[frac_cols].clip(0, 1).fillna(0)
        logger.info(f"LULC fetched: {len(df)} cells, mean built-up={df['lulc_builtup_frac'].mean():.2f}")
        return df

    except Exception as e:
        logger.error(f"GEE LULC fetch failed: {e}")
        return None


def fetch_ndvi(
    grid_gdf: gpd.GeoDataFrame,
    project_id: str,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch median NDVI per grid cell from Sentinel-2 SR.
    Returns DataFrame: cell_id, ndvi_mean, ndvi_std
    """
    if not _init_gee(project_id):
        return None

    if date_start is None or date_end is None:
        date_start, date_end = _get_optimal_date_range()

    try:
        import ee
        ee_fc = _gdf_to_ee_feature_collection(grid_gdf)
        scale = _get_scale_for_grid(grid_gdf, 20)

        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')
            return image.addBands(ndvi)

        def mask_s2_clouds(image):
            qa = image.select('QA60')
            cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)
            cirrus_mask = qa.bitwiseAnd(1 << 11).eq(0)
            return image.updateMask(cloud_mask).updateMask(cirrus_mask)

        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(date_start, date_end)
              .filterBounds(ee_fc)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
              .map(mask_s2_clouds)
              .map(add_ndvi))

        median_ndvi = s2.select('ndvi').median()

        reduced = median_ndvi.reduceRegions(
            collection=ee_fc,
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.stdDev(), sharedInputs=True
            ),
            scale=scale,
            crs='EPSG:4326',
        )

        features = reduced.getInfo()['features']
        rows = []
        for f in features:
            props = f['properties']
            rows.append({
                'cell_id': int(props.get('cell_id', -1)),
                'ndvi_mean': props.get('mean', np.nan),
                'ndvi_std': props.get('stdDev', np.nan),
            })

        df = pd.DataFrame(rows)
        df['ndvi_mean'] = df['ndvi_mean'].clip(-1, 1)
        logger.info(f"NDVI fetched: {len(df)} cells, mean NDVI={df['ndvi_mean'].mean():.3f}")
        return df

    except Exception as e:
        logger.error(f"GEE NDVI fetch failed: {e}")
        return None


def fetch_ghsl(
    grid_gdf: gpd.GeoDataFrame,
    project_id: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch GHSL population density per grid cell.
    Returns DataFrame: cell_id, pop_density (persons/km²)
    """
    if not _init_gee(project_id):
        return None
    try:
        import ee
        ee_fc = _gdf_to_ee_feature_collection(grid_gdf)
        scale = _get_scale_for_grid(grid_gdf, 100)

        # GHSL population 2020 (100m resolution)
        ghsl = ee.ImageCollection('JRC/GHSL/P2023A/GHS_POP').filterDate('2020-01-01', '2021-01-01').first()

        reduced = ghsl.reduceRegions(
            collection=ee_fc,
            reducer=ee.Reducer.mean(),
            scale=scale,
            crs='EPSG:4326',
        )

        features = reduced.getInfo()['features']
        rows = []
        for f in features:
            props = f['properties']
            rows.append({
                'cell_id': int(props.get('cell_id', -1)),
                'pop_density': max(0, props.get('mean', 0) or 0),
            })

        df = pd.DataFrame(rows)
        logger.info(f"GHSL fetched: {len(df)} cells, mean pop={df['pop_density'].mean():.0f}/km²")
        return df

    except Exception as e:
        logger.error(f"GEE GHSL fetch failed: {e}")
        return None


def fetch_all_gee_parallel(
    grid_gdf: gpd.GeoDataFrame,
    project_id: str,
    date_start: str,
    date_end: str,
) -> dict:
    """
    Fetch LST, LULC, NDVI, and GHSL in PARALLEL using ThreadPoolExecutor.

    GEE calls are I/O-bound (waiting for server-side computation) so
    threading gives near-linear speedup: ~4× faster than sequential.

    Returns dict: {'lst': df, 'lulc': df, 'ndvi': df, 'ghsl': df}
    Each value is a DataFrame or None on failure.
    """
    logger.info("Starting parallel GEE data fetch (LST + LULC + NDVI + GHSL simultaneously)...")

    tasks = {
        'lst':  lambda: fetch_lst(grid_gdf, project_id, date_start, date_end),
        'lulc': lambda: fetch_lulc(grid_gdf, project_id),
        'ndvi': lambda: fetch_ndvi(grid_gdf, project_id, date_start, date_end),
        'ghsl': lambda: fetch_ghsl(grid_gdf, project_id),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_key = {executor.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
                logger.info(f"GEE parallel fetch complete: {key}")
            except Exception as e:
                logger.error(f"GEE parallel fetch failed for {key}: {e}")
                results[key] = None

    return results


def fetch_multi_year_lst(
    grid_gdf: gpd.GeoDataFrame,
    project_id: str,
    years: list = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch LST for multiple years (same Apr–Jun window) to show trend.
    Returns long-format DataFrame: cell_id, year, lst_celsius

    OPTIMIZED: fetches all years in parallel instead of sequentially.
    """
    if years is None:
        years = [2015, 2019, 2023]
    if not _init_gee(project_id):
        return None

    logger.info(f"Fetching multi-year LST trend in parallel for years: {years}")

    def _fetch_year(year):
        date_start = f"{year}-04-01"
        date_end = f"{year}-06-30"
        logger.info(f"Fetching LST trend data for {year}")
        df = fetch_lst(grid_gdf, project_id, date_start, date_end)
        if df is not None:
            df['year'] = year
        return df

    all_dfs = []
    with ThreadPoolExecutor(max_workers=len(years)) as executor:
        future_to_year = {executor.submit(_fetch_year, year): year for year in years}
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            try:
                df = future.result()
                if df is not None:
                    logger.info(f"Multi-year LST: {len(df)} cells for {year}")
                    all_dfs.append(df)
            except Exception as e:
                logger.error(f"Multi-year LST fetch failed for {year}: {e}")

    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


def is_gee_available(project_id: str) -> bool:
    """Check if GEE is initialized and ready."""
    return _init_gee(project_id)
