"""
feature_engineering.py — Assemble and derive physics-informed features.

Merges all data sources into a single per-cell feature table.
Key physics-informed features derived here:
  - net_radiation_proxy: surface energy balance approximation
  - albedo_proxy: from land-cover class fractions
  - ndbi: normalized difference built-up index (proxy)
  - heat_capacity_proxy: thermal mass estimate from land cover

Memory management: intermediate DataFrames dropped after merge.
"""

import gc
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ESA WorldCover class → approximate albedo values (literature values)
LULC_ALBEDO = {
    'builtup': 0.15,      # Mixed concrete/asphalt/roofs
    'green': 0.20,        # Vegetation (varies 0.1–0.25)
    'water': 0.07,        # Open water
    'bare': 0.30,         # Bare soil/rock
    'default': 0.18,      # Mixed/unknown
}

# Emissivity by surface type (for net radiation calculation)
LULC_EMISSIVITY = {
    'builtup': 0.92,
    'green': 0.97,
    'water': 0.99,
    'bare': 0.90,
}

STEFAN_BOLTZMANN = 5.67e-8  # W/m²/K⁴


def compute_albedo(
    lulc_builtup: float,
    lulc_green: float,
    lulc_water: float,
    lulc_bare: float,
) -> float:
    """
    Compute weighted albedo from land cover fractions.
    This is the physics-informed albedo proxy used in net_radiation_proxy.
    """
    total = lulc_builtup + lulc_green + lulc_water + lulc_bare
    if total <= 0:
        return LULC_ALBEDO['default']

    # Normalize fractions
    b = lulc_builtup / total
    g = lulc_green / total
    w = lulc_water / total
    ba = lulc_bare / total

    albedo = (b * LULC_ALBEDO['builtup']
              + g * LULC_ALBEDO['green']
              + w * LULC_ALBEDO['water']
              + ba * LULC_ALBEDO['bare'])
    return float(np.clip(albedo, 0.05, 0.55))


def compute_emissivity(
    lulc_builtup: float,
    lulc_green: float,
    lulc_water: float,
    lulc_bare: float,
) -> float:
    """Compute weighted surface emissivity from land cover fractions."""
    total = lulc_builtup + lulc_green + lulc_water + lulc_bare
    if total <= 0:
        return 0.95

    b = lulc_builtup / total
    g = lulc_green / total
    w = lulc_water / total
    ba = lulc_bare / total

    return float(b * LULC_EMISSIVITY['builtup']
                 + g * LULC_EMISSIVITY['green']
                 + w * LULC_EMISSIVITY['water']
                 + ba * LULC_EMISSIVITY['bare'])


def compute_net_radiation_proxy(
    albedo: float,
    solar_radiation_wm2: float,
    air_temp_c: float,
    emissivity: float,
) -> float:
    """
    Simplified surface energy balance net radiation (W/m²).

    Rn = (1 - α) × Rg - ε × σ × T⁴
    where:
      α = surface albedo
      Rg = incoming shortwave radiation (W/m²)
      ε = surface emissivity
      σ = Stefan-Boltzmann constant
      T = air temperature (K) as proxy for surface radiative cooling

    Higher Rn → more energy available to heat surface → higher LST.
    This gives the model a physically meaningful input rooted in SEB theory.
    """
    T_K = air_temp_c + 273.15
    Rn = (1 - albedo) * solar_radiation_wm2 - emissivity * STEFAN_BOLTZMANN * T_K ** 4
    return float(Rn)


def compute_ndbi_proxy(lulc_builtup: float, lulc_green: float) -> float:
    """
    Proxy for NDBI (Normalized Difference Built-up Index) from LULC fractions.
    NDBI = (built-up - vegetation) / (built-up + vegetation)
    Range: -1 to +1. Higher values → more built-up → hotter.
    """
    denom = lulc_builtup + lulc_green
    if denom < 0.001:
        return 0.0
    return float(np.clip((lulc_builtup - lulc_green) / denom, -1, 1))


def build_feature_table(
    grid_gdf: gpd.GeoDataFrame,
    lst_df: Optional[pd.DataFrame],
    lulc_df: Optional[pd.DataFrame],
    ndvi_df: Optional[pd.DataFrame],
    osm_dfs: Dict[str, pd.DataFrame],
    weather_dict: Dict[str, float],
    ghsl_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge all data sources into a single per-cell feature DataFrame.

    Returns:
        DataFrame with columns:
          cell_id, centroid_lat, centroid_lon,
          lst_celsius (target),
          ndvi_mean, lulc_builtup_frac, lulc_green_frac,
          lulc_water_frac, lulc_bare_frac,
          industrial_frac, commercial_frac, residential_frac, park_frac,
          building_density, road_density_m_per_km2, dist_water_km,
          pop_density, air_temp_mean_c, relative_humidity_mean,
          wind_speed_mean_ms, solar_radiation_Wm2,
          albedo_proxy, emissivity_proxy, net_radiation_proxy, ndbi_proxy
    """
    logger.info("Building feature table...")

    # Start with grid geometry
    base = grid_gdf[['cell_id', 'centroid_lat', 'centroid_lon']].copy()

    # --- Merge LST (target variable) ---
    if lst_df is not None and not lst_df.empty:
        base = base.merge(lst_df[['cell_id', 'lst_celsius']], on='cell_id', how='left')
    else:
        logger.warning("LST data unavailable — LST will be estimated via proxy")
        base['lst_celsius'] = np.nan
    del lst_df
    gc.collect()

    # --- Merge LULC ---
    if lulc_df is not None and not lulc_df.empty:
        base = base.merge(lulc_df, on='cell_id', how='left')
    else:
        for col in ['lulc_builtup_frac', 'lulc_green_frac', 'lulc_water_frac', 'lulc_bare_frac']:
            base[col] = 0.0
    del lulc_df
    gc.collect()

    # --- Merge NDVI ---
    if ndvi_df is not None and not ndvi_df.empty:
        base = base.merge(ndvi_df[['cell_id', 'ndvi_mean']], on='cell_id', how='left')
    else:
        # Estimate NDVI from LULC green fraction as fallback
        base['ndvi_mean'] = (base.get('lulc_green_frac', pd.Series(0.0)) * 0.7).clip(0, 1)
    del ndvi_df
    gc.collect()

    # --- Merge OSM features ---
    for key, df in osm_dfs.items():
        if df is not None and not df.empty and key != 'cell_id':
            cols_to_merge = [c for c in df.columns if c != 'cell_id']
            base = base.merge(df[['cell_id'] + cols_to_merge], on='cell_id', how='left')
    del osm_dfs
    gc.collect()

    # --- Merge GHSL ---
    if ghsl_df is not None and not ghsl_df.empty:
        base = base.merge(ghsl_df[['cell_id', 'pop_density']], on='cell_id', how='left')
    else:
        base['pop_density'] = 0.0
    del ghsl_df
    gc.collect()

    # --- Add city-level weather (same for all cells) ---
    base['air_temp_mean_c'] = weather_dict.get('air_temp_mean_c', 35.0)
    base['relative_humidity_mean'] = weather_dict.get('relative_humidity_mean', 45.0)
    base['wind_speed_mean_ms'] = weather_dict.get('wind_speed_mean_ms', 2.5)
    base['solar_radiation_Wm2'] = weather_dict.get('shortwave_radiation_Wm2',
                                                     weather_dict.get('solar_radiation_Wm2', 250.0))

    # --- Fill missing values with sensible defaults ---
    default_fills = {
        'lulc_builtup_frac': 0.3,
        'lulc_green_frac': 0.15,
        'lulc_water_frac': 0.02,
        'lulc_bare_frac': 0.05,
        'ndvi_mean': 0.15,
        'industrial_frac': 0.0,
        'commercial_frac': 0.0,
        'residential_frac': 0.3,
        'park_frac': 0.05,
        'building_density': 0.2,
        'building_count': 0,
        'road_density_m_per_km2': 1000.0,
        'dist_water_km': 5.0,
        'pop_density': 500.0,
    }
    for col, default in default_fills.items():
        if col not in base.columns:
            base[col] = default
        else:
            base[col] = base[col].fillna(default)

    # --- Compute physics-informed derived features (VECTORIZED for speed) ---
    logger.info("Computing physics-informed derived features (vectorized)...")

    b = base['lulc_builtup_frac']
    g = base['lulc_green_frac']
    w = base['lulc_water_frac']
    ba = base['lulc_bare_frac']

    # Normalize fractions to sum to 1 (avoid division by zero)
    total = (b + g + w + ba).clip(lower=1e-6)
    b_n, g_n, w_n, ba_n = b / total, g / total, w / total, ba / total

    # Albedo proxy (weighted by LULC fractions) — fully vectorized
    base['albedo_proxy'] = (
        b_n * LULC_ALBEDO['builtup']
        + g_n * LULC_ALBEDO['green']
        + w_n * LULC_ALBEDO['water']
        + ba_n * LULC_ALBEDO['bare']
    ).clip(0.05, 0.55)

    # Emissivity proxy — vectorized
    base['emissivity_proxy'] = (
        b_n * LULC_EMISSIVITY['builtup']
        + g_n * LULC_EMISSIVITY['green']
        + w_n * LULC_EMISSIVITY['water']
        + ba_n * LULC_EMISSIVITY['bare']
    ).clip(0.85, 0.99)

    # Net radiation proxy (core physics-informed feature) — vectorized
    T_K = base['air_temp_mean_c'] + 273.15
    base['net_radiation_proxy'] = (
        (1 - base['albedo_proxy']) * base['solar_radiation_Wm2']
        - base['emissivity_proxy'] * STEFAN_BOLTZMANN * T_K ** 4
    )

    # NDBI proxy — vectorized
    denom = (b + g).clip(lower=1e-6)
    raw_ndbi = (b - g) / denom
    base['ndbi_proxy'] = np.where(denom < 0.001, 0.0, raw_ndbi).clip(-1, 1)

    # Urban heat loading proxy (combines building density + road density + built-up fraction)
    base['urban_heat_load'] = (
        base['lulc_builtup_frac'] * 0.4
        + base['building_density'] * 0.3
        + (base['road_density_m_per_km2'] / 5000.0).clip(0, 1) * 0.3
    )

    # Evapotranspiration proxy (green cover + moisture availability — cools via latent heat)
    base['et_proxy'] = (
        base['ndvi_mean'] * 0.5
        + base['lulc_green_frac'] * 0.3
        + base['lulc_water_frac'] * 0.2
    )

    # LST proxy estimation (for GEE fallback mode)
    # Physics: LST ~ f(net radiation, urban heat load, evapotranspiration, wind cooling)
    base['lst_proxy'] = (
        base['air_temp_mean_c']
        + base['net_radiation_proxy'] * 0.012          # radiation → sensible heat
        + base['urban_heat_load'] * 8.0                # urban heating
        - base['et_proxy'] * 5.0                       # evaporative cooling
        - base['wind_speed_mean_ms'] * 0.8             # wind mixing
        - base['dist_water_km'].clip(0, 10) * 0.15     # water body cooling
    )

    # If GEE LST not available, use the physics proxy
    lst_missing = base['lst_celsius'].isna().mean()
    if lst_missing > 0.5:
        logger.warning(f"{lst_missing:.0%} of cells have no GEE LST. "
                       "Using physics-informed proxy (labeled in UI).")
        base['lst_celsius'] = base['lst_celsius'].fillna(base['lst_proxy'])
        base['lst_source'] = 'proxy'
    else:
        # Fill remaining NaN cells with proxy where GEE had cloud cover
        base['lst_celsius'] = base['lst_celsius'].fillna(base['lst_proxy'])
        base['lst_source'] = 'gee'

    # Final sanity clamp: Indian summer LST range 20–65°C
    # DATA VALIDATION LAYER
    n_anomalies = 0
    
    # 1. Check for impossible LST vs Air Temp divergence
    # Typical LST is up to 15-20C hotter than air temp in extreme urban cases.
    # If LST is > 30C hotter than air temp, it's likely a sensor/cloud error.
    lst_air_diff = base['lst_celsius'] - base['air_temp_mean_c']
    invalid_diff = (lst_air_diff > 30) | (lst_air_diff < -15)
    if invalid_diff.any():
        logger.warning(f"Data Validation: {invalid_diff.sum()} cells have physically improbable LST vs Air Temp differences. Capping.")
        base.loc[lst_air_diff > 30, 'lst_celsius'] = base['air_temp_mean_c'] + 30
        base.loc[lst_air_diff < -15, 'lst_celsius'] = base['air_temp_mean_c'] - 15
        n_anomalies += invalid_diff.sum()
        
    # 2. Strict Range Checks
    out_of_bounds = (base['lst_celsius'] < 10) | (base['lst_celsius'] > 75)
    if out_of_bounds.any():
        logger.warning(f"Data Validation: {out_of_bounds.sum()} cells have LST outside absolute bounds (10-75C).")
        n_anomalies += out_of_bounds.sum()

    base['lst_celsius'] = base['lst_celsius'].clip(20, 65)
    
    # Add validation flag to dataframe for UI
    base.attrs['validation_warnings'] = n_anomalies

    # Normalize road density for model input
    base['road_density_norm'] = (base['road_density_m_per_km2'] / 5000.0).clip(0, 1)

    # Normalize population density
    base['pop_density_norm'] = np.log1p(base['pop_density']) / 10.0

    n_valid_lst = base['lst_celsius'].notna().sum()
    logger.info(
        f"Feature table built: {len(base)} cells, "
        f"{n_valid_lst} with LST data, "
        f"LST mean={base['lst_celsius'].mean():.1f}°C"
    )

    return base


def get_feature_columns() -> list:
    """Return ordered list of ML feature columns (excluding target and metadata)."""
    return [
        'ndvi_mean',
        'lulc_builtup_frac',
        'lulc_green_frac',
        'lulc_water_frac',
        'lulc_bare_frac',
        'industrial_frac',
        'commercial_frac',
        'residential_frac',
        'park_frac',
        'building_density',
        'road_density_norm',
        'dist_water_km',
        'pop_density_norm',
        'albedo_proxy',
        'net_radiation_proxy',
        'ndbi_proxy',
        'urban_heat_load',
        'et_proxy',
        'relative_humidity_mean',
        'wind_speed_mean_ms',
    ]
