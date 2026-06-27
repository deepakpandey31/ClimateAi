"""
hotspot_detection.py — Getis-Ord Gi* spatial statistics for hotspot identification.

Uses esda + libpysal for statistically rigorous hotspot detection.
This is materially more rigorous than top-N sorting — identifies
spatial *clusters* of heat, not just isolated hot pixels.

Each detected hotspot is reverse-geocoded to a real locality name.
"""

import time
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_lst_anomaly(
    grid_gdf: gpd.GeoDataFrame,
    feature_df: pd.DataFrame,
    lst_col: str = 'lst_celsius',
) -> gpd.GeoDataFrame:
    """
    Compute LST z-score (anomaly) per cell relative to city mean.
    Merges back into the grid GeoDataFrame.

    Returns GeoDataFrame with additional columns:
        lst_celsius, lst_anomaly_c (raw difference), lst_zscore
    """
    merged = grid_gdf.merge(
        feature_df[['cell_id', lst_col]].rename(columns={lst_col: 'lst_celsius'}),
        on='cell_id', how='left'
    )

    valid = merged['lst_celsius'].dropna()
    city_mean = float(valid.mean())
    city_std = float(valid.std())

    merged['lst_anomaly_c'] = merged['lst_celsius'] - city_mean
    if city_std > 0:
        merged['lst_zscore'] = merged['lst_anomaly_c'] / city_std
    else:
        merged['lst_zscore'] = 0.0

    merged['city_mean_lst'] = city_mean
    merged['city_std_lst'] = city_std

    logger.info(f"LST anomaly: city mean={city_mean:.1f}°C, std={city_std:.2f}°C")
    return merged


def getis_ord_hotspots(
    grid_lst_gdf: gpd.GeoDataFrame,
    lst_col: str = 'lst_celsius',
    significance_threshold: float = 0.05,
    n_neighbors: int = 8,
) -> gpd.GeoDataFrame:
    """
    Apply Getis-Ord Gi* spatial statistic to identify significant heat clusters.

    Args:
        grid_lst_gdf: GeoDataFrame with LST values
        lst_col: Column with LST values
        significance_threshold: p-value cutoff for significance (default 0.05)
        n_neighbors: K nearest neighbors for spatial weights

    Returns:
        GeoDataFrame with additional columns:
            gi_zscore, gi_pvalue, is_hotspot (bool), hotspot_category
    """
    try:
        from esda.getisord import G_Local
        from libpysal.weights import KNN
    except ImportError:
        logger.error("esda/libpysal not installed. Falling back to z-score ranking.")
        return _fallback_hotspot_detection(grid_lst_gdf, lst_col)

    gdf = grid_lst_gdf.copy()

    # Drop cells with missing LST
    valid_mask = gdf[lst_col].notna()
    if valid_mask.sum() < 30:
        logger.warning(f"Only {valid_mask.sum()} cells with LST. Using z-score fallback.")
        return _fallback_hotspot_detection(gdf, lst_col)

    gdf_valid = gdf[valid_mask].copy().reset_index(drop=True)

    # Build K-nearest-neighbor spatial weights from cell centroids
    try:
        # Use centroid coordinates for weights (faster than polygon weights)
        coords = np.column_stack([gdf_valid.geometry.centroid.x,
                                   gdf_valid.geometry.centroid.y])
        w = KNN.from_array(coords, k=n_neighbors)
        w.transform = 'R'  # Row-standardize

        # Getis-Ord Gi* (star=True includes self in neighborhood)
        # permutations=0 → analytical (asymptotic) p-values: INSTANT, no Monte Carlo
        gi = G_Local(gdf_valid[lst_col].values, w, star=True, permutations=0)

        gdf_valid['gi_zscore'] = gi.z_sim
        gdf_valid['gi_pvalue'] = gi.p_sim

    except Exception as e:
        logger.error(f"Gi* calculation failed: {e}. Using z-score fallback.")
        return _fallback_hotspot_detection(gdf, lst_col)

    # Classify hotspots/coldspots by z-score + significance
    def classify(row):
        if row['gi_pvalue'] > significance_threshold:
            return 'Not significant'
        if row['gi_zscore'] >= 2.576:
            return 'Hotspot (99%)'
        elif row['gi_zscore'] >= 1.960:
            return 'Hotspot (95%)'
        elif row['gi_zscore'] >= 1.645:
            return 'Hotspot (90%)'
        elif row['gi_zscore'] <= -2.576:
            return 'Coldspot (99%)'
        elif row['gi_zscore'] <= -1.960:
            return 'Coldspot (95%)'
        elif row['gi_zscore'] <= -1.645:
            return 'Coldspot (90%)'
        else:
            return 'Not significant'

    gdf_valid['hotspot_category'] = gdf_valid.apply(classify, axis=1)
    gdf_valid['is_hotspot'] = gdf_valid['hotspot_category'].str.startswith('Hotspot')

    # Merge back to full grid
    merge_cols = ['cell_id', 'gi_zscore', 'gi_pvalue', 'is_hotspot', 'hotspot_category']
    gdf = gdf.merge(gdf_valid[merge_cols], on='cell_id', how='left')
    gdf['is_hotspot'] = gdf['is_hotspot'].fillna(False)
    gdf['hotspot_category'] = gdf['hotspot_category'].fillna('No data')

    n_hotspots = gdf['is_hotspot'].sum()
    logger.info(f"Gi* hotspot analysis: {n_hotspots} significant hotspot cells "
                f"({n_hotspots/len(gdf)*100:.1f}% of city)")

    return gdf


def _fallback_hotspot_detection(
    gdf: gpd.GeoDataFrame,
    lst_col: str,
) -> gpd.GeoDataFrame:
    """
    Fallback when esda is unavailable: use simple z-score ranking.
    Labeled clearly in UI as 'simplified hotspot detection'.
    """
    gdf = gdf.copy()
    zscore = (gdf[lst_col] - gdf[lst_col].mean()) / gdf[lst_col].std()
    gdf['gi_zscore'] = zscore
    gdf['gi_pvalue'] = 0.05  # Placeholder
    gdf['is_hotspot'] = zscore >= 1.645  # 90% one-tailed
    gdf['hotspot_category'] = gdf['is_hotspot'].map(
        {True: 'Hotspot (z-score)', False: 'Not significant'}
    )
    logger.warning("Using z-score fallback for hotspot detection (esda unavailable)")
    return gdf


def get_top_hotspot_clusters(
    hotspot_gdf: gpd.GeoDataFrame,
    n: int = 5,
    min_distance_km: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    Select top N spatially distinct hotspot clusters.

    Clusters cells by proximity — takes the hottest representative cell
    from each cluster, ensuring geographic spread (not all in one zone).

    Args:
        hotspot_gdf: GeoDataFrame with is_hotspot and lst_celsius
        n: Number of top hotspot clusters to return
        min_distance_km: Minimum distance between selected hotspot centroids

    Returns:
        GeoDataFrame of top N hotspot representative cells
    """
    hot_cells = hotspot_gdf[hotspot_gdf['is_hotspot']].copy()

    if hot_cells.empty:
        # Fall back to top N by raw LST
        logger.warning("No statistically significant hotspots. Using top-N by LST anomaly.")
        hot_cells = hotspot_gdf.nlargest(n * 5, 'lst_celsius')

    if len(hot_cells) <= n:
        return hot_cells

    # Sort by temperature (hottest first)
    hot_cells = hot_cells.sort_values('lst_celsius', ascending=False).reset_index(drop=True)

    # Greedy spatial selection: pick hottest, then only cells far enough away
    selected_indices = []
    selected_centroids = []
    min_dist_deg = min_distance_km / 111.0  # approx degrees

    for idx, row in hot_cells.iterrows():
        centroid = (row.geometry.centroid.x, row.geometry.centroid.y)

        if not selected_centroids:
            selected_indices.append(idx)
            selected_centroids.append(centroid)
        else:
            # Check distance to all already-selected centroids
            too_close = any(
                np.sqrt((centroid[0] - sc[0])**2 + (centroid[1] - sc[1])**2) < min_dist_deg
                for sc in selected_centroids
            )
            if not too_close:
                selected_indices.append(idx)
                selected_centroids.append(centroid)

        if len(selected_indices) >= n:
            break

    result = hot_cells.loc[selected_indices].copy()
    logger.info(f"Selected {len(result)} spatially distinct hotspot clusters")
    return result


_REVERSE_GEOCODE_CACHE = {}


def name_hotspots(
    hotspot_cells: gpd.GeoDataFrame,
    reverse_geocode_fn,
) -> gpd.GeoDataFrame:
    """
    Attach real locality names to hotspot centroids via reverse geocoding.
    Uses parallel fetching and caching to keep runtime under 1 second.
    """
    from concurrent.futures import ThreadPoolExecutor
    global _REVERSE_GEOCODE_CACHE

    hotspot_cells = hotspot_cells.copy()
    hotspot_cells['centroid_lat'] = hotspot_cells.geometry.centroid.y
    hotspot_cells['centroid_lon'] = hotspot_cells.geometry.centroid.x

    coords = []
    for _, row in hotspot_cells.iterrows():
        lat = round(row['centroid_lat'], 4)
        lon = round(row['centroid_lon'], 4)
        coords.append((lat, lon))

    def _fetch_name(lat, lon):
        key = (lat, lon)
        if key in _REVERSE_GEOCODE_CACHE:
            return _REVERSE_GEOCODE_CACHE[key]
        try:
            name = reverse_geocode_fn(lat, lon)
            _REVERSE_GEOCODE_CACHE[key] = name
            return name
        except Exception:
            return f"({lat:.3f}°N, {lon:.3f}°E)"

    names = []
    # Use max_workers=len(coords) to fetch all names in parallel
    with ThreadPoolExecutor(max_workers=max(1, len(coords))) as executor:
        futures = [executor.submit(_fetch_name, lat, lon) for lat, lon in coords]
        for fut in futures:
            try:
                names.append(fut.result())
            except Exception:
                names.append("Unknown locality")

    hotspot_cells['locality_name'] = names
    return hotspot_cells


def compute_heat_vulnerability_index(
    feature_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute Heat Vulnerability Index per cell.

    HVI = normalized(LST anomaly × √pop_density)
    Flags zones that are both hot AND densely populated — highest harm to people.
    Normalized to 0–100 scale.

    Returns: Series of HVI values indexed by cell_id
    """
    df = feature_df[['cell_id', 'lst_celsius', 'pop_density']].copy()
    df['lst_anomaly'] = df['lst_celsius'] - df['lst_celsius'].mean()
    df['lst_anomaly'] = df['lst_anomaly'].clip(0, None)  # Only positive anomalies matter

    # Square root of population density to reduce outlier influence
    df['pop_sqrt'] = np.sqrt(df['pop_density'].clip(0))

    df['hvi_raw'] = df['lst_anomaly'] * df['pop_sqrt']

    # Normalize to 0–100
    hvi_min = df['hvi_raw'].min()
    hvi_max = df['hvi_raw'].max()
    if hvi_max > hvi_min:
        df['hvi'] = (df['hvi_raw'] - hvi_min) / (hvi_max - hvi_min) * 100
    else:
        df['hvi'] = 0.0

    return df.set_index('cell_id')['hvi']
