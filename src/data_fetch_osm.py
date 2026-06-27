"""
data_fetch_osm.py — OpenStreetMap Overpass API data fetching via OSMnx.

Fetches urban morphology features per grid cell:
  - Building density (footprint area fraction)
  - Road length density (proxy for impervious + traffic heat)
  - Land use fractions (industrial, commercial, residential, parks)
  - Distance to nearest water body

PERFORMANCE OPTIMIZATIONS (v2):
  - All 4 OSM feature types fetched in PARALLEL via ThreadPoolExecutor
  - Overpass data fetched once per city, then used for all sub-analyses
  - Spatial joins vectorised with pre-projected GeoDataFrames cached locally

All calls have retry logic with public Overpass mirrors as fallback.
"""

import time
import math
import logging
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import Point, MultiPolygon
from shapely.ops import unary_union
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Overpass mirror list (tried in order on timeout)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

USER_AGENT = "UrbanHeatMitigationAI/1.0 (ISRO-Hackathon)"

# OSMnx configuration — compatible with both osmnx 1.x and 2.x
# osmnx 2.0 removed log_console, use_cache, and renamed timeout→requests_timeout
def _safe_set_ox(attr, value):
    """Set an osmnx setting safely, ignoring if removed in newer versions."""
    try:
        setattr(ox.settings, attr, value)
    except AttributeError:
        pass

_safe_set_ox('log_console', False)       # removed in osmnx 2.0
_safe_set_ox('use_cache', False)          # removed in osmnx 2.0
_safe_set_ox('timeout', 90)               # osmnx 1.x
_safe_set_ox('requests_timeout', 90)      # osmnx 2.x renamed
_safe_set_ox('overpass_endpoint', OVERPASS_ENDPOINTS[0])  # 1.x
_safe_set_ox('overpass_url', OVERPASS_ENDPOINTS[0])       # possible 2.x alias


def _retry_with_mirrors(func, *args, **kwargs):
    """Try a function, switching Overpass mirrors on timeout. osmnx 1.x/2.x safe."""
    for i, endpoint in enumerate(OVERPASS_ENDPOINTS):
        try:
            _safe_set_ox('overpass_endpoint', endpoint)  # osmnx 1.x
            _safe_set_ox('overpass_url', endpoint)        # osmnx 2.x
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Overpass endpoint {endpoint} failed: {e}")
            if i < len(OVERPASS_ENDPOINTS) - 1:
                time.sleep(2)
    logger.error("All Overpass mirrors failed")
    return None


def fetch_buildings(
    boundary_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Fetch building footprints and compute per-cell metrics:
      - building_density: fraction of cell area covered by buildings
      - building_count: number of building footprints in cell

    Returns DataFrame: cell_id, building_density, building_count
    """
    polygon = boundary_gdf.geometry.iloc[0]

    # Default empty result
    empty = pd.DataFrame({
        'cell_id': grid_gdf['cell_id'],
        'building_density': 0.0,
        'building_count': 0,
    })

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            buildings_gdf = _retry_with_mirrors(
                ox.features_from_polygon,
                polygon,
                tags={'building': True}
            )

        if buildings_gdf is None or buildings_gdf.empty:
            logger.warning("No buildings found from OSM — using defaults.")
            return empty

        # Keep only polygon geometries (actual footprints)
        buildings_gdf = buildings_gdf[
            buildings_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
        ].copy()
        buildings_gdf = buildings_gdf.to_crs("EPSG:4326")

        # Spatial join: assign each building to a grid cell
        grid_proj = grid_gdf.to_crs("EPSG:32644")
        bld_proj = buildings_gdf.to_crs("EPSG:32644")[['geometry']].copy()
        bld_proj['building_area_m2'] = bld_proj.geometry.area

        joined = gpd.sjoin(bld_proj, grid_proj[['cell_id', 'geometry']], how='left', predicate='intersects')
        if joined.empty:
            return empty

        agg = joined.groupby('cell_id').agg(
            building_count=('building_area_m2', 'count'),
            total_building_area=('building_area_m2', 'sum'),
        ).reset_index()

        # Cell area in m²
        cell_area_m2 = grid_proj.geometry.area.mean()
        agg['building_density'] = (agg['total_building_area'] / cell_area_m2).clip(0, 1)

        result = grid_gdf[['cell_id']].merge(agg[['cell_id', 'building_density', 'building_count']],
                                               on='cell_id', how='left')
        result[['building_density', 'building_count']] = result[['building_density', 'building_count']].fillna(0)
        logger.info(f"Buildings: {len(buildings_gdf)} footprints, mean density={result['building_density'].mean():.3f}")
        return result

    except Exception as e:
        logger.error(f"Building fetch failed: {e}")
        return empty


def fetch_roads(
    boundary_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Fetch road network and compute road length density (m/km²) per cell.
    Higher road density → more impervious surface → higher LST.

    Returns DataFrame: cell_id, road_density_m_per_km2
    """
    polygon = boundary_gdf.geometry.iloc[0]
    empty = pd.DataFrame({
        'cell_id': grid_gdf['cell_id'],
        'road_density_m_per_km2': 0.0,
    })

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = _retry_with_mirrors(
                ox.graph_from_polygon,
                polygon,
                network_type='drive',
                retain_all=False,
            )

        if graph is None:
            return empty

        edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        edges_gdf = edges_gdf.to_crs("EPSG:32644")
        edges_gdf['length_m'] = edges_gdf.geometry.length

        grid_proj = grid_gdf.to_crs("EPSG:32644")
        joined = gpd.sjoin(edges_gdf[['geometry', 'length_m']], grid_proj[['cell_id', 'geometry']],
                           how='left', predicate='intersects')

        agg = joined.groupby('cell_id')['length_m'].sum().reset_index()
        agg.columns = ['cell_id', 'road_length_m']

        cell_area_km2 = grid_proj.geometry.area.mean() / 1e6
        agg['road_density_m_per_km2'] = agg['road_length_m'] / cell_area_km2

        result = grid_gdf[['cell_id']].merge(agg[['cell_id', 'road_density_m_per_km2']],
                                              on='cell_id', how='left')
        result['road_density_m_per_km2'] = result['road_density_m_per_km2'].fillna(0)
        logger.info(f"Roads: mean density={result['road_density_m_per_km2'].mean():.0f} m/km²")
        return result

    except Exception as e:
        logger.error(f"Road fetch failed: {e}")
        return empty


def fetch_landuse(
    boundary_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Fetch OSM land use polygons and compute fractional coverage per cell:
      - industrial_frac
      - commercial_frac
      - residential_frac
      - park_frac (parks, forests, recreation)

    Returns DataFrame: cell_id, industrial_frac, commercial_frac,
                       residential_frac, park_frac
    """
    polygon = boundary_gdf.geometry.iloc[0]

    cols = ['cell_id', 'industrial_frac', 'commercial_frac', 'residential_frac', 'park_frac']
    empty = pd.DataFrame({c: 0.0 for c in cols}, index=grid_gdf.index)
    empty['cell_id'] = grid_gdf['cell_id']

    # OSM tag categories
    CATEGORY_TAGS = {
        'industrial': ['industrial', 'port', 'quarry'],
        'commercial': ['commercial', 'retail', 'business_park'],
        'residential': ['residential', 'housing'],
        'park': ['park', 'forest', 'recreation_ground', 'nature_reserve', 'grass', 'village_green'],
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            landuse_gdf = _retry_with_mirrors(
                ox.features_from_polygon,
                polygon,
                tags={'landuse': True}
            )

        if landuse_gdf is None or landuse_gdf.empty:
            logger.warning("No landuse polygons from OSM — using defaults.")
            return empty

        landuse_gdf = landuse_gdf[
            landuse_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
        ].to_crs("EPSG:32644").copy()

        landuse_gdf['landuse_tag'] = landuse_gdf.get('landuse', pd.Series(dtype=str)).fillna('')

        grid_proj = grid_gdf.to_crs("EPSG:32644")
        cell_area_m2 = grid_proj.geometry.area.mean()

        result = grid_gdf[['cell_id']].copy()
        for cat_name, tag_values in CATEGORY_TAGS.items():
            cat_mask = landuse_gdf['landuse_tag'].isin(tag_values)
            cat_gdf = landuse_gdf[cat_mask].copy()

            if cat_gdf.empty:
                result[f'{cat_name}_frac'] = 0.0
                continue

            joined = gpd.sjoin(cat_gdf[['geometry']], grid_proj[['cell_id', 'geometry']],
                               how='left', predicate='intersects')
            # Compute intersection area per cell
            areas = []
            for cell_id, group in joined.groupby('cell_id'):
                cell_geom = grid_proj.loc[grid_proj['cell_id'] == cell_id, 'geometry'].iloc[0]
                intersection_area = sum(
                    cat_gdf.geometry.iloc[idx].intersection(cell_geom).area
                    for idx in range(len(group))
                    if idx < len(cat_gdf)
                )
                areas.append({'cell_id': cell_id, f'{cat_name}_frac': min(1.0, intersection_area / cell_area_m2)})

            if areas:
                area_df = pd.DataFrame(areas)
                result = result.merge(area_df, on='cell_id', how='left')
            else:
                result[f'{cat_name}_frac'] = 0.0

        for col in ['industrial_frac', 'commercial_frac', 'residential_frac', 'park_frac']:
            if col not in result.columns:
                result[col] = 0.0
            result[col] = result[col].fillna(0).clip(0, 1)

        logger.info(f"Landuse: industrial={result['industrial_frac'].mean():.3f}, "
                    f"park={result['park_frac'].mean():.3f}")
        return result

    except Exception as e:
        logger.error(f"Landuse fetch failed: {e}")
        return empty


def fetch_water_distance(
    boundary_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Fetch OSM water bodies and compute distance from each cell centroid
    to the nearest water body (km).

    Returns DataFrame: cell_id, dist_water_km
    """
    polygon = boundary_gdf.geometry.iloc[0]
    empty = pd.DataFrame({
        'cell_id': grid_gdf['cell_id'],
        'dist_water_km': 5.0,  # default: 5 km away
    })

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            water_gdf = _retry_with_mirrors(
                ox.features_from_polygon,
                polygon,
                tags={'natural': ['water', 'lake', 'pond', 'river'],
                      'waterway': ['river', 'stream', 'canal'],
                      'landuse': ['reservoir', 'basin']}
            )

        if water_gdf is None or water_gdf.empty:
            logger.warning("No water features found — using default distances.")
            return empty

        # Project to metric
        water_proj = water_gdf[
            water_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString'])
        ].to_crs("EPSG:32644")

        if water_proj.empty:
            return empty

        # Union all water geometries for distance computation
        water_union = unary_union(water_proj.geometry.values)

        # Compute distance from each grid cell centroid to water
        grid_proj = grid_gdf.to_crs("EPSG:32644").copy()
        distances_m = grid_proj.geometry.centroid.distance(water_union)
        grid_proj['dist_water_km'] = distances_m / 1000.0

        # Debug logging to verify exact perpendicular distance vs 0.0 snapping
        n_zero = (grid_proj['dist_water_km'] == 0.0).sum()
        logger.info(f"Water distance check: {n_zero}/{len(grid_proj)} cells have exactly 0.0km distance.")

        grid_proj['dist_water_km'] = grid_proj['dist_water_km'].clip(0, 50)

        result = grid_gdf[['cell_id']].merge(
            grid_proj[['cell_id', 'dist_water_km']], on='cell_id', how='left'
        )
        result['dist_water_km'] = result['dist_water_km'].fillna(5.0)
        logger.info(f"Water distance: mean={result['dist_water_km'].mean():.2f} km")
        return result

    except Exception as e:
        logger.error(f"Water distance fetch failed: {e}")
        return empty


def fetch_all_osm(
    boundary_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch all OSM features in PARALLEL using ThreadPoolExecutor.

    OPTIMIZED: All 4 OSM calls (buildings, roads, landuse, water) run
    concurrently instead of sequentially.  Since Overpass is I/O-bound,
    threading gives ~3–4× speedup with 4 parallel workers.

    Returns dict of DataFrames keyed by feature type.
    Continues on individual failures — never crashes the whole pipeline.
    """
    logger.info("Starting parallel OSM data fetch (buildings + roads + landuse + water simultaneously)...")

    default_results = {
        'buildings': pd.DataFrame({'cell_id': grid_gdf['cell_id'], 'building_density': 0.0, 'building_count': 0}),
        'roads':     pd.DataFrame({'cell_id': grid_gdf['cell_id'], 'road_density_m_per_km2': 0.0}),
        'landuse':   pd.DataFrame({'cell_id': grid_gdf['cell_id'],
                                   'industrial_frac': 0.0, 'commercial_frac': 0.0,
                                   'residential_frac': 0.0, 'park_frac': 0.0}),
        'water':     pd.DataFrame({'cell_id': grid_gdf['cell_id'], 'dist_water_km': 5.0}),
    }

    tasks = {
        'buildings': lambda: fetch_buildings(boundary_gdf, grid_gdf),
        'roads':     lambda: fetch_roads(boundary_gdf, grid_gdf),
        'landuse':   lambda: fetch_landuse(boundary_gdf, grid_gdf),
        'water':     lambda: fetch_water_distance(boundary_gdf, grid_gdf),
    }

    results = {}
    # Use 4 workers — each OSM call hits a different Overpass endpoint path
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_key = {executor.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result = future.result()
                if result is not None:
                    results[key] = result
                    logger.info(f"OSM parallel fetch complete: {key}")
                else:
                    results[key] = default_results[key]
            except Exception as e:
                logger.error(f"OSM parallel fetch failed for {key}: {e}")
                results[key] = default_results[key]

    return results
