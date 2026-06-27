"""
geocode.py — City boundary resolution + adaptive grid generation.
Uses Nominatim (via OSMnx) with strict rate limiting. Never hardcodes
city boundaries — every lookup is live from OpenStreetMap at runtime.
"""

import time
import math
import logging
import warnings
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box, Point
from shapely.ops import unary_union
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Nominatim rate limit: 1 request / second (enforced globally)
_NOMINATIM_DELAY = 1.1  # seconds between requests
_LAST_NOMINATIM_CALL = [0.0]  # mutable for closure

# Target grid cell counts — MINIMIZED for sub-minute analysis.
# GEE reduceRegions time scales ~linearly with feature count.
# 150 cells vs 2500 cells = ~15x faster GEE, with no meaningful UHI quality loss.
# OPTIMIZED: smaller cell counts → faster GEE reduceRegions + smaller FeatureCollection
_GRID_TARGETS = [
    (300,    100),   # city area km² < 300  → 100 cells
    (1000,   150),   # < 1000 km²           → 150 cells
    (3000,   175),   # < 3000 km²           → 175 cells
    (float('inf'), 200),  # any size         → 200 cells max
]

USER_AGENT = "UrbanHeatMitigationAI/1.0 (ISRO-Hackathon; contact: research@urbanheat.ai)"


def _rate_limit():
    """Enforce Nominatim 1 req/sec rate limit."""
    elapsed = time.time() - _LAST_NOMINATIM_CALL[0]
    if elapsed < _NOMINATIM_DELAY:
        time.sleep(_NOMINATIM_DELAY - elapsed)
    _LAST_NOMINATIM_CALL[0] = time.time()


def geocode_city(city_name: str) -> dict:
    """
    Resolve a city name → boundary polygon + metadata.

    Returns:
        dict with keys:
            boundary_gdf   : GeoDataFrame (EPSG:4326) with city polygon
            centroid_lat   : float
            centroid_lon   : float
            display_name   : str
            area_km2       : float
            bbox           : (minx, miny, maxx, maxy)
    Raises:
        ValueError if city cannot be resolved.
    """
    # osmnx settings — set safely for both 1.x and 2.x compatibility
    for _attr, _val in [
        ('log_console', False),      # removed in osmnx 2.0
        ('use_cache', False),         # removed in osmnx 2.0
        ('http_accept_language', 'en'),  # still present in 2.x
    ]:
        try:
            setattr(ox.settings, _attr, _val)
        except AttributeError:
            pass

    # Try with ", India" appended for precision
    queries_to_try = [
        f"{city_name}, India",
        city_name,
        {"city": city_name, "country": "India"},
    ]

    boundary_gdf = None
    for query in queries_to_try:
        try:
            _rate_limit()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boundary_gdf = ox.geocode_to_gdf(query)
            if boundary_gdf is not None and not boundary_gdf.empty:
                logger.info(f"Geocoded '{city_name}' using query: {query}")
                break
        except Exception as e:
            logger.warning(f"Geocode attempt failed for query '{query}': {e}")
            time.sleep(1.0)

    if boundary_gdf is None or boundary_gdf.empty:
        raise ValueError(
            f"Could not find boundary for city '{city_name}'. "
            "Try adding state: e.g. 'Bhopal, Madhya Pradesh'"
        )

    # Ensure single polygon (dissolve if multipolygon)
    if len(boundary_gdf) > 1:
        dissolved = boundary_gdf.dissolve()
        boundary_gdf = gpd.GeoDataFrame(geometry=[dissolved.geometry.iloc[0]], crs="EPSG:4326")

    boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
    geom = boundary_gdf.geometry.iloc[0]
    centroid = geom.centroid
    bbox = geom.bounds  # (minx, miny, maxx, maxy)

    # Area in km² (reproject to metric CRS for accuracy)
    area_km2 = boundary_gdf.to_crs("EPSG:32644").geometry.area.iloc[0] / 1e6

    display_name = boundary_gdf.get("display_name", [city_name])[0] if "display_name" in boundary_gdf.columns else city_name

    return {
        "boundary_gdf": boundary_gdf,
        "centroid_lat": centroid.y,
        "centroid_lon": centroid.x,
        "display_name": display_name,
        "area_km2": area_km2,
        "bbox": bbox,
    }


def build_grid(boundary_gdf: gpd.GeoDataFrame, target_cells: Optional[int] = None) -> gpd.GeoDataFrame:
    """
    Build an adaptive rectangular grid over the city boundary.

    Cell size is chosen so total cells ≈ target_cells (adaptive by area).
    Only cells whose centroid falls inside the boundary polygon are kept.

    Returns:
        GeoDataFrame with columns: geometry (cell polygon), cell_id, centroid_lat, centroid_lon
    """
    boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
    geom = boundary_gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = geom.bounds

    # Project to metric for area calculation
    gdf_metric = boundary_gdf.to_crs("EPSG:32644")
    area_km2 = gdf_metric.geometry.area.iloc[0] / 1e6

    # Choose target cell count adaptively
    if target_cells is None:
        for threshold, count in _GRID_TARGETS:
            if area_km2 < threshold:
                target_cells = count
                break

    # Cell size in degrees (approx 111 km/degree latitude)
    cell_km = math.sqrt(area_km2 / target_cells)
    cell_deg_lat = cell_km / 111.0
    cell_deg_lon = cell_km / (111.0 * math.cos(math.radians((miny + maxy) / 2)))

    # Generate grid
    xs = np.arange(minx, maxx, cell_deg_lon)
    ys = np.arange(miny, maxy, cell_deg_lat)

    cells = []
    cell_id = 0
    for y in ys:
        for x in xs:
            cell_geom = box(x, y, x + cell_deg_lon, y + cell_deg_lat)
            centroid = cell_geom.centroid
            # Keep only cells whose centroid is inside the boundary
            if geom.contains(centroid):
                cells.append({
                    "cell_id": cell_id,
                    "geometry": cell_geom,
                    "centroid_lat": centroid.y,
                    "centroid_lon": centroid.x,
                    "cell_size_km": cell_km,
                })
                cell_id += 1

    if not cells:
        raise ValueError("Grid generation produced no cells — check city boundary.")

    grid_gdf = gpd.GeoDataFrame(cells, crs="EPSG:4326")
    logger.info(
        f"Grid: {len(grid_gdf)} cells, "
        f"cell_size≈{cell_km:.1f} km, "
        f"city_area≈{area_km2:.0f} km²"
    )
    return grid_gdf


def reverse_geocode(lat: float, lon: float, timeout: float = 5.0) -> str:
    """
    Reverse geocode a coordinate to a locality/neighbourhood name.
    Falls back to 'Unknown locality' on failure. Rate-limited.
    """
    _rate_limit()
    try:
        geolocator = Nominatim(user_agent=USER_AGENT)
        location = geolocator.reverse(
            (lat, lon),
            exactly_one=True,
            timeout=timeout,
            language="en",
        )
        if location is None:
            return f"({lat:.3f}°N, {lon:.3f}°E)"

        addr = location.raw.get("address", {})
        # Try progressively coarser locality names
        for field in ["neighbourhood", "suburb", "residential", "quarter",
                      "village", "town", "city_district", "district", "county"]:
            if field in addr:
                return addr[field]
        return addr.get("city", addr.get("state", location.address.split(",")[0]))

    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.warning(f"Reverse geocode failed for ({lat}, {lon}): {e}")
        return f"({lat:.3f}°N, {lon:.3f}°E)"
    except Exception as e:
        logger.warning(f"Reverse geocode unexpected error: {e}")
        return f"({lat:.3f}°N, {lon:.3f}°E)"


def get_city_bbox_ee_format(bbox: Tuple) -> list:
    """Convert shapely bounds tuple → [minx, miny, maxx, maxy] list for EE."""
    minx, miny, maxx, maxy = bbox
    return [minx, miny, maxx, maxy]
