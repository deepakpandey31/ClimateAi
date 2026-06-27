"""
cache.py — Disk-based caching layer for all API responses.
Keyed by (city_slug, date_str, data_type). Uses diskcache for
arbitrary Python objects + parquet for large DataFrames.
"""

import hashlib
import os
import gc
import logging
import diskcache
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Root cache directory — use /tmp/climateai_cache on Linux (e.g. Streamlit Cloud) to avoid permission/read-only issues
if os.name == 'posix':
    CACHE_DIR = Path("/tmp/climateai_cache")
else:
    CACHE_DIR = Path(__file__).parent.parent / "cache"

try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache = diskcache.Cache(str(CACHE_DIR), size_limit=2 * 1024 ** 3)  # 2 GB max
except Exception as e:
    # Fallback to temp directory of the OS if we cannot write to CACHE_DIR
    import tempfile
    CACHE_DIR = Path(tempfile.gettempdir()) / "climateai_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache = diskcache.Cache(str(CACHE_DIR), size_limit=2 * 1024 ** 3)


def _make_key(city: str, date_str: str, data_type: str) -> str:
    raw = f"{city.lower().strip()}|{date_str}|{data_type}"
    return hashlib.sha1(raw.encode()).hexdigest()


def get(city: str, date_str: str, data_type: str):
    """Return cached value or None."""
    key = _make_key(city, date_str, data_type)
    # Check parquet first for DataFrames
    parquet_path = CACHE_DIR / f"{key}.parquet"
    if parquet_path.exists():
        try:
            try:
                import geopandas as gpd
                df = gpd.read_parquet(parquet_path)
            except Exception:
                df = pd.read_parquet(parquet_path)
            logger.debug(f"Cache HIT (parquet): {data_type} for {city}")
            return df
        except Exception as e:
            logger.warning(f"Parquet cache read failed: {e}")
    # Check diskcache for other objects
    val = _cache.get(key, default=None)
    if val is not None:
        logger.debug(f"Cache HIT (disk): {data_type} for {city}")
    return val


def set(city: str, date_str: str, data_type: str, value) -> None:
    """Store value. DataFrames stored as parquet; everything else in diskcache."""
    key = _make_key(city, date_str, data_type)
    try:
        if isinstance(value, pd.DataFrame) and len(value) > 100:
            parquet_path = CACHE_DIR / f"{key}.parquet"
            value.to_parquet(parquet_path, index=False)
            logger.debug(f"Cache SET (parquet): {data_type} for {city}")
        else:
            _cache.set(key, value, expire=86400 * 14)  # 14 days TTL
            logger.debug(f"Cache SET (disk): {data_type} for {city}")
    except Exception as e:
        logger.warning(f"Cache write failed for {data_type}/{city}: {e}")


def clear_city(city: str) -> None:
    """Remove all cache entries for a given city."""
    city_slug = city.lower().strip()
    removed = 0
    for path in CACHE_DIR.glob("*.parquet"):
        # We can't reverse SHA1, so we iterate all keys in diskcache instead
        path.unlink(missing_ok=True)
        removed += 1
    # Clear diskcache entries (brute-force: evict all expired, then rebuild)
    _cache.clear()
    logger.info(f"Cleared cache for city hint '{city_slug}' ({removed} parquet files removed)")


def cache_exists(city: str, date_str: str, data_type: str) -> bool:
    """Check if a cache entry exists without loading it."""
    key = _make_key(city, date_str, data_type)
    parquet_path = CACHE_DIR / f"{key}.parquet"
    return parquet_path.exists() or (key in _cache)


def get_cache_stats() -> dict:
    return {
        "cache_dir": str(CACHE_DIR),
        "disk_cache_size_mb": round(_cache.volume() / 1024 ** 2, 1),
        "parquet_files": len(list(CACHE_DIR.glob("*.parquet"))),
    }
