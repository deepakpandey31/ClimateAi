"""
data_fetch_weather.py — Meteorological data from Open-Meteo and NASA POWER.

Both APIs are completely free with no API key required.
Data is fetched at city centroid level (city-wide average) and
applied uniformly to all grid cells — fine for UHI analysis where
the intra-city temperature variation comes from LST, not weather.
"""

import logging
import time
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

REQUEST_TIMEOUT = 30  # seconds


def _get_analysis_date_range(months: int = 3) -> tuple:
    """Return date range for the most recent complete data window."""
    end = datetime.utcnow() - timedelta(days=5)  # data may lag ~5 days
    start = end - timedelta(days=30 * months)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def fetch_openmeteo(
    lat: float,
    lon: float,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> Dict[str, float]:
    """
    Fetch historical weather from Open-Meteo Archive API.
    Returns dict of scalar values (means over the date range):
        air_temp_mean_c, air_temp_max_c, relative_humidity_mean,
        wind_speed_mean_ms, precipitation_sum_mm
    Falls back to forecast API if archive unavailable.
    """
    # Cap end date to 2 days ago to prevent future date 400 Bad Request on the archive API
    yesterday = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
    if date_end is not None and date_end > yesterday:
        date_end = yesterday
    if date_start is not None and date_end is not None and date_start > date_end:
        date_start = (datetime.strptime(date_end, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

    if date_start is None or date_end is None:
        date_start, date_end = _get_analysis_date_range()

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_start,
        "end_date": date_end,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_max",
            "wind_speed_10m_max",
            "precipitation_sum",
            "shortwave_radiation_sum",
        ],
        "timezone": "Asia/Kolkata",
    }

    defaults = {
        "air_temp_mean_c": 35.0,
        "air_temp_max_c": 42.0,
        "relative_humidity_mean": 45.0,
        "wind_speed_mean_ms": 2.5,
        "precipitation_sum_mm": 10.0,
        "shortwave_radiation_Wm2": 250.0,
    }

    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})
        if not daily or "temperature_2m_mean" not in daily:
            raise ValueError("Unexpected Open-Meteo response structure")

        def safe_mean(arr):
            vals = [v for v in arr if v is not None]
            return float(np.mean(vals)) if vals else np.nan

        result = {
            "air_temp_mean_c": safe_mean(daily.get("temperature_2m_mean", [])),
            "air_temp_max_c": safe_mean(daily.get("temperature_2m_max", [])),
            "relative_humidity_mean": safe_mean(daily.get("relative_humidity_2m_max", [])),
            "wind_speed_mean_ms": safe_mean(daily.get("wind_speed_10m_max", [])) * 0.6,  # max→mean factor
            "precipitation_sum_mm": safe_mean(daily.get("precipitation_sum", [])),
            "shortwave_radiation_Wm2": safe_mean(daily.get("shortwave_radiation_sum", [])) / 0.0864,  # MJ/m²/day → W/m²
        }

        # Replace NaN with defaults
        for key, default in defaults.items():
            if key in result and (result[key] is None or np.isnan(result[key])):
                result[key] = default

        logger.info(f"Open-Meteo: T_mean={result['air_temp_mean_c']:.1f}°C, "
                    f"RH={result['relative_humidity_mean']:.0f}%, "
                    f"radiation={result['shortwave_radiation_Wm2']:.0f} W/m²")
        return result

    except Exception as e:
        logger.warning(f"Open-Meteo fetch failed ({e}), using defaults.")
        return defaults


def fetch_nasa_power(
    lat: float,
    lon: float,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> Dict[str, float]:
    """
    Fetch solar radiation and meteorology from NASA POWER API.
    Used to cross-check Open-Meteo and fill solar radiation if needed.

    Returns: solar_radiation_Wm2, temp_mean_c, wind_speed_ms
    """
    # Cap end date to 2 days ago to prevent future date request errors
    yesterday = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
    if date_end is not None and date_end > yesterday:
        date_end = yesterday
    if date_start is not None and date_end is not None and date_start > date_end:
        date_start = (datetime.strptime(date_end, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

    if date_start is None or date_end is None:
        date_start, date_end = _get_analysis_date_range(months=2)

    # Convert dates to NASA POWER format (YYYYMMDD)
    start_fmt = date_start.replace("-", "")
    end_fmt = date_end.replace("-", "")

    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS10M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_fmt,
        "end": end_fmt,
        "format": "JSON",
    }

    defaults = {
        "solar_radiation_Wm2": 250.0,
        "temp_mean_c": 35.0,
        "wind_speed_ms": 2.5,
    }

    try:
        response = requests.get(NASA_POWER_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        props = data.get("properties", {}).get("parameter", {})
        if not props:
            raise ValueError("Empty NASA POWER response")

        def extract_mean(param_data: dict) -> float:
            vals = [v for v in param_data.values() if v is not None and v > -900]
            return float(np.mean(vals)) if vals else np.nan

        solar = props.get("ALLSKY_SFC_SW_DWN", {})  # kWh/m²/day
        temp = props.get("T2M", {})  # °C
        wind = props.get("WS10M", {})  # m/s

        result = {
            "solar_radiation_Wm2": extract_mean(solar) * 1000 / 24,  # kWh/m²/day → W/m²
            "temp_mean_c": extract_mean(temp),
            "wind_speed_ms": extract_mean(wind),
        }

        for key, default in defaults.items():
            if np.isnan(result.get(key, np.nan)):
                result[key] = default

        logger.info(f"NASA POWER: solar={result['solar_radiation_Wm2']:.0f} W/m², "
                    f"T={result['temp_mean_c']:.1f}°C")
        return result

    except Exception as e:
        logger.warning(f"NASA POWER fetch failed ({e}), using defaults.")
        return defaults


def compute_heat_index(air_temp_c: float, relative_humidity: float) -> float:
    """
    Compute Steadman Heat Index (°C) from air temperature and RH.
    Used for heat-health risk flagging against IMD thresholds.
    Formula: Rothfusz regression (valid for T > 27°C, RH > 40%)
    """
    T_f = air_temp_c * 9/5 + 32  # to Fahrenheit
    RH = relative_humidity

    if T_f < 80:  # Heat index not applicable at low temperatures
        return air_temp_c

    HI_f = (-42.379
            + 2.04901523 * T_f
            + 10.14333127 * RH
            - 0.22475541 * T_f * RH
            - 0.00683783 * T_f ** 2
            - 0.05481717 * RH ** 2
            + 0.00122874 * T_f ** 2 * RH
            + 0.00085282 * T_f * RH ** 2
            - 0.00000199 * T_f ** 2 * RH ** 2)

    # Adjustment for low RH
    if RH < 13 and 80 <= T_f <= 112:
        adj = ((13 - RH) / 4) * np.sqrt((17 - abs(T_f - 95)) / 17)
        HI_f -= adj
    elif RH > 85 and 80 <= T_f <= 87:
        adj = ((RH - 85) / 10) * ((87 - T_f) / 5)
        HI_f += adj

    return (HI_f - 32) * 5/9  # Back to Celsius


def classify_imd_heatwave(heat_index_c: float, region: str = "plains") -> dict:
    """
    Classify heat-health risk using IMD heatwave thresholds.
    Returns: {"level": str, "color": str, "description": str}
    """
    # IMD thresholds (India Meteorological Department)
    # Plains: Heat Wave ≥ 40°C, Severe Heat Wave ≥ 45°C
    if heat_index_c >= 47:
        return {"level": "Extreme", "color": "#8B0000", "description": "Extreme heat danger — IMD Red Alert"}
    elif heat_index_c >= 45:
        return {"level": "Severe", "color": "#FF0000", "description": "Severe heat wave — IMD Orange Alert"}
    elif heat_index_c >= 40:
        return {"level": "Heat Wave", "color": "#FF8C00", "description": "Heat wave conditions — IMD Yellow Alert"}
    elif heat_index_c >= 35:
        return {"level": "Hot", "color": "#FFA500", "description": "Very hot conditions — stay hydrated"}
    else:
        return {"level": "Normal", "color": "#228B22", "description": "Normal conditions"}
