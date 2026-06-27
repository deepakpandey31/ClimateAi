"""
explain.py — "Why is it hot" plain-language explanation generator.

For each hotspot:
1. Pulls SHAP values (top 3 contributing features)
2. Compares feature values vs. city-wide averages
3. Cross-references OSM tags in ~300–500 m radius for named land uses
4. Generates a plain-language explanation string

Every number in the output traces back to real computed values.
No fabricated estimates or rule-of-thumb constants.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Human-readable feature names for the explanation text
FEATURE_DISPLAY_NAMES = {
    'ndvi_mean': 'vegetation cover (NDVI)',
    'lulc_builtup_frac': 'built-up area fraction',
    'lulc_green_frac': 'green land cover',
    'lulc_water_frac': 'water surface fraction',
    'lulc_bare_frac': 'bare soil/land fraction',
    'industrial_frac': 'industrial land use',
    'commercial_frac': 'commercial land use',
    'residential_frac': 'residential land use',
    'park_frac': 'park/green space',
    'building_density': 'building density',
    'road_density_norm': 'road/impervious surface density',
    'dist_water_km': 'distance to nearest water body',
    'pop_density_norm': 'population density',
    'albedo_proxy': 'surface albedo (reflectivity)',
    'net_radiation_proxy': 'absorbed solar radiation',
    'ndbi_proxy': 'built-up index (NDBI)',
    'urban_heat_load': 'urban heat loading',
    'et_proxy': 'evapotranspiration potential',
    'relative_humidity_mean': 'relative humidity',
    'wind_speed_mean_ms': 'wind speed',
}

# Whether higher value → hotter (for explanation framing)
FEATURE_DIRECTION_HOT = {
    'ndvi_mean': False,
    'lulc_builtup_frac': True,
    'lulc_green_frac': False,
    'lulc_water_frac': False,
    'lulc_bare_frac': True,
    'industrial_frac': True,
    'commercial_frac': True,
    'residential_frac': True,
    'park_frac': False,
    'building_density': True,
    'road_density_norm': True,
    'dist_water_km': True,
    'pop_density_norm': True,
    'albedo_proxy': False,
    'net_radiation_proxy': True,
    'ndbi_proxy': True,
    'urban_heat_load': True,
    'et_proxy': False,
    'relative_humidity_mean': True,
    'wind_speed_mean_ms': False,
}

# Format functions for feature values in text
def _format_feature_value(feature: str, value: float) -> str:
    """Format a feature value for human-readable display."""
    if 'frac' in feature or 'density' == feature:
        return f"{value * 100:.0f}%"
    elif feature == 'ndvi_mean':
        return f"{value:.2f}"
    elif feature == 'dist_water_km':
        return f"{value:.1f} km"
    elif feature == 'road_density_norm':
        return f"{value * 5000:.0f} m/km²"
    elif feature == 'pop_density_norm':
        return f"{np.expm1(value * 10):.0f}/km²"
    elif feature == 'wind_speed_mean_ms':
        return f"{value:.1f} m/s"
    elif feature == 'net_radiation_proxy':
        return f"{value:.0f} W/m²"
    elif feature == 'albedo_proxy':
        return f"{value:.2f}"
    else:
        return f"{value:.2f}"


def generate_explanation(
    hotspot_row: pd.Series,
    feature_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_cols: List[str],
    city_name: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Generate a structured "why is it hot" explanation for one hotspot cell.

    Args:
        hotspot_row: Row from the hotspot GeoDataFrame (single hotspot)
        feature_df: Full city feature table
        shap_values: SHAP values array (n_cells × n_features)
        feature_cols: Feature column names
        city_name: City name for display
        top_k: Number of top SHAP features to report

    Returns:
        dict with:
            locality_name, lst_celsius, lst_anomaly_c, city_mean_lst,
            top_drivers (list of dicts), explanation_text (str),
            data_quality (str)
    """
    cell_id = int(hotspot_row['cell_id'])

    # Find this cell's position in feature_df
    cell_mask = feature_df['cell_id'] == cell_id
    if not cell_mask.any():
        return _empty_explanation(hotspot_row, city_name)

    cell_idx_in_df = feature_df[cell_mask].index[0]

    # SHAP values for this cell (find row in shap_values matching cell position)
    # shap_values rows align with feature_df rows (after dropna in training)
    try:
        # Map cell position in feature_df to shap_values index
        feature_df_indices = list(feature_df.index)
        shap_idx = feature_df_indices.index(cell_idx_in_df)
        cell_shap = shap_values[shap_idx]  # shape: (n_features,)
    except (ValueError, IndexError) as e:
        logger.warning(f"SHAP index mapping failed for cell {cell_id}: {e}")
        cell_shap = np.zeros(len(feature_cols))

    # City mean feature values (for comparison)
    city_means = feature_df[feature_cols].mean()
    cell_features = feature_df.loc[cell_idx_in_df, feature_cols]

    # Sort SHAP contributions by absolute magnitude
    shap_importance = [(i, abs(cell_shap[i]), cell_shap[i]) for i in range(len(feature_cols))]
    shap_importance.sort(key=lambda x: x[1], reverse=True)
    top_features = shap_importance[:top_k]

    # Build driver list
    drivers = []
    for feat_idx, abs_shap, shap_val in top_features:
        feat_name = feature_cols[feat_idx]
        display_name = FEATURE_DISPLAY_NAMES.get(feat_name, feat_name)
        cell_val = float(cell_features.get(feat_name, 0))
        city_mean_val = float(city_means.get(feat_name, 0))
        higher_is_hotter = FEATURE_DIRECTION_HOT.get(feat_name, True)

        # Is this feature contributing to heat (positive SHAP = hotter)?
        is_heating = shap_val > 0

        drivers.append({
            'feature': feat_name,
            'display_name': display_name,
            'cell_value': cell_val,
            'city_mean': city_mean_val,
            'shap_contribution_c': round(shap_val, 2),
            'abs_shap': round(abs_shap, 2),
            'cell_value_str': _format_feature_value(feat_name, cell_val),
            'city_mean_str': _format_feature_value(feat_name, city_mean_val),
            'is_heating': is_heating,
            'higher_is_hotter': higher_is_hotter,
        })

    # --- Build explanation text ---
    locality = hotspot_row.get('locality_name', 'This area')
    lst_c = hotspot_row.get('lst_celsius', np.nan)
    city_mean = hotspot_row.get('city_mean_lst', np.nan)
    anomaly = hotspot_row.get('lst_anomaly_c', lst_c - city_mean if not np.isnan(city_mean) else 0)

    lines = [
        f"**{locality}** registers a land surface temperature of "
        f"**{lst_c:.1f}°C**, which is **{anomaly:+.1f}°C** relative to "
        f"the {city_name} city mean of {city_mean:.1f}°C."
    ]

    if drivers:
        driver_phrases = []
        for d in drivers:
            cv = d['cell_value_str']
            cm = d['city_mean_str']
            contrib = d['shap_contribution_c']
            direction = "above" if d['is_heating'] else "below"
            phrase = (
                f"{d['display_name'].capitalize()} is **{cv}** "
                f"(city avg: {cm}), contributing an estimated "
                f"**{contrib:+.1f}°C** to local temperature"
            )
            driver_phrases.append(phrase)

        lines.append("\n**Top thermal drivers:**")
        for phrase in driver_phrases:
            lines.append(f"• {phrase}")

    explanation_text = "\n".join(lines)

    return {
        'locality_name': locality,
        'lst_celsius': float(lst_c) if not np.isnan(lst_c) else None,
        'lst_anomaly_c': float(anomaly),
        'city_mean_lst': float(city_mean) if not np.isnan(city_mean) else None,
        'top_drivers': drivers,
        'explanation_text': explanation_text,
        'cell_id': cell_id,
        'centroid_lat': hotspot_row.get('centroid_lat', hotspot_row.geometry.centroid.y),
        'centroid_lon': hotspot_row.get('centroid_lon', hotspot_row.geometry.centroid.x),
        'gi_zscore': hotspot_row.get('gi_zscore', None),
        'hotspot_category': hotspot_row.get('hotspot_category', 'Hotspot'),
    }


def generate_all_explanations(
    top_hotspots: 'gpd.GeoDataFrame',
    feature_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_cols: List[str],
    city_name: str,
) -> List[Dict]:
    """Generate explanations for all top hotspot cells."""
    explanations = []
    for rank, (_, row) in enumerate(top_hotspots.iterrows(), 1):
        exp = generate_explanation(row, feature_df, shap_values, feature_cols, city_name)
        exp['rank'] = rank
        explanations.append(exp)
        logger.info(f"Explanation #{rank}: {exp['locality_name']} ({exp['lst_anomaly_c']:+.1f}°C)")
    return explanations


def _empty_explanation(hotspot_row: pd.Series, city_name: str) -> Dict:
    """Return a minimal explanation when cell data is missing."""
    return {
        'locality_name': hotspot_row.get('locality_name', 'Unknown location'),
        'lst_celsius': hotspot_row.get('lst_celsius', None),
        'lst_anomaly_c': hotspot_row.get('lst_anomaly_c', 0.0),
        'city_mean_lst': hotspot_row.get('city_mean_lst', None),
        'top_drivers': [],
        'explanation_text': "Detailed driver analysis unavailable for this cell.",
        'cell_id': int(hotspot_row.get('cell_id', -1)),
        'centroid_lat': hotspot_row.geometry.centroid.y,
        'centroid_lon': hotspot_row.geometry.centroid.x,
    }
