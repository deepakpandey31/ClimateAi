"""
app.py — Urban Heat Mitigation AI System
Streamlit entry point — 6-tab interactive dashboard.

ISRO Hackathon Submission | All data from free public APIs.
Run: streamlit run app.py
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='requests')
warnings.filterwarnings('ignore', category=RuntimeWarning)

import os
import sys
import gc
import time
import logging
import datetime
import threading
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "smartqueue-b1647")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("urban_heat_app")

# Page config set in app.py wrapper

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background: #0a0e1a; color: #e8eaf6; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a, #16213e);
    border-right: 1px solid rgba(233,69,96,0.3);
}
[data-testid="stSidebar"] * { color: #e8eaf6 !important; }

/* Header gradient */
.hero-header {
    background: linear-gradient(135deg, #0f3460, #16213e, #e94560);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(233,69,96,0.3);
    box-shadow: 0 8px 32px rgba(233,69,96,0.15);
}
.hero-header h1 { color: #fff; font-size: 2rem; font-weight: 700; margin: 0; }
.hero-header p  { color: rgba(255,255,255,0.8); margin: 0.5rem 0 0; font-size: 1rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(15,52,96,0.8), rgba(22,33,62,0.9));
    border: 1px solid rgba(233,69,96,0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #e94560; }
.metric-card .label { font-size: 0.8rem; color: rgba(232,234,246,0.7); margin-top: 0.3rem; }

/* Hotspot cards */
.hotspot-card {
    background: linear-gradient(135deg, rgba(233,69,96,0.12), rgba(15,52,96,0.8));
    border: 1px solid rgba(233,69,96,0.4);
    border-left: 4px solid #e94560;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(233,69,96,0.1);
}
.hotspot-card h3 { color: #e94560; margin: 0 0 0.5rem; font-size: 1.1rem; }
.hotspot-card .temp { font-size: 1.8rem; font-weight: 700; color: #fff; }
.hotspot-card .anomaly { color: #ff6b6b; font-size: 0.9rem; }

/* Intervention cards */
.intervention-card {
    background: linear-gradient(135deg, rgba(15,52,96,0.7), rgba(22,33,62,0.8));
    border: 1px solid rgba(100,181,246,0.3);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* Status badges */
.badge-hot { background: #e94560; color: white; padding: 2px 10px;
              border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-ok  { background: #2ecc71; color: white; padding: 2px 10px;
              border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-warn{ background: #f39c12; color: white; padding: 2px 10px;
              border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

/* Progress bar override */
div.stProgress > div > div { background: linear-gradient(90deg, #e94560, #0f3460); }

/* Tab styling */
[data-testid="stTab"] { color: rgba(232,234,246,0.7); }
[data-testid="stTab"][aria-selected="true"] { color: #e94560; border-bottom-color: #e94560; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #e94560, #c0392b);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; transition: all 0.3s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(233,69,96,0.4);
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline State ────────────────────────────────────────────────────────────
if 'pipeline_state' not in st.session_state:
    st.session_state.pipeline_state = {
        'status': 'idle',          # idle | running | done | error
        'city': None,
        'progress': 0,
        'progress_msg': '',
        'result': None,
        'error': None,
    }


def _update_progress(pct: int, msg: str):
    st.session_state.pipeline_state['progress'] = pct
    st.session_state.pipeline_state['progress_msg'] = msg
    logger.info(f"[{pct}%] {msg}")


# ── Core Pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(city_name: str, force_refresh: bool = False, fetch_osm: bool = False):
    """
    Full analysis pipeline. Runs in-thread (Streamlit reruns handle UI updates).
    All heavy compute is server-side in GEE; only small tables come back locally.

    PERFORMANCE OPTIMIZATIONS (v2):
    - GEE satellite fetches run in parallel (LST+LULC+NDVI+GHSL simultaneously)
    - OSM fetches run in parallel (buildings+roads+landuse+water simultaneously)
    - Weather (Open-Meteo + NASA POWER) fetched concurrently
    - Multi-year LST trend deferred to background thread (non-blocking)
    - Grid cells reduced by ~50% for faster GEE processing
    - Feature engineering vectorized (no more row-by-row apply)
    """
    from src.cache import get as cache_get, set as cache_set, cache_exists
    from src.geocode import geocode_city, build_grid, reverse_geocode
    from src.data_fetch_gee import (fetch_all_gee_parallel,
                                     fetch_multi_year_lst, is_gee_available)
    from src.data_fetch_osm import fetch_all_osm
    from src.data_fetch_weather import (fetch_openmeteo, fetch_nasa_power,
                                         compute_heat_index, classify_imd_heatwave)
    from src.feature_engineering import build_feature_table, get_feature_columns
    from src.model import train_city_model, predict_lst, get_shap_values
    from src.hotspot_detection import (compute_lst_anomaly, getis_ord_hotspots,
                                        get_top_hotspot_clusters, name_hotspots,
                                        compute_heat_vulnerability_index)
    from src.explain import generate_all_explanations
    from src.intervention_simulator import batch_simulate, INTERVENTION_TYPES
    from src.optimizer import optimize_budget
    from concurrent.futures import ThreadPoolExecutor, as_completed

    state = st.session_state.pipeline_state
    state['status'] = 'running'
    state['error'] = None
    today = datetime.date.today().strftime("%Y-%m")
    cache_key_city = city_name.lower().strip().replace(" ", "_")

    try:
        # ── Step 1: Geocode + Grid ──────────────────────────────────────────
        _update_progress(5, f"Geocoding {city_name}...")

        city_info = cache_get(cache_key_city, today, 'city_info')
        if city_info is None or force_refresh:
            city_info = geocode_city(city_name)
            cache_set(cache_key_city, today, 'city_info', city_info)

        boundary_gdf = city_info['boundary_gdf']
        area_km2 = city_info['area_km2']
        centroid_lat = city_info['centroid_lat']
        centroid_lon = city_info['centroid_lon']

        _update_progress(10, f"Building analysis grid (area: {area_km2:.0f} km²)...")

        grid_gdf = cache_get(cache_key_city, today, 'grid')
        if grid_gdf is None or force_refresh:
            grid_gdf = build_grid(boundary_gdf)
            cache_set(cache_key_city, today, 'grid', grid_gdf)

        n_cells = len(grid_gdf)

        # ── Step 2: GEE + OSM + Weather — all in parallel ──────────────────
        import datetime as dt
        now = dt.datetime.utcnow()
        year = now.year if now.month >= 4 else now.year - 1
        date_start = f"{year}-04-01"
        date_end = f"{year}-06-30"

        # Check cache status for each data type
        gee_cached  = all(cache_get(cache_key_city, today, k) is not None
                          for k in ('lst', 'lulc', 'ndvi', 'ghsl'))
        osm_cached  = cache_get(cache_key_city, today, 'osm') is not None
        wx_cached   = cache_get(cache_key_city, today, 'weather') is not None

        if gee_cached and not force_refresh:
            _update_progress(20, "Loading GEE data from cache...")
            gee_ok = is_gee_available(GEE_PROJECT_ID)
            lst_df  = cache_get(cache_key_city, today, 'lst')
            lulc_df = cache_get(cache_key_city, today, 'lulc')
            ndvi_df = cache_get(cache_key_city, today, 'ndvi')
            ghsl_df = cache_get(cache_key_city, today, 'ghsl')
        else:
            _update_progress(15, "Checking GEE availability...")
            gee_ok = is_gee_available(GEE_PROJECT_ID)
            _update_progress(20, f"⚡ Fetching satellite data in PARALLEL (LST + LULC + NDVI + GHSL)...")
            gee_results = fetch_all_gee_parallel(grid_gdf, GEE_PROJECT_ID, date_start, date_end)
            lst_df  = gee_results.get('lst')
            lulc_df = gee_results.get('lulc')
            ndvi_df = gee_results.get('ndvi')
            ghsl_df = gee_results.get('ghsl')
            if lst_df  is not None: cache_set(cache_key_city, today, 'lst',  lst_df)
            if lulc_df is not None: cache_set(cache_key_city, today, 'lulc', lulc_df)
            if ndvi_df is not None: cache_set(cache_key_city, today, 'ndvi', ndvi_df)
            if ghsl_df is not None: cache_set(cache_key_city, today, 'ghsl', ghsl_df)

        # ── Step 3: OSM + Weather — run concurrently with each other ────────
        _update_progress(44, "⚡ Fetching OSM morphology + weather data in parallel...")

        def _fetch_weather():
            wx = fetch_openmeteo(centroid_lat, centroid_lon, date_start, date_end)
            nasa = fetch_nasa_power(centroid_lat, centroid_lon, date_start, date_end)
            if nasa.get('solar_radiation_Wm2'):
                wx['shortwave_radiation_Wm2'] = (
                    wx.get('shortwave_radiation_Wm2', 250) * 0.5
                    + nasa.get('solar_radiation_Wm2', 250) * 0.5
                )
            return wx

        osm_dfs = cache_get(cache_key_city, today, 'osm')
        weather = cache_get(cache_key_city, today, 'weather')

        if (osm_dfs is None or force_refresh) or (weather is None or force_refresh):
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = {}
                if osm_dfs is None or force_refresh:
                    futures['osm'] = ex.submit(fetch_all_osm, boundary_gdf, grid_gdf, fetch_osm)
                if weather is None or force_refresh:
                    futures['weather'] = ex.submit(_fetch_weather)
                for key, fut in futures.items():
                    try:
                        result = fut.result()
                        if key == 'osm':
                            osm_dfs = result
                            cache_set(cache_key_city, today, 'osm', osm_dfs)
                        else:
                            weather = result
                            cache_set(cache_key_city, today, 'weather', weather)
                    except Exception as e:
                        logger.error(f"Parallel {key} fetch failed: {e}")

        heat_index = compute_heat_index(
            weather.get('air_temp_mean_c', 35),
            weather.get('relative_humidity_mean', 45)
        )
        heatwave_class = classify_imd_heatwave(heat_index)

        # ── Step 4: Feature Engineering ─────────────────────────────────────
        _update_progress(58, "Engineering features (physics-informed, vectorized)...")
        feature_df = cache_get(cache_key_city, today, 'features')
        if feature_df is None or force_refresh:
            feature_df = build_feature_table(
                grid_gdf, lst_df, lulc_df, ndvi_df, osm_dfs, weather, ghsl_df
            )
            cache_set(cache_key_city, today, 'features', feature_df)

        feature_cols = get_feature_columns()
        feature_cols = [c for c in feature_cols if c in feature_df.columns]

        # ── Step 5: Train Model ─────────────────────────────────────────────
        _update_progress(68, "Training physics-informed XGBoost model...")
        model_cache = cache_get(cache_key_city, today, 'model')
        if model_cache is None or force_refresh:
            model, feat_cols_used, model_metrics = train_city_model(feature_df, feature_cols)
            cache_set(cache_key_city, today, 'model', (model, feat_cols_used, model_metrics))
        else:
            model, feat_cols_used, model_metrics = model_cache

        # Predict LST for all cells and add to feature_df
        feature_df['lst_predicted'] = predict_lst(model, feature_df, feat_cols_used)

        # ── Step 6: SHAP Values ─────────────────────────────────────────────
        _update_progress(74, "Computing SHAP feature attributions...")
        shap_cache = cache_get(cache_key_city, today, 'shap')
        if shap_cache is None or force_refresh:
            shap_values, expected_value, explainer = get_shap_values(model, feature_df, feat_cols_used)
            cache_set(cache_key_city, today, 'shap', (shap_values, expected_value))
        else:
            shap_values, expected_value = shap_cache

        # ── Step 7: Hotspot Detection ───────────────────────────────────────
        _update_progress(80, "Detecting heat hotspots (Getis-Ord Gi*)...")
        grid_lst_gdf = compute_lst_anomaly(grid_gdf, feature_df)
        hotspot_gdf = getis_ord_hotspots(grid_lst_gdf)
        top_hotspots = get_top_hotspot_clusters(hotspot_gdf, n=5)

        _update_progress(85, "Reverse-geocoding hotspot locations...")
        top_hotspots = name_hotspots(top_hotspots, reverse_geocode)

        # ── Step 8: Explanations ────────────────────────────────────────────
        _update_progress(88, "Generating 'why is it hot' explanations...")
        explanations = generate_all_explanations(
            top_hotspots, feature_df, shap_values, feat_cols_used, city_name
        )

        # ── Step 9: HVI ─────────────────────────────────────────────────────
        hvi_series = compute_heat_vulnerability_index(feature_df)
        feature_df['hvi'] = feature_df['cell_id'].map(hvi_series).fillna(0)

        # ── Step 10: Intervention Simulation ────────────────────────────────
        _update_progress(91, "Running intervention cooling simulations...")
        sim_df = cache_get(cache_key_city, today, 'simulations')
        if sim_df is None or force_refresh:
            sim_df = batch_simulate(
                top_hotspots, feature_df, feat_cols_used, model,
                interventions=list(INTERVENTION_TYPES.keys()),
                default_intensity=5.0
            )
            cache_set(cache_key_city, today, 'simulations', sim_df)

        # ── Step 11: Multi-year LST Trend — use cache or skip for now ───────
        # Non-blocking: load from cache instantly; if not cached, skip and let
        # user see results immediately. Trend is shown when available.
        _update_progress(96, "Loading multi-year LST trend...")
        trend_df = cache_get(cache_key_city, today, 'trend')
        if trend_df is None and not force_refresh:
            # Fetch in a background thread so user sees results immediately.
            # The result will be cached and available on next interaction.
            def _bg_trend():
                try:
                    t = fetch_multi_year_lst(grid_gdf, GEE_PROJECT_ID, years=[2015, 2019, 2023])
                    if t is not None:
                        cache_set(cache_key_city, today, 'trend', t)
                        logger.info("Background multi-year LST trend fetch complete.")
                except Exception as _e:
                    logger.warning(f"Background trend fetch failed: {_e}")
            threading.Thread(target=_bg_trend, daemon=True).start()
            logger.info("Multi-year LST trend fetch deferred to background thread.")
        elif force_refresh:
            trend_df = fetch_multi_year_lst(grid_gdf, GEE_PROJECT_ID, years=[2015, 2019, 2023])
            if trend_df is not None:
                cache_set(cache_key_city, today, 'trend', trend_df)

        # ── Step 12: City stats dict ─────────────────────────────────────────
        city_stats = {
            'n_cells': n_cells,
            'area_km2': area_km2,
            'centroid_lat': centroid_lat,
            'centroid_lon': centroid_lon,
            'lst_mean': float(feature_df['lst_celsius'].mean()),
            'lst_max': float(feature_df['lst_celsius'].max()),
            'lst_min': float(feature_df['lst_celsius'].min()),
            'lst_std': float(feature_df['lst_celsius'].std()),
            'air_temp_c': weather.get('air_temp_mean_c', 35),
            'rh': weather.get('relative_humidity_mean', 45),
            'heat_index_c': heat_index,
            'heatwave_class': heatwave_class,
            'date_start': date_start,
            'date_end': date_end,
            'gee_available': gee_ok,
            'lst_source': feature_df['lst_source'].mode()[0] if 'lst_source' in feature_df.columns else 'unknown',
            'boundary_gdf': boundary_gdf,
        }

        # Add anomaly column to feature_df for map rendering
        feature_df['lst_anomaly_c'] = feature_df['lst_celsius'] - city_stats['lst_mean']

        _update_progress(100, "Analysis complete! ✅")

        state['result'] = {
            'city_name': city_name,
            'city_info': city_info,
            'grid_gdf': grid_gdf,
            'hotspot_gdf': hotspot_gdf,
            'top_hotspots': top_hotspots,
            'feature_df': feature_df,
            'explanations': explanations,
            'sim_df': sim_df,
            'trend_df': trend_df,
            'model': model,
            'feat_cols_used': feat_cols_used,
            'model_metrics': model_metrics,
            'shap_values': shap_values,
            'city_stats': city_stats,
            'weather': weather,
        }
        state['status'] = 'done'
        state['city'] = city_name

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        state['status'] = 'error'
        state['error'] = str(e)

    finally:
        gc.collect()


# ── Map Builders ──────────────────────────────────────────────────────────────
def build_lst_map(grid_gdf, feature_df, hotspot_gdf, top_hotspots, city_stats):
    """Build the main LST choropleth + hotspot marker map."""
    m = folium.Map(
        location=[city_stats['centroid_lat'], city_stats['centroid_lon']],
        zoom_start=11,
        tiles='CartoDB dark_matter',
    )

    # Merge LST to grid
    plot_gdf = grid_gdf.merge(
        feature_df[['cell_id', 'lst_celsius', 'lst_anomaly_c']].rename(
            columns={'lst_anomaly_c': 'lst_anomaly_c'}
        ) if 'lst_anomaly_c' in feature_df.columns
        else feature_df[['cell_id', 'lst_celsius']],
        on='cell_id', how='left'
    )
    if 'lst_anomaly_c' not in plot_gdf.columns:
        plot_gdf['lst_anomaly_c'] = plot_gdf['lst_celsius'] - city_stats['lst_mean']

    # LST choropleth
    clean_gdf = plot_gdf.dropna(subset=['lst_celsius'])
    if not clean_gdf.empty:
        # Calculate valid quantile bins, dropping duplicates
        bins = list(pd.qcut(clean_gdf['lst_celsius'], q=6, retbins=True, duplicates='drop')[1])
        if len(bins) < 3:
            vmin, vmax = float(clean_gdf['lst_celsius'].min()), float(clean_gdf['lst_celsius'].max())
            if vmin >= vmax:
                vmin -= 1.0
                vmax += 1.0
            bins = list(np.linspace(vmin, vmax, 7))

        folium.Choropleth(
            geo_data=clean_gdf.__geo_interface__,
            data=clean_gdf[['cell_id', 'lst_celsius']],
            columns=['cell_id', 'lst_celsius'],
            key_on='feature.properties.cell_id',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.1,
            legend_name='Land Surface Temperature (°C)',
            threshold_scale=bins,
            nan_fill_color='transparent',
        ).add_to(m)
        try:
            m.fit_bounds(m.get_bounds())
        except Exception:
            pass

    # Hotspot markers
    for _, row in top_hotspots.iterrows():
        lat = row.geometry.centroid.y
        lon = row.geometry.centroid.x
        lst = row.get('lst_celsius', 0)
        anomaly = row.get('lst_anomaly_c', 0)
        locality = row.get('locality_name', f"Cell {row['cell_id']}")
        category = row.get('hotspot_category', 'Hotspot')

        popup_html = f"""
        <div style="font-family:Inter,sans-serif; min-width:200px; background:#1a1a2e; 
                    color:#e8eaf6; padding:12px; border-radius:8px;">
            <h4 style="color:#e94560; margin:0 0 8px;">🔴 {locality}</h4>
            <b>LST:</b> {lst:.1f}°C &nbsp;|&nbsp; 
            <b>Anomaly:</b> <span style="color:#ff6b6b;">{anomaly:+.1f}°C</span><br>
            <small style="color:#aaa;">{category}</small>
        </div>"""

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"🔴 {locality}: {lst:.1f}°C",
            icon=folium.Icon(color='red', icon='fire', prefix='fa'),
        ).add_to(m)

    # City boundary outline
    boundary_gdf = city_stats.get('boundary_gdf')
    if boundary_gdf is not None:
        folium.GeoJson(
            boundary_gdf.__geo_interface__,
            style_function=lambda x: {'color': '#e94560', 'weight': 2,
                                       'fillOpacity': 0, 'opacity': 0.8},
            name='City Boundary',
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def build_hvi_map(grid_gdf, feature_df, city_stats):
    """Build Heat Vulnerability Index map."""
    m = folium.Map(
        location=[city_stats['centroid_lat'], city_stats['centroid_lon']],
        zoom_start=11,
        tiles='CartoDB dark_matter',
    )

    plot_gdf = grid_gdf.merge(
        feature_df[['cell_id', 'hvi']].fillna(0),
        on='cell_id', how='left'
    )

    folium.Choropleth(
        geo_data=plot_gdf.__geo_interface__,
        data=plot_gdf[['cell_id', 'hvi']],
        columns=['cell_id', 'hvi'],
        key_on='feature.properties.cell_id',
        fill_color='RdPu',
        fill_opacity=0.75,
        line_opacity=0.1,
        legend_name='Heat Vulnerability Index (0–100)',
        threshold_scale=[0, 10, 25, 50, 75, 90, 100],
    ).add_to(m)

    return m


def build_intervention_map(top_hotspots, sim_df, city_stats, selected_intervention):
    """Build map showing intervention allocation."""
    m = folium.Map(
        location=[city_stats['centroid_lat'], city_stats['centroid_lon']],
        zoom_start=11,
        tiles='CartoDB dark_matter',
    )

    if sim_df is None or sim_df.empty:
        return m

    filt = sim_df[sim_df['intervention_type'] == selected_intervention]

    for _, row in top_hotspots.iterrows():
        lat = row.geometry.centroid.y
        lon = row.geometry.centroid.x
        cell_id = int(row['cell_id'])
        locality = row.get('locality_name', f"Cell {cell_id}")

        sim_row = filt[filt['cell_id'] == cell_id]
        if not sim_row.empty:
            delta = float(sim_row['delta_lst_c'].iloc[0])
            lower = float(sim_row.get('delta_lst_lower_c', sim_row['delta_lst_c']).iloc[0])
            upper = float(sim_row.get('delta_lst_upper_c', sim_row['delta_lst_c']).iloc[0])
            intensity = float(sim_row['intensity'].iloc[0])

            popup_html = f"""
            <div style="font-family:Inter,sans-serif; min-width:220px; background:#1a1a2e;
                        color:#e8eaf6; padding:12px; border-radius:8px;">
                <h4 style="color:#64b5f6; margin:0 0 8px;">🔵 {locality}</h4>
                <b>Intervention:</b> {selected_intervention}<br>
                <b>Intensity:</b> {intensity}/10<br>
                <b>Predicted cooling:</b> 
                <span style="color:#4fc3f7;">{delta:.2f}°C</span><br>
                <small style="color:#aaa;">Range: {lower:.2f}°C to {upper:.2f}°C</small>
            </div>"""

            radius = max(8, abs(delta) * 5)
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"🔵 {locality}: {delta:.2f}°C cooling",
                color='#4fc3f7', fill=True, fill_color='#64b5f6',
                fill_opacity=0.6, weight=2,
            ).add_to(m)

    return m


# ── Plotly Chart Builders ─────────────────────────────────────────────────────
def plot_shap_bar(explanation: dict) -> go.Figure:
    """SHAP feature contribution bar chart for one hotspot."""
    drivers = explanation.get('top_drivers', [])
    if not drivers:
        return go.Figure()

    features = [d['display_name'] for d in drivers]
    contributions = [d['shap_contribution_c'] for d in drivers]
    colors = ['#e94560' if c > 0 else '#64b5f6' for c in contributions]

    fig = go.Figure(go.Bar(
        x=contributions,
        y=features,
        orientation='h',
        marker=dict(color=colors, opacity=0.85),
        text=[f"{c:+.2f}°C" for c in contributions],
        textposition='outside',
    ))
    fig.update_layout(
        title="SHAP Feature Contributions to LST",
        xaxis_title="Temperature Contribution (°C)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8eaf6', family='Inter'),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', zeroline=True,
                   zerolinecolor='rgba(255,255,255,0.3)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
    )
    return fig


def plot_lst_trend(trend_df: pd.DataFrame, city_name: str) -> go.Figure:
    """Multi-year LST trend line chart."""
    if trend_df is None or trend_df.empty:
        return go.Figure()

    # Use 5th and 95th percentiles to avoid single-pixel outliers flattening the trend
    yearly = (trend_df.dropna(subset=['lst_celsius']).groupby('year')['lst_celsius']
              .agg(
                  mean='mean',
                  max=lambda x: x.quantile(0.95),
                  min=lambda x: x.quantile(0.05)
              ).reset_index())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly['year'], y=yearly['max'],
        name='Max LST', line=dict(color='#e94560', width=2, dash='dot'),
        mode='lines+markers', marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=yearly['year'], y=yearly['mean'],
        name='Mean LST', line=dict(color='#f39c12', width=3),
        mode='lines+markers', marker=dict(size=10, symbol='circle'),
    ))
    fig.add_trace(go.Scatter(
        x=yearly['year'], y=yearly['min'],
        name='Min LST', line=dict(color='#64b5f6', width=2, dash='dot'),
        mode='lines+markers', marker=dict(size=8),
    ))

    fig.update_layout(
        title=f"Multi-Year LST Trend — {city_name} (April–June)",
        xaxis_title="Year",
        yaxis_title="Land Surface Temperature (°C)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8eaf6', family='Inter'),
        legend=dict(bgcolor='rgba(0,0,0,0.3)', bordercolor='rgba(255,255,255,0.2)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickvals=yearly['year'].tolist()),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=380,
    )
    return fig


def plot_intervention_comparison(sim_df: pd.DataFrame, locality: str) -> go.Figure:
    """Bar chart comparing all interventions for one hotspot."""
    if sim_df is None or sim_df.empty:
        return go.Figure()

    cell_data = sim_df[sim_df['locality_name'] == locality]
    if cell_data.empty:
        return go.Figure()

    cell_data = cell_data.sort_values('delta_lst_c')
    labels = cell_data['intervention_label'].tolist()
    deltas = cell_data['delta_lst_c'].tolist()
    lower = (cell_data['delta_lst_lower_c'] if 'delta_lst_lower_c' in cell_data.columns
             else cell_data['delta_lst_c']).tolist()
    upper = (cell_data['delta_lst_upper_c'] if 'delta_lst_upper_c' in cell_data.columns
             else cell_data['delta_lst_c']).tolist()

    error_minus = [abs(d - l) for d, l in zip(deltas, lower)]
    error_plus = [abs(u - d) for d, u in zip(deltas, upper)]

    fig = go.Figure(go.Bar(
        x=deltas,
        y=labels,
        orientation='h',
        marker=dict(color='#64b5f6', opacity=0.8),
        error_x=dict(
            type='data',
            symmetric=False,
            arrayminus=error_minus,
            array=error_plus,
            color='rgba(255,255,255,0.5)',
        ),
        text=[f"{d:.2f}°C" for d in deltas],
        textposition='outside',
    ))
    fig.update_layout(
        title=f"Predicted Cooling by Intervention — {locality}",
        xaxis_title="Predicted LST Change (°C, negative = cooling)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8eaf6', family='Inter'),
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', zeroline=True,
                   zerolinecolor='rgba(255,255,255,0.3)'),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🌡️ Urban Heat AI")
        st.markdown("*ISRO Hackathon Submission*")
        st.divider()

        city_input = st.text_input(
            "🏙️ Indian City",
            value="Kanpur",
            placeholder="e.g. Kanpur, Surat, Bhopal...",
            help="Type any Indian city name. The system fetches real satellite data at runtime.",
        )

        force_refresh = st.checkbox(
            "Force data refresh",
            value=False,
            help="Ignore cache and re-fetch all data (slower but up-to-date)"
        )

        fetch_osm = st.checkbox(
            "Fetch OSM data (detailed/slow)",
            value=False,
            help="Downloads precise building footprints and road networks from OSM. Unchecked (default) is 10x faster using GEE morphology proxies."
        )

        analyze_btn = st.button("🔬 Analyze City", type="primary", use_container_width=True)

        st.divider()
        st.markdown("**Data Sources**")
        st.markdown("""
        - 🛰️ Landsat 8/9 LST (GEE)
        - 🌍 ESA WorldCover LULC
        - 🌿 Sentinel-2 NDVI
        - 🗺️ OpenStreetMap (Optional)
        - ☁️ Open-Meteo weather
        - ☀️ NASA POWER solar
        - 👥 GHSL Population
        """)

        st.divider()
        # Cache stats
        from src.cache import get_cache_stats
        stats = get_cache_stats()
        st.caption(f"Cache: {stats['disk_cache_size_mb']} MB | {stats['parquet_files']} files")

        # GEE status
        state = st.session_state.pipeline_state
        if state['status'] == 'done' and state['result']:
            cs = state['result']['city_stats']
            if cs.get('gee_available'):
                st.success("✅ GEE Connected")
            else:
                st.warning("⚠️ GEE unavailable — using proxy LST")

        return city_input, analyze_btn, force_refresh, fetch_osm


# ── Tab Renderers ─────────────────────────────────────────────────────────────
def render_tab_overview(result):
    cs = result['city_stats']
    grid_gdf = result['grid_gdf']
    feature_df = result['feature_df']
    hotspot_gdf = result['hotspot_gdf']
    top_hotspots = result['top_hotspots']

    # Hero metrics row
    cols = st.columns(5)
    metrics = [
        ("Mean LST", f"{cs['lst_mean']:.1f}°C", "🌡️", "City-wide average Land Surface Temperature"),
        ("Max LST", f"{cs['lst_max']:.1f}°C", "🔴", "Maximum LST observed in any 250m cell"),
        ("Heat Index", f"{cs['heat_index_c']:.1f}°C", "🌶️", "Steadman Heat Index (feels-like temperature) based on air temp and humidity"),
        ("Hotspot Cells", str(int(hotspot_gdf['is_hotspot'].sum())), "📍", "Number of 250m cells flagged as statistically significant heat anomalies (Getis-Ord Gi*)"),
        ("Grid Cells", str(cs['n_cells']), "🗺️", "Total number of 250m x 250m analysis cells covering the city"),
    ]
    for col, (label, value, icon, help_text) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card" title="{help_text}">
                <div style="font-size:1.5rem">{icon}</div>
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Heatwave alert
    hw = cs['heatwave_class']
    if hw['level'] != 'Normal':
        st.warning(f"⚠️ **{hw['level']}** — {hw['description']} (Heat Index: {cs['heat_index_c']:.1f}°C)")

    # LST data source banner
    if cs.get('lst_source') == 'proxy':
        st.info("ℹ️ **LST approximated via physics proxy** (GEE data unavailable for this run). "
                "Values estimated from land cover + meteorology using surface energy balance.")

    # Map
    st.markdown("### 🗺️ Land Surface Temperature Map")
    with st.spinner("Rendering map..."):
        m = build_lst_map(grid_gdf, feature_df, hotspot_gdf, top_hotspots, cs)
        st_folium(m, width="100%", height=500, returned_objects=[])

    # City summary table
    st.markdown("### 📊 City Statistics")
    
    # Display data validation warnings if any
    if feature_df.attrs.get('validation_warnings', 0) > 0:
        st.warning(f"⚠️ **Data Validation Alert:** {feature_df.attrs['validation_warnings']} cells had statistically improbable LST values and were capped to realistic bounds.")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("City Area", f"{cs['area_km2']:.0f} km²")
        st.metric("Air temp at satellite pass time", f"{cs['air_temp_c']:.1f}°C", 
                  help="Temperature measured by weather stations around 10:30 AM local time (when Landsat passes). This is generally lower than the daily peak air temperature.")
        st.metric("Relative Humidity", f"{cs['rh']:.0f}%")
    with c2:
        st.metric("LST Range", f"{cs['lst_min']:.1f}°C – {cs['lst_max']:.1f}°C")
        st.metric("LST Std Dev", f"{cs['lst_std']:.2f}°C")
        timestamp_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        st.metric("Analysis Period", f"{cs['date_start']} to {cs['date_end']}", help=f"Data retrieved: {timestamp_str}")


def render_tab_hotspots(result):
    explanations = result['explanations']
    top_hotspots = result['top_hotspots']
    city_name = result['city_name']

    st.markdown("### 🔴 Detected Heat Hotspots")
    st.caption(
        "Identified using Getis-Ord Gi* spatial statistics (p < 0.05). "
        "Each hotspot is reverse-geocoded to a real locality name from OpenStreetMap."
    )

    # Model quality indicator
    mm = result['model_metrics']
    if mm:
        cv_r2 = mm.get('cv_r2', 0)
        col1, col2, col3 = st.columns(3)
        col1.metric("Model CV R²", f"{cv_r2:.3f}", help="Cross-validation R². >0.6 is good.")
        col2.metric("CV RMSE", f"{mm.get('cv_rmse_c', 0):.2f}°C")
        col3.metric("Training cells", str(mm.get('n_cells', 0)))

        if cv_r2 < 0.3:
            st.warning("ℹ️ Model R² is low — city may have homogeneous land cover "
                       "or GEE LST coverage is sparse. SHAP attributions are still directionally valid.")

    st.divider()

    for exp in explanations:
        rank = exp.get('rank', '?')
        locality = exp.get('locality_name', 'Unknown')
        lst = exp.get('lst_celsius', 0) or 0
        anomaly = exp.get('lst_anomaly_c', 0) or 0
        category = exp.get('hotspot_category', 'Hotspot')

        with st.expander(f"🔴 Hotspot #{rank} — {locality} ({lst:.1f}°C, {anomaly:+.1f}°C)", expanded=(rank == 1)):
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.markdown(f"""
                <div class="hotspot-card">
                    <h3>{locality}</h3>
                    <div class="temp">{lst:.1f}°C</div>
                    <div class="anomaly">{anomaly:+.1f}°C above city mean</div>
                    <br>
                    <small style="color:#aaa;">{category}</small><br>
                    <small style="color:#aaa;">
                        {exp.get('centroid_lat', 0):.4f}°N, {exp.get('centroid_lon', 0):.4f}°E
                    </small>
                </div>""", unsafe_allow_html=True)

                if exp.get('top_drivers'):
                    st.markdown("**Feature vs. City Average:**")
                    for driver in exp['top_drivers']:
                        direction_emoji = "🔴" if driver['is_heating'] else "🔵"
                        st.markdown(
                            f"{direction_emoji} **{driver['display_name']}**: "
                            f"{driver['cell_value_str']} "
                            f"*(city avg: {driver['city_mean_str']})* "
                            f"→ `{driver['shap_contribution_c']:+.2f}°C`"
                        )

            with c2:
                fig = plot_shap_bar(exp)
                if fig.data:
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown(exp.get('explanation_text', ''), unsafe_allow_html=False)


def render_tab_interventions(result):
    from src.intervention_simulator import simulate_intervention, INTERVENTION_TYPES

    explanations = result['explanations']
    sim_df = result['sim_df']
    feature_df = result['feature_df']
    model = result['model']
    feat_cols = result['feat_cols_used']
    top_hotspots = result['top_hotspots']
    cs = result['city_stats']

    st.markdown("### 🌿 Cooling Intervention Simulator")
    st.caption(
        "Feature modifications feed back through the same trained city model. "
        "Every °C estimate is model-derived, not a generic rule-of-thumb. "
        "Physical assumptions are documented in `src/intervention_simulator.py`."
    )

    # Hotspot selector
    localities = [e.get('locality_name', f"Hotspot {e.get('rank')}") for e in explanations]
    selected_locality = st.selectbox("Select Hotspot", localities)

    # Find the selected explanation and cell
    selected_exp = next((e for e in explanations if e.get('locality_name') == selected_locality), None)
    if not selected_exp:
        st.warning("Hotspot not found.")
        return

    cell_id = selected_exp['cell_id']
    cell_mask = feature_df['cell_id'] == cell_id
    if not cell_mask.any():
        st.warning("Cell features not found.")
        return

    cell_features = feature_df[cell_mask].iloc[0]

    # Intervention controls
    c1, c2 = st.columns(2)
    with c1:
        intervention_type = st.selectbox(
            "Intervention Type",
            options=list(INTERVENTION_TYPES.keys()),
            format_func=lambda k: INTERVENTION_TYPES[k]['label'],
        )
    with c2:
        intensity = st.slider(
            "Intensity (1=minimal → 10=maximum)",
            min_value=1.0, max_value=10.0, value=5.0, step=0.5,
        )

    # Live simulation
    with st.spinner("Computing cooling prediction..."):
        sim = simulate_intervention(model, cell_features, feat_cols, intervention_type, intensity)

    # Display result
    delta = sim['delta_lst_c']
    lower = sim['delta_lst_lower_c']
    upper = sim['delta_lst_upper_c']

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline LST", f"{sim['baseline_lst_c']:.1f}°C")
    col2.metric("Predicted LST after intervention", f"{sim['predicted_lst_c']:.1f}°C",
                delta=f"{delta:.2f}°C", delta_color="inverse")
    col3.metric("Uncertainty Range", f"{lower:.2f}°C to {upper:.2f}°C")

    st.markdown(f"""
    <div class="intervention-card">
        <b>{INTERVENTION_TYPES[intervention_type]['label']}</b><br>
        Predicted cooling: <span style="color:#4fc3f7; font-size:1.2rem; font-weight:700;">
            {delta:.2f}°C</span> 
        (90% CI: {lower:.2f}°C to {upper:.2f}°C)<br>
        Resource required: {sim['resource_amount']:.0f} {sim['resource_unit']} 
        (at intensity {intensity:.0f}/10)<br>
        <small style="color:#aaa;">
            Uncertainty estimated from XGBoost tree-level predictions (staged prediction spread)
        </small>
    </div>""", unsafe_allow_html=True)

    # Feature deltas explanation
    if sim['feature_deltas']:
        st.markdown("**Feature changes applied to model:**")
        for feat, delta_val in sim['feature_deltas'].items():
            if abs(delta_val) > 0.001:
                st.caption(f"• `{feat}`: {delta_val:+.4f}")

    st.divider()

    # All interventions comparison chart
    st.markdown("### 📊 All Interventions Comparison")
    fig = plot_intervention_comparison(sim_df, selected_locality)
    if fig.data:
        st.plotly_chart(fig, use_container_width=True)

    # Intervention map
    st.markdown("### 🗺️ Intervention Map")
    m = build_intervention_map(top_hotspots, sim_df, cs, intervention_type)
    st_folium(m, width="100%", height=400, returned_objects=[])


def render_tab_optimizer(result):
    from src.optimizer import optimize_budget

    sim_df = result['sim_df']
    feature_df = result['feature_df']
    explanations = result['explanations']
    cs = result['city_stats']

    st.markdown("### 💰 Budget-Constrained Optimizer")
    st.caption(
        "PuLP linear programming allocates resources across hotspots to maximize "
        "population-weighted LST reduction. Change the budget to get a different allocation."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        budget_type = st.selectbox(
            "Budget Type",
            options=['saplings', 'cool_roof_m2', 'rupees'],
            format_func=lambda x: {
                'saplings': '🌳 Number of Saplings',
                'cool_roof_m2': '🏠 Cool Roof Area (m²)',
                'rupees': '💵 Budget in ₹',
            }[x],
        )
    with c2:
        budget_defaults = {'saplings': 5000, 'cool_roof_m2': 50000, 'rupees': 2000000}
        budget = st.number_input(
            f"Available Budget",
            min_value=100,
            max_value=10000000,
            value=budget_defaults.get(budget_type, 5000),
            step=500,
        )
    with c3:
        optimize_btn = st.button("🔧 Optimize Allocation", type="primary")

    if optimize_btn or 'optimizer_result' not in st.session_state:
        with st.spinner("Running PuLP optimizer..."):
            pop_weights = {
                int(row['cell_id']): float(row.get('pop_density_norm', 1.0))
                for _, row in feature_df.iterrows()
            }
            opt_result = optimize_budget(sim_df, budget, budget_type, pop_weights)
            st.session_state['optimizer_result'] = opt_result
    else:
        opt_result = st.session_state.get('optimizer_result', {})

    if not opt_result or not opt_result.get('allocations'):
        st.warning("No viable allocations found. Try increasing budget or checking hotspot data.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cooling (sum)", f"{opt_result['total_delta_lst_c']:.2f}°C")
    col2.metric("Budget Used", f"{opt_result['total_cost']:,.0f} / {budget:,.0f}")
    col3.metric("Budget Utilization", f"{opt_result['budget_utilization']*100:.0f}%")
    col4.metric("Hotspots Addressed", str(opt_result['n_hotspots_addressed']))

    st.caption(f"Solver: {opt_result['solver_status']} | "
               f"Weighted by population exposure (GHSL) + LST anomaly severity")

    # Allocation table
    st.markdown("### 📋 Recommended Allocation")
    alloc_rows = []
    for alloc in opt_result['allocations']:
        alloc_rows.append({
            'Locality': alloc['locality_name'],
            'Intervention': alloc.get('intervention_label', alloc['intervention_type']),
            'Intensity': f"{alloc.get('effective_intensity', '?')}/10",
            'Predicted Cooling (°C)': f"{alloc['delta_lst_c']:.2f}",
            'Uncertainty Range': f"{alloc.get('delta_lst_lower_c', 0):.2f} to {alloc.get('delta_lst_upper_c', 0):.2f}",
            'Cost': f"{alloc['cost']:,.0f} {alloc['resource_unit']}",
            'Pop Weight': f"{alloc.get('pop_weight', 1):.2f}",
        })

    if alloc_rows:
        alloc_df = pd.DataFrame(alloc_rows)
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    # Allocation bar chart
    if alloc_rows:
        fig = go.Figure(go.Bar(
            x=[r['Locality'] for r in alloc_rows],
            y=[float(r['Predicted Cooling (°C)']) for r in alloc_rows],
            marker=dict(color='#64b5f6', opacity=0.8),
            text=[r['Intervention'] for r in alloc_rows],
            textposition='outside',
        ))
        fig.update_layout(
            title="Predicted Cooling per Hotspot (Optimized Allocation)",
            yaxis_title="Cooling (°C)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8eaf6', family='Inter'),
            height=380,
            xaxis=dict(tickangle=-15),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_tab_trends(result):
    trend_df = result.get('trend_df')
    feature_df = result['feature_df']
    grid_gdf = result['grid_gdf']
    cs = result['city_stats']
    city_name = result['city_name']

    st.markdown("### 📈 Multi-Year LST Trend (2015 – 2023)")

    if trend_df is not None and not trend_df.empty:
        fig = plot_lst_trend(trend_df, city_name)
        st.plotly_chart(fig, use_container_width=True)

        # Year-over-year change
        yearly = trend_df.groupby('year')['lst_celsius'].mean()
        if len(yearly) >= 2:
            yoy_change = float(yearly.iloc[-1] - yearly.iloc[0])
            years_span = yearly.index[-1] - yearly.index[0]
            st.metric(
                f"Mean LST change ({yearly.index[0]}–{yearly.index[-1]})",
                f"{yoy_change:+.1f}°C",
                help=f"Change in city-mean LST over {years_span} years (peak summer Apr–Jun)"
            )
    else:
        st.info("Multi-year trend data not available (GEE may be unavailable or quota exhausted). "
                "Re-run the analysis with GEE connected to see historical trends.")

    st.divider()

    # Heat Vulnerability Index map
    st.markdown("### 🏘️ Heat Vulnerability Index")
    st.caption(
        "HVI = LST anomaly × √(population density). "
        "Identifies zones that are both hot AND densely inhabited — "
        "highest health risk, not just the hottest pixels."
    )

    m = build_hvi_map(grid_gdf, feature_df, cs)
    st_folium(m, width="100%", height=450, returned_objects=[])

    # Heat health risk
    st.divider()
    st.markdown("### ☀️ Heat-Health Risk")
    hw = cs['heatwave_class']
    hi = cs['heat_index_c']
    at = cs['air_temp_c']
    rh = cs['rh']

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(15,52,96,0.8), rgba(22,33,62,0.9));
                border: 2px solid {hw['color']}; border-radius:12px; padding:1.5rem;">
        <h3 style="color:{hw['color']}; margin:0 0 1rem;">
            IMD Heat Alert: {hw['level']}
        </h3>
        <p style="color:#e8eaf6; margin:0.3rem 0;">
            <b>Air Temperature:</b> {at:.1f}°C &nbsp;|&nbsp; 
            <b>Relative Humidity:</b> {rh:.0f}% &nbsp;|&nbsp;
            <b>Steadman Heat Index:</b> {hi:.1f}°C
        </p>
        <p style="color:rgba(232,234,246,0.8); margin:0.5rem 0 0;">{hw['description']}</p>
        <small style="color:rgba(232,234,246,0.5);">
            IMD thresholds: Heat Wave ≥ 40°C | Severe ≥ 45°C | Extreme ≥ 47°C (Heat Index)
        </small>
    </div>""", unsafe_allow_html=True)


def render_tab_report(result):
    from src.report_generator import generate_markdown_report, generate_pdf_report

    cs = result['city_stats']
    explanations = result['explanations']
    sim_df = result['sim_df']
    model_metrics = result['model_metrics']
    city_name = result['city_name']
    opt_result = st.session_state.get('optimizer_result')

    st.markdown("### 📄 Download Analysis Report")
    st.caption("Complete summary of hotspots, drivers, interventions, and optimizer results.")

    today = datetime.date.today().strftime("%Y-%m-%d")

    c1, c2 = st.columns(2)
    with c1:
        md_content = generate_markdown_report(
            city_name, today, cs, explanations, sim_df,
            opt_result, model_metrics, cs.get('lst_source', 'gee')
        )
        st.download_button(
            "📝 Download Markdown Report",
            data=md_content,
            file_name=f"urban_heat_{city_name.lower().replace(' ', '_')}_{today}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with c2:
        pdf_bytes = generate_pdf_report(
            city_name, today, cs, explanations, sim_df,
            opt_result, model_metrics, cs.get('lst_source', 'gee')
        )
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"urban_heat_{city_name.lower().replace(' ', '_')}_{today}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # Preview in app
    st.divider()
    st.markdown("**Report Preview:**")
    with st.expander("Show Markdown Preview", expanded=False):
        st.markdown(md_content)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Sidebar
    city_input, analyze_btn, force_refresh, fetch_osm = render_sidebar()

    # Hero header
    st.markdown(f"""
    <div class="hero-header">
        <h1>🌡️ Urban Heat Mitigation AI</h1>
        <p>Physics-informed geospatial ML system for Indian city heat stress analysis — ISRO Hackathon</p>
    </div>""", unsafe_allow_html=True)

    state = st.session_state.pipeline_state

    # Trigger pipeline
    if analyze_btn and city_input:
        state['status'] = 'running'
        state['progress'] = 0
        state['progress_msg'] = 'Starting...'
        state['city'] = city_input
        state['fetch_osm'] = fetch_osm
        state['result'] = None
        state['error'] = None
        st.rerun()

    # Running state
    if state['status'] == 'running':
        st.markdown(f"### 🔄 Analyzing **{state['city']}**...")
        progress_bar = st.progress(state['progress'] / 100)
        status_text = st.empty()
        status_text.info(f"⏳ {state['progress_msg'] or 'Initializing pipeline...'}")

        with st.spinner("Running analysis pipeline (takes less than a minute)..."):
            run_pipeline(state['city'], force_refresh, state.get('fetch_osm', False))
            st.rerun()
        return

    # Error state
    if state['status'] == 'error':
        st.error(f"❌ Pipeline failed: {state['error']}")
        st.markdown("""
        **Troubleshooting:**
        1. Ensure GEE is authenticated: `python -c "import ee; ee.Authenticate()"`
        2. Check your `.env` file has the correct `GEE_PROJECT_ID`
        3. Try a different city name (append ", India" if needed)
        4. Check your internet connection (OSM Overpass + Open-Meteo both required)
        """)
        return

    # Idle state
    if state['status'] == 'idle' or state['result'] is None:
        st.markdown("""
        ## Welcome to Urban Heat Mitigation AI 🛰️
        
        This system identifies urban heat stress hotspots in any Indian city using:
        - **Landsat 8/9 satellite data** for real Land Surface Temperature
        - **Physics-informed XGBoost** with monotonic constraints (more vegetation = cooler, always)
        - **Getis-Ord Gi*** spatial statistics for statistically rigorous hotspot detection
        - **SHAP explainability** to quantify *why* each zone is hot
        - **Counterfactual simulations** — every intervention's °C estimate from the model itself
        
        **👈 Type a city name and click "Analyze City" to begin.**
        
        *Tested cities: Kanpur, Surat, Bhopal, Coimbatore, Patna, Delhi, Hyderabad*
        """)

        # Quick-start buttons
        st.markdown("**Quick start:**")
        q_cols = st.columns(5)
        for col, city in zip(q_cols, ["Kanpur", "Surat", "Bhopal", "Coimbatore", "Patna"]):
            if col.button(city, use_container_width=True):
                state['status'] = 'running'
                state['progress'] = 0
                state['progress_msg'] = 'Starting...'
                state['city'] = city
                state['fetch_osm'] = False
                state['result'] = None
                state['error'] = None
                st.rerun()
        return

    # Results state — show tabs
    result = state['result']
    city_name = result['city_name']

    st.success(f"✅ Analysis complete for **{city_name}** "
               f"({result['city_stats']['n_cells']} grid cells)")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ Overview Map",
        "🔴 Hotspot Analysis",
        "🌿 Interventions",
        "💰 Budget Optimizer",
        "📈 Trends & Vulnerability",
        "📄 Download Report",
    ])

    with tab1:
        render_tab_overview(result)
    with tab2:
        render_tab_hotspots(result)
    with tab3:
        render_tab_interventions(result)
    with tab4:
        render_tab_optimizer(result)
    with tab5:
        render_tab_trends(result)
    with tab6:
        render_tab_report(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error("🚨 Critical Error inside main():")
        st.code(traceback.format_exc())
