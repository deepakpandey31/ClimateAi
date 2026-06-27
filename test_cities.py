import sys
import logging
from pathlib import Path
import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)

from src.geocode import geocode_city, build_grid
from src.data_fetch_gee import fetch_lst, fetch_lulc, fetch_ndvi, fetch_ghsl, fetch_multi_year_lst, _init_gee
from src.data_fetch_osm import fetch_all_osm
from src.data_fetch_weather import fetch_openmeteo
from src.feature_engineering import build_feature_table, get_feature_columns
from src.model import train_city_model
from src.hotspot_detection import compute_lst_anomaly, getis_ord_hotspots, get_top_hotspot_clusters
from src.intervention_simulator import batch_simulate

GEE_PROJECT_ID = "smartqueue-b1647"
_init_gee(GEE_PROJECT_ID)

for city_name in ["Kanpur", "Surat"]:
    print(f"\n==============================")
    print(f"RUNNING E2E FOR {city_name}")
    print(f"==============================")
    city_info = geocode_city(city_name)
    grid_gdf = build_grid(city_info['boundary_gdf'])
    print(f"Grid: {len(grid_gdf)} cells")
    
    date_start, date_end = "2023-04-01", "2023-06-30"
    
    lst_df = fetch_lst(grid_gdf, GEE_PROJECT_ID, date_start, date_end)
    lulc_df = fetch_lulc(grid_gdf, GEE_PROJECT_ID)
    ndvi_df = fetch_ndvi(grid_gdf, GEE_PROJECT_ID, date_start, date_end)
    ghsl_df = fetch_ghsl(grid_gdf, GEE_PROJECT_ID)
    
    osm_dfs = fetch_all_osm(city_info['boundary_gdf'], grid_gdf)
    weather = fetch_openmeteo(city_info['centroid_lat'], city_info['centroid_lon'], date_start, date_end)
    
    feature_df = build_feature_table(grid_gdf, lst_df, lulc_df, ndvi_df, osm_dfs, weather, ghsl_df)
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in feature_df.columns]
    
    print(f"Validation Anomalies Capped: {feature_df.attrs.get('validation_warnings', 0)}")
    
    model, feat_cols_used, model_metrics = train_city_model(feature_df, feature_cols)
    print(f"Model Training Results:")
    print(f"  CV R2    = {model_metrics['cv_r2']:.3f}")
    print(f"  Train R2 = {model_metrics['train_r2']:.3f}")
    
    grid_lst_gdf = compute_lst_anomaly(grid_gdf, feature_df)
    hotspot_gdf = getis_ord_hotspots(grid_lst_gdf)
    top_hotspots = get_top_hotspot_clusters(hotspot_gdf, n=3)
    
    sim_df = batch_simulate(top_hotspots, feature_df, feat_cols_used, model, default_intensity=5.0)
    if not sim_df.empty:
        max_cooling = sim_df['delta_lst_c'].min()
        print(f"Intervention Simulator max cooling constraint check: {max_cooling:.2f} °C (Should be >= -4.0)")
    
    print("Testing multi-year fetch...")
    trend_df = fetch_multi_year_lst(grid_gdf, GEE_PROJECT_ID, years=[2019, 2023])
    if trend_df is not None:
        yearly = (trend_df.dropna(subset=['lst_celsius']).groupby('year')['lst_celsius']
                  .agg(max=lambda x: x.quantile(0.95), min=lambda x: x.quantile(0.05)).reset_index())
        print("Trend Data (95th/5th percentiles):")
        print(yearly)
    
    # Check OSM water distance
    avg_water_dist = feature_df['dist_water_km'].mean()
    min_water_dist = feature_df['dist_water_km'].min()
    max_water_dist = feature_df['dist_water_km'].max()
    print(f"Water Distance Stats -> Avg: {avg_water_dist:.2f} km, Min: {min_water_dist:.2f} km, Max: {max_water_dist:.2f} km")
