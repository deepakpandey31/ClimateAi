# Urban Heat Mitigation AI System 🌡️🛰️

An ISRO hackathon submission that identifies urban heat stress hotspots in any Indian city, quantifies drivers using physics-informed ML, and simulates cooling interventions — all from free public APIs, zero hardcoded data.

---

## Quick Setup (under 15 minutes)

### 1. Python Environment
```bash
cd urban-heat-ai
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Google Earth Engine (FREE — required for satellite LST data)
1. Visit https://earthengine.google.com → **Sign Up**
2. Select **"Noncommercial / Research"** — no credit card needed
3. After approval, your Cloud Project ID appears at https://console.cloud.google.com (looks like `ee-yourname-12345`)
4. Run once to store credentials locally:
   ```bash
   python -c "import ee; ee.Authenticate()"
   ```
   A browser window opens — log in with your Google account and allow access.

### 3. Configure your Project ID
Copy `.env.example` to `.env` and fill in your project ID:
```
GEE_PROJECT_ID=your-cloud-project-id
```

### 4. Run the App
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## No other sign-ups required
| Service | Key required? |
|---|---|
| Nominatim (OpenStreetMap) | ❌ No |
| Google Earth Engine | ✅ Free sign-up (above) |
| ESA WorldCover via GEE | ❌ (included in GEE) |
| Open-Meteo weather | ❌ No |
| NASA POWER solar | ❌ No |
| OSM Overpass API | ❌ No |

---

## Usage
1. Type any Indian city name (Kanpur, Surat, Bhopal, Coimbatore, Patna…)
2. Hit **Analyze City** — the pipeline runs in the background (~2–5 min first run)
3. Explore hotspot cards, intervention sliders, and the budget optimizer
4. Download the PDF report for your submission

---

## Architecture
```
City name → geocode + boundary (Nominatim/OSMnx)
  → adaptive grid (1,000–4,000 cells, ~200–300 m)
  → GEE server-side: LST (Landsat 8/9) + LULC (ESA WorldCover) + NDVI (Sentinel-2) + population (GHSL)
  → OSM: buildings, roads, landuse, water bodies
  → Open-Meteo: air temp, humidity, wind
  → NASA POWER: solar radiation
  → physics-informed feature table (per cell)
  → XGBoost model with monotonic constraints + SHAP
  → Getis-Ord Gi* hotspot detection
  → Nominatim reverse geocode → real locality names
  → "Why is it hot" explanation engine
  → Counterfactual intervention simulator
  → PuLP budget optimizer
  → Streamlit + Folium dashboard
```

## Memory Budget (designed for 8 GB RAM laptops)
- No raster tiles ever downloaded — all satellite reductions happen server-side in GEE
- Grid capped at 4,000 cells max; adaptive cell size by city area
- All API responses cached to `cache/` directory
- XGBoost capped at 200 trees / depth 6 (~150–300 MB RAM peak)

## Tech Stack
100% free & open source: Python 3.10+, Streamlit, Folium, GeoPandas, OSMnx, Earth Engine API, XGBoost, SHAP, esda/libpysal, PuLP, ReportLab
