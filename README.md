# Smart Distribution Grid – Demo App (Streamlit)

This repository contains a **Streamlit Project** for an **AI-Assisted Smart Distribution Grid Design**. It visualizes **transformer–pole–customer** relations on a map, builds routes, computes a simplified **voltage drop score**, and showcases **anomaly detection / forecasting** with lightweight ML models.

## Features
- 🗺️ **Map & Routing:** Route from demand point to a selected transformer; distinguish **existing** vs **proposed** poles along the route.
- ⚡ **Voltage Drop Score:** Simplified **k·L·N** formula per route.
- 🤖 **ML Predictions (Demo):** RandomForest / LightGBM for quick predictions (can run on synthetic data when real data is absent).
- 🔍 **Anomaly Detection:** Rolling window logic + user controls (test window, contamination, rolling length) with summary metrics.
- 🧭 **Parameter Controls:** Aggregation, holdout length, anomaly ratio, rolling window, etc.
- 🧩 **Modular Functions:** Single source of truth for `vdrop_kLN`, separate modules for routing/distance/outputs.

## Tech Stack
- **Python 3.10+**
- **Streamlit**, **streamlit-folium**, **folium**
- **pandas**, **numpy**, **scikit-learn**, **lightgbm**
- **shapely**, **geopy**, **pyproj**
- (Optional) **prophet**, **openpyxl**, **plotly**

## Architecture / How it works
- **Map:** Produced with `folium` and embedded via `streamlit-folium`.
- **Route:** Built from demand point → selected transformer; total length is computed.
- **Voltage Score:** `vdrop_kLN(k, L, N)` where L = route length, N = pole/span count. *k* bundles material/cross-section/resistivity/power factor.
- **Snapping:** Match route vertices to **existing poles** using a tolerance (avoid exact float equality checks).
- **ML/Anomaly:** Train RF/LGBM on synthetic or real data; run rolling window anomalies with user parameters.
