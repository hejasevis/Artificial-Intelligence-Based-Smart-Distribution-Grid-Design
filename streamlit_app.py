import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from shapely.geometry import LineString
from geopy.distance import geodesic
import json

# =============== Optional libs ===============
try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except Exception:
    HAS_PYPROJ = False

try:
    from scipy.spatial import cKDTree
    HAS_KDTREE = True
except Exception:
    HAS_KDTREE = False

# ML & Viz
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Yapay Zeka ile Akıllı Dağıtım Şebekesi Tasarımı", layout="wide")
st.title("🔌 Yapay Zeka ile Akıllı Dağıtım Şebekesi Tasarımı — v2")

# ===================== MENU =====================
selected = option_menu(
    menu_title="",
    options=["Talep Girdisi", "Gerilim Düşümü", "Forecasting", "Arıza/Anomali"],
    icons=["geo-alt-fill", "activity", "graph-up-arrow", "exclamation-triangle-fill"],
    menu_icon="cast",
    default_index=0,
    styles={
        "container": {"padding": "5!important", "background-color": "#f0f2f6"},
        "icon": {"color": "orange", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "2px", "font-weight": "600"},
        "nav-link-selected": {"background-color": "#fcd769", "color": "white", "font-weight": "700"},
    }
)

# ===================== HELPERS =====================
def get_transformers():
    if not HAS_PYPROJ:
        raise RuntimeError("pyproj mevcut değil")
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    bwd = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return fwd, bwd

@st.cache_data
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    return d

@st.cache_data
def load_default_data():
    """Try to read user files from working dir. Falls back to uploaders if not found."""
    direk_df = None
    trafo_df = None
    ext_df = None
    try:
        direk_df = pd.read_excel("Direk Sorgu Sonuçları.xlsx")
        direk_df = norm_cols(direk_df)
    except Exception:
        pass
    try:
        trafo_df = pd.read_excel("Trafo Sorgu Sonuçları.xlsx")
        trafo_df = norm_cols(trafo_df)
    except Exception:
        pass
    try:
        ext_df = pd.read_csv("smart_grid_dataset.csv")
        ext_df = norm_cols(ext_df)
    except Exception:
        pass
    return direk_df, trafo_df, ext_df

# Electrical model (approx 3-phase)
def vdrop_percent(P_kw, L_m, Vn_kV=0.4, cosphi=0.9, R_ohm_km=0.642, X_ohm_km=0.083):
    try:
        L_km = float(L_m) / 1000.0
        V = float(Vn_kV) * 1e3
        P = float(P_kw) * 1e3
        I = P / (np.sqrt(3) * V * float(cosphi))
        # sinφ from cosφ
        sinphi = np.sqrt(max(0.0, 1.0 - float(cosphi) ** 2))
        Z_proj = float(R_ohm_km) * L_km * float(cosphi) + float(X_ohm_km) * L_km * sinphi
        dV = 100.0 * (np.sqrt(3) * I * Z_proj) / V
        return float(dV)
    except Exception:
        return np.nan

# Build KDTree for snapping
@st.cache_data
def build_kdtree(points_xy):
    if not HAS_KDTREE:
        return None
    arr = np.array(points_xy)
    if len(arr) == 0:
        return None
    return cKDTree(arr)

# Route builder: interpolate points along straight line and snap to nearest poles

def dedup_seq(seq):
    out = []
    for p in seq:
        if not out or (p[0] != out[-1][0] or p[1] != out[-1][1]):
            out.append(p)
    return out


def build_route_and_stats(demand_latlon, trafo_latlon, poles_latlon, max_span=40.0, snap_radius=30.0):
    """Returns: route_latlon, total_len_m, used_count, proposed_count, spans_m"""
    try:
        fwd, bwd = get_transformers()
        to_xy = lambda lon, lat: fwd.transform(lon, lat)
        to_lonlat = lambda x, y: bwd.transform(x, y)

        demand_xy = to_xy(demand_latlon[1], demand_latlon[0])
        trafo_xy = to_xy(trafo_latlon[1], trafo_latlon[0])
        line_xy = LineString([demand_xy, trafo_xy])

        poles_xy = [to_xy(lon, lat) for (lat, lon) in poles_latlon]
        tree = build_kdtree(poles_xy)

        length = line_xy.length
        distances = list(np.arange(0, length, float(max_span))) + [length]
        pts = [line_xy.interpolate(d) for d in distances]

        used_idx = set()
        route_xy = []
        for p in pts:
            px, py = p.x, p.y
            snapped = False
            if tree is not None:
                dist, idx = tree.query([px, py], k=1)
                if dist <= float(snap_radius):
                    route_xy.append(tuple(poles_xy[idx])); used_idx.add(int(idx)); snapped = True
            if not snapped:
                route_xy.append((px, py))

        if route_xy:
            route_xy[0] = demand_xy
            route_xy[-1] = trafo_xy

        route_xy = dedup_seq(route_xy)
        spans = [LineString(route_xy[i:i+2]).length for i in range(len(route_xy)-1)]
        total_len_m = sum(spans)
        used_count = len(used_idx)
        proposed_count = max(0, len(route_xy) - used_count - 2)

        final_path = [(to_lonlat(x, y)[1], to_lonlat(x, y)[0]) for (x, y) in route_xy]
        return final_path, total_len_m, used_count, proposed_count, spans

    except Exception:
        # fallback geodesic straight line
        total_len_m = geodesic(demand_latlon, trafo_latlon).meters
        final_path = [demand_latlon, trafo_latlon]
        return final_path, total_len_m, 0, 1, [total_len_m]

# ===================== DATA =====================
# GitHub yükleme desteği
import io, requests

@st.cache_data(show_spinner=False, ttl=600)
def fetch_github_file(raw_url: str, headers: dict | None = None) -> bytes:
    if not raw_url:
        raise ValueError("raw_url boş")
    headers = headers or {}
    r = requests.get(raw_url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.content

@st.cache_data
def load_from_github(direk_url: str | None, trafo_url: str | None, ext_url: str | None, token: str | None = None):
    headers = {"Authorization": f"token {token}"} if token else None
    direk_df = trafo_df = ext_df = None
    try:
        if direk_url:
            raw = fetch_github_file(direk_url, headers)
            direk_df = pd.read_excel(io.BytesIO(raw))
            direk_df = norm_cols(direk_df)
    except Exception as e:
        st.warning(f"Direk GitHub yüklemesi başarısız: {e}")
    try:
        if trafo_url:
            raw = fetch_github_file(trafo_url, headers)
            trafo_df = pd.read_excel(io.BytesIO(raw))
            trafo_df = norm_cols(trafo_df)
    except Exception as e:
        st.warning(f"Trafo GitHub yüklemesi başarısız: {e}")
    try:
        if ext_url:
            raw = fetch_github_file(ext_url, headers)
            # CSV/Parquet autodetect by extension
            if ext_url.lower().endswith('.parquet'):
                import pyarrow as pa, pyarrow.parquet as pq  # noqa: F401
                ext_df = pd.read_parquet(io.BytesIO(raw))
            else:
                ext_df = pd.read_csv(io.BytesIO(raw))
            ext_df = norm_cols(ext_df)
    except Exception as e:
        st.warning(f"Ek veri GitHub yüklemesi başarısız: {e}")
    return direk_df, trafo_df, ext_df


direk_df, trafo_df, ext_df = load_default_data()

with st.sidebar.expander("📂 Veri Kaynakları"):
    st.write("Varsayılan: çalışma dizinindeki dosyalar. Alternatif olarak **GitHub raw URL** girin veya dosya yükleyin.")

    # GitHub Inputs
    st.markdown("**GitHub Raw URL'leri**")
    gh_direk = st.text_input("Direk (XLSX) Raw URL", placeholder="https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/Direk.xlsx")
    gh_trafo = st.text_input("Trafo (XLSX) Raw URL", placeholder="https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/Trafo.xlsx")
    gh_ext = st.text_input("Kaggle/Ek (CSV/Parquet) Raw URL", placeholder="https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/data.csv")
    gh_token = st.secrets.get("GITHUB_TOKEN", None) if hasattr(st, 'secrets') else None
    colb1, colb2 = st.columns([1,1])
    with colb1:
        use_github = st.checkbox("GitHub'dan yükle", value=False)
    with colb2:
        if st.button("🔄 GitHub verisini çek"):
            st.cache_data.clear()

    # File uploaders
    up_direk = st.file_uploader("veya Direk verisi (XLSX) yükle", type=["xlsx"], key="upl_direk")
    up_trafo = st.file_uploader("veya Trafo verisi (XLSX) yükle", type=["xlsx"], key="upl_trafo")
    up_ext = st.file_uploader("veya Kaggle/ek veri (CSV/Parquet)", type=["csv","parquet"], key="upl_ext")

    if use_github and (gh_direk or gh_trafo or gh_ext):
        g_direk, g_trafo, g_ext = load_from_github(gh_direk or None, gh_trafo or None, gh_ext or None, gh_token)
        direk_df = g_direk or direk_df
        trafo_df = g_trafo or trafo_df
        ext_df = g_ext or ext_df

    if up_direk is not None:
        direk_df = norm_cols(pd.read_excel(up_direk))
    if up_trafo is not None:
        trafo_df = norm_cols(pd.read_excel(up_trafo))
    if up_ext is not None:
        if up_ext.name.lower().endswith('.parquet'):
            ext_df = pd.read_parquet(up_ext)
        else:
            ext_df = norm_cols(pd.read_csv(up_ext))

# ===================== PAGE 1: Talep Girdisi =====================
if selected == "Talep Girdisi":
    if (direk_clean is None) or (trafo_clean is None):
        st.error("Direk ve trafo verileri olmadan bu sayfa çalışmaz. Sol menüden dosyaları yükleyin.")
        st.stop()

    st.sidebar.header("⚙️ Hat Parametreleri")
    max_span = st.sidebar.number_input("Maks. direk aralığı (m)", 20, 120, 40, 5)
    snap_radius = st.sidebar.number_input("Mevcut direğe snap yarıçapı (m)", 5, 120, 30, 5)

    with st.sidebar.expander("🔌 Elektrik Parametreleri"):
        Vn_kV = st.number_input("Nominal gerilim (kV)", 0.4, 34.5, 0.4, 0.1)
        pf = st.number_input("Güç faktörü (pf)", 0.5, 1.0, 0.9, 0.05)
        R_ohm_km = st.number_input("R (Ω/km)", 0.05, 1.5, 0.642, 0.01)
        X_ohm_km = st.number_input("X (Ω/km)", 0.01, 1.0, 0.083, 0.01)
        drop_threshold_pct = st.number_input("Gerilim düşümü eşiği (%)", 1.0, 15.0, 5.0, 0.5)

    # --- Map for demand selection ---
    st.subheader("📍 Talep Noktasını Seçin (Harita)")
    center_lat = float(direk_clean['lat'].mean())
    center_lon = float(direk_clean['lon'].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16, control_scale=True)

    poles_group = folium.FeatureGroup(name="Direkler (Mevcut)", show=True)
    trafos_group = folium.FeatureGroup(name="Trafolar", show=True)

    for _, r in direk_clean.iterrows():
        folium.CircleMarker([r['lat'], r['lon']], radius=4, color="blue",
                            fill=True, fill_opacity=0.7,
                            tooltip=f"Direk: {r.get('pole_code','-')}",
                            popup=f"AssetID: {r.get('asset_id','-')}").add_to(poles_group)

    for _, r in trafo_clean.iterrows():
        folium.Marker([r['lat'], r['lon']],
                      tooltip=f"Trafo: {r.get('name','-')}",
                      popup=f"Güç: {r.get('kva','?')} kVA\nAssetID: {r.get('asset_id','-')}",
                      icon=folium.Icon(color="orange", icon="bolt", prefix="fa")).add_to(trafos_group)

    poles_group.add_to(m); trafos_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=600, width="100%", returned_objects=["last_clicked"], key="select_map_v2")

    if "demand_point" not in st.session_state:
        st.session_state["demand_point"] = None
    if map_data and map_data.get("last_clicked"):
        st.session_state["demand_point"] = (
            float(map_data["last_clicked"]["lat"]),
            float(map_data["last_clicked"]["lng"]),
        )
    if st.session_state["demand_point"] is None:
        st.info("📍 Haritadan bir noktaya tıkla."); st.stop()

    new_lat, new_lon = st.session_state["demand_point"]
    st.success(f"Yeni talep noktası: ({new_lat:.6f}, {new_lon:.6f})")

    st.subheader("⚡ Talep Edilen Yük (kW)")
    user_kw = st.slider("Talep edilen güç", 1, 500, 120, 5, key="kw_slider_v2")

    # --- Choose best trafo by route-based ΔV (top-5 evaluated) ---
    def eval_trafo(row):
        t_latlon = (float(row['lat']), float(row['lon']))
        poles_latlon = list(zip(direk_clean['lat'].astype(float), direk_clean['lon'].astype(float)))
        route, Lm, used, prop, spans = build_route_and_stats((new_lat,new_lon), t_latlon, poles_latlon,
                                                             max_span=max_span, snap_radius=snap_radius)
        dv = vdrop_percent(user_kw, Lm, Vn_kV, pf, R_ohm_km, X_ohm_km)
        cap_ok = False
        try:
            kva = float(row.get('kva', np.nan))
            cap_ok = (kva * pf) >= user_kw
        except Exception:
            pass
        return {
            'name': row.get('name','-'), 'kva': row.get('kva', None), 'lat': t_latlon[0], 'lon': t_latlon[1],
            'route': route, 'L_m': Lm, 'dv_pct': dv, 'cap_ok': cap_ok, 'used_cnt': used, 'new_cnt': prop
        }

    # Evaluate top-N nearest by geodesic first for speed
    trafo_local = trafo_clean.copy()
    trafo_local['geo_dist'] = trafo_local.apply(lambda r: geodesic((new_lat,new_lon),(float(r['lat']),float(r['lon']))).meters, axis=1)
    topN = trafo_local.sort_values('geo_dist').head(8)
    evals = [eval_trafo(r) for _, r in topN.iterrows()]
    cand_df = pd.DataFrame(evals).sort_values(by=['cap_ok','dv_pct','new_cnt','L_m'], ascending=[False, True, True, True])

    with st.expander("📈 En Uygun Trafo Adayları (rota ve ΔV ile)"):
        st.dataframe(cand_df[['name','kva','L_m','dv_pct','cap_ok','new_cnt']].rename(columns={
            'name':'Montaj Yeri','kva':'Gücü[kVA]','L_m':'Rota (m)','dv_pct':'ΔV (%)','cap_ok':'Kapasite Uygun','new_cnt':'Yeni Direk'
        }), use_container_width=True)

    best = cand_df.iloc[0]

    # --- Draw result map ---
    m2 = folium.Map(location=[new_lat, new_lon], zoom_start=16, control_scale=True)

    # Existing poles (blue)
    for _, r in direk_clean.iterrows():
        folium.CircleMarker([r['lat'], r['lon']], radius=4, color="blue",
                            fill=True, fill_opacity=0.7,
                            tooltip=f"Direk: {r.get('pole_code','-')}").add_to(m2)

    # Trafo markers (orange)
    for _, r in trafo_clean.iterrows():
        folium.Marker([r['lat'], r['lon']],
                      tooltip=f"Trafo: {r.get('name','-')}",
                      icon=folium.Icon(color="orange", icon="bolt", prefix="fa")).add_to(m2)

    # Demand & chosen trafo
    folium.Marker((new_lat, new_lon), icon=folium.Icon(color="red"), tooltip="Talep Noktası").add_to(m2)
    folium.Marker((best['lat'], best['lon']), icon=folium.Icon(color="orange", icon="bolt", prefix="fa"), tooltip="Seçilen Trafo").add_to(m2)

    # Route polyline
    if len(best['route']) >= 2:
        # Identify snapped vs new points for coloring
        try:
            fwd, bwd = get_transformers()
            to_xy = lambda lon, lat: fwd.transform(lon, lat)
            poles_xy = [to_xy(lon, lat) for (lat, lon) in zip(direk_clean['lat'], direk_clean['lon'])]
            snapped_set = set(poles_xy)
            # rebuild XY for route
            route_xy = [to_xy(lon, lat) for (lat, lon) in best['route']]
            for (lat, lon), (x,y) in zip(best['route'], route_xy):
                if (x,y) in snapped_set:
                    folium.CircleMarker((lat, lon), radius=5, color="blue", fill=True, fill_opacity=0.9, tooltip="Mevcut Direk (rota)").add_to(m2)
                else:
                    folium.CircleMarker((lat, lon), radius=5, color="purple", fill=True, fill_opacity=0.9, tooltip="Önerilen Yeni Direk").add_to(m2)
        except Exception:
            pass

        folium.PolyLine(best['route'], color="green", weight=4, opacity=0.9,
                        tooltip=f"Hat uzunluğu ≈ {best['L_m']:.1f} m").add_to(m2)
    else:
        st.warning("Hat noktaları üretilemedi.")

    st.subheader("🧾 Hat Özeti")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Uzunluk", f"{best['L_m']:.1f} m")
    c2.metric("Kullanılan Mevcut Direk", f"{int(best['used_cnt'])}")
    c3.metric("Önerilen Yeni Direk", f"{int(best['new_cnt'])}")
    avg_span = (np.mean(best['L_m']/max(1,len(best['route'])-1)) if best['L_m']>0 else 0)
    c4.metric("Ortalama Direk Aralığı", f"{avg_span:.1f} m")

    if best['dv_pct'] > drop_threshold_pct:
        st.error(f"⚠️ Gerilim düşümü %{best['dv_pct']:.2f} — eşik %{drop_threshold_pct:.1f} üstü.")
    else:
        st.success(f"✅ ΔV %{best['dv_pct']:.2f} ≤ %{drop_threshold_pct:.1f}")

    st.subheader("📡 Oluşturulan Şebeke Hattı")
    st_folium(m2, height=620, width="100%", key="result_map_v2")

    # Download GeoJSON export
    try:
        gj = {
            "type":"FeatureCollection",
            "features":[
                {"type":"Feature","geometry":{"type":"LineString","coordinates":[[lon,lat] for (lat,lon) in best['route']]},
                 "properties":{"L_m":best['L_m'],"dv_pct":best['dv_pct']}}
            ]
        }
        st.download_button("⬇️ GeoJSON (Rota)", data=json.dumps(gj), mime="application/geo+json", file_name="rota.geojson")
    except Exception:
        pass

# ===================== PAGE 2: Gerilim Düşümü (Synthetic) =====================
elif selected == "Gerilim Düşümü":
    st.subheader("📉 Gerilim Düşümü — Sentetik Senaryo Oyun Alanı")
    st.caption("Parametrelerle oynayarak ΔV davranışını görselleştirin. Plotly grafikleri interaktiftir.")

    with st.sidebar.expander("🔌 Parametreler"):
        Vn_kV = st.number_input("Nominal gerilim (kV)", 0.4, 34.5, 0.4, 0.1, key="gd_vn")
        pf = st.number_input("Güç faktörü (pf)", 0.5, 1.0, 0.9, 0.05, key="gd_pf")
        R_ohm_km = st.number_input("R (Ω/km)", 0.05, 1.5, 0.642, 0.01, key="gd_R")
        X_ohm_km = st.number_input("X (Ω/km)", 0.01, 1.0, 0.083, 0.01, key="gd_X")

    # Synthetic grid
    Ls = np.linspace(10, 2000, 120)  # meters
    loads = np.linspace(5, 400, 120)  # kW
    mesh = [(L, P) for L in Ls for P in loads]
    df = pd.DataFrame(mesh, columns=["L_m","P_kw"])
    df["dv_pct"] = df.apply(lambda r: vdrop_percent(r.P_kw, r.L_m, Vn_kV, pf, R_ohm_km, X_ohm_km), axis=1)

    # Heatmap
    fig_hm = px.density_heatmap(df, x="L_m", y="P_kw", z="dv_pct", nbinsx=40, nbinsy=40, histfunc="avg",
                                title="ΔV (%) Isı Haritası (L vs P)", template="plotly_white")
    fig_hm.update_layout(xaxis_title="Hat Uzunluğu (m)", yaxis_title="Yük (kW)")
    st.plotly_chart(fig_hm, use_container_width=True)

    # Isoline-like line slices
    sel_load = st.slider("Kesit için Yük (kW)", 5, 400, 120, 5)
    df_slice = df[df.P_kw==sel_load]
    fig_ln = px.line(df_slice, x="L_m", y="dv_pct", markers=True, template="plotly_white",
                     title=f"ΔV (%) — Sabit Yük: {sel_load} kW")
    fig_ln.update_layout(xaxis_title="Hat Uzunluğu (m)", yaxis_title="ΔV (%)")
    st.plotly_chart(fig_ln, use_container_width=True)

# ===================== PAGE 3: Forecasting =====================
elif selected == "Forecasting":
    st.subheader("📈 Yük Tahmini (Forecasting)")
    st.caption("Demo: sentetik zaman serisi + basit modeller. Gerçek sayaç/hat verinizle değiştirin.")

    # Build synthetic daily load with weekly seasonality + trend
    rng = np.random.default_rng(7)
    days = pd.date_range("2024-01-01", periods=365, freq="D")
    base = 300 + 0.1*np.arange(len(days))
    weekly = 40*np.sin(2*np.pi*days.dayofweek/7)
    noise = rng.normal(0, 15, len(days))
    y = base + weekly + noise
    ts = pd.DataFrame({"ds": days, "y": y})

    # Try statsmodels Holt-Winters; fallback to rolling mean
    yhat = None
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(ts.y, trend='add', seasonal='add', seasonal_periods=7)
        fit = model.fit(optimized=True)
        fut = fit.forecast(60)
        fc = pd.DataFrame({"ds": pd.date_range(days[-1] + pd.Timedelta(days=1), periods=60), "yhat": fut.values})
        yhat = fc
    except Exception:
        roll = ts.y.rolling(7, min_periods=1).mean()
        fut = np.repeat(roll.iloc[-1], 60)
        yhat = pd.DataFrame({"ds": pd.date_range(days[-1] + pd.Timedelta(days=1), periods=60), "yhat": fut})

    fig_fc = px.line(title="Günlük Yük — Geçmiş ve Tahmin", template="plotly_white")
    fig_fc.add_scatter(x=ts.ds, y=ts.y, mode="lines", name="Geçmiş")
    fig_fc.add_scatter(x=yhat.ds, y=yhat.yhat, mode="lines", name="Tahmin")
    fig_fc.update_layout(xaxis_title="Tarih", yaxis_title="kW")
    st.plotly_chart(fig_fc, use_container_width=True)

    st.info("Gerçek kullanım: sayaç/trafo günlük/yarım saatlik verinizi yükleyin, bu sayfada modele sokalım. ARIMA/Prophet opsiyonel eklenebilir.")

# ===================== PAGE 4: Arıza/Anomali =====================
elif selected == "Arıza/Anomali":
    st.subheader("🚨 Arıza & Anomali Tespiti — Demo")
    st.caption("IsolationForest ile akım/gerilim/kW sentetik veride anomali işaretleme.")

    rng = np.random.default_rng(42)
    n = 800
    df = pd.DataFrame({
        'V': rng.normal(230, 3.0, n),
        'I': rng.normal(40, 5.0, n),
        'P': lambda d: d['V']*d['I']/1000.0
    })
    df['P'] = df['V'] * df['I'] / 1000.0

    # Inject anomalies
    idx = rng.choice(n, size=20, replace=False)
    df.loc[idx, 'V'] += rng.normal(-30, 6, len(idx))
    df.loc[idx, 'I'] += rng.normal(35, 8, len(idx))

    iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=7)
    preds = iso.fit_predict(df[['V','I','P']])
    df['anomaly'] = (preds == -1).astype(int)

    fig_sc = px.scatter(df, x='I', y='V', color=df['anomaly'].map({0:'Normal',1:'Anomali'}),
                        title='Akım-Volt Çaprazı — Anomaliler', template='plotly_white')
    st.plotly_chart(fig_sc, use_container_width=True)

    rate = df['anomaly'].mean()*100
    st.metric("Anomali Oranı", f"%{rate:.2f}")

    with st.expander("Aykırı Nokta Tablosu"):
        st.dataframe(df[df['anomaly']==1].head(50), use_container_width=True)

    st.info("Gerçek sahada: olay logları + SCADA ölçümleri + röle kayıtları ile eğitilmiş bir model önerilir. Özellik zenginleştirme (harmonikler, THD, sıfır-seq.), zaman bağlamı ve lokasyon ilişkisi (GNN) eklenebilir.")
