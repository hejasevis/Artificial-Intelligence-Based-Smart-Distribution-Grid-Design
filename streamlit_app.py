# app.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from shapely.geometry import LineString
from geopy.distance import geodesic
import plotly.express as px
from sklearn.ensemble import IsolationForest

# --- Opsiyonel hızlandırıcılar (varsa kullanılır) ---
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

# Yol rotası için (opsiyonel)
try:
    import osmnx as ox
    import networkx as nx
    HAS_OSMNX = True
    ox.settings.use_cache = True
except Exception:
    HAS_OSMNX = False

# ===================== GENEL =====================
st.set_page_config(page_title="Yapay Zeka ile Akıllı Dağıtım Şebekesi Tasarımı", layout="wide")
st.title("🔌 Yapay Zeka ile Akıllı Dağıtım Şebekesi Tasarımı")

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

# ===================== VERİ =====================
@st.cache_data
def load_data():
    direk_df = pd.read_excel("Direk Sorgu Sonuçları.xlsx")
    trafo_df = pd.read_excel("Trafo Sorgu Sonuçları.xlsx")
    ext_df   = pd.read_csv("smart_grid_dataset.csv")  # diğer sayfalarda opsiyonel
    return direk_df, trafo_df, ext_df

direk_df, trafo_df, ext_df = load_data()

# Beklenen kolonları hızlıca çek
direk_clean = (
    direk_df[["AssetID", "Direk Kodu", "Enlem", "Boylam"]]
    .dropna(subset=["Enlem", "Boylam"])
    .copy()
)
trafo_clean = (
    trafo_df[["AssetID", "Montaj Yeri", "Gücü[kVA]", "Enlem", "Boylam"]]
    .dropna(subset=["Enlem", "Boylam"])
    .copy()
)

# ===================== YARDIMCI =====================
def get_transformers():
    if not HAS_PYPROJ:
        raise RuntimeError("pyproj mevcut değil")
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    bwd = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return fwd, bwd

def vdrop_percent(P_kw, L_m, Vn_kV=0.4, cosphi=0.9, R_ohm_km=0.642, X_ohm_km=0.083):
    """3-faz yaklaşık gerilim düşümü (%)"""
    try:
        L_km = float(L_m) / 1000.0
        V = float(Vn_kV) * 1e3
        P = float(P_kw) * 1e3
        I = P / (np.sqrt(3) * V * float(cosphi))
        sinphi = np.sqrt(max(0.0, 1.0 - float(cosphi) ** 2))
        Z_proj = float(R_ohm_km) * L_km * float(cosphi) + float(X_ohm_km) * L_km * sinphi
        dV = 100.0 * (np.sqrt(3) * I * Z_proj) / V
        return float(dV)
    except Exception:
        return np.nan

def dedup_seq(seq):
    out = []
    for p in seq:
        if not out or (p[0] != out[-1][0] or p[1] != out[-1][1]):
            out.append(p)
    return out

def build_kdtree(points_xy):
    if not HAS_KDTREE:
        return None
    arr = np.array(points_xy)
    if len(arr) == 0:
        return None
    return cKDTree(arr)

def build_route_and_stats(demand_latlon, trafo_latlon, poles_latlon, max_span=40.0, snap_radius=30.0):
    """
    (OSM yoksa) düz hat üzerinde örnekle + KD-Tree snap
    Returns: route_latlon, total_len_m, used_count, proposed_count, spans_m
    """
    try:
        fwd, bwd = get_transformers()
        to_xy = lambda lon, lat: fwd.transform(lon, lat)
        to_lonlat = lambda x, y: bwd.transform(x, y)

        demand_xy = to_xy(demand_latlon[1], demand_latlon[0])
        trafo_xy  = to_xy(trafo_latlon[1], trafo_latlon[0])
        line_xy   = LineString([demand_xy, trafo_xy])

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
            route_xy[0]  = demand_xy
            route_xy[-1] = trafo_xy

        route_xy = dedup_seq(route_xy)
        spans = [LineString(route_xy[i:i+2]).length for i in range(len(route_xy)-1)]
        total_len_m   = sum(spans)
        used_count    = len(used_idx)
        proposed_count= max(0, len(route_xy) - used_count - 2)

        final_path = [(to_lonlat(x, y)[1], to_lonlat(x, y)[0]) for (x, y) in route_xy]
        return final_path, total_len_m, used_count, proposed_count, spans

    except Exception:
        # fallback: düz geodesic
        total_len_m = geodesic(demand_latlon, trafo_latlon).meters
        final_path = [demand_latlon, trafo_latlon]
        return final_path, total_len_m, 0, 1, [total_len_m]

# ---------- OSM Yol Rotalama ----------
def _shortest_road_route(demand_latlon, trafo_latlon):
    """OSM yol grafiği üzerinde en kısa rota (lat,lon)"""
    if not HAS_OSMNX:
        raise RuntimeError("osmnx yok")
    (lat1, lon1), (lat2, lon2) = demand_latlon, trafo_latlon
    min_lat, max_lat = min(lat1, lat2), max(lat1, lat2)
    min_lon, max_lon = min(lon1, lon2), max(lon1, lon2)
    expand = 0.01  # ~1 km civarı güvenlik payı
    north, south = max_lat + expand, min_lat - expand
    east, west   = max_lon + expand, min_lon - expand
    G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    orig = ox.distance.nearest_nodes(G, lon1, lat1)
    dest = ox.distance.nearest_nodes(G, lon2, lat2)
    path = nx.shortest_path(G, orig, dest, weight="length")
    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]  # (lat, lon)
    return coords

def _polyline_length_m(coords):
    L = 0.0
    for i in range(len(coords)-1):
        L += geodesic(coords[i], coords[i+1]).meters
    return L

def _sample_along_polyline(coords, step_m):
    """Polylineda her step_m metre için nokta üret (lat,lon)."""
    if len(coords) < 2:
        return coords[:]
    out = [coords[0]]
    carry = 0.0
    for i in range(len(coords)-1):
        a, b = coords[i], coords[i+1]
        seg = geodesic(a, b).meters
        if seg == 0:
            continue
        dist_along = carry
        while dist_along + step_m <= seg:
            t = (dist_along + step_m) / seg
            lat = a[0] + t * (b[0] - a[0])
            lon = a[1] + t * (b[1] - a[1])
            out.append((lat, lon))
            dist_along += step_m
        carry = seg - dist_along
    if out[-1] != coords[-1]:
        out.append(coords[-1])
    return out

def build_route_via_roads(demand_latlon, trafo_latlon, poles_latlon, max_span=40.0, snap_radius=30.0):
    """
    Yol (OSM) üzerinden rota + direk yerleştirme.
    Döndürür: route_latlon, total_len_m, used_count, proposed_count
    """
    road_line = _shortest_road_route(demand_latlon, trafo_latlon)
    total_len_m = _polyline_length_m(road_line)
    samples = _sample_along_polyline(road_line, max_span)

    # Mevcut direklere snap (geodesic)
    used = 0
    proposed = 0
    route_pts = []
    for s in samples:
        best = None
        best_d = None
        for p in poles_latlon:
            d = geodesic(s, p).meters
            if best_d is None or d < best_d:
                best_d = d; best = p
        if best_d is not None and best_d <= float(snap_radius):
            route_pts.append(best); used += 1
        else:
            route_pts.append(s); proposed += 1

    if route_pts:
        route_pts[0] = demand_latlon
        route_pts[-1] = trafo_latlon

    route_pts = dedup_seq(route_pts)
    return route_pts, total_len_m, used, max(0, proposed - 2)

# ===================== SAYFA 1: Talep Girdisi =====================
if selected == "Talep Girdisi":
    st.sidebar.header("⚙️ Hat Parametreleri")
    max_span     = st.sidebar.number_input("Maks. direk aralığı (m)", 20, 120, 40, 5)
    snap_radius  = st.sidebar.number_input("Mevcut direğe snap yarıçapı (m)", 5, 120, 30, 5)
    use_roads    = st.sidebar.checkbox("Yolu Takip Et (OSM ile)", value=True if HAS_OSMNX else False,
                                       help="OSM varsa rota yollardan geçer; yoksa doğrusal yöntem kullanılır.")

    with st.sidebar.expander("🔌 Elektrik Parametreleri"):
        Vn_kV   = st.number_input("Nominal gerilim (kV)", 0.4, 34.5, 0.4, 0.1)
        pf      = st.number_input("Güç faktörü (pf)", 0.5, 1.0, 0.9, 0.05)
        R_ohm_km= st.number_input("R (Ω/km)", 0.05, 1.5, 0.642, 0.01)
        X_ohm_km= st.number_input("X (Ω/km)", 0.01, 1.0, 0.083, 0.01)
        drop_threshold_pct = st.number_input("Gerilim düşümü eşiği (%)", 1.0, 15.0, 5.0, 0.5)

    st.subheader("📍 Talep Noktasını Seçin (Harita)")
    center_lat = float(direk_clean["Enlem"].mean())
    center_lon = float(direk_clean["Boylam"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16, control_scale=True)

    poles_group = folium.FeatureGroup(name="Direkler (Mevcut)", show=True)
    trafos_group = folium.FeatureGroup(name="Trafolar", show=True)

    for _, r in direk_clean.iterrows():
        folium.CircleMarker([r["Enlem"], r["Boylam"]], radius=4, color="blue",
                            fill=True, fill_opacity=0.7,
                            tooltip=f"Direk: {r['Direk Kodu']}",
                            popup=f"AssetID: {r['AssetID']}").add_to(poles_group)

    for _, r in trafo_clean.iterrows():
        folium.Marker([r["Enlem"], r["Boylam"]],
                      tooltip=f"Trafo: {r['Montaj Yeri']}",
                      popup=f"Güç: {r['Gücü[kVA]']} kVA\nAssetID: {r['AssetID']}",
                      icon=folium.Icon(color="orange", icon="bolt", prefix="fa")).add_to(trafos_group)

    poles_group.add_to(m); trafos_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=620, width="100%", returned_objects=["last_clicked"], key="select_map_basic")

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
    user_kw = st.slider("Talep edilen güç", 1, 500, 120, 5, key="kw_slider_basic")

    # Trafo değerlendirme
    def eval_trafo(row):
        t_latlon = (float(row["Enlem"]), float(row["Boylam"]))
        poles_latlon = list(zip(direk_clean["Enlem"].astype(float), direk_clean["Boylam"].astype(float)))

        if use_roads and HAS_OSMNX:
            route, Lm, used, prop = build_route_via_roads(
                (new_lat, new_lon), t_latlon, poles_latlon,
                max_span=max_span, snap_radius=snap_radius
            )
        else:
            route, Lm, used, prop, _ = build_route_and_stats(
                (new_lat, new_lon), t_latlon, poles_latlon,
                max_span=max_span, snap_radius=snap_radius
            )

        dv = vdrop_percent(user_kw, Lm, Vn_kV, pf, R_ohm_km, X_ohm_km)
        cap_ok = False
        try:
            kva = float(row["Gücü[kVA]"])
            cap_ok = (kva * pf) >= user_kw
        except Exception:
            pass
        return {
            "Montaj Yeri": row["Montaj Yeri"],
            "Gücü[kVA]": row["Gücü[kVA]"],
            "lat": t_latlon[0], "lon": t_latlon[1],
            "route": route, "L_m": Lm, "ΔV (%)": dv, "Kapasite Uygun": cap_ok,
            "Kullanılan Direk": used, "Yeni Direk": prop
        }

    trafo_local = trafo_clean.copy()
    trafo_local["geo_dist"] = trafo_local.apply(
        lambda r: geodesic((new_lat, new_lon), (float(r["Enlem"]), float(r["Boylam"]))).meters, axis=1
    )
    topN = trafo_local.sort_values("geo_dist").head(8)
    evals = [eval_trafo(r) for _, r in topN.iterrows()]
    cand_df = pd.DataFrame(evals).sort_values(by=["Kapasite Uygun", "ΔV (%)", "Yeni Direk", "L_m"],
                                              ascending=[False, True, True, True])

    with st.expander("📈 En Uygun Trafo Adayları (rota ve ΔV ile)"):
        st.dataframe(cand_df[["Montaj Yeri", "Gücü[kVA]", "L_m", "ΔV (%)", "Kapasite Uygun", "Yeni Direk"]],
                     use_container_width=True)

    best = cand_df.iloc[0]

    # 400 kVA üstü uyarı
    try:
        if float(best["Gücü[kVA]"]) > 400:
            st.warning("Mevcut trafo gücü 400 kVA'dan büyük; yeni trafo gerekebilir.")
    except Exception:
        pass

    # Sonuç haritası
    m2 = folium.Map(location=[new_lat, new_lon], zoom_start=16, control_scale=True)

    # Mevcut direkler (mavi)
    for _, r in direk_clean.iterrows():
        folium.CircleMarker([r["Enlem"], r["Boylam"]], radius=4, color="blue",
                            fill=True, fill_opacity=0.7,
                            tooltip=f"Direk: {r['Direk Kodu']}").add_to(m2)

    # Trafolar (turuncu)
    for _, r in trafo_clean.iterrows():
        folium.Marker([r["Enlem"], r["Boylam"]],
                      tooltip=f"Trafo: {r['Montaj Yeri']}",
                      icon=folium.Icon(color="orange", icon="bolt", prefix="fa")).add_to(m2)

    # Talep & seçilen trafo
    folium.Marker((new_lat, new_lon), icon=folium.Icon(color="red"), tooltip="Talep Noktası").add_to(m2)
    folium.Marker((best["lat"], best["lon"]), icon=folium.Icon(color="orange", icon="bolt", prefix="fa"),
                  tooltip="Seçilen Trafo").add_to(m2)

    # Rota + nokta tipleri (mavi=mevcut, mor=yeni)
    if len(best["route"]) >= 2:
        try:
            if HAS_PYPROJ:
                fwd, _ = get_transformers()
                to_xy = lambda lon, lat: fwd.transform(lon, lat)
                poles_xy = [to_xy(lon, lat) for (lat, lon) in zip(direk_clean["Enlem"], direk_clean["Boylam"])]
                snapped_set = set(poles_xy)
                route_xy = [to_xy(lon, lat) for (lat, lon) in best["route"]]
                for (lat, lon), (x, y) in zip(best["route"], route_xy):
                    if (x, y) in snapped_set:
                        folium.CircleMarker((lat, lon), radius=5, color="blue", fill=True, fill_opacity=0.9,
                                            tooltip="Mevcut Direk (rota)").add_to(m2)
                    else:
                        folium.CircleMarker((lat, lon), radius=5, color="purple", fill=True, fill_opacity=0.9,
                                            tooltip="Önerilen Yeni Direk").add_to(m2)
        except Exception:
            pass

        folium.PolyLine(best["route"], color="green", weight=4, opacity=0.9,
                        tooltip=f"Hat uzunluğu ≈ {best['L_m']:.1f} m").add_to(m2)
    else:
        st.warning("Hat noktaları üretilemedi.")

    st.subheader("🧾 Hat Özeti")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Uzunluk", f"{best['L_m']:.1f} m")
    c2.metric("Kullanılan Mevcut Direk", f"{int(best['Kullanılan Direk'])}")
    c3.metric("Önerilen Yeni Direk", f"{int(best['Yeni Direk'])}")
    # Ortalama açıklık (basit tahmin)
    avg_span = best["L_m"] / max(1, (len(best["route"]) - 1))
    c4.metric("Ortalama Direk Aralığı", f"{avg_span:.1f} m")

    if best["ΔV (%)"] > drop_threshold_pct:
        st.error(f"⚠️ Gerilim düşümü %{best['ΔV (%)']:.2f} — eşik %{drop_threshold_pct:.1f} üstü.")
    else:
        st.success(f"✅ ΔV %{best['ΔV (%)']:.2f} ≤ %{drop_threshold_pct:.1f}")

    st.subheader("📡 Oluşturulan Şebeke Hattı")
    st_folium(m2, height=620, width="100%", key="result_map_basic")

# ===================== SAYFA 2: Gerilim Düşümü (Sentetik) =====================
elif selected == "Gerilim Düşümü":
    st.subheader("📉 Gerilim Düşümü — Sentetik Senaryo")
    st.caption("Parametrelerle oynayarak ΔV davranışını görün.")

    with st.sidebar.expander("🔌 Parametreler"):
        Vn_kV   = st.number_input("Nominal gerilim (kV)", 0.4, 34.5, 0.4, 0.1, key="gd_vn")
        pf      = st.number_input("Güç faktörü (pf)", 0.5, 1.0, 0.9, 0.05, key="gd_pf")
        R_ohm_km= st.number_input("R (Ω/km)", 0.05, 1.5, 0.642, 0.01, key="gd_R")
        X_ohm_km= st.number_input("X (Ω/km)", 0.01, 1.0, 0.083, 0.01, key="gd_X")

    Ls    = np.linspace(10, 2000, 120)   # m
    loads = np.linspace(5,  400,  120)   # kW
    mesh = [(L, P) for L in Ls for P in loads]
    df = pd.DataFrame(mesh, columns=["L_m","P_kw"])
    df["dv_pct"] = df.apply(lambda r: vdrop_percent(r.P_kw, r.L_m, Vn_kV, pf, R_ohm_km, X_ohm_km), axis=1)

    fig_hm = px.density_heatmap(df, x="L_m", y="P_kw", z="dv_pct", nbinsx=40, nbinsy=40, histfunc="avg",
                                title="ΔV (%) Isı Haritası (L vs P)", template="plotly_white")
    fig_hm.update_layout(xaxis_title="Hat Uzunluğu (m)", yaxis_title="Yük (kW)")
    st.plotly_chart(fig_hm, use_container_width=True)

    sel_load = st.slider("Kesit için Yük (kW)", 5, 400, 120, 5)
    df_slice = df[df.P_kw == sel_load]
    fig_ln = px.line(df_slice, x="L_m", y="dv_pct", markers=True, template="plotly_white",
                     title=f"ΔV (%) — Sabit Yük: {sel_load} kW")
    fig_ln.update_layout(xaxis_title="Hat Uzunluğu (m)", yaxis_title="ΔV (%)")
    st.plotly_chart(fig_ln, use_container_width=True)

# ===================== SAYFA 3: Forecasting =====================
elif selected == "Forecasting":
    st.subheader("📈 Yük Tahmini (Forecasting) — Demo")
    st.caption("Sentetik günlük seri + Holt-Winters (yoksa rolling mean) ile 60 gün tahmin.")

    rng = np.random.default_rng(7)
    days = pd.date_range("2024-01-01", periods=365, freq="D")
    base = 300 + 0.1*np.arange(len(days))
    weekly = 40*np.sin(2*np.pi*days.dayofweek/7)
    noise = rng.normal(0, 15, len(days))
    y = base + weekly + noise
    ts = pd.DataFrame({"ds": days, "y": y})

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(ts.y, trend="add", seasonal="add", seasonal_periods=7)
        fit = model.fit(optimized=True)
        fut = fit.forecast(60)
        fc = pd.DataFrame({"ds": pd.date_range(days[-1] + pd.Timedelta(days=1), periods=60), "yhat": fut.values})
    except Exception:
        roll = ts.y.rolling(7, min_periods=1).mean()
        fut = np.repeat(roll.iloc[-1], 60)
        fc = pd.DataFrame({"ds": pd.date_range(days[-1] + pd.Timedelta(days=1), periods=60), "yhat": fut})

    fig_fc = px.line(title="Günlük Yük — Geçmiş ve Tahmin", template="plotly_white")
    fig_fc.add_scatter(x=ts.ds, y=ts.y, mode="lines", name="Geçmiş")
    fig_fc.add_scatter(x=fc.ds, y=fc.yhat, mode="lines", name="Tahmin")
    fig_fc.update_layout(xaxis_title="Tarih", yaxis_title="kW")
    st.plotly_chart(fig_fc, use_container_width=True)

# ===================== SAYFA 4: Arıza / Anomali =====================
elif selected == "Arıza/Anomali":
    st.subheader("🚨 Arıza & Anomali Tespiti — Demo")
    st.caption("IsolationForest ile sentetik V-I-P üzerinde anomali işaretleme.")

    rng = np.random.default_rng(42)
    n = 800
    V = rng.normal(230, 3.0, n)
    I = rng.normal(40, 5.0, n)
    P = V * I / 1000.0
    df = pd.DataFrame({"V": V, "I": I, "P": P})

    # anomali enjekte
    idx = rng.choice(n, size=20, replace=False)
    df.loc[idx, "V"] += rng.normal(-30, 6, len(idx))
    df.loc[idx, "I"] += rng.normal(35, 8, len(idx))
    df["P"] = df["V"] * df["I"] / 1000.0

    iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=7)
    preds = iso.fit_predict(df[["V", "I", "P"]])
    df["anomaly"] = (preds == -1).astype(int)

    fig_sc = px.scatter(
        df, x="I", y="V",
        color=df["anomaly"].map({0: "Normal", 1: "Anomali"}),
        title="Akım–Volt Dağılımı — Anomaliler", template="plotly_white"
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    rate = df["anomaly"].mean() * 100
    st.metric("Anomali Oranı", f"%{rate:.2f}")

    with st.expander("Aykırı Nokta Tablosu"):
        st.dataframe(df[df["anomaly"] == 1].head(50), use_container_width=True)
