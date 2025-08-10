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

# --- Opsiyonel hızlandırıcılar ---
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
    # XLSX için openpyxl gerektir
    try:
        direk_df = pd.read_excel("Direk Sorgu Sonuçları.xlsx", engine="openpyxl")
        trafo_df = pd.read_excel("Trafo Sorgu Sonuçları.xlsx", engine="openpyxl")
    except ImportError:
        st.error("`openpyxl` eksik. requirements.txt'e `openpyxl` ekleyip yeniden deploy et.")
        st.stop()
    except FileNotFoundError as e:
        st.error(f"Dosya bulunamadı: {e}")
        st.stop()

    try:
        ext_df = pd.read_csv("smart_grid_dataset.csv")
    except Exception:
        ext_df = pd.DataFrame()

    return direk_df, trafo_df, ext_df

direk_df, trafo_df, ext_df = load_data()

# Beklenen kolonları çek
direk_clean = (
    direk_df[["AssetID", "Direk Kodu", "Enlem", "Boylam"]]
    .dropna(subset=["Enlem", "Boylam"]).copy()
)
trafo_clean = (
    trafo_df[["AssetID", "Montaj Yeri", "Gücü[kVA]", "Enlem", "Boylam"]]
    .dropna(subset=["Enlem", "Boylam"]).copy()
)

# ===================== YARDIMCI =====================
def get_transformers():
    if not HAS_PYPROJ:
        raise RuntimeError("pyproj mevcut değil")
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    bwd = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return fwd, bwd

# Gerilim düşümü: %e = k * L * N
def vdrop_kLN(L_m: float, P_kw: float, k: float) -> float:
    try:
        return float(k) * float(L_m) * float(P_kw)
    except Exception:
        return float("nan")

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
    Rota ve istatistikleri döndürür.
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
        total_len_m = geodesic(demand_latlon, trafo_latlon).meters
        final_path = [demand_latlon, trafo_latlon]
        return final_path, total_len_m, 0, 1, [total_len_m]

# ===================== SAYFA 1: Talep Girdisi =====================
if selected == "Talep Girdisi":
    st.sidebar.header("⚙️ Hat Parametreleri")
    max_span     = st.sidebar.number_input("Maks. direk aralığı (m)", 20, 120, 40, 5)
    snap_radius  = st.sidebar.number_input("Mevcut direğe snap yarıçapı (m)", 5, 120, 30, 5)

    with st.sidebar.expander("🔌 Elektrik Parametreleri"):
        k_const = st.number_input("k sabiti", 0.0, 1.0, 0.0001, 0.0001)
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
                            tooltip=f"Direk: {r['Direk Kodu']}").add_to(poles_group)

    for _, r in trafo_clean.iterrows():
        folium.Marker([r["Enlem"], r["Boylam"]],
                      tooltip=f"Trafo: {r['Montaj Yeri']}",
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
        st.info("📍 Haritadan bir talep noktası seçiniz."); st.stop()

    new_lat, new_lon = st.session_state["demand_point"]
    st.success(f"Yeni talep noktası: ({new_lat:.6f}, {new_lon:.6f})")

    st.subheader("⚡ Talep Edilen Yük (kW)")
    user_kw = st.slider("Talep edilen güç", 1, 500, 120, 5, key="kw_slider_basic")

    # Trafo adayları: en yakın 8 → rota üret → k·L·N ile düşüm hesapla
    def eval_trafo(row):
        t_latlon = (float(row["Enlem"]), float(row["Boylam"]))
        poles_latlon = list(zip(direk_clean["Enlem"].astype(float), direk_clean["Boylam"].astype(float)))
        route, Lm, used, prop, spans = build_route_and_stats((new_lat, new_lon), t_latlon, poles_latlon,
                                                             max_span=max_span, snap_radius=snap_radius)
        dv = vdrop_kLN(Lm, user_kw, k_const)  # k·L·N
        cap_ok = False
        try:
            kva = float(row["Gücü[kVA]"])
            cap_ok = (kva * 0.8) >= user_kw  # pf=0.8
        except Exception:
            pass
        return {
            "Montaj Yeri": row["Montaj Yeri"],
            "Gücü[kVA]": row["Gücü[kVA]"],
            "lat": t_latlon[0], "lon": t_latlon[1],
            "route": route, "L_m": Lm,
            "Gerilim Düşümü (%)": dv, "Kapasite Uygun": cap_ok,
            "Kullanılan Direk": used, "Yeni Direk": prop
        }

    trafo_local = trafo_clean.copy()
    trafo_local["geo_dist"] = trafo_local.apply(
        lambda r: geodesic((new_lat, new_lon), (float(r["Enlem"]), float(r["Boylam"]))).meters, axis=1
    )
    topN = trafo_local.sort_values("geo_dist").head(8)
    evals = [eval_trafo(r) for _, r in topN.iterrows()]
    cand_df = pd.DataFrame(evals).sort_values(
        by=["Kapasite Uygun", "Gerilim Düşümü (%)", "Yeni Direk", "L_m"],
        ascending=[False, True, True, True]
    )

    with st.expander("📈 En Uygun Trafo Adayları"):
        st.dataframe(
            cand_df[["Montaj Yeri", "Gücü[kVA]", "L_m", "Gerilim Düşümü (%)", "Kapasite Uygun", "Yeni Direk"]],
            use_container_width=True
        )

    best = cand_df.iloc[0]

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

    # Rota üzerinde mevcut (mavi) / yeni (mor) direkler
    try:
        if HAS_PYPROJ and len(best["route"]) >= 2:
            fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
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

    if len(best["route"]) >= 2:
        folium.PolyLine(best["route"], color="green", weight=4, opacity=0.9,
                        tooltip=f"Hat uzunluğu ≈ {best['L_m']:.1f} m").add_to(m2)
    else:
        st.warning("Hat noktaları üretilemedi.")

    # Özet metrikler
    st.subheader("🧾 Hat Özeti")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Uzunluk", f"{best['L_m']:.1f} m")
    c2.metric("Kullanılan Mevcut Direk", f"{int(best['Kullanılan Direk'])}")
    c3.metric("Önerilen Yeni Direk", f"{int(best['Yeni Direk'])}")
    avg_span = best["L_m"] / max(1, len(best["route"]) - 1)
    c4.metric("Ortalama Direk Aralığı", f"{avg_span:.1f} m")

    dv_val = float(best["Gerilim Düşümü (%)"])
    try:
        best_kva = float(best["Gücü[kVA]"])
    except Exception:
        best_kva = None

    if dv_val > drop_threshold_pct:
        st.error(f"⚠️ Gerilim düşümü %{dv_val:.2f} — eşik %{drop_threshold_pct:.1f} üstü.")
    else:
        st.success(f"✅ Gerilim düşümü %{dv_val:.2f} ≤ %{drop_threshold_pct:.1f}")

    if (best_kva is not None) and (best_kva > 400):
        st.warning("ℹ️ Mevcut trafo gücü 400 kVA üzerinde — **ek trafo gerekebilir**.")

    # Durum kartı (sayfa içi hesap)
    durum_val = dv_val <= drop_threshold_pct
    bg = "#0ea65d" if durum_val else "#ef4444"
    txt = "Eşik altında — Tasarım uygun." if durum_val else "Eşik üstünde — İyileştirme gerek."
    st.markdown(
        f"""<div style="background:{bg};padding:16px;border-radius:14px;color:white;font-weight:600;">{txt}</div>""",
        unsafe_allow_html=True
    )

    st.subheader("📡 Oluşturulan Şebeke Hattı")
    st_folium(m2, height=620, width="100%", key="result_map_basic")

# ===================== SAYFA 2: Gerilim Düşümü =====================
elif selected == "Gerilim Düşümü":
    st.subheader("📉 Gerilim Düşümü — Gerçek Veri & AI (Trafo Bazlı)")

    # ===== Girdiler =====
    c0, c1, c2, c3 = st.columns([1,1,1,1])
    with c0:
        k_const = st.number_input("k sabiti", 0.0, 1.0, 0.0001, 0.0001)
    with c1:
        thr_pct = st.number_input("Eşik (%)", 0.5, 20.0, 5.0, 0.5)
    with c2:
        L_in = st.number_input("Hat Uzunluğu L (m)", 10, 10000, 600, 10)
    with c3:
        N_in = st.number_input("Yük N (kW)", 1, 5000, 200, 1)

    k_in = k_const

    # ===== Formül Hesap Fonksiyonu =====
    def vdrop_kLN(L_m, P_kw, k):
        try:
            return float(k) * float(L_m) * float(P_kw)
        except:
            return float("nan")

    # ===== Eğitim Verisi =====
    def build_training_df(ext_df):
        if ext_df is not None and not ext_df.empty:
            cols_lower = {c.lower(): c for c in ext_df.columns}
            needs = ["l_m", "p_kw", "k", "dv_pct"]
            if all(n in cols_lower for n in needs):
                df = pd.DataFrame({
                    "L_m":    ext_df[cols_lower["l_m"]],
                    "P_kw":   ext_df[cols_lower["p_kw"]],
                    "k":      ext_df[cols_lower["k"]],
                    "dv_pct": ext_df[cols_lower["dv_pct"]],
                }).dropna()
                df["dv_pct"] = df["dv_pct"].clip(0, 15)
                return df

        # fallback sentetik
        rng = np.random.default_rng(0)
        n = 3000
        L = rng.uniform(10, 3000, n)
        P = rng.uniform(1, 600, n)
        k_vals = np.clip(rng.normal(loc=k_const if k_const > 0 else 1e-4,
                                    scale=0.25 * (k_const if k_const > 0 else 1e-4),
                                    size=n), 1e-6, 1.0)
        dv = np.clip(k_vals * L * P * rng.normal(1.0, 0.03, size=n), 0, 15)
        return pd.DataFrame({"L_m": L, "P_kw": P, "k": k_vals, "dv_pct": dv})

    train_df = build_training_df(ext_df)

    # ===== Model Eğitimi =====
    @st.cache_resource
    def train_regressor(df):
        X = df[["L_m", "P_kw", "k"]]
        y = df["dv_pct"]
        try:
            from lightgbm import LGBMRegressor
            reg = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=64, random_state=42)
        except:
            from sklearn.ensemble import RandomForestRegressor
            reg = RandomForestRegressor(n_estimators=350, random_state=42, n_jobs=-1)
        reg.fit(X, y)
        return reg

    try:
        reg = train_regressor(train_df)
    except:
        reg = None

    # ===== Tek Trafo Seçimi =====
    if 'trafo_df' in locals():
        trafo_names = trafo_df["Montaj Yeri"].dropna().unique()
        trafo_sec = st.selectbox("🔌 Trafo Seçin", options=trafo_names)
        trafo_row = trafo_df[trafo_df["Montaj Yeri"] == trafo_sec].iloc[0]
        trafo_power = trafo_row.get("Gücü[kVA]", 100)

        # 5 direk varsayımı
        direk_sayisi = 5
        L_list = np.linspace(L_in, L_in + 500, direk_sayisi)
        N_list = np.linspace(N_in, N_in + 50, direk_sayisi)

        dloc = pd.DataFrame({
            "L_m": L_list,
            "P_kw": N_list,
            "k": [k_in] * direk_sayisi
        })

        # Formül
        dloc["Formül (%)"] = dloc.apply(lambda r: vdrop_kLN(r["L_m"], r["P_kw"], r["k"]), axis=1)

        # AI Tahmini
        if reg is not None:
            dloc_pred = dloc[["L_m", "P_kw", "k"]]  # Model ile uyumlu kolonlar
            dloc["Tahmin (%)"] = reg.predict(dloc_pred)
        else:
            dloc["Tahmin (%)"] = float("nan")

        # Gösterim
        st.dataframe(dloc)

        fig = px.line(dloc, x="L_m", y=["Formül (%)", "Tahmin (%)"], markers=True,
                      title=f"Gerilim Düşümü — {trafo_sec}")
        st.plotly_chart(fig, use_container_width=True)

    # ===== Örnek Kullanıcı Hesabı =====
    dv_formula = vdrop_kLN(L_in, N_in, k_in)
    if reg is not None:
        Xq = pd.DataFrame([{"L_m": L_in, "P_kw": N_in, "k": k_in}])
        dv_ai = float(reg.predict(Xq)[0])
    else:
        dv_ai = float("nan")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📐 Formül (k·L·N)", f"%{dv_formula:.2f}")
    m2.metric("🤖 AI Tahmini", f"%{dv_ai:.2f}" if np.isfinite(dv_ai) else "—")
    m3.metric("🎯 Eşik", f"%{thr_pct:.2f}")
    durum_val = (dv_ai if np.isfinite(dv_ai) else dv_formula) <= thr_pct
    m4.metric("Durum", "✅ Uygun" if durum_val else "❌ Uygunsuz")


# ===================== SAYFA 3: Forecasting (Sadece Prophet) =====================
elif selected == "Forecasting":
    st.subheader("📈 Yük Tahmini (Forecasting) — Prophet")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        horizon = st.number_input("Tahmin ufku (gün)", 7, 180, 30, 1)
    with c2:
        holdout_days = st.number_input("Test penceresi (gün)", 7, 90, 30, 1)
    with c3:
        agg = st.selectbox("Zaman toplaması", ["Günlük Ortalama", "Günlük Toplam"], index=0)

    if ext_df is None or ext_df.empty:
        st.error("smart_grid_dataset.csv bulunamadı/boş."); st.stop()

    cols_lower = {c.lower(): c for c in ext_df.columns}
    time_col = next((cols_lower[k] for k in ["timestamp","datetime","date","tarih","ds"] if k in cols_lower), None)
    if time_col is None:
        for c in ext_df.columns:
            parsed = pd.to_datetime(ext_df[c], errors="coerce")
            if parsed.notna().mean() > 0.6:
                time_col = c; break
    load_col = next((cols_lower[k] for k in ["load_kw","load","power_kw","kw","value","y"] if k in cols_lower), None)
    if load_col is None:
        numeric_candidates = [c for c in ext_df.columns if pd.api.types.is_numeric_dtype(ext_df[c])]
        load_col = numeric_candidates[0] if numeric_candidates else None

    if time_col is None or load_col is None:
        st.error("CSV'de zaman/yük kolonları tespit edilemedi."); st.stop()

    df_raw = ext_df[[time_col, load_col]].rename(columns={time_col: "ds", load_col: "y"}).copy()
    df_raw["ds"] = pd.to_datetime(df_raw["ds"], errors="coerce")
    df_raw["y"]  = pd.to_numeric(df_raw["y"], errors="coerce")
    df_raw = df_raw.dropna(subset=["ds","y"]).sort_values("ds")
    if df_raw.empty:
        st.error("Seçilen kolonlardan tarih/yük üretilemedi."); st.stop()

    if "Ortalama" in agg:
        series = df_raw.set_index("ds")["y"].resample("D").mean().interpolate("time")
    else:
        series = df_raw.set_index("ds")["y"].resample("D").sum().interpolate("time")
    ts = series.reset_index().rename(columns={"index":"ds"})
    if len(ts) <= holdout_days + 30:
        st.error("Zaman serisi kısa. Test penceresini küçült veya veri aralığını artır."); st.stop()

    cutoff = ts["ds"].max() - pd.Timedelta(days=int(holdout_days))
    train = ts[ts["ds"] <= cutoff].copy()
    test  = ts[ts["ds"] >  cutoff].copy()

    try:
        from prophet import Prophet
    except Exception as e:
        st.error(f"Prophet yüklenemedi: {e} (requirements.txt'e 'prophet' ekleyin)"); st.stop()

    m = Prophet(seasonality_mode="additive", yearly_seasonality=False, daily_seasonality=False)
    m.add_seasonality(name="weekly", period=7, fourier_order=6)
    m.fit(train.rename(columns={"ds":"ds","y":"y"}))

    test_pred = m.predict(test[["ds"]])
    yhat_test = pd.Series(test_pred["yhat"].values, index=test["ds"].values)

    future = m.make_future_dataframe(periods=int(horizon), freq="D", include_history=False)
    fut = m.predict(future)
    fc = pd.DataFrame({"ds": fut["ds"], "yhat": fut["yhat"], "yhat_low": fut["yhat_lower"], "yhat_high": fut["yhat_upper"]})

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], mode="lines", name="Gerçek (Train)"))
    fig.add_trace(go.Scatter(x=test["ds"],  y=test["y"],  mode="lines", name="Gerçek (Test)"))
    fig.add_trace(go.Scatter(x=yhat_test.index, y=yhat_test.values, mode="lines", name="Test Tahmini"))
    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", name="İleri Tahmin"))
    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_low"],  mode="lines", name="Alt Band", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_high"], mode="lines", name="Üst Band", line=dict(dash="dot")))
    fig.update_layout(template="plotly_white", title="Prophet — Geçmiş, Test ve İleri Tahmin",
                      xaxis_title="Tarih", yaxis_title="kW", legend_title="Seri")
    st.plotly_chart(fig, use_container_width=True)

    out = fc[["ds","yhat","yhat_low","yhat_high"]].rename(
        columns={"ds":"tarih","yhat":"tahmin_kw","yhat_low":"alt","yhat_high":"üst"}
    )
    st.download_button("📥 Tahmini CSV indir",
                       data=out.to_csv(index=False).encode("utf-8"),
                       file_name="forecast_prophet.csv", mime="text/csv")

    st.divider()

    # Metrikler (altta)
    import numpy as np
    def _rmse(y_true, y_pred):
        yt, yp = np.array(y_true), np.array(y_pred)
        return float(np.sqrt(np.mean((yt - yp)**2)))
    def _mae(y_true, y_pred):
        yt, yp = np.array(y_true), np.array(y_pred)
        return float(np.mean(np.abs(yt - yp)))
    def _mape(y_true, y_pred):
        yt, yp = np.array(y_true), np.array(y_pred)
        mask = yt != 0
        return float(np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100) if mask.sum() else np.nan
    def _rmse_pct(y_true, y_pred):
        rm = _rmse(y_true, y_pred)
        denom = float(np.mean(np.abs(y_true)))
        return float(rm/denom*100) if denom > 0 else np.nan

    y_test = test.set_index("ds")["y"]
    rmse  = _rmse(y_test.values, yhat_test.values)
    mae   = _mae(y_test.values, yhat_test.values)
    mape  = _mape(y_test.values, yhat_test.values)
    rmsep = _rmse_pct(y_test.values, yhat_test.values)

    with st.expander("📊 Model Sonuçları"):
        cM1, cM2, cM3, cM4 = st.columns(4)
        cM1.metric("RMSE", f"{rmse:,.2f}")
        cM2.metric("MAE",  f"{mae:,.2f}")
        cM3.metric("MAPE", f"%{mape:,.2f}" if np.isfinite(mape) else "—")
        cM4.metric("RMSE%", f"%{rmsep:,.2f}" if np.isfinite(rmsep) else "—")

# ===================== SAYFA 4: Arıza / Anomali ====================
elif selected == "Arıza/Anomali":
    st.subheader("🚨 Arıza & Anomali Tespiti — Demo (IsolationForest)")

    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(42)
    n = 800
    V = rng.normal(230, 3.0, n)
    I = rng.normal(40, 5.0, n)
    P = V * I / 1000.0
    df = pd.DataFrame({"V": V, "I": I, "P": P})

    idx = rng.choice(n, size=20, replace=False)
    df.loc[idx, "V"] += rng.normal(-30, 6, len(idx))
    df.loc[idx, "I"] += rng.normal(35, 8, len(idx))
    df["P"] = df["V"] * df["I"] / 1000.0

    iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=7)
    preds = iso.fit_predict(df[["V", "I", "P"]])
    df["anomaly"] = (preds == -1).astype(int)

    fig_sc = px.scatter(df, x="I", y="V",
                        color=df["anomaly"].map({0: "Normal", 1: "Anomali"}),
                        title="Akım–Volt Dağılımı — Anomaliler", template="plotly_white")
    st.plotly_chart(fig_sc, use_container_width=True)

    rate = df["anomaly"].mean() * 100
    st.metric("Anomali Oranı", f"%{rate:.2f}")

    with st.expander("Aykırı Nokta Tablosu"):
        st.dataframe(df[df["anomaly"] == 1].head(50), use_container_width=True)
