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
    options=["Talep Verisi Girişi", "Gerilim Düşümü Analizi", "Talep / Tüketim Tahmini", "Arıza ve Anomali Tespiti"],
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
if selected == "Talep Verisi Girişi":
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

# ===================== SAYFA 2: Gerilim Düşümü — Gerçek Veri & AI =====================
elif selected == "Gerilim Düşümü Analizi":
    st.subheader("📉 Gerilim Düşümü ")

    # ------- Girdiler (sayfa içi) -------
    c0, c1, c2, c3 = st.columns([1,1,1,1])
    with c0:
        k_const = st.number_input("k sabiti", 0.0, 1.0, 0.0001, 0.0001, key="gd_k_inline")
    with c1:
        thr_pct = st.number_input("Eşik (%)", 0.5, 20.0, 5.0, 0.5, key="gd_thr_inline")
    with c2:
        L_in = st.number_input("Hat Uzunluğu L (m)", 10, 10000, 600, 10)
    with c3:
        N_in = st.number_input("Yük N (kW)", 1, 5000, 200, 1)

    k_in = k_const  # aynı k'yı hem örnek hesapta hem AI tahminde kullanacağız

    # ------- Formül -------
    def vdrop_kLN(L_m: float, P_kw: float, k: float) -> float:
        try:
            return float(k) * float(L_m) * float(P_kw)
        except Exception:
            return float("nan")

    # ------- Eğitim verisi: ext_df varsa kullan, yoksa sentetik (tamamı %15 ile sınırlandırılır) -------
    def build_training_df(ext_df):
        try:
            cols_lower = {c.lower(): c for c in ext_df.columns}
        except Exception:
            cols_lower = {}
        needs = ["l_m", "p_kw", "k", "dv_pct"]
        if ext_df is not None and len(ext_df) > 0 and all(n in cols_lower for n in needs):
            df = pd.DataFrame({
                "L_m":    ext_df[cols_lower["l_m"]],
                "P_kw":   ext_df[cols_lower["p_kw"]],
                "k":      ext_df[cols_lower["k"]],
                "dv_pct": ext_df[cols_lower["dv_pct"]],
            }).dropna()
            df["dv_pct"] = df["dv_pct"].clip(0, 15)  # max %15
            return df

        # fallback: sentetik (ölçekler daraltıldı, üst sınır %15)
        rng = np.random.default_rng(0)
        n = 3000
        L = rng.uniform(10, 3000, n)
        P = rng.uniform(1,  600,  n)
        k_vals = rng.normal(loc=k_const if k_const > 0 else 1e-4,
                            scale=0.25 * (k_const if k_const > 0 else 1e-4),
                            size=n)
        k_vals = np.clip(k_vals, 1e-6, 1.0)
        dv = k_vals * L * P * rng.normal(1.0, 0.03, size=n)  # küçük ölçüm hatası
        dv = np.clip(dv, 0, 15)  # max %15
        return pd.DataFrame({"L_m": L, "P_kw": P, "k": k_vals, "dv_pct": dv})

    train_df = build_training_df(ext_df)

    # ------- Model eğitimi (LightGBM yoksa RF'ye düş) -------
    @st.cache_resource
    def train_regressor(df: pd.DataFrame):
        X = df[["L_m", "P_kw", "k"]]
        y = df["dv_pct"]
        try:
            from lightgbm import LGBMRegressor
            reg = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=64, random_state=42)
        except Exception:
            from sklearn.ensemble import RandomForestRegressor
            reg = RandomForestRegressor(n_estimators=350, random_state=42, n_jobs=-1)
        reg.fit(X, y)
        return reg

    try:
        reg = train_regressor(train_df)
    except Exception:
        reg = None

    # ------- Örnek tahmin -------
    dv_formula = vdrop_kLN(L_in, N_in, k_in)
    if reg is not None:
        Xq = pd.DataFrame([{"L_m": L_in, "P_kw": N_in, "k": k_in}])
        dv_ai = float(reg.predict(Xq)[0])
    else:
        dv_ai = float("nan")

    # tekil çıktıları da 15'e kırp
    if np.isfinite(dv_formula):
        dv_formula = float(np.clip(dv_formula, 0, 15))
    if np.isfinite(dv_ai):
        dv_ai = float(np.clip(dv_ai, 0, 15))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📐 Hesaplama", f"%{dv_formula:.2f}")
    m2.metric("🤖 AI Tahmini", f"%{dv_ai:.2f}" if np.isfinite(dv_ai) else "—")
    m3.metric("🎯 Eşik", f"%{thr_pct:.2f}")
    durum_val = (dv_ai if np.isfinite(dv_ai) else dv_formula) <= thr_pct
    m4.metric("Durum", "✅ Uygun" if durum_val else "❌ Uygunsuz")

    st.divider()

    # ================== Trafo bazlı basit karşılaştırma (FİX 5 DİREK, %15 clip) ==================
    st.markdown("### 🔌 Trafo Seçin")

    trafo_names = trafo_df["Montaj Yeri"].dropna().astype(str).unique().tolist()
    if len(trafo_names) == 0:
        st.info("Trafo verisi yok."); st.stop()

    trafo_sel = st.selectbox("Trafo Seçin", options=trafo_names)

    # 1) Seçilen trafo konumu
    trow = trafo_df[trafo_df["Montaj Yeri"].astype(str) == trafo_sel].iloc[0]
    t_coord = (float(trow["Enlem"]), float(trow["Boylam"]))

    # 2) Direkler: en yakın 5 direk (fix)
    dloc = direk_df.dropna(subset=["Enlem", "Boylam"]).copy()
    if len(dloc) == 0:
        st.error("Direk verisi yok."); st.stop()

    dloc["Mesafe (m)"] = dloc.apply(
        lambda r: geodesic((float(r["Enlem"]), float(r["Boylam"])), t_coord).meters, axis=1
    )
    dloc = dloc.sort_values("Mesafe (m)").head(5).reset_index(drop=True)  # <<< FİX 5 DİREK

    # 3) Yük (kW): yoksa sentetik, var ise sayısallaştır
    rng = np.random.default_rng(42)
    if "Yük (kW)" in dloc.columns:
        dloc["Yük (kW)"] = pd.to_numeric(dloc["Yük (kW)"], errors="coerce").fillna(
            rng.integers(10, 300, size=len(dloc))
        )
    else:
        dloc["Yük (kW)"] = rng.integers(10, 300, size=len(dloc))

    # 4) Gerçek (formül) ve AI tahmini
    dloc["Gerçek (%)"] = dloc.apply(lambda r: vdrop_kLN(r["Mesafe (m)"], r["Yük (kW)"], k_in), axis=1)
    if reg is not None:
        Xb = dloc.rename(columns={"Mesafe (m)": "L_m", "Yük (kW)": "P_kw"})[["L_m", "P_kw"]].copy()
        Xb["k"] = k_in
        dloc["Tahmin (%)"] = reg.predict(Xb)
    else:
        dloc["Tahmin (%)"] = np.nan

    # 5) ÜST SINIR: max %15’e clip
    dloc["Gerçek (%)"]  = dloc["Gerçek (%)"].clip(upper=15)
    dloc["Tahmin (%)"]  = dloc["Tahmin (%)"].clip(upper=15)

    # 6) Basit performans
    valid = dloc[["Gerçek (%)", "Tahmin (%)"]].dropna()
    if len(valid) >= 3:
        from sklearn.metrics import r2_score, mean_squared_error
        r2  = r2_score(valid["Gerçek (%)"], valid["Tahmin (%)"])
        mse = mean_squared_error(valid["Gerçek (%)"], valid["Tahmin (%)"])
    else:
        r2 = mse = float("nan")

    # 7) Grafik: Çizgi grafiği (Direk Kodu bazlı)
    import plotly.express as px
    x_labels = dloc["Direk Kodu"].astype(str).fillna("—")
    plot_df = dloc.assign(**{"Direk": x_labels})[["Direk", "Gerçek (%)", "Tahmin (%)"]]

    fig_cmp = px.line(
        plot_df,
        x="Direk",
        y=["Gerçek (%)", "Tahmin (%)"],
        markers=True,
        template="plotly_white",
        title=f"{trafo_sel} — Gerçek vs AI"
    )
    fig_cmp.add_hline(y=thr_pct, line_dash="dot", annotation_text=f"Eşik %{thr_pct:.2f}")
    fig_cmp.update_layout(
        xaxis_title="Direk",
        yaxis_title="Gerilim Düşümü (%)"
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # 8) Expander içinde R², MSE ve tablo
    with st.expander("📊 Detaylı Sonuçlar"):
        st.markdown(f"**R²:** {r2:.3f}" if np.isfinite(r2) else "**R²:** —")
        st.markdown(f"**MSE:** {mse:.4f}" if np.isfinite(mse) else "**MSE:** —")
        st.dataframe(
            dloc[["Direk Kodu","Mesafe (m)","Yük (kW)","Gerçek (%)","Tahmin (%)"]]
            .style.format({
                "Mesafe (m)":"{:.0f}",
                "Yük (kW)":"{:.0f}",
                "Gerçek (%)":"{:.2f}",
                "Tahmin (%)":"{:.2f}"
            }),
            use_container_width=True
        )



# ===================== SAYFA 3: Forecasting (Zaman Serisi Tahmini) =====================
elif selected == "Talep / Tüketim Tahmini":
    st.subheader("📈 Zaman Serisi Tahmini ")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        horizon = st.number_input("Öngörü Süresi (Gün)", 7, 180, 30, 1)
    with c2:
        holdout_days = st.number_input("Test Veri Aralığı (Gün)", 7, 90, 30, 1)
    with c3:
        agg = st.selectbox("Zaman Ölçeği", ["Günlük Ortalama", "Günlük Toplam"], index=0)

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

# ===================== SAYFA 4: Arıza / Anomali Tespiti (sabit parametreler + şık metrikler) =====================
elif selected == "Arıza ve Anomali Tespiti":
    st.subheader("🚨 Arıza & Anomali Tespiti")

    # ---- Sabitler (kullanıcıdan sormuyoruz) ----
    AGG_MODE = "mean"     # Günlük Ortalama
    CONTAM   = 0.03       # Anomali oranı
    HOLDOUT  = 30         # Test penceresi (gün)
    ROLL_WIN = 7          # Rolling pencere (gün)

    # ---- Veri kontrol ----
    if ext_df is None or ext_df.empty:
        st.error("smart_grid_dataset.csv bulunamadı/boş."); st.stop()

    # timestamp / load kolonlarını bul
    cols_lower = {c.lower(): c for c in ext_df.columns}
    time_col = next((cols_lower[k] for k in ["timestamp","datetime","date","tarih","ds"] if k in cols_lower), None)
    load_col = next((cols_lower[k] for k in ["load_kw","load","power_kw","kw","value","y"] if k in cols_lower), None)

    if time_col is None:
        for c in ext_df.columns:
            if pd.to_datetime(ext_df[c], errors="coerce").notna().mean() > 0.6:
                time_col = c; break
    if load_col is None:
        numc = [c for c in ext_df.columns if pd.api.types.is_numeric_dtype(ext_df[c])]
        load_col = numc[0] if numc else None

    if time_col is None or load_col is None:
        st.error("CSV’de zaman ve yük kolonu bulunamadı."); st.stop()

    df = ext_df[[time_col, load_col]].rename(columns={time_col:"ds", load_col:"y"}).copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"]  = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds","y"]).sort_values("ds")

    # ---- Günlük toplama (sabit: Ortalama) ----
    s = df.set_index("ds")["y"].resample("D").mean().interpolate("time")
    ts = s.reset_index().rename(columns={"index":"ds"})

    if len(ts) <= HOLDOUT + 30:
        st.error("Zaman serisi kısa. Test veri aralığını küçültmek veya veri aralığını artırmak gerekli olabilir."); st.stop()

    # ---- Özellikler ----
    ts["lag1"] = ts["y"].shift(1)
    ts["lag2"] = ts["y"].shift(2)
    ts["lag3"] = ts["y"].shift(3)
    ts["diff1"] = ts["y"].diff(1)
    ts["pct1"]  = ts["y"].pct_change(1).replace([np.inf, -np.inf], np.nan)
    ts["roll_mean"] = ts["y"].rolling(ROLL_WIN, min_periods=1).mean()
    ts["roll_std"]  = ts["y"].rolling(ROLL_WIN, min_periods=1).std().fillna(0.0)

    feats = ["y","lag1","lag2","lag3","diff1","pct1","roll_mean","roll_std"]
    ts_feats = ts.dropna(subset=["lag3"]).copy()  # ilk 3 gün düşer

    # ---- Train/Test böl ----
    cutoff = ts_feats["ds"].max() - pd.Timedelta(days=HOLDOUT)
    train = ts_feats[ts_feats["ds"] <= cutoff].copy()
    test  = ts_feats[ts_feats["ds"] >  cutoff].copy()
    if len(train) < 20 or len(test) < 5:
        st.error("Eğitim/test için yeterli veri yok. Parametreleri yeniden değerlendir."); st.stop()

    # ---- Model ----
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=300, contamination=CONTAM, random_state=7)
    iso.fit(train[feats])

    # Skor & tahmin
    ts_feats["score"] = iso.decision_function(ts_feats[feats])  # büyük = normal, küçük = anomali
    ts_feats["pred"]  = iso.predict(ts_feats[feats])            # 1 normal, -1 anomali
    ts_feats["anomaly"] = (ts_feats["pred"] == -1).astype(int)

    # ---- Basit arıza tipi etiketleme ----
    med_std = float(ts_feats["roll_std"].median())
    def fault_type(row):
        if row["anomaly"] != 1:
            return "Normal"
        if row["diff1"] > 2.5 * med_std:
            return "Ani Artış (Spike)"
        if row["diff1"] < -2.5 * med_std:
            return "Ani Düşüş (Drop)"
        if abs(row["roll_std"]) < 1e-6:
            return "Düz Çizgi (Flatline)"
        return "Aykırı"
    ts_feats["tip"] = ts_feats.apply(fault_type, axis=1)

        # ---- Parametreler ----
    with st.expander("⚙️ Parametreler"):
        cpa, cpb, cpc, cpd = st.columns(4)
        with cpa:
            AGG_MODE = st.selectbox("Zaman Ölçeği", ["Günlük Ortalama"], index=0)
        with cpb:
            HOLDOUT = st.number_input("Test Veri Aralığı (Gün)", min_value=1, max_value=365,
                                      value=HOLDOUT, step=1)
        with cpc:
            CONTAM = st.number_input("Anomali Oranı", min_value=0.0, max_value=1.0,
                                     value=float(CONTAM), step=0.01, format="%.2f")
        with cpd:
            ROLL_WIN = st.number_input("Hareketli Ortalama Penceresi (Gün)", min_value=1, max_value=365,
                                       value=ROLL_WIN, step=1)


    # ---- Grafik ----
    import plotly.graph_objects as go
    base = ts_feats[ts_feats["anomaly"] == 0]
    outl = ts_feats[ts_feats["anomaly"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base["ds"], y=base["y"], mode="lines", name="Seri (Normal)"))
    fig.add_trace(go.Scatter(x=outl["ds"], y=outl["y"], mode="markers", name="Anomali",
                             marker=dict(size=9, symbol="x")))
    fig.update_layout(template="plotly_white",
                      title="Zaman Serisi ve Tespit Edilen Anomaliler",
                      xaxis_title="Tarih", yaxis_title="kW")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Şık metrik kutuları (grafiğin ALTINDA) ----
    total = int(len(ts_feats))
    anom  = int(outl.shape[0])
    rate  = (anom / total * 100.0) if total > 0 else 0.0

    st.markdown("### 📊 Anomali İstatistikleri")
    st.write("")  # biraz boşluk

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(label="📄 Toplam Kayıt", value=f"{total:,}")
    with c2:
        st.metric(label="🚨 Anomali Sayısı", value=f"{anom:,}")
    with c3:
        st.metric(label="📈 Anomali Oranı", value=f"%{rate:.2f}")

    st.divider()


    # ---- (Opsiyonel)  ----
    anomalies = outl[["ds","y","score","diff1","pct1","tip"]].sort_values("ds")
    with st.expander("📋 Anomali Tablosu"):
        st.dataframe(anomalies, use_container_width=True)
      