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
    direk_df = pd.read_excel("Direk Sorgu Sonuçları.xlsx")
    trafo_df = pd.read_excel("Trafo Sorgu Sonuçları.xlsx")
    ext_df   = pd.read_csv("smart_grid_dataset.csv")  
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

# --- Gerilim düşümü (senin istediğin model): %e = k * L * N ---
# L: metre, N: kW, k: sabit
def vdrop_pct_kLN(L_m: float, P_kw: float, k: float = 0.0001) -> float:
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
        # fallback: düz geodesic
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
        st.info("📍 Haritadan bir talep noktası belirleyiniz."); st.stop()

    new_lat, new_lon = st.session_state["demand_point"]
    st.success(f"Yeni talep noktası: ({new_lat:.6f}, {new_lon:.6f})")

    st.subheader("⚡ Talep Edilen Yük (kW)")
    user_kw = st.slider("Talep edilen güç", 1, 500, 120, 5, key="kw_slider_basic")

    # Trafo değerlendirme: önce geodesic'e göre en yakın 8'i seç, her birinde rota üret ve gerilim düşümünü hesapla
    def eval_trafo(row):
        t_latlon = (float(row["Enlem"]), float(row["Boylam"]))
        poles_latlon = list(zip(direk_clean["Enlem"].astype(float), direk_clean["Boylam"].astype(float)))
        route, Lm, used, prop, spans = build_route_and_stats((new_lat, new_lon), t_latlon, poles_latlon,
                                                             max_span=max_span, snap_radius=snap_radius)
        dv = vdrop_pct_kLN(Lm, user_kw, k_const)  # <-- k·L·N
        cap_ok = False
        try:
            kva = float(row["Gücü[kVA]"])
            cap_ok = (kva * 0.8) >= user_kw  # pf=0.8 varsayımı (isteğe bağlı parametreleşir)
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

    with st.expander("📈 En Uygun Trafo Adayları (rota ve gerilim düşümü ile)"):
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

    # Rota + rota üzerindeki nokta tipleri (mavi=mevcut, mor=yeni)
    if len(best["route"]) >= 2:
        try:
            if HAS_PYPROJ:
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
    # Ortalama açıklık (yaklaşık)
    avg_span = best["L_m"] / max(1, len(best["route"]) - 1)
    c4.metric("Ortalama Direk Aralığı", f"{avg_span:.1f} m")

    # Bildirimler (ikili koşul)
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
        if dv_val > drop_threshold_pct:
            st.error("⚠️ Mevcut trafo gücü 400 kVA üzerinde ve gerilim düşümü eşiği aşılıyor — **ek trafo gerekebilir**.")
        else:
            st.warning("ℹ️ Mevcut trafo gücü 400 kVA üzerinde — **ek trafo gerekebilir**.")

    st.subheader("📡 Oluşturulan Şebeke Hattı")
    st_folium(m2, height=620, width="100%", key="result_map_basic")



    # ====== A) Renkli durum kartı + Gauge ======
    # Renkli kart
    bg = "#0ea65d" if durum_val else "#ef4444"
    txt = "Eşik altında — Tasarım uygun." if durum_val else "Eşik üstünde — İyileştirme gerek."
    st.markdown(
        f"""
        <div style="background:{bg};padding:16px;border-radius:14px;color:white;font-weight:600;">
            {txt}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ================== TRAFO KARŞILAŞTIRMA (Sidebar seçmeli) ==================
    st.divider()
    st.markdown("### 🔌 Trafo Karşılaştırma — Formül vs AI")

    # --- Sidebar: trafo seçimi ve referans nokta ---
    st.sidebar.header("Trafo Karşılaştırma")
    trafo_names = trafo_clean["Montaj Yeri"].dropna().astype(str).unique().tolist()
    sel_trafos = st.sidebar.multiselect("Trafo seç", options=trafo_names, default=trafo_names[:5])
    ref_lat = st.sidebar.number_input("Referans Enlem (°)", value=float(trafo_clean["Enlem"].mean()))
    ref_lon = st.sidebar.number_input("Referans Boylam (°)", value=float(trafo_clean["Boylam"].mean()))
    sort_by = st.sidebar.selectbox("Sırala", ["Farka göre (büyük→küçük)", "AI düşüme göre (küçük→büyük)", "Formül düşüme göre (küçük→büyük)"])

    if len(sel_trafos) == 0:
        st.info("Sidebardan en az bir trafo seç.")
    else:
        # Seçilen trafoları al
        traf = trafo_clean[trafo_clean["Montaj Yeri"].astype(str).isin(sel_trafos)].copy()

        # L: referans → trafo jeodezik mesafe (m)
        def _dist_to_ref(row):
            try:
                return geodesic((ref_lat, ref_lon), (float(row["Enlem"]), float(row["Boylam"]))).meters
            except Exception:
                return np.nan

        traf["L_m"] = traf.apply(_dist_to_ref, axis=1)
        traf = traf.dropna(subset=["L_m"]).reset_index(drop=True)

        # Hesaplar (N=kW ve k sabiti olarak sayfanın üstteki L/N/k girişlerindeki N_in ve k_in kullanıyoruz)
        traf["Formül_%"] = traf["L_m"].apply(lambda L: vdrop_kLN(L, N_in, k_in))
        if reg is not None:
            Xbatch = pd.DataFrame({"L_m": traf["L_m"], "P_kw": N_in, "k": k_in})
            traf["AI_%"] = reg.predict(Xbatch)
        else:
            traf["AI_%"] = np.nan

        traf["Fark_%"] = traf["AI_%"] - traf["Formül_%"]

        # Sıralama
        if sort_by.startswith("Farka"):
            traf = traf.sort_values("Fark_%", ascending=False)
        elif sort_by.startswith("AI"):
            traf = traf.sort_values("AI_%", ascending=True)
        else:
            traf = traf.sort_values("Formül_%", ascending=True)

        # Tablo
        st.dataframe(
            traf[["Montaj Yeri","Gücü[kVA]","L_m","Formül_%","AI_%","Fark_%"]]
            .rename(columns={"L_m":"L (m)","Formül_%":"Gerilim Düşümü (Formül, %)","AI_%":"Gerilim Düşümü (AI, %)","Fark_%":"Fark (AI–Formül, %)"})
            .style.format({"L (m)":"{:.0f}","Gerilim Düşümü (Formül, %)":"{:.2f}","Gerilim Düşümü (AI, %)":"{:.2f}","Fark (AI–Formül, %)":"{:+.2f}"}),
            use_container_width=True
        )

        # Bar grafik: AI vs Formül (yan yana)
        plot_df = traf[["Montaj Yeri","Formül_%","AI_%"]].melt(id_vars="Montaj Yeri",
                        var_name="Yöntem", value_name="Düşüm (%)")
        fig_bar = px.bar(plot_df, x="Montaj Yeri", y="Düşüm (%)", color="Yöntem",
                         barmode="group", template="plotly_white",
                         title=f"Seçilen Trafolar için Gerilim Düşümü — N={N_in} kW, k={k_in}")
        fig_bar.add_hline(y=thr_pct, line_dash="dot", annotation_text=f"Eşik %{thr_pct:.2f}")
        fig_bar.update_layout(xaxis_tickangle=20)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Fark grafiği
        fig_diff = px.bar(traf, x="Montaj Yeri", y="Fark_%", template="plotly_white",
                          title="AI – Formül Farkı (%, + pozitif = AI daha yüksek)")
        fig_diff.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig_diff, use_container_width=True)

# ===================== SAYFA 2: Gerilim Düşümü — Gerçek Veri & AI (Trafo Bazlı Özet) =====================
elif selected == "Gerilim Düşümü":
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
    m1.metric("📐 Formül (k·L·N)", f"%{dv_formula:.2f}")
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
        Xb = dloc[["Mesafe (m)", "Yük (kW)"]].copy()
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

# ===================== SAYFA 3: Forecasting =====================
elif selected == "Forecasting":
    st.subheader("📈 Yük Tahmini (Forecasting) — Günlük")

    # --------- Girdiler (sayfa içi) ---------
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        horizon = st.number_input("Tahmin ufku (gün)", 7, 120, 30, 1)
    with c2:
        agg = st.selectbox("Zaman toplaması", ["Günlük Ortalama", "Günlük Toplam"], index=0)
    with c3:
        season = st.selectbox("Mevsimsellik", ["Haftalık (7)", "Aylık (~30)"], index=0)
    with c4:
        holdout_days = st.number_input("Test penceresi (gün)", 7, 90, 30, 1)

    season_periods = 7 if "7" in season else 30

    # --------- Veri Hazırlama ---------
    # ext_df global'den geliyor (smart_grid_dataset.csv)
    df_raw = None
    if ext_df is not None and len(ext_df) > 0:
        # kolon adayları
        cols = {c.lower(): c for c in ext_df.columns}
        # zaman kolonu
        time_col = None
        for k in ["timestamp", "datetime", "date", "ds"]:
            if k in cols:
                time_col = cols[k]; break
        # yük kolonu
        load_col = None
        for k in ["load", "y", "power_kw", "kw", "value"]:
            if k in cols:
                load_col = cols[k]; break

        if time_col and load_col:
            df_raw = ext_df[[time_col, load_col]].rename(columns={time_col:"ds", load_col:"y"}).copy()

    # Eğer dataset uygun değilse sentetik üret
    if df_raw is None or len(df_raw) < 30:
        rng = np.random.default_rng(7)
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        base = 300 + 0.15*np.arange(len(dates))
        weekly = 35*np.sin(2*np.pi*dates.dayofweek/7)
        noise = rng.normal(0, 14, len(dates))
        df_raw = pd.DataFrame({"ds": dates, "y": base + weekly + noise})

    # Zaman temizliği
    df_raw["ds"] = pd.to_datetime(df_raw["ds"])
    df_raw = df_raw.sort_values("ds")

    # Günlük resample (mean/sum)
    rule = "D"
    if "Ortalama" in agg:
        series = df_raw.set_index("ds")["y"].resample(rule).mean().interpolate("time")
    else:
        series = df_raw.set_index("ds")["y"].resample(rule).sum().interpolate("time")

    ts = series.reset_index().rename(columns={"index":"ds"})
    ts = ts.dropna()
    if len(ts) <= holdout_days + max(14, season_periods*2):
        st.error("Zaman serisi çok kısa. Daha uzun veri sağlayın veya test penceresini küçültün.")
        st.stop()

    # Train/Test ayır
    cutoff = ts["ds"].max() - pd.Timedelta(days=int(holdout_days))
    train = ts[ts["ds"] <= cutoff].copy()
    test  = ts[ts["ds"] >  cutoff].copy()

    # --------- Model: Holt-Winters (fallback: rolling mean) ---------
    y_train = train.set_index("ds")["y"]
    y_test  = test.set_index("ds")["y"]

    fc = None
    yhat_test = None
    used_model = "Holt-Winters"

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal="add",
            seasonal_periods=season_periods
        )
        fit = model.fit(optimized=True, use_brute=True)
        # test dönemi tahmini
        yhat_test = fit.forecast(len(y_test))
        # ileri tahmin
        fut = fit.forecast(int(horizon))
        fc = pd.DataFrame({"ds": pd.date_range(ts["ds"].max() + pd.Timedelta(days=1), periods=int(horizon), freq="D"),
                           "yhat": fut.values})
        # basit güven aralığı (residual std)
        resid = (y_train - fit.fittedvalues).dropna()
        s = float(resid.std()) if len(resid) else 0.0
        fc["yhat_low"]  = fc["yhat"] - 1.96*s
        fc["yhat_high"] = fc["yhat"] + 1.96*s
    except Exception:
        used_model = "Rolling Mean"
        roll = y_train.rolling(window=season_periods, min_periods=1).mean()
        last = float(roll.iloc[-1])
        # test tahmini
        yhat_test = pd.Series(last, index=y_test.index)
        # ileri tahmin
        future_idx = pd.date_range(ts["ds"].max() + pd.Timedelta(days=1), periods=int(horizon), freq="D")
        fc = pd.DataFrame({"ds": future_idx, "yhat": last})
        fc["yhat_low"]  = fc["yhat"]
        fc["yhat_high"] = fc["yhat"]

    # --------- Metrikler ---------
    def _mape(y_true, y_pred):
        yt = np.array(y_true); yp = np.array(y_pred)
        mask = yt != 0
        if mask.sum() == 0: return np.nan
        return float(np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100)

    rmse = float(np.sqrt(np.mean((y_test.values - yhat_test.values)**2)))
    mae  = float(np.mean(np.abs(y_test.values - yhat_test.values)))
    mape = _mape(y_test.values, yhat_test.values)

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Model", used_model)
    k2.metric("RMSE", f"{rmse:,.2f}")
    k3.metric("MAE",  f"{mae:,.2f}")
    k4.metric("MAPE", f"%{mape:,.2f}" if np.isfinite(mape) else "—")

    st.divider()

    # --------- Grafik ---------
    import plotly.express as px
    hist = ts.copy()
    hist = hist.rename(columns={"y":"Gerçek"})
    fut = fc.rename(columns={"yhat":"Tahmin", "yhat_low":"Alt", "yhat_high":"Üst"})

    fig = px.line(template="plotly_white", title="Geçmiş ve İleri Tahmin")
    fig.add_scatter(x=hist["ds"], y=hist["Gerçek"], mode="lines", name="Gerçek")
    # test dönemi vurgusu
    fig.add_scatter(x=y_test.index, y=yhat_test.values, mode="lines", name="Test Tahmini")
    fig.add_scatter(x=fut["ds"], y=fut["Tahmin"], mode="lines", name="İleri Tahmin")
    fig.add_scatter(x=fut["ds"], y=fut["Alt"], mode="lines", name="Alt Band", line=dict(dash="dot"))
    fig.add_scatter(x=fut["ds"], y=fut["Üst"], mode="lines", name="Üst Band", line=dict(dash="dot"))
    fig.update_layout(xaxis_title="Tarih", yaxis_title="kW")
    st.plotly_chart(fig, use_container_width=True)

    # --------- İndirilebilir çıktı ---------
    out = fut[["ds","Tahmin","Alt","Üst"]].copy()
    out_csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Tahmini CSV indir", data=out_csv, file_name="forecast.csv", mime="text/csv")


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
