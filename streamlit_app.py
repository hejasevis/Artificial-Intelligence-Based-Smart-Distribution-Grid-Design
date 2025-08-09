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

# ===================== SAYFA 2: Gerilim Düşümü (k·L·N) =====================
elif selected == "Gerilim Düşümü":
    st.subheader("📉 Gerilim Düşümü Analizi — k·L·N Modeli")
    st.caption("Bu sayfa, senin istediğin formülle çalışır: **Gerilim Düşümü (%) = k × Hat Uzunluğu (L, m) × Yük (N, kW)**")

    with st.expander("Model Özeti", expanded=True):
        st.markdown("""
- **Formül:** `Gerilim Düşümü (%) = k · L · N`  
  - `k`: sabit (saha/iletken/şartlara göre ayarlanır)  
  - `L`: hat uzunluğu (metre)  
  - `N`: yük (kW)  
- **Amaç:** L ve N değiştikçe gerilim düşümünün nasıl davrandığını görmek; hangi **eşik** altında kaldığımızı görsel olarak ayırt etmek.
        """)

    # --- Kontroller (sol)
    with st.sidebar.expander("🔧 Parametreler", expanded=True):
        k_const = st.number_input("k sabiti", 0.0, 1.0, 0.0001, 0.0001, key="gd_k")
        drop_threshold_pct = st.number_input("Eşik (Gerilim Düşümü, %)", 0.5, 20.0, 5.0, 0.5, key="gd_thr")

        st.markdown("**Kısayol (örnek ayarlar)**")
        c1, c2 = st.columns(2)
        if c1.button("Konut (düşük k)", use_container_width=True):
            k_const = 0.00008
        if c2.button("Sanayi (yüksek k)", use_container_width=True):
            k_const = 0.00015

        st.divider()
        # Tekil kesit incelemeleri için seçimler
        L_fixed = st.slider("Sabit L (m) — P'ye karşı eğri", 10, 3000, 400, 10)
        P_fixed = st.slider("Sabit N (kW) — L'ye karşı eğri", 1, 800, 150, 1)

    # --- Veri ızgarası
    Ls    = np.linspace(10, 3000, 150)   # m
    loads = np.linspace(1,  800,  150)   # kW
    L_grid, P_grid = np.meshgrid(Ls, loads)
    dv_grid = k_const * L_grid * P_grid  # k·L·N

    # ========= ÜST KARTLAR =========
    max_dv   = float(np.nanmax(dv_grid))
    median_dv= float(np.nanmedian(dv_grid))
    feas_pct = float((dv_grid <= drop_threshold_pct).sum() / dv_grid.size * 100)

    m1, m2, m3 = st.columns(3)
    m1.metric("Maks. Gerilim Düşümü", f"%{max_dv:.2f}")
    m2.metric("Medyan Gerilim Düşümü", f"%{median_dv:.2f}")
    m3.metric("Eşik Altı Alan", f"%{feas_pct:.1f}")

    st.divider()

    # ========= A) Eşik Alanı Haritası (uygun/uygunsuz) =========
    st.markdown("### 🗺️ Eşik Alanı — Uygun/Uygunsuz (Isı Haritası)")
    feas = (dv_grid <= drop_threshold_pct).astype(int)
    # 0=Uygunsuz(>eşik), 1=Uygun(≤eşik). Görselleştirmede 0/1 yerine etiket kullanacağız.
    feas_df = pd.DataFrame(feas, index=[f"{int(p)}" for p in loads], columns=[f"{int(l)}" for l in Ls])
    fig_feas = px.imshow(
        feas_df.values,
        x=feas_df.columns, y=feas_df.index,
        aspect="auto",
        color_continuous_scale=[[0, "#f94144"], [1, "#90be6d"]],
        title=f"Eşik: %{drop_threshold_pct:.2f} — Yeşil=Uygun, Kırmızı=Uygunsuz",
        labels=dict(x="Hat Uzunluğu L (m)", y="Yük N (kW)", color="Uygunluk"),
        origin="lower"
    )
    fig_feas.update_coloraxes(showscale=False)
    st.plotly_chart(fig_feas, use_container_width=True)

    st.caption("**Not:** k sabiti büyüdükçe veya eşik küçüldükçe yeşil alan daralır.")

    st.divider()

    # ========= B) Tekil Kesitler: L'ye ve N'e karşı eğriler =========
    st.markdown("### 📈 Kesit Grafikleri (Okunabilir Karşılaştırma)")

    # (1) Sabit N=P_fixed iken L'ye karşı
    dv_L_curve = k_const * Ls * P_fixed
    fig_ln_L = px.line(
        x=Ls, y=dv_L_curve, markers=True, template="plotly_white",
        title=f"Gerilim Düşümü (%) — Sabit Yük: {P_fixed} kW (L'ye karşı)"
    )
    fig_ln_L.add_hline(y=drop_threshold_pct, line_dash="dot", annotation_text=f"Eşik %{drop_threshold_pct:.2f}")
    fig_ln_L.update_layout(xaxis_title="Hat Uzunluğu L (m)", yaxis_title="Gerilim Düşümü (%)")

    # (2) Sabit L=L_fixed iken N'e karşı
    dv_P_curve = k_const * L_fixed * loads
    fig_ln_P = px.line(
        x=loads, y=dv_P_curve, markers=True, template="plotly_white",
        title=f"Gerilim Düşümü (%) — Sabit Hat Uzunluğu: {L_fixed} m (N'e karşı)"
    )
    fig_ln_P.add_hline(y=drop_threshold_pct, line_dash="dot", annotation_text=f"Eşik %{drop_threshold_pct:.2f}")
    fig_ln_P.update_layout(xaxis_title="Yük N (kW)", yaxis_title="Gerilim Düşümü (%)")

    cL, cP = st.columns(2)
    cL.plotly_chart(fig_ln_L, use_container_width=True)
    cP.plotly_chart(fig_ln_P, use_container_width=True)

    st.divider()

    # ========= C) Senaryo Karşılaştırma (A/B) =========
    st.markdown("### 🥊 Senaryo Karşılaştırma (A/B)")

    sA1, sA2, sA3 = st.columns(3)
    with sA1: L_A = st.number_input("Senaryo A — L (m)", 10, 5000, 600, 10)
    with sA2: P_A = st.number_input("Senaryo A — N (kW)", 1,  2000, 200, 1)
    with sA3: k_A = st.number_input("Senaryo A — k", 0.0, 1.0, float(k_const), 0.0001)

    sB1, sB2, sB3 = st.columns(3)
    with sB1: L_B = st.number_input("Senaryo B — L (m)", 10, 5000, 400, 10)
    with sB2: P_B = st.number_input("Senaryo B — N (kW)", 1,  2000, 120, 1)
    with sB3: k_B = st.number_input("Senaryo B — k", 0.0, 1.0, float(k_const), 0.0001)

    dv_A = k_A * L_A * P_A
    dv_B = k_B * L_B * P_B

    a1, a2, a3 = st.columns(3)
    a1.metric("A — Gerilim Düşümü", f"%{dv_A:.2f}")
    a2.metric("B — Gerilim Düşümü", f"%{dv_B:.2f}")
    diff_txt = "A daha kötü" if dv_A > dv_B else ("B daha kötü" if dv_B > dv_A else "Eşit")
    a3.metric("Karar", diff_txt)

    # Çubuk grafikte göster
    comp_df = pd.DataFrame({
        "Senaryo": ["A", "B"],
        "Gerilim Düşümü (%)": [dv_A, dv_B]
    })
    fig_bar = px.bar(comp_df, x="Senaryo", y="Gerilim Düşümü (%)", text_auto=True, template="plotly_white",
                     title="A/B Karşılaştırma")
    fig_bar.add_hline(y=drop_threshold_pct, line_dash="dot", annotation_text=f"Eşik %{drop_threshold_pct:.2f}")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.info("👉 İpucu: Eşiği, k’yı, L ve N değerlerini oynatarak hangi parametrenin düşümü daha hızlı artırdığını sezgisel görün.")


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
