import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from shapely.geometry import LineString
from geopy.distance import geodesic

# --- Opsiyonel ---
try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except Exception:
    HAS_PYPROJ = False

# Sayfalar
st.set_page_config(page_title="Yapay Zeka ile Akıllı Dağıtım Şebekesi Tasarımı", layout="wide")
st.title("🔌 Yapay Zeka ile Akıllı Dağıtım Şebekesi Tasarımı")

selected = option_menu(
    menu_title="",
    options=["Talep Girdisi", "Gerilim Düşümü", "Yük Tahmini", "Anomali Tespiti"],
    icons=["geo-alt-fill", "activity", "graph-up", "exclamation-triangle"],
    menu_icon="cast",
    default_index=0,
    styles={
        "container": {"padding": "5!important", "background-color": "#f0f2f6"},
        "icon": {"color": "orange", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "2px", "font-weight": "600"},
        "nav-link-selected": {"background-color": "#fcd769", "color": "white", "font-weight": "700"},
    }
)

# Ek Fonksiyonlar
def get_transformers():
    if not HAS_PYPROJ:
        raise RuntimeError("pyproj mevcut değil")
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    bwd = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return fwd, bwd

def interpolate_point_along_route(route_line_xy, bwd, distance_m):
    distance_m = max(0.0, min(distance_m, route_line_xy.length))
    p = route_line_xy.interpolate(distance_m)
    x, y = p.x, p.y
    lon, lat = bwd.transform(x, y)
    return float(lat), float(lon), (x, y)

def pick_nearest_existing_pole_xy(poles_xy, target_xy):
    if not poles_xy:
        return None, None
    tx, ty = target_xy
    best = None
    best_d2 = None
    for (x, y) in poles_xy:
        d2 = (x - tx) ** 2 + (y - ty) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best = (x, y)
    return best, best_d2

def suggest_kva_from_kw(load_kw):
    target = load_kw * 1.25
    for step in [250, 400, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150]:
        if step >= target:
            return step
    return 3150

def next_two_positions(total_len_m, Lmax):
    """Analitik: hattı Lmax'a göre böl. 0->yok, 1->tek, 2->iki ara trafo."""
    if total_len_m <= Lmax:
        return []
    if total_len_m <= 2 * Lmax:
        return [min(Lmax, total_len_m - Lmax)]
    return [Lmax, total_len_m - Lmax]

# ML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

@st.cache_resource
def train_simple_models(n=3000, k_drop=0.0001, thr_pct=5.0):
    rng = np.random.default_rng(0)
    total_len = rng.uniform(50, 2000, n)
    load_kw = rng.uniform(10, 300, n)
    kva = rng.choice([250, 400, 630, 800, 1000], n)
    thr = thr_pct / 100.0
    Lmax = thr / (k_drop * load_kw)
    need_two = (total_len > 2 * Lmax).astype(int)
    pos = np.clip(Lmax, 0, total_len - Lmax)
    X = np.c_[total_len, load_kw, kva]
    cls = RandomForestClassifier(n_estimators=120, random_state=42).fit(X, need_two)
    reg = RandomForestRegressor(n_estimators=160, random_state=42).fit(X, pos)
    return cls, reg

# Veri
@st.cache_data
def load_data():
    direk_df = pd.read_excel("Direk Sorgu Sonuçları.xlsx")
    trafo_df = pd.read_excel("Trafo Sorgu Sonuçları.xlsx")
    return direk_df, trafo_df

direk_df, trafo_df = load_data()

# Sayfa 1 - Talep Girdisi
if selected == "Talep Girdisi":
    direk_clean = (
        direk_df[["AssetID", "Direk Kodu", "Enlem", "Boylam"]]
        .dropna(subset=["Enlem", "Boylam"]).copy()
    )
    direk_clean["Direk Kodu"] = direk_clean["Direk Kodu"].fillna("Bilinmiyor")

    trafo_clean = (
        trafo_df[["AssetID", "Montaj Yeri", "Gücü[kVA]", "Enlem", "Boylam"]]
        .dropna(subset=["Enlem", "Boylam"]).copy()
    )

    # Sidebar
    st.sidebar.header("⚙️ Hat Parametreleri")
    max_span = st.sidebar.number_input("Maks. direk aralığı (m)", 20, 100, 40, 5)
    snap_radius = st.sidebar.number_input("Mevcut direğe snap yarıçapı (m)", 10, 60, 30, 5)

    with st.sidebar.expander("🔧 Gelişmiş (Trafo Önerisi)"):
        drop_threshold_pct = st.number_input("Gerilim düşümü eşiği (%)", 1.0, 15.0, 5.0, 0.5)
        snap_tr_radius = st.number_input("Trafo/direk snap yarıçapı (m)", 10, 120, 50, 5)
        pf = st.number_input("Güç faktörü (pf)", 0.5, 1.0, 0.8, 0.05)
        use_ml = st.checkbox("Öneri için ML kullan (deneysel)", True)
        draw_new_poles = st.checkbox("Ara (mor) direkleri çiz", True)
        if st.button("🧹 Önbelleği Temizle ve Yeniden Çalıştır"):
            st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

    # Harita - Seçim
    st.subheader("📍 Talep Noktasını Seçin")
    center_lat = float(direk_clean["Enlem"].mean())
    center_lon = float(direk_clean["Boylam"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16, control_scale=True)

    poles_group = folium.FeatureGroup(name="Direkler (Mevcut)", show=True)
    trafos_group = folium.FeatureGroup(name="Trafos", show=True)

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
    map_data = st_folium(m, height=650, width="100%", returned_objects=["last_clicked"], key="select_map")

    if "demand_point" not in st.session_state:
        st.session_state["demand_point"] = None
    if map_data and map_data.get("last_clicked"):
        st.session_state["demand_point"] = (
            float(map_data["last_clicked"]["lat"]),
            float(map_data["last_clicked"]["lng"]),
        )
    if st.session_state["demand_point"] is None:
        st.info("📍 Lütfen haritadan bir noktaya tıklayınız."); st.stop()


    new_lat, new_lon = st.session_state["demand_point"]
    st.success(f"Yeni talep noktası: ({new_lat:.6f}, {new_lon:.6f})")

    st.subheader("⚡ Talep Edilen Yük (kW)")
    user_kw = st.slider("Talep edilen güç", 1, 300, 100, 5, key="kw_slider")
    k_drop = 0.0001

    # En yakın trafo
    def distance_to_demand(row):
        return geodesic((new_lat, new_lon), (row["Enlem"], row["Boylam"])) .meters

    trafo_local = trafo_clean.copy()
    trafo_local["Mesafe (m)"] = trafo_local.apply(distance_to_demand, axis=1)
    trafo_local["Gerilim Düşümü (%)"] = k_drop * trafo_local["Mesafe (m)"] * user_kw
    sorted_trafo = trafo_local.sort_values(by="Mesafe (m)").reset_index(drop=True)
    recommended = sorted_trafo.iloc[0]

    # Kapasite
    try:
        trafo_power_kva = float(recommended["Gücü[kVA]"])
    except Exception:
        trafo_power_kva = None

    if trafo_power_kva is not None:
        trafo_capacity_kw = trafo_power_kva * 0.8  # pf varsayılan 0.8; aşağıda kullanıcı pf ile tekrar hesaplanır
        if user_kw > trafo_capacity_kw:
            st.error(
                f"Talep {user_kw:.0f} kW, seçilen mevcut trafo kapasitesini aşıyor "
                f"({trafo_power_kva:.0f} kVA × pf=0.80 ≈ {trafo_capacity_kw:.0f} kW). "
                "Ek/yenileme trafo gerekir."
            )

    with st.expander("📈 En Yakın 5 Trafo "):
        st.dataframe(
            sorted_trafo[["Montaj Yeri", "Gücü[kVA]", "Mesafe (m)", "Gerilim Düşümü (%)"]].head(5),
            use_container_width=True,
        )

    # Hat
    m2 = folium.Map(location=[new_lat, new_lon], zoom_start=16, control_scale=True)
    for _, r in direk_clean.iterrows():
        folium.CircleMarker([r["Enlem"], r["Boylam"]], radius=4, color="blue",
                            fill=True, fill_opacity=0.7,
                            tooltip=f"Direk: {r['Direk Kodu']}",
                            popup=f"AssetID: {r['AssetID']}").add_to(m2)
    for _, r in trafo_local.iterrows():
        folium.Marker([r["Enlem"], r["Boylam"]],
                      tooltip=f"Trafo: {r['Montaj Yeri']}",
                      popup=f"Güç: {r['Gücü[kVA]']} kVA\nAssetID: {r['AssetID']}",
                      icon=folium.Icon(color="orange", icon="bolt", prefix="fa")).add_to(m2)

    try:
        fwd, bwd = get_transformers()
        to_xy = lambda lon, lat: fwd.transform(lon, lat)
        to_lonlat = lambda x, y: bwd.transform(x, y)

        demand_xy = to_xy(new_lon, new_lat)
        trafo_xy = to_xy(float(recommended["Boylam"]), float(recommended["Enlem"]))
        line_xy = LineString([demand_xy, trafo_xy])

        poles_xy = [to_xy(lon, lat) for lon, lat in zip(direk_clean["Boylam"], direk_clean["Enlem"]) ]

        length = line_xy.length
        distances = list(np.arange(0, length, float(max_span))) + [length]
        pts = [line_xy.interpolate(d) for d in distances]

        route_xy, used_idx = [], set()
        for p in pts:
            px, py = p.x, p.y
            best_i, best_d2 = None, None
            for i, (x, y) in enumerate(poles_xy):
                d2 = (x - px) ** 2 + (y - py) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2, best_i = d2, i
            if best_i is not None and (best_d2 ** 0.5) <= float(snap_radius):
                route_xy.append(poles_xy[best_i]); used_idx.add(best_i)
            else:
                route_xy.append((px, py))

        if route_xy:
            route_xy[0] = demand_xy
            route_xy[-1] = trafo_xy

        final_path = [(to_lonlat(x, y)[1], to_lonlat(x, y)[0]) for (x, y) in route_xy]

        route_line_xy = LineString(route_xy)
        total_len_m = route_line_xy.length
        spans = [LineString(route_xy[i:i + 2]).length for i in range(len(route_xy) - 1)]
        avg_span = float(np.mean(spans)) if spans else 0.0
        used_count = len(used_idx)
        proposed_count = max(0, len(route_xy) - used_count - 2)

        if draw_new_poles:
            used_xy_set = set(poles_xy[i] for i in used_idx)
            for (x, y), (lat, lon) in zip(route_xy, final_path):
                if (x, y) not in used_xy_set:
                    folium.CircleMarker((lat, lon), radius=5, color="purple",
                                        fill=True, fill_opacity=0.9,
                                        tooltip="Önerilen Yeni Direk").add_to(m2)

    except Exception:
        final_path = [(new_lat, new_lon),
                      (float(recommended["Enlem"]), float(recommended["Boylam"]))]
        total_len_m = geodesic((new_lat, new_lon),
                               (float(recommended["Enlem"]), float(recommended["Boylam"])) ).meters
        avg_span = total_len_m
        used_count = 0
        proposed_count = 1
        route_line_xy = None
        poles_xy = []

    folium.Marker((new_lat, new_lon), icon=folium.Icon(color="red"), tooltip="Talep Noktası").add_to(m2)
    folium.Marker((float(recommended["Enlem"]), float(recommended["Boylam"])),
                  icon=folium.Icon(color="orange", icon="bolt", prefix="fa"),
                  tooltip="Seçilen Mevcut Trafo").add_to(m2)

    if len(final_path) >= 2:
        folium.PolyLine(final_path, color="green", weight=4, opacity=0.9,
                        tooltip=f"Hat uzunluğu ≈ {total_len_m:.1f} m").add_to(m2)
    else:
        st.warning("Hat noktaları üretilemedi.")

    st.subheader("🧾 Hat Özeti")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Uzunluk", f"{total_len_m:.1f} m")
    c2.metric("Kullanılan Mevcut Direk", f"{used_count}")
    c3.metric("Önerilen Yeni Direk", f"{proposed_count}")
    c4.metric("Ortalama Direk Aralığı", f"{avg_span:.1f} m")

    if recommended["Gerilim Düşümü (%)"] > 5.0:
        st.error(f"⚠️ Gerilim düşümü %{recommended['Gerilim Düşümü (%)']:.2f} — yeni trafo gerekebilir.")
    else:
        st.success(f"✅ En uygun trafo: {recommended['Montaj Yeri']}, %{recommended['Gerilim Düşümü (%)']:.2f} gerilim düşümü")

    # Yeni Trafo
    if (trafo_power_kva is not None) and (trafo_power_kva > 400) and (route_line_xy is not None):
        Lmax = (drop_threshold_pct / 100.0) / (k_drop * float(user_kw))
        pos_list = next_two_positions(total_len_m, Lmax)

        if use_ml:
            try:
                cls, reg = train_simple_models(k_drop=k_drop, thr_pct=drop_threshold_pct)
                X_now = np.array([[total_len_m, float(user_kw), trafo_power_kva]])
                need_two_ml = bool(cls.predict(X_now)[0])
                if not need_two_ml:
                    dpos_ml = float(reg.predict(X_now)[0])
                    if len(pos_list) == 1:
                        pos_list = [0.5 * (pos_list[0] + dpos_ml)]
                    elif len(pos_list) == 0:
                        pos_list = [dpos_ml]
                else:
                    if len(pos_list) < 2:
                        pos_list = [min(Lmax, total_len_m / 2), max(total_len_m - Lmax, total_len_m / 2)]
            except Exception:
                st.info("ML destekli öneri çalışmadı; analitik sonuç kullanılıyor.")

        if len(pos_list) == 0:
            st.info(f"Bu hat tek parçada %{drop_threshold_pct:.1f} eşiğin altında; yeni trafoya gerek yok.")
        else:
            try:
                fwd, bwd = get_transformers()
                _ = poles_xy
            except Exception:
                fwd, bwd = None, None

            suggestions = []
            for dpos in pos_list:
                if fwd and bwd and route_line_xy is not None:
                    lat, lon, (tx, ty) = interpolate_point_along_route(route_line_xy, bwd, dpos)
                    nearest_xy, d2 = pick_nearest_existing_pole_xy(poles_xy, (tx, ty))
                    snapped = False
                    if nearest_xy is not None and (d2 ** 0.5) <= float(snap_tr_radius):
                        nx, ny = nearest_xy
                        nlon, nlat = bwd.transform(nx, ny)
                        lat, lon = float(nlat), float(nlon)
                        snapped = True
                else:
                    lat, lon, snapped = np.nan, np.nan, False

                suggestions.append({
                    "Sıra": len(suggestions) + 1,
                    "Hat Boyunca Konum (m)": round(float(dpos), 1),
                    "Lat": round(float(lat), 6) if pd.notna(lat) else None,
                    "Lon": round(float(lon), 6) if pd.notna(lon) else None,
                    "Konum Türü": "Mevcut direğe snap" if snapped else "Yeni konum",
                    "Tahmini kVA": suggest_kva_from_kw(float(user_kw)),
                })

            capacity_note = ""
            if trafo_power_kva is not None:
                cap_kw = trafo_power_kva * pf
                if user_kw > cap_kw:
                    capacity_note = " (mevcut trafo kapasitesi AŞILIYOR)"

            st.subheader("🔋 Önerilen Yeni Trafolar")
            df_sug = pd.DataFrame(
                suggestions,
                columns=["Sıra", "Hat Boyunca Konum (m)", "Lat", "Lon", "Konum Türü", "Tahmini kVA"],
            )
            st.dataframe(df_sug, use_container_width=True)

        
            if len(pos_list) == 1:
                st.warning(
                    f"Mevcut trafo {trafo_power_kva:.0f} kVA{capacity_note}. "
                    f"%{drop_threshold_pct:.1f} gerilim eşiğine göre **1 yeni trafo** önerildi."
                )
            else:
                st.error(
                    f"Mevcut trafo {trafo_power_kva:.0f} kVA{capacity_note}. "
                    f"%{drop_threshold_pct:.1f} gerilim eşiğine göre **2 yeni trafo** önerildi."
                )
    else:
        st.info("Yeni trafo öneri tablosu yalnızca **mevcut trafo gücü > 400 kVA** olduğunda üretilir.")

    st.subheader("📡 Oluşturulan Şebeke Hattı")
    st_folium(m2, height=650, width="100%", key="result_map")

# Sayfa 2 - Gerilim Düşünmü
elif selected == "Gerilim Düşümü":
    st.subheader("📉 Gerilim Düşümü Tahmini")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    import plotly.express as px

    direk_clean = direk_df.dropna(subset=["Enlem", "Boylam"]).copy()
    trafo_names = trafo_df["Montaj Yeri"].dropna().unique()
    if len(trafo_names) == 0:
        st.error("Trafo verisi bulunamadı."); st.stop()

    trafo_sec = st.selectbox("🔌 Trafo Seçin", options=trafo_names)
    trafo_row = trafo_df[trafo_df["Montaj Yeri"] == trafo_sec].iloc[0]
    trafo_coord = (float(trafo_row["Enlem"]), float(trafo_row["Boylam"]))
    trafo_power = float(trafo_row["Gücü[kVA]"])

    direk_clean["Mesafe (m)"] = direk_clean.apply(
        lambda r: geodesic((r["Enlem"], r["Boylam"]), trafo_coord).meters, axis=1
    )
    rng = np.random.default_rng(42)
    direk_clean["Yük (kW)"] = rng.integers(10, 300, size=len(direk_clean))
    direk_clean["Trafo_Gucu (kVA)"] = trafo_power

    k = 0.0001
    direk_clean["Gerilim Düşümü (%)"] = k * direk_clean["Mesafe (m)"] * direk_clean["Yük (kW)"]

    X = direk_clean[["Mesafe (m)", "Yük (kW)", "Trafo_Gucu (kVA)"]]
    y = direk_clean["Gerilim Düşümü (%)"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    st.markdown(f"**R²:** `{r2:.4f}` — **MSE:** `{mse:.6f}`")

    import plotly.express as px
    chart_df = pd.DataFrame({"Gerçek (%)": y_test.values[:200], "Tahmin (%)": y_pred[:200]})
    fig = px.line(chart_df, markers=True, template="plotly_white",
                  title="Gerilim Düşümü (%) — Tahmin vs Gerçek")
    fig.update_layout(yaxis_title="Gerilim Düşümü (%)", xaxis_title="Veri Noktası", title_font_size=20)
    st.plotly_chart(fig, use_container_width=True)

    # SHAP
    try:
        import shap
        with st.expander("🔎 SHAP açıklamaları (isteğe bağlı)"):
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test.iloc[:200])
            shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns)
            mean_abs = shap_df.abs().mean().sort_values(ascending=False).reset_index()
            mean_abs.columns = ["Özellik", "Ortalama |SHAP|"]
            fig2 = px.bar(mean_abs, x="Özellik", y="Ortalama |SHAP|", text_auto=True,
                          title="Özellik Etkileri (SHAP — Ortalama Mutlak)", template="plotly_white")
            fig2.update_layout(yaxis_title="Etki Büyüklüğü", xaxis_title="Özellik")
            st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        st.info("SHAP yüklenemedi (ortam desteklemiyor olabilir).")

# Yük Tahmini
elif selected == "Yük Tahmini":
    st.subheader("📈 Yük Tahmini (Sentetik Veri, Saatlik)")

    import plotly.express as px
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    # Sentetik veri üretimi
    rng = np.random.default_rng(123)
    hours = pd.date_range("2025-01-01", periods=24*120, freq="H")  # 120 gün saatlik
    df = pd.DataFrame({"ts": hours})
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["month"] = df["ts"].dt.month

    # Sıcaklık
    base_temp = 15 + 10*np.sin(2*np.pi*(df["ts"].dt.dayofyear/365))
    daily_cycle = 3*np.sin(2*np.pi*(df["hour"]/24))
    df["temp"] = base_temp + daily_cycle + rng.normal(0, 1.0, len(df))

    # Talep
    workday = (df["dow"] < 5).astype(int)
    peak = 20 + 8*np.sin(2*np.pi*(df["hour"]/24 - 0.2))  # akşam tepe
    season = 10 + 6*np.sin(2*np.pi*(df["ts"].dt.dayofyear/365 - 0.1))
    noise = rng.normal(0, 1.5, len(df))
    df["load_kw"] = 50 + peak + season + 0.8*(25 - abs(df["temp"]-25)) + 2*workday + noise
    df["load_kw"] = df["load_kw"].clip(lower=10)

    # Train
    feat = ["hour", "dow", "month", "temp", "load_lag1", "load_lag24", "load_lag168"]
    df["load_lag1"] = df["load_kw"].shift(1)
    df["load_lag24"] = df["load_kw"].shift(24)
    df["load_lag168"] = df["load_kw"].shift(168)
    df = df.dropna().reset_index(drop=True)

    split_idx = int(len(df)*0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train[feat], train["load_kw"]
    X_test, y_test = test[feat], test["load_kw"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    st.markdown(f"**MAE:** `{mae:.2f} kW`  — veri: 120 gün saatlik (sentetik)")

    # Grafik: gerçek vs tahmin
    show_last = 24*7
    plot_df = pd.DataFrame({
        "Tarih": test["ts"].iloc[-show_last:],
        "Gerçek (kW)": y_test.iloc[-show_last:].values,
        "Tahmin (kW)": y_pred[-show_last:]
    })
    fig = px.line(plot_df, x="Tarih", y=["Gerçek (kW)", "Tahmin (kW)"], markers=True, template="plotly_white",
                  title="Son 7 Gün — Gerçek vs Tahmin")
    fig.update_layout(legend_title=None, xaxis_title="", yaxis_title="kW")
    st.plotly_chart(fig, use_container_width=True)

    # SHAP
    try:
        import shap
        with st.expander("🧠 SHAP Açıklamaları"):
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test.iloc[:300])
            shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns)
            mean_abs = shap_df.abs().mean().sort_values(ascending=False).reset_index()
            mean_abs.columns = ["Özellik", "Ortalama |SHAP|"]
            fig2 = px.bar(mean_abs, x="Özellik", y="Ortalama |SHAP|", text_auto=True,
                          title="Özellik Etkileri (SHAP — Ortalama Mutlak)", template="plotly_white")
            fig2.update_layout(yaxis_title="Etki Büyüklüğü", xaxis_title="Özellik")
            st.plotly_chart(fig2, use_container_width=True)

            # Tek gözlem açıklaması (tool-tip gibi)
            idx = st.slider("Örnek index (test)", 0, len(X_test)-1, 0)
            single = X_test.iloc[[idx]]
            single_pred = model.predict(single)[0]
            single_sv = explainer(single)
            contrib = pd.Series(single_sv.values[0], index=X_train.columns).sort_values(key=lambda x: -abs(x))
            topk = contrib.head(8).reset_index()
            topk.columns = ["Özellik", "SHAP Katkısı"]
            fig3 = px.bar(topk, x="Özellik", y="SHAP Katkısı", text_auto=True,
                          title=f"Seçili Nokta Tahmin Açıklaması — Tahmin: {single_pred:.1f} kW", template="plotly_white")
            fig3.update_layout(yaxis_title="Katkı (±)", xaxis_title="Özellik")
            st.plotly_chart(fig3, use_container_width=True)
    except Exception:
        st.info("SHAP yüklenemedi (ortam desteklemiyor olabilir).")

# Anomali / Arıza Tespiti
elif selected == "Anomali Tespiti":
    st.subheader("🚨 Anomali / Arıza Tespiti (Residual + IsolationForest)")

    import plotly.express as px
    from sklearn.ensemble import IsolationForest
    from sklearn.ensemble import RandomForestRegressor

    # Aynı sentetik veriyi kullan (yük tahmini ile tutarlı)
    rng = np.random.default_rng(999)
    hours = pd.date_range("2025-01-01", periods=24*60, freq="H")
    df = pd.DataFrame({"ts": hours})
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["month"] = df["ts"].dt.month
    base_temp = 15 + 10*np.sin(2*np.pi*(df["ts"].dt.dayofyear/365))
    daily_cycle = 3*np.sin(2*np.pi*(df["hour"]/24))
    df["temp"] = base_temp + daily_cycle + rng.normal(0, 1.0, len(df))
    workday = (df["dow"] < 5).astype(int)
    peak = 20 + 8*np.sin(2*np.pi*(df["hour"]/24 - 0.2))
    season = 10 + 6*np.sin(2*np.pi*(df["ts"].dt.dayofyear/365 - 0.1))
    noise = rng.normal(0, 1.5, len(df))
    df["load_kw"] = 50 + peak + season + 0.8*(25 - abs(df["temp"]-25)) + 2*workday + noise
    df["load_kw"] = df["load_kw"].clip(lower=10)

    # Yapay arızalar / anomaliler (spike & dip)
    idx_spike = rng.choice(len(df), size=15, replace=False)
    df.loc[idx_spike, "load_kw"] += rng.uniform(15, 30, size=len(idx_spike))
    idx_dip = rng.choice(len(df), size=10, replace=False)
    df.loc[idx_dip, "load_kw"] -= rng.uniform(12, 24, size=len(idx_dip))

    # Özellikler + model
    df["load_lag1"] = df["load_kw"].shift(1)
    df["load_lag24"] = df["load_kw"].shift(24)
    df["load_lag168"] = df["load_kw"].shift(168)
    df = df.dropna().reset_index(drop=True)
    feat = ["hour", "dow", "month", "temp", "load_lag1", "load_lag24", "load_lag168"]

    split_idx = int(len(df)*0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    X_train, y_train = train[feat], train["load_kw"]
    X_test, y_test = test[feat], test["load_kw"]

    fmodel = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
    y_hat = fmodel.predict(X_test)
    residual = y_test.values - y_hat

    # Residual z-score
    res_mu, res_sigma = residual.mean(), residual.std() + 1e-6
    z = (residual - res_mu) / res_sigma
    res_flag = (np.abs(z) > 3.0)  # eşiği slider yap
    thr = st.slider("Z-skor eşiği", 2.0, 5.0, 3.0, 0.1)
    res_flag = (np.abs(z) > thr)

    # IsolationForest
    iso = IsolationForest(contamination=0.02, random_state=42).fit(X_train)
    iso_flag = iso.predict(X_test) == -1

    out = test[["ts"]].copy()
    out["Gerçek (kW)"] = y_test.values
    out["Tahmin (kW)"] = y_hat
    out["Residual"] = residual
    out["Z-skor"] = z
    out["Anomali (Z)"] = res_flag
    out["Anomali (IF)"] = iso_flag
    out["Anomali"] = out["Anomali (Z)"] | out["Anomali (IF)"]

    # Grafik
    fig = px.line(out, x="ts", y=["Gerçek (kW)", "Tahmin (kW)"], template="plotly_white",
                  title="Gerçek vs Tahmin — Anomali İşaretleri")
    # Nokta işaretleri
    anomalies = out[out["Anomali"]]
    if not anomalies.empty:
        fig.add_scatter(x=anomalies["ts"], y=anomalies["Gerçek (kW)"],
                        mode="markers", name="Anomali",
                        marker=dict(size=9, symbol="x"))
    fig.update_layout(xaxis_title="", yaxis_title="kW", legend_title=None)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Not: Anomali = (|residual| z-skoru > eşik) ∨ (IsolationForest = anomali).")

