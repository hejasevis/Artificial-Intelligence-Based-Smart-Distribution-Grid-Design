import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from shapely.geometry import LineString
from geopy.distance import geodesic

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except Exception:
    HAS_PYPROJ = False

st.set_page_config(page_title="Akıllı Şebeke (AI) – Entegrasyon", layout="wide")
st.title("🔌 Akıllı Şebeke (AI) – Entegrasyonlu Uygulama")

@st.cache_data
def load_csvs():
    direk = pd.read_csv("direkler.csv")
    trafo = pd.read_csv("trafolar.csv")
    try:
        olcum = pd.read_csv("olcum_yuk.csv", parse_dates=["timestamp"])
    except Exception:
        olcum = None
    return direk, trafo, olcum

direk_df, trafo_df, olcum_df = load_csvs()

selected = option_menu(
    menu_title="",
    options=["Talep Girdisi","Gerilim Düşümü","Yük Tahmini","Anomali Tespiti"],
    icons=["geo-alt-fill","activity","graph-up","exclamation-triangle"],
    menu_icon="cast",
    default_index=0
)

def get_transformers():
    if not HAS_PYPROJ:
        raise RuntimeError("pyproj mevcut değil")
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    bwd = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return fwd, bwd

def suggest_kVA(load_kw):
    for step in [250,400,630,800,1000,1250,1600]:
        if step >= load_kw*1.25: return step
    return 2000

# --- Talep Girdisi (aynı mantık, CSV'lerle) ---
if selected == "Talep Girdisi":
    st.sidebar.header("⚙️ Hat Parametreleri")
    max_span = st.sidebar.number_input("Maks. direk aralığı (m)", 20, 100, 40, 5)
    snap_radius = st.sidebar.number_input("Mevcut direğe snap yarıçapı (m)", 10, 60, 30, 5)
    with st.sidebar.expander("🔧 Gelişmiş"):
        drop_threshold_pct = st.number_input("Gerilim düşümü eşiği (%)", 1.0, 15.0, 5.0, 0.5)
        snap_tr_radius = st.number_input("Trafo/direk snap yarıçapı (m)", 10, 120, 50, 5)
        pf = st.number_input("Güç faktörü (pf)", 0.5, 1.0, 0.8, 0.05)

    direk_clean = direk_df.dropna(subset=["Enlem","Boylam"]).copy()
    trafo_clean = trafo_df.dropna(subset=["Enlem","Boylam","Gucu_kVA"]).copy()

    st.subheader("📍 Talep Noktası Seç")
    center = [float(direk_clean["Enlem"].mean()), float(direk_clean["Boylam"].mean())]
    m = folium.Map(location=center, zoom_start=15, control_scale=True)
    for _, r in direk_clean.iterrows():
        folium.CircleMarker([r["Enlem"],r["Boylam"]], radius=4, color="blue", fill=True).add_to(m)
    for _, r in trafo_clean.iterrows():
        folium.Marker([r["Enlem"],r["Boylam"]], tooltip=f"{r['Montaj Yeri']} ({int(r['Gucu_kVA'])} kVA)",
                      icon=folium.Icon(color="orange", icon="bolt", prefix="fa")).add_to(m)
    m.add_child(folium.LatLngPopup())
    map_data = st_folium(m, height=550, width="100%", returned_objects=["last_clicked"], key="dem")

    if map_data and map_data.get("last_clicked"):
        new_lat = float(map_data["last_clicked"]["lat"])
        new_lon = float(map_data["last_clicked"]["lng"])
        st.success(f"Talep: ({new_lat:.6f}, {new_lon:.6f})")
        user_kw = st.slider("Talep edilen güç (kW)", 1, 300, 120, 5)
        k_drop = 1e-4

        # en yakın trafo
        trafo_clean["Mesafe (m)"] = trafo_clean.apply(lambda r: geodesic((new_lat,new_lon),(r["Enlem"],r["Boylam"])).meters, axis=1)
        trafo_clean["Gerilim Düşümü (%)"] = k_drop * trafo_clean["Mesafe (m)"] * user_kw
        rec = trafo_clean.sort_values("Mesafe (m)").iloc[0]

        trafo_capacity_kw = rec["Gucu_kVA"] * pf
        if user_kw > trafo_capacity_kw:
            st.error(f"Talep {user_kw:.0f} kW, mevcut trafo kapasitesini aşıyor (≈{trafo_capacity_kw:.0f} kW).")

        # Bilgi kutuları
        c1,c2,c3 = st.columns(3)
        c1.metric("En Yakın Trafo", rec["Montaj Yeri"])
        c2.metric("Trafo Gücü", f"{int(rec['Gucu_kVA'])} kVA")
        c3.metric("ΔV (tahmini)", f"%{rec['Gerilim Düşümü (%)']:.2f}")

        # 400 kVA kuralı → yeni trafo öner
        if rec["Gucu_kVA"] > 400 and rec["Gerilim Düşümü (%)"] > drop_threshold_pct:
            st.warning(f"Mevcut trafo {int(rec['Gucu_kVA'])} kVA. %{drop_threshold_pct:.1f} eşiğine göre **ek trafo** gerekir.")
            st.info("Not: Hat uzunluğu metinde gösterilmiyor; karar gerilim eşiği ve kapasiteye göre verildi.")

# --- Gerilim Düşümü (CSV ile) ---
elif selected == "Gerilim Düşümü":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    import plotly.express as px

    trafo_names = trafo_df["Montaj Yeri"].dropna().unique()
    if len(trafo_names)==0: st.stop()
    trafo_sec = st.selectbox("🔌 Trafo", trafo_names)
    trow = trafo_df[trafo_df["Montaj Yeri"]==trafo_sec].iloc[0]
    tcoord = (float(trow["Enlem"]), float(trow["Boylam"]))
    tp = float(trow["Gucu_kVA"])

    d = direk_df.dropna(subset=["Enlem","Boylam"]).copy()
    d["Mesafe (m)"] = d.apply(lambda r: geodesic((r["Enlem"],r["Boylam"]), tcoord).meters, axis=1)
    rng = np.random.default_rng(42)
    d["Yük (kW)"] = rng.integers(10, 300, size=len(d))
    d["Trafo_Gucu (kVA)"] = tp
    k = 1e-4
    d["Gerilim Düşümü (%)"] = k * d["Mesafe (m)"] * d["Yük (kW)"]

    X = d[["Mesafe (m)","Yük (kW)","Trafo_Gucu (kVA)"]]
    y = d["Gerilim Düşümü (%)"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor(n_estimators=120, random_state=42).fit(Xtr,ytr)
    yhat = model.predict(Xte)

    r2 = r2_score(yte,yhat); mse = mean_squared_error(yte,yhat)
    st.markdown(f"**R²:** `{r2:.3f}` — **MSE:** `{mse:.6f}`")

    chart = pd.DataFrame({"Gerçek (%)": yte.values[:200], "Tahmin (%)": yhat[:200]})
    fig = px.line(chart, markers=True, template="plotly_white", title="ΔV — Tahmin vs Gerçek")
    st.plotly_chart(fig, use_container_width=True)

    try:
        import shap
        with st.expander("🔎 SHAP"):
            explainer = shap.Explainer(model, Xtr)
            sv = explainer(Xte.iloc[:150])
            shap_df = pd.DataFrame(sv.values, columns=Xtr.columns).abs().mean().sort_values(ascending=False).reset_index()
            shap_df.columns = ["Özellik","Ortalama |SHAP|"]
            fig2 = px.bar(shap_df, x="Özellik", y="Ortalama |SHAP|", text_auto=True, template="plotly_white",
                          title="Özellik Etkileri (SHAP)")
            st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        st.info("SHAP ortamda devre dışı.")

# --- Yük Tahmini (smart_grid_dataset normalize edilirse) ---
elif selected == "Yük Tahmini":
    import plotly.express as px
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    if olcum_df is None or olcum_df.empty:
        st.info("olcum_yuk.csv bulunamadı.")
    else:
        df = olcum_df.copy()
        df["hour"] = df["timestamp"].dt.hour
        df["dow"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["load_lag1"] = df["P_kW"].shift(1)
        df["load_lag24"] = df["P_kW"].shift(24)
        df["load_lag168"] = df["P_kW"].shift(168)
        df = df.dropna().reset_index(drop=True)
        feat = ["hour","dow","month","load_lag1","load_lag24","load_lag168"]

        split = int(len(df)*0.8)
        train, test = df.iloc[:split], df.iloc[split:]
        Xtr, ytr = train[feat], train["P_kW"]
        Xte, yte = test[feat], test["P_kW"]

        model = RandomForestRegressor(n_estimators=200, random_state=42).fit(Xtr,ytr)
        yhat = model.predict(Xte)
        mae = mean_absolute_error(yte,yhat)
        st.markdown(f"**MAE:** `{mae:.2f} kW`")

        show_last = min(24*7, len(test))
        plot_df = pd.DataFrame({"Tarih": test["timestamp"].iloc[-show_last:],
                                "Gerçek (kW)": yte.iloc[-show_last:].values,
                                "Tahmin (kW)": yhat[-show_last:]})
        fig = px.line(plot_df, x="Tarih", y=["Gerçek (kW)","Tahmin (kW)"], markers=True, template="plotly_white",
                      title="Gerçek vs Tahmin (Son 7 Gün)")
        st.plotly_chart(fig, use_container_width=True)

        try:
            import shap
            with st.expander("🧠 SHAP"):
                explainer = shap.Explainer(model, Xtr)
                sv = explainer(Xte.iloc[:300])
                shap_df = pd.DataFrame(sv.values, columns=Xtr.columns).abs().mean().sort_values(ascending=False).reset_index()
                shap_df.columns = ["Özellik","Ortalama |SHAP|"]
                fig2 = px.bar(shap_df, x="Özellik", y="Ortalama |SHAP|", text_auto=True, template="plotly_white",
                              title="Özellik Etkileri (SHAP — Yük Tahmini)")
                st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.info("SHAP ortamda devre dışı.")

# --- Anomali Tespiti (ölçümden) ---
elif selected == "Anomali Tespiti":
    import plotly.express as px
    from sklearn.ensemble import IsolationForest
    from sklearn.ensemble import RandomForestRegressor

    if olcum_df is None or olcum_df.empty:
        st.info("olcum_yuk.csv bulunamadı.")
    else:
        df = olcum_df.copy().sort_values("timestamp")
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        df["dow"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
        df["month"] = pd.to_datetime(df["timestamp"]).dt.month
        df["load_lag1"] = df["P_kW"].shift(1)
        df["load_lag24"] = df["P_kW"].shift(24)
        df["load_lag168"] = df["P_kW"].shift(168)
        df = df.dropna().reset_index(drop=True)
        feat = ["hour","dow","month","load_lag1","load_lag24","load_lag168"]

        split = int(len(df)*0.8)
        train, test = df.iloc[:split], df.iloc[split:]
        Xtr, ytr = train[feat], train["P_kW"]
        Xte, yte = test[feat], test["P_kW"]

        fmodel = RandomForestRegressor(n_estimators=200, random_state=42).fit(Xtr,ytr)
        yhat = fmodel.predict(Xte)
        residual = yte.values - yhat
        z = (residual - residual.mean()) / (residual.std() + 1e-6)

        thr = st.slider("Z-skor eşiği", 2.0, 5.0, 3.0, 0.1)
        z_flag = (np.abs(z) > thr)

        iso = IsolationForest(contamination=0.02, random_state=42).fit(Xtr)
        iso_flag = (iso.predict(Xte) == -1)

        out = test[["timestamp"]].copy()
        out["Gerçek (kW)"] = yte.values
        out["Tahmin (kW)"] = yhat
        out["Z-skor"] = z
        out["Anomali"] = z_flag | iso_flag

        fig = px.line(out, x="timestamp", y=["Gerçek (kW)","Tahmin (kW)"], template="plotly_white",
                      title="Anomali İşaretleri")
        anomalies = out[out["Anomali"]]
        if not anomalies.empty:
            fig.add_scatter(x=anomalies["timestamp"], y=anomalies["Gerçek (kW)"],
                            mode="markers", name="Anomali")
        st.plotly_chart(fig, use_container_width=True)
