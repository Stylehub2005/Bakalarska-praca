import os
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

STD_CUSTOMER = "customer_id"
ANALYSES_DIR = "data/analyses"


def rfm_path(dataset_id: str) -> str:
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_rfm.parquet")


def clusters_path(dataset_id: str) -> str:
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_clusters.parquet")


def load_rfm(dataset_id: str) -> pd.DataFrame | None:
    # Prefer session, else load from disk
    df_rfm = st.session_state.get("df_rfm")
    if df_rfm is not None and not df_rfm.empty:
        return df_rfm
    p = rfm_path(dataset_id)
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None


def save_clusters_to_disk(df_clusters: pd.DataFrame, dataset_id: str) -> None:
    os.makedirs(ANALYSES_DIR, exist_ok=True)
    df_clusters.to_parquet(clusters_path(dataset_id), index=False)


def load_clusters_from_disk(dataset_id: str) -> pd.DataFrame | None:
    p = clusters_path(dataset_id)
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None


def delete_clusters_from_disk(dataset_id: str) -> None:
    p = clusters_path(dataset_id)
    if os.path.exists(p):
        os.remove(p)


def prepare_features(df_rfm: pd.DataFrame, features: list[str], scaler_name: str):
    X = df_rfm[features].copy()

    # Basic safety: drop inf / nan
    X = X.replace([float("inf"), float("-inf")], pd.NA).dropna()

    scaler = StandardScaler() if scaler_name == "StandardScaler" else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled


st.title("🧠 Segmentácia zákazníkov")

dataset_id = st.session_state.get("active_dataset_id")
if not dataset_id:
    st.warning("Nie je nastavený aktívny dataset. Vráť sa na stránku načítania dát a vyber dataset.")
    st.stop()

df_rfm = load_rfm(dataset_id)
if df_rfm is None or df_rfm.empty:
    st.warning("Najprv spusti RFM analýzu (stránka **RFM analýza**).")
    st.stop()

st.markdown(
    """
Táto stránka vykoná **zhlukovú analýzu** zákazníkov na základe RFM metrík.
Výsledné klastre predstavujú segmenty zákazníkov s podobným nákupným správaním a kúpnu silou.
Výsledok sa ukladá do histórie (`data/analyses/`).
"""
)

# --- Controls
st.subheader("Nastavenie segmentácie")

features_all = ["recency", "frequency", "monetary", "R_score", "F_score", "M_score", "RFM_sum"]
default_features = ["recency", "frequency", "monetary"]

features = st.multiselect(
    "Vyber vstupné znaky (features)",
    options=features_all,
    default=default_features
)

scaler_name = st.selectbox("Normalizácia", ["StandardScaler", "MinMaxScaler"], index=0)
k = st.slider("Počet klastrov (k)", min_value=2, max_value=10, value=4)

random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)

saved_exists = os.path.exists(clusters_path(dataset_id))

col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 2])
with col1:
    run_kmeans = st.button("▶️ Spustiť K-Means", type="primary", disabled=(len(features) < 2))
with col2:
    load_saved = st.button("♻️ Načítať uložené klastre", disabled=not saved_exists)
with col3:
    delete_saved = st.button("🗑 Zmazať uložené klastre", disabled=not saved_exists)
with col4:
    st.caption(f"Aktívny dataset ID: `{dataset_id}` | uložené klastre: {'áno' if saved_exists else 'nie'}")

if delete_saved:
    delete_clusters_from_disk(dataset_id)
    st.session_state.pop("df_clusters", None)
    st.success("Uložené klastre boli zmazané.")
    st.rerun()

if load_saved:
    loaded = load_clusters_from_disk(dataset_id)
    if loaded is None or loaded.empty:
        st.warning("Nepodarilo sa načítať uložené klastre.")
    else:
        st.session_state["df_clusters"] = loaded
        st.success("Uložené klastre boli načítané (bez prepočtu).")

# Autoload once
if "df_clusters" not in st.session_state and saved_exists:
    auto = load_clusters_from_disk(dataset_id)
    if auto is not None and not auto.empty:
        st.session_state["df_clusters"] = auto
        st.info("Našiel som uložené klastre a načítal som ich automaticky.")

# Run k-means
if run_kmeans:
    X_raw, X_scaled = prepare_features(df_rfm, features, scaler_name)

    # Align df_rfm to X_raw index after dropping NaNs
    df_aligned = df_rfm.loc[X_raw.index].copy()

    model = KMeans(n_clusters=k, random_state=int(random_state), n_init="auto")
    labels = model.fit_predict(X_scaled)

    df_clusters = df_aligned.copy()
    df_clusters["cluster"] = labels

    # Quality metrics
    inertia = float(model.inertia_)
    sil = None
    if k >= 2 and len(df_clusters) > k:
        try:
            sil = float(silhouette_score(X_scaled, labels))
        except Exception:
            sil = None

    # Store metrics in session
    st.session_state["cluster_metrics"] = {
        "k": k,
        "features": features,
        "scaler": scaler_name,
        "inertia": inertia,
        "silhouette": sil,
    }

    st.session_state["df_clusters"] = df_clusters
    save_clusters_to_disk(df_clusters, dataset_id)

    st.success("K-Means segmentácia bola vykonaná a uložená do histórie.")

df_clusters = st.session_state.get("df_clusters")
if df_clusters is None or df_clusters.empty:
    st.warning("Zatiaľ nie sú k dispozícii klastre. Klikni na **Spustiť K-Means** alebo načítaj uložené.")
    st.stop()

# --- Outputs
st.subheader("Kvalita modelu")
metrics = st.session_state.get("cluster_metrics", {})
c1, c2, c3 = st.columns(3)
c1.metric("k", str(metrics.get("k", "—")))
c2.metric("Inertia", f"{metrics.get('inertia', float('nan')):.2f}" if "inertia" in metrics else "—")
sil_val = metrics.get("silhouette", None)
c3.metric("Silhouette", f"{sil_val:.3f}" if isinstance(sil_val, float) else "—")

st.subheader("Zákazníci s priradeným klastrom (top 50)")
st.dataframe(df_clusters.sort_values(["cluster", "monetary"], ascending=[True, False]).head(50),
             use_container_width=True)

# Cluster profiles
st.subheader("Profil klastrov (priemery)")
cluster_profile = (
    df_clusters.groupby("cluster")
    .agg(
        customers=(STD_CUSTOMER, "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_R=("R_score", "mean"),
        avg_F=("F_score", "mean"),
        avg_M=("M_score", "mean"),
        avg_RFM_sum=("RFM_sum", "mean"),
    )
    .reset_index()
    .sort_values("customers", ascending=False)
)
st.dataframe(cluster_profile, use_container_width=True)

# Visualizations
st.subheader("Vizualizácia klastrov")

# 2D scatter (choose axes)
colA, colB = st.columns(2)
with colA:
    x_axis = st.selectbox("X axis", ["recency", "frequency", "monetary"], index=0)
with colB:
    y_axis = st.selectbox("Y axis", ["recency", "frequency", "monetary"], index=2)

fig2d = px.scatter(
    df_clusters,
    x=x_axis,
    y=y_axis,
    color="cluster",
    hover_data=[STD_CUSTOMER, "RFM_score", "RFM_sum", "Segment_label"],
    title="2D scatter (colored by cluster)"
)
st.plotly_chart(fig2d, use_container_width=True)

# 3D scatter
fig3d = px.scatter_3d(
    df_clusters,
    x="recency",
    y="frequency",
    z="monetary",
    color="cluster",
    hover_data=[STD_CUSTOMER, "RFM_score", "RFM_sum", "Segment_label"],
    title="3D scatter (R, F, M)"
)
st.plotly_chart(fig3d, use_container_width=True)

st.info("✅ Segmentácia je hotová. Ďalšie kroky: **Trendy** a **Marketing Insights**.")