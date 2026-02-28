import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

STD_CUSTOMER = "customer_id"
ANALYSES_DIR = "data/analyses"


def rfm_path(dataset_id: str) -> str:
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_rfm.parquet")


def clusters_path(dataset_id: str) -> str:
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_clusters.parquet")


def clusters_meta_path(dataset_id: str) -> str:
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_clusters_meta.json")


def load_rfm(dataset_id: str) -> pd.DataFrame | None:
    df_rfm = st.session_state.get("df_rfm")
    if df_rfm is not None and not df_rfm.empty:
        return df_rfm
    p = rfm_path(dataset_id)
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None


def save_clusters_to_disk(df_clusters: pd.DataFrame, meta: dict, dataset_id: str) -> None:
    os.makedirs(ANALYSES_DIR, exist_ok=True)
    df_clusters.to_parquet(clusters_path(dataset_id), index=False)
    with open(clusters_meta_path(dataset_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_clusters_from_disk(dataset_id: str) -> tuple[pd.DataFrame | None, dict | None]:
    p = clusters_path(dataset_id)
    mp = clusters_meta_path(dataset_id)
    df = pd.read_parquet(p) if os.path.exists(p) else None
    meta = None
    if os.path.exists(mp):
        with open(mp, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return df, meta


def delete_clusters_from_disk(dataset_id: str) -> None:
    p = clusters_path(dataset_id)
    mp = clusters_meta_path(dataset_id)
    if os.path.exists(p):
        os.remove(p)
    if os.path.exists(mp):
        os.remove(mp)


def prepare_features(df_rfm: pd.DataFrame, features: list[str], scaler_name: str):
    X = df_rfm[features].copy()
    X = X.replace([float("inf"), float("-inf")], pd.NA).dropna()

    scaler = StandardScaler() if scaler_name == "StandardScaler" else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled


def compute_silhouette(X_scaled, labels) -> float | None:
    # silhouette requires at least 2 clusters (excluding noise label -1)
    uniq = set(labels)
    non_noise = [u for u in uniq if u != -1]
    if len(non_noise) < 2:
        return None
    # also need enough samples
    if len(labels) <= len(non_noise):
        return None
    try:
        return float(silhouette_score(X_scaled, labels))
    except Exception:
        return None


def add_cluster_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def to_label(x):
        try:
            xi = int(x)
        except Exception:
            return str(x)
        return "Noise / Outliers" if xi == -1 else f"Cluster {xi}"
    out["cluster_label"] = out["cluster"].apply(to_label)
    out["is_noise"] = out["cluster"].astype(int) == -1
    return out


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

Podporované algoritmy:
- **K-Means** (centroid-based, vyžaduje počet klastrov k)
- **DBSCAN** (density-based, vie nájsť aj *noise/outliers*)
- **Hierarchical / Agglomerative** (hierarchické zhlukovanie)

Výsledok sa ukladá do histórie (`data/analyses/`).
"""
)

# ---------------- Controls ----------------
st.subheader("Nastavenie segmentácie")

features_all = ["recency", "frequency", "monetary", "R_score", "F_score", "M_score", "RFM_sum"]
default_features = ["recency", "frequency", "monetary"]

features = st.multiselect(
    "Vyber vstupné znaky (features)",
    options=features_all,
    default=default_features
)

scaler_name = st.selectbox("Normalizácia", ["StandardScaler", "MinMaxScaler"], index=0)

algorithm = st.selectbox(
    "Algoritmus",
    ["K-Means", "DBSCAN", "Hierarchical (Agglomerative)"],
    index=0
)

# Parameters per algorithm
k = None
random_state = None
eps = None
min_samples = None
linkage = None
metric = None

if algorithm == "K-Means":
    k = st.slider("Počet klastrov (k)", min_value=2, max_value=10, value=4)
    random_state = st.number_input("Random state", min_value=0, max_value=100_000, value=42, step=1)

elif algorithm == "DBSCAN":
    st.caption("Tip: DBSCAN je citlivý na škálovanie, preto normalizácia je dôležitá.")
    eps = st.slider("eps (radius)", min_value=0.05, max_value=5.0, value=0.5, step=0.05)
    min_samples = st.slider("min_samples", min_value=2, max_value=50, value=10, step=1)

else:  # Hierarchical
    k = st.slider("Počet klastrov (k)", min_value=2, max_value=10, value=4)
    linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"], index=0)
    # ward requires euclidean
    if linkage == "ward":
        metric = "euclidean"
        st.info("Linkage 'ward' vyžaduje metrickú vzdialenosť 'euclidean'.")
    else:
        metric = st.selectbox("Metric", ["euclidean", "manhattan", "cosine"], index=0)

saved_exists = os.path.exists(clusters_path(dataset_id))

col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 2])
with col1:
    run_cluster = st.button("▶️ Spustiť segmentáciu", type="primary", disabled=(len(features) < 2))
with col2:
    load_saved = st.button("♻️ Načítať uložené klastre", disabled=not saved_exists)
with col3:
    delete_saved = st.button("🗑 Zmazať uložené klastre", disabled=not saved_exists)
with col4:
    st.caption(f"Aktívny dataset ID: `{dataset_id}` | uložené klastre: {'áno' if saved_exists else 'nie'}")

# ---------------- Delete / Load ----------------
if delete_saved:
    delete_clusters_from_disk(dataset_id)
    st.session_state.pop("df_clusters", None)
    st.session_state.pop("cluster_meta", None)
    st.success("Uložené klastre boli zmazané.")
    st.rerun()

if load_saved:
    loaded_df, loaded_meta = load_clusters_from_disk(dataset_id)
    if loaded_df is None or loaded_df.empty:
        st.warning("Nepodarilo sa načítať uložené klastre.")
    else:
        st.session_state["df_clusters"] = loaded_df
        st.session_state["cluster_meta"] = loaded_meta or {}
        st.success("Uložené klastre boli načítané (bez prepočtu).")

# Autoload once
if "df_clusters" not in st.session_state and saved_exists:
    auto_df, auto_meta = load_clusters_from_disk(dataset_id)
    if auto_df is not None and not auto_df.empty:
        st.session_state["df_clusters"] = auto_df
        st.session_state["cluster_meta"] = auto_meta or {}
        st.info("Našiel som uložené klastre a načítal som ich automaticky.")

# ---------------- Run segmentation ----------------
if run_cluster:
    X_raw, X_scaled = prepare_features(df_rfm, features, scaler_name)
    df_aligned = df_rfm.loc[X_raw.index].copy()

    inertia = None
    labels = None

    if algorithm == "K-Means":
        model = KMeans(n_clusters=int(k), random_state=int(random_state), n_init="auto")
        labels = model.fit_predict(X_scaled)
        inertia = float(model.inertia_)

        meta = {
            "algorithm": "K-Means",
            "k": int(k),
            "random_state": int(random_state),
            "scaler": scaler_name,
            "features": features,
            "inertia": inertia,
        }

    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = model.fit_predict(X_scaled)

        meta = {
            "algorithm": "DBSCAN",
            "eps": float(eps),
            "min_samples": int(min_samples),
            "scaler": scaler_name,
            "features": features,
            "inertia": None,
        }

    else:  # Hierarchical / Agglomerative
        # sklearn API: metric parameter (ward only euclidean)
        try:
            model = AgglomerativeClustering(n_clusters=int(k), linkage=str(linkage), metric=str(metric))
        except TypeError:
            # older versions used 'affinity'
            model = AgglomerativeClustering(n_clusters=int(k), linkage=str(linkage), affinity=str(metric))

        labels = model.fit_predict(X_scaled)

        meta = {
            "algorithm": "Hierarchical (Agglomerative)",
            "k": int(k),
            "linkage": str(linkage),
            "metric": str(metric),
            "scaler": scaler_name,
            "features": features,
            "inertia": None,
        }

    # Build output
    df_clusters = df_aligned.copy()
    df_clusters["cluster"] = labels
    df_clusters = add_cluster_label(df_clusters)

    # Quality metrics
    sil = compute_silhouette(X_scaled, labels)
    meta["silhouette"] = sil

    st.session_state["df_clusters"] = df_clusters
    st.session_state["cluster_meta"] = meta

    save_clusters_to_disk(df_clusters, meta, dataset_id)
    st.success("Segmentácia bola vykonaná a uložená do histórie.")

# ---------------- Outputs ----------------
df_clusters = st.session_state.get("df_clusters")
meta = st.session_state.get("cluster_meta", {})

if df_clusters is None or df_clusters.empty:
    st.warning("Zatiaľ nie sú k dispozícii klastre. Klikni na **Spustiť segmentáciu** alebo načítaj uložené.")
    st.stop()

st.subheader("Kvalita a konfigurácia modelu")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Algoritmus", meta.get("algorithm", "—"))
c2.metric("Inertia", f"{meta['inertia']:.2f}" if isinstance(meta.get("inertia"), (int, float)) else "—")
sil_val = meta.get("silhouette")
c3.metric("Silhouette", f"{sil_val:.3f}" if isinstance(sil_val, float) else "—")
c4.metric("Scaler", meta.get("scaler", "—"))
st.caption(f"Features: {meta.get('features', [])}")

# Quick counts (incl. noise)
st.subheader("Počty zákazníkov v klastroch")
counts = df_clusters["cluster_label"].value_counts().reset_index()
counts.columns = ["cluster_label", "customers"]
st.dataframe(counts, use_container_width=True)

st.subheader("Zákazníci s priradeným klastrom (top 50)")
sort_cols = ["cluster", "monetary"] if "monetary" in df_clusters.columns else ["cluster"]
st.dataframe(
    df_clusters.sort_values(sort_cols, ascending=[True, False] if len(sort_cols) == 2 else True).head(50),
    use_container_width=True,
)

# Cluster profiles
st.subheader("Profil klastrov (priemery)")
profile = (
    df_clusters.groupby("cluster_label")
    .agg(
        customers=(STD_CUSTOMER, "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_R=("R_score", "mean"),
        avg_F=("F_score", "mean"),
        avg_M=("M_score", "mean"),
        avg_RFM_sum=("RFM_sum", "mean"),
        noise=("is_noise", "sum"),
    )
    .reset_index()
    .sort_values("customers", ascending=False)
)
st.dataframe(profile, use_container_width=True)

# Visualizations
st.subheader("Vizualizácia klastrov")

colA, colB = st.columns(2)
with colA:
    x_axis = st.selectbox("X axis", ["recency", "frequency", "monetary"], index=0)
with colB:
    y_axis = st.selectbox("Y axis", ["recency", "frequency", "monetary"], index=2)

hover_cols = [STD_CUSTOMER]
for c in ["RFM_score", "RFM_sum", "Segment_label", "cluster_label"]:
    if c in df_clusters.columns:
        hover_cols.append(c)

fig2d = px.scatter(
    df_clusters,
    x=x_axis,
    y=y_axis,
    color="cluster_label",
    hover_data=hover_cols,
    title="2D scatter (colored by cluster)",
)
st.plotly_chart(fig2d, use_container_width=True)

fig3d = px.scatter_3d(
    df_clusters,
    x="recency",
    y="frequency",
    z="monetary",
    color="cluster_label",
    hover_data=hover_cols,
    title="3D scatter (R, F, M)",
)
st.plotly_chart(fig3d, use_container_width=True)

st.info("✅ Segmentácia je hotová. Ďalšie kroky: **Trendy** a **Marketing Insights**.")