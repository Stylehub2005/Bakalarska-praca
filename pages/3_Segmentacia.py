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
SETTINGS_PATH = "data/settings.json"

DEFAULT_SETTINGS = {
    "rfm_weights": {"R": 1.0, "F": 1.0, "M": 1.0},
    "default_scaler": "StandardScaler",
    "auto_k": {"k_min": 2, "k_max": 10},
    "segmentation_default_algorithm": "K-Means",
}


# ---------------- SETTINGS ----------------

def load_settings():
    s = st.session_state.get("settings")

    if isinstance(s, dict):
        merged = DEFAULT_SETTINGS.copy()
        merged.update(s)
        merged["rfm_weights"] = {**DEFAULT_SETTINGS["rfm_weights"], **merged.get("rfm_weights", {})}
        merged["auto_k"] = {**DEFAULT_SETTINGS["auto_k"], **merged.get("auto_k", {})}
        return merged

    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            s2 = json.load(f)

        merged = DEFAULT_SETTINGS.copy()
        merged.update(s2)
        st.session_state["settings"] = merged
        return merged

    return DEFAULT_SETTINGS


# ---------------- PATHS ----------------

def rfm_path(dataset_id):
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_rfm.parquet")


def clusters_path(dataset_id):
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_clusters.parquet")


# ---------------- LOAD RFM ----------------

def load_rfm(dataset_id):

    df_rfm = st.session_state.get("df_rfm")

    if df_rfm is not None:
        return df_rfm

    path = rfm_path(dataset_id)

    if os.path.exists(path):
        return pd.read_parquet(path)

    return None


# ---------------- SCALER ----------------

def scaler_from_name(name):

    if name == "StandardScaler":
        return StandardScaler()

    return MinMaxScaler()


# ---------------- FEATURE PREP ----------------

def prepare_features(df_rfm, features, scaler_name, weights):

    X = df_rfm[features].copy()

    for col in features:

        if "recency" in col:
            X[col] *= weights.get("R", 1)

        if "frequency" in col:
            X[col] *= weights.get("F", 1)

        if "monetary" in col:
            X[col] *= weights.get("M", 1)

    scaler = scaler_from_name(scaler_name)

    X_scaled = scaler.fit_transform(X)

    return X, X_scaled


# ---------------- AUTO K ----------------

def compute_k_metrics(X_scaled, k_min, k_max):

    results = []

    for k in range(k_min, k_max + 1):

        model = KMeans(n_clusters=k, random_state=42, n_init="auto")

        labels = model.fit_predict(X_scaled)

        inertia = model.inertia_

        sil = None

        try:
            sil = silhouette_score(X_scaled, labels)
        except:
            pass

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette": sil
        })

    return pd.DataFrame(results)


# ---------------- UI ----------------

st.title("🧠 Segmentácia zákazníkov")

settings = load_settings()

weights = settings["rfm_weights"]

dataset_id = st.session_state.get("active_dataset_id")

if not dataset_id:
    st.warning("No active dataset")
    st.stop()

df_rfm = load_rfm(dataset_id)

if df_rfm is None:
    st.warning("Run RFM analysis first")
    st.stop()


# ---------------- CONTROLS ----------------

st.subheader("Segmentation settings")

features_all = [
    "recency",
    "frequency",
    "monetary",
    "R_score",
    "F_score",
    "M_score",
    "RFM_sum",
    "RFM_weighted_sum",
]

features = st.multiselect(
    "Features",
    features_all,
    default=["recency", "frequency", "monetary"]
)

scaler_name = st.selectbox(
    "Scaler",
    ["StandardScaler", "MinMaxScaler"]
)

algorithm = st.selectbox(
    "Algorithm",
    ["K-Means", "DBSCAN", "Hierarchical"]
)

k_min = settings["auto_k"]["k_min"]
k_max = settings["auto_k"]["k_max"]

k = None

if algorithm in ["K-Means", "Hierarchical"]:
    k = st.slider("Number of clusters", k_min, k_max, 4)

run_cluster = st.button("▶️ Run segmentation")

auto_k_button = st.button("🔍 Auto detect optimal k")


# ---------------- AUTO K ----------------

if auto_k_button:

    X_raw, X_scaled = prepare_features(df_rfm, features, scaler_name, weights)

    k_metrics = compute_k_metrics(X_scaled, k_min, k_max)

    st.session_state["k_metrics"] = k_metrics

    best_row = k_metrics.sort_values("silhouette", ascending=False).iloc[0]

    st.session_state["best_k"] = int(best_row["k"])


k_metrics = st.session_state.get("k_metrics")

if k_metrics is not None:

    st.subheader("Auto-k analysis")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.line(
            k_metrics,
            x="k",
            y="inertia",
            markers=True,
            title="Elbow method"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.line(
            k_metrics,
            x="k",
            y="silhouette",
            markers=True,
            title="Silhouette score"
        )

        st.plotly_chart(fig, use_container_width=True)

    best_k = st.session_state.get("best_k")

    if best_k:
        st.success(f"Recommended k = {best_k}")


# ---------------- RUN SEGMENTATION ----------------

if run_cluster:

    X_raw, X_scaled = prepare_features(df_rfm, features, scaler_name, weights)

    if algorithm == "K-Means":

        model = KMeans(n_clusters=k, random_state=42, n_init="auto")

        labels = model.fit_predict(X_scaled)

    elif algorithm == "DBSCAN":

        model = DBSCAN()

        labels = model.fit_predict(X_scaled)

    else:

        # FIXED HIERARCHICAL CLUSTERING

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
            metric="euclidean",
            distance_threshold=None
        )

        labels = model.fit_predict(X_scaled)

    df_clusters = df_rfm.copy()

    df_clusters["cluster"] = labels

    st.session_state["df_clusters"] = df_clusters

    st.success("Segmentation completed")


df_clusters = st.session_state.get("df_clusters")

if df_clusters is None:
    st.stop()


# ---------------- CLUSTER SIZE ----------------

st.subheader("Cluster sizes")

counts = df_clusters["cluster"].value_counts().reset_index()

counts.columns = ["cluster", "customers"]

st.dataframe(counts, use_container_width=True)

fig = px.bar(
    counts,
    x="cluster",
    y="customers",
    title="Cluster size"
)

st.plotly_chart(fig, use_container_width=True)


# ---------------- PROFILE ----------------

st.subheader("Cluster profile")

profile = (
    df_clusters
    .groupby("cluster")
    .agg(
        customers=(STD_CUSTOMER, "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    )
    .reset_index()
)

st.dataframe(profile, use_container_width=True)


# ---------------- VISUALIZATION ----------------

st.subheader("Cluster visualization")

fig = px.scatter(
    df_clusters,
    x="recency",
    y="monetary",
    color="cluster",
)

st.plotly_chart(fig, use_container_width=True)

fig3d = px.scatter_3d(
    df_clusters,
    x="recency",
    y="frequency",
    z="monetary",
    color="cluster",
)

st.plotly_chart(fig3d, use_container_width=True)

st.success("Segmentation ready")