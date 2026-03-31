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
        return merged

    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            s2 = json.load(f)
        merged = DEFAULT_SETTINGS.copy()
        merged.update(s2)
        st.session_state["settings"] = merged
        return merged

    return DEFAULT_SETTINGS


# ---------------- LOAD ----------------

def load_rfm(dataset_id):
    df = st.session_state.get("df_rfm")
    if df is not None:
        return df
    path = os.path.join(ANALYSES_DIR, f"{dataset_id}_rfm.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


# ---------------- PREP ----------------

def scaler_from_name(name):
    return StandardScaler() if name == "StandardScaler" else MinMaxScaler()


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

    return X_scaled


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

# ✅ ДОБАВЛЕНО ОБЪЯСНЕНИЕ
st.markdown("""
## 🎯 Na čo slúži táto stránka?

Táto stránka umožňuje rozdeliť zákazníkov do skupín (**segmentov**), ktoré majú podobné správanie.

👉 Výsledok:
- skupiny zákazníkov (napr. VIP, low-value, rizikoví)
- lepšie cielenie marketingu
- pochopenie zákazníckeho správania

---

## ⚙️ Čo tu nastavuješ?

Tu ovplyvňuješ:
- **aké dáta použijeme (features)**
- **ako ich upravíme (scaler)**
- **aký algoritmus použijeme**
- **koľko segmentov vytvoríme (k)**

💡 Každá zmena → iný výsledok segmentácie.
""")

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

st.subheader("⚙️ Segmentation settings")

# ✅ FEATURES EXPLAINED
st.markdown("""
### 📊 Features (premenné)

Vyberáš, podľa čoho sa budú zákazníci deliť:

- **recency** → ako nedávno nakúpil  
- **frequency** → ako často nakupuje  
- **monetary** → koľko míňa  

💡 Odporúčanie: použi RFM (recency + frequency + monetary)
""")

features_all = [
    "recency", "frequency", "monetary",
    "R_score", "F_score", "M_score",
    "RFM_sum", "RFM_weighted_sum"
]

features = st.multiselect(
    "Features",
    features_all,
    default=["recency", "frequency", "monetary"]
)

# ✅ SCALER EXPLAINED
st.markdown("""
### ⚖️ Scaler (normalizácia)

Upravuje rozsah dát:

- **StandardScaler** → štandardný (najčastejšie)  
- **MinMaxScaler** → škáluje na 0–1  

👉 Dôležité, lebo bez toho by napr. monetary dominovalo.
""")

scaler_name = st.selectbox(
    "Scaler",
    ["StandardScaler", "MinMaxScaler"]
)

# ✅ ALGORITHM EXPLAINED
st.markdown("""
### 🤖 Algorithm

- **K-Means** → najčastejší, potrebuje počet klastrov (k)  
- **DBSCAN** → hľadá hustoty (nájde outliers)  
- **Hierarchical** → stromová segmentácia  

💡 Odporúčanie: začni s K-Means
""")

algorithm = st.selectbox(
    "Algorithm",
    ["K-Means", "DBSCAN", "Hierarchical"]
)

k_min = settings["auto_k"]["k_min"]
k_max = settings["auto_k"]["k_max"]

k = None
if algorithm in ["K-Means", "Hierarchical"]:
    st.markdown("""
### 🔢 Number of clusters (k)

Koľko skupín chceš vytvoriť.

- malé k → veľké skupiny  
- veľké k → detailné segmenty  

👉 Použi Auto-k pre odporúčanie.
""")

    k = st.slider("Number of clusters", k_min, k_max, 4)

# ---------------- BUTTONS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    run_cluster = st.button("▶️ Run segmentation", type="primary")

with col2:
    auto_k_button = st.button("🔍 Auto detect optimal k")

with col3:
    reset_button = st.button("🔄 Reset")

# ---------------- RESET ----------------

if reset_button:
    st.session_state.pop("df_clusters", None)
    st.session_state.pop("k_metrics", None)
    st.session_state.pop("best_k", None)

    st.success("Settings reset")
    st.rerun()

# ---------------- WARNING ----------------

if "df_clusters" in st.session_state:
    st.info("⚠️ Segmentation already computed. Press RESET to change parameters.")

# ---------------- AUTO K ----------------

if auto_k_button:
    X_scaled = prepare_features(df_rfm, features, scaler_name, weights)

    k_metrics = compute_k_metrics(X_scaled, k_min, k_max)

    st.session_state["k_metrics"] = k_metrics

    best = k_metrics.sort_values("silhouette", ascending=False).iloc[0]
    st.session_state["best_k"] = int(best["k"])

k_metrics = st.session_state.get("k_metrics")

if k_metrics is not None:

    st.subheader("📊 Auto-k analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(px.line(k_metrics, x="k", y="inertia", markers=True))

    with col2:
        st.plotly_chart(px.line(k_metrics, x="k", y="silhouette", markers=True))

    best_k = st.session_state.get("best_k")
    if best_k:
        st.success(f"Recommended k = {best_k}")

# ---------------- RUN ----------------

if run_cluster:

    X_scaled = prepare_features(df_rfm, features, scaler_name, weights)

    if algorithm == "K-Means":
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")

    elif algorithm == "DBSCAN":
        model = DBSCAN()

    else:
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

# ---------------- OUTPUT ----------------

df_clusters = st.session_state.get("df_clusters")

if df_clusters is None:
    st.stop()

st.subheader("Cluster sizes")

counts = df_clusters["cluster"].value_counts().reset_index()
counts.columns = ["cluster", "customers"]

st.dataframe(counts)
st.plotly_chart(px.bar(counts, x="cluster", y="customers"))

st.subheader("Cluster profile")

profile = df_clusters.groupby("cluster").agg(
    customers=(STD_CUSTOMER, "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
).reset_index()

st.dataframe(profile)

st.subheader("Visualization")

st.plotly_chart(px.scatter(df_clusters, x="recency", y="monetary", color="cluster"))

st.plotly_chart(px.scatter_3d(
    df_clusters,
    x="recency",
    y="frequency",
    z="monetary",
    color="cluster"
))

st.success("Segmentation ready")