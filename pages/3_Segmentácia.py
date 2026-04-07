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


def load_rfm(dataset_id):
    df = st.session_state.get("df_rfm")
    if df is not None:
        return df
    path = os.path.join(ANALYSES_DIR, f"{dataset_id}_rfm.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


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


def compute_k_metrics(X_scaled, k_min, k_max):
    results = []

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)

        inertia = model.inertia_

        sil = None
        try:
            sil = silhouette_score(X_scaled, labels)
        except Exception:
            pass

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette": sil
        })

    df = pd.DataFrame(results).sort_values("k").reset_index(drop=True)

    df["inertia_drop"] = df["inertia"].shift(1) - df["inertia"]
    df["inertia_drop_pct"] = (df["inertia_drop"] / df["inertia"].shift(1) * 100).round(2)

    elbow_scores = [None] * len(df)
    for i in range(1, len(df) - 1):
        prev_drop = df.loc[i - 1, "inertia"] - df.loc[i, "inertia"]
        next_drop = df.loc[i, "inertia"] - df.loc[i + 1, "inertia"]
        elbow_scores[i] = prev_drop - next_drop

    df["elbow_score"] = elbow_scores

    return df


def _normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.nunique(dropna=True) <= 1:
        return pd.Series([1.0] * len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def recommend_best_k(k_metrics: pd.DataFrame) -> tuple[int | None, pd.DataFrame]:
    df = k_metrics.copy()

    valid = df.dropna(subset=["silhouette"]).copy()
    if valid.empty:
        return None, df

    sil_k2 = valid.loc[valid["k"] == 2, "silhouette"]
    sil_k2 = float(sil_k2.iloc[0]) if not sil_k2.empty else None

    valid_3plus = valid[valid["k"] >= 3].copy()

    if not valid_3plus.empty:
        valid_3plus["silhouette_norm"] = _normalize_series(valid_3plus["silhouette"])
        valid_3plus["elbow_score_filled"] = valid_3plus["elbow_score"].fillna(0)
        valid_3plus["elbow_norm"] = _normalize_series(valid_3plus["elbow_score_filled"])
        valid_3plus["recommendation_score"] = (
            0.6 * valid_3plus["silhouette_norm"] + 0.4 * valid_3plus["elbow_norm"]
        )

        best_3plus = valid_3plus.sort_values(
            ["recommendation_score", "silhouette", "elbow_score"],
            ascending=False
        ).iloc[0]

        best_3plus_k = int(best_3plus["k"])
        best_3plus_sil = float(best_3plus["silhouette"])

        # Ak je k=2 iba mierne lepšie ako 3+, preferujeme interpretovateľnejšie riešenie s 3+ klastrami.
        if sil_k2 is not None and (sil_k2 - best_3plus_sil) > 0.15:
            return 2, df

        return best_3plus_k, df

    best = valid.sort_values("silhouette", ascending=False).iloc[0]
    return int(best["k"]), df


st.title("🧠 Segmentácia zákazníkov")

st.markdown("""
## 🎯 Na čo slúži táto stránka?

Táto stránka umožňuje rozdeliť zákazníkov do skupín (**segmentov**), ktoré majú podobné správanie.

👉 Výsledok:
- skupiny zákazníkov
- lepšie cielenie marketingu
- pochopenie zákazníckeho správania
""")

settings = load_settings()
weights = settings["rfm_weights"]

dataset_id = st.session_state.get("active_dataset_id")

if not dataset_id:
    st.warning("Nie je aktívny dataset")
    st.stop()

df_rfm = load_rfm(dataset_id)

if df_rfm is None:
    st.warning("Najprv spusti RFM analýzu")
    st.stop()

st.subheader("⚙️ Nastavenia segmentácie")

features_all = [
    "recency", "frequency", "monetary",
    "R_score", "F_score", "M_score",
    "RFM_sum", "RFM_weighted_sum"
]

features = st.multiselect(
    "Premenné (features)",
    features_all,
    default=["recency", "frequency", "monetary"]
)

scaler_name = st.selectbox(
    "Normalizácia (scaler)",
    ["StandardScaler", "MinMaxScaler"]
)

algorithm = st.selectbox(
    "Algoritmus",
    ["K-Means", "DBSCAN", "Hierarchical"]
)

k_min = settings["auto_k"]["k_min"]
k_max = settings["auto_k"]["k_max"]

k = None
if algorithm in ["K-Means", "Hierarchical"]:
    k = st.slider("Počet klastrov (k)", k_min, k_max, 4)

col1, col2, col3 = st.columns(3)

with col1:
    run_cluster = st.button("▶️ Spustiť segmentáciu", type="primary")

with col2:
    auto_k_button = st.button("🔍 Automaticky nájsť optimálne k")

with col3:
    reset_button = st.button("🔄 Resetovať")

if reset_button:
    st.session_state.pop("df_clusters", None)
    st.session_state.pop("k_metrics", None)
    st.session_state.pop("best_k", None)

    st.success("Nastavenia boli resetované")
    st.rerun()

if auto_k_button:
    X_scaled = prepare_features(df_rfm, features, scaler_name, weights)

    k_metrics = compute_k_metrics(X_scaled, k_min, k_max)
    best_k, k_metrics = recommend_best_k(k_metrics)

    st.session_state["k_metrics"] = k_metrics
    st.session_state["best_k"] = best_k

k_metrics = st.session_state.get("k_metrics")

if k_metrics is not None:
    st.subheader("📊 Analýza optimálneho k")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.line(
                k_metrics,
                x="k",
                y="inertia",
                markers=True,
                labels={
                    "k": "Počet klastrov",
                    "inertia": "Inertia (variancia)"
                }
            ),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.line(
                k_metrics,
                x="k",
                y="silhouette",
                markers=True,
                labels={
                    "k": "Počet klastrov",
                    "silhouette": "Silhouette skóre"
                }
            ),
            use_container_width=True
        )

    st.markdown("### 🧾 Diagnostika výberu k")

    preview = k_metrics.copy()
    preview = preview.rename(columns={
        "k": "Počet klastrov",
        "inertia": "Inertia",
        "silhouette": "Silhouette skóre",
        "inertia_drop": "Pokles inertia",
        "inertia_drop_pct": "Pokles inertia (%)",
        "elbow_score": "Elbow skóre"
    })

    st.dataframe(preview, use_container_width=True)

    best_k = st.session_state.get("best_k")

    if best_k:
        st.success(f"Odporúčané k = {best_k}")
        st.info(
            "Odporúčanie je založené na kombinácii **Silhouette skóre** a **Elbow princípu**, "
            "nie iba na maximálnej hodnote silhouette. "
            "Tým sa znižuje riziko, že model automaticky preferuje príliš jednoduché riešenie s k = 2."
        )

if run_cluster:
    X_scaled = prepare_features(df_rfm, features, scaler_name, weights)

    if algorithm == "K-Means":
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init="auto"
        )

    elif algorithm == "DBSCAN":
        model = DBSCAN()

    else:
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
            metric="euclidean"
        )

    labels = model.fit_predict(X_scaled)

    df_clusters = df_rfm.copy()
    df_clusters["cluster"] = labels

    st.session_state["df_clusters"] = df_clusters

    st.success("Segmentácia bola dokončená")

df_clusters = st.session_state.get("df_clusters")

if df_clusters is None:
    st.stop()

st.subheader("Veľkosti klastrov")

counts = df_clusters["cluster"].value_counts().reset_index()
counts.columns = ["Klaster", "Počet zákazníkov"]

st.dataframe(counts)

st.plotly_chart(
    px.bar(
        counts,
        x="Klaster",
        y="Počet zákazníkov",
        title="Veľkosť jednotlivých klastrov"
    ),
    use_container_width=True
)

st.subheader("Profil klastrov")

profile = df_clusters.groupby("cluster").agg(
    customers=(STD_CUSTOMER, "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
).reset_index()

profile.columns = [
    "Klaster",
    "Počet zákazníkov",
    "Priemerná recencia",
    "Priemerná frekvencia",
    "Priemerná hodnota nákupov"
]

st.dataframe(profile)

st.subheader("Vizualizácia")

st.plotly_chart(
    px.scatter(
        df_clusters,
        x="recency",
        y="monetary",
        color="cluster",
        labels={
            "recency": "Recencia",
            "monetary": "Hodnota nákupov",
            "cluster": "Klaster"
        },
        title="Rozdelenie klastrov (2D)"
    ),
    use_container_width=True
)

st.plotly_chart(
    px.scatter_3d(
        df_clusters,
        x="recency",
        y="frequency",
        z="monetary",
        color="cluster",
        labels={
            "recency": "Recencia",
            "frequency": "Frekvencia",
            "monetary": "Hodnota nákupov",
            "cluster": "Klaster"
        },
        title="Rozdelenie klastrov (3D)"
    ),
    use_container_width=True
)

st.success("Segmentácia pripravená")