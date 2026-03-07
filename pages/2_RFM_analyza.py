import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

STD_CUSTOMER = "customer_id"
STD_DATE = "transaction_date"
STD_AMOUNT = "amount"

ANALYSES_DIR = "data/analyses"
SETTINGS_PATH = "data/settings.json"

DEFAULT_SETTINGS = {
    "rfm_weights": {"R": 1.0, "F": 1.0, "M": 1.0},
    "default_scaler": "StandardScaler",
    "auto_k": {"k_min": 2, "k_max": 10},
    "segmentation_default_algorithm": "K-Means",
}


def load_settings() -> dict:
    s = st.session_state.get("settings")
    if isinstance(s, dict):
        merged = DEFAULT_SETTINGS.copy()
        merged.update(s)
        merged["rfm_weights"] = {**DEFAULT_SETTINGS["rfm_weights"], **merged.get("rfm_weights", {})}
        merged["auto_k"] = {**DEFAULT_SETTINGS["auto_k"], **merged.get("auto_k", {})}
        return merged

    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                s2 = json.load(f)
            merged = DEFAULT_SETTINGS.copy()
            merged.update(s2)
            merged["rfm_weights"] = {**DEFAULT_SETTINGS["rfm_weights"], **merged.get("rfm_weights", {})}
            merged["auto_k"] = {**DEFAULT_SETTINGS["auto_k"], **merged.get("auto_k", {})}
            st.session_state["settings"] = merged
            return merged
        except Exception:
            pass

    st.session_state["settings"] = DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()


def rfm_path(dataset_id: str) -> str:
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_rfm.parquet")


def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    rfm = (
        df.groupby(STD_CUSTOMER)
        .agg(
            recency=(STD_DATE, lambda x: (snapshot_date - x.max()).days),
            frequency=(STD_DATE, "count"),
            monetary=(STD_AMOUNT, "sum"),
        )
        .reset_index()
    )
    return rfm


def rfm_scoring_quintiles(rfm: pd.DataFrame) -> pd.DataFrame:
    scored = rfm.copy()

    scored["_r_rank"] = scored["recency"].rank(method="first")
    scored["_f_rank"] = scored["frequency"].rank(method="first")
    scored["_m_rank"] = scored["monetary"].rank(method="first")

    scored["R_score"] = pd.qcut(scored["_r_rank"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    scored["F_score"] = pd.qcut(scored["_f_rank"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    scored["M_score"] = pd.qcut(scored["_m_rank"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    scored["RFM_score"] = (
        scored["R_score"].astype(str) + scored["F_score"].astype(str) + scored["M_score"].astype(str)
    )

    scored["RFM_sum"] = scored["R_score"] + scored["F_score"] + scored["M_score"]

    scored = scored.drop(columns=["_r_rank", "_f_rank", "_m_rank"])

    return scored


def add_weighted_scores(scored: pd.DataFrame, weights: dict) -> pd.DataFrame:
    wR = float(weights.get("R", 1.0))
    wF = float(weights.get("F", 1.0))
    wM = float(weights.get("M", 1.0))

    df = scored.copy()

    df["R_weighted"] = df["R_score"] * wR
    df["F_weighted"] = df["F_score"] * wF
    df["M_weighted"] = df["M_score"] * wM

    df["FM_weighted"] = df["F_weighted"] + df["M_weighted"]

    df["RFM_weighted_sum"] = df["R_weighted"] + df["F_weighted"] + df["M_weighted"]

    return df


def describe_segments_weighted(scored: pd.DataFrame, weights: dict) -> pd.DataFrame:

    wR = float(weights.get("R", 1.0))
    wF = float(weights.get("F", 1.0))
    wM = float(weights.get("M", 1.0))

    max_R = 5.0 * wR
    max_FM = 5.0 * wF + 5.0 * wM

    df = scored.copy()

    def label(row):

        r = row["R_weighted"]
        fm = row["FM_weighted"]

        if r >= 0.8 * max_R and fm >= 0.8 * max_FM:
            return "VIP / Champions"

        if r >= 0.8 * max_R and fm >= 0.6 * max_FM:
            return "Loyal / Active"

        if r >= 0.6 * max_R and fm >= 0.6 * max_FM:
            return "Potential Loyalists"

        if r <= 0.4 * max_R and fm >= 0.7 * max_FM:
            return "At Risk (High value)"

        if r <= 0.4 * max_R and fm <= 0.4 * max_FM:
            return "Lost"

        if r >= 0.8 * max_R and fm <= 0.4 * max_FM:
            return "New / Low spend"

        return "Regular"

    df["Segment_label"] = df.apply(label, axis=1)

    return df


def save_rfm_to_disk(df_rfm: pd.DataFrame, dataset_id: str) -> None:
    os.makedirs(ANALYSES_DIR, exist_ok=True)
    df_rfm.to_parquet(rfm_path(dataset_id), index=False)


def load_rfm_from_disk(dataset_id: str) -> pd.DataFrame | None:
    path = rfm_path(dataset_id)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def delete_rfm_from_disk(dataset_id: str) -> None:
    path = rfm_path(dataset_id)
    if os.path.exists(path):
        os.remove(path)


st.title("📊 RFM analýza")

settings = load_settings()
weights = settings.get("rfm_weights", {"R": 1.0, "F": 1.0, "M": 1.0})

df = st.session_state.get("df_transactions")
dataset_id = st.session_state.get("active_dataset_id")

if df is None or df.empty:
    st.warning("Najprv načítaj dáta na stránke **Načítanie a overenie dát**.")
    st.stop()

if not dataset_id:
    st.warning("Nie je nastavený aktívny dataset.")
    st.stop()

min_date = df[STD_DATE].min()
max_date = df[STD_DATE].max()

default_snapshot = (max_date + pd.Timedelta(days=1)).date()

snapshot_date_ui = st.date_input(
    "Snapshot date",
    value=default_snapshot,
    min_value=min_date.date(),
    max_value=(max_date + pd.Timedelta(days=365)).date(),
)

snapshot_date = pd.to_datetime(snapshot_date_ui)

saved_exists = os.path.exists(rfm_path(dataset_id))

col1, col2, col3 = st.columns(3)

with col1:
    run_calc = st.button("▶️ Spustiť výpočet RFM", type="primary")

with col2:
    load_saved = st.button("♻️ Načítať uložený RFM", disabled=not saved_exists)

with col3:
    delete_saved = st.button("🗑 Zmazať uložený RFM", disabled=not saved_exists)

if delete_saved:
    delete_rfm_from_disk(dataset_id)
    st.session_state.pop("df_rfm", None)
    st.success("RFM deleted.")
    st.rerun()

if load_saved:
    loaded = load_rfm_from_disk(dataset_id)
    if loaded is not None:
        st.session_state["df_rfm"] = loaded

if run_calc:

    rfm = compute_rfm(df, snapshot_date)

    rfm_scored = rfm_scoring_quintiles(rfm)

    rfm_scored = add_weighted_scores(rfm_scored, weights)

    rfm_scored = describe_segments_weighted(rfm_scored, weights)

    st.session_state["df_rfm"] = rfm_scored

    save_rfm_to_disk(rfm_scored, dataset_id)

    st.success("RFM calculated and saved.")

df_rfm = st.session_state.get("df_rfm")

if df_rfm is None:
    st.warning("Run RFM calculation first.")
    st.stop()

st.subheader("RFM table")

st.dataframe(
    df_rfm.sort_values("RFM_weighted_sum", ascending=False).head(50),
    use_container_width=True,
)

st.subheader("Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Customers", df_rfm[STD_CUSTOMER].nunique())
c2.metric("Avg Recency", round(df_rfm["recency"].mean(), 1))
c3.metric("Avg Frequency", round(df_rfm["frequency"].mean(), 2))
c4.metric("Avg Monetary", round(df_rfm["monetary"].mean(), 2))

st.subheader("Distributions")

colA, colB = st.columns(2)

with colA:
    st.plotly_chart(px.histogram(df_rfm, x="recency", nbins=40))

with colB:
    st.plotly_chart(px.histogram(df_rfm, x="frequency", nbins=40))

st.plotly_chart(px.histogram(df_rfm, x="monetary", nbins=40))

# -------- Frequency category bar chart --------

st.subheader("Frequency categories")

bins = [0, 1, 2, 5, 10, 20, df_rfm["frequency"].max() + 1]

labels = [
    "1 purchase",
    "2 purchases",
    "3–5 purchases",
    "6–10 purchases",
    "11–20 purchases",
    "20+ purchases",
]

df_rfm["frequency_group"] = pd.cut(
    df_rfm["frequency"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)

freq_counts = (
    df_rfm["frequency_group"]
    .value_counts()
    .sort_index()
    .reset_index()
)

freq_counts.columns = ["Frequency group", "Customers"]

st.plotly_chart(
    px.bar(freq_counts, x="Frequency group", y="Customers")
)

# -------- Monetary boxplot --------

st.subheader("Monetary outliers")

st.plotly_chart(
    px.box(
        df_rfm,
        y="monetary",
        points="outliers",
        title="Monetary distribution with outliers",
    )
)

# -------- Segment overview --------

st.subheader("Segment overview")

seg_counts = (
    df_rfm["Segment_label"]
    .value_counts()
    .reset_index()
)

seg_counts.columns = ["Segment", "Customers"]

st.plotly_chart(
    px.bar(seg_counts, x="Segment", y="Customers")
)

st.info("✅ RFM ready. Continue to customer segmentation.")