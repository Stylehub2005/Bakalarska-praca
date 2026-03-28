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


# ================= UI =================

st.title("📊 RFM analýza")


st.markdown("""
## 📊 Čo je RFM analýza?

RFM = **Recency, Frequency, Monetary**

- **Recency (R)** – koľko dní od posledného nákupu  
- **Frequency (F)** – počet nákupov  
- **Monetary (M)** – celková hodnota nákupov  

👉 Používa sa na segmentáciu zákazníkov podľa ich hodnoty.
""")

settings = load_settings()
weights = settings.get("rfm_weights", {"R": 1.0, "F": 1.0, "M": 1.0})

df = st.session_state.get("df_transactions")
dataset_id = st.session_state.get("active_dataset_id")

if df is None or df.empty:
    st.warning("Najprv načítaj dáta.")
    st.stop()

if not dataset_id:
    st.warning("Nie je aktívny dataset.")
    st.stop()

min_date = df[STD_DATE].min()
max_date = df[STD_DATE].max()

snapshot_date = pd.to_datetime(
    st.date_input("Snapshot date", value=(max_date + pd.Timedelta(days=1)).date())
)

if st.button("▶️ Spustiť výpočet RFM"):

    rfm = compute_rfm(df, snapshot_date)
    rfm_scored = rfm_scoring_quintiles(rfm)
    rfm_scored = add_weighted_scores(rfm_scored, weights)
    rfm_scored = describe_segments_weighted(rfm_scored, weights)

    st.session_state["df_rfm"] = rfm_scored
    save_rfm_to_disk(rfm_scored, dataset_id)

df_rfm = st.session_state.get("df_rfm")

if df_rfm is None:
    st.stop()


st.markdown("""
### 🧾 Vysvetlenie stĺpcov

- recency → nižšie = lepšie  
- frequency → vyššie = lepšie  
- monetary → vyššie = lepšie  
- Segment_label → typ zákazníka
""")

st.dataframe(df_rfm.head(50))

# ================= DISTRIBUTIONS =================

st.subheader("Distributions")

st.plotly_chart(px.histogram(df_rfm, x="recency"))
st.plotly_chart(px.histogram(df_rfm, x="frequency"))

st.markdown("### 📈 Interpretácia dát")

if df_rfm["frequency"].skew() > 1:
    st.warning("Väčšina zákazníkov nakupuje málo → typické pre e-commerce")

# ================= FIXED FREQUENCY =================

st.subheader("Frequency categories")

bins = [0,1,2,3,5,10,50]
labels = ["1","2","3","4-5","6-10","10+"]

df_rfm["frequency_group"] = pd.cut(df_rfm["frequency"], bins=bins, labels=labels)

freq_counts = (
    df_rfm["frequency_group"]
    .value_counts()
    .sort_index()
    .reset_index()
)

freq_counts.columns = ["Frequency group", "Customers"]

st.plotly_chart(
    px.bar(
        freq_counts,
        x="Frequency group",
        y="Customers",
        title="Customer distribution by purchase frequency"
    ),
    use_container_width=True
)

# ================= OUTLIERS =================

st.subheader("Monetary outliers")

q99 = df_rfm["monetary"].quantile(0.99)

st.markdown(f"""
Top 1% zákazníkov generuje extrémny revenue.
""")

st.plotly_chart(px.box(df_rfm, y="monetary"))

# ================= SEGMENTS =================

st.subheader("Segment overview")

seg_counts = (
    df_rfm["Segment_label"]
    .value_counts()
    .reset_index()
)

seg_counts.columns = ["Segment", "Customers"]

st.plotly_chart(
    px.bar(
        seg_counts,
        x="Segment",
        y="Customers"
    ),
    use_container_width=True
)