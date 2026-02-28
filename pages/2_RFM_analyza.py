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
    # Prefer session (so changes apply immediately), else load from file
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

    # stabilize qcut with ranks
    scored["_r_rank"] = scored["recency"].rank(method="first")
    scored["_f_rank"] = scored["frequency"].rank(method="first")
    scored["_m_rank"] = scored["monetary"].rank(method="first")

    # Recency reversed: smaller recency => higher score
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
    """
    Marketing labels driven by weighted scores.
    Thresholds are proportional to maximum possible weighted values.
    """
    wR = float(weights.get("R", 1.0))
    wF = float(weights.get("F", 1.0))
    wM = float(weights.get("M", 1.0))

    max_R = 5.0 * wR
    max_FM = 5.0 * wF + 5.0 * wM

    df = scored.copy()

    def label(row):
        r = row["R_weighted"]
        fm = row["FM_weighted"]

        # proportional thresholds
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
    st.warning("Nie je nastavený aktívny dataset. Vráť sa na stránku načítania dát a vyber dataset.")
    st.stop()

st.markdown(
    f"""
RFM analýza vypočíta:
- **Recency** – dni od posledného nákupu (nižšie = lepšie)
- **Frequency** – počet transakcií (vyššie = lepšie)
- **Monetary** – celková útrata (vyššie = lepšie)

✅ Nastavené váhy (zo stránky Nastavenia): **R={weights.get('R',1.0):.1f}, F={weights.get('F',1.0):.1f}, M={weights.get('M',1.0):.1f}**  
Výsledkom je aj **RFM_weighted_sum** a marketingové labely založené na vážených skóre.
"""
)

min_date = df[STD_DATE].min()
max_date = df[STD_DATE].max()
default_snapshot = (max_date + pd.Timedelta(days=1)).date()

snapshot_date_ui = st.date_input(
    "Snapshot date (reference date for Recency)",
    value=default_snapshot,
    min_value=min_date.date(),
    max_value=(max_date + pd.Timedelta(days=365)).date(),
)
snapshot_date = pd.to_datetime(snapshot_date_ui)

saved_exists = os.path.exists(rfm_path(dataset_id))
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 2])

with col1:
    run_calc = st.button("▶️ Spustiť výpočet RFM", type="primary")
with col2:
    load_saved = st.button("♻️ Načítať uložený RFM", disabled=not saved_exists)
with col3:
    delete_saved = st.button("🗑 Zmazať uložený RFM", disabled=not saved_exists)
with col4:
    st.caption(f"Aktívny dataset ID: `{dataset_id}` | uložený RFM: {'áno' if saved_exists else 'nie'}")

if delete_saved:
    delete_rfm_from_disk(dataset_id)
    st.session_state.pop("df_rfm", None)
    st.success("Uložený RFM bol zmazaný.")
    st.rerun()

if load_saved:
    loaded = load_rfm_from_disk(dataset_id)
    if loaded is None or loaded.empty:
        st.warning("Uložený RFM sa nepodarilo načítať.")
    else:
        st.session_state["df_rfm"] = loaded
        st.success("Uložený RFM bol načítaný (bez prepočtu).")

if run_calc:
    rfm = compute_rfm(df, snapshot_date)
    rfm_scored = rfm_scoring_quintiles(rfm)
    rfm_scored = add_weighted_scores(rfm_scored, weights)
    rfm_scored = describe_segments_weighted(rfm_scored, weights)

    st.session_state["df_rfm"] = rfm_scored
    save_rfm_to_disk(rfm_scored, dataset_id)
    st.success("RFM bol vypočítaný a uložený do histórie.")

# autoload once if no session
if "df_rfm" not in st.session_state:
    auto = load_rfm_from_disk(dataset_id)
    if auto is not None and not auto.empty:
        st.session_state["df_rfm"] = auto
        st.info("Našiel som uložený RFM a načítal som ho automaticky.")

df_rfm = st.session_state.get("df_rfm")
if df_rfm is None or df_rfm.empty:
    st.warning("Zatiaľ nie je k dispozícii RFM výsledok. Klikni na **Spustiť výpočet RFM**.")
    st.stop()

# ----- OUTPUTS -----
st.subheader("RFM table (top 50 by RFM_weighted_sum)")
sort_col = "RFM_weighted_sum" if "RFM_weighted_sum" in df_rfm.columns else "RFM_sum"
st.dataframe(df_rfm.sort_values(sort_col, ascending=False).head(50), use_container_width=True)

st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{df_rfm[STD_CUSTOMER].nunique():,}")
c2.metric("Avg Recency (days)", f"{df_rfm['recency'].mean():.1f}")
c3.metric("Avg Frequency", f"{df_rfm['frequency'].mean():.2f}")
c4.metric("Avg Monetary", f"{df_rfm['monetary'].mean():.2f}")

st.subheader("Distributions")
colA, colB = st.columns(2)
with colA:
    st.plotly_chart(px.histogram(df_rfm, x="recency", nbins=40, title="Recency distribution (days)"),
                    use_container_width=True)
with colB:
    st.plotly_chart(px.histogram(df_rfm, x="frequency", nbins=40, title="Frequency distribution"),
                    use_container_width=True)

st.plotly_chart(px.histogram(df_rfm, x="monetary", nbins=40, title="Monetary distribution"),
                use_container_width=True)

st.subheader("Segment overview")
seg_counts = (
    df_rfm["Segment_label"]
    .value_counts()
    .rename_axis("Segment_label")
    .reset_index(name="count")
)
st.plotly_chart(px.bar(seg_counts, x="Segment_label", y="count", title="Customers per segment (weighted labels)"),
                use_container_width=True)

st.subheader("Segment characteristics (average metrics)")
seg_profile = (
    df_rfm.groupby("Segment_label")
    .agg(
        customers=(STD_CUSTOMER, "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_R=("R_score", "mean"),
        avg_F=("F_score", "mean"),
        avg_M=("M_score", "mean"),
        avg_RFM_sum=("RFM_sum", "mean"),
        avg_RFM_weighted=("RFM_weighted_sum", "mean"),
    )
    .reset_index()
    .sort_values("customers", ascending=False)
)
st.dataframe(seg_profile, use_container_width=True)

st.info("✅ RFM je pripravené. Pokračuj na stránku **Segmentácia zákazníkov**.")