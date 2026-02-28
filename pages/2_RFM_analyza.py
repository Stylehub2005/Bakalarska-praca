import os
import pandas as pd
import streamlit as st
import plotly.express as px

STD_CUSTOMER = "customer_id"
STD_DATE = "transaction_date"
STD_AMOUNT = "amount"

ANALYSES_DIR = "data/analyses"


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


def describe_segments(scored: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    df["FM_sum"] = df["F_score"] + df["M_score"]

    def label(row):
        r = row["R_score"]
        fm = row["FM_sum"]
        if r >= 4 and fm >= 8:
            return "VIP / Champions"
        if r >= 4 and fm >= 6:
            return "Loyal / Active"
        if r >= 3 and fm >= 6:
            return "Potential Loyalists"
        if r <= 2 and fm >= 7:
            return "At Risk (High value)"
        if r <= 2 and fm <= 4:
            return "Lost"
        if r >= 4 and fm <= 4:
            return "New / Low spend"
        return "Regular"

    df["Segment_label"] = df.apply(label, axis=1)
    return df.drop(columns=["FM_sum"])


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

df = st.session_state.get("df_transactions")
dataset_id = st.session_state.get("active_dataset_id")

if df is None or df.empty:
    st.warning("Najprv načítaj dáta na stránke **Načítanie a overenie dát**.")
    st.stop()

if not dataset_id:
    st.warning("Nie je nastavený aktívny dataset. Vráť sa na stránku načítania dát a vyber dataset.")
    st.stop()

st.markdown(
    """
RFM analýza vypočíta:
- **Recency** – dni od posledného nákupu (nižšie = lepšie)
- **Frequency** – počet transakcií (vyššie = lepšie)
- **Monetary** – celková útrata (vyššie = lepšie)

Táto stránka podporuje **históriu analýz**: výsledok sa ukladá do `data/analyses/` a dá sa načítať bez prepočtu.
"""
)

# Snapshot date selection
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

# Buttons row
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

# Delete saved
if delete_saved:
    delete_rfm_from_disk(dataset_id)
    st.session_state.pop("df_rfm", None)
    st.success("Uložený RFM bol zmazaný.")
    st.rerun()

# Load saved
if load_saved:
    loaded = load_rfm_from_disk(dataset_id)
    if loaded is None or loaded.empty:
        st.warning("Uložený RFM sa nepodarilo načítať (súbor chýba alebo je prázdny).")
    else:
        st.session_state["df_rfm"] = loaded
        st.success("Uložený RFM bol načítaný (bez prepočtu).")

# Run calculation
if run_calc:
    rfm = compute_rfm(df, snapshot_date)
    rfm_scored = rfm_scoring_quintiles(rfm)
    rfm_scored = describe_segments(rfm_scored)

    st.session_state["df_rfm"] = rfm_scored
    save_rfm_to_disk(rfm_scored, dataset_id)

    st.success("RFM bol vypočítaný a uložený do histórie.")
    # не rerun — чтобы сразу показать результаты ниже

# If nothing in session yet, try autoload once
if "df_rfm" not in st.session_state:
    auto = load_rfm_from_disk(dataset_id)
    if auto is not None and not auto.empty:
        st.session_state["df_rfm"] = auto
        st.info("Našiel som uložený RFM a načítal som ho automaticky. (Môžeš prepočítať tlačidlom vyššie.)")

df_rfm = st.session_state.get("df_rfm")
if df_rfm is None or df_rfm.empty:
    st.warning("Zatiaľ nie je k dispozícii RFM výsledok. Klikni na **Spustiť výpočet RFM**.")
    st.stop()

# ----- OUTPUTS -----
st.subheader("RFM table (top 50 by RFM_sum)")
st.dataframe(df_rfm.sort_values("RFM_sum", ascending=False).head(50), use_container_width=True)

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
st.plotly_chart(px.bar(seg_counts, x="Segment_label", y="count", title="Customers per segment (rule-based labels)"),
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
    )
    .reset_index()
    .sort_values("customers", ascending=False)
)
st.dataframe(seg_profile, use_container_width=True)

st.info("✅ RFM je pripravené. Pokračuj na stránku **Segmentácia zákazníkov**.")