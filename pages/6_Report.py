import os
import json
import streamlit as st
import pandas as pd

STD_CUSTOMER = "customer_id"
STD_DATE = "transaction_date"
STD_AMOUNT = "amount"

DATA_DIR = "data"
ANALYSES_DIR = os.path.join(DATA_DIR, "analyses")


def dataset_registry_path():
    return os.path.join(DATA_DIR, "registry.json")


def load_registry():
    if not os.path.exists(dataset_registry_path()):
        return {"datasets": [], "active_dataset_id": None}
    with open(dataset_registry_path(), "r", encoding="utf-8") as f:
        return json.load(f)


def rfm_path(dataset_id: str) -> str:
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_rfm.parquet")


def clusters_path(dataset_id: str) -> str:
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_clusters.parquet")


def clusters_meta_path(dataset_id: str) -> str:
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_clusters_meta.json")


def safe_read_parquet(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def build_summary(df_tx: pd.DataFrame, df_rfm: pd.DataFrame | None, df_clusters: pd.DataFrame | None) -> pd.DataFrame:
    rows = len(df_tx)
    customers = df_tx[STD_CUSTOMER].nunique()
    min_date = df_tx[STD_DATE].min()
    max_date = df_tx[STD_DATE].max()
    revenue = df_tx[STD_AMOUNT].sum()
    avg_order = df_tx[STD_AMOUNT].mean()

    out = {
        "rows": [rows],
        "customers": [customers],
        "date_min": [str(min_date)],
        "date_max": [str(max_date)],
        "total_revenue": [float(revenue)],
        "avg_order_value": [float(avg_order)],
        "has_rfm": [df_rfm is not None and not df_rfm.empty],
        "has_clusters": [df_clusters is not None and not df_clusters.empty],
    }

    # If clustering exists, add number of segments
    if df_clusters is not None and "cluster_label" in df_clusters.columns:
        out["segments_count"] = [int(df_clusters["cluster_label"].nunique())]
    elif df_clusters is not None and "cluster" in df_clusters.columns:
        out["segments_count"] = [int(df_clusters["cluster"].nunique())]
    else:
        out["segments_count"] = [0]

    return pd.DataFrame(out)


st.title("📄 Report & Export")

dataset_id = st.session_state.get("active_dataset_id")
df_tx = st.session_state.get("df_transactions")

if not dataset_id or df_tx is None or df_tx.empty:
    st.warning("Najprv načítaj dáta na stránke **Načítanie a overenie dát**.")
    st.stop()

# Ensure types
df_tx = df_tx.copy()
df_tx[STD_DATE] = pd.to_datetime(df_tx[STD_DATE], errors="coerce")
df_tx[STD_AMOUNT] = pd.to_numeric(df_tx[STD_AMOUNT], errors="coerce")
df_tx = df_tx.dropna(subset=[STD_DATE, STD_AMOUNT, STD_CUSTOMER])

# Load analyses from disk if available
df_rfm = safe_read_parquet(rfm_path(dataset_id))
df_clusters = safe_read_parquet(clusters_path(dataset_id))

registry = load_registry()

st.markdown(
    """
Táto stránka umožňuje export výsledkov analýzy pre marketingové použitie.
Môžeš stiahnuť:
- transakčné dáta (štandardizované),
- RFM tabuľku,
- segmentáciu (klastre),
- súhrnný report (summary).
"""
)

st.subheader("Aktívny dataset")
st.code(f"dataset_id = {dataset_id}")

# Show registry info for this dataset if exists
ds_meta = None
for d in registry.get("datasets", []):
    if d.get("id") == dataset_id:
        ds_meta = d
        break

if ds_meta:
    st.write("Original name:", ds_meta.get("original_name"))
    st.write("Created at:", ds_meta.get("created_at"))
    st.write("Rows:", ds_meta.get("rows"))
    st.write("Customers:", ds_meta.get("customers"))

st.divider()

st.subheader("Dostupné uložené analýzy (história)")
files_info = [
    ("RFM", rfm_path(dataset_id)),
    ("Clusters", clusters_path(dataset_id)),
    ("Clusters meta", clusters_meta_path(dataset_id)),
]
for label, path in files_info:
    st.write(f"- {label}: {'✅' if os.path.exists(path) else '❌'}  ({path})")

st.divider()

# Summary report
st.subheader("Summary report")
summary_df = build_summary(df_tx, df_rfm, df_clusters)
st.dataframe(summary_df, use_container_width=True)

summary_csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download summary report (CSV)",
    data=summary_csv,
    file_name=f"{dataset_id}_summary.csv",
    mime="text/csv"
)

st.divider()

# Exports
st.subheader("Export súborov")

tx_csv = df_tx.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download transactions (CSV)",
    data=tx_csv,
    file_name=f"{dataset_id}_transactions.csv",
    mime="text/csv"
)

if df_rfm is not None and not df_rfm.empty:
    rfm_csv = df_rfm.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download RFM (CSV)",
        data=rfm_csv,
        file_name=f"{dataset_id}_rfm.csv",
        mime="text/csv"
    )
else:
    st.info("RFM zatiaľ nie je dostupné. Spusti RFM analýzu, aby sa dalo exportovať.")

if df_clusters is not None and not df_clusters.empty:
    clusters_csv = df_clusters.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download clusters (CSV)",
        data=clusters_csv,
        file_name=f"{dataset_id}_clusters.csv",
        mime="text/csv"
    )
else:
    st.info("Segmentácia zatiaľ nie je dostupná. Spusti Segmentáciu, aby sa dala exportovať.")

st.success("✅ Export pripravený. Posledná stránka: **Nastavenia segmentácie**.")