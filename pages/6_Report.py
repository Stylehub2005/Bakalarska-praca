import os
import json
import zipfile
import io
import streamlit as st
import pandas as pd

STD_CUSTOMER = "customer_id"
STD_DATE = "transaction_date"
STD_AMOUNT = "amount"

DATA_DIR = "data"
ANALYSES_DIR = os.path.join(DATA_DIR, "analyses")


# ---------------- PATHS ----------------

def dataset_registry_path():
    return os.path.join(DATA_DIR, "registry.json")


def rfm_path(dataset_id):
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_rfm.parquet")


def clusters_path(dataset_id):
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_clusters.parquet")


# ---------------- LOAD ----------------

def load_registry():
    if not os.path.exists(dataset_registry_path()):
        return {"datasets": [], "active_dataset_id": None}
    with open(dataset_registry_path(), "r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_parquet(path):
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except:
            return None
    return None


# ---------------- UI ----------------

st.title("📄 Report & Export")

dataset_id = st.session_state.get("active_dataset_id")
df_tx = st.session_state.get("df_transactions")

if not dataset_id or df_tx is None or df_tx.empty:
    st.warning("Najprv načítaj dáta.")
    st.stop()

# Clean
df_tx = df_tx.copy()
df_tx[STD_DATE] = pd.to_datetime(df_tx[STD_DATE], errors="coerce")
df_tx[STD_AMOUNT] = pd.to_numeric(df_tx[STD_AMOUNT], errors="coerce")
df_tx = df_tx.dropna(subset=[STD_DATE, STD_AMOUNT, STD_CUSTOMER])

# Load data (ВАЖНО: session + disk)
df_rfm = st.session_state.get("df_rfm")
if df_rfm is None:
    df_rfm = safe_read_parquet(rfm_path(dataset_id))

df_clusters = st.session_state.get("df_clusters")
if df_clusters is None:
    df_clusters = safe_read_parquet(clusters_path(dataset_id))

registry = load_registry()

# ---------------- KPI ----------------

st.subheader("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", f"{len(df_tx):,}")
col2.metric("Customers", f"{df_tx[STD_CUSTOMER].nunique():,}")
col3.metric("Revenue", f"{df_tx[STD_AMOUNT].sum():.2f}")
col4.metric("Avg order", f"{df_tx[STD_AMOUNT].mean():.2f}")

# ---------------- STATUS ----------------

st.subheader("📁 Analysis status")

rfm_ok = df_rfm is not None and not df_rfm.empty
clusters_ok = df_clusters is not None and not df_clusters.empty

st.write(f"RFM: {'✅' if rfm_ok else '❌'}")
st.write(f"Segmentation: {'✅' if clusters_ok else '❌'}")

if clusters_ok:
    st.success("Segmentácia je dostupná")
else:
    st.warning("Segmentácia nie je dostupná")

# ---------------- SUMMARY ----------------

st.subheader("Summary")

summary = pd.DataFrame({
    "rows": [len(df_tx)],
    "customers": [df_tx[STD_CUSTOMER].nunique()],
    "revenue": [df_tx[STD_AMOUNT].sum()],
    "avg_order": [df_tx[STD_AMOUNT].mean()],
    "has_rfm": [rfm_ok],
    "has_clusters": [clusters_ok],
})

st.dataframe(summary)

# ---------------- EXPORT ----------------

st.subheader("📥 Export")

tx_csv = df_tx.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download transactions",
    tx_csv,
    f"{dataset_id}_transactions.csv"
)

if rfm_ok:
    rfm_csv = df_rfm.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download RFM",
        rfm_csv,
        f"{dataset_id}_rfm.csv"
    )

if clusters_ok:
    clusters_csv = df_clusters.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download clusters",
        clusters_csv,
        f"{dataset_id}_clusters.csv"
    )

# ---------------- ZIP EXPORT ----------------

st.subheader("📦 Export all (ZIP)")

if st.button("Create ZIP export"):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as z:

        z.writestr("transactions.csv", tx_csv)

        if rfm_ok:
            z.writestr("rfm.csv", df_rfm.to_csv(index=False))

        if clusters_ok:
            z.writestr("clusters.csv", df_clusters.to_csv(index=False))

    st.download_button(
        "⬇️ Download ZIP",
        zip_buffer.getvalue(),
        f"{dataset_id}_export.zip",
        mime="application/zip"
    )

st.success("✅ Report ready")