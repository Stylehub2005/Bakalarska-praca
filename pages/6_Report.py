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


def dataset_registry_path():
    return os.path.join(DATA_DIR, "registry.json")


def rfm_path(dataset_id):
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_rfm.parquet")


def clusters_path(dataset_id):
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_clusters.parquet")


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



st.title("📄 Report a export")


st.markdown("""
## 🎯 Účel stránky

Táto stránka slúži ako **finálny výstup analýzy**.

👉 Pomáha odpovedať na otázky:
- Aký veľký je dataset?
- Koľko máme zákazníkov?
- Aké sú celkové tržby?
- Je analýza kompletná?
- Čo môžeme exportovať pre marketing?

👉 Používa sa najmä pre:
- reporting (manažment)
- export dát do marketingových nástrojov
""")

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

# Load analyses
df_rfm = st.session_state.get("df_rfm")
if df_rfm is None:
    df_rfm = safe_read_parquet(rfm_path(dataset_id))

df_clusters = st.session_state.get("df_clusters")
if df_clusters is None:
    df_clusters = safe_read_parquet(clusters_path(dataset_id))


st.subheader("📊 Prehľad")

col1, col2, col3, col4 = st.columns(4)

rows = len(df_tx)
customers = df_tx[STD_CUSTOMER].nunique()
revenue = df_tx[STD_AMOUNT].sum()
avg_order = df_tx[STD_AMOUNT].mean()

col1.metric("Počet riadkov", f"{rows:,}")
col2.metric("Počet zákazníkov", f"{customers:,}")
col3.metric("Tržby", f"{revenue:.2f}")
col4.metric("Priemerná objednávka", f"{avg_order:.2f}")


st.markdown("### 📈 Interpretácia")

if customers > 0:
    st.info(f"Priemerný zákazník má približne {rows/customers:.1f} objednávok")

if revenue / customers > avg_order * 2:
    st.info("Existujú zákazníci s vysokou hodnotou (VIP potenciál)")


st.subheader("📁 Stav analýzy")

rfm_ok = df_rfm is not None and not df_rfm.empty
clusters_ok = df_clusters is not None and not df_clusters.empty

st.write(f"RFM: {'✅ dostupné' if rfm_ok else '❌ chýba'}")
st.write(f"Segmentácia: {'✅ dostupná' if clusters_ok else '❌ chýba'}")

if not rfm_ok:
    st.warning("⚠️ Najprv spusti RFM analýzu")

if rfm_ok and not clusters_ok:
    st.warning("⚠️ Spusti segmentáciu pre marketingové využitie")

if clusters_ok:
    st.success("✅ Analýza je kompletná – pripravená na marketing")


st.subheader("📊 Súhrn")

summary = pd.DataFrame({
    "Počet riadkov": [rows],
    "Počet zákazníkov": [customers],
    "Tržby": [revenue],
    "Priemerná objednávka": [avg_order],
    "RFM dostupné": [rfm_ok],
    "Segmentácia dostupná": [clusters_ok],
})

st.dataframe(summary)


st.subheader("📥 Export dát")

st.markdown("""
👉 Tieto dáta môžeš použiť:
- v CRM systémoch
- v e-mail marketingu
- v BI nástrojoch (Power BI, Tableau)
""")

tx_csv = df_tx.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Stiahnuť transakcie",
    tx_csv,
    f"{dataset_id}_transactions.csv"
)

if rfm_ok:
    rfm_csv = df_rfm.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Stiahnuť RFM",
        rfm_csv,
        f"{dataset_id}_rfm.csv"
    )

if clusters_ok:
    clusters_csv = df_clusters.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Stiahnuť klastre",
        clusters_csv,
        f"{dataset_id}_clusters.csv"
    )


st.subheader("📦 Export všetkého")

if st.button("Vytvoriť ZIP export"):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.writestr("transactions.csv", tx_csv)

        if rfm_ok:
            z.writestr("rfm.csv", df_rfm.to_csv(index=False))

        if clusters_ok:
            z.writestr("clusters.csv", df_clusters.to_csv(index=False))

    st.download_button(
        "⬇️ Stiahnuť ZIP",
        zip_buffer.getvalue(),
        f"{dataset_id}_export.zip",
        mime="application/zip"
    )


st.success("✅ Report je pripravený")

st.markdown("""
## 🚀 Čo ďalej?

👉 Použi tieto dáta na:
- marketingové kampane
- segmentáciu zákazníkov
- personalizáciu ponúk
""")