import streamlit as st
import pandas as pd

STD_CUSTOMER = "customer_id"
STD_AMOUNT = "amount"

st.set_page_config(
    page_title="Segmify",
    layout="wide"
)

# ---------------- TITLE ----------------

st.title("📊 Segmify - systém segmentácie zákazníkov")

st.markdown("""
### 🎓 Bakalárska práca – projekt

Táto aplikácia umožňuje:

- Analýzu správania zákazníkov
- RFM segmentáciu
- Zhlukovanie zákazníkov
- Monitorovanie trendov
- Marketingové poznatky

""")

st.divider()

# ---------------- LOAD DATA ----------------

df_tx = st.session_state.get("df_transactions")
df_rfm = st.session_state.get("df_rfm")
df_clusters = st.session_state.get("df_clusters")

# ---------------- KPI ----------------

st.subheader("📊 Prehľad")

col1, col2, col3, col4 = st.columns(4)

if df_tx is not None and not df_tx.empty:
    col1.metric("Customers", f"{df_tx[STD_CUSTOMER].nunique():,}")
    col2.metric("Transactions", f"{len(df_tx):,}")
    col3.metric("Revenue", f"{df_tx[STD_AMOUNT].sum():.2f}")
    col4.metric("Avg order", f"{df_tx[STD_AMOUNT].mean():.2f}")
else:
    st.info("No dataset loaded yet. Please upload data first.")

st.divider()


st.subheader("⚙️ Analytický proces")

st.markdown("""
1️⃣ Načítanie dát  
2️⃣ RFM analýza  
3️⃣ Segmentácia zákazníkov  
4️⃣ Trendy a monitoring  
5️⃣ Marketing insights  
6️⃣ Report & export  
""")

st.divider()


st.subheader("📁 Stav systému")

col1, col2, col3 = st.columns(3)

col1.metric("Dáta načítané", "✅" if df_tx is not None else "❌")
col2.metric("RFM pripravené", "✅" if df_rfm is not None else "❌")
col3.metric("Segmentácia pripravená", "✅" if df_clusters is not None else "❌")

st.divider()

# ---------------- NAVIGATION ----------------

st.subheader("🚀 Rýchla navigácia")

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("pages/1_Nacitanie_dat.py", label="📂 Načítanie dát")

with col2:
    st.page_link("pages/2_RFM_analyza.py", label="📊 RFM analýza")

with col3:
    st.page_link("pages/3_Segmentacia.py", label="🧠 Segmentácia")

col4, col5, col6 = st.columns(3)

with col4:
    st.page_link("pages/4_Trendy.py", label="📈 Trendy")

with col5:
    st.page_link("pages/5_Marketing_Insights.py", label="🎯 Marketing Insights")

with col6:
    st.page_link("pages/6_Report.py", label="📄 Report")

st.page_link("pages/7_Nastavenia.py", label="⚙️ Nastavenia")

st.divider()


st.markdown("""
---
💡 Tento systém pomáha identifikovať hodnotné segmenty zákazníkov a zlepšiť marketingové stratégie.

👨‍🎓 Projekt bakalárskej práce – systém segmentácie zákazníkov
""")
#https://bakalarska-praca-k6jwcdmpgsjuck4qyftlee.streamlit.app/