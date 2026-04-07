import os
import json
import hashlib
from datetime import datetime
import streamlit as st
import pandas as pd

STD_CUSTOMER = "customer_id"
STD_DATE = "transaction_date"
STD_AMOUNT = "amount"

DATA_DIR = "data"
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
ANALYSES_DIR = os.path.join(DATA_DIR, "analyses")
REGISTRY_PATH = os.path.join(DATA_DIR, "registry.json")



def ensure_storage():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(ANALYSES_DIR, exist_ok=True)


def load_registry():
    ensure_storage()
    if not os.path.exists(REGISTRY_PATH):
        return {"datasets": [], "active_dataset_id": None}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def dataset_path(dataset_id):
    return os.path.join(DATASETS_DIR, f"{dataset_id}.csv")


def delete_dataset(dataset_id):
    p = dataset_path(dataset_id)
    if os.path.exists(p):
        os.remove(p)



def load_csv_safely(file):
    for sep in [",", ";", "\t", "|"]:
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=sep)
            if df.shape[1] >= 2:
                return df
        except:
            pass
    file.seek(0)
    return pd.read_csv(file, sep=None, engine="python")



def to_standard(df, c, d, mode, a=None, q=None, p=None):
    df = df.copy()

    df = df.rename(columns={c: STD_CUSTOMER, d: STD_DATE})

    if mode == "Amount":
        df = df.rename(columns={a: STD_AMOUNT})
    else:
        df[STD_AMOUNT] = pd.to_numeric(df[q], errors="coerce") * pd.to_numeric(df[p], errors="coerce")

    return df[[STD_CUSTOMER, STD_DATE, STD_AMOUNT]]


def clean(df):
    df = df.copy()
    df[STD_DATE] = pd.to_datetime(df[STD_DATE], errors="coerce")
    df[STD_AMOUNT] = pd.to_numeric(df[STD_AMOUNT], errors="coerce")
    df = df.dropna()
    df = df[df[STD_AMOUNT] > 0]
    return df


st.title("📂 Načítanie dát")

st.markdown("""
### 📌 Ako pripraviť dataset

Táto aplikácia pracuje s **transakčnými dátami o nákupoch zákazníkov**.

Dataset musí obsahovať:

- 🧑 **Customer ID** – identifikátor zákazníka  
- 📅 **Dátum transakcie**  
- 💰 **Hodnota transakcie**:
  - buď ako **amount**
  - alebo ako **Quantity × Price**

💡 **Poznámka:**  
Pre **RFM analýzu** nie je potrebné evidovať konkrétny produkt alebo typ nákupu.  
Model analyzuje najmä **nákupné správanie zákazníka**, teda:
- ako **nedávno** zákazník nakúpil,
- ako **často** nakupuje,
- a akú má **celkovú hodnotu nákupov**.

---

### ✅ Príklad:

Customer | Date | Amount  
123 | 2024-01-01 | 100  

alebo:

Customer | Date | Quantity | Price  
123 | 2024-01-01 | 2 | 50  

---

### ⚠️ Dôležité:
- každý stĺpec môže byť použitý iba raz  
- dátum musí byť validný  
- amount musí byť číslo > 0  
""")

registry = load_registry()


st.subheader("🗃 História datasetov")

datasets = registry["datasets"]

if datasets:

    labels = []
    id_map = {}

    for d in datasets:
        label = f"{d['created_at']} | rows={d['rows']} customers={d['customers']}"
        labels.append(label)
        id_map[label] = d["id"]

    selected = st.selectbox("Vyber dataset", labels)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Nastaviť ako aktívny"):
            dataset_id = id_map[selected]

            df = pd.read_csv(dataset_path(dataset_id))
            df[STD_DATE] = pd.to_datetime(df[STD_DATE])
            df[STD_AMOUNT] = pd.to_numeric(df[STD_AMOUNT])

            st.session_state["df_transactions"] = df
            st.session_state["active_dataset_id"] = dataset_id

            registry["active_dataset_id"] = dataset_id
            save_registry(registry)

            st.success("Dataset nastavený ako aktívny")
            st.rerun()

    with col2:
        if st.button("🗑 Zmazať dataset"):
            dataset_id = id_map[selected]

            registry["datasets"] = [d for d in datasets if d["id"] != dataset_id]

            delete_dataset(dataset_id)
            save_registry(registry)

            st.warning("Dataset zmazaný")
            st.rerun()

else:
    st.info("Žiadne datasety")


st.divider()


st.subheader("⬆️ Nahrať CSV")

file = st.file_uploader("Upload CSV")

if file:

    df_raw = load_csv_safely(file)

    st.dataframe(df_raw.head())

    cols = list(df_raw.columns)

    st.subheader("🧩 Mapovanie")

    st.caption("Vyber stĺpec identifikujúci zákazníka")
    customer = st.selectbox("Customer", cols)

    st.caption("Vyber dátum transakcie")
    date = st.selectbox("Date", cols)

    mode = st.radio("Mode", ["Amount", "Quantity×Price"])

    if mode == "Amount":
        st.caption("Vyber celkovú hodnotu objednávky")
        amount = st.selectbox("Amount", cols)
        qty = price = None
    else:
        st.caption("Vyber množstvo produktov")
        qty = st.selectbox("Quantity", cols)

        st.caption("Vyber cenu za jednotku")
        price = st.selectbox("Price", cols)
        amount = None

    st.warning("⚠️ Každý stĺpec musí byť unikátny")


    selected_cols = [customer, date]

    if mode == "Amount":
        selected_cols.append(amount)
    else:
        selected_cols.extend([qty, price])

    if len(set(selected_cols)) != len(selected_cols):
        st.error("❌ Duplicitné stĺpce!")
        st.stop()

    if st.button("💾 Uložiť dataset"):

        try:
            df = to_standard(df_raw, customer, date, mode, amount, qty, price)
            df = clean(df)

            csv = df.to_csv(index=False).encode()
            dataset_id = hashlib.sha256(csv).hexdigest()[:16]

            with open(dataset_path(dataset_id), "wb") as f:
                f.write(csv)

            registry = load_registry()

            registry["datasets"].insert(0, {
                "id": dataset_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "rows": len(df),
                "customers": df[STD_CUSTOMER].nunique()
            })

            registry["active_dataset_id"] = dataset_id
            save_registry(registry)

            st.session_state["df_transactions"] = df
            st.session_state["active_dataset_id"] = dataset_id

            st.success("✅ Dataset uložený")

        except Exception as e:
            st.error("Chyba")
            st.exception(e)