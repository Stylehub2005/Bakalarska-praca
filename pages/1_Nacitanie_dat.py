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


# ---------- STORAGE ----------
def ensure_storage():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(ANALYSES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def load_registry():
    ensure_storage()
    if not os.path.exists(REGISTRY_PATH):
        return {"datasets": [], "active_dataset_id": None}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def dataset_file_path(dataset_id):
    return os.path.join(DATASETS_DIR, f"{dataset_id}.csv")


def delete_dataset_files(dataset_id):
    p = dataset_file_path(dataset_id)
    if os.path.exists(p):
        os.remove(p)

    for fname in os.listdir(ANALYSES_DIR):
        if fname.startswith(f"{dataset_id}_"):
            try:
                os.remove(os.path.join(ANALYSES_DIR, fname))
            except:
                pass


# ---------- CSV ----------
def load_csv_safely(uploaded_file):
    for sep in [",", ";", "\t", "|"]:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=sep)
            if df.shape[1] >= 2:
                return df
        except:
            continue
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, sep=None, engine="python")


def guess_index(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return cols.index(lower_map[cand.lower()])
    return 0


# ---------- TRANSFORM ----------
def to_standard_schema(df_raw, customer_col, date_col, amount_mode, amount_col=None, qty_col=None, price_col=None):
    df = df_raw.copy()

    df = df.rename(columns={customer_col: STD_CUSTOMER, date_col: STD_DATE})

    if amount_mode == "Dataset obsahuje amount":
        if amount_col not in df.columns:
            raise ValueError("❌ Stĺpec amount neexistuje.")
        df = df.rename(columns={amount_col: STD_AMOUNT})

    else:
        if qty_col not in df.columns or price_col not in df.columns:
            raise ValueError("❌ Stĺpce Quantity alebo Price neexistujú.")

        # 🔥 VALIDÁCIA
        if not pd.api.types.is_numeric_dtype(df[qty_col]):
            raise ValueError("❌ Quantity musí byť číselný stĺpec.")

        if not pd.api.types.is_numeric_dtype(df[price_col]):
            raise ValueError("❌ Price musí byť číselný stĺpec.")

        qty = pd.to_numeric(df[qty_col], errors="coerce")
        price = pd.to_numeric(df[price_col], errors="coerce")

        df[STD_AMOUNT] = qty * price

    return df[[STD_CUSTOMER, STD_DATE, STD_AMOUNT]].copy()


def basic_cleaning(df):
    df = df.copy()
    df[STD_CUSTOMER] = df[STD_CUSTOMER].astype(str).str.strip()
    df[STD_DATE] = pd.to_datetime(df[STD_DATE], errors="coerce")
    df[STD_AMOUNT] = pd.to_numeric(df[STD_AMOUNT], errors="coerce")

    df = df.dropna()
    df = df[df[STD_AMOUNT] > 0]

    return df


# ---------- UI ----------
st.title("📂 Načítanie a overenie dát")

# ---------- INFO ----------
st.info("""
📌 **Aký dataset nahrať?**

• customer_id  
• transaction_date  
• amount  

alebo  

• Quantity + Price  

→ aplikácia vypočíta amount automaticky
""")

# example
example_df = pd.DataFrame({
    "customer_id": ["C001", "C001", "C002"],
    "transaction_date": ["2023-01-01", "2023-01-10", "2023-01-05"],
    "amount": [100, 50, 200]
})

st.download_button(
    "⬇️ Stiahnuť ukážkový dataset",
    data=example_df.to_csv(index=False).encode("utf-8"),
    file_name="example_dataset.csv"
)

registry = load_registry()

# ---------- HISTORY ----------
st.subheader("🗃 História datasetov")

datasets = registry.get("datasets", [])

if datasets:
    options = []
    id_map = {}

    for d in datasets:
        label = f"{d['created_at']} | {d.get('original_name')}"
        options.append(label)
        id_map[label] = d["id"]

    selected = st.selectbox("Vyber dataset", options)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Nastaviť ako aktívny"):
            ds_id = id_map[selected]
            registry["active_dataset_id"] = ds_id
            save_registry(registry)

            path = dataset_file_path(ds_id)
            if os.path.exists(path):
                df = pd.read_csv(path)
                df[STD_DATE] = pd.to_datetime(df[STD_DATE], errors="coerce")
                df[STD_AMOUNT] = pd.to_numeric(df[STD_AMOUNT], errors="coerce")
                df = df.dropna()

                st.session_state["df_transactions"] = df
                st.session_state["active_dataset_id"] = ds_id

            st.success("Aktívny dataset nastavený")
            st.rerun()

    with col2:
        if st.button("🗑 Zmazať dataset"):
            ds_id = id_map[selected]
            registry["datasets"] = [d for d in datasets if d["id"] != ds_id]
            save_registry(registry)
            delete_dataset_files(ds_id)
            st.warning("Dataset zmazaný")
            st.rerun()

else:
    st.info("Zatiaľ žiadne datasety")

st.divider()

# ---------- UPLOAD ----------
st.subheader("⬆️ Nahrať dataset")

file = st.file_uploader("Vyber CSV súbor", type=["csv"])

if file is None:
    st.stop()

df_raw = load_csv_safely(file)

st.dataframe(df_raw.head())

cols = list(df_raw.columns)

st.subheader("🧩 Mapovanie stĺpcov")

customer_col = st.selectbox("Zákazník", cols)
date_col = st.selectbox("Dátum", cols)

amount_mode = st.radio(
    "Zdroj hodnoty",
    ["Dataset obsahuje amount", "Vypočítať Quantity × Price"]
)

if amount_mode == "Dataset obsahuje amount":
    amount_col = st.selectbox("Amount", cols)
    qty_col = price_col = None
else:
    st.warning("⚠️ Vyber číselné stĺpce pre Quantity a Price")
    qty_col = st.selectbox("Quantity", cols)
    price_col = st.selectbox("Price", cols)
    amount_col = None

# ---------- SAVE ----------
if st.button("💾 Uložiť dataset", type="primary"):

    try:
        df = to_standard_schema(df_raw, customer_col, date_col, amount_mode, amount_col, qty_col, price_col)
    except Exception as e:
        st.error(str(e))
        st.stop()

    df = basic_cleaning(df)

    if df.empty:
        st.warning("Žiadne validné dáta")
        st.stop()

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    dataset_id = hashlib.sha256(csv_bytes).hexdigest()[:16]

    path = dataset_file_path(dataset_id)
    with open(path, "wb") as f:
        f.write(csv_bytes)

    registry["datasets"].append({
        "id": dataset_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "original_name": file.name
    })

    registry["active_dataset_id"] = dataset_id
    save_registry(registry)

    st.session_state["df_transactions"] = df
    st.session_state["active_dataset_id"] = dataset_id

    st.success("✅ Dataset uložený")