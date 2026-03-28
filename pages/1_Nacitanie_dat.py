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


# ---------- Storage ----------
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
        json.dump(registry, f, ensure_ascii=False, indent=2)


def dataset_file_path(dataset_id):
    return os.path.join(DATASETS_DIR, f"{dataset_id}.csv")


# ---------- CSV ----------
def load_csv_safely(uploaded_file):
    for sep in [",", ";", "\t", "|"]:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=sep)
            if df.shape[1] >= 2:
                return df
        except:
            pass
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, sep=None, engine="python")


# ---------- CORE ----------
def to_standard_schema(df, customer_col, date_col, amount_mode, amount_col=None, qty_col=None, price_col=None):
    df = df.copy()

    # rename safely
    rename_map = {
        customer_col: STD_CUSTOMER,
        date_col: STD_DATE
    }

    df = df.rename(columns=rename_map)

    if amount_mode == "Dataset obsahuje amount":
        if amount_col not in df.columns:
            raise ValueError("Amount column not found")
        df = df.rename(columns={amount_col: STD_AMOUNT})

    else:
        if qty_col not in df.columns or price_col not in df.columns:
            raise ValueError("Quantity alebo Price stĺpec neexistuje")

        qty = pd.to_numeric(df[qty_col], errors="coerce")
        price = pd.to_numeric(df[price_col], errors="coerce")

        df[STD_AMOUNT] = qty * price

    # FINAL CHECK
    required = [STD_CUSTOMER, STD_DATE, STD_AMOUNT]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    return df[required]


def basic_cleaning(df):
    df = df.copy()

    df[STD_CUSTOMER] = df[STD_CUSTOMER].astype(str).str.strip()
    df[STD_DATE] = pd.to_datetime(df[STD_DATE], errors="coerce")
    df[STD_AMOUNT] = pd.to_numeric(df[STD_AMOUNT], errors="coerce")

    df = df.dropna()
    df = df[df[STD_AMOUNT] > 0]

    return df


def dataset_stats(df):
    return {
        "rows": len(df),
        "customers": df[STD_CUSTOMER].nunique(),
        "revenue": df[STD_AMOUNT].sum()
    }


# ---------- UI ----------
st.title("📂 Načítanie dát")

st.info("Nahraj CSV súbor a nastav mapovanie stĺpcov.")

registry = load_registry()

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

df_raw = load_csv_safely(uploaded_file)

st.subheader("Preview")
st.dataframe(df_raw.head())

cols = list(df_raw.columns)

# ---------- MAPPING ----------
st.subheader("🧩 Mapovanie")

st.warning("⚠️ Každý stĺpec musí byť unikátny")

customer_col = st.selectbox("Customer", cols)
date_col = st.selectbox("Date", cols)

amount_mode = st.radio(
    "Zdroj hodnoty",
    ["Dataset obsahuje amount", "Vypočítať Quantity × Price"]
)

if amount_mode == "Dataset obsahuje amount":
    amount_col = st.selectbox("Amount", cols)
    qty_col = None
    price_col = None
else:
    qty_col = st.selectbox("Quantity", cols)
    price_col = st.selectbox("Price", cols)
    amount_col = None


# ---------- VALIDATION ----------
selected_cols = [customer_col, date_col]

if amount_mode == "Dataset obsahuje amount":
    selected_cols.append(amount_col)
else:
    selected_cols.extend([qty_col, price_col])

if len(set(selected_cols)) != len(selected_cols):
    st.error("❌ Vybral si rovnaký stĺpec viackrát!")
    st.stop()


# ---------- SAVE ----------
if st.button("💾 Uložiť dataset"):

    try:
        df_std = to_standard_schema(
            df_raw,
            customer_col,
            date_col,
            amount_mode,
            amount_col,
            qty_col,
            price_col
        )

        df_clean = basic_cleaning(df_std)

        if df_clean.empty:
            st.error("❌ Dataset je prázdny po čistení")
            st.stop()

        # save
        ensure_storage()
        csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
        dataset_id = hashlib.sha256(csv_bytes).hexdigest()[:16]

        with open(dataset_file_path(dataset_id), "wb") as f:
            f.write(csv_bytes)

        registry = load_registry()

        registry["datasets"].insert(0, {
            "id": dataset_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows": len(df_clean),
            "customers": df_clean[STD_CUSTOMER].nunique()
        })

        registry["active_dataset_id"] = dataset_id
        save_registry(registry)

        st.session_state["df_transactions"] = df_clean
        st.session_state["active_dataset_id"] = dataset_id

        stats = dataset_stats(df_clean)

        st.success("✅ Dataset uložený")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", stats["rows"])
        c2.metric("Customers", stats["customers"])
        c3.metric("Revenue", f"{stats['revenue']:.2f}")

    except Exception as e:
        st.error("❌ Chyba pri spracovaní datasetu")
        st.exception(e)