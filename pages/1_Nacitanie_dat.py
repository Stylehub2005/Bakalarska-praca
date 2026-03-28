import os
import json
import hashlib
from datetime import datetime
import streamlit as st
import pandas as pd

# Internal standard columns used across the whole app:
STD_CUSTOMER = "customer_id"
STD_DATE = "transaction_date"
STD_AMOUNT = "amount"

DATA_DIR = "data"
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
ANALYSES_DIR = os.path.join(DATA_DIR, "analyses")
REGISTRY_PATH = os.path.join(DATA_DIR, "registry.json")


# ---------- Storage helpers ----------
def ensure_storage():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(ANALYSES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def load_registry() -> dict:
    ensure_storage()
    if not os.path.exists(REGISTRY_PATH):
        return {"datasets": [], "active_dataset_id": None}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict) -> None:
    ensure_storage()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def bytes_sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dataset_file_path(dataset_id: str) -> str:
    return os.path.join(DATASETS_DIR, f"{dataset_id}.csv")


def analysis_file_path(dataset_id: str, kind: str) -> str:
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_{kind}.parquet")


def delete_dataset_files(dataset_id: str) -> None:
    p = dataset_file_path(dataset_id)
    if os.path.exists(p):
        os.remove(p)

    for fname in os.listdir(ANALYSES_DIR):
        if fname.startswith(f"{dataset_id}_"):
            try:
                os.remove(os.path.join(ANALYSES_DIR, fname))
            except OSError:
                pass


# ---------- CSV read + mapping ----------
def load_csv_safely(uploaded_file) -> pd.DataFrame:
    for sep in [",", ";", "\t", "|"]:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=sep)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, sep=None, engine="python")


def guess_index(cols: list[str], candidates: list[str]) -> int:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return cols.index(lower_map[cand.lower()])
    return 0


def to_standard_schema(
    df_raw: pd.DataFrame,
    customer_col: str,
    date_col: str,
    amount_mode: str,
    amount_col: str | None = None,
    qty_col: str | None = None,
    price_col: str | None = None,
) -> pd.DataFrame:
    df = df_raw.copy()

    df = df.rename(columns={customer_col: STD_CUSTOMER, date_col: STD_DATE})

    if amount_mode == "Dataset obsahuje amount":
        if not amount_col:
            raise ValueError("Amount column is not selected.")
        df = df.rename(columns={amount_col: STD_AMOUNT})
    else:
        if not qty_col or not price_col:
            raise ValueError("Quantity/Price columns are not selected.")
        qty = pd.to_numeric(df[qty_col], errors="coerce")
        price = pd.to_numeric(df[price_col], errors="coerce")
        df[STD_AMOUNT] = qty * price

    return df[[STD_CUSTOMER, STD_DATE, STD_AMOUNT]].copy()


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[STD_CUSTOMER] = df[STD_CUSTOMER].astype(str).str.strip()
    df[STD_DATE] = pd.to_datetime(df[STD_DATE], errors="coerce")
    df[STD_AMOUNT] = pd.to_numeric(df[STD_AMOUNT], errors="coerce")

    df = df.dropna(subset=[STD_CUSTOMER, STD_DATE, STD_AMOUNT])
    df = df[df[STD_CUSTOMER] != ""]
    df = df[df[STD_CUSTOMER].str.lower() != "nan"]
    df = df[df[STD_AMOUNT] > 0]
    return df


def dataset_stats(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "customers": int(df[STD_CUSTOMER].nunique()),
        "min_date": df[STD_DATE].min(),
        "max_date": df[STD_DATE].max(),
        "total_revenue": float(df[STD_AMOUNT].sum()),
        "avg_check": float(df[STD_AMOUNT].mean()) if len(df) else 0.0,
    }


# ---------- UI ----------
st.title("📂 Načítanie a overenie dát")

# ---------- NEW UX BLOCK ----------
st.info("""
📌 **Aký dataset nahrať?**

Aplikácia pracuje s transakčnými dátami zákazníkov.

**Minimálne požiadavky:**
• customer_id → identifikátor zákazníka  
• transaction_date → dátum nákupu  
• amount → hodnota nákupu  

---

📊 Alternatíva:
• Quantity + Price → automatický výpočet amount

---

⚠️ Každý riadok = jedna transakcia
""")

example_df = pd.DataFrame({
    "customer_id": ["C001", "C001", "C002"],
    "transaction_date": ["2023-01-01", "2023-01-10", "2023-01-05"],
    "amount": [100, 50, 200]
})

st.download_button(
    "⬇️ Stiahnuť ukážkový dataset",
    data=example_df.to_csv(index=False).encode("utf-8"),
    file_name="example_dataset.csv",
    mime="text/csv"
)

st.markdown("""
Táto stránka umožňuje:
- nahrať dataset,
- vybrať aktívny dataset,
- nastaviť mapovanie,
- pripraviť dáta pre analýzu.
""")

registry = load_registry()

st.subheader("🗃 História datasetov")

datasets = registry.get("datasets", [])
active_id = registry.get("active_dataset_id")

if datasets:
    options = []
    id_by_label = {}
    for d in datasets:
        label = f"{d['created_at']} | {d.get('original_name','dataset')}"
        options.append(label)
        id_by_label[label] = d["id"]

    selected_label = st.selectbox("Vyber dataset", options)

    if st.button("✅ Nastaviť ako aktívny"):
        registry["active_dataset_id"] = id_by_label[selected_label]
        save_registry(registry)
        st.rerun()

else:
    st.info("Zatiaľ nemáš žiadne datasety.")

st.divider()

st.subheader("⬆️ Nahrať dataset (CSV)")

uploaded_file = st.file_uploader("Vyber CSV súbor", type=["csv"])

if uploaded_file is None:
    st.stop()

df_raw = load_csv_safely(uploaded_file)

st.dataframe(df_raw.head(10))

cols = list(df_raw.columns)

st.subheader("🧩 Mapovanie stĺpcov")

customer_col = st.selectbox("Zákazník (customer_id)", cols)
date_col = st.selectbox("Dátum (transaction_date)", cols)

amount_mode = st.radio(
    "Zdroj hodnoty",
    ["Dataset obsahuje amount", "Vypočítať Quantity × Price"]
)

if amount_mode == "Dataset obsahuje amount":
    amount_col = st.selectbox("Amount", cols)
    qty_col = price_col = None
else:
    qty_col = st.selectbox("Quantity", cols)
    price_col = st.selectbox("Price", cols)
    amount_col = None

if st.button("💾 Uložiť dataset", type="primary"):

    df_std = to_standard_schema(
        df_raw, customer_col, date_col,
        amount_mode, amount_col, qty_col, price_col
    )

    df_clean = basic_cleaning(df_std)

    csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
    dataset_id = bytes_sha256(csv_bytes)[:16]

    path = dataset_file_path(dataset_id)
    with open(path, "wb") as f:
        f.write(csv_bytes)

    registry["datasets"].append({
        "id": dataset_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "original_name": uploaded_file.name
    })

    registry["active_dataset_id"] = dataset_id
    save_registry(registry)

    st.session_state["df_transactions"] = df_clean

    st.success("✅ Dataset uložený")