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
    # kind example: "rfm"
    return os.path.join(ANALYSES_DIR, f"{dataset_id}_{kind}.parquet")


def delete_dataset_files(dataset_id: str) -> None:
    p = dataset_file_path(dataset_id)
    if os.path.exists(p):
        os.remove(p)

    # delete analyses linked to dataset_id
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

    if amount_mode == "Amount column exists":
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

st.markdown(
    """
Táto stránka umožňuje:
- nahrať nový dataset (CSV) a uložiť ho do histórie,
- vybrať aktívny dataset z histórie,
- nastaviť mapovanie stĺpcov (customer/date/amount),
- vyčistiť dáta a uložiť ich do session pre ďalšie kroky.
"""
)

registry = load_registry()

# --- History / Active dataset selector
st.subheader("🗃 História datasetov")

datasets = registry.get("datasets", [])
active_id = registry.get("active_dataset_id")

if datasets:
    # show list with a friendly label
    options = []
    id_by_label = {}
    for d in datasets:
        label = f"{d['created_at']} | {d.get('original_name','dataset')} | rows={d.get('rows','?')} customers={d.get('customers','?')}"
        options.append(label)
        id_by_label[label] = d["id"]

    default_label = None
    if active_id:
        for d in datasets:
            if d["id"] == active_id:
                default_label = f"{d['created_at']} | {d.get('original_name','dataset')} | rows={d.get('rows','?')} customers={d.get('customers','?')}"
                break

    selected_label = st.selectbox(
        "Vyber aktívny dataset",
        options,
        index=options.index(default_label) if default_label in options else 0
    )

    colA, colB, colC = st.columns([1, 1, 2])

    with colA:
        if st.button("✅ Nastaviť ako aktívny"):
            registry["active_dataset_id"] = id_by_label[selected_label]
            save_registry(registry)
            st.success("Aktívny dataset bol nastavený.")
            st.rerun()

    with colB:
        if st.button("🗑 Zmazať vybraný dataset"):
            ds_id = id_by_label[selected_label]
            # remove from registry
            registry["datasets"] = [d for d in registry["datasets"] if d["id"] != ds_id]
            if registry.get("active_dataset_id") == ds_id:
                registry["active_dataset_id"] = registry["datasets"][0]["id"] if registry["datasets"] else None
            save_registry(registry)
            # remove files
            delete_dataset_files(ds_id)
            # clear session if it referenced it
            if st.session_state.get("active_dataset_id") == ds_id:
                st.session_state.pop("df_transactions", None)
                st.session_state.pop("df_rfm", None)
                st.session_state.pop("active_dataset_id", None)
            st.warning("Dataset bol odstránený (aj uložené analýzy).")
            st.rerun()

else:
    st.info("Zatiaľ nie sú uložené žiadne datasety. Nahraj prvý CSV nižšie.")

st.divider()

# --- Upload new dataset
st.subheader("⬆️ Nahrať nový dataset (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    # If we have an active dataset, load it to session so other pages work
    if registry.get("active_dataset_id"):
        ds_id = registry["active_dataset_id"]
        csv_path = dataset_file_path(ds_id)
        if os.path.exists(csv_path):
            # load active dataset into session on-demand
            df_raw = pd.read_csv(csv_path)
            # it is already standardized (customer_id/transaction_date/amount) because we store cleaned version
            df_raw[STD_DATE] = pd.to_datetime(df_raw[STD_DATE], errors="coerce")
            df_raw[STD_AMOUNT] = pd.to_numeric(df_raw[STD_AMOUNT], errors="coerce")
            df_raw = df_raw.dropna(subset=[STD_CUSTOMER, STD_DATE, STD_AMOUNT])
            st.session_state["df_transactions"] = df_raw
            st.session_state["active_dataset_id"] = ds_id
            st.success("Aktívny dataset je načítaný zo histórie a pripravený na analýzu.")
    st.stop()

# read raw file
try:
    df_raw = load_csv_safely(uploaded_file)
except Exception as e:
    st.error("Nepodarilo sa načítať CSV. Skontroluj formát súboru.")
    st.exception(e)
    st.stop()

st.subheader("Preview (raw)")
st.dataframe(df_raw.head(15), use_container_width=True)

cols = list(df_raw.columns)
if not cols:
    st.error("Dataset neobsahuje žiadne stĺpce.")
    st.stop()

# mapping UI
st.subheader("🧩 Mapovanie stĺpcov (nastavenie povinných polí)")

customer_default = guess_index(cols, ["customer_id", "CustomerID", "Customer ID", "Customer"])
date_default = guess_index(cols, ["transaction_date", "InvoiceDate", "date", "Date", "Invoice Date"])

customer_col = st.selectbox("Customer ID column", cols, index=customer_default)
date_col = st.selectbox("Transaction date column", cols, index=date_default)

amount_mode = st.radio(
    "Amount source",
    ["Amount column exists", "Compute as Quantity × Price"],
    horizontal=True
)

amount_col = None
qty_col = None
price_col = None

if amount_mode == "Amount column exists":
    amount_default = guess_index(cols, ["amount", "Total", "Revenue", "TotalPrice", "Price"])
    amount_col = st.selectbox("Amount column", cols, index=amount_default)
else:
    qty_default = guess_index(cols, ["Quantity", "qty", "count", "Count"])
    price_default = guess_index(cols, ["Price", "UnitPrice", "Unit Price"])
    qty_col = st.selectbox("Quantity column", cols, index=qty_default)
    price_col = st.selectbox("Price column", cols, index=price_default)

apply = st.button("✅ Uložiť do histórie a nastaviť ako aktívny", type="primary")

if not apply:
    st.info("Nastav mapovanie stĺpcov a klikni na tlačidlo vyššie.")
    st.stop()

# standardize + clean
try:
    df_std = to_standard_schema(
        df_raw=df_raw,
        customer_col=customer_col,
        date_col=date_col,
        amount_mode=amount_mode,
        amount_col=amount_col,
        qty_col=qty_col,
        price_col=price_col,
    )
except Exception as e:
    st.error("Chyba pri mapovaní stĺpcov. Skontroluj výber.")
    st.exception(e)
    st.stop()

df_clean = basic_cleaning(df_std)
if df_clean.empty:
    st.warning("Po čistení nezostali žiadne platné záznamy.")
    st.stop()

stats = dataset_stats(df_clean)

# save dataset file (standardized cleaned CSV)
ensure_storage()
csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
dataset_id = bytes_sha256(csv_bytes)[:16]  # short stable id

path = dataset_file_path(dataset_id)
with open(path, "wb") as f:
    f.write(csv_bytes)

# update registry (avoid duplicates)
created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
original_name = getattr(uploaded_file, "name", "uploaded.csv")

registry = load_registry()
if not any(d["id"] == dataset_id for d in registry["datasets"]):
    registry["datasets"].insert(
        0,
        {
            "id": dataset_id,
            "created_at": created_at,
            "original_name": original_name,
            "rows": stats["rows"],
            "customers": stats["customers"],
            "min_date": str(stats["min_date"]),
            "max_date": str(stats["max_date"]),
            "mapping": {
                "customer_col": customer_col,
                "date_col": date_col,
                "amount_mode": amount_mode,
                "amount_col": amount_col,
                "qty_col": qty_col,
                "price_col": price_col,
            },
        },
    )

registry["active_dataset_id"] = dataset_id
save_registry(registry)

# put into session for next pages
st.session_state["df_transactions"] = df_clean
st.session_state["active_dataset_id"] = dataset_id
st.session_state["mapping_info"] = registry["datasets"][0]["mapping"]

st.success("✅ Dataset uložený do histórie a nastavený ako aktívny.")
st.write("Aktívny dataset ID:", dataset_id)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{stats['rows']:,}")
c2.metric("Unique customers", f"{stats['customers']:,}")
c3.metric("Total revenue", f"{stats['total_revenue']:.2f}")
c4.metric("Avg. check", f"{stats['avg_check']:.2f}")

st.write("Date range:", stats["min_date"], "→", stats["max_date"])