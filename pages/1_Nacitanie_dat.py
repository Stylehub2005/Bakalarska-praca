import streamlit as st
import pandas as pd

# Internal standard columns used across the whole app:
STD_CUSTOMER = "customer_id"
STD_DATE = "transaction_date"
STD_AMOUNT = "amount"


def load_csv_safely(uploaded_file) -> pd.DataFrame:
    """Try common separators, then fall back to auto sniffing."""
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
    """Return a reasonable default index in selectbox based on common column names."""
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
    """Create standardized df with columns: customer_id, transaction_date, amount."""
    df = df_raw.copy()

    # Rename chosen cols to internal names
    rename_map = {
        customer_col: STD_CUSTOMER,
        date_col: STD_DATE,
    }
    df = df.rename(columns=rename_map)

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

    # Keep only required internal columns (plus optional extras if you want later)
    return df[[STD_CUSTOMER, STD_DATE, STD_AMOUNT]].copy()


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize IDs (string is safest)
    df[STD_CUSTOMER] = df[STD_CUSTOMER].astype(str).str.strip()

    # Parse date and amount
    df[STD_DATE] = pd.to_datetime(df[STD_DATE], errors="coerce")
    df[STD_AMOUNT] = pd.to_numeric(df[STD_AMOUNT], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=[STD_CUSTOMER, STD_DATE, STD_AMOUNT])

    # Remove empty/invalid customer ids
    df = df[df[STD_CUSTOMER] != ""]
    df = df[df[STD_CUSTOMER].str.lower() != "nan"]

    # Keep positive amounts only (typical purchase transactions)
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


st.title("📂 Načítanie a overenie dát")

st.markdown(
    """
Táto stránka slúži na:
1) **Import CSV datasetu**
2) **Mapovanie stĺpcov** (používateľ nastaví, ktoré stĺpce predstavujú customer/date/amount)
3) **Základné čistenie a validáciu**
4) Uloženie výsledku do `session_state` pre ďalšie kroky (RFM, segmentácia, trendy).
"""
)

with st.expander("✅ Povinné polia (interný formát)", expanded=False):
    st.write("Aplikácia pracuje s internými stĺpcami:")
    st.code("\n".join([STD_CUSTOMER, STD_DATE, STD_AMOUNT]))
    st.write(
        "Ak tvoj dataset používa iné názvy, nastavíš ich cez mapovanie nižšie. "
        "Ak nemáš priamo 'amount', vie sa vypočítať ako Quantity × Price."
    )

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Nahraj CSV súbor, aby sme mohli pokračovať.")
    st.stop()

# --- Load raw CSV
try:
    df_raw = load_csv_safely(uploaded_file)
except Exception as e:
    st.error("Nepodarilo sa načítať CSV. Skontroluj formát súboru.")
    st.exception(e)
    st.stop()

st.subheader("Preview (raw)")
st.dataframe(df_raw.head(15), use_container_width=True)

cols = list(df_raw.columns)
if len(cols) == 0:
    st.error("Dataset neobsahuje žiadne stĺpce.")
    st.stop()

# --- Mapping UI
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

apply = st.button("✅ Použiť mapovanie", type="primary")

if not apply:
    st.info("Nastav mapovanie stĺpcov a klikni na **Použiť mapovanie**.")
    st.stop()

# --- Standardize + clean
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

st.subheader("Preview (standardized & cleaned)")
st.dataframe(df_clean.head(15), use_container_width=True)

if df_clean.empty:
    st.warning(
        "Po čistení nezostali žiadne platné záznamy. "
        "Skontroluj dátumový stĺpec, customer id a hodnoty amount/quantity/price."
    )
    st.stop()

# --- Stats
stats = dataset_stats(df_clean)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{stats['rows']:,}")
c2.metric("Unique customers", f"{stats['customers']:,}")
c3.metric("Total revenue", f"{stats['total_revenue']:.2f}")
c4.metric("Avg. check", f"{stats['avg_check']:.2f}")

st.write("Date range:", stats["min_date"], "→", stats["max_date"])

# --- Save to session for the next pages
st.session_state["df_transactions"] = df_clean
st.session_state["mapping_info"] = {
    "customer_col": customer_col,
    "date_col": date_col,
    "amount_mode": amount_mode,
    "amount_col": amount_col,
    "qty_col": qty_col,
    "price_col": price_col,
}

st.success("✅ Dáta sú pripravené a uložené. Môžeš pokračovať na stránku **RFM analýza**.")