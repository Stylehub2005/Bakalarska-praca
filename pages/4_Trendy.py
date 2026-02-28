import streamlit as st
import pandas as pd
import plotly.express as px

STD_CUSTOMER = "customer_id"
STD_DATE = "transaction_date"
STD_AMOUNT = "amount"


def get_transactions():
    df = st.session_state.get("df_transactions")
    if df is None or df.empty:
        return None
    df = df.copy()
    df[STD_DATE] = pd.to_datetime(df[STD_DATE], errors="coerce")
    df[STD_AMOUNT] = pd.to_numeric(df[STD_AMOUNT], errors="coerce")
    df = df.dropna(subset=[STD_DATE, STD_AMOUNT, STD_CUSTOMER])
    return df


def get_clusters():
    df = st.session_state.get("df_clusters")
    if df is None or df.empty:
        return None
    cols = [STD_CUSTOMER]
    if "cluster" in df.columns:
        cols.append("cluster")
    if "Segment_label" in df.columns:
        cols.append("Segment_label")
    return df[cols].drop_duplicates(STD_CUSTOMER)


def add_segments(df_tx: pd.DataFrame, df_cl: pd.DataFrame | None) -> pd.DataFrame:
    if df_cl is None:
        tmp = df_tx.copy()
        tmp["cluster"] = "—"
        tmp["Segment_label"] = "—"
        return tmp
    return df_tx.merge(df_cl, on=STD_CUSTOMER, how="left")


def resample_freq(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    tmp = df.set_index(STD_DATE).sort_index()
    out = tmp.resample(rule).agg(
        revenue=(STD_AMOUNT, "sum"),
        transactions=(STD_AMOUNT, "count"),
        active_customers=(STD_CUSTOMER, "nunique"),
    )
    out["avg_order_value"] = out["revenue"] / out["transactions"]
    return out.reset_index()


st.title("📈 Trendy a monitoring")

df_tx = get_transactions()
if df_tx is None:
    st.warning("Najprv načítaj dáta na stránke **Načítanie a overenie dát**.")
    st.stop()

df_cl = get_clusters()
df = add_segments(df_tx, df_cl)

st.markdown(
    """
Táto stránka slúži na monitoring trendov:
- vývoj tržieb v čase,
- počet aktívnych zákazníkov,
- priemerná hodnota objednávky,
- vývoj segmentov (cluster / marketing label) v čase.

**Výpočet a grafy sa spustia až po kliknutí na tlačidlo v sekcii Filtre.**
"""
)

# Defaults
min_d, max_d = df[STD_DATE].min(), df[STD_DATE].max()

if "trendy_start" not in st.session_state:
    st.session_state["trendy_start"] = min_d.date()
if "trendy_end" not in st.session_state:
    st.session_state["trendy_end"] = max_d.date()
if "trendy_gran" not in st.session_state:
    st.session_state["trendy_gran"] = "Month"
if "trendy_segmode" not in st.session_state:
    st.session_state["trendy_segmode"] = "cluster"
if "trendy_run" not in st.session_state:
    st.session_state["trendy_run"] = False

# --- Filters in a form (apply only on submit)
st.subheader("Filtre")

with st.form("trendy_filters"):
    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])

    with c1:
        start = st.date_input(
            "Od",
            value=st.session_state["trendy_start"],
            min_value=min_d.date(),
            max_value=max_d.date(),
        )

    with c2:
        end = st.date_input(
            "Do",
            value=st.session_state["trendy_end"],
            min_value=min_d.date(),
            max_value=max_d.date(),
        )

    with c3:
        gran = st.selectbox(
            "Agregácia",
            ["Day", "Week", "Month"],
            index=["Day", "Week", "Month"].index(st.session_state["trendy_gran"]),
        )

    seg_mode = st.radio(
        "Vývoj segmentov podľa",
        ["cluster", "Segment_label"],
        index=["cluster", "Segment_label"].index(st.session_state["trendy_segmode"]),
        horizontal=True,
    )

    submitted = st.form_submit_button("▶️ Spustiť monitoring", type="primary")

if submitted:
    st.session_state["trendy_start"] = start
    st.session_state["trendy_end"] = end
    st.session_state["trendy_gran"] = gran
    st.session_state["trendy_segmode"] = seg_mode
    st.session_state["trendy_run"] = True

# If user hasn’t run it yet, show hint and stop
if not st.session_state["trendy_run"]:
    st.info("Nastav filtre a klikni na **Spustiť monitoring**.")
    st.stop()

# --- Apply stored filters
start_ts = pd.to_datetime(st.session_state["trendy_start"])
end_ts = pd.to_datetime(st.session_state["trendy_end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
gran = st.session_state["trendy_gran"]
seg_mode = st.session_state["trendy_segmode"]

df_f = df[(df[STD_DATE] >= start_ts) & (df[STD_DATE] <= end_ts)].copy()
if df_f.empty:
    st.warning("Žiadne dáta pre zvolené obdobie.")
    st.stop()

rule = {"Day": "D", "Week": "W", "Month": "M"}[gran]
trend = resample_freq(df_f, rule)

# --- KPIs
st.subheader("KPI")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Revenue", f"{df_f[STD_AMOUNT].sum():.2f}")
k2.metric("Transactions", f"{len(df_f):,}")
k3.metric("Active customers", f"{df_f[STD_CUSTOMER].nunique():,}")
k4.metric("Avg order value", f"{df_f[STD_AMOUNT].mean():.2f}")

# --- Trend charts
st.subheader("Trendy")

st.plotly_chart(
    px.line(trend, x=STD_DATE, y="revenue", title=f"Revenue trend ({gran})"),
    use_container_width=True,
)

st.plotly_chart(
    px.line(trend, x=STD_DATE, y="active_customers", title=f"Active customers trend ({gran})"),
    use_container_width=True,
)

st.plotly_chart(
    px.line(trend, x=STD_DATE, y="avg_order_value", title=f"Average order value ({gran})"),
    use_container_width=True,
)

# --- Segment trends
st.subheader("Vývoj segmentov")

if seg_mode in df_f.columns and df_f[seg_mode].notna().any() and (df_f[seg_mode] != "—").any():
    df_seg = df_f.dropna(subset=[seg_mode]).copy()
    df_seg = df_seg.set_index(STD_DATE).sort_index()

    seg_counts = (
        df_seg.groupby([pd.Grouper(freq=rule), seg_mode])[STD_CUSTOMER]
        .nunique()
        .reset_index(name="active_customers")
    )

    st.plotly_chart(
        px.line(
            seg_counts,
            x=STD_DATE,
            y="active_customers",
            color=seg_mode,
            title=f"Active customers by {seg_mode} ({gran})",
        ),
        use_container_width=True,
    )

    seg_rev = (
        df_seg.groupby([pd.Grouper(freq=rule), seg_mode])[STD_AMOUNT]
        .sum()
        .reset_index(name="revenue")
    )

    st.plotly_chart(
        px.area(
            seg_rev,
            x=STD_DATE,
            y="revenue",
            color=seg_mode,
            title=f"Revenue by {seg_mode} ({gran})",
        ),
        use_container_width=True,
    )
else:
    st.info(
        "Segmenty nie sú dostupné. Najprv spusti **Segmentácia** (K-Means), "
        "potom sa tu zobrazí vývoj segmentov v čase."
    )

st.success("✅ Monitoring je pripravený. Ďalšia stránka: **Marketing Insights**.")