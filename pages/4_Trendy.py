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


def add_segments(df_tx, df_cl):
    if df_cl is None:
        tmp = df_tx.copy()
        tmp["cluster"] = "—"
        tmp["Segment_label"] = "—"
        return tmp
    return df_tx.merge(df_cl, on=STD_CUSTOMER, how="left")


def resample_freq(df, rule):
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
    st.warning("Najprv načítaj dáta.")
    st.stop()

df_cl = get_clusters()
df = add_segments(df_tx, df_cl)

# -------- FILTERS --------
min_d, max_d = df[STD_DATE].min(), df[STD_DATE].max()

with st.form("filters"):
    c1, c2, c3 = st.columns(3)

    with c1:
        start = st.date_input("Od", min_d.date())
    with c2:
        end = st.date_input("Do", max_d.date())
    with c3:
        gran = st.selectbox("Agregácia", ["Day", "Week", "Month"])

    seg_mode = st.radio("Segment by", ["cluster", "Segment_label"], horizontal=True)

    run = st.form_submit_button("▶️ Spustiť monitoring")

if not run:
    st.stop()

df_f = df[(df[STD_DATE] >= pd.to_datetime(start)) & (df[STD_DATE] <= pd.to_datetime(end))]

rule = {"Day": "D", "Week": "W", "Month": "M"}[gran]
trend = resample_freq(df_f, rule)

# -------- KPI --------
st.subheader("KPI")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue", f"{df_f[STD_AMOUNT].sum():.2f}")
c2.metric("Transactions", len(df_f))
c3.metric("Customers", df_f[STD_CUSTOMER].nunique())
c4.metric("Avg order", f"{df_f[STD_AMOUNT].mean():.2f}")

# -------- TRENDS --------
st.subheader("Trendy")

st.plotly_chart(px.line(trend, x=STD_DATE, y="revenue"))
st.plotly_chart(px.line(trend, x=STD_DATE, y="active_customers"))
st.plotly_chart(px.line(trend, x=STD_DATE, y="avg_order_value"))

# -------- SEGMENT TRENDS --------
st.subheader("Segment trends")

if seg_mode in df_f.columns:

    seg_rev = (
        df_f.groupby([pd.Grouper(key=STD_DATE, freq=rule), seg_mode])[STD_AMOUNT]
        .sum()
        .reset_index()
    )

    st.plotly_chart(px.area(seg_rev, x=STD_DATE, y=STD_AMOUNT, color=seg_mode))


# -------- 1. Revenue share --------
st.subheader("📊 Revenue share by segment")

if seg_mode in df_f.columns:

    seg_summary = (
        df_f.groupby(seg_mode)
        .agg(
            customers=(STD_CUSTOMER, "nunique"),
            revenue=(STD_AMOUNT, "sum"),
            avg_order=(STD_AMOUNT, "mean"),
        )
        .reset_index()
    )

    total = seg_summary["revenue"].sum()
    seg_summary["share_%"] = (seg_summary["revenue"] / total * 100).round(2)

    st.dataframe(seg_summary)

    fig = px.pie(seg_summary, names=seg_mode, values="revenue")
    st.plotly_chart(fig)

# -------- 2. Segment comparison --------
st.subheader("📋 Segment comparison")

if seg_mode in df_f.columns:

    comp = (
        df_f.groupby(seg_mode)
        .agg(
            customers=(STD_CUSTOMER, "nunique"),
            transactions=(STD_AMOUNT, "count"),
            revenue=(STD_AMOUNT, "sum"),
        )
        .reset_index()
    )

    comp["avg_order"] = comp["revenue"] / comp["transactions"]

    st.dataframe(comp.sort_values("revenue", ascending=False))

# -------- 3. Growth / decline --------
st.subheader("📈 Segment growth")

if seg_mode in df_f.columns:

    df_f["month"] = df_f[STD_DATE].dt.to_period("M")

    growth = (
        df_f.groupby(["month", seg_mode])[STD_AMOUNT]
        .sum()
        .reset_index()
    )

    pivot = growth.pivot(index="month", columns=seg_mode, values=STD_AMOUNT)

    growth_rate = pivot.pct_change().mean().sort_values(ascending=False)

    gdf = growth_rate.reset_index()
    gdf.columns = [seg_mode, "growth"]

    st.dataframe(gdf)

    st.success(f"🚀 Top growing: {gdf.iloc[0][seg_mode]}")
    st.warning(f"📉 Worst: {gdf.iloc[-1][seg_mode]}")

st.success("Monitoring ready")