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


def build_growth_diagnostics(df_f, seg_mode):
    tmp = df_f.copy()
    tmp["month"] = tmp[STD_DATE].dt.to_period("M")

    growth = (
        tmp
        .groupby(["month", seg_mode])[STD_AMOUNT]
        .sum()
        .reset_index()
    )

    pivot = growth.pivot(
        index="month",
        columns=seg_mode,
        values=STD_AMOUNT
    ).sort_index()

    growth_change = pivot.pct_change()

    growth_rate = growth_change.mean().sort_values(ascending=False)
    gdf = growth_rate.reset_index()
    gdf.columns = ["Segment", "Priemerný rast"]

    events = []
    for segment in growth_change.columns:
        series = growth_change[segment].dropna()
        if series.empty:
            continue

        best_period = series.idxmax()
        worst_period = series.idxmin()
        best_value = series.max()
        worst_value = series.min()

        prev_best = best_period - 1
        prev_worst = worst_period - 1

        events.append({
            "Segment": segment,
            "Najlepší rast (%)": round(best_value * 100, 2),
            "Obdobie rastu": f"{prev_best} → {best_period}",
            "Najväčší pokles (%)": round(worst_value * 100, 2),
            "Obdobie poklesu": f"{prev_worst} → {worst_period}",
        })

    events_df = pd.DataFrame(events)

    best_event = None
    worst_event = None

    if not events_df.empty:
        best_event = events_df.loc[events_df["Najlepší rast (%)"].idxmax()].to_dict()
        worst_event = events_df.loc[events_df["Najväčší pokles (%)"].idxmin()].to_dict()

    return gdf, events_df, best_event, worst_event


st.title("📈 Trendy a monitorovanie")

st.markdown("""
## 🎯 Na čo slúži táto stránka?

Táto stránka slúži na sledovanie vývoja biznisu v čase:

- 📈 ako sa menia tržby  
- 👥 ako rastie počet zákazníkov  
- 💰 ako sa vyvíja hodnota objednávok  
- 🧩 ako sa správajú jednotlivé segmenty  

👉 Cieľ: odhaliť trendy, problémy a príležitosti.
""")

df_tx = get_transactions()
if df_tx is None:
    st.warning("Najprv načítaj dáta.")
    st.stop()

df_cl = get_clusters()
df = add_segments(df_tx, df_cl)


min_d, max_d = df[STD_DATE].min(), df[STD_DATE].max()

with st.form("filters"):

    c1, c2, c3 = st.columns(3)

    with c1:
        start = st.date_input("Od dátumu", min_d.date())

    with c2:
        end = st.date_input("Do dátumu", max_d.date())

    with c3:
        gran = st.selectbox(
            "Agregácia",
            ["Day", "Week", "Month"],
            format_func=lambda x: {
                "Day": "Deň",
                "Week": "Týždeň",
                "Month": "Mesiac"
            }[x]
        )

    seg_mode = st.radio(
        "Segmentovať podľa",
        ["cluster", "Segment_label"],
        format_func=lambda x: {
            "cluster": "Klaster",
            "Segment_label": "Segment"
        }[x],
        horizontal=True
    )

    run = st.form_submit_button("▶️ Spustiť monitoring")

if not run:
    st.stop()

df_f = df[
    (df[STD_DATE] >= pd.to_datetime(start))
    & (df[STD_DATE] <= pd.to_datetime(end))
]

rule = {
    "Day": "D",
    "Week": "W",
    "Month": "M"
}[gran]

trend = resample_freq(df_f, rule)


st.subheader("KPI")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Tržby",
    f"{df_f[STD_AMOUNT].sum():.2f}"
)

c2.metric(
    "Počet nákupov",
    len(df_f)
)

c3.metric(
    "Počet zákazníkov",
    df_f[STD_CUSTOMER].nunique()
)

c4.metric(
    "Priemerná hodnota objednávky",
    f"{df_f[STD_AMOUNT].mean():.2f}"
)


st.subheader("📈 Hlavné trendy")

fig1 = px.line(
    trend,
    x=STD_DATE,
    y="revenue",
    title="Vývoj tržieb",
    labels={
        STD_DATE: "Dátum",
        "revenue": "Tržby"
    }
)

fig1.update_layout(height=300)

st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(
    trend,
    x=STD_DATE,
    y="active_customers",
    title="Vývoj počtu zákazníkov",
    labels={
        STD_DATE: "Dátum",
        "active_customers": "Aktívni zákazníci"
    }
)

fig2.update_layout(height=300)

st.plotly_chart(fig2, use_container_width=True)

fig3 = px.line(
    trend,
    x=STD_DATE,
    y="avg_order_value",
    title="Priemerná hodnota objednávky",
    labels={
        STD_DATE: "Dátum",
        "avg_order_value": "Priemerná hodnota objednávky"
    }
)

fig3.update_layout(height=300)

st.plotly_chart(fig3, use_container_width=True)


st.subheader("🧩 Trendy segmentov")

if seg_mode in df_f.columns:

    seg_rev = (
        df_f
        .groupby([
            pd.Grouper(key=STD_DATE, freq=rule),
            seg_mode
        ])[STD_AMOUNT]
        .sum()
        .reset_index()
    )

    st.markdown("### Súhrnný pohľad")

    fig = px.line(
        seg_rev,
        x=STD_DATE,
        y=STD_AMOUNT,
        color=seg_mode,
        labels={
            STD_DATE: "Dátum",
            STD_AMOUNT: "Tržby",
            "cluster": "Klaster",
            "Segment_label": "Segment"
        },
        title="Vývoj tržieb podľa segmentov"
    )

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Detailné grafy jednotlivých segmentov")

    segments = seg_rev[seg_mode].dropna().unique().tolist()

    for segment in segments:
        seg_single = seg_rev[seg_rev[seg_mode] == segment].copy()

        fig_single = px.line(
            seg_single,
            x=STD_DATE,
            y=STD_AMOUNT,
            markers=True,
            title=f"Segment {segment} – vývoj tržieb",
            labels={
                STD_DATE: "Dátum",
                STD_AMOUNT: "Tržby"
            }
        )
        fig_single.update_layout(height=300)
        st.plotly_chart(fig_single, use_container_width=True)


st.subheader("📊 Podiel tržieb podľa segmentu")

if seg_mode in df_f.columns:

    seg_summary = (
        df_f
        .groupby(seg_mode)
        .agg(
            customers=(STD_CUSTOMER, "nunique"),
            revenue=(STD_AMOUNT, "sum"),
            avg_order=(STD_AMOUNT, "mean"),
        )
        .reset_index()
    )

    total = seg_summary["revenue"].sum()

    seg_summary["share_%"] = (
        seg_summary["revenue"] / total * 100
    ).round(2)

    seg_summary.columns = [
        "Segment",
        "Počet zákazníkov",
        "Tržby",
        "Priemerná objednávka",
        "Podiel %"
    ]

    st.dataframe(seg_summary)

    fig = px.pie(
        seg_summary,
        names="Segment",
        values="Tržby",
        title="Podiel segmentov na tržbách"
    )

    fig.update_layout(height=350)

    st.plotly_chart(fig, use_container_width=True)


st.subheader("📋 Porovnanie segmentov")

if seg_mode in df_f.columns:

    comp = (
        df_f
        .groupby(seg_mode)
        .agg(
            customers=(STD_CUSTOMER, "nunique"),
            transactions=(STD_AMOUNT, "count"),
            revenue=(STD_AMOUNT, "sum"),
        )
        .reset_index()
    )

    comp["avg_order"] = comp["revenue"] / comp["transactions"]

    comp.columns = [
        "Segment",
        "Počet zákazníkov",
        "Počet nákupov",
        "Tržby",
        "Priemerná objednávka"
    ]

    st.dataframe(
        comp.sort_values("Tržby", ascending=False)
    )


st.subheader("📈 Rast segmentov")

if seg_mode in df_f.columns:

    gdf, events_df, best_event, worst_event = build_growth_diagnostics(df_f, seg_mode)

    st.dataframe(gdf)

    if best_event is not None:
        st.success(
            f"🚀 Najrýchlejšie rástol segment **{best_event['Segment']}** "
            f"v období **{best_event['Obdobie rastu']}** "
            f"({best_event['Najlepší rast (%)']:.2f} %)."
        )

    if worst_event is not None:
        st.warning(
            f"📉 Najväčší pokles mal segment **{worst_event['Segment']}** "
            f"v období **{worst_event['Obdobie poklesu']}** "
            f"({worst_event['Najväčší pokles (%)']:.2f} %)."
        )

    st.markdown("### Detail zmien podľa období")
    st.dataframe(events_df, use_container_width=True)

st.success("Monitoring pripravený")