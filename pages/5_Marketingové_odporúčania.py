import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px

STD_CUSTOMER = "customer_id"
SETTINGS_PATH = "data/settings.json"

DEFAULT_SETTINGS = {
    "rfm_weights": {"R": 1.0, "F": 1.0, "M": 1.0},
    "default_scaler": "StandardScaler",
    "auto_k": {"k_min": 2, "k_max": 10},
    "segmentation_default_algorithm": "K-Means",
    "segment_rules": {
        "vip_r_min": 0.8,
        "vip_fm_min": 0.8,
        "loyal_r_min": 0.8,
        "loyal_fm_min": 0.6,
        "potential_r_min": 0.6,
        "potential_fm_min": 0.6,
        "risk_r_max": 0.4,
        "risk_fm_min": 0.7,
        "lost_r_max": 0.4,
        "lost_fm_max": 0.4,
        "new_r_min": 0.8,
        "new_fm_max": 0.4,
    },
}


def ensure_data_dir():
    os.makedirs("data", exist_ok=True)


def load_settings():
    s = st.session_state.get("settings")
    if isinstance(s, dict):
        merged = DEFAULT_SETTINGS.copy()
        merged.update(s)
        merged["rfm_weights"] = {**DEFAULT_SETTINGS["rfm_weights"], **merged.get("rfm_weights", {})}
        merged["auto_k"] = {**DEFAULT_SETTINGS["auto_k"], **merged.get("auto_k", {})}
        merged["segment_rules"] = {**DEFAULT_SETTINGS["segment_rules"], **merged.get("segment_rules", {})}
        return merged

    ensure_data_dir()
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                s2 = json.load(f)
            merged = DEFAULT_SETTINGS.copy()
            merged.update(s2)
            merged["rfm_weights"] = {**DEFAULT_SETTINGS["rfm_weights"], **merged.get("rfm_weights", {})}
            merged["auto_k"] = {**DEFAULT_SETTINGS["auto_k"], **merged.get("auto_k", {})}
            merged["segment_rules"] = {**DEFAULT_SETTINGS["segment_rules"], **merged.get("segment_rules", {})}
            st.session_state["settings"] = merged
            return merged
        except Exception:
            pass

    st.session_state["settings"] = DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()


def get_best_available_df():
    dfc = st.session_state.get("df_clusters")
    if dfc is not None and not dfc.empty:
        return dfc.copy(), "clusters"
    dfr = st.session_state.get("df_rfm")
    if dfr is not None and not dfr.empty:
        return dfr.copy(), "rfm"
    return None, None


def segment_recommendation(row: pd.Series) -> str:
    label = row.get("Segment_label", None)
    if not isinstance(label, str) or label == "—":
        label = row.get("cluster_label", None)

    if isinstance(label, str):
        if "VIP" in label or "Champion" in label:
            return "Udržať a odmeniť: VIP program, exkluzívne ponuky, skorší prístup."
        if "Loyal" in label or "Active" in label:
            return "Upsell/Cross-sell: balíčky, odporúčané produkty, vernostný bonus."
        if "Potential" in label:
            return "Podporiť ďalší nákup: personalizované kupóny, onboarding séria."
        if "At Risk" in label:
            return "Reaktivácia: zľava s časovým limitom, win-back e-mail/SMS."
        if "Lost" in label:
            return "Win-back alebo vyradenie: silná motivácia alebo znížiť frekvenciu kontaktu."
        if "New" in label:
            return "Onboarding: uvítacia kampaň, vysvetliť hodnotu, odporúčania."
        if "Regular" in label:
            return "Štandardná komunikácia: personalizácia, testovanie ponúk a udržiavanie angažovanosti."
        return "Štandardná komunikácia: personalizácia podľa produktov a kategórií."

    avg_r = row.get("avg_R", None)
    avg_f = row.get("avg_F", None)
    avg_m = row.get("avg_M", None)
    if all(isinstance(x, (int, float)) for x in [avg_r, avg_f, avg_m]):
        if avg_r >= 4 and (avg_f + avg_m) >= 7:
            return "VIP / top zákazníci: odmeny, exkluzívne ponuky, retencia."
        if avg_r <= 2 and (avg_f + avg_m) >= 7:
            return "Rizikoví s vysokou hodnotou: reaktivácia (win-back)."
        if avg_r <= 2 and (avg_f + avg_m) <= 5:
            return "Stratení / nízka hodnota: obmedziť rozpočet, iba lacné kanály."
        return "Bežní zákazníci: personalizácia a testovanie ponúk (A/B)."

    return "Odporúčanie nie je dostupné (chýbajú údaje)."


st.title("🎯 Marketingové odporúčania")

settings = load_settings()
rules = settings["segment_rules"]

df, mode = get_best_available_df()
if df is None:
    st.warning("Najprv spusti **RFM analýzu** a ideálne aj **segmentáciu**.")
    st.stop()

st.markdown("""
## 🧠 Ako vznikli segmenty?

Segmenty sú vytvorené na základe **RFM skóre**:

- **R (Recency)** – čerstvosť nákupu  
- **F (Frequency)** – počet nákupov  
- **M (Monetary)** – hodnota nákupov  

Každý zákazník dostane skóre **1–5** pre každú metriku.

👉 Následne kombinujeme:
- **R (aktivita)**  
- **F + M (hodnota zákazníka)**  

Na základe toho vznikajú segmenty.
""")

with st.expander("📐 Logika segmentácie (pravidlá)"):
    st.markdown(f"""
**VIP / Šampióni**  
R ≥ **{rules["vip_r_min"] * 100:.0f} %** maxima a (F+M) ≥ **{rules["vip_fm_min"] * 100:.0f} %** maxima

**Lojálni / Aktívni**  
R ≥ **{rules["loyal_r_min"] * 100:.0f} %** a (F+M) ≥ **{rules["loyal_fm_min"] * 100:.0f} %**

**Potenciálne lojálni**  
R ≥ **{rules["potential_r_min"] * 100:.0f} %** a (F+M) ≥ **{rules["potential_fm_min"] * 100:.0f} %**

**Ohrození**  
R ≤ **{rules["risk_r_max"] * 100:.0f} %** a (F+M) ≥ **{rules["risk_fm_min"] * 100:.0f} %**

**Stratení**  
R ≤ **{rules["lost_r_max"] * 100:.0f} %** a (F+M) ≤ **{rules["lost_fm_max"] * 100:.0f} %**

**Noví / Nízka útrata**  
R ≥ **{rules["new_r_min"] * 100:.0f} %** a (F+M) ≤ **{rules["new_fm_max"] * 100:.0f} %**
""")
    st.caption("Tieto pravidlá je možné upraviť na stránke Nastavenia.")

st.markdown("### 🏷 Význam segmentov")

segment_info = pd.DataFrame({
    "Segment": [
        "VIP / Šampióni",
        "Lojálni / Aktívni",
        "Potenciálne lojálni",
        "Ohrození",
        "Stratení",
        "Noví / Nízka útrata",
        "Bežní"
    ],
    "Význam": [
        "Najlepší zákazníci – časté a hodnotné nákupy",
        "Stabilní zákazníci – pravidelne nakupujú",
        "Majú potenciál stať sa lojálnymi zákazníkmi",
        "Hodnotní zákazníci, ktorí prestávajú nakupovať",
        "Stratení zákazníci",
        "Noví zákazníci alebo zákazníci s nízkou útratou",
        "Priemerní zákazníci"
    ]
})

st.dataframe(segment_info, use_container_width=True)

st.markdown("""
Táto stránka transformuje analytické výsledky na **marketingovo interpretovateľné výstupy**:
- prehľad segmentov a klastrov,
- ich charakteristiky,
- odporúčania kampaní,
- export zoznamu zákazníkov pre cielený marketing.
""")

st.markdown("""
## 🎯 Ako vznikli odporúčania?

Odporúčania vychádzajú z marketingovej praxe:

- zákazníci s vysokou hodnotou → **udržať (retencia)**
- zákazníci v riziku → **reaktivovať (win-back)**
- noví zákazníci → **onboarding**
- slabší zákazníci → **zvýšiť angažovanosť**
""")

with st.expander("📊 Prečo práve tieto odporúčania?"):
    st.markdown("""
**VIP / Šampióni**  
→ vysoká hodnota → treba ich udržať  
→ VIP program, exkluzívne ponuky  

**Ohrození**  
→ kedysi hodnotní, teraz neaktívni  
→ reaktivačné kampane  

**Noví zákazníci**  
→ ešte nepoznajú značku  
→ onboarding, edukácia  

**Stratení**  
→ nízka pravdepodobnosť návratu  
→ obmedzené marketingové náklady  
""")

seg_key = None
if "cluster_label" in df.columns:
    seg_key = "cluster_label"
elif "cluster" in df.columns:
    seg_key = "cluster"
elif "Segment_label" in df.columns:
    seg_key = "Segment_label"
else:
    seg_key = "Segment_label"
    df["Segment_label"] = "Všetci zákazníci"

st.subheader("Prehľad segmentov")

agg_map = {
    STD_CUSTOMER: "count",
}
for col in ["recency", "frequency", "monetary", "R_score", "F_score", "M_score", "RFM_sum"]:
    if col in df.columns:
        agg_map[col] = "mean"

summary = df.groupby(seg_key).agg(agg_map).reset_index()
summary = summary.rename(columns={STD_CUSTOMER: "customers"})

total_customers = summary["customers"].sum()
summary["share_pct"] = (summary["customers"] / total_customers) * 100

rename_cols = {}
if "R_score" in summary.columns:
    rename_cols["R_score"] = "Priemerné R skóre"
if "F_score" in summary.columns:
    rename_cols["F_score"] = "Priemerné F skóre"
if "M_score" in summary.columns:
    rename_cols["M_score"] = "Priemerné M skóre"
if "RFM_sum" in summary.columns:
    rename_cols["RFM_sum"] = "Priemerný súčet RFM"

summary = summary.rename(columns=rename_cols)

summary = summary.rename(columns={
    seg_key: "Segment",
    "customers": "Počet zákazníkov",
    "share_pct": "Podiel (%)"
})

if "Segment_label" in summary.columns and "Segment_label" != "Segment":
    summary = summary.rename(columns={"Segment_label": "Označenie segmentu"})

if "Priemerné R skóre" in summary.columns:
    summary["avg_R"] = summary["Priemerné R skóre"]
if "Priemerné F skóre" in summary.columns:
    summary["avg_F"] = summary["Priemerné F skóre"]
if "Priemerné M skóre" in summary.columns:
    summary["avg_M"] = summary["Priemerné M skóre"]

summary["Odporúčanie"] = summary.apply(segment_recommendation, axis=1)
summary = summary.sort_values("Počet zákazníkov", ascending=False)

st.dataframe(summary, use_container_width=True)

fig = px.bar(
    summary,
    x="Segment",
    y="Počet zákazníkov",
    title="Veľkosť segmentov (počet zákazníkov)",
    labels={
        "Segment": "Segment",
        "Počet zákazníkov": "Počet zákazníkov"
    }
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Výber segmentu a export zákazníkov")

segments = summary["Segment"].astype(str).tolist()
selected = st.selectbox("Vyber segment", segments, index=0)

df_seg = df[df[seg_key].astype(str) == str(selected)].copy()
st.write(f"Zákazníci v segmente: **{len(df_seg):,}**")

if "monetary" in df_seg.columns:
    df_seg = df_seg.sort_values("monetary", ascending=False)

cols_to_show = [STD_CUSTOMER]
for c in ["recency", "frequency", "monetary", "RFM_score", "RFM_sum", "Segment_label", "cluster", "cluster_label"]:
    if c in df_seg.columns and c not in cols_to_show:
        cols_to_show.append(c)

preview_df = df_seg[cols_to_show].head(100).copy()

rename_preview = {
    STD_CUSTOMER: "ID zákazníka",
    "recency": "Recencia",
    "frequency": "Frekvencia",
    "monetary": "Monetárna hodnota",
    "RFM_score": "RFM skóre",
    "RFM_sum": "Súčet RFM",
    "Segment_label": "Označenie segmentu",
    "cluster": "Klaster",
    "cluster_label": "Označenie klastra"
}
preview_df = preview_df.rename(columns={k: v for k, v in rename_preview.items() if k in preview_df.columns})

st.dataframe(preview_df, use_container_width=True)

export_cols = [STD_CUSTOMER]
if "Segment_label" in df_seg.columns:
    export_cols.append("Segment_label")
if "cluster" in df_seg.columns:
    export_cols.append("cluster")
if "cluster_label" in df_seg.columns:
    export_cols.append("cluster_label")

csv_bytes = df_seg[export_cols].drop_duplicates().to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Stiahnuť zoznam zákazníkov (CSV)",
    data=csv_bytes,
    file_name=f"customers_{str(selected).replace(' ', '_')}.csv",
    mime="text/csv"
)

st.success("✅ Marketingové odporúčania sú pripravené. Ďalší krok: stránka **Report a export**.")