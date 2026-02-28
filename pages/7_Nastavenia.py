import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

SETTINGS_PATH = "data/settings.json"
DEFAULT_SETTINGS = {
    "rfm_weights": {"R": 1.0, "F": 1.0, "M": 1.0},
    "default_scaler": "StandardScaler",
    "auto_k": {"k_min": 2, "k_max": 10},
    "segmentation_default_algorithm": "K-Means",
}


def ensure_data_dir():
    os.makedirs("data", exist_ok=True)


def load_settings() -> dict:
    ensure_data_dir()
    if not os.path.exists(SETTINGS_PATH):
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        # merge with defaults (forward-compatible)
        merged = DEFAULT_SETTINGS.copy()
        merged.update(s)
        merged["rfm_weights"] = {**DEFAULT_SETTINGS["rfm_weights"], **merged.get("rfm_weights", {})}
        merged["auto_k"] = {**DEFAULT_SETTINGS["auto_k"], **merged.get("auto_k", {})}
        return merged
    except Exception:
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    ensure_data_dir()
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def get_rfm_for_auto_k():
    # Use clusters DF if exists (it already contains RFM columns), else df_rfm
    df = st.session_state.get("df_rfm")
    if df is not None and not df.empty:
        return df.copy()
    return None


def scaler_from_name(name: str):
    return StandardScaler() if name == "StandardScaler" else MinMaxScaler()


st.title("⚙ Nastavenia segmentácie")

st.markdown(
    """
Táto stránka slúži na nastavenie parametrov segmentácie a automatický výber počtu klastrov.
Nastavenia sa ukladajú do:
- `data/settings.json`
- `st.session_state["settings"]`

Neskôr budú tieto nastavenia použité na stránkach **RFM analýza** a **Segmentácia**.
"""
)

settings = load_settings()

# ---------- Settings form ----------
st.subheader("Základné nastavenia")

with st.form("settings_form"):
    st.write("### Váhy RFM (pre interpretáciu / prípadné váženie)")
    c1, c2, c3 = st.columns(3)
    with c1:
        wR = st.number_input("Weight Recency (R)", min_value=0.0, max_value=10.0, value=float(settings["rfm_weights"]["R"]), step=0.1)
    with c2:
        wF = st.number_input("Weight Frequency (F)", min_value=0.0, max_value=10.0, value=float(settings["rfm_weights"]["F"]), step=0.1)
    with c3:
        wM = st.number_input("Weight Monetary (M)", min_value=0.0, max_value=10.0, value=float(settings["rfm_weights"]["M"]), step=0.1)

    st.write("### Predvolené nastavenia segmentácie")
    default_scaler = st.selectbox(
        "Default scaler",
        ["StandardScaler", "MinMaxScaler"],
        index=["StandardScaler", "MinMaxScaler"].index(settings.get("default_scaler", "StandardScaler"))
    )

    default_algo = st.selectbox(
        "Default algorithm",
        ["K-Means", "DBSCAN", "Hierarchical (Agglomerative)"],
        index=["K-Means", "DBSCAN", "Hierarchical (Agglomerative)"].index(
            settings.get("segmentation_default_algorithm", "K-Means")
        )
    )

    st.write("### Auto-k rozsah (pre K-Means / Hierarchical)")
    k_min = st.slider("k_min", 2, 10, int(settings["auto_k"]["k_min"]))
    k_max = st.slider("k_max", 2, 15, int(settings["auto_k"]["k_max"]))

    save_btn = st.form_submit_button("💾 Uložiť nastavenia", type="primary")

if save_btn:
    # fix range
    if k_max < k_min:
        k_min, k_max = k_max, k_min

    settings["rfm_weights"] = {"R": float(wR), "F": float(wF), "M": float(wM)}
    settings["default_scaler"] = default_scaler
    settings["segmentation_default_algorithm"] = default_algo
    settings["auto_k"] = {"k_min": int(k_min), "k_max": int(k_max)}

    save_settings(settings)
    st.session_state["settings"] = settings
    st.success("✅ Nastavenia boli uložené.")

st.divider()

# ---------- Auto-k analysis ----------
st.subheader("📌 Automatický výber počtu klastrov (Elbow + Silhouette)")

df_rfm = get_rfm_for_auto_k()

if df_rfm is None or df_rfm.empty:
    st.info("Najprv vypočítaj RFM (stránka **RFM analýza**). Potom bude možné analyzovať auto-k.")
    st.stop()

st.markdown(
    """
Auto-k analýza používa vstupné RFM metriky (predvolené: recency, frequency, monetary).
Vypočíta:
- **Elbow (inertia)** pre K-Means
- **Silhouette score** (kvalita klastrov) pre rôzne hodnoty k

Výpočet spustíš tlačidlom.
"""
)

features = st.multiselect(
    "Features pre auto-k",
    options=["recency", "frequency", "monetary", "R_score", "F_score", "M_score", "RFM_sum"],
    default=["recency", "frequency", "monetary"]
)

scaler_name = st.selectbox(
    "Scaler pre auto-k",
    ["StandardScaler", "MinMaxScaler"],
    index=["StandardScaler", "MinMaxScaler"].index(settings.get("default_scaler", "StandardScaler"))
)

k_min = int(settings["auto_k"]["k_min"])
k_max = int(settings["auto_k"]["k_max"])
k_max = max(k_max, k_min + 1)

run_auto_k = st.button("▶️ Spustiť auto-k analýzu", type="primary", disabled=(len(features) < 2))

if not run_auto_k:
    st.stop()

# Prepare X
X = df_rfm[features].replace([float("inf"), float("-inf")], pd.NA).dropna()
if X.empty or len(X) < 10:
    st.warning("Nedostatok dát pre auto-k (príliš málo zákazníkov po čistení).")
    st.stop()

scaler = scaler_from_name(scaler_name)
X_scaled = scaler.fit_transform(X)

# Compute metrics for each k
rows = []
for k in range(k_min, k_max + 1):
    if k >= len(X_scaled):
        break

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)
    inertia = float(km.inertia_)

    sil = None
    # silhouette requires at least 2 clusters and enough samples
    if len(X_scaled) > k and len(set(labels)) > 1:
        try:
            sil = float(silhouette_score(X_scaled, labels))
        except Exception:
            sil = None

    rows.append({"k": k, "inertia": inertia, "silhouette": sil})

df_k = pd.DataFrame(rows)

st.write("Výsledky:")
st.dataframe(df_k, use_container_width=True)

# Charts
st.plotly_chart(
    px.line(df_k, x="k", y="inertia", markers=True, title="Elbow curve (Inertia vs k)"),
    use_container_width=True
)

if df_k["silhouette"].notna().any():
    st.plotly_chart(
        px.line(df_k, x="k", y="silhouette", markers=True, title="Silhouette score vs k"),
        use_container_width=True
    )
else:
    st.info("Silhouette score nie je dostupné (napr. vznikol 1 klaster pre niektoré k).")

# Suggested k
best_k = None
if df_k["silhouette"].notna().any():
    best_k = int(df_k.loc[df_k["silhouette"].idxmax(), "k"])

if best_k is not None:
    st.success(f"✅ Odporúčané k podľa max silhouette: **{best_k}**")
else:
    st.info("Odporúčanie k: použite elbow (zlom v krivke inertia).")

# Store auto-k results in session
st.session_state["auto_k_results"] = {
    "features": features,
    "scaler": scaler_name,
    "k_table": df_k,
    "best_k_silhouette": best_k,
}