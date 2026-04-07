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


def load_settings():
    ensure_data_dir()
    if not os.path.exists(SETTINGS_PATH):
        return DEFAULT_SETTINGS.copy()

    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)

        merged = DEFAULT_SETTINGS.copy()
        merged.update(s)
        merged["rfm_weights"] = {**DEFAULT_SETTINGS["rfm_weights"], **merged.get("rfm_weights", {})}
        merged["auto_k"] = {**DEFAULT_SETTINGS["auto_k"], **merged.get("auto_k", {})}
        return merged

    except:
        return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    ensure_data_dir()
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def get_rfm_for_auto_k():
    df = st.session_state.get("df_rfm")
    if df is not None and not df.empty:
        return df.copy()
    return None


def scaler_from_name(name):
    return StandardScaler() if name == "StandardScaler" else MinMaxScaler()

st.title("⚙ Nastavenia")

st.markdown("""
## 🎯 Účel nastavení

Táto stránka umožňuje nastaviť parametre, ktoré ovplyvňujú:
- výpočet RFM skóre
- segmentáciu zákazníkov
- kvalitu clusteringu

👉 Správne nastavenie môže výrazne zmeniť výsledky analýzy.
""")

settings = load_settings()


st.subheader("Základné nastavenia")

with st.form("settings_form"):

    st.write("### Váhy RFM")

    st.markdown("""
👉 Určuje **dôležitosť jednotlivých faktorov**:

- **Recency (R)** – ako nedávno zákazník nakupoval  
- **Frequency (F)** – ako často nakupuje  
- **Monetary (M)** – koľko míňa  

💡 Príklad:
- zvýšiť R → dôležitejší je „nedávno aktívny zákazník“
- zvýšiť M → dôležitejší je zákazník s vysokou útratou
""")

    c1, c2, c3 = st.columns(3)

    with c1:
        wR = st.number_input("Váha recencie (R)", 0.0, 10.0, float(settings["rfm_weights"]["R"]), step=0.1)

    with c2:
        wF = st.number_input("Váha frekvencie (F)", 0.0, 10.0, float(settings["rfm_weights"]["F"]), step=0.1)

    with c3:
        wM = st.number_input("Váha monetárnej hodnoty (M)", 0.0, 10.0, float(settings["rfm_weights"]["M"]), step=0.1)

    st.info("👉 Predvolené nastavenie = všetky váhy sú rovnaké → klasická RFM analýza")

    st.write("### Normalizácia dát")

    st.markdown("""
👉 Normalizácia zabezpečí, že všetky premenné majú rovnaký vplyv.

- **StandardScaler** → normalizuje na priemer 0 (najčastejšie používané)
- **MinMaxScaler** → škáluje hodnoty na interval 0–1

💡 Použi:
- StandardScaler → vo väčšine prípadov
- MinMaxScaler → ak chceš interpretovateľné rozsahy
""")

    default_scaler = st.selectbox(
        "Predvolený scaler",
        ["StandardScaler", "MinMaxScaler"],
        index=["StandardScaler", "MinMaxScaler"].index(settings["default_scaler"])
    )

    st.write("### Algoritmus segmentácie")

    st.markdown("""
👉 Vyber algoritmus clusteringu:

- **K-Means** → najčastejší, rýchly, potrebuje k
- **DBSCAN** → nájde hustoty (detekcia odľahlých hodnôt)
- **Hierarchical** → stromová segmentácia

💡 Odporúčanie:
- použi **K-Means** pre väčšinu prípadov
""")

    algo_options = ["K-Means", "DBSCAN", "Hierarchical (Agglomerative)"]
    saved_algo = settings["segmentation_default_algorithm"]
    if saved_algo == "Hierarchical":
        saved_algo = "Hierarchical (Agglomerative)"

    default_algo = st.selectbox(
        "Predvolený algoritmus",
        algo_options,
        index=algo_options.index(saved_algo)
    )

    st.write("### Auto-k (výber počtu klastrov)")

    st.markdown("""
👉 Určuje rozsah testovaných hodnôt k.

- k_min → minimálny počet klastrov
- k_max → maximálny počet klastrov

💡 Príliš malé k → slabá segmentácia  
💡 Príliš veľké k → nadmerná segmentácia
""")

    k_min = st.slider("Minimálny počet klastrov (k_min)", 2, 10, int(settings["auto_k"]["k_min"]))
    k_max = st.slider("Maximálny počet klastrov (k_max)", 2, 15, int(settings["auto_k"]["k_max"]))

    save_btn = st.form_submit_button("💾 Uložiť nastavenia", type="primary")
if save_btn:

    if k_max < k_min:
        k_min, k_max = k_max, k_min

    normalized_algo = default_algo
    if default_algo == "Hierarchical (Agglomerative)":
        normalized_algo = "Hierarchical"

    settings["rfm_weights"] = {"R": wR, "F": wF, "M": wM}
    settings["default_scaler"] = default_scaler
    settings["segmentation_default_algorithm"] = normalized_algo
    settings["auto_k"] = {"k_min": k_min, "k_max": k_max}

    save_settings(settings)
    st.session_state["settings"] = settings

    st.success("✅ Nastavenia uložené")

st.divider()


st.subheader("📊 Auto-k analýza")

st.markdown("""
👉 Táto analýza pomáha nájsť optimálny počet klastrov.

Používajú sa:
- **Elbow metóda** → hľadá „zlom“
- **Silhouette skóre** → hodnotí kvalitu klastrov
""")

df_rfm = get_rfm_for_auto_k()

if df_rfm is None:
    st.warning("Najprv spusti RFM analýzu")
    st.stop()

features = st.multiselect(
    "Premenné",
    ["recency", "frequency", "monetary", "R_score", "F_score", "M_score"],
    default=["recency", "frequency", "monetary"]
)

scaler_name = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler"])

run = st.button("▶️ Spustiť analýzu", type="primary")

if not run:
    st.stop()

X = df_rfm[features].dropna()

scaler = scaler_from_name(scaler_name)
X_scaled = scaler.fit_transform(X)

rows = []

for k in range(settings["auto_k"]["k_min"], settings["auto_k"]["k_max"] + 1):

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)

    inertia = km.inertia_
    sil = silhouette_score(X_scaled, labels) if k > 1 else None

    rows.append({"k": k, "inertia": inertia, "silhouette": sil})

df_k = pd.DataFrame(rows)

preview_df = df_k.rename(columns={
    "k": "Počet klastrov",
    "inertia": "Inertia",
    "silhouette": "Silhouette skóre"
})

st.dataframe(preview_df)

st.plotly_chart(
    px.line(
        df_k,
        x="k",
        y="inertia",
        title="Elbow metóda",
        labels={
            "k": "Počet klastrov",
            "inertia": "Inertia"
        }
    ),
    use_container_width=True
)

st.plotly_chart(
    px.line(
        df_k,
        x="k",
        y="silhouette",
        title="Silhouette skóre",
        labels={
            "k": "Počet klastrov",
            "silhouette": "Silhouette skóre"
        }
    ),
    use_container_width=True
)

best_k = df_k.loc[df_k["silhouette"].idxmax(), "k"]

st.success(f"👉 Odporúčané k: {best_k}")