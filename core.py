import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

STD_CUSTOMER = "customer_id"
STD_DATE = "transaction_date"
STD_AMOUNT = "amount"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[STD_DATE] = pd.to_datetime(df[STD_DATE], errors="coerce")
    df[STD_AMOUNT] = pd.to_numeric(df[STD_AMOUNT], errors="coerce")

    df = df.dropna(subset=[STD_CUSTOMER, STD_DATE, STD_AMOUNT])

    df = df[df[STD_AMOUNT] > 0]
    df = df[df[STD_CUSTOMER].astype(str).str.strip() != ""]

    return df


def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame(columns=[STD_CUSTOMER, "recency", "frequency", "monetary"])

    rfm = (
        df.groupby(STD_CUSTOMER)
        .agg(
            recency=(STD_DATE, lambda x: (snapshot_date - x.max()).days),
            frequency=(STD_DATE, "count"),
            monetary=(STD_AMOUNT, "sum"),
        )
        .reset_index()
    )

    return rfm


def rfm_scoring_quintiles(rfm: pd.DataFrame) -> pd.DataFrame:

    if rfm.empty:
        return rfm

    df = rfm.copy()

    df["_r_rank"] = df["recency"].rank(method="first")
    df["_f_rank"] = df["frequency"].rank(method="first")
    df["_m_rank"] = df["monetary"].rank(method="first")

    df["R_score"] = pd.qcut(df["_r_rank"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df["F_score"] = pd.qcut(df["_f_rank"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["M_score"] = pd.qcut(df["_m_rank"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    df["RFM_score"] = (
        df["R_score"].astype(str)
        + df["F_score"].astype(str)
        + df["M_score"].astype(str)
    )

    df["RFM_sum"] = df["R_score"] + df["F_score"] + df["M_score"]

    df = df.drop(columns=["_r_rank", "_f_rank", "_m_rank"])

    return df


def add_weighted_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:

    if df.empty:
        return df

    wR = float(weights.get("R", 1.0))
    wF = float(weights.get("F", 1.0))
    wM = float(weights.get("M", 1.0))

    out = df.copy()

    out["R_weighted"] = out["R_score"] * wR
    out["F_weighted"] = out["F_score"] * wF
    out["M_weighted"] = out["M_score"] * wM

    out["FM_weighted"] = out["F_weighted"] + out["M_weighted"]
    out["RFM_weighted_sum"] = (
        out["R_weighted"] + out["F_weighted"] + out["M_weighted"]
    )

    return out


def describe_segments_weighted(df: pd.DataFrame, weights: dict) -> pd.DataFrame:

    if df.empty:
        return df

    wR = float(weights.get("R", 1.0))
    wF = float(weights.get("F", 1.0))
    wM = float(weights.get("M", 1.0))

    max_R = 5.0 * wR
    max_FM = 5.0 * wF + 5.0 * wM

    out = df.copy()

    def label(row):
        r = row["R_weighted"]
        fm = row["FM_weighted"]

        if r >= 0.8 * max_R and fm >= 0.8 * max_FM:
            return "VIP / Champions"

        if r >= 0.8 * max_R and fm >= 0.6 * max_FM:
            return "Loyal / Active"

        if r >= 0.6 * max_R and fm >= 0.6 * max_FM:
            return "Potential Loyalists"

        if r <= 0.4 * max_R and fm >= 0.7 * max_FM:
            return "At Risk"

        if r <= 0.4 * max_R and fm <= 0.4 * max_FM:
            return "Lost"

        if r >= 0.8 * max_R and fm <= 0.4 * max_FM:
            return "New"

        return "Regular"

    out["Segment_label"] = out.apply(label, axis=1)

    return out


def scaler_from_name(name: str):
    if name == "StandardScaler":
        return StandardScaler()
    return MinMaxScaler()


def prepare_features(df: pd.DataFrame, features: list, scaler_name: str, weights: dict):

    if df is None or df.empty:
        return None

    X = df[features].copy()

    # Apply weights
    for col in features:
        if "recency" in col:
            X[col] *= weights.get("R", 1)
        if "frequency" in col:
            X[col] *= weights.get("F", 1)
        if "monetary" in col:
            X[col] *= weights.get("M", 1)

    scaler = scaler_from_name(scaler_name)
    X_scaled = scaler.fit_transform(X)

    return X_scaled