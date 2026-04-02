import pandas as pd
from datetime import datetime
from core import compute_rfm




def test_compute_rfm_basic():
    df = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "transaction_date": [
            "2023-01-01",
            "2023-01-10",
            "2023-01-05"
        ],
        "amount": [100, 200, 50]
    })

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    snapshot = pd.to_datetime("2023-01-11")

    rfm = compute_rfm(df, snapshot)

    c1 = rfm[rfm["customer_id"] == 1].iloc[0]

    assert c1["frequency"] == 2
    assert c1["monetary"] == 300
    assert c1["recency"] == 1



def test_single_purchase_customer():
    df = pd.DataFrame({
        "customer_id": [1],
        "transaction_date": ["2023-01-01"],
        "amount": [100]
    })

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    snapshot = pd.to_datetime("2023-01-10")

    rfm = compute_rfm(df, snapshot)

    row = rfm.iloc[0]

    assert row["frequency"] == 1
    assert row["monetary"] == 100
    assert row["recency"] == 9


# =============================
# 🧪 SCENARIO 2: empty data
# =============================

def test_empty_dataframe():
    df = pd.DataFrame(columns=["customer_id", "transaction_date", "amount"])

    snapshot = pd.to_datetime("2023-01-10")

    rfm = compute_rfm(df, snapshot)

    assert rfm.empty


# =============================
# 🧪 SCENARIO 3: outliers
# =============================

def test_outliers_handling():
    df = pd.DataFrame({
        "customer_id": [1, 1],
        "transaction_date": ["2023-01-01", "2023-01-02"],
        "amount": [100, 1000000]  # outlier
    })

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    snapshot = pd.to_datetime("2023-01-10")

    rfm = compute_rfm(df, snapshot)

    row = rfm.iloc[0]

    assert row["monetary"] == 1000100