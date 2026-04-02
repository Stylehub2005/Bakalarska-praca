import pandas as pd
from core import clean


def test_clean_removes_invalid():
    df = pd.DataFrame({
        "customer_id": [1, 2],
        "transaction_date": ["2023-01-01", None],
        "amount": [100, -50]
    })

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    cleaned = clean(df)

    assert len(cleaned) == 1
    assert cleaned["amount"].iloc[0] == 100