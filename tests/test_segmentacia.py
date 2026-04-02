import pandas as pd
from core import prepare_features


def test_prepare_features_scaling():
    df = pd.DataFrame({
        "recency": [10, 20],
        "frequency": [1, 2],
        "monetary": [100, 200]
    })

    weights = {"R": 1, "F": 1, "M": 1}

    X = prepare_features(df, ["recency", "frequency", "monetary"], "StandardScaler", weights)

    assert X.shape == (2, 3)