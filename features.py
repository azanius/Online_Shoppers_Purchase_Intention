import numpy as np
import pandas as pd

# The categorical columns we one-hot encode (same as the notebook).
CATEGORICAL_COLS = ["Month", "VisitorType", "Weekend"]


def add_engineered_features(df):
    df = df.copy()  # never modify the caller's frame

    total_pages = (
        df["Administrative"] + df["Informational"] + df["ProductRelated"]
    ).astype(float)
    total_duration = (
        df["Administrative_Duration"]
        + df["Informational_Duration"]
        + df["ProductRelated_Duration"]
    ).astype(float)

    df["total_pages"] = total_pages
    df["total_duration"] = total_duration
    # np.where guards divide-by-zero: a session with 0 pages gets 0.0 instead of an error.
    df["avg_time_per_page"] = np.where(total_pages > 0, total_duration / total_pages, 0.0)
    df["product_focus"] = np.where(
        total_pages > 0, df["ProductRelated"].astype(float) / total_pages, 0.0
    )
    df["has_page_value"] = (df["PageValues"] > 0).astype(int)

    return df

def preprocess(df):
    df = add_engineered_features(df)
    df = df.copy()
    df["Weekend"] = df["Weekend"].astype(str)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True) #i used drop_first=True to avoid the dummy-variable trap
    return df
