import pandas as pd

# the categorical columns i one-hot encode (same as the notebook).
# OperatingSystems, Browser, Region and TrafficType are stored as numbers but are really
# category codes, so i encode them too. Weekend is already True/False and Revenue is the target.
CATEGORICAL_COLS = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]


def preprocess(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True) #i used drop_first=True to avoid the dummy-variable trap
    return df
