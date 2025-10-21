import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_minmax(df: pd.DataFrame, cols):
    scaler = MinMaxScaler()
    out = df.copy()
    out[cols] = scaler.fit_transform(out[cols])
    return out

def standardize(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        out[c] = (out[c] - out[c].mean()) / (out[c].std(ddof=0) if out[c].std(ddof=0) else 1.0)
    return out

def unify_columns_lower(df: pd.DataFrame):
    return df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})

def clean_strings(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    return out
