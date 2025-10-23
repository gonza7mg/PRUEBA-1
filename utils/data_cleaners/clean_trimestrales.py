import pandas as pd
from utils.data_prep import (
    unify_columns_lower, clean_strings, coerce_numeric, build_period_column,
    drop_dupes_and_aggregate, group_small_ops
)

_TEXT = ["operador","servicio"]

def clean_trimestrales(df: pd.DataFrame) -> pd.DataFrame:
    df = unify_columns_lower(df)
    df = clean_strings(df, [c for c in _TEXT if c in df.columns])
    df = coerce_numeric(df, prefer_comma_decimal=True)
    df = build_period_column(df)

    keys = [k for k in ["periodo","operador","servicio"] if k in df.columns]
    if keys:
        df = drop_dupes_and_aggregate(df, keys)

    if "operador" in df.columns:
        df = group_small_ops(df, top_n=5)

    if "periodo" in df.columns:
        df = df.sort_values("periodo")

    prefer = [c for c in ["periodo","operador","servicio","valor"] if c in df.columns]
    rest = [c for c in df.columns if c not in prefer]
    return df[prefer + rest]
