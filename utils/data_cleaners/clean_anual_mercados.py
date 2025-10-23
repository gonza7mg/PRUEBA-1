import pandas as pd
from utils.data_prep import (
    unify_columns_lower, clean_strings, coerce_numeric, build_period_column,
    drop_dupes_and_aggregate, group_small_ops
)

_TEXT = ["operador","mercado"]

def clean_anual_mercados(df: pd.DataFrame) -> pd.DataFrame:
    df = unify_columns_lower(df)
    df = clean_strings(df, [c for c in _TEXT if c in df.columns])
    df = coerce_numeric(df, prefer_comma_decimal=True)
    df = build_period_column(df)

    # normalizar etiquetas de mercado (si vienen variantes)
    if "mercado" in df.columns:
        df["mercado"] = df["mercado"].str.replace(r"\s+", " ", regex=True).str.strip()

    keys = [k for k in ["periodo","mercado","operador"] if k in df.columns]
    if keys:
        df = drop_dupes_and_aggregate(df, keys)

    if "operador" in df.columns:
        df = group_small_ops(df, top_n=5)

    if "periodo" in df.columns:
        df = df[df["periodo"].dt.year.between(2010, 2100)]

    prefer = [c for c in ["periodo","mercado","operador","valor"] if c in df.columns]
    rest = [c for c in df.columns if c not in prefer]
    return df[prefer + rest]
