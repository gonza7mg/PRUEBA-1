import pandas as pd
from utils.data_prep import (
    unify_columns_lower, clean_strings, coerce_numeric, build_period_column,
    drop_dupes_and_aggregate, group_small_ops, normalize_province_name
)

_TEXT = ["operador","provincia","ccaa","tecnologia","tecnologia_de_acceso","servicio"]

def clean_provinciales(df: pd.DataFrame) -> pd.DataFrame:
    df = unify_columns_lower(df)
    df = clean_strings(df, [c for c in _TEXT if c in df.columns])
    df = coerce_numeric(df, prefer_comma_decimal=True)
    df = build_period_column(df)

    # normalizar nombre de provincia sencillo (puedes ampliar mapeo en data_prep)
    if "provincia" in df.columns:
        df["provincia"] = df["provincia"].apply(normalize_province_name)

    # elegir llave: periodo + provincia + (operador/tecnologia/servicio si existen)
    base = ["periodo","provincia"]
    extras = [c for c in ["operador","tecnologia","tecnologia_de_acceso","servicio"] if c in df.columns]
    keys = [k for k in base + extras]
    if keys:
        df = drop_dupes_and_aggregate(df, keys)

    if "operador" in df.columns:
        df = group_small_ops(df, top_n=5)

    if "periodo" in df.columns:
        df = df.sort_values(["periodo","provincia"])

    prefer = [c for c in ["periodo","provincia","ccaa","operador","tecnologia","valor"] if c in df.columns]
    rest = [c for c in df.columns if c not in prefer]
    return df[prefer + rest]
