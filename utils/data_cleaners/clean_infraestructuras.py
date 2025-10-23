import pandas as pd
from utils.data_prep import (
    unify_columns_lower, clean_strings, coerce_numeric, build_period_column,
    drop_dupes_and_aggregate, normalize_province_name
)

_TEXT = ["provincia","ccaa","tecnologia","tipo_de_estaciones_base","tecnologia_de_acceso","unidades","servicio"]

def clean_infraestructuras(df: pd.DataFrame) -> pd.DataFrame:
    df = unify_columns_lower(df)
    df = clean_strings(df, [c for c in _TEXT if c in df.columns])
    df = coerce_numeric(df, prefer_comma_decimal=True)
    df = build_period_column(df)

    if "provincia" in df.columns:
        df["provincia"] = df["provincia"].apply(normalize_province_name)

    # elegir llave: periodo + provincia + tecnologia/variable infra
    keys = [k for k in ["periodo","provincia","tecnologia","tecnologia_de_acceso","tipo_de_estaciones_base","servicio"] if k in df.columns]
    if keys:
        df = drop_dupes_and_aggregate(df, keys)

    if "periodo" in df.columns:
        df = df.sort_values(["periodo","provincia"])

    prefer = [c for c in ["periodo","provincia","ccaa","tecnologia","unidades","valor"] if c in df.columns]
    rest = [c for c in df.columns if c not in prefer]
    return df[prefer + rest]
