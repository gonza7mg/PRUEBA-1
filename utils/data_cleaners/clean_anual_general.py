import pandas as pd
from utils.data_prep import (
    unify_columns_lower, clean_strings, coerce_numeric, build_period_column,
    drop_dupes_and_aggregate, group_small_ops
)

_TEXT = ["operador","servicio","tipo_de_ingreso"]

def clean_anual_general(df: pd.DataFrame) -> pd.DataFrame:
    df = unify_columns_lower(df)
    df = clean_strings(df, [c for c in _TEXT if c in df.columns])
    df = coerce_numeric(df, prefer_comma_decimal=True)
    df = build_period_column(df)

    # nombre de valor típico (valor/importe/ingresos/lineas...)
    value_col = "valor" if "valor" in df.columns else None
    if not value_col:
        for cand in ["importe","ingresos","lineas","clientes","miles_de_euros"]:
            if cand in df.columns:
                value_col = cand; break

    # eliminar duplicados por llaves típicas
    keys = [k for k in ["periodo","operador","servicio","tipo_de_ingreso"] if k in df.columns]
    if keys:
        df = drop_dupes_and_aggregate(df, keys)

    # agrupar pequeños
    if "operador" in df.columns:
        df = group_small_ops(df, top_n=5)

    # filtrar años razonables (opcional)
    if "periodo" in df.columns:
        df = df[df["periodo"].dt.year.between(2010, 2100)]

    # reordenar columnas
    prefer = [c for c in ["periodo","operador","servicio","tipo_de_ingreso", value_col] if c and c in df.columns]
    rest = [c for c in df.columns if c not in prefer]
    return df[prefer + rest]
