"""
Descarga los 6 recursos CNMC, limpia y agrupa operadores pequeños en 'Otros',
y guarda CSV “limpios” reproducibles en data/clean/.
"""
import os
import pandas as pd
from utils.cnmc_ckan import fetch_resource
from utils.data_prep import (
    unify_columns_lower, clean_strings, coerce_numeric,
    build_period_column, drop_dupes_and_aggregate, group_small_ops
)

RESOURCES = {
    "anual_datos_generales": "5e2d8f37-2385-4774-82ec-365cd83d65bd",
    "anual_mercados": "7afbf769-655d-4b43-b49f-95c2919ec1fe",
    "mensual": "3632297f-07d8-480c-aca5-c987dcde0ccb",
    "provinciales": "1efe6d64-72a8-4f45-a36c-691054f3e277",
    "trimestrales": "5da45f2f-e596-4940-b682-eab18e85288a",
    "infraestructuras": "baab2a5e-cc52-4704-a799-a28b19223a3b",
}

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

TEXT_COLS_CANDIDATES = [
    "operador","servicio","concepto","tipo_de_paquete","tipo_de_ingreso",
    "provincia","ccaa","tecnologia_de_acceso","tipo_de_ba_mayorista",
    "tipo_de_estaciones_base","unidades"
]

def basic_clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = unify_columns_lower(df)
    df = clean_strings(df, [c for c in TEXT_COLS_CANDIDATES if c in df.columns])
    df = coerce_numeric(df, prefer_comma_decimal=True)
    df = build_period_column(df)

    # Llaves típicas para colapsar duplicados (usamos las que existan)
    candidate_keys = ["periodo","operador","servicio","provincia","ccaa","tecnologia_de_acceso","concepto"]
    keys = [k for k in candidate_keys if k in df.columns]
    if keys:
        df = drop_dupes_and_aggregate(df, keys=keys)

    # Agrupar operadores pequeños en 'Otros'
    if "operador" in df.columns:
        df = group_small_ops(df, top_n=5, col_op="operador")

    return df

def run():
    for name, rid in RESOURCES.items():
        print(f"Descargando {name} ({rid})...")
        df = fetch_resource(rid)
        raw_path = os.path.join(RAW_DIR, f"{name}.csv")
        df.to_csv(raw_path, index=False, encoding="utf-8")
        print(f"  → Guardado RAW: {raw_path} ({len(df)} filas)")

        dfc = basic_clean_pipeline(df)
        clean_path = os.path.join(CLEAN_DIR, f"{name}_clean.csv")
        dfc.to_csv(clean_path, index=False, encoding="utf-8")
        print(f"  → Guardado CLEAN: {clean_path} ({len(dfc)} filas)")

if __name__ == "__main__":
    run()
