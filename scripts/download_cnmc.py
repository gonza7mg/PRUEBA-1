import os, pandas as pd
from utils.cnmc_ckan import fetch_resource
from utils.data_prep import unify_columns_lower, clean_strings

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

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = unify_columns_lower(df)
    text_cols = [c for c in ["servicio","concepto","operador","tipo_de_paquete","tipo_de_ingreso",
                             "provincia","ccaa","tecnología_de_acceso","tipo_de_ba_mayorista",
                             "tipo_de_estaciones_base","unidades"] if c in df.columns]
    df = clean_strings(df, text_cols)
    return df

def run():
    for name, rid in RESOURCES.items():
        print(f"Descargando {name} ({rid})...")
        df = fetch_resource(rid)
        raw_path = os.path.join(RAW_DIR, f"{name}.csv")
        df.to_csv(raw_path, index=False)
        dfc = basic_clean(df)
        clean_path = os.path.join(CLEAN_DIR, f"{name}_clean.csv")
        dfc.to_csv(clean_path, index=False)
        print(f"Guardado: {raw_path} / {clean_path} (filas={len(dfc)})")

if __name__ == "__main__":
    run()
