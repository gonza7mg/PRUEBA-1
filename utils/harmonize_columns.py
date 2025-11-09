# utils/harmonize_columns.py
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------
# Diccionarios de equivalencias
# ----------------------------
CANON = {
    # temporales
    "anio": ["anio", "anno", "year"],
    "fecha": ["fecha", "mes", "periodo"],
    "trimestre": ["trimestre", "quarter", "q"],
    # entidades
    "operador": ["operador", "empresa", "operadora", "nombre_operador"],
    "provincia": ["provincia", "prov.", "province"],
    "tecnologia": ["tecnologia", "tecnología", "tecno", "technology"],
    "mercado": ["mercado", "tipo_de_mercado", "segmento", "market"],
    # métricas típicas
    "ingresos": ["ingresos", "ingresos_totales", "ingresos_total"],
    "gastos": ["gastos", "costes", "costos"],
    "inversiones": ["inversiones", "capex", "inversion"],
    "ebitda": ["ebitda", "resultado_bruto_explotacion"],
    "empleados_por_operador": ["empleados_por_operador", "empleados"],
    "lineas": ["lineas", "lineas_activas", "abonados", "suscriptores"],
    "penetracion": ["penetracion", "penetración", "penetracion_100hab"],
    "cuota": ["cuota", "cuota_mercado", "market_share"],
}

REVERSE = {alt: canon for canon, alts in CANON.items() for alt in alts}

# ----------------------------
# Funciones de armonización
# ----------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza nombres de columnas a snake_case y mapea a CANON si procede."""
    new_cols = {}
    for c in df.columns:
        c0 = c.strip()
        c1 = (
            c0.lower()
              .replace("á","a").replace("é","e").replace("í","i")
              .replace("ó","o").replace("ú","u").replace("ñ","n")
        )
        c1 = re.sub(r"[^\w]+", "_", c1).strip("_")
        new_cols[c] = REVERSE.get(c1, c1)
    return df.rename(columns=new_cols)

def derive_year(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura 'anio'. Si no existe, lo deriva desde 'trimestre' o 'fecha/periodo'."""
    if "anio" in df.columns:
        return df
    if "trimestre" in df.columns:
        # acepta formas como 2023Q1 o 2023-T1
        year = df["trimestre"].astype(str).str.extract(r"(\d{4})")[0]
        df["anio"] = pd.to_numeric(year, errors="coerce")
        return df
    for cand in ["fecha", "periodo", "mes"]:
        if cand in df.columns:
            s = pd.to_datetime(df[cand], errors="coerce")
            if s.notna().any():
                df["anio"] = s.dt.year
                return df
            # si no es parseable, intenta extraer yyyy
            year = df[cand].astype(str).str.extract(r"(\d{4})")[0]
            df["anio"] = pd.to_numeric(year, errors="coerce")
            return df
    return df

def coerce_numeric(df: pd.DataFrame, prefer: list[str] | None = None) -> pd.DataFrame:
    """Convierte a numéricas columnas típicamente cuantitativas (si hay separadores europeos)."""
    numeric_like = {"ingresos","gastos","inversiones","ebitda","empleados_por_operador","lineas","penetracion","cuota"}
    if prefer:
        numeric_like |= set(prefer)
    for c in df.columns:
        if c in numeric_like or df[c].dtype == "object":
            s = (df[c].astype(str)
                    .str.replace(".", "", regex=False)   # miles europeos
                    .str.replace(",", ".", regex=False)) # decimales
            df[c] = pd.to_numeric(s, errors="ignore")
    return df

def normalize_operators(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombre de operadores (strip, title)."""
    if "operador" in df.columns:
        df["operador"] = (df["operador"]
                          .astype(str)
                          .str.strip()
                          .str.replace(r"\s+", " ", regex=True)
                          .str.title())
        # Correcciones frecuentes
        repl = {
            "Vodafon": "Vodafone",
            "Mas Movil": "MásMóvil",
            "Mas Movil Group": "MásMóvil",
            "Yoigo Telecom": "Yoigo",
        }
        df["operador"] = df["operador"].replace(repl)
    return df

def normalize_province(df: pd.DataFrame) -> pd.DataFrame:
    if "provincia" in df.columns:
        df["provincia"] = (df["provincia"].astype(str)
                           .str.strip()
                           .str.replace(r"\s+", " ", regex=True)
                           .str.title())
    return df

def add_quarter_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Si hay 'trimestre', añade 'anio' y 'q' numérico si se puede."""
    if "trimestre" in df.columns:
        s = df["trimestre"].astype(str)
        df["q"] = pd.to_numeric(s.str.extract(r"Q(\d)")[0], errors="coerce")
        if "anio" not in df.columns:
            df = derive_year(df)
    return df

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coloca columnas clave al inicio para homogeneidad visual."""
    priority = [c for c in ["anio","trimestre","q","fecha","operador","provincia","tecnologia","mercado"] if c in df.columns]
    rest = [c for c in df.columns if c not in priority]
    return df[priority + rest]

def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    df = derive_year(df)
    df = add_quarter_parts(df)
    df = normalize_operators(df)
    df = normalize_province(df)
    df = coerce_numeric(df)
    df = reorder_columns(df)
    return df

# Para reporte de cambios básicos
def profile(df: pd.DataFrame) -> dict:
    return {
        "filas": len(df),
        "columnas": len(df.columns),
        "nulos_totales": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "tiene_anio": "anio" in df.columns,
        "tiene_trimestre": "trimestre" in df.columns,
        "tiene_operador": "operador" in df.columns,
        "tiene_provincia": "provincia" in df.columns,
    }
