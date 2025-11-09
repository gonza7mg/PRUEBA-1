# utils/harmonize_columns.py
import re
from typing import Dict, Any
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1) Normalización de nombres de columnas (y unicidad)
# ---------------------------------------------------------------------

# Si quieres mapear nombres "especiales" a un estándar, añade aquí
REVERSE: Dict[str, str] = {
    # ejemplos:
    # "año": "anio",
    # "años": "anio",
    # "year": "anio",
    # "fecha_periodo": "periodo",
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los nombres de columnas: minúsculas, sin acentos, guiones bajos,
    y garantiza unicidad (añade sufijos __1, __2 si hay duplicados tras normalizar).
    """
    new_cols = {}
    for c in df.columns:
        c1 = (str(c).strip().lower()
              .replace("á", "a").replace("é", "e").replace("í", "i")
              .replace("ó", "o").replace("ú", "u").replace("ñ", "n"))
        c1 = re.sub(r"[^\w]+", "_", c1).strip("_")
        new_cols[c] = REVERSE.get(c1, c1)

    df = df.rename(columns=new_cols)

    # Hacer nombres únicos si hay duplicados
    seen: Dict[str, int] = {}
    uniq = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            uniq.append(c)
        else:
            seen[c] += 1
            uniq.append(f"{c}__{seen[c]}")  # p.ej., ingresos__1
    df.columns = uniq
    return df


# ---------------------------------------------------------------------
# 2) Conversión robusta a numérico (soporta duplicadas)
# ---------------------------------------------------------------------

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte a numérico las columnas relevantes:
    - Si df[c] es DataFrame (por duplicados), convierte cada subcolumna.
    - Sustituye errores por NaN (trazabilidad) y elimina warnings futuros.
    """
    numeric_like = {
        "ingresos", "gastos", "inversiones", "ebitda",
        "empleados_por_operador", "lineas", "penetracion", "cuota",
        "ingresos_por_operador", "gastos_por_operador", "inversiones_por_operador",
        "paquetes"  # por si aparece como cuenta
    }

    def to_num_series(s: pd.Series) -> pd.Series:
        """Convierte texto tipo '1.234,56' → 1234.56, devolviendo NaN si no puede."""
        if s.dtype.kind in "biufc":
            return s
        s2 = (s.astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False))
        return pd.to_numeric(s2, errors="coerce")

    for c in df.columns:
        obj = df[c]
        if isinstance(obj, pd.DataFrame):  # caso columnas duplicadas
            for sub in obj.columns:
                df[sub] = to_num_series(obj[sub])
        else:
            if obj.dtype == "object" or c in numeric_like:
                df[c] = to_num_series(obj)

    return df


# ---------------------------------------------------------------------
# 3) Normalización de operador y derivación temporal
# ---------------------------------------------------------------------

# Mapa de alias de operadores → nombre estándar
OPERATOR_MAP: Dict[str, str] = {
    "movistar": "Movistar",
    "telefonica": "Movistar", "telefónica": "Movistar", "telefónica de españa": "Movistar",
    "vodafone": "Vodafone", "vodafon": "Vodafone",
    "orange": "Orange",
    "jazztel": "Jazztel",
    "yoigo": "Yoigo",
    "ono": "Ono",
    "masmovil": "MásMóvil", "másmovil": "MásMóvil", "mas móvil": "MásMóvil", "más móvil": "MásMóvil",
    "otros": "Otros",
}

def normalize_operators(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la columna 'operador' si existe."""
    if "operador" in df.columns:
        s = df["operador"].astype(str).str.strip().str.lower()
        df["operador"] = s.map(lambda x: OPERATOR_MAP.get(x, x.title()))
    return df

def derive_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva/normaliza columna de año en 'anio' (Int64).
    Busca entre anio, año, ano, year, o extrae de 'periodo' si es fecha.
    """
    year_cols = ["anio", "año", "ano", "year"]
    found = None
    for c in year_cols:
        if c in df.columns:
            found = c
            break
    if found:
        df["anio"] = pd.to_numeric(df[found], errors="coerce").astype("Int64")

    if "anio" not in df.columns and "periodo" in df.columns:
        dt = pd.to_datetime(df["periodo"], errors="coerce", dayfirst=True)
        df["anio"] = dt.dt.year.astype("Int64")

    return df


# ---------------------------------------------------------------------
# 4) Pipeline de armonización y un perfil/resumen
# ---------------------------------------------------------------------

def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline simple de armonización:
    1) normaliza nombres de columnas (y unicidad)
    2) deriva 'anio'
    3) normaliza 'operador'
    4) convierte a numérico
    """
    df = standardize_columns(df)
    df = derive_year(df)
    df = normalize_operators(df)
    df = coerce_numeric(df)
    return df

def profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un pequeño perfil/resumen (útil para logs).
    """
    summary: Dict[str, Any] = {
        "rows": len(df),
        "cols": df.shape[1],
        "nulls": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }
    return pd.DataFrame([summary])
