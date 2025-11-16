# utils/harmonize_columns.py

import re
from typing import Dict, Any
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

# ---------------------------------------------------------------------
# 1) Normalización de nombres de columnas (y unicidad)
# ---------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    """
    Elimina tildes/acentos de un string.
    """
    import unicodedata
    if not isinstance(s, str):
        s = str(s)
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


# Si quieres mapear nombres "especiales" a un estándar, añade aquí
REVERSE: Dict[str, str] = {
    # ejemplos:
    # "año": "anio",
    # "años": "anio",
    # "year": "anio",
    # "ingreso": "ingresos",
    # ...
}


def _normalize_colname(c: str) -> str:
    """
    Minúsculas, sin acentos, espacios→_ y limpieza de símbolos en nombres de columnas.
    Aplica también el mapeo REVERSE cuando proceda.
    """
    if not isinstance(c, str):
        c = str(c)
    base = _strip_accents(c).strip().lower()
    base = re.sub(r"\s+", "_", base)
    base = re.sub(r"[^a-z0-9_]", "", base)

    if base in REVERSE:
        base = REVERSE[base]
    return base


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas y asegura unicidad (evita duplicados).
    """
    cols = list(df.columns)
    new_cols = []
    seen = {}
    for c in cols:
        nc = _normalize_colname(c)
        if nc in seen:
            seen[nc] += 1
            nc = f"{nc}__{seen[nc]}"
        else:
            seen[nc] = 0
        new_cols.append(nc)
    out = df.copy()
    out.columns = new_cols
    return out


# ---------------------------------------------------------------------
# 2) Derivación de año
# ---------------------------------------------------------------------

def derive_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva/normaliza columna de año en 'anio' (Int64).
    Busca entre anio, año, ano, year, o extrae de 'periodo' si es fecha.
    """
    out = df.copy()
    year_cols = ["anio", "año", "ano", "year"]
    found = None
    for c in year_cols:
        if c in out.columns:
            found = c
            break

    if found:
        out["anio"] = pd.to_numeric(out[found], errors="coerce").astype("Int64")

    if "anio" not in out.columns and "periodo" in out.columns:
        dt = pd.to_datetime(out["periodo"], errors="coerce", dayfirst=True)
        if dt.notna().sum() > 0:
            out["anio"] = dt.dt.year.astype("Int64")

    return out


# ---------------------------------------------------------------------
# 3) Normalización de operadores
# ---------------------------------------------------------------------

OPERATOR_MAP: Dict[str, str] = {
    # Normalizaciones típicas; amplía según necesidad.
    "movistar": "Movistar",
    "telefonica": "Movistar",
    "telefónica": "Movistar",
    "orange": "Orange",
    "vodafone": "Vodafone",
    "yoigo": "Yoigo",
    "masmovil": "MásMóvil",
    "másmovil": "MásMóvil",
    "mas movil": "MásMóvil",
    "más móvil": "MásMóvil",
    "otros": "Otros",
}


def normalize_operators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza la columna 'operador' si existe.
    """
    out = df.copy()
    if "operador" in out.columns:
        s = out["operador"].astype(str).str.strip().str.lower()
        out["operador"] = s.map(lambda x: OPERATOR_MAP.get(x, x.title()))
    return out


# ---------------------------------------------------------------------
# 4) Conversión robusta a numérico
# ---------------------------------------------------------------------

_NUMERIC_LIKE = {
    "ingresos",
    "gastos",
    "inversiones",
    "ebitda",
    "unidades",
    "lineas",
    "lineas_o_accesos",
    "empleados",
    "empleados_por_operador",
    "ingresos_por_operador",
    "inversiones_por_operador",
    "gastos_por_operador",
    "tasa_de_penetracion",
    "portabilidades",
}


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte a numérico las columnas que "parecen" cuantitativas.
    No revienta si alguna columna no existe.
    """
    out = df.copy()
    for c in list(out.columns):
        lc = c.lower()
        if lc in _NUMERIC_LIKE or re.search(r"(importe|monto|cuota|hhi|penetra)", lc):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ---------------------------------------------------------------------
# 5) Pipeline base de armonización (lo que ya teníais)
# ---------------------------------------------------------------------

def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline simple de armonización:
    1) normaliza nombres de columnas (y unicidad)
    2) deriva 'anio'
    3) normaliza 'operador'
    4) convierte a numérico
    """
    out = standardize_columns(df)
    out = derive_year(out)
    out = normalize_operators(out)
    out = coerce_numeric(out)
    return out


# ---------------------------------------------------------------------
# 6) Normalización de 'periodo' según dataset
# ---------------------------------------------------------------------

def _parse_trimestre_to_period(s: str):
    """
    Convierte '2010T1', '2010-Q1', '2010q3' → último día del trimestre.
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().upper()
    m = re.match(r"^(\d{4})[TQ](\d)$", s)
    if not m:
        return pd.NaT
    year = int(m.group(1))
    q = int(m.group(2))
    if q not in (1, 2, 3, 4):
        return pd.NaT
    month = q * 3
    return (pd.Timestamp(year=year, month=month, day=1) + MonthEnd(0))


def _parse_month_to_period(s: str):
    """
    Convierte '2010-03', '2010/03', '2010-03-01', 'mar-10'… a último día de mes.
    """
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()

    # Caso típico YYYY-MM o YYYY-MM-DD
    dt = pd.to_datetime(s, errors="ignore", dayfirst=False)
    if isinstance(dt, pd.Timestamp):
        return dt + MonthEnd(0)

    try:
        dt2 = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt2):
            return pd.NaT
        return dt2 + MonthEnd(0)
    except Exception:
        return pd.NaT


def normalize_periodo(df: pd.DataFrame, dataset: str | None = None) -> pd.DataFrame:
    """
    Asegura que exista una columna 'periodo' de tipo datetime64[ns].
    Usa diferentes estrategias según las columnas/dataset.
    """
    out = df.copy()

    # 1) Si ya hay 'periodo' y se puede parsear, se respeta.
    if "periodo" in out.columns:
        dt = pd.to_datetime(out["periodo"], errors="coerce", dayfirst=True)
        if dt.notna().sum() > 0:
            out["periodo"] = dt
            return out

    # 2) Dataset trimestral con columna 'trimestre'
    if dataset == "trimestrales" and "trimestre" in out.columns:
        out["periodo"] = out["trimestre"].astype(str).map(_parse_trimestre_to_period)
        return out

    # 3) Dataset mensual con columna 'mes'
    if dataset == "mensual" and "mes" in out.columns:
        out["periodo"] = out["mes"].astype(str).map(_parse_month_to_period)
        return out

    # 4) Dataset provinciales/infraestructuras/anuales solo con 'anio' / 'anno'
    if "anio" in out.columns and "periodo" not in out.columns:
        out["periodo"] = pd.to_datetime(
            out["anio"].astype("Int64").astype("string") + "-01-01",
            errors="coerce"
        )
        return out

    if "anno" in out.columns and "periodo" not in out.columns:
        out["periodo"] = pd.to_datetime(
            out["anno"].astype("Int64").astype("string") + "-01-01",
            errors="coerce"
        )
        return out

    return out


# ---------------------------------------------------------------------
# 7) Llaves semánticas
# ---------------------------------------------------------------------

def add_semantic_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea llaves semánticas estándar cuando existan las columnas necesarias.
    - periodo–operador–servicio
    - periodo–provincia–tecnologia
    """
    out = df.copy()

    if "periodo" in out.columns and pd.api.types.is_datetime64_any_dtype(out["periodo"]):
        period_str = out["periodo"].dt.strftime("%Y-%m-%d").fillna("")

        if {"operador", "servicio"}.issubset(out.columns):
            out["key_periodo_operador_servicio"] = (
                period_str + "|" +
                out["operador"].astype(str).str.strip() + "|" +
                out["servicio"].astype(str).str.strip()
            )

        if {"provincia", "tecnologia"}.issubset(out.columns):
            out["key_periodo_provincia_tecnologia"] = (
                period_str + "|" +
                out["provincia"].astype(str).str.strip() + "|" +
                out["tecnologia"].astype(str).str.strip()
            )

    return out


# ---------------------------------------------------------------------
# 8) Métricas derivadas: cuotas y HHI
# ---------------------------------------------------------------------

def _add_quota_and_hhi(df: pd.DataFrame,
                       group_keys: list[str],
                       value_col: str,
                       quota_col: str,
                       hhi_col: str) -> pd.DataFrame:
    out = df.copy()
    if not set(group_keys).issubset(out.columns) or value_col not in out.columns:
        return out

    gtot = out.groupby(group_keys)[value_col].transform("sum")
    out[quota_col] = np.where(gtot > 0, out[value_col] / gtot, np.nan)

    def _calc_hhi(s: pd.Series) -> float:
        s2 = s.fillna(0.0)
        return float((s2 ** 2).sum())

    out[hhi_col] = out.groupby(group_keys)[quota_col].transform(_calc_hhi)
    return out


def add_derived_metrics(df: pd.DataFrame, dataset: str | None = None) -> pd.DataFrame:
    """
    Añade métricas derivadas cuando tiene sentido:
    - cuotas e HHI por ingresos (cuando hay ingresos_por_operador)
    - cuotas e HHI por líneas (cuando hay 'lineas')
    No revienta si faltan columnas; simplemente no crea nada.
    """
    out = df.copy()

    # Cuotas e HHI por ingresos (anual/mensual/trimestral, mercados, etc.)
    if "ingresos_por_operador" in out.columns:
        keys: list[str] = []
        if "periodo" in out.columns:
            keys.append("periodo")
        for k in ["pais", "mercado", "tipo_de_mercado", "servicio"]:
            if k in out.columns:
                keys.append(k)
        # añadir operador para que la cuota sea por operador
        if "operador" in out.columns:
            keys.append("operador")
        keys = list(dict.fromkeys(keys))  # quitar duplicados manteniendo orden
        if len(keys) >= 2:
            out = _add_quota_and_hhi(
                out,
                group_keys=[g for g in keys if g != "operador"],
                value_col="ingresos_por_operador",
                quota_col="cuota_ingresos",
                hhi_col="hhi_ingresos",
            )

    # Cuotas e HHI por líneas (provinciales, etc.)
    if "lineas" in out.columns:
        keys2: list[str] = []
        if "periodo" in out.columns:
            keys2.append("periodo")
        for k in ["provincia", "servicio", "tecnologia"]:
            if k in out.columns:
                keys2.append(k)
        if "operador" in out.columns:
            keys2.append("operador")
        keys2 = list(dict.fromkeys(keys2))
        if len(keys2) >= 2:
            out = _add_quota_and_hhi(
                out,
                group_keys=[g for g in keys2 if g != "operador"],
                value_col="lineas",
                quota_col="cuota_lineas",
                hhi_col="hhi_lineas",
            )

    return out


# ---------------------------------------------------------------------
# 9) Pipeline completo para la capa FINAL
# ---------------------------------------------------------------------

def harmonize_full(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Pipeline completo de armonización para la capa FINAL:
    1) harmonize() existente (columnas, operadores, tipos, anio)
    2) normaliza 'periodo' según dataset
    3) añade llaves semánticas
    4) añade métricas derivadas (cuotas, HHI, etc.)
    """
    out = harmonize(df)
    out = normalize_periodo(out, dataset=dataset)
    out = add_semantic_keys(out)
    out = add_derived_metrics(out, dataset=dataset)
    return out


# ---------------------------------------------------------------------
# 10) Perfilado sencillo (para final_report)
# ---------------------------------------------------------------------

def profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un pequeño perfil/resumen (útil para logs).
    """
    summary: Dict[str, Any] = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "nulls": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }
    return pd.DataFrame([summary])
