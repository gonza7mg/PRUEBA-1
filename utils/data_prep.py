import re
import unicodedata
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

# ─────────────────────────────────────────────────────────────
# Helpers de texto / columnas
# ─────────────────────────────────────────────────────────────
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def unify_columns_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Minúsculas, sin acentos, espacios→_ y limpieza de símbolos en nombres de columnas."""
    mapper = {}
    for c in df.columns:
        c2 = _strip_accents(str(c)).strip().lower().replace(" ", "_")
        c2 = re.sub(r"[^a-z0-9_]", "_", c2)
        c2 = re.sub(r"_+", "_", c2).strip("_")
        mapper[c] = c2
    return df.rename(columns=mapper)

def clean_strings(df: pd.DataFrame, cols):
    """Trim seguro en columnas categóricas."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    return out

# ─────────────────────────────────────────────────────────────
# Conversión numérica robusta
# ─────────────────────────────────────────────────────────────
def coerce_numeric(df: pd.DataFrame, prefer_comma_decimal: bool = True):
    """
    Convierte columnas object a numéricas cuando sea posible.
    Gestiona formatos tipo '1.234,56' y reemplaza '-' / '' por NaN.
    """
    out = df.copy()
    for c in out.columns:
        if not is_numeric_dtype(out[c]):
            s = out[c].astype(str)
            s = s.replace({"-": None, "": None, "None": None})
            if prefer_comma_decimal:
                s = s.str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False)
            out[c] = pd.to_numeric(s, errors="ignore")
    return out

# ─────────────────────────────────────────────────────────────
# Periodo canónico
# ─────────────────────────────────────────────────────────────
def build_period_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea 'periodo' canónico si no existe:
      - mensual: usa columnas 'anio|ano|año' y 'mes'
      - trimestral: 'anio' + 'trimestre' → mes fin (03/06/09/12)
      - anual: 'anio' → YYYY-01-01
      - si hay 'fecha', se intenta parsear
    """
    out = df.copy()
    if "periodo" in out.columns:
        # intentar convertir a datetime por si viene como string
        out["periodo"] = pd.to_datetime(out["periodo"], errors="coerce")
        return out

    cols = [c.lower() for c in out.columns]
    colmap = {c.lower(): c for c in out.columns}

    def has(col): return col in colmap

    year_col = None
    for y in ["anio","ano","año","anno","year"]:
        if has(y):
            year_col = colmap[y]
            break

    # mensual
    if year_col and has("mes"):
        y = pd.to_numeric(out[year_col], errors="coerce").fillna(2000).astype(int)
        m = pd.to_numeric(out[colmap["mes"]], errors="coerce").fillna(1).astype(int).clip(1,12)
        out["periodo"] = pd.to_datetime(dict(year=y, month=m, day=1))
        return out

    # trimestral
    for qname in ["trimestre","trim","quarter"]:
        if year_col and has(qname):
            y = pd.to_numeric(out[year_col], errors="coerce").fillna(2000).astype(int)
            q = pd.to_numeric(out[colmap[qname]], errors="coerce").fillna(1).astype(int).clip(1,4)
            month_map = {1:3, 2:6, 3:9, 4:12}
            out["periodo"] = pd.to_datetime(dict(year=y, month=q.map(month_map), day=1))
            return out

    # anual
    if year_col:
        y = pd.to_numeric(out[year_col], errors="coerce").fillna(2000).astype(int)
        out["periodo"] = pd.to_datetime(dict(year=y, month=1, day=1))
        return out

    # fecha
    if has("fecha"):
        out["periodo"] = pd.to_datetime(out[colmap["fecha"]], errors="coerce")

    return out

# ─────────────────────────────────────────────────────────────
# Duplicados y agregación por llaves
# ─────────────────────────────────────────────────────────────
def drop_dupes_and_aggregate(df: pd.DataFrame, keys: list[str], agg_map: dict | None = None):
    out = df.copy()
    if not keys:
        return out.drop_duplicates()
    if out.duplicated(subset=keys).sum() == 0:
        return out
    if agg_map is None:
        agg_map = {}
        for c in out.columns:
            if c in keys: 
                continue
            agg_map[c] = "sum" if is_numeric_dtype(out[c]) else "first"
    return out.groupby(keys, dropna=False, as_index=False).agg(agg_map)

# ─────────────────────────────────────────────────────────────
# Agrupar operadores pequeños
# ─────────────────────────────────────────────────────────────
def group_small_ops(df: pd.DataFrame, top_n: int = 5, col_op: str = "operador"):
    out = df.copy()
    if col_op not in out.columns:
        return out
    top_ops = out[col_op].value_counts().nlargest(top_n).index
    out[col_op] = out[col_op].where(out[col_op].isin(top_ops), "Otros")
    return out

# ─────────────────────────────────────────────────────────────
# Normalizar provincias (acento/alias simples)
# ─────────────────────────────────────────────────────────────
_PROV_FIX = {
    "a coruna": "a coruña",
    "araba alava": "araba/álava",
    "alava": "araba/álava",
    "castellon": "castellón",
    "illes balears": "islas baleares",
    "girona": "gerona",  # si quieres mapear a una sola forma; ajusta a tu preferencia
}

def normalize_province_name(name: str) -> str:
    if pd.isna(name): return name
    s = _strip_accents(str(name)).lower().strip()
    return _PROV_FIX.get(s, str(name).strip())
