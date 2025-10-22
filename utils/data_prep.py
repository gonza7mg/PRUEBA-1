import re
import unicodedata
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler

# ---------------------------
# Helpers de texto y columnas
# ---------------------------
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def unify_columns_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Minúsculas, trim y espacios→guion_bajo en nombres de columnas."""
    mapper = {}
    for c in df.columns:
        c2 = _strip_accents(c).strip().lower().replace(" ", "_")
        c2 = re.sub(r"[^a-z0-9_]", "_", c2)  # caracteres raros
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

# ---------------------------
# Conversión numérica robusta
# ---------------------------
def coerce_numeric(df: pd.DataFrame, prefer_comma_decimal: bool = True):
    """
    Convierte columnas object a numéricas cuando sea posible.
    Gestiona formatos tipo '1.234,56' o '-' como NaN.
    """
    out = df.copy()
    for c in out.columns:
        if not is_numeric_dtype(out[c]):
            s = out[c].astype(str)
            s = s.replace({"-": None, "": None, "None": None})
            if prefer_comma_decimal:
                # quita puntos de miles y cambia coma por punto
                s = s.str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False)
            out[c] = pd.to_numeric(s, errors="ignore")
    return out

# ---------------------------
# Periodos y llaves canónicas
# ---------------------------
def build_period_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea 'periodo' canónico.
    - anual: anno|anio|año
    - mensual: anio + mes
    - trimestral: anio + trimestre (marca mes: 03,06,09,12)
    Si ya existe 'periodo', lo respeta.
    """
    out = df.copy()
    if "periodo" in out.columns:
        return out

    # nombres posibles
    year_cols = [c for c in out.columns if c in ["ano","año","anio","anno","year","any","anual","anio_"]]
    month_cols = [c for c in out.columns if c in ["mes","mes_num","month"]]
    quarter_cols = [c for c in out.columns if c in ["trimestre","quarter","trim"]]

    year = None
    if year_cols:
        year = out[year_cols[0]]
    elif "fecha" in out.columns:
        year = pd.to_datetime(out["fecha"], errors="coerce").dt.year

    # mensual
    if year is not None and month_cols:
        m = pd.to_numeric(out[month_cols[0]], errors="coerce").fillna(1).astype(int).clip(1,12)
        y = pd.to_numeric(year, errors="coerce").fillna(2000).astype(int)
        out["periodo"] = pd.to_datetime(dict(year=y, month=m, day=1))
        return out

    # trimestral
    if year is not None and quarter_cols:
        q = pd.to_numeric(out[quarter_cols[0]], errors="coerce").fillna(1).astype(int).clip(1,4)
        y = pd.to_numeric(year, errors="coerce").fillna(2000).astype(int)
        month_map = {1:3, 2:6, 3:9, 4:12}
        out["periodo"] = pd.to_datetime(dict(year=y, month=q.map(month_map), day=1))
        return out

    # anual
    if year is not None:
        y = pd.to_numeric(year, errors="coerce").fillna(2000).astype(int)
        out["periodo"] = pd.to_datetime(dict(year=y, month=1, day=1))
        return out

    return out

# ---------------------------
# Duplicados y agregación
# ---------------------------
def drop_dupes_and_aggregate(df: pd.DataFrame, keys: list[str], agg_map: dict | None = None):
    """
    Elimina duplicados por llaves. Si hay múltiples filas por key, agrega:
    - por defecto: sum en numéricas, first en categóricas.
    - o usa 'agg_map' si lo pasas.
    """
    out = df.copy()
    if not keys:
        return out.drop_duplicates()
    # si no hay duplicados, devuelve igual
    if out.duplicated(subset=keys).sum() == 0:
        return out

    if agg_map is None:
        agg_map = {}
        for c in out.columns:
            if c in keys: 
                continue
            agg_map[c] = "sum" if is_numeric_dtype(out[c]) else "first"
    out = out.groupby(keys, dropna=False, as_index=False).agg(agg_map)
    return out

# ---------------------------
# Agrupar operadores pequeños
# ---------------------------
def group_small_ops(df: pd.DataFrame, top_n: int = 5, col_op: str = "operador"):
    """
    Agrupa operadores no incluidos en el top_n (por volumen de filas) en 'Otros'.
    Mantiene el total y simplifica el análisis.
    """
    out = df.copy()
    if col_op not in out.columns:
        return out
    top_ops = out[col_op].value_counts().nlargest(top_n).index
    out[col_op] = out[col_op].where(out[col_op].isin(top_ops), "Otros")
    return out

# ---------------------------
# Normalización numérica (opcional)
# ---------------------------
def normalize_minmax(df: pd.DataFrame, cols):
    """Min–Max 0–1 para columnas numéricas."""
    out = df.copy()
    if not cols:
        return out
    scaler = MinMaxScaler()
    out[cols] = scaler.fit_transform(out[cols])
    return out

# ---------------------------
# KPIs básicos (opcionales)
# ---------------------------
def add_kpis_basic(df: pd.DataFrame,
                   value_col: str,                   # p.ej., "lineas" o "ingresos"
                   keys_by: list[str],               # p.ej., ["periodo","operador","servicio"]
                   denom_for_share: list[str] = None # p.ej., ["periodo","servicio"]
                   ):
    """
    Añade KPIs básicos:
      - net_adds (Δ valor vs periodo anterior dentro de keys_by sin 'periodo')
      - growth_rate (t/t-1 - 1)
      - market_share (valor / total por 'denom_for_share')
    Requisitos: 'periodo' en keys_by para net/growth.
    """
    out = df.copy()
    if "periodo" in keys_by and value_col in out.columns:
        # net_adds y growth dentro de cada grupo (sin periodo)
        group_keys = [k for k in keys_by if k != "periodo"]
        out = out.sort_values(["periodo"] + group_keys)
        out["net_adds"] = out.groupby(group_keys)[value_col].diff()
        out["growth_rate"] = out.groupby(group_keys)[value_col].pct_change()
    # market share
    if denom_for_share and value_col in out.columns:
        denom = out.groupby(denom_for_share, dropna=False)[value_col].transform("sum")
        out["market_share"] = out[value_col] / denom.replace(0, pd.NA)
    return out
