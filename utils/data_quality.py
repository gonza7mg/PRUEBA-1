import pandas as pd
import numpy as np
from datetime import datetime

# ---------- Utilidades ----------
PROVINCIAS_INE = {
    # listado mínimo; amplíalo si quieres validación completa
    "alava", "araba/álava", "albacete", "alicante", "almeria", "a coruña",
    "asturias", "avila", "badajoz", "barcelona", "burgos", "caceres", "cadiz",
    "cantabria", "castellon", "ciudad real", "cordoba", "cuenca", "girona",
    "granada", "guadalajara", "guipuzcoa", "huelva", "huesca", "illes balears",
    "jaen", "la rioja", "leon", "lleida", "lugo", "madrid", "malaga", "murcia",
    "navarra", "ourense", "palencia", "pontevedra", "salamanca", "segovia",
    "sevilla", "soria", "tarragona", "santa cruz de tenerife", "teruel",
    "toledo", "valencia", "valladolid", "vizcaya", "zamora", "zaragoza",
    "las palmas"
}

def _to_lower_str(x):
    if pd.isna(x): return x
    return str(x).strip().lower()

def _is_num_col(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _iQR_outliers(s: pd.Series):
    q1 = s.quantile(0.25); q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5*iqr; high = q3 + 1.5*iqr
    return (s < low) | (s > high)

def _rolling_zscore(s: pd.Series, window=6):
    mu = s.rolling(window, min_periods=int(window/2)).mean()
    sd = s.rolling(window, min_periods=int(window/2)).std(ddof=0)
    z = (s - mu) / sd.replace(0, np.nan)
    return z

# ---------- Chequeos por dimensión ----------
def check_completeness(df: pd.DataFrame):
    rows = []
    total = len(df)
    for c in df.columns:
        nnull = int(df[c].isna().sum())
        rows.append({
            "dimension": "completitud",
            "check": f"nulos_columna:{c}",
            "affected": nnull,
            "pct_rows": (nnull/total if total else 0),
            "notes": ""
        })
    # Huecos temporales si existe periodo
    if "periodo" in df.columns and pd.api.types.is_datetime64_any_dtype(df["periodo"]):
        years = sorted(df["periodo"].dt.year.dropna().unique())
        if len(years) >= 2:
            missing_years = []
            for y in range(years[0], years[-1]+1):
                if y not in years:
                    missing_years.append(y)
            rows.append({
                "dimension": "completitud",
                "check": "huecos_anuales",
                "affected": len(missing_years),
                "pct_rows": 0,
                "notes": f"faltan: {missing_years}" if missing_years else "OK"
            })
    return pd.DataFrame(rows)

def check_validity(df: pd.DataFrame):
    rows = []
    # no-negativos en numéricos típicos
    for c in df.columns:
        if _is_num_col(df[c]):
            neg = int((df[c] < 0).sum())
            rows.append({
                "dimension": "validez",
                "check": f"no_negativos:{c}",
                "affected": neg,
                "pct_rows": (neg/len(df) if len(df) else 0),
                "notes": "valores negativos donde no deberían"
            })
    # fechas razonables
    if "periodo" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["periodo"]):
            y = df["periodo"].dt.year.dropna()
            out_range = int((~y.between(2010, datetime.now().year + 1)).sum())
            rows.append({
                "dimension": "validez",
                "check": "rango_periodo",
                "affected": out_range,
                "pct_rows": (out_range/len(df) if len(df) else 0),
                "notes": "años fuera de rango 2010–hoy+1"
            })
        else:
            rows.append({
                "dimension": "validez",
                "check": "periodo_parseable",
                "affected": len(df),
                "pct_rows": 1.0,
                "notes": "columna 'periodo' no es datetime"
            })
    return pd.DataFrame(rows)

def check_consistency_business(df: pd.DataFrame):
    rows = []
    # Si hay cuotas de mercado, chequear ~100%
    if all(c in df.columns for c in ["periodo","mercado","operador"]) and ("cuota" in df.columns or "valor" in df.columns):
        col = "cuota" if "cuota" in df.columns else "valor"
        grp = df.groupby(["periodo","mercado"], dropna=False)[col].sum().reset_index()
        if col == "cuota":
            # debería sumar ~100 (o 1 si viene en [0,1])
            tot100 = (grp[col].round(2).between(99, 101)).mean()
            tot1 = (grp[col].round(3).between(0.99, 1.01)).mean()
            ok = max(tot100, tot1)
            rows.append({
                "dimension": "consistencia",
                "check": "suma_cuotas_≈100%",
                "affected": int((1-ok)*len(grp)),
                "pct_rows": float(1-ok),
                "notes": f"grupos OK≈ {ok:.2%}"
            })
    return pd.DataFrame(rows)

def check_uniqueness(df: pd.DataFrame, keys):
    rows = []
    if keys and all(k in df.columns for k in keys):
        dups = int(df.duplicated(subset=keys).sum())
        rows.append({
            "dimension": "unicidad",
            "check": f"duplicados:{'+'.join(keys)}",
            "affected": dups,
            "pct_rows": (dups/len(df) if len(df) else 0),
            "notes": ""
        })
    return pd.DataFrame(rows)

def check_integrity_refs(df: pd.DataFrame):
    rows = []
    if "provincia" in df.columns:
        prov = df["provincia"].astype(str).str.strip().str.lower()
        bad = int(~prov.isin(PROVINCIAS_INE).sum()) if len(prov) else 0
        rows.append({
            "dimension": "integridad",
            "check": "provincia_catalogo_INE",
            "affected": bad,
            "pct_rows": (bad/len(df) if len(df) else 0),
            "notes": "normalizar nombres si procede"
        })
    return pd.DataFrame(rows)

def check_accuracy_outliers(df: pd.DataFrame, value_cols=None, time_col="periodo", key_cols=None):
    rows = []
    if value_cols is None:
        value_cols = [c for c in df.columns if _is_num_col(df[c])]
    for c in value_cols:
        s = df[c].dropna()
        if s.empty: 
            continue
        # IQR univariante
        mask = _iQR_outliers(s)
        rows.append({
            "dimension": "exactitud",
            "check": f"outliers_IQR:{c}",
            "affected": int(mask.sum()),
            "pct_rows": (mask.mean() if len(s) else 0),
            "notes": "posibles valores atípicos"
        })
        # z-score temporal por clave (si hay fecha)
        if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]) and key_cols:
            try:
                affected = 0; total = 0
                gcols = [k for k in key_cols if k in df.columns]
                for _, sub in df.dropna(subset=[time_col]).sort_values(time_col).groupby(gcols, dropna=False):
                    if c not in sub.columns: 
                        continue
                    z = _rolling_zscore(sub[c].astype(float))
                    n = int((z.abs() > 3).sum())
                    affected += n; total += len(sub)
                rows.append({
                    "dimension": "exactitud",
                    "check": f"outliers_temporales_z3:{c}",
                    "affected": affected,
                    "pct_rows": (affected/total if total else 0),
                    "notes": "saltos anómalos en serie temporal"
                })
            except Exception:
                pass
    return pd.DataFrame(rows)

def check_timeliness(df: pd.DataFrame):
    rows = []
    if "periodo" in df.columns and pd.api.types.is_datetime64_any_dtype(df["periodo"]):
        last = df["periodo"].max()
        rows.append({
            "dimension": "actualidad",
            "check": "ultimo_periodo",
            "affected": 0,
            "pct_rows": 0,
            "notes": str(last.date()) if pd.notna(last) else "NA"
        })
    return pd.DataFrame(rows)

# ---------- Orquestador por dataset ----------
DEFAULT_KEYS = {
    "anual": ["periodo","operador","servicio"],
    "mensual": ["periodo","operador","servicio"],
    "trimestral": ["periodo","operador","servicio"],
    "provincial": ["periodo","provincia"],
    "infraestructuras": ["periodo","provincia","tecnologia"]
}

def run_quality_suite(df: pd.DataFrame, dataset_hint: str | None = None):
    # inferir claves
    keys = None
    if dataset_hint:
        for k,v in DEFAULT_KEYS.items():
            if k in dataset_hint.lower():
                keys = [c for c in v if c in df.columns]
                break
    if keys is None:
        candidates = ["periodo","operador","servicio","provincia","mercado","tecnologia"]
        keys = [c for c in candidates if c in df.columns]

    # suite
    reports = []
    reports.append(check_completeness(df))
    reports.append(check_validity(df))
    reports.append(check_consistency_business(df))
    reports.append(check_uniqueness(df, keys))
    reports.append(check_integrity_refs(df))
    reports.append(check_accuracy_outliers(df, value_cols=None, time_col="periodo", key_cols=keys))
    reports.append(check_timeliness(df))

    out = pd.concat([r for r in reports if r is not None and not r.empty], ignore_index=True)
    score = (1 - out["pct_rows"].clip(0,1)).mean() if not out.empty else 1.0
    out.attrs["quality_score"] = round(float(score)*100, 2)
    return out
