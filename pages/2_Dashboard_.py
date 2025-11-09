# pages/3_Dashboard.py
import os
import io
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Dashboard – DSS CNMC", layout="wide", page_icon="📊")
st.title("📊 Dashboard – DSS Telecomunicaciones (CNMC / FINAL)")

# -----------------------------
# Rutas fijas a los FINAL CSVs
# -----------------------------
FINAL = {
    "anual_datos_generales": "data/final/anual_datos_generales_final.csv",
    "anual_mercados":        "data/final/anual_mercados_final.csv",
    "mensual":               "data/final/mensual_final.csv",
    "provinciales":          "data/final/provinciales_final.csv",
    "trimestrales":          "data/final/trimestrales_final.csv",
    "infraestructuras":      "data/final/infraestructuras_final.csv",
}

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # asegurar 'periodo' datetime si existe
    if "periodo" in df.columns:
        try:
            df["periodo"] = pd.to_datetime(df["periodo"], errors="coerce")
        except Exception:
            pass
    df.columns = [c.strip() for c in df.columns]
    return df

def available_cols(df, substrs, numeric=False):
    cols = []
    for c in df.columns:
        if all(s.lower() in c.lower() for s in substrs):
            if numeric:
                if pd.api.types.is_numeric_dtype(df[c]):
                    cols.append(c)
            else:
                cols.append(c)
    return cols

def pick_first(df, candidates, numeric=False):
    for pat in candidates:
        cols = available_cols(df, [pat], numeric=numeric)
        if cols:
            return cols[0]
    return None

def safe_sum(df, col):
    try:
        return float(pd.to_numeric(df[col], errors="coerce").sum())
    except Exception:
        return np.nan

def hhi_from_shares(share_series):
    """HHI asumiendo share en 0-100 o 0-1."""
    s = pd.to_numeric(share_series, errors="coerce").dropna()
    if s.max() <= 1.5:
        s = s * 100.0
    return float((s**2).sum())

# -----------------------------------
# Carga de todos los datasets FINAL
# -----------------------------------
dfs = {k: load_csv(p) for k,p in FINAL.items()}

missing = [k for k,v in dfs.items() if v is None or v.empty]
if missing:
    st.warning(f"Faltan datasets FINAL o están vacíos: {', '.join(missing)}. El dashboard mostrará lo disponible.")

# -----------------------------------
# Filtros globales (sidebar)
# -----------------------------------
st.sidebar.header("Filtros globales")

# Construir lista de series de 'periodo' válidas (datetime y con algún valor no-NaT)
all_periods = []
for df in dfs.values():
    if isinstance(df, pd.DataFrame) and "periodo" in df.columns:
        s = pd.to_datetime(df["periodo"], errors="coerce")
        if s.notna().any():
            all_periods.append(s)

# Calcular pmin/pmax de forma segura
if all_periods:
    mins = [s.min() for s in all_periods if s.notna().any()]
    maxs = [s.max() for s in all_periods if s.notna().any()]
    if mins and maxs:
        pmin = min(mins)
        pmax = max(maxs)
    else:
        pmin = None
        pmax = None
else:
    pmin = None
    pmax = None

# Slider de años (solo si tenemos rango válido)
if pmin is not None and pmax is not None:
    year_min = int(pmin.year)
    year_max = int(pmax.year)
    default_low = max(year_min, year_max - 5)
    if default_low > year_max:
        default_low = year_min
    year_range = st.sidebar.slider(
        "Rango de años",
        min_value=year_min,
        max_value=year_max,
        value=(default_low, year_max),
    )
else:
    year_range = None
    st.sidebar.caption("No se detectó una columna de **periodo** con valores válidos en los datasets cargados.")

# Listas maestras de dimensiones comunes
operadores, servicios, provincias, tecnologias = set(), set(), set(), set()

for df in dfs.values():
    if df is None:
        continue
    for cname, holder in [("operador", operadores), ("servicio", servicios), ("provincia", provincias), ("tecnologia", tecnologias)]:
        if cname in df.columns:
            holder.update(df[cname].dropna().astype(str).unique().tolist())

oper_sel = st.sidebar.multiselect("Operadores", sorted(list(operadores)) if operadores else [])
serv_sel = st.sidebar.multiselect("Servicios", sorted(list(servicios)) if servicios else [])
prov_sel = st.sidebar.multiselect("Provincias", sorted(list(provincias)) if provincias else [])
tec_sel  = st.sidebar.multiselect("Tecnologías", sorted(list(tecnologias)) if tecnologias else [])

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    out = df.copy()
    if year_range and "periodo" in out.columns and pd.api.types.is_datetime64_any_dtype(out["periodo"]):
        out = out[(out["periodo"].dt.year >= year_range[0]) & (out["periodo"].dt.year <= year_range[1])]
    if oper_sel and "operador" in out.columns:
        out = out[out["operador"].astype(str).isin(oper_sel)]
    if serv_sel and "servicio" in out.columns:
        out = out[out["servicio"].astype(str).isin(serv_sel)]
    if prov_sel and "provincia" in out.columns:
        out = out[out["provincia"].astype(str).isin(prov_sel)]
    if tec_sel and "tecnologia" in out.columns:
        out = out[out["tecnologia"].astype(str).isin(tec_sel)]
    return out

dfs_f = {k: apply_filters(v) for k,v in dfs.items()}

# -----------------------------------
# KPIs principales
# -----------------------------------
st.subheader("KPIs principales")
c1, c2, c3, c4 = st.columns(4)

# KPI 1 – Líneas activas (mensual)
if dfs_f["mensual"] is not None:
    dfm = dfs_f["mensual"]
    col = pick_first(dfm, ["linea","línea","abonados"], numeric=True) or pick_first(dfm, ["valor","total"], numeric=True)
    if col:
        total_lines = safe_sum(dfm, col)
        c1.metric("Líneas activas", f"{total_lines:,.0f}".replace(",", "."))
    else:
        c1.metric("Líneas activas", "NA")
else:
    c1.metric("Líneas activas", "NA")

# KPI 2 – Ingresos (trimestral)
if dfs_f["trimestrales"] is not None:
    dft = dfs_f["trimestrales"]
    col = pick_first(dft, ["ingres"], numeric=True)
    if col:
        total_rev = safe_sum(dft, col)
        c2.metric("Ingresos totales (€)", f"{total_rev:,.0f}".replace(",", "."))
    else:
        c2.metric("Ingresos", "NA")
else:
    c2.metric("Ingresos", "NA")

# KPI 3 – Cobertura 5G
if dfs_f["infraestructuras"] is not None:
    dfi = dfs_f["infraestructuras"]
    col = pick_first(dfi, ["5g","cov"], numeric=True)
    if col:
        cov = pd.to_numeric(dfi[col], errors="coerce").mean()
        c3.metric("Cobertura 5G (%)", f"{cov:.1f}")
    else:
        c3.metric("Cobertura 5G", "NA")
else:
    c3.metric("Cobertura 5G", "NA")

# KPI 4 – HHI (mercados)
if dfs_f["anual_mercados"] is not None:
    dfm = dfs_f["anual_mercados"]
    col = pick_first(dfm, ["cuota","share"], numeric=True)
    if col and "mercado" in dfm.columns:
        sub = dfm.copy()
        if "periodo" in sub.columns:
            sub = sub[sub["periodo"] == sub["periodo"].max()]
        hhi = sub.groupby("mercado")[col].apply(hhi_from_shares).mean()
        c4.metric("HHI medio", f"{hhi:,.0f}".replace(",", "."))
    else:
        c4.metric("HHI", "NA")
else:
    c4.metric("HHI", "NA")

# -----------------------------------
# Tabs
# -----------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Panorama general",
    "Mercado y competencia",
    "Infraestructura",
    "Territorial",
    "Indicadores cruzados",
])

# ---- TAB 1 ----
with tab1:
    st.subheader("Evolución temporal")
    dfm = dfs_f["mensual"]
    if dfm is not None and "periodo" in dfm.columns:
        ycol = pick_first(dfm, ["linea","línea","abonados","valor"], numeric=True)
        if ycol:
            g = dfm.groupby("periodo", as_index=False)[ycol].sum()
            st.plotly_chart(px.line(g, x="periodo", y=ycol, title="Evolución mensual"), use_container_width=True)
    else:
        st.info("No hay datos mensuales para graficar.")

# ---- TAB 2 ----
with tab2:
    st.subheader("Cuotas y concentración")
    dfm = dfs_f["anual_mercados"]
    if dfm is not None and all(c in dfm.columns for c in ["mercado","operador"]):
        cuota_col = pick_first(dfm, ["cuota","share"], numeric=True)
        mercados = sorted(dfm["mercado"].dropna().unique())
        merc_sel = st.selectbox("Mercado", mercados)
        sub = dfm[dfm["mercado"] == merc_sel]
        if "periodo" in sub.columns:
            lastp = sub["periodo"].max()
            last = sub[sub["periodo"] == lastp]
            fig = px.pie(last, names="operador", values=cuota_col, title=f"Cuota de mercado – {merc_sel}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos de mercados anuales.")

# ---- TAB 3 ----
with tab3:
    st.subheader("Infraestructura y despliegue")
    dfi = dfs_f["infraestructuras"]
    if dfi is not None:
        col = pick_first(dfi, ["5g","cov","km","nodo"], numeric=True)
        if col:
            g = dfi.groupby("periodo", as_index=False)[col].sum()
            st.plotly_chart(px.line(g, x="periodo", y=col, title=f"Evolución {col}"), use_container_width=True)
    else:
        st.info("No hay datos de infraestructura.")

# ---- TAB 4 ----
with tab4:
    st.subheader("Distribución territorial")
    dfp = dfs_f["provinciales"]
    if dfp is not None and "provincia" in dfp.columns:
        col = pick_first(dfp, ["linea","valor","ingres"], numeric=True)
        if col:
            g = dfp.groupby("provincia", as_index=False)[col].mean().sort_values(col, ascending=False)
            st.plotly_chart(px.bar(g.head(20), x="provincia", y=col, title="Top 20 provincias"), use_container_width=True)
    else:
        st.info("No hay datos provinciales para mostrar.")

# ---- TAB 5 ----
with tab5:
    st.subheader("Relaciones entre variables (cross-dataset)")
    dft, dfi = dfs_f["trimestrales"], dfs_f["infraestructuras"]
    if dft is not None and dfi is not None:
        colx = pick_first(dft, ["ingres"], numeric=True)
        coly = pick_first(dfi, ["5g","cov","nodo"], numeric=True)
        if colx and coly:
            gt = dft.groupby("periodo", as_index=False)[colx].sum()
            gi = dfi.groupby("periodo", as_index=False)[coly].sum()
            merged = pd.merge(gt, gi, on="periodo", how="inner")
            st.plotly_chart(px.scatter(merged, x=colx, y=coly, trendline="ols", title="Ingresos vs Infraestructura"),
                            use_container_width=True)
    else:
        st.info("No hay datos para comparar.")
