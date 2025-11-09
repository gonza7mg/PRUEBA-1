# pages/2_Dashboard_.py
import os
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dashboard – DSS CNMC (FINAL)", layout="wide")

# -------------------------------------------------------------------
# Localización de ficheros FINAL
# -------------------------------------------------------------------
FINAL = {
    "Anual – Datos generales": "data/final/anual_datos_generales_final.csv",
    "Anual – Mercados":        "data/final/anual_mercados_final.csv",
    "Mensual":                 "data/final/mensual_final.csv",
    "Provinciales":            "data/final/provinciales_final.csv",
    "Trimestrales":            "data/final/trimestrales_final.csv",
    "Infraestructuras":        "data/final/infraestructuras_final.csv",
}

# -------------------------------------------------------------------
# Utilidades robustas
# -------------------------------------------------------------------
def uniquify_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen, new = {}, []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 0
            new.append(c)
    df.columns = new
    return df

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = uniquify_columns(df)
    df.columns = [c.strip() for c in df.columns]
    # normalizaciones típicas
    ren = {}
    for c in df.columns:
        low = c.lower()
        if low in {"operador", "operadora"}:
            ren[c] = "operador"
        if low in {"mercado", "tipo_de_mercado"}:
            ren[c] = "mercado"
        if low in {"provincia", "prov"}:
            ren[c] = "provincia"
        if low in {"tecnologia", "tecnologías"}:
            ren[c] = "tecnologia"
        if low in {"cuota_pct", "cuota%"}:
            ren[c] = "cuota"
    if ren:
        df = df.rename(columns=ren)
    if "periodo" in df.columns:
        df["periodo"] = pd.to_datetime(df["periodo"], errors="coerce")
    return df

def numeric_cols(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

def cat_cols(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    return [c for c in df.columns if df[c].dtype == "object" and c not in exclude]

def pick_first(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for cand in candidates:
        for c in df.columns:
            if c.lower() == cand:
                return c
    return None

def gauge(value: Optional[float], title: str, suffix: str = "", min_v: float = 0, max_v: float = 100):
    if value is None or np.isfinite(value) is False:
        fig = go.Figure(go.Indicator(mode="number", value=0, number={'suffix': f" {suffix}"},
                                     title={'text': title}))
    else:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': f" {suffix}"},
            title={'text': title},
            gauge={
                'axis': {'range': [min_v, max_v]},
                'bar': {'thickness': 0.25},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "lightgray",
            }
        ))
    fig.update_layout(height=190, margin=dict(l=10, r=10, t=30, b=0))
    return fig

def series_line(df: pd.DataFrame, x: str, y: str, color: Optional[str], title: str):
    if df.empty:
        return None
    fig = px.line(df, x=x, y=y, color=color, markers=True, title=title)
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def bars(df: pd.DataFrame, x: str, y: str, color: Optional[str], title: str, stacked: bool=False):
    if df.empty:
        return None
    fig = px.bar(df, x=x, y=y, color=color, title=title)
    if stacked:
        fig.update_layout(barmode="stack")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def ensure_pct(c: pd.Series) -> pd.Series:
    """Adapta 0-1 a 0-100 si procede."""
    if c.max() <= 1.5:
        return c * 100
    return c

# -------------------------------------------------------------------
# Carga y selección del dataset FINAL
# -------------------------------------------------------------------
existing = {n: p for n, p in FINAL.items() if os.path.exists(p)}
if not existing:
    st.warning("No hay ficheros en data/final/. Ejecuta primero el pipeline FINAL.")
    st.stop()

dfs = {n: load_csv(p) for n, p in existing.items() if load_csv(p) is not None}
names_ok = list(dfs.keys())

st.title("📊 Dashboard – DSS Telecomunicaciones (CNMC / FINAL)")

# Filtros globales
fl, _ = st.columns([1, 4])
with fl:
    dataset = st.selectbox("Fuente", names_ok, index=0)

df = dfs[dataset].copy()
st.caption(f"{dataset}: {len(df):,} filas × {df.shape[1]} columnas")

# -------------------------------------------------------------------
# Detección de métricas/categorías “típicas”
# -------------------------------------------------------------------
col_lineas = pick_first(df, ["lineas_activas", "lineas", "abonados", "suscriptores", "lineas_totales"])
col_ingresos = pick_first(df, ["ingresos", "ingresos_totales", "ingresos_por_operador", "revenue"])
col_cob_5g = pick_first(df, ["cobertura_5g", "cobertura5g", "5g"])
col_operador = "operador" if "operador" in df.columns else None
col_prov = "provincia" if "provincia" in df.columns else None
col_tecn = "tecnologia" if "tecnologia" in df.columns else None

# -------------------------------------------------------------------
# KPI ROW (4 tarjetas con gauge/indicador)
# -------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

# KPI 1: Líneas activas (suma)
try:
    v = float(df[col_lineas].fillna(0).sum()) if col_lineas else None
except Exception:
    v = None
with k1:
    st.plotly_chart(gauge(v if v is not None else 0, "Líneas activas (suma)", "", 0, v if v and v>0 else 100), use_container_width=True)

# KPI 2: Ingresos totales
try:
    ing = float(df[col_ingresos].fillna(0).sum()) if col_ingresos else None
except Exception:
    ing = None
with k2:
    st.plotly_chart(gauge(ing if ing is not None else 0, "Ingresos totales (€)", "", 0, ing if ing and ing>0 else 100), use_container_width=True)

# KPI 3: Cobertura 5G media
try:
    cov = df[col_cob_5g].dropna().astype(float)
    cov = ensure_pct(cov) if not cov.empty else cov
    cov_val = float(cov.mean()) if not cov.empty else None
except Exception:
    cov_val = None
with k3:
    st.plotly_chart(gauge(cov_val if cov_val is not None else 0, "Cobertura 5G media", "%"), use_container_width=True)

# KPI 4: HHI aprox (si hay operador y cuota o ingresos)
hhi = None
try:
    if col_operador and ("cuota" in df.columns or col_ingresos):
        if "cuota" in df.columns:
            shares = df.groupby(col_operador, as_index=False)["cuota"].sum()["cuota"]
            if shares.max() <= 1.5:
                shares = shares
            else:
                shares = shares / 100.0
        else:
            g = df.groupby(col_operador, as_index=False)[col_ingresos].sum()
            if g[col_ingresos].sum() > 0:
                shares = g[col_ingresos] / g[col_ingresos].sum()
            else:
                shares = None
        if shares is not None:
            hhi = float((shares.clip(lower=0) ** 2).sum() * 10_000)
except Exception:
    hhi = None

with k4:
    st.plotly_chart(gauge(hhi if hhi is not None else 0, "HHI (aprox.)", "", 0, 10_000), use_container_width=True)

st.divider()

# -------------------------------------------------------------------
# BLOQUE 2: dos series temporales + barras (estilo “executive”)
# -------------------------------------------------------------------
row1_c1, row1_c2 = st.columns(2)

# Serie temporal 1: si hay periodo y líneas/ingresos
if "periodo" in df.columns and pd.api.types.is_datetime64_any_dtype(df["periodo"]):
    # a) líneas/ingresos en el tiempo (suma por periodo)
    if col_lineas:
        g = df.dropna(subset=["periodo"])[["periodo", col_lineas]].groupby("periodo", as_index=False)[col_lineas].sum()
        fig = series_line(g.sort_values("periodo"), "periodo", col_lineas, None, "Evolución líneas")
        with row1_c1:
            if fig: st.plotly_chart(fig, use_container_width=True)
    elif col_ingresos:
        g = df.dropna(subset=["periodo"])[["periodo", col_ingresos]].groupby("periodo", as_index=False)[col_ingresos].sum()
        fig = series_line(g.sort_values("periodo"), "periodo", col_ingresos, None, "Evolución ingresos")
        with row1_c1:
            if fig: st.plotly_chart(fig, use_container_width=True)

    # b) cuota por operador en el tiempo (si existe)
    if col_operador and "cuota" in df.columns:
        base = df.dropna(subset=["periodo"])[["periodo", "operador", "cuota"]].copy()
        base["cuota"] = ensure_pct(base["cuota"])
        g = base.groupby(["periodo", "operador"], as_index=False)["cuota"].sum()
        fig = series_line(g.sort_values("periodo"), "periodo", "cuota", "operador", "Cuotas (%) por operador")
        with row1_c2:
            if fig: st.plotly_chart(fig, use_container_width=True)
else:
    with row1_c1:
        st.info("No hay columna 'periodo' parseada como fecha.")
    with row1_c2:
        st.empty()

st.divider()

# -------------------------------------------------------------------
# BLOQUE 3: Barras por operador + stacked por último periodo
# -------------------------------------------------------------------
row2_c1, row2_c2 = st.columns(2)

# Top operadores por líneas o ingresos
if col_operador:
    metric = col_lineas or col_ingresos
    if metric:
        g = df.groupby(col_operador, as_index=False)[metric].sum().sort_values(metric, ascending=False).head(10)
        fig = bars(g, x=col_operador, y=metric, color=None, title=f"Top 10 operadores por {metric}")
        with row2_c1:
            if fig: st.plotly_chart(fig, use_container_width=True)
    else:
        with row2_c1:
            st.info("No hay métrica numérica para agrupar por operador.")
else:
    with row2_c1:
        st.info("Este dataset no tiene 'operador'.")

# Stacked por operador en último periodo (si hay periodo)
if col_operador and "periodo" in df.columns and pd.api.types.is_datetime64_any_dtype(df["periodo"]):
    metric = col_lineas or col_ingresos or ("cuota" if "cuota" in df.columns else None)
    if metric:
        dft = df.dropna(subset=["periodo"])
        if not dft.empty:
            last = dft["periodo"].max()
            sub = dft[dft["periodo"] == last]
            if metric == "cuota":
                sub = sub.assign(cuota=ensure_pct(sub["cuota"]))
            g = sub.groupby([col_operador], as_index=False)[metric].sum().sort_values(metric, ascending=False)
            fig = bars(g, x=col_operador, y=metric, color=None, title=f"{metric} por operador – {last.date()}", stacked=False)
            with row2_c2:
                if fig: st.plotly_chart(fig, use_container_width=True)
        else:
            with row2_c2:
                st.info("No hay registros fechados para el stacked del último periodo.")
    else:
        with row2_c2:
            st.info("No hay métrica numérica para el stacked por operador.")
else:
    with row2_c2:
        st.info("No es posible construir el stacked (falta 'operador' o 'periodo').")

st.divider()

# -------------------------------------------------------------------
# BLOQUE 4: Territorial y tecnología
# -------------------------------------------------------------------
row3_c1, row3_c2 = st.columns(2)

# Provincias (Top-N)
if col_prov:
    metric = col_lineas or col_ingresos
    if metric:
        g = df.groupby(col_prov, as_index=False)[metric].sum().sort_values(metric, ascending=False).head(15)
        fig = bars(g, x=col_prov, y=metric, color=None, title=f"Top 15 provincias por {metric}")
        with row3_c1:
            if fig: st.plotly_chart(fig, use_container_width=True)
    else:
        with row3_c1:
            st.info("No hay métrica para agrupar por provincia.")
else:
    with row3_c1:
        st.info("No existe columna 'provincia'.")

# Infraestructura por tecnología (si procede)
if col_tecn:
    metric = pick_first(df, ["nodos", "km_red", "capacidad", "cobertura_5g", "cobertura5g"])
    if metric:
        g = df.groupby(col_tecn, as_index=False)[metric].sum().sort_values(metric, ascending=False)
        fig = bars(g, x=col_tecn, y=metric, color=None, title=f"{metric} por tecnología")
        with row3_c2:
            if fig: st.plotly_chart(fig, use_container_width=True)
    else:
        with row3_c2:
            st.info("No hay métrica de infraestructura detectable para tecnología.")
else:
    with row3_c2:
        st.info("No existe columna 'tecnologia' en este dataset.")
