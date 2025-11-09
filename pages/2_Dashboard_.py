# pages/2_Dashboard_.py
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dashboard – DSS CNMC (FINAL)", layout="wide")

FINAL = {
    "Anual – Datos generales": "data/final/anual_datos_generales_final.csv",
    "Anual – Mercados":        "data/final/anual_mercados_final.csv",
    "Mensual":                 "data/final/mensual_final.csv",
    "Provinciales":            "data/final/provinciales_final.csv",
    "Trimestrales":            "data/final/trimestrales_final.csv",
    "Infraestructuras":        "data/final/infraestructuras_final.csv",
}

# ----------------------- Utils robustas -----------------------
def uniquify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Evita nombres de columna duplicados (causan el error de 'no 1-dimensional')."""
    seen: dict[str, int] = {}
    new = []
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
    # normalización de nombres frecuentes
    ren = {}
    for c in df.columns:
        low = c.lower()
        if low in {"operador", "operadora"}:
            ren[c] = "operador"
        if low in {"mercado", "tipo_de_mercado"}:
            ren[c] = "mercado"
        if low in {"provincia", "prov"}:
            ren[c] = "provincia"
        if low in {"tecnologia", "tecnologías", "technology"}:
            ren[c] = "tecnologia"
        if low in {"cuota_pct", "cuota%"}:
            ren[c] = "cuota"
    if ren:
        df = df.rename(columns=ren)
    # periodo a datetime si existe
    if "periodo" in df.columns:
        df["periodo"] = pd.to_datetime(df["periodo"], errors="coerce")
    return df

def has(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)

def numeric_cols(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

def cat_cols(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    return [c for c in df.columns if df[c].dtype == "object" and c not in exclude]

def safe_line(df, x, y, color=None, title=""):
    if df is not None and not df.empty:
        fig = px.line(df, x=x, y=y, color=color, markers=True, title=title)
        st.plotly_chart(fig, use_container_width=True)

def safe_bar(df, x, y, color=None, title=""):
    if df is not None and not df.empty:
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------- Carga de datos -----------------------
existing = {n: p for n, p in FINAL.items() if os.path.exists(p)}
if not existing:
    st.warning("No hay ficheros en data/final/. Ejecuta el pipeline para generarlos.")
    st.stop()

dfs = {n: load_csv(p) for n, p in existing.items()}
names_ok = [n for n, d in dfs.items() if d is not None and not d.empty]

st.title("Dashboard – DSS Telecomunicaciones (CNMC / FINAL)")

col_sel, _ = st.columns([1, 3])
with col_sel:
    dataset = st.selectbox("Dataset", names_ok, index=0)

df = dfs[dataset]
st.success(f"{dataset}: {len(df):,} filas × {df.shape[1]} columnas")

with st.expander("Vista previa (50 filas)", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

st.divider()

# ----------------------- KPIs sencillos -----------------------
st.subheader("KPIs")
k1, k2, k3, k4 = st.columns(4)

lineas_col = next((c for c in df.columns if c.lower() in
                   {"lineas_activas", "lineas", "abonados", "suscriptores", "lineas_totales"}), None)
ing_col = next((c for c in df.columns if c.lower() in
                {"ingresos", "revenue", "ingresos_totales", "ingresos_por_operador"}), None)
cov_col = next((c for c in df.columns if c.lower() in {"cobertura_5g", "cobertura5g", "5g"}), None)

try:
    k1.metric("Líneas activas", "0" if lineas_col is None else f"{int(df[lineas_col].fillna(0).sum()):,}")
except Exception:
    k1.metric("Líneas activas", "0")

try:
    if ing_col:
        total = df[ing_col].fillna(0).sum()
        k2.metric("Ingresos totales (€)", f"{total:,.0f}" if total >= 1_000_000 else f"{total:,.2f}")
    else:
        k2.metric("Ingresos totales (€)", "0")
except Exception:
    k2.metric("Ingresos totales (€)", "0")

try:
    if cov_col:
        cov = df[cov_col].dropna()
        k3.metric("Cobertura 5G (%)", "NA" if cov.empty else f"{cov.mean():.1f}%")
    else:
        k3.metric("Cobertura 5G (%)", "NA")
except Exception:
    k3.metric("Cobertura 5G (%)", "NA")

# HHI estimado si hay operador y cuotas o una métrica
hhi_val = None
try:
    if has(df, ["operador"]) and ("cuota" in df.columns or ing_col is not None):
        base = df.copy()
        if "cuota" in base.columns:
            tmp = base.groupby("operador", as_index=False)["cuota"].sum()
            if tmp["cuota"].max() <= 1.5:
                shares = tmp["cuota"].clip(lower=0)
            else:
                shares = (tmp["cuota"].clip(lower=0) / 100.0)
            hhi_val = float((shares ** 2).sum() * 10_000)
        elif ing_col is not None:
            g = base.groupby("operador", as_index=False)[ing_col].sum()
            if g[ing_col].sum() > 0:
                shares = g[ing_col] / g[ing_col].sum()
                hhi_val = float((shares ** 2).sum() * 10_000)
except Exception:
    hhi_val = None
k4.metric("HHI (aprox.)", "NA" if hhi_val is None else f"{hhi_val:,.0f}")

st.divider()

# ----------------------- Pestañas -----------------------
tabs = st.tabs(["Temporal", "Categorías", "Competencia", "Territorial", "Infraestructura"])

# -------- Temporal
with tabs[0]:
    st.markdown("### Serie temporal")
    if "periodo" in df.columns and pd.api.types.is_datetime64_any_dtype(df["periodo"]):
        nums = numeric_cols(df, exclude=["periodo"])
        if not nums:
            st.info("No hay métricas numéricas para trazar.")
        else:
            y = st.selectbox("Métrica", nums, key="t_y")
            color = st.selectbox("Desagregar por", ["(ninguna)"] + cat_cols(df), key="t_color")
            base = df.dropna(subset=["periodo"])[["periodo", y] + ([color] if color != "(ninguna)" else [])]
            gcols = ["periodo"] + ([color] if color != "(ninguna)" else [])
            g = base.groupby(gcols, as_index=False)[y].sum()
            safe_line(g.sort_values("periodo"), x="periodo", y=y, color=(color if color != "(ninguna)" else None),
                      title=f"Evolución de {y}")
    else:
        st.info("No hay columna de fecha parseada como 'periodo'.")

# -------- Categorías (Top-N)
with tabs[1]:
    st.markdown("### Top categorías")
    cats = cat_cols(df)
    nums = numeric_cols(df)
    if not cats or not nums:
        st.info("Se necesita al menos 1 categórica y 1 numérica.")
    else:
        c = st.selectbox("Categoría", cats, key="c_cat")
        v = st.selectbox("Métrica", nums, key="c_val")
        topn = st.slider("Top N", 5, 50, 15, key="c_top")
        # agrupar siempre con selección de columnas -> evita el error de Grouper
        base = df[[c, v]].copy()
        g = base.groupby(c, as_index=False)[v].sum().sort_values(v, ascending=False).head(topn)
        safe_bar(g, x=c, y=v, title=f"Top {topn} por {c}")

# -------- Competencia
with tabs[2]:
    st.markdown("### Cuotas por operador y HHI por periodo")
    if "operador" not in df.columns:
        st.info("No existe columna 'operador' en este dataset.")
    else:
        # Cuotas
        if "cuota" in df.columns:
            base = df[["operador", "cuota"] + (["periodo"] if "periodo" in df.columns else [])].copy()
            # normaliza a 0-100 si viene 0-1
            if base["cuota"].max() <= 1.5:
                base["cuota"] = base["cuota"] * 100
            g = base.groupby("operador", as_index=False)["cuota"].sum().sort_values("cuota", ascending=False)
            safe_bar(g, x="operador", y="cuota", title="Cuotas (%)")
        elif ing_col is not None:
            base = df[["operador", ing_col] + (["periodo"] if "periodo" in df.columns else [])].copy()
            g = base.groupby("operador", as_index=False)[ing_col].sum()
            if g[ing_col].sum() > 0:
                g["cuota_est_%"] = g[ing_col] / g[ing_col].sum() * 100
                safe_bar(g.sort_values("cuota_est_%", ascending=False), x="operador", y="cuota_est_%", title="Cuotas estimadas (%)")
            else:
                st.info("No hay datos para estimar cuotas.")
        else:
            st.info("No hay 'cuota' ni una métrica para estimarla.")

        # HHI en el tiempo (si hay periodo)
        if "periodo" in df.columns and pd.api.types.is_datetime64_any_dtype(df["periodo"]):
            if "cuota" in df.columns:
                tmp = df[["periodo", "operador", "cuota"]].dropna(subset=["periodo"]).copy()
                if tmp["cuota"].max() <= 1.5:
                    tmp["cuota"] = tmp["cuota"]
                else:
                    tmp["cuota"] = tmp["cuota"] / 100.0
                g = tmp.groupby(["periodo", "operador"])["cuota"].sum().reset_index()
                hhi = g.groupby("periodo")["cuota"].apply(lambda s: (s**2).sum()*10_000).reset_index(name="HHI")
                safe_line(hhi.sort_values("periodo"), x="periodo", y="HHI", title="HHI por periodo")
            elif ing_col is not None:
                tmp = df[["periodo", "operador", ing_col]].dropna(subset=["periodo"]).copy()
                g = tmp.groupby(["periodo", "operador"])[ing_col].sum().reset_index()
                g["total"] = g.groupby("periodo")[ing_col].transform("sum")
                g = g[g["total"] > 0]
                g["sq"] = (g[ing_col] / g["total"]) ** 2
                hhi = g.groupby("periodo")["sq"].sum().reset_index(name="HHI")
                safe_line(hhi.sort_values("periodo"), x="periodo", y="HHI", title="HHI por periodo")
        else:
            st.caption("No hay columna temporal válida para HHI temporal.")

# -------- Territorial
with tabs[3]:
    st.markdown("### Distribución por provincia")
    if "provincia" not in df.columns:
        st.info("No existe columna 'provincia' en este dataset.")
    else:
        nums = numeric_cols(df)
        if not nums:
            st.info("No hay métricas numéricas para trazar por provincia.")
        else:
            val = st.selectbox("Métrica", nums, key="t_val")
            g = df.groupby("provincia", as_index=False)[val].sum().sort_values(val, ascending=False)
            topn = st.slider("Top N", 10, 52, 20, key="t_top")
            safe_bar(g.head(topn), x="provincia", y=val, title=f"Top {topn} provincias por {val}")
            if "periodo" in df.columns and pd.api.types.is_datetime64_any_dtype(df["periodo"]):
                pick = st.selectbox("Provincia (serie temporal)", sorted(df["provincia"].dropna().unique().tolist()))
                dft = df[df["provincia"] == pick].dropna(subset=["periodo"])
                if not dft.empty:
                    g = dft.groupby("periodo", as_index=False)[val].sum()
                    safe_line(g.sort_values("periodo"), x="periodo", y=val, title=f"Evolución – {pick}")

# -------- Infraestructura
with tabs[4]:
    st.markdown("### Infraestructura por tecnología")
    tech = "tecnologia" if "tecnologia" in df.columns else None
    met = next((c for c in df.columns if c.lower() in
                {"cobertura_5g", "cobertura", "nodos", "km_red", "capacidad", "unidades"}), None)
    if not tech or not met:
        st.info("No se han encontrado columnas de tecnología y métrica de infraestructura.")
    else:
        g = df.groupby(tech, as_index=False)[met].sum().sort_values(met, ascending=False)
        safe_bar(g, x=tech, y=met, title=f"{met} por {tech}")
        if "periodo" in df.columns and pd.api.types.is_datetime64_any_dtype(df["periodo"]):
            pick = st.selectbox("Tecnología (serie temporal)", ["(todas)"] + sorted(df[tech].dropna().unique().tolist()))
            dft = df.dropna(subset=["periodo"])
            if pick != "(todas)":
                dft = dft[dft[tech] == pick]
            g = dft.groupby("periodo", as_index=False)[met].sum()
            safe_line(g.sort_values("periodo"), x="periodo", y=met, title=f"Evolución {met}")
