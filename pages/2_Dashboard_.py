# pages/2_Dashboard_.py
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(page_title="Dashboard – DSS CNMC", layout="wide", page_icon="📊")
st.title("📊 Dashboard – DSS Telecomunicaciones (CNMC / FINAL)")

FINAL = {
    "Anual – Datos generales": "data/final/anual_datos_generales_final.csv",
    "Anual – Mercados":        "data/final/anual_mercados_final.csv",
    "Mensual":                 "data/final/mensual_final.csv",
    "Provinciales":            "data/final/provinciales_final.csv",
    "Trimestrales":            "data/final/trimestrales_final.csv",
    "Infraestructuras":        "data/final/infraestructuras_final.csv",
}

# ------------------------------------------------------------
# Utils robustas
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "periodo" in df.columns:
        df["periodo"] = pd.to_datetime(df["periodo"], errors="coerce")
    # Normaliza nombres típicos si existen en alguna variante
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {"operador", "operadora"}:
            rename[c] = "operador"
        if cl in {"mercado", "tipo_de_mercado"}:
            rename[c] = "mercado"
        if cl in {"provincia", "prov"}:
            rename[c] = "provincia"
        if cl in {"tecnologia", "tecnologías", "technology"}:
            rename[c] = "tecnologia"
    if rename:
        df = df.rename(columns=rename)
    return df

def has(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)

def first(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def numeric_cols(df: pd.DataFrame, exclude: Iterable[str]=()) -> list[str]:
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

def cat_cols(df: pd.DataFrame, exclude: Iterable[str]=()) -> list[str]:
    return [c for c in df.columns
            if df[c].dtype == "object" and c not in exclude]

def kpi_metric(label: str, value, help_txt: str | None = None):
    col = st.container()
    if help_txt:
        with col:
            st.caption(help_txt)
    st.metric(label, "NA" if value is None else value)

def safe_plot(fig):
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Carga de datasets final
# ------------------------------------------------------------
existing = {name: p for name, p in FINAL.items() if os.path.exists(p)}
if not existing:
    st.warning("No se han encontrado CSV en **data/final/**. Ejecuta el pipeline de limpieza final.")
    st.stop()

# Carga todos (para KPIs globales) y deja selector principal
dfs = {name: load_csv(path) for name, path in existing.items()}
names_ok = [n for n,d in dfs.items() if d is not None and not d.empty]

left, right = st.columns([1,2])
with left:
    dataset = st.selectbox("Dataset principal", names_ok, index=0)
df = dfs[dataset]

st.success(f"{dataset}: {len(df):,} filas × {df.shape[1]} columnas")
with st.expander("Vista previa (primeras 50 filas)", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

st.divider()

# ------------------------------------------------------------
# KPIs principales (robustos)
# ------------------------------------------------------------
st.subheader("KPIs principales")
k1, k2, k3, k4 = st.columns(4)

# KPI 1: líneas activas (si existe una métrica compatible)
lineas_col = first(df, ["lineas_activas", "lineas", "abonados", "suscriptores", "lineas_totales"])
lineas_val = None
if lineas_col is not None:
    try:
        lineas_val = f"{int(df[lineas_col].fillna(0).sum()):,}"
    except Exception:
        lineas_val = None
k1.metric("Líneas activas", "0" if lineas_val is None else lineas_val)

# KPI 2: ingresos totales
ing_col = first(df, ["ingresos", "ingreso", "revenue", "ingresos_por_operador", "ingresos_totales"])
ing_val = None
if ing_col is not None:
    try:
        ing_val = df[ing_col].fillna(0).sum()
        # si parece millones -> no tocar; si es muy grande, formatea
        if ing_val >= 1_000_000:
            ing_val_fmt = f"{ing_val:,.0f}"
        else:
            ing_val_fmt = f"{ing_val:,.2f}"
    except Exception:
        ing_val_fmt = None
else:
    ing_val_fmt = None
k2.metric("Ingresos totales (€)", ing_val_fmt if ing_val_fmt is not None else "0")

# KPI 3: cobertura 5G (si hay %)
cov_col = first(df, ["cobertura_5g", "cobertura5g", "5g"])
cov_val = None
if cov_col is not None:
    try:
        cov_val = df[cov_col].dropna()
        cov_val = None if cov_val.empty else f"{cov_val.mean():.1f}%"
    except Exception:
        cov_val = None
k3.metric("Cobertura 5G (%)", cov_val if cov_val else "NA")

# KPI 4: HHI (si se puede calcular)
# Necesitamos cuota o podemos estimar por ingresos repartidos por operador en un periodo reciente
hhi_val = None
if has(df, ["operador"]) and ("cuota" in df.columns or ing_col is not None):
    try:
        if "cuota" in df.columns:
            # cuota en %, 0–100 o 0–1
            latest = df
            if "periodo" in df.columns:
                latest = df[df["periodo"] == df["periodo"].max()]
            s = latest.groupby("operador")["cuota"].sum()
            if s.max() <= 1.5:  # está en 0–1
                shares = s.clip(lower=0).fillna(0)
            else:
                shares = (s.clip(lower=0).fillna(0) / 100.0)
            hhi_val = round(float((shares ** 2).sum() * 10_000), 0)
        elif ing_col is not None:
            latest = df
            if "periodo" in df.columns:
                latest = df[df["periodo"] == df["periodo"].max()]
            g = latest.groupby("operador")[ing_col].sum()
            if g.sum() > 0:
                shares = g / g.sum()
                hhi_val = round(float((shares ** 2).sum() * 10_000), 0)
    except Exception:
        hhi_val = None
k4.metric("HHI medio", "NA" if hhi_val is None else f"{hhi_val:,.0f}")

st.divider()

# ------------------------------------------------------------
# TABS – cada bloque se autocontrola
# ------------------------------------------------------------
tabs = st.tabs(["Panorama general", "Mercado y competencia", "Infraestructura", "Territorial", "Indicadores cruzados"])

# ------------- TAB 1: Panorama general ----------------------
with tabs[0]:
    st.markdown("#### Evolución temporal")
    # Detecta columna temporal
    time_col = "periodo" if "periodo" in df.columns and pd.api.types.is_datetime64_any_dtype(df["periodo"]) else None
    if time_col is None:
        st.info("No hay columna temporal parseada como fecha. Este bloque se omite.")
    else:
        # Métricas numéricas disponibles
        nums = numeric_cols(df, exclude=[time_col])
        if not nums:
            st.info("No hay columnas numéricas que graficar.")
        else:
            y_col = st.selectbox("Métrica (y)", nums, key="t1_y")
            color_col = st.selectbox("Desagregar por (opcional)", ["(sin desagregar)"] + cat_cols(df), key="t1_col")
            group = color_col if color_col != "(sin desagregar)" else None
            agg = df[[time_col, y_col] + ([group] if group else [])].dropna(subset=[time_col]).copy()
            gcols = [time_col] + ([group] if group else [])
            g = agg.groupby(gcols, as_index=False)[y_col].sum()
            fig = px.line(g, x=time_col, y=y_col, color=group, markers=True, title=f"Evolución temporal – {y_col}")
            safe_plot(fig)

    st.markdown("#### Top categorías")
    cats = cat_cols(df)
    nums = numeric_cols(df)
    if not cats or not nums:
        st.info("Se necesita al menos 1 categórica y 1 numérica.")
    else:
        c = st.selectbox("Categoría", cats, key="t1_cat")
        v = st.selectbox("Métrica", nums, key="t1_val")
        topn = st.slider("Top N", 5, 50, 15, key="t1_topn")
        g = df.groupby(c, as_index=False)[v].sum().sort_values(v, ascending=False).head(topn)
        fig = px.bar(g, x=c, y=v, title=f"Top {topn} por {c}")
        safe_plot(fig)

# ------------- TAB 2: Mercado y competencia -----------------
with tabs[1]:
    st.markdown("#### Cuotas por operador / mercado")
    # Intento 1: si ya hay 'cuota'
    if has(df, ["operador"]) and ("cuota" in df.columns):
        base = df
        # filtros opcionales
        filt_cols = []
        if "mercado" in df.columns:
            mercados = ["(todos)"] + sorted(df["mercado"].dropna().unique().tolist())
            pick_m = st.selectbox("Mercado", mercados, key="t2_m")
            if pick_m != "(todos)":
                base = base[base["mercado"] == pick_m]
                filt_cols.append(("mercado", pick_m))
        if "periodo" in df.columns and pd.api.types.is_datetime64_any_dtype(df["periodo"]):
            periods = base["periodo"].dropna().sort_values().unique()
            pick_p = st.selectbox("Periodo", ["(todos)"] + [pd.to_datetime(p).date() for p in periods], key="t2_p")
            if pick_p != "(todos)":
                base = base[base["periodo"].dt.date == pick_p]
                filt_cols.append(("periodo", str(pick_p)))
        g = base.groupby("operador", as_index=False)["cuota"].sum().sort_values("cuota", ascending=False)
        # normaliza a % si viene en [0,1]
        if g["cuota"].max() <= 1.5:
            g["cuota"] = g["cuota"] * 100
        fig = px.bar(g, x="operador", y="cuota", title="Cuotas de mercado (%)")
        safe_plot(fig)
    # Intento 2: computar cuotas a partir de ingresos (o métrica)
    elif has(df, ["operador"]):
        nums = numeric_cols(df)
        if not nums:
            st.info("No hay 'cuota' ni métricas numéricas para estimarla.")
        else:
            metric = st.selectbox("Métrica para cuotas (suma por operador)", nums, key="t2_val")
            base = df[[metric, "operador"] + (["mercado"] if "mercado" in df.columns else []) + (["periodo"] if "periodo" in df.columns else [])].copy()
            if "mercado" in base.columns:
                mercados = ["(todos)"] + sorted(base["mercado"].dropna().unique().tolist())
                pick_m = st.selectbox("Mercado", mercados, key="t2m2")
                if pick_m != "(todos)":
                    base = base[base["mercado"] == pick_m]
            if "periodo" in base.columns and pd.api.types.is_datetime64_any_dtype(base["periodo"]):
                periods = base["periodo"].dropna().sort_values().unique()
                pick_p = st.selectbox("Periodo", ["(todos)"] + [pd.to_datetime(p).date() for p in periods], key="t2p2")
                if pick_p != "(todos)":
                    base = base[base["periodo"].dt.date == pick_p]
            g = base.groupby("operador", as_index=False)[metric].sum()
            if g[metric].sum() > 0:
                g["cuota_estimada_%"] = g[metric] / g[metric].sum() * 100
                fig = px.bar(g.sort_values("cuota_estimada_%", ascending=False), x="operador", y="cuota_estimada_%", title="Cuotas estimadas (%)")
                safe_plot(fig)
            else:
                st.info("No hay datos suficientes para estimar cuotas.")
    else:
        st.info("Para este bloque se necesita columna **operador**.")

    st.markdown("#### HHI por periodo (si procede)")
    if has(df, ["operador"]) and ("cuota" in df.columns or ing_col is not None):
        base = df.copy()
        if "periodo" in base.columns and pd.api.types.is_datetime64_any_dtype(base["periodo"]):
            if "cuota" in base.columns:
                tmp = base[["periodo", "operador", "cuota"]].dropna(subset=["periodo"])
                # normaliza
                if tmp["cuota"].max() <= 1.5:
                    tmp["cuota"] = tmp["cuota"].clip(lower=0)
                else:
                    tmp["cuota"] = (tmp["cuota"].clip(lower=0) / 100.0)
                h = tmp.groupby(["periodo", "operador"])["cuota"].sum().reset_index()
                h = h.groupby("periodo")["cuota"].apply(lambda s:(s**2).sum()*10_000).reset_index(name="HHI")
            else:
                val = first(df, ["ingresos", "revenue", "ingresos_totales", "ingresos_por_operador"])
                if val is not None:
                    tmp = base[["periodo", "operador", val]].dropna(subset=["periodo"])
                    g = tmp.groupby(["periodo", "operador"])[val].sum().reset_index()
                    g_tot = g.groupby("periodo")[val].transform("sum")
                    shares = g[val] / g_tot.replace(0, np.nan)
                    g["sq"] = (shares ** 2)
                    h = g.groupby("periodo")["sq"].sum().reset_index(name="HHI")
                else:
                    h = None
            if h is not None and not h.empty:
                fig = px.line(h.sort_values("periodo"), x="periodo", y="HHI", markers=True, title="HHI por periodo")
                safe_plot(fig)
            else:
                st.info("No se pudo calcular el HHI con los datos disponibles.")
        else:
            st.info("No hay columna temporal válida para trazar HHI en el tiempo.")
    else:
        st.info("No hay 'operador' y 'cuota' / métrica para calcular HHI.")

# ------------- TAB 3: Infraestructura -----------------------
with tabs[2]:
    st.markdown("#### Cobertura / infraestructura por tecnología")
    # buscamos columnas típicas
    tech = first(df, ["tecnologia", "tecnologías"])
    metric = first(df, ["cobertura_5g", "cobertura", "nodos", "km_red", "capacidad", "unidades"])
    if tech and metric:
        base = df[[tech, metric] + (["periodo"] if "periodo" in df.columns else [])].copy()
        agg = base.groupby(tech, as_index=False)[metric].sum().sort_values(metric, ascending=False)
        fig = px.bar(agg, x=tech, y=metric, title=f"{metric} por {tech}")
        safe_plot(fig)
        if "periodo" in base.columns and pd.api.types.is_datetime64_any_dtype(base["periodo"]):
            pick_t = st.selectbox("Tecnología (serie temporal)", ["(todas)"] + sorted(base[tech].dropna().unique().tolist()), key="t3_tech")
            dft = base.dropna(subset=["periodo"]).copy()
            if pick_t != "(todas)":
                dft = dft[dft[tech] == pick_t]
            g = dft.groupby("periodo", as_index=False)[metric].sum()
            fig = px.line(g.sort_values("periodo"), x="periodo", y=metric, markers=True, title=f"Evolución {metric}")
            safe_plot(fig)
    else:
        st.info("No se encuentran columnas típicas de infraestructura (p. ej. 'tecnologia' y alguna métrica).")

# ------------- TAB 4: Territorial ---------------------------
with tabs[3]:
    st.markdown("#### Distribución por provincia")
    prov = first(df, ["provincia"])
    if prov is None:
        st.info("No hay columna **provincia** en este dataset.")
    else:
        nums = numeric_cols(df)
        if not nums:
            st.info("No hay columnas numéricas para graficar por provincia.")
        else:
            val = st.selectbox("Métrica territorial", nums, key="t4_val")
            g = df.groupby(prov, as_index=False)[val].sum().sort_values(val, ascending=False)
            topn = st.slider("Top N", 10, 52, 20, key="t4_top")
            fig = px.bar(g.head(topn), x=prov, y=val, title=f"Top {topn} provincias por {val}")
            safe_plot(fig)
            if "periodo" in df.columns and pd.api.types.is_datetime64_any_dtype(df["periodo"]):
                st.markdown("##### Evolución por provincia (selección)")
                pick_p = st.selectbox("Provincia", sorted(df[prov].dropna().unique().tolist()), key="t4_prov")
                dft = df[df[prov] == pick_p].dropna(subset=["periodo"])
                if not dft.empty:
                    g = dft.groupby("periodo", as_index=False)[val].sum()
                    fig = px.line(g.sort_values("periodo"), x="periodo", y=val, markers=True, title=f"Evolución – {pick_p}")
                    safe_plot(fig)

# ------------- TAB 5: Indicadores cruzados ------------------
with tabs[4]:
    st.markdown("#### Dispersión entre métricas")
    nums = numeric_cols(df)
    if len(nums) < 2:
        st.info("Se necesitan al menos 2 columnas numéricas para hacer scatter.")
    else:
        x = st.selectbox("X", nums, key="t5_x")
        y = st.selectbox("Y", [c for c in nums if c != x], key="t5_y")
        color = st.selectbox("Color (opcional)", ["(sin color)"] + cat_cols(df), key="t5_c")
        d = df[[x, y] + ([color] if color != "(sin color)" else [])].copy()
        fig = px.scatter(d, x=x, y=y, color=(color if color != "(sin color)" else None),
                         trendline="ols", title=f"{x} vs {y}")
        safe_plot(fig)

    st.markdown("#### Tabla dinámica (pivot)")
    cats = cat_cols(df)
    nums = numeric_cols(df)
    if len(cats) >= 1 and nums:
        r = st.selectbox("Filas", cats, key="t5_r")
        c = st.selectbox("Columnas (opcional)", ["(ninguna)"] + [x for x in cats if x != r], key="t5_c2")
        v = st.selectbox("Valor (suma)", nums, key="t5_v")
        tmp = df[[r] + ([c] if c != "(ninguna)" else []) + [v]].copy()
        if c == "(ninguna)":
            pvt = tmp.groupby(r, as_index=False)[v].sum().sort_values(v, ascending=False)
        else:
            pvt = pd.pivot_table(tmp, index=r, columns=c, values=v, aggfunc="sum", fill_value=0)
        st.dataframe(pvt, use_container_width=True)
    else:
        st.info("Para la tabla dinámica se necesita al menos 1 categórica y 1 numérica.")
