# pages/3_Dashboard_Simple.py
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard (simple) – DSS CNMC", layout="wide", page_icon="📉")
st.title("📉 Dashboard simple – CNMC (datasets FINAL)")

# ------------------------------
# Rutas (usamos SOLO data/final)
# ------------------------------
FINAL = [
    ("Anual – Datos generales", "data/final/anual_datos_generales_final.csv"),
    ("Anual – Mercados",        "data/final/anual_mercados_final.csv"),
    ("Mensual",                 "data/final/mensual_final.csv"),
    ("Provinciales",            "data/final/provinciales_final.csv"),
    ("Trimestrales",            "data/final/trimestrales_final.csv"),
    ("Infraestructuras",        "data/final/infraestructuras_final.csv"),
]

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # intenta parsear "periodo" si existe
    if "periodo" in df.columns:
        df["periodo"] = pd.to_datetime(df["periodo"], errors="coerce")
    return df

# ------------------------------
# Selección de dataset y vista
# ------------------------------
existing = [(name, p) for name, p in FINAL if os.path.exists(p)]
if not existing:
    st.warning("No se han encontrado CSV en data/final/. Ejecuta el pipeline de limpieza final primero.")
    st.stop()

name = st.sidebar.selectbox("Dataset FINAL", [n for n, _ in existing])
path = dict(existing)[name]
df = load_csv(path)

if df is None or df.empty:
    st.warning("El CSV está vacío o no pudo cargarse.")
    st.stop()

st.success(f"{name}: {len(df):,} filas × {df.shape[1]} columnas")
with st.expander("Vista previa (primeras 50 filas)", expanded=True):
    st.dataframe(df.head(50), use_container_width=True)

# ------------------------------
# Panel de KPIs muy básicos
# ------------------------------
st.subheader("KPIs rápidos")
c1, c2, c3, c4 = st.columns(4)
n_rows = len(df)
n_cols = df.shape[1]
nulls_total = int(df.isna().sum().sum())
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
c1.metric("Filas", f"{n_rows:,}")
c2.metric("Columnas", f"{n_cols}")
c3.metric("Total nulos", f"{nulls_total:,}")
c4.metric("Numéricas", f"{len(num_cols)}")

st.divider()

# =========================================================
# BLOQUE 1 — Serie temporal (si hay columna fecha/periodo)
# =========================================================
st.subheader("1) Serie temporal (si hay fecha)")

# Detecta columna temporal: periodo o cualquier datetime
date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
if not date_cols:
    # intenta parsear columnas que "parezcan" fechas
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["periodo", "fecha", "date", "anio", "año"])]
    parsed = []
    for c in candidates:
        try:
            tmp = pd.to_datetime(df[c], errors="coerce")
            if tmp.notna().any():
                df[c] = tmp
                parsed.append(c)
        except Exception:
            pass
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

if date_cols:
    time_col = st.selectbox("Columna temporal", date_cols, index=0, key="time_col")
    # numéricas agregables
    num_cols_safe = [c for c in num_cols if c != time_col]
    if num_cols_safe:
        y_col = st.selectbox("Métrica (y)", num_cols_safe, key="y_time")
        # Optional desagregación por categoría
        cat_cols = [c for c in df.columns if df[c].dtype == "object"]
        group_col = st.selectbox("Desagregar (opcional)", ["(sin desagregar)"] + cat_cols, key="grp_time")
        dfi = df[[time_col, y_col] + ([group_col] if group_col != "(sin desagregar)" else [])].copy()
        dfi = dfi.dropna(subset=[time_col])
        agg = dfi.groupby(([time_col, group_col] if group_col != "(sin desagregar)" else [time_col]), as_index=False)[y_col].sum()
        fig = px.line(agg, x=time_col, y=y_col, color=(group_col if group_col != "(sin desagregar)" else None), markers=True,
                      title=f"Evolución temporal – {y_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay columnas numéricas para graficar en el tiempo.")
else:
    st.info("No se encontró ninguna columna de fecha/periodo. Este bloque se omite.")

st.divider()

# =========================================================
# BLOQUE 2 — Barras por categoría (top N)
# =========================================================
st.subheader("2) Barras por categoría")

cat_cols = [c for c in df.columns if df[c].dtype == "object"]
if cat_cols and num_cols:
    cat = st.selectbox("Categoría", cat_cols, key="cat_bar")
    val = st.selectbox("Métrica (y)", num_cols, key="val_bar")
    top_n = st.slider("Top N", 5, 50, 20, key="topn_bar")
    g = df.groupby(cat, as_index=False)[val].sum().sort_values(val, ascending=False).head(top_n)
    fig = px.bar(g, x=cat, y=val, title=f"Top {top_n} por {cat}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Se necesitan al menos 1 columna categórica y 1 numérica.")

st.divider()

# =========================================================
# BLOQUE 3 — Dispersión entre numéricas
# =========================================================
st.subheader("3) Dispersión entre variables numéricas")

if len(num_cols) >= 2:
    x = st.selectbox("Eje X", num_cols, key="x_scatter")
    y = st.selectbox("Eje Y", [c for c in num_cols if c != x], key="y_scatter")
    color = st.selectbox("Color (opcional)", ["(sin color)"] + cat_cols, key="col_scatter")
    df_plot = df[[x, y] + ([color] if color != "(sin color)" else [])].copy()
    fig = px.scatter(df_plot, x=x, y=y, color=(color if color != "(sin color)" else None), trendline="ols",
                     title=f"Dispersión {x} vs {y}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Se necesitan al menos 2 columnas numéricas para el scatter.")
