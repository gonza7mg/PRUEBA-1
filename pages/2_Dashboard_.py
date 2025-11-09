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
    # normalizar columnas a lower snake básico para búsqueda flexible
    # pero conservamos originales para display
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

def pick_many(df, candidates):
    out=[]
    for pat in candidates:
        cols = available_cols(df, [pat], numeric=False)
        out += cols
    # unique preserving order
    seen=set(); res=[]
    for c in out:
        if c not in seen:
            seen.add(c); res.append(c)
    return res

def safe_sum(df, col):
    try:
        return float(pd.to_numeric(df[col], errors="coerce").sum())
    except Exception:
        return np.nan

def yoy_growth(series: pd.Series):
    """Calcula crecimiento interanual por índice temporal (freq anual/mensual/trimestral)."""
    s = pd.to_numeric(series, errors="coerce")
    return (s - s.shift(12)) / s.shift(12) * 100

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

# Determinar conjunto de periodos disponible (mínimo común denominador)
all_periods = []
for df in dfs.values():
    if isinstance(df, pd.DataFrame) and "periodo" in df.columns and pd.api.types.is_datetime64_any_dtype(df["periodo"]):
        all_periods.append(df["periodo"])
if all_periods:
    pmin = min([s.min() for s in all_periods if s.notna().any()])
    pmax = max([s.max() for s in all_periods if s.notna().any()])
else:
    pmin = None; pmax = None

if pmin is not None and pmax is not None:
    year_min = int(pmin.year); year_max = int(pmax.year)
    year_range = st.sidebar.slider("Rango de años", min_value=year_min, max_value=year_max,
                                   value=(max(year_min, year_max-5), year_max))
else:
    year_range = None

# Listas maestras de dimensiones comunes
operadores = set()
servicios = set()
provincias = set()
tecnologias = set()

for name, df in dfs.items():
    if df is None: 
        continue
    for cset, holder in [
        (["operador"], operadores),
        (["servicio"], servicios),
        (["provincia"], provincias),
        (["tecnologia","tecnología"], tecnologias),
    ]:
        for c in cset:
            if c in df.columns:
                holder.update(df[c].dropna().astype(str).unique().tolist())

oper_sel = st.sidebar.multiselect("Operadores", sorted(list(operadores)) if operadores else [])
serv_sel = st.sidebar.multiselect("Servicios", sorted(list(servicios)) if servicios else [])
prov_sel = st.sidebar.multiselect("Provincias", sorted(list(provincias)) if provincias else [])
tec_sel  = st.sidebar.multiselect("Tecnologías", sorted(list(tecnologias)) if tecnologias else [])

def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
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
    if tec_sel and ("tecnologia" in out.columns or "tecnología" in out.columns):
        tcol = "tecnologia" if "tecnologia" in out.columns else "tecnología"
        out = out[out[tcol].astype(str).isin(tec_sel)]
    return out

dfs_f = {k: apply_global_filters(v) for k,v in dfs.items()}

# -----------------------------------
# KPIs principales (Top)
# -----------------------------------
st.subheader("KPIs principales")

c1, c2, c3, c4 = st.columns(4)

# KPI 1: Líneas activas (busca columna por patrones)
if dfs_f["mensual"] is not None:
    dfm = dfs_f["mensual"]
    # heurística para elegir métrica de líneas
    line_col = pick_first(dfm, ["linea", "línea", "abonados", "suscrip"], numeric=True) or \
               pick_first(dfm, ["valor", "total"], numeric=True)
    if line_col and "periodo" in dfm.columns:
        total_lines = safe_sum(dfm, line_col)
        with c1:
            st.metric("Líneas activas (mensual, filtro aplicado)", f"{total_lines:,.0f}".replace(",", "."))
else:
    with c1:
        st.metric("Líneas activas", "NA")

# KPI 2: Ingresos (trimestrales)
if dfs_f["trimestrales"] is not None:
    dft = dfs_f["trimestrales"]
    inc_col = pick_first(dft, ["ingres"], numeric=True) or pick_first(dft, ["valor","importe"], numeric=True)
    if inc_col:
        total_rev = safe_sum(dft, inc_col)
        with c2:
            st.metric("Ingresos (trimestral)", f"{total_rev:,.0f} €".replace(",", "."))
else:
    with c2:
        st.metric("Ingresos", "NA")

# KPI 3: Cobertura 5G (infraestructuras)
if dfs_f["infraestructuras"] is not None:
    dfi = dfs_f["infraestructuras"]
    cov5g_col = pick_first(dfi, ["5g", "cobertura5", "poblacion_5g", "población_5g", "cov_5g"], numeric=True)
    if cov5g_col:
        cov5g = pd.to_numeric(dfi[cov5g_col], errors="coerce")
        with c3:
            st.metric("Cobertura 5G (media)", f"{cov5g.mean():.1f}%")
    else:
        with c3:
            st.metric("Cobertura 5G", "NA")
else:
    with c3:
        st.metric("Cobertura 5G", "NA")

# KPI 4: HHI (anual_mercados)
if dfs_f["anual_mercados"] is not None:
    dfam = dfs_f["anual_mercados"]
    cuota_col = pick_first(dfam, ["cuota"], numeric=True) or pick_first(dfam, ["share"], numeric=True)
    group_cols = [c for c in ["periodo","mercado"] if c in dfam.columns]
    if cuota_col and group_cols:
        # HHI del último periodo disponible
        lastp = dfam["periodo"].max() if "periodo" in dfam.columns else None
        sub = dfam.copy()
        if lastp is not None:
            sub = sub[sub["periodo"] == lastp]
        hhi = sub.groupby([c for c in group_cols if c != "periodo"])[cuota_col].apply(hhi_from_shares).mean()
        with c4:
            st.metric("HHI medio (últ. periodo)", f"{hhi:,.0f}".replace(",", "."))
    else:
        with c4:
            st.metric("HHI", "NA")
else:
    with c4:
        st.metric("HHI", "NA")

st.caption("Los KPIs dependen de las columnas detectadas automáticamente. Puedes afinar con los filtros o las selecciones de cada bloque.")

# -----------------------------------
# Tabs por bloque analítico
# -----------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Panorama general",
    "Mercado y competencia",
    "Infraestructura y despliegue",
    "Análisis territorial",
    "Indicadores cruzados",
])

# ========== TAB 1: PANORAMA GENERAL ==========
with tab1:
    st.subheader("Evolución temporal de variables clave")
    colA, colB = st.columns(2)

    # Mensual: líneas (o valor)
    if dfs_f["mensual"] is not None and "periodo" in dfs_f["mensual"].columns:
        dfm = dfs_f["mensual"].copy()
        num_cols = [c for c in dfm.columns if pd.api.types.is_numeric_dtype(dfm[c])]
        ycol = pick_first(dfm, ["linea","línea","abonados","valor"], numeric=True) or st.selectbox(
            "Mensual: elige métrica numérica", num_cols, key="m_ycol")
        if ycol:
            g = dfm.groupby("periodo", as_index=False)[ycol].sum()
            fig = px.line(g, x="periodo", y=ycol, title=f"Mensual – {ycol} (agregado)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        with colA:
            st.info("Mensual no disponible.")

    # Trimestral: ingresos
    if dfs_f["trimestrales"] is not None and "periodo" in dfs_f["trimestrales"].columns:
        dft = dfs_f["trimestrales"].copy()
        ycol = pick_first(dft, ["ingres"], numeric=True) or pick_first(dft, ["valor","importe"], numeric=True)
        if not ycol:
            num_cols = [c for c in dft.columns if pd.api.types.is_numeric_dtype(dft[c])]
            ycol = st.selectbox("Trimestral: elige métrica numérica", num_cols, key="t_ycol")
        if ycol:
            g = dft.groupby("periodo", as_index=False)[ycol].sum()
            fig = px.area(g, x="periodo", y=ycol, title=f"Trimestral – {ycol} (agregado)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        with colB:
            st.info("Trimestrales no disponible.")

# ========== TAB 2: MERCADO Y COMPETENCIA ==========
with tab2:
    st.subheader("Cuotas y concentración")

    dfam = dfs_f["anual_mercados"]
    if dfam is None:
        st.info("No hay datos en Anual – Mercados.")
    else:
        # Selectores
        c1, c2, c3 = st.columns([2,1,1])
        mercado_col = "mercado" if "mercado" in dfam.columns else None
        operador_col = "operador" if "operador" in dfam.columns else None
        cuota_col = pick_first(dfam, ["cuota", "share"], numeric=True)

        if not all([mercado_col, operador_col, cuota_col]):
            st.warning("No se detectaron columnas estándar para cuotas. Revisa el CSV final.")
        else:
            mercados = sorted(dfam[mercado_col].dropna().unique().tolist())
            merc_sel = c1.selectbox("Mercado", mercados)
            last_or_all = c2.radio("Periodo", ["Último", "Todos"], horizontal=True)
            topn = int(c3.number_input("Top operadores (para gráfico)", 3, 10, 5))

            sub = dfam[dfam[mercado_col] == merc_sel].copy()
            if "periodo" in sub.columns and last_or_all == "Último":
                lastp = sub["periodo"].max()
                sub = sub[sub["periodo"] == lastp]

            # Market share (barras apiladas o pie si es un único periodo)
            if sub["periodo"].nunique() == 1 if "periodo" in sub.columns else True:
                fig = px.pie(sub.sort_values(cuota_col, ascending=False).head(topn),
                             names=operador_col, values=cuota_col,
                             title=f"Cuota de mercado – {merc_sel}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                g = sub.groupby(["periodo", operador_col], as_index=False)[cuota_col].sum()
                top_ops = g.groupby(operador_col)[cuota_col].mean().nlargest(topn).index
                g = g[g[operador_col].isin(top_ops)]
                fig = px.area(g, x="periodo", y=cuota_col, color=operador_col,
                              title=f"Evolución cuota – {merc_sel}")
                st.plotly_chart(fig, use_container_width=True)

            # HHI por periodo
            if "periodo" in sub.columns:
                g = sub.groupby("periodo").apply(lambda d: hhi_from_shares(d[cuota_col])).reset_index(name="HHI")
                fig2 = px.line(g, x="periodo", y="HHI", title=f"HHI – {merc_sel}")
                st.plotly_chart(fig2, use_container_width=True)

# ========== TAB 3: INFRAESTRUCTURA Y DESPLIEGUE ==========
with tab3:
    st.subheader("Cobertura y capacidades de red")

    dfi = dfs_f["infraestructuras"]
    if dfi is None:
        st.info("Infraestructuras no disponible.")
    else:
        # Selección de métrica
        num_cols = [c for c in dfi.columns if pd.api.types.is_numeric_dtype(dfi[c])]
        guess = pick_first(dfi, ["5g","cobertura","km","nodos","estaciones","hogares"], numeric=True)
        metrica = st.selectbox("Métrica (infraestructura)", num_cols, index=(num_cols.index(guess) if guess in num_cols else 0) if num_cols else 0)

        dims = [c for c in ["periodo","tecnologia","operador","provincia"] if c in dfi.columns]
        col1, col2 = st.columns([2,1])

        if "periodo" in dfi.columns:
            g = dfi.groupby("periodo", as_index=False)[metrica].sum()
            fig = px.line(g, x="periodo", y=metrica, title=f"Evolución {metrica}")
            col1.plotly_chart(fig, use_container_width=True)

        cat = None
        for cand in ["tecnologia","operador"]:
            if cand in dfi.columns:
                cat = cand; break
        if cat:
            g2 = dfi.groupby(cat, as_index=False)[metrica].sum().sort_values(metrica, ascending=False).head(12)
            fig2 = px.bar(g2, x=cat, y=metrica, title=f"{metrica} por {cat}")
            col2.plotly_chart(fig2, use_container_width=True)

# ========== TAB 4: ANÁLISIS TERRITORIAL ==========
with tab4:
    st.subheader("Comparativa provincial")

    dfp = dfs_f["provinciales"]
    if dfp is None:
        st.info("Provinciales no disponible.")
    else:
        if "provincia" not in dfp.columns:
            st.warning("No existe columna 'provincia' en provinciales_final.csv")
        else:
            num_cols = [c for c in dfp.columns if pd.api.types.is_numeric_dtype(dfp[c])]
            metrica = pick_first(dfp, ["linea","línea","penetr","valor","ingres"], numeric=True) or \
                      st.selectbox("Elige métrica provincial", num_cols, key="prov_metrica")
            if metrica:
                # Ranking
                g = dfp.groupby("provincia", as_index=False)[metrica].mean().sort_values(metrica, ascending=False)
                st.plotly_chart(px.bar(g.head(20), x="provincia", y=metrica, title=f"Top 20 provincias por {metrica}"),
                                use_container_width=True)

                # Intento de mapa: si hay geojson o centroides
                geojson_path = "data/geo/provincias.geojson"
                centroids_path = "data/geo/provincias_centroides.csv"

                if os.path.exists(geojson_path):
                    try:
                        with open(geojson_path, "r", encoding="utf-8") as f:
                            gj = json.load(f)
                        # Suponiendo que las features tienen 'provincia' o 'name'
                        figmap = px.choropleth(
                            g,
                            geojson=gj,
                            locations="provincia",
                            color=metrica,
                            featureidkey="properties.name",
                            title=f"Mapa provincial – {metrica}",
                            color_continuous_scale="Viridis"
                        )
                        figmap.update_geos(fitbounds="locations", visible=False)
                        st.plotly_chart(figmap, use_container_width=True)
                    except Exception as e:
                        st.info(f"No se pudo dibujar el mapa con GeoJSON ({e}). Se muestra ranking.")
                elif os.path.exists(centroids_path):
                    try:
                        cen = pd.read_csv(centroids_path)
                        # columnas esperadas: provincia, lat, lon
                        merged = pd.merge(
                            g, cen[["provincia","lat","lon"]],
                            on="provincia", how="inner"
                        )
                        st.map(merged.rename(columns={"lat":"latitude","lon":"longitude"})[["latitude","longitude"]])
                    except Exception as e:
                        st.info(f"No se pudo usar centroides ({e}). Se muestra ranking.")
                else:
                    st.caption("Para el mapa, añade `data/geo/provincias.geojson` o `data/geo/provincias_centroides.csv` (provincia, lat, lon).")

# ========== TAB 5: INDICADORES CRUZADOS ==========
with tab5:
    st.subheader("Relaciones entre variables (cross-dataset)")

    dft = dfs_f["trimestrales"]
    dfi = dfs_f["infraestructuras"]

    if dft is None or dfi is None:
        st.info("Se requieren 'trimestrales_final.csv' e 'infraestructuras_final.csv' para este apartado.")
    else:
        # Elegir métricas
        inc_col = pick_first(dft, ["ingres"], numeric=True) or pick_first(dft, ["valor","importe"], numeric=True)
        infra_col = pick_first(dfi, ["5g","cov","km","nodos","estaciones","hogares"], numeric=True)

        colL, colR = st.columns(2)
        with colL:
            inc_col = st.selectbox("Métrica de ingresos (trimestral)", 
                                   [c for c in dft.columns if pd.api.types.is_numeric_dtype(dft[c])],
                                   index=([c for c in dft.columns if c==inc_col].index(inc_col) if inc_col in dft.columns else 0))
        with colR:
            infra_col = st.selectbox("Métrica de infraestructura", 
                                     [c for c in dfi.columns if pd.api.types.is_numeric_dtype(dfi[c])],
                                     index=([c for c in dfi.columns if c==infra_col].index(infra_col) if infra_col in dfi.columns else 0))

        # Agregación por periodo y (si existe) operador
        keys_t = [c for c in ["periodo","operador"] if c in dft.columns]
        keys_i = [c for c in ["periodo","operador"] if c in dfi.columns]
        gt = dft.groupby(keys_t, as_index=False)[inc_col].sum() if keys_t else None
        gi = dfi.groupby(keys_i, as_index=False)[infra_col].sum() if keys_i else None

        if gt is not None and gi is not None:
            on = [c for c in ["periodo","operador"] if c in gt.columns and c in gi.columns]
            if not on:
                # si no hay claves comunes, agregamos solo por periodo
                if "periodo" in dft.columns and "periodo" in dfi.columns:
                    gt2 = dft.groupby("periodo", as_index=False)[inc_col].sum()
                    gi2 = dfi.groupby("periodo", as_index=False)[infra_col].sum()
                    merged = pd.merge(gt2, gi2, on="periodo", how="inner")
                else:
                    merged = None
            else:
                merged = pd.merge(gt, gi, on=on, how="inner")

            if merged is not None and not merged.empty:
                xcol = inc_col; ycol = infra_col
                title = f"Relación {inc_col} vs {infra_col}"
                color = "operador" if "operador" in merged.columns else None
                fig = px.scatter(merged, x=xcol, y=ycol, color=color, hover_data=on,
                                 trendline="ols", title=title)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No fue posible cruzar métricas con claves comunes; revisa las columnas disponibles.")
        else:
            st.info("No hay columnas numéricas detectadas para el cruce.")

# -----------------------------------
# Nota final
# -----------------------------------
st.caption("""
Este dashboard usa **datasets FINAL** y heurísticas para detectar columnas estándar (ej.: *periodo, operador, servicio, provincia, cuota, ingresos, líneas, tecnología*).
Si una métrica no aparece bien, elige la columna desde los selectores del bloque correspondiente o ajusta los nombres en el CSV final.
""")
