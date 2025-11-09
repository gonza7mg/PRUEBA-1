# pages/3_Dashboard.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard DSS – CNMC", page_icon="📊", layout="wide")

# --------------------------
# Helpers
# --------------------------
@st.cache_data(show_spinner=False)
def load_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"No se pudo cargar {path}: {e}")
        return pd.DataFrame()

def CAGR(series: pd.Series) -> float:
    """Crecimiento compuesto anual (suponiendo datos anuales en orden temporal)"""
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    first, last = s.iloc[0], s.iloc[-1]
    n_years = len(s) - 1
    if first <= 0:
        return np.nan
    return (last / first) ** (1 / n_years) - 1

def hhi(shares: pd.Series) -> float:
    """Índice HHI (shares en % o 0-1). Devuelve 0-10000 en escala estándar."""
    s = shares.dropna().astype(float)
    if s.max() <= 1.0:  # si viene en proporciones
        s = s * 100
    return float(((s) ** 2).sum())

def pct_change(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    return (s.iloc[-1] - s.iloc[-2]) / abs(s.iloc[-2]) if s.iloc[-2] != 0 else np.nan

# --------------------------
# Datos CLEAN (rutas)
# --------------------------
PATHS = {
    "anual_dg": "data/clean/anual_datos_generales_clean.csv",
    "anual_merc": "data/clean/anual_mercados_clean.csv",
    "mensual": "data/clean/mensual_clean.csv",
    "prov": "data/clean/provinciales_clean.csv",
    "trim": "data/clean/trimestrales_clean.csv",
    "infra": "data/clean/infraestructuras_clean.csv",
}

dg   = load_csv_safe(PATHS["anual_dg"])      # anual – datos generales
merc = load_csv_safe(PATHS["anual_merc"])    # anual – mercados
mens = load_csv_safe(PATHS["mensual"])       # mensual
prov = load_csv_safe(PATHS["prov"])          # provinciales
tri  = load_csv_safe(PATHS["trim"])          # trimestrales
inf  = load_csv_safe(PATHS["infra"])         # infraestructuras

# Normalizaciones suaves por si cambian nombres
for df in [dg, merc, mens, prov, tri, inf]:
    if "anno" in df.columns:
        df.rename(columns={"anno": "anio"}, inplace=True)
    if "año" in df.columns:
        df.rename(columns={"año": "anio"}, inplace=True)

# --------------------------
# Filtros globales
# --------------------------
st.title("📊 Dashboard DSS – CNMC (CLEAN)")
st.caption("Todos los gráficos se alimentan de data/clean/*.csv tras la limpieza y normalización.")

# inferencia rápida de rangos
years_candidates = []
for df in [dg, merc, mens, prov, tri, inf]:
    if "anio" in df.columns:
        y = pd.to_numeric(df["anio"], errors="coerce")
        years_candidates.extend(y.dropna().astype(int).tolist())
if years_candidates:
    min_year, max_year = int(np.min(years_candidates)), int(np.max(years_candidates))
else:
    min_year, max_year = 2000, 2025

c1, c2, c3 = st.columns([1,1,2])
with c1:
    year_from, year_to = st.slider("Rango de años", min_year, max_year, (max(min_year, max_year-5), max_year))
with c2:
    operador_sel = st.selectbox(
        "Operador (opcional)", 
        sorted(list(set(dg.get("operador", pd.Series([])).dropna().unique()) 
                    | set(merc.get("operador", pd.Series([])).dropna().unique())
                    | set(tri.get("operador", pd.Series([])).dropna().unique())) ) or ["(Todos)"]
    )
with c3:
    tecnologia_sel = st.multiselect(
        "Tecnología (opcional)",
        sorted(list(set(mens.get("tecnologia", pd.Series([])).dropna().unique()) 
                    | set(prov.get("tecnologia", pd.Series([])).dropna().unique())
                    | set(inf.get("tecnologia", pd.Series([])).dropna().unique()))),
        []
    )

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "anio" in out.columns:
        out = out[(out["anio"] >= year_from) & (out["anio"] <= year_to)]
    if operador_sel and operador_sel != "(Todos)" and "operador" in out.columns:
        out = out[out["operador"] == operador_sel]
    if tecnologia_sel and "tecnologia" in out.columns:
        out = out[out["tecnologia"].isin(tecnologia_sel)]
    return out

dg_f   = apply_filters(dg)
merc_f = apply_filters(merc)
mens_f = apply_filters(mens)
prov_f = apply_filters(prov)
tri_f  = apply_filters(tri)
inf_f  = apply_filters(inf)

# --------------------------
# PESTAÑAS
# --------------------------
tabs = st.tabs(["📌 Resumen", "💶 Económico", "🏷️ Mercado", "🗺️ Territorial", "📈 Operativo", "🧪 Calidad"])

# ==========================
# 1) RESUMEN
# ==========================
with tabs[0]:
    st.subheader("Resumen ejecutivo")

    # KPIs básicos (robustos: intentan distintas columnas si varían nombres)
    def first_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    col_ingresos = first_col(dg_f, ["ingresos", "ingresos_total", "ingresos_por_operador"])
    col_inv      = first_col(tri_f, ["inversiones", "capex", "inversion"])
    col_lineas   = first_col(mens_f, ["lineas_activas", "lineas"])
    col_hogares  = first_col(prov_f, ["penetracion", "lineas"])  # proxy territorial

    total_ingresos = dg_f[col_ingresos].sum() if col_ingresos else np.nan
    total_inversion = tri_f[col_inv].sum() if col_inv else np.nan
    total_lineas = mens_f[col_lineas].iloc[-1] if (col_lineas and not mens_f.empty) else np.nan

    # CAGR de ingresos anuales (si hay columna de anio e ingresos)
    cagr_ing = np.nan
    if col_ingresos and "anio" in dg_f.columns:
        by_year = dg_f.groupby("anio")[col_ingresos].sum().sort_index()
        cagr_ing = CAGR(by_year)

    cA, cB, cC, cD = st.columns(4)
    with cA: st.metric("Ingresos (periodo filtrado)", f"{total_ingresos:,.0f}")
    with cB: st.metric("Inversión total", f"{total_inversion:,.0f}" if not np.isnan(total_inversion) else "—")
    with cC: st.metric("Líneas activas (último mes)", f"{total_lineas:,.0f}" if not np.isnan(total_lineas) else "—")
    with cD: st.metric("CAGR Ingresos", f"{cagr_ing*100:,.2f}%" if not np.isnan(cagr_ing) else "—")

    # Evolución temporal (ingresos) y líneas
    if col_ingresos and "anio" in dg_f.columns:
        by_year = dg_f.groupby("anio")[col_ingresos].sum().reset_index().sort_values("anio")
        fig = px.line(by_year, x="anio", y=col_ingresos, markers=True, title="Evolución de ingresos (anual)")
        st.plotly_chart(fig, use_container_width=True)

    if col_lineas and "anio" not in mens_f.columns and not mens_f.empty:
        # Si mensual tiene 'fecha' o 'mes'
        time_col = "fecha" if "fecha" in mens_f.columns else ("mes" if "mes" in mens_f.columns else None)
        if time_col:
            by_month = mens_f.groupby(time_col)[col_lineas].sum().reset_index()
            fig2 = px.line(by_month, x=time_col, y=col_lineas, markers=False, title="Evolución de líneas activas (mensual)")
            st.plotly_chart(fig2, use_container_width=True)

# ==========================
# 2) ECONÓMICO (Trimestral / Anual)
# ==========================
with tabs[1]:
    st.subheader("Desempeño económico-financiero")

    col_ing_trim = first_col(tri_f, ["ingresos", "ingresos_totales"])
    col_inv_trim = first_col(tri_f, ["inversiones", "capex", "inversion"])
    col_ebitda   = first_col(tri_f, ["ebitda"])

    if not tri_f.empty and col_ing_trim:
        fig = px.line(tri_f.sort_values(["anio"]), x="anio", y=col_ing_trim, color="operador" if "operador" in tri_f.columns else None,
                      markers=True, title="Ingresos por operador (trimestral/anualizado)")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if not tri_f.empty and col_inv_trim:
            figi = px.bar(tri_f, x="anio", y=col_inv_trim, color="operador" if "operador" in tri_f.columns else None,
                          barmode="group", title="Inversión (CAPEX) por año")
            st.plotly_chart(figi, use_container_width=True)
    with c2:
        if not tri_f.empty and col_ebitda:
            figm = px.line(tri_f.sort_values(["anio"]), x="anio", y=col_ebitda,
                           color="operador" if "operador" in tri_f.columns else None,
                           markers=True, title="EBITDA por operador")
            st.plotly_chart(figm, use_container_width=True)

# ==========================
# 3) MERCADO (Cuotas, HHI)
# ==========================
with tabs[2]:
    st.subheader("Estructura de mercado y competencia")

    # Cuotas por año y operador
    cuota_col = first_col(merc_f, ["cuota_mercado", "cuota", "participacion"])
    if not merc_f.empty and cuota_col:
        # Último año del filtro
        last_year = merc_f["anio"].dropna().max() if "anio" in merc_f.columns else None
        if last_year is not None:
            df_last = merc_f[merc_f["anio"] == last_year]
            fig = px.bar(df_last.sort_values(cuota_col, ascending=False),
                         x="operador" if "operador" in df_last.columns else None,
                         y=cuota_col, color="operador" if "operador" in df_last.columns else None,
                         title=f"Cuotas de mercado – {int(last_year)}")
            st.plotly_chart(fig, use_container_width=True)

        # HHI por año
        if "anio" in merc_f.columns and "operador" in merc_f.columns:
            hhi_by_year = (merc_f.groupby("anio")
                                   .apply(lambda g: hhi(g[cuota_col]))
                                   .reset_index(name="HHI"))
            fig_hhi = px.line(hhi_by_year, x="anio", y="HHI", markers=True, title="Índice HHI (concentración del mercado)")
            st.plotly_chart(fig_hhi, use_container_width=True)

# ==========================
# 4) TERRITORIAL (Provinciales + Infra)
# ==========================
with tabs[3]:
    st.subheader("Distribución territorial y despliegue")

    # Penetración media por provincia (si existe)
    pen_col = first_col(prov_f, ["penetracion", "lineas_por_100_hab", "lineas_per_capita"])
    if not prov_f.empty and pen_col:
        # Top / bottom provincias
        topn = st.slider("Top/Bottom provincias", 3, 20, 10)
        last_year = prov_f["anio"].max() if "anio" in prov_f.columns else None
        dfp = prov_f if last_year is None else prov_f[prov_f["anio"] == last_year]
        if "provincia" in dfp.columns:
            c1, c2 = st.columns(2)
            with c1:
                top = dfp.sort_values(pen_col, ascending=False).head(topn)
                fig_top = px.bar(top, x="provincia", y=pen_col, title=f"Top {topn} provincias por penetración")
                st.plotly_chart(fig_top, use_container_width=True)
            with c2:
                bot = dfp.sort_values(pen_col, ascending=True).head(topn)
                fig_bot = px.bar(bot, x="provincia", y=pen_col, title=f"Bottom {topn} provincias por penetración")
                st.plotly_chart(fig_bot, use_container_width=True)

    # Infraestructuras por tecnología
    acc_col = first_col(inf_f, ["accesos", "nodos", "km_red", "unidades"])
    if not inf_f.empty and acc_col:
        fig_inf = px.bar(inf_f, x="anio" if "anio" in inf_f.columns else None,
                         y=acc_col, color="tecnologia" if "tecnologia" in inf_f.columns else None,
                         barmode="group", title="Despliegue de infraestructuras por tecnología")
        st.plotly_chart(fig_inf, use_container_width=True)

# ==========================
# 5) OPERATIVO (Mensual)
# ==========================
with tabs[4]:
    st.subheader("Dinámica operativa (mensual)")

    alt_col  = first_col(mens_f, ["altas", "altas_mensuales"])
    baj_col  = first_col(mens_f, ["bajas", "bajas_mensuales"])
    lin_col  = first_col(mens_f, ["lineas_activas", "lineas"])

    # Identificar la columna temporal (fecha/mes)
    time_col = "fecha" if "fecha" in mens_f.columns else ("mes" if "mes" in mens_f.columns else None)

    if time_col and (alt_col or baj_col or lin_col):
        c1, c2 = st.columns(2)
        with c1:
            if alt_col and baj_col:
                tmp = mens_f[[time_col, alt_col, baj_col]].groupby(time_col).sum().reset_index()
                tmp["netas"] = tmp[alt_col] - tmp[baj_col]
                fig_ops = px.line(tmp, x=time_col, y=["netas", alt_col, baj_col],
                                  title="Altas, bajas y netas", markers=True)
                st.plotly_chart(fig_ops, use_container_width=True)
        with c2:
            if lin_col:
                tmp2 = mens_f[[time_col, lin_col]].groupby(time_col).sum().reset_index()
                fig_lin = px.line(tmp2, x=time_col, y=lin_col, title="Líneas activas (mensual)")
                st.plotly_chart(fig_lin, use_container_width=True)

# ==========================
# 6) CALIDAD (resumen rápido)
# ==========================
with tabs[5]:
    st.subheader("Calidad de datos (resumen)")

    # Indicadores sencillos por dataset
    def quick_quality(df: pd.DataFrame, nombre: str):
        n_rows, n_cols = df.shape
        n_nulls = int(df.isna().sum().sum())
        n_dups  = int(df.duplicated().sum())
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric(f"{nombre}: filas", f"{n_rows:,}")
        with c2: st.metric(f"{nombre}: columnas", f"{n_cols}")
        with c3: st.metric(f"{nombre}: nulos", f"{n_nulls:,}")
        with c4: st.metric(f"{nombre}: duplicados", f"{n_dups:,}")

    quick_quality(dg_f,   "Anual – Datos generales")
    quick_quality(merc_f, "Anual – Mercados")
    quick_quality(mens_f, "Mensual")
    quick_quality(prov_f, "Provinciales")
    quick_quality(tri_f,  "Trimestrales")
    quick_quality(inf_f,  "Infraestructuras")

    st.caption("Para detalle completo, usa la página 'Calidad de Datos' con la suite de evaluación.")
