# pages/2_Dashboard_.py
import os
from typing import Optional, Iterable, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dashboard – DSS CNMC (FINAL)", layout="wide")

# -------------------------------------------------------------------
# Rutas a ficheros FINAL
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

FINAL_FILES: Dict[str, str] = {
    "anual_general": os.path.join(BASE_DIR, "data", "final", "anual_datos_generales_final.csv"),
    "anual_mercados": os.path.join(BASE_DIR, "data", "final", "anual_mercados_final.csv"),
    "mensual": os.path.join(BASE_DIR, "data", "final", "mensual_final.csv"),
    "provinciales": os.path.join(BASE_DIR, "data", "final", "provinciales_final.csv"),
    "trimestrales": os.path.join(BASE_DIR, "data", "final", "trimestrales_final.csv"),
    "infraestructuras": os.path.join(BASE_DIR, "data", "final", "infraestructuras_final.csv"),
}

# -------------------------------------------------------------------
# Utilidades de carga y ayuda
# -------------------------------------------------------------------


@st.cache_data
def load_final(name: str) -> pd.DataFrame:
    path = FINAL_FILES[name]
    df = pd.read_csv(path)
    if "periodo" in df.columns:
        df["periodo"] = pd.to_datetime(df["periodo"], errors="coerce")
    return df


def pick_first(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_pct(series: pd.Series) -> pd.Series:
    """Devuelve serie en porcentaje (0–100) asumiendo que puede venir 0–1 o 0–100."""
    s = series.astype(float)
    if len(s) == 0:
        return s
    if s.max() <= 1.0:
        return s * 100.0
    return s


# Coordenadas aproximadas por provincia para mapa
PROV_COORDS: Dict[str, Tuple[float, float]] = {
    "Álava": (42.85, -2.68),
    "Albacete": (38.99, -1.86),
    "Alicante": (38.35, -0.48),
    "Almería": (36.84, -2.46),
    "Asturias": (43.36, -5.85),
    "Ávila": (40.65, -4.69),
    "Badajoz": (38.88, -6.97),
    "Barcelona": (41.39, 2.17),
    "Burgos": (42.34, -3.70),
    "Cáceres": (39.47, -6.37),
    "Cádiz": (36.53, -6.29),
    "Cantabria": (43.46, -3.81),
    "Castellón": (39.98, -0.04),
    "Ciudad Real": (38.99, -3.92),
    "Córdoba": (37.88, -4.78),
    "Cuenca": (40.07, -2.14),
    "Girona": (41.98, 2.82),
    "Granada": (37.18, -3.60),
    "Guadalajara": (40.63, -3.17),
    "Gipuzkoa": (43.31, -1.99),
    "Huelva": (37.26, -6.95),
    "Huesca": (42.14, -0.41),
    "Illes Balears": (39.57, 2.65),
    "Islas Baleares": (39.57, 2.65),
    "Jaén": (37.77, -3.79),
    "La Rioja": (42.46, -2.45),
    "León": (42.60, -5.57),
    "Lleida": (41.62, 0.62),
    "Lugo": (43.01, -7.56),
    "Madrid": (40.42, -3.70),
    "Málaga": (36.72, -4.42),
    "Murcia": (37.99, -1.13),
    "Navarra": (42.82, -1.65),
    "Ourense": (42.34, -7.86),
    "Palencia": (42.01, -4.53),
    "Pontevedra": (42.43, -8.64),
    "Salamanca": (40.97, -5.66),
    "Segovia": (40.95, -4.12),
    "Sevilla": (37.39, -5.99),
    "Soria": (41.77, -2.47),
    "Tarragona": (41.12, 1.25),
    "Santa Cruz de Tenerife": (28.46, -16.25),
    "Teruel": (40.34, -1.11),
    "Toledo": (39.86, -4.02),
    "Valencia": (39.47, -0.38),
    "Valladolid": (41.65, -4.72),
    "Bizkaia": (43.26, -2.93),
    "Zamora": (41.50, -5.74),
    "Zaragoza": (41.65, -0.88),
    "A Coruña": (43.36, -8.41),
    "Las Palmas": (28.10, -15.41),
    "Ceuta": (35.89, -5.32),
    "Melilla": (35.29, -2.94),
}

# -------------------------------------------------------------------
# Carga de datos
# -------------------------------------------------------------------

df_anual_general = load_final("anual_general")
df_anual_mercados = load_final("anual_mercados")
df_mensual = load_final("mensual")
df_prov = load_final("provinciales")
df_trim = load_final("trimestrales")
df_infra = load_final("infraestructuras")

# Años disponibles (para filtros globales)
all_years_series = []
if "periodo" in df_anual_mercados.columns:
    all_years_series.append(df_anual_mercados["periodo"].dt.year.dropna())
if "periodo" in df_prov.columns:
    all_years_series.append(df_prov["periodo"].dt.year.dropna())

all_years = sorted(pd.concat(all_years_series).unique())
min_year, max_year = int(all_years[0]), int(all_years[-1])

# -------------------------------------------------------------------
# FILTROS LATERALES
# -------------------------------------------------------------------

st.sidebar.title("Filtros estratégicos")

year_range = st.sidebar.slider(
    "Años (anual / provincial)",
    min_value=min_year,
    max_value=max_year,
    value=(max_year - 4, max_year),
    step=1,
)

# Mercado/servicio para anual_mercados
servicios = sorted(df_anual_mercados["servicio"].dropna().unique().tolist())
servicio_sel = st.sidebar.selectbox(
    "Servicio / mercado (anual)",
    options=["Todos"] + servicios,
    index=0,
)

# Operadores
operadores = sorted(df_anual_mercados["operador"].dropna().unique().tolist())
ops_default = [op for op in operadores if op.lower() in {"movistar", "orange", "vodafone", "másmóvil", "masmovil"}]
if not ops_default:
    ops_default = operadores[:5]

ops_sel = st.sidebar.multiselect(
    "Operadores (anual / mensual)",
    options=operadores,
    default=ops_default,
)

st.sidebar.markdown("---")
st.sidebar.caption("Datos CNMC (capas CLEAN → FINAL).")

# -------------------------------------------------------------------
# Filtro base sobre anual_mercados
# -------------------------------------------------------------------

df_anual = df_anual_mercados.copy()
if "periodo" in df_anual.columns:
    df_anual["anio"] = df_anual["periodo"].dt.year.astype("Int64")
else:
    df_anual["anio"] = np.nan

mask_year = df_anual["anio"].between(year_range[0], year_range[1])
df_anual = df_anual[mask_year]

if servicio_sel != "Todos":
    df_anual = df_anual[df_anual["servicio"] == servicio_sel]

if ops_sel:
    df_anual = df_anual[df_anual["operador"].isin(ops_sel)]

# -------------------------------------------------------------------
# Layout principal
# -------------------------------------------------------------------

st.title("Dashboard estratégico de operadores – CNMC")

st.markdown(
    """
Este dashboard explota la **capa FINAL** armonizada de la CNMC para
apoyar decisiones estratégicas de operadores de telecomunicaciones:

- Evolución de **ingresos por operador y mercado**.
- **Cuotas de mercado** y concentración (HHI).
- Distribución territorial (**penetración por provincia**).
- Visión táctica mensual y alguna referencia trimestral.
"""
)

st.divider()

# -------------------------------------------------------------------
# BLOQUE 1: KPIs ejecutivos (anual_mercados)
# -------------------------------------------------------------------

if df_anual.empty:
    latest_year = year_range[1]
    df_latest = df_anual_mercados.copy()
    df_latest["anio"] = df_latest["periodo"].dt.year.astype("Int64")
    df_latest = df_latest[df_latest["anio"] == latest_year]
else:
    latest_year = int(df_anual["anio"].max())
    df_latest = df_anual[df_anual["anio"] == latest_year]

total_ingresos_latest = df_latest["ingresos_por_operador"].sum()

prev_year = latest_year - 1
df_prev = df_anual_mercados.copy()
if "periodo" in df_prev.columns:
    df_prev["anio"] = df_prev["periodo"].dt.year.astype("Int64")
else:
    df_prev["anio"] = np.nan
df_prev = df_prev[df_prev["anio"] == prev_year]
if servicio_sel != "Todos":
    df_prev = df_prev[df_prev["servicio"] == servicio_sel]
if ops_sel:
    df_prev = df_prev[df_prev["operador"].isin(ops_sel)]
total_ingresos_prev = df_prev["ingresos_por_operador"].sum()

growth = None
if total_ingresos_prev > 0:
    growth = (total_ingresos_latest / total_ingresos_prev - 1) * 100

# Cuota top operador
cuotas_latest = df_latest.copy()
if "cuota_ingresos" in cuotas_latest.columns:
    cuotas_latest["cuota_pct"] = ensure_pct(cuotas_latest["cuota_ingresos"])
    cuotas_agg = (
        cuotas_latest.groupby("operador", as_index=False)["cuota_pct"]
        .mean()
        .sort_values("cuota_pct", ascending=False)
    )
    top_row = cuotas_agg.iloc[0] if not cuotas_agg.empty else None
else:
    top_row = None
    cuotas_agg = pd.DataFrame(columns=["operador", "cuota_pct"])

# HHI medio
if "hhi_ingresos" in df_latest.columns:
    hhi_latest_series = df_latest["hhi_ingresos"].dropna()
    hhi_latest = float(hhi_latest_series.mean() * 10_000) if not hhi_latest_series.empty else None
else:
    hhi_latest = None

# Número de operadores activos
num_ops_latest = df_latest["operador"].nunique()

# Mostrar KPIs
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        f"Ingresos {latest_year} (M€ aprox.)",
        value=f"{total_ingresos_latest:,.0f}",
        delta=f"{growth:+.1f} % vs {prev_year}" if growth is not None else None,
        help="Suma de ingresos_por_operador en el periodo y filtros seleccionados.",
    )

with k2:
    if top_row is not None:
        st.metric(
            "Operador líder (ingresos)",
            value=f"{top_row['operador']}",
            delta=f"{top_row['cuota_pct']:.1f} %",
            help="Operador con mayor cuota media de ingresos en el último año.",
        )
    else:
        st.metric("Operador líder (ingresos)", value="N/D")

with k3:
    st.metric(
        "Operadores activos",
        value=str(num_ops_latest),
        help="Número de operadores con ingresos en el último año filtrado.",
    )

with k4:
    if hhi_latest is not None:
        st.metric(
            "HHI (ingresos)",
            value=f"{hhi_latest:,.0f}",
            help="Índice de Herfindahl-Hirschman basado en cuotas de ingresos. >2500 indica alta concentración.",
        )
    else:
        st.metric("HHI (ingresos)", value="N/D")

st.divider()

# -------------------------------------------------------------------
# BLOQUE 2: Series anuales por operador y estructura de mercado
# -------------------------------------------------------------------

st.subheader("Evolución anual por operador y estructura competitiva")

row1_c1, row1_c2 = st.columns((2, 1))

# Serie de ingresos por operador
with row1_c1:
    if df_anual.empty:
        st.info("No hay datos para la combinación de filtros seleccionada.")
    else:
        df_series = df_anual.dropna(subset=["ingresos_por_operador"])
        df_series = df_series.sort_values("periodo")
        fig = px.line(
            df_series,
            x="periodo",
            y="ingresos_por_operador",
            color="operador",
            markers=True,
            labels={
                "periodo": "Periodo",
                "ingresos_por_operador": "Ingresos por operador (M€)",
                "operador": "Operador",
            },
            title="Ingresos por operador (anual)",
        )
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

# Barras de cuota de mercado en último año
with row1_c2:
    if not cuotas_agg.empty:
        fig_bar = px.bar(
            cuotas_agg.head(10),
            x="operador",
            y="cuota_pct",
            labels={"operador": "Operador", "cuota_pct": "Cuota ingresos (%)"},
            title=f"Cuota de ingresos por operador – {latest_year}",
        )
        fig_bar.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No hay datos de cuota de ingresos disponibles para este filtro.")

st.divider()

# -------------------------------------------------------------------
# BLOQUE 3: Visión territorial (provincias)
# -------------------------------------------------------------------

st.subheader("Cobertura y penetración por provincia")

df_prov = df_prov.copy()
if "periodo" in df_prov.columns:
    df_prov["anio"] = df_prov["periodo"].dt.year.astype("Int64")
else:
    df_prov["anio"] = np.nan

prov_years = sorted(df_prov["anio"].dropna().unique().tolist())
year_prov_sel = st.slider(
    "Año para mapa provincial",
    min_value=int(prov_years[0]),
    max_value=int(prov_years[-1]),
    value=int(prov_years[-1]),
    step=1,
)

serv_prov_opts = sorted(df_prov["servicio"].dropna().unique().tolist())
serv_prov_sel = st.selectbox(
    "Servicio (provincial)",
    options=["Todos"] + serv_prov_opts,
    index=0,
)

dfp = df_prov[df_prov["anio"] == year_prov_sel].copy()
if serv_prov_sel != "Todos":
    dfp = dfp[dfp["servicio"] == serv_prov_sel]

row2_c1, row2_c2 = st.columns((1.5, 1))

if not dfp.empty:
    g = (
        dfp.groupby("provincia", as_index=False)[["tasa_de_penetracion", "lineas_o_accesos"]]
        .mean()
        .sort_values("tasa_de_penetracion", ascending=False)
    )
    g["coords"] = g["provincia"].map(PROV_COORDS.get)
    g["lat"] = g["coords"].apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    g["lon"] = g["coords"].apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
    g = g.dropna(subset=["lat", "lon"])

    with row2_c1:
        if g.empty:
            st.info("No hay coordenadas definidas para las provincias de este conjunto.")
        else:
            map_df = g.rename(columns={"lat": "latitude", "lon": "longitude"}).copy()
            map_df = map_df.dropna(subset=["latitude", "longitude"])
            # Para evitar errores, solo pasamos lat/lon a st.map
            st.map(map_df[["latitude", "longitude"]])
            st.caption("Distribución geográfica de provincias con datos disponibles.")

    with row2_c2:
        fig_prov = px.bar(
            g.head(15),
            x="provincia",
            y="tasa_de_penetracion",
            labels={
                "provincia": "Provincia",
                "tasa_de_penetracion": "Tasa de penetración (líneas/100 hab.)",
            },
            title=f"Top provincias por penetración – {year_prov_sel}",
        )
        fig_prov.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_prov, use_container_width=True)
else:
    with row2_c1:
        st.info("No hay datos provinciales para el año/servicio seleccionados.")
    with row2_c2:
        st.empty()

st.divider()

# -------------------------------------------------------------------
# BLOQUE 4: Visión táctica mensual y referencia trimestral
# -------------------------------------------------------------------

st.subheader("Detalle mensual y trimestral (táctico)")

row3_c1, row3_c2 = st.columns(2)

# Mensual: cuotas por líneas
with row3_c1:
    if not df_mensual.empty:
        dfm = df_mensual.copy()
        if "periodo" in dfm.columns:
            dfm["anio"] = dfm["periodo"].dt.year.astype("Int64")
        else:
            dfm["anio"] = np.nan
        mask_m = dfm["anio"].between(year_range[0], year_range[1])
        dfm = dfm[mask_m]
        if ops_sel:
            dfm = dfm[dfm["operador"].isin(ops_sel)]
        if dfm.empty:
            st.info("No hay datos mensuales para los filtros seleccionados.")
        else:
            fig_m = px.line(
                dfm.sort_values("periodo"),
                x="periodo",
                y="lineas",
                color="operador",
                markers=True,
                labels={
                    "periodo": "Periodo",
                    "lineas": "Líneas / accesos",
                    "operador": "Operador",
                },
                title="Líneas por operador (mensual, muestra)",
            )
            fig_m.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("No hay datos mensuales en la capa FINAL.")

# Trimestral: ingresos y HHI (si disponibles)
with row3_c2:
    if not df_trim.empty:
        dft = df_trim.copy()
        if "periodo" in dft.columns:
            dft["anio"] = dft["periodo"].dt.year.astype("Int64")
        else:
            dft["anio"] = np.nan
        mask_t = dft["anio"].between(year_range[0], year_range[1])
        dft = dft[mask_t]
        if ops_sel:
            dft = dft[dft["operador"].isin(ops_sel)]

        if dft.empty:
            st.info("No hay datos trimestrales para los filtros seleccionados.")
        else:
            fig_t = px.bar(
                dft,
                x="periodo",
                y="ingresos_por_operador",
                color="operador",
                labels={
                    "periodo": "Periodo",
                    "ingresos_por_operador": "Ingresos por operador",
                    "operador": "Operador",
                },
                title="Ingresos por operador (trimestral, muestra)",
            )
            fig_t.update_layout(height=300, barmode="stack", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No hay datos trimestrales en la capa FINAL.")

st.divider()

# -------------------------------------------------------------------
# BLOQUE 5: Infraestructura por tecnología (si aplica)
# -------------------------------------------------------------------

st.subheader("Infraestructura por tecnología (visión estática)")

if df_infra.empty:
    st.info("No hay datos de infraestructuras en la capa FINAL.")
else:
    col_tecn = pick_first(df_infra, ["tecnologia_de_acceso", "tecnologia"])
    metric_infra = pick_first(
        df_infra,
        ["lineas_o_accesos", "estaciones_base", "trafico_de_datos", "nodos_radio"],
    )

    if col_tecn is not None and metric_infra is not None:
        g_inf = (
            df_infra.groupby(col_tecn, as_index=False)[metric_infra]
            .sum()
            .sort_values(metric_infra, ascending=False)
        )
        fig_inf = px.bar(
            g_inf,
            x=col_tecn,
            y=metric_infra,
            labels={col_tecn: "Tecnología", metric_infra: "Intensidad"},
            title="Infraestructura por tecnología (última foto disponible)",
        )
        fig_inf.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_inf, use_container_width=True)
    else:
        st.info("No se ha podido identificar una métrica clara de infraestructura.")
