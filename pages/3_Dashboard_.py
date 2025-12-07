# pages/2_Dashboard_.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path
import json

st.set_page_config(
    page_title="Dashboard Mercado Telecom – CNMC",
    page_icon="📊",
    layout="wide",
)

# =====================================================================
# RUTAS DE FICHEROS: FINAL + CLEAN + RAW
# =====================================================================

FINAL_FILES = {
    "anual_mercados":       Path("data/final/anual_mercados_final.csv"),
    "anual_datos_generales":Path("data/final/anual_datos_generales_final.csv"),
    "provinciales":         Path("data/final/provinciales_final.csv"),
    "mensual":              Path("data/final/mensual_final.csv"),
    "trimestrales":         Path("data/final/trimestrales_final.csv"),
    "infraestructuras":     Path("data/final/infraestructuras_final.csv"),
}

CLEAN_FILES = {
    "anual_mercados":       Path("data/clean/anual_mercados_clean.csv"),
    "anual_datos_generales":Path("data/clean/anual_datos_generales_clean.csv"),
    "provinciales":         Path("data/clean/provinciales_clean.csv"),
    "mensual":              Path("data/clean/mensual_clean.csv"),
    "trimestrales":         Path("data/clean/trimestrales_clean.csv"),
    "infraestructuras":     Path("data/clean/infraestructuras_clean.csv"),
}

RAW_FILES = {
    "anual_mercados": Path("data/raw/anual_mercados.csv"),
    "mensual":        Path("data/raw/mensual.csv"),
    "trimestrales":   Path("data/raw/trimestrales.csv"),
}

DIM_COL_CANDIDATES = {
    "common": [
        "periodo", "pais", "provincia", "ccaa", "mes", "trimestre",
        "servicio", "concepto", "operador",
        "tipo_de_mercado", "tipo_de_ingreso", "tipo_de_cliente",
        "segmento", "tipo_de_trafico", "tipo_de_contrato",
        "tipo_de_linea", "tipo_de_mensaje", "tipo_de_trafico_de_mensaje",
        "tecnologia_de_acceso", "tecnologia_de_acceso_baf",
        "tipo_de_ba_mayorista", "tipo_de_estaciones_base",
        "tipo_de_ba_mayorista",
    ]
}

# Colores corporativos por operador principales
COLOR_OPERADORES = {
    "Movistar":   "#0072C6",
    "TELEFONICA": "#0072C6",
    "Telefónica": "#0072C6",
    "Vodafone":   "#E60000",
    "VODAFONE":   "#E60000",
    "Orange":     "#FF6600",
    "ORANGE":     "#FF6600",
}

GEO_PROV_PATH = Path("data/geo/provincias_es.geojson")

# Mapeo de nombres entre CNMC y tu geojson (properties.provincia)
PROV_NAME_MAP = {
    "Coruña. A": "La Coruña",
    "Balears. Illes": "Islas Baleares",
    "Rioja. La": "La Rioja",
    "Palmas. Las": "Las Palmas",
    "Araba/Álava": "Álava",
    "Gipuzkoa": "Guipúzcoa",
    "Bizkaia": "Vizcaya",
    "Castellón/Castelló": "Castellón",
    "Valencia/València": "Valencia",
    "Girona": "Gerona",
}

# =====================================================================
# CARGA DE DATOS
# =====================================================================

@st.cache_data(show_spinner=False)
def load_merged_dataset(name: str) -> pd.DataFrame:
    """
    Carga el dataset 'final' y, si existe, añade dimensiones desde el 'clean'
    usando el id. Pensado para vistas estructurales (anual, provincial).
    """
    final_path = FINAL_FILES.get(name)
    clean_path = CLEAN_FILES.get(name)

    df_final = None
    df_clean = None

    if final_path is not None and final_path.exists():
        df_final = pd.read_csv(final_path)

    if clean_path is not None and clean_path.exists():
        df_clean = pd.read_csv(clean_path)

    if df_final is None and df_clean is None:
        return pd.DataFrame()

    if df_final is None:
        df = df_clean.copy()
    elif df_clean is None:
        df = df_final.copy()
    else:
        df_final = df_final.copy()
        df_clean = df_clean.copy()

        if "id" not in df_final.columns or "id" not in df_clean.columns:
            df = df_final.copy()
        else:
            dim_cols = ["id"]
            for c in DIM_COL_CANDIDATES["common"]:
                if c in df_clean.columns:
                    dim_cols.append(c)
            dim_cols += [
                c for c in df_clean.columns
                if c not in dim_cols and df_clean[c].dtype == "object"
            ]
            dim_cols = list(dict.fromkeys(dim_cols))

            dims = df_clean[dim_cols].drop_duplicates(subset=["id"])
            df = df_final.merge(dims, on="id", how="left", suffixes=("", "_dim"))

            for c in ["servicio", "concepto", "operador", "tipo_de_mercado",
                      "provincia", "ccaa"]:
                if c in df.columns and f"{c}_dim" in df.columns:
                    df[c] = df[c].fillna(df[f"{c}_dim"])
                    df.drop(columns=[f"{c}_dim"], inplace=True)

    if "anio" not in df.columns and "anno" in df.columns:
        df["anio"] = df["anno"]
    if "anio" in df.columns:
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")

    return df


@st.cache_data(show_spinner=False)
def load_raw_dataset(name: str) -> pd.DataFrame:
    """
    Carga directamente el dataset RAW (datos originales CNMC).
    """
    path = RAW_FILES.get(name)
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    if "anio" not in df.columns and "anno" in df.columns:
        df["anio"] = df["anno"]
    if "anio" in df.columns:
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    return df


# =====================================================================
# UTILIDADES COMUNES
# =====================================================================

def safe_year_range(df: pd.DataFrame):
    if "anio" not in df.columns:
        return (None, None)
    vals = df["anio"].dropna().astype(int)
    if vals.empty:
        return (None, None)
    return (int(vals.min()), int(vals.max()))


def ensure_month_date(col_mes: pd.Series, col_anio: pd.Series | None = None):
    """Convierte columna 'mes' en datetime. Admite formato 'YYYY-MM' o números."""
    s = col_mes.astype(str)
    if col_anio is not None and not col_anio.isna().all():
        y = col_anio.astype("Int64")
        m = pd.to_numeric(s, errors="coerce")
        return pd.to_datetime({"year": y, "month": m, "day": 1}, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def compute_hhi(shares: pd.Series) -> float:
    """HHI en escala 0–10.000 (habitual en competencia)."""
    return float(((shares * 100) ** 2).sum())


# =====================================================================
# LAYOUT
# =====================================================================

st.title("📊 Dashboard del mercado de telecomunicaciones")
st.caption(
    "Evolución de ingresos, cuotas y concentración, distribución territorial "
    "y visión táctica mensual / trimestral del mercado telecom."
)

# ---------------------------------------------------------------------
# Cargamos datasets necesarios (estructural y táctica)
# ---------------------------------------------------------------------
anual_merc_merged = load_merged_dataset("anual_mercados")
anual_merc_raw    = load_raw_dataset("anual_mercados")
anual_dg          = load_merged_dataset("anual_datos_generales")
prov              = load_merged_dataset("provinciales")

mensual_raw       = load_raw_dataset("mensual")
trimestral_raw    = load_raw_dataset("trimestrales")

anual_merc = anual_merc_merged

if anual_merc_raw.empty and anual_merc_merged.empty:
    st.error("No se ha podido cargar ningún dataset anual de mercados (RAW ni final).")
    st.stop()

# =====================================================================
# 1. Evolución de ingresos por operador y mercado (USANDO RAW)
# =====================================================================

st.header("1. Evolución de ingresos por operador y mercado (datos RAW)")

df_ing = anual_merc_raw.copy() if not anual_merc_raw.empty else anual_merc_merged.copy()

year_min, year_max = safe_year_range(df_ing)
if year_min is None:
    year_min, year_max = 2010, 2024

c1, c2, c3 = st.columns(3)
with c1:
    anio_range = st.slider(
        "Años (intervalo de análisis)",
        int(year_min),
        int(year_max),
        (int(year_min), int(year_max)),
        step=1,
        key="ev_anio_range",
    )
with c2:
    mercados = sorted(df_ing["tipo_de_mercado"].dropna().unique()) \
        if "tipo_de_mercado" in df_ing.columns else []
    mercado_sel = st.selectbox(
        "Tipo de mercado",
        ["(todos)"] + mercados,
        index=0,
        key="ev_mercado",
    )
with c3:
    servicios = sorted(df_ing["servicio"].dropna().unique()) \
        if "servicio" in df_ing.columns else []
    servicio_sel = st.multiselect(
        "Servicio / mercado",
        servicios,
        default=servicios if servicios else [],
        key="ev_servicio",
    )

df_ev = df_ing[
    (df_ing["anio"] >= anio_range[0]) &
    (df_ing["anio"] <= anio_range[1])
].copy()

if mercado_sel != "(todos)" and "tipo_de_mercado" in df_ev.columns:
    df_ev = df_ev[df_ev["tipo_de_mercado"] == mercado_sel]

if servicio_sel and "servicio" in df_ev.columns:
    df_ev = df_ev[df_ev["servicio"].isin(servicio_sel)]

if "operador" in df_ev.columns:
    all_ops = sorted(df_ev["operador"].dropna().unique())
else:
    all_ops = []

if "ingresos_por_operador" in df_ev.columns and "operador" in df_ev.columns:
    top_ops = (
        df_ev.groupby("operador", dropna=True)["ingresos_por_operador"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
else:
    top_ops = all_ops[:5]

operadores_sel = st.multiselect(
    "Operadores",
    all_ops,
    default=top_ops,
    key="ev_operadores",
)

if operadores_sel:
    df_ev = df_ev[df_ev["operador"].isin(operadores_sel)]

if df_ev.empty:
    st.warning("No hay datos para la combinación de filtros seleccionada.")
else:
    df_plot = df_ev.copy()
    if "ingresos_por_operador" in df_plot.columns:
        df_plot["ingresos_plot"] = df_plot["ingresos_por_operador"]
    else:
        df_plot["ingresos_plot"] = df_plot.get("ingresos", 0)

    # AGRUPAMOS POR AÑO Y OPERADOR -> UNA LÍNEA CONTINUA POR OPERADOR
    df_line = (
        df_plot.groupby(["anio", "operador"], as_index=False)["ingresos_plot"]
        .sum()
        .sort_values("anio")
    )

    fig_ev = px.line(
        df_line,
        x="anio",
        y="ingresos_plot",
        color="operador",
        color_discrete_map=COLOR_OPERADORES,
        markers=False,
        labels={
            "anio": "Año",
            "ingresos_plot": "Ingresos (M€)",
            "operador": "Operador",
        },
        title="Evolución de ingresos por operador (M€) – datos RAW",
    )
    fig_ev.update_traces(mode="lines", line=dict(width=3))
    fig_ev.update_layout(legend_title_text="Operador")
    st.plotly_chart(fig_ev, use_container_width=True)

    st.markdown("**Resumen por operador y servicio (media anual en el periodo seleccionado – RAW)**")
    resumen = (
        df_plot
        .groupby(["operador", "servicio"], dropna=False)["ingresos_plot"]
        .mean()
        .reset_index()
        .rename(columns={"ingresos_plot": "ingresos_medios_M€"})
        .sort_values("ingresos_medios_M€", ascending=False)
    )
    st.dataframe(resumen, use_container_width=True)


# =====================================================================
# 2. Cuotas de mercado y concentración (HHI)
# =====================================================================

st.header("2. Cuotas de mercado y concentración (HHI)")

df_hhi = df_ing.copy()
years_hhi = sorted(df_hhi["anio"].dropna().astype(int).unique())

if not years_hhi:
    st.warning("No hay años disponibles para calcular cuotas y HHI.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        anio_hhi = st.selectbox(
            "Año de referencia",
            years_hhi,
            index=len(years_hhi) - 1,
            key="hhi_anio",
        )
    with c2:
        mercados_hhi = sorted(df_hhi["tipo_de_mercado"].dropna().unique()) \
            if "tipo_de_mercado" in df_hhi.columns else []
        mercado_hhi = st.selectbox(
            "Tipo de mercado (HHI)",
            ["(todos)"] + mercados_hhi,
            index=0,
            key="hhi_mercado",
        )
    with c3:
        servicios_hhi = sorted(df_hhi["servicio"].dropna().unique()) \
            if "servicio" in df_hhi.columns else []
        servicio_hhi = st.selectbox(
            "Servicio / mercado (HHI)",
            ["(todos)"] + servicios_hhi,
            index=0,
            key="hhi_servicio",
        )

    mask = df_hhi["anio"] == anio_hhi
    if mercado_hhi != "(todos)" and "tipo_de_mercado" in df_hhi.columns:
        mask &= df_hhi["tipo_de_mercado"] == mercado_hhi
    if servicio_hhi != "(todos)" and "servicio" in df_hhi.columns:
        mask &= df_hhi["servicio"] == servicio_hhi

    df_hhi_y = df_hhi[mask].copy()

    if df_hhi_y.empty:
        st.warning("No hay datos para esa combinación de año / mercado / servicio.")
    else:
        if "ingresos_por_operador" in df_hhi_y.columns:
            df_hhi_y = (
                df_hhi_y.groupby("operador", dropna=True)["ingresos_por_operador"]
                .sum()
                .reset_index()
            )
            df_hhi_y = df_hhi_y[df_hhi_y["ingresos_por_operador"] > 0]
            total = df_hhi_y["ingresos_por_operador"].sum()
            df_hhi_y["cuota"] = df_hhi_y["ingresos_por_operador"] / total
        else:
            st.warning(
                "No existe la columna 'ingresos_por_operador'; se usará 'ingresos'."
            )
            df_hhi_y = (
                df_hhi_y.groupby("operador", dropna=True)["ingresos"]
                .sum()
                .reset_index()
            )
            df_hhi_y = df_hhi_y[df_hhi_y["ingresos"] > 0]
            total = df_hhi_y["ingresos"].sum()
            df_hhi_y["cuota"] = df_hhi_y["ingresos"] / total

        df_hhi_y["cuota_%"] = df_hhi_y["cuota"] * 100

        hhi = compute_hhi(df_hhi_y["cuota"])
        if hhi < 1500:
            grado = "Baja concentración"
        elif hhi < 2500:
            grado = "Concentración moderada"
        else:
            grado = "Alta concentración"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("HHI (0–10.000)", f"{hhi:,.0f}")
        with c2:
            st.metric("Número de operadores", f"{len(df_hhi_y)}")
        with c3:
            st.metric("Grado de concentración", grado)

        df_hhi_plot = df_hhi_y.sort_values("cuota_%", ascending=False).reset_index(drop=True)

        fig_cuotas = px.bar(
            df_hhi_plot,
            x="operador",
            y="cuota_%",
            color="operador",
            color_discrete_map=COLOR_OPERADORES,
            text=df_hhi_plot["cuota_%"].map(lambda x: f"{x:.1f}%"),
            labels={
                "operador": "Operador",
                "cuota_%": "Cuota de mercado (% ingresos)",
            },
            title=f"Cuotas de mercado por ingresos – {anio_hhi} (datos RAW)",
        )
        fig_cuotas.update_traces(textposition="outside")
        fig_cuotas.update_yaxes(range=[0, max(df_hhi_plot["cuota_%"]) * 1.2])
        st.plotly_chart(fig_cuotas, use_container_width=True)

        # NUEVA GRÁFICA: NÚMERO TOTAL DE OPERADORES POR AÑO
        st.markdown("**Número total de operadores activos por año (ingresos > 0)**")

        df_ops = df_hhi.copy()
        if "ingresos_por_operador" in df_ops.columns:
            metric_col = "ingresos_por_operador"
        else:
            metric_col = "ingresos" if "ingresos" in df_ops.columns else None

        if metric_col is not None and "operador" in df_ops.columns and "anio" in df_ops.columns:
            df_ops_agg = (
                df_ops.groupby(["anio", "operador"], as_index=False)[metric_col]
                .sum()
            )
            df_ops_agg = df_ops_agg[df_ops_agg[metric_col] > 0]

            df_ops_year = (
                df_ops_agg.groupby("anio")["operador"]
                .nunique()
                .reset_index(name="num_operadores")
                .sort_values("anio")
            )

            fig_ops = px.line(
                df_ops_year,
                x="anio",
                y="num_operadores",
                markers=False,
                labels={
                    "anio": "Año",
                    "num_operadores": "Número de operadores con ingresos",
                },
                title="Evolución del número total de operadores activos por año",
            )
            fig_ops.update_traces(mode="lines", line=dict(width=3))
            st.plotly_chart(fig_ops, use_container_width=True)
        else:
            st.info(
                "No se ha podido calcular el número de operadores por año "
                "porque faltan columnas de ingresos u operador."
            )


# =====================================================================
# 3. Distribución territorial (mapas y heatmap)
# =====================================================================

st.header("3. Distribución territorial: penetración y volumen por provincia")

if prov.empty:
    st.warning("No se ha podido cargar el dataset provincial.")
else:
    df_prov = prov.copy()

    if "anio" not in df_prov.columns and "anno" in df_prov.columns:
        df_prov["anio"] = df_prov["anno"]

    if "provincia" not in df_prov.columns or df_prov["provincia"].isna().all():
        st.warning("La columna 'provincia' está vacía en el dataset provincial.")
    else:
        df_prov["provincia_key"] = df_prov["provincia"].replace(PROV_NAME_MAP)

        if "anio" in df_prov.columns:
            years_prov = sorted(df_prov["anio"].dropna().astype(int).unique())
            selected_year = st.selectbox(
                "Año a mostrar en los mapas provinciales",
                years_prov,
                index=len(years_prov) - 1,
                key="prov_map_year",
            ) if years_prov else None
        else:
            selected_year = None

        # --- 3.1 Mapa de tasa de penetración ---
        if "tasa_de_penetracion" not in df_prov.columns:
            st.warning("El dataset provincial no contiene 'tasa_de_penetracion'.")
        else:
            st.subheader("3.1 Mapa de tasa de penetración por provincia")

            df_pen = df_prov[df_prov["tasa_de_penetracion"].notna()].copy()
            if selected_year is not None:
                df_pen = df_pen[df_pen["anio"] == selected_year]

            if df_pen.empty:
                st.info("No hay datos de tasa de penetración para el año seleccionado.")
            else:
                pen_grp = (
                    df_pen.groupby(["provincia", "provincia_key"], as_index=False)["tasa_de_penetracion"]
                    .mean()
                )

                if GEO_PROV_PATH.exists():
                    with open(GEO_PROV_PATH, "r", encoding="utf-8") as f:
                        geojson_prov = json.load(f)

                    fig_map_pen = px.choropleth(
                        pen_grp,
                        geojson=geojson_prov,
                        locations="provincia_key",
                        featureidkey="properties.provincia",
                        color="tasa_de_penetracion",
                        hover_name="provincia",
                        color_continuous_scale="Viridis",
                        labels={
                            "tasa_de_penetracion": "Tasa de penetración (líneas / 100 hab.)",
                        },
                        title=(
                            f"Tasa de penetración por provincia – {selected_year}"
                            if selected_year else
                            "Tasa de penetración por provincia"
                        ),
                    )
                    fig_map_pen.update_geos(fitbounds="locations", visible=False)
                    st.plotly_chart(fig_map_pen, use_container_width=True)
                else:
                    st.info(
                        "No se ha encontrado `data/geo/provincias_es.geojson`. "
                        "Se muestra ranking en barras en lugar de mapa."
                    )
                    pen_sorted = pen_grp.sort_values(
                        "tasa_de_penetracion", ascending=False
                    )
                    fig_pen_bar = px.bar(
                        pen_sorted,
                        x="provincia",
                        y="tasa_de_penetracion",
                        labels={
                            "provincia": "Provincia",
                            "tasa_de_penetracion": "Tasa de penetración (líneas / 100 hab.)",
                        },
                        title=(
                            f"Tasa de penetración por provincia – {selected_year}"
                            if selected_year else
                            "Tasa de penetración por provincia"
                        ),
                    )
                    fig_pen_bar.update_xaxes(tickangle=60)
                    st.plotly_chart(fig_pen_bar, use_container_width=True)

                st.markdown("**Top 5 provincias por penetración**")
                st.dataframe(
                    pen_grp.sort_values("tasa_de_penetracion", ascending=False)
                    .head(5)[["provincia", "tasa_de_penetracion"]],
                    use_container_width=True,
                )

        # --- 3.2 Mapa de volumen global (no tasas) ---
        st.subheader("3.2 Mapa de volumen total por provincia (no tasa)")

        df_vol = df_prov.copy()
        if selected_year is not None and "anio" in df_vol.columns:
            df_vol = df_vol[df_vol["anio"] == selected_year]

        if df_vol.empty:
            st.info("No hay datos provinciales para el año seleccionado.")
        else:
            cand_vol_cols = [
                "lineas_o_accesos",
                "unidades",
                "estaciones_base",
            ]
            vol_col = None
            for c in cand_vol_cols:
                if c in df_vol.columns and df_vol[c].notna().sum() > 0:
                    vol_col = c
                    break
            if vol_col is None:
                num_cols = [
                    c for c in df_vol.columns
                    if df_vol[c].dtype != "object"
                    and c not in ["anio", "tasa_de_penetracion", "id", "anno"]
                ]
                vol_col = num_cols[0] if num_cols else None

            if vol_col is None:
                st.info("No se ha encontrado ningún indicador numérico de volumen para el mapa.")
            else:
                vol_grp = (
                    df_vol.groupby(["provincia", "provincia_key"], as_index=False)[vol_col]
                    .sum()
                )
                if GEO_PROV_PATH.exists():
                    with open(GEO_PROV_PATH, "r", encoding="utf-8") as f:
                        geojson_prov = json.load(f)

                    fig_map_vol = px.choropleth(
                        vol_grp,
                        geojson=geojson_prov,
                        locations="provincia_key",
                        featureidkey="properties.provincia",
                        color=vol_col,
                        hover_name="provincia",
                        color_continuous_scale="Blues",
                        labels={vol_col: vol_col.replace("_", " ").title()},
                        title=(
                            f"Volumen total por provincia – {vol_col.replace('_',' ').title()} ({selected_year})"
                            if selected_year else
                            f"Volumen total por provincia – {vol_col.replace('_',' ').title()}"
                        ),
                    )
                    fig_map_vol.update_geos(fitbounds="locations", visible=False)
                    st.plotly_chart(fig_map_vol, use_container_width=True)
                else:
                    st.info(
                        "No se ha encontrado `data/geo/provincias_es.geojson`. "
                        "Se muestra ranking en barras en lugar de mapa."
                    )
                    vol_sorted = vol_grp.sort_values(vol_col, ascending=False)
                    fig_vol_bar = px.bar(
                        vol_sorted,
                        x="provincia",
                        y=vol_col,
                        labels={
                            "provincia": "Provincia",
                            vol_col: vol_col.replace("_", " ").title(),
                        },
                        title=(
                            f"Volumen total por provincia – {vol_col.replace('_',' ').title()} ({selected_year})"
                            if selected_year else
                            f"Volumen total por provincia – {vol_col.replace('_',' ').title()}"
                        ),
                    )
                    fig_vol_bar.update_xaxes(tickangle=60)
                    st.plotly_chart(fig_vol_bar, use_container_width=True)

                st.markdown("**Top 5 provincias por volumen**")
                st.dataframe(
                    vol_grp.sort_values(vol_col, ascending=False)
                    .head(5)[["provincia", vol_col]],
                    use_container_width=True,
                )

        # --- 3.3 Heatmap histórico de penetración por provincia ---
        st.subheader("3.3 Evolución histórica de la penetración por provincia")

        if "tasa_de_penetracion" in df_prov.columns and "anio" in df_prov.columns:
            df_pen2 = df_prov[df_prov["tasa_de_penetracion"].notna()].copy()
            if not df_pen2.empty:
                tabla = df_pen2.pivot_table(
                    index="provincia",
                    columns="anio",
                    values="tasa_de_penetracion",
                    aggfunc="mean",
                )
                tabla = tabla.sort_index(axis=1)

                fig_heat = px.imshow(
                    tabla,
                    aspect="auto",
                    labels={
                        "x": "Año",
                        "y": "Provincia",
                        "color": "Tasa de penetración (líneas / 100 hab.)",
                    },
                    title="Mapa de calor de penetración por provincia y año",
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                st.markdown(
                    "Cada fila es una provincia y cada columna un año, con su tasa media de penetración."
                )
            else:
                st.info("No hay datos suficientes para construir el heatmap provincial.")
        else:
            st.info("Faltan columnas 'tasa_de_penetracion' o 'anio' para el heatmap.")


# =====================================================================
# 4. Visión táctica mensual y trimestral por servicio (USANDO RAW)
# =====================================================================

st.header("4. Visión táctica mensual y trimestral por servicio (datos RAW)")

tab_mensual, tab_trimestral = st.tabs(["Mensual (táctica)", "Trimestral (estructura)"])

# ----------------- 4.1 MENSUAL – visión táctica (RAW) -----------------
with tab_mensual:
    st.subheader("4.1 Evolución mensual por servicio y operador (datos raw CNMC)")

    if mensual_raw.empty:
        st.warning("No se ha podido cargar el dataset mensual RAW (data/raw/mensual.csv).")
    else:
        df_m = mensual_raw.copy()

        if "fecha" in df_m.columns:
            df_m["fecha"] = pd.to_datetime(df_m["fecha"], errors="coerce")
        elif "mes" in df_m.columns and "anio" in df_m.columns:
            df_m["fecha"] = ensure_month_date(df_m["mes"], df_m["anio"])
        elif "mes" in df_m.columns:
            df_m["fecha"] = pd.to_datetime(df_m["mes"], errors="coerce")
        else:
            df_m["fecha"] = pd.NaT

        df_m = df_m[df_m["fecha"].notna()]

        if df_m.empty or "servicio" not in df_m.columns:
            st.info("No hay información suficiente (fecha/servicio) en el dataset mensual RAW.")
        else:
            cand_indic_m = ["ingresos", "lineas", "líneas", "portabilidades", "unidades"]
            indicador_m = None
            for c in cand_indic_m:
                if c in df_m.columns and df_m[c].notna().sum() > 0:
                    indicador_m = c
                    break
            if indicador_m is None:
                num_cols_m = [
                    c for c in df_m.columns
                    if df_m[c].dtype != "object"
                    and c not in ["anio", "mes", "fecha", "id"]
                ]
                indicador_m = num_cols_m[0] if num_cols_m else None

            if indicador_m is None:
                st.info("No se ha encontrado ningún indicador numérico mensual para graficar (RAW).")
            else:
                servicios_m = sorted(df_m["servicio"].dropna().unique())
                col1, col2 = st.columns([2, 1])
                with col1:
                    servicio_m_sel = st.selectbox(
                        "Servicio principal para visión táctica (RAW)",
                        ["(todos)"] + servicios_m,
                        index=0,
                        key="mens_servicio_sel_raw",
                    )
                with col2:
                    st.write("")
                    st.write(f"Indicador usado: **{indicador_m.replace('_',' ').title()}**")

                df_m_srv = (
                    df_m.groupby(["fecha", "servicio"], as_index=False)[indicador_m]
                    .sum()
                    .sort_values("fecha")
                )

                if servicio_m_sel == "(todos)":
                    st.markdown("**Evolución mensual por servicio (visión global del mercado – RAW)**")
                    fig_m_all = px.line(
                        df_m_srv,
                        x="fecha",
                        y=indicador_m,
                        color="servicio",
                        markers=False,
                        labels={
                            "fecha": "Mes",
                            "servicio": "Servicio",
                            indicador_m: indicador_m.replace("_", " ").title(),
                        },
                        title=f"Evolución mensual del mercado por servicio – {indicador_m.replace('_',' ').title()} (raw)",
                    )
                    fig_m_all.update_traces(mode="lines", line=dict(width=3))
                    st.plotly_chart(fig_m_all, use_container_width=True)

                    st.markdown("**Servicios con mayor crecimiento en los últimos 12 meses (RAW)**")
                    last_dates = sorted(df_m_srv["fecha"].unique())[-12:]
                    df_last12 = df_m_srv[df_m_srv["fecha"].isin(last_dates)].copy()
                    resumen_12m = (
                        df_last12.groupby("servicio")[indicador_m]
                        .agg(["first", "last"])
                        .rename(columns={"first": "valor_inicio", "last": "valor_fin"})
                    )
                    resumen_12m["var_abs"] = resumen_12m["valor_fin"] - resumen_12m["valor_inicio"]
                    resumen_12m["var_%"] = (
                        (resumen_12m["var_abs"] / resumen_12m["valor_inicio"].replace({0: np.nan})) * 100
                    )
                    resumen_12m = resumen_12m.sort_values("var_%", ascending=False).reset_index()

                    fig_rank = px.bar(
                        resumen_12m,
                        x="servicio",
                        y="var_%",
                        labels={
                            "servicio": "Servicio",
                            "var_%": "Crecimiento % últimos 12 meses",
                        },
                        title="Ranking de servicios por crecimiento % en los últimos 12 meses (RAW)",
                    )
                    fig_rank.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_rank, use_container_width=True)
                    st.dataframe(resumen_12m, use_container_width=True)

                else:
                    st.markdown(
                        f"**Evolución mensual del servicio {servicio_m_sel} por operador (RAW)**"
                    )
                    df_m_serv = df_m[df_m["servicio"] == servicio_m_sel].copy()
                    if "operador" not in df_m_serv.columns or df_m_serv["operador"].isna().all():
                        st.info(
                            "No hay desglose por operador para este servicio en el dataset mensual RAW. "
                            "Se muestra sólo la serie agregada."
                        )
                        df_m_agg = (
                            df_m_serv.groupby("fecha", as_index=False)[indicador_m]
                            .sum()
                            .sort_values("fecha")
                        )
                        fig_m_serv = px.line(
                            df_m_agg,
                            x="fecha",
                            y=indicador_m,
                            markers=False,
                            labels={
                                "fecha": "Mes",
                                indicador_m: indicador_m.replace("_", " ").title(),
                            },
                            title=f"Evolución mensual – {servicio_m_sel} (raw)",
                        )
                        fig_m_serv.update_traces(mode="lines", line=dict(width=3))
                        st.plotly_chart(fig_m_serv, use_container_width=True)
                    else:
                        df_m_serv_agg = (
                            df_m_serv.groupby(["fecha", "operador"], as_index=False)[indicador_m]
                            .sum()
                            .sort_values("fecha")
                        )
                        fig_m_serv = px.line(
                            df_m_serv_agg,
                            x="fecha",
                            y=indicador_m,
                            color="operador",
                            color_discrete_map=COLOR_OPERADORES,
                            markers=False,
                            labels={
                                "fecha": "Mes",
                                "operador": "Operador",
                                indicador_m: indicador_m.replace("_", " ").title(),
                            },
                            title=f"Evolución mensual – {servicio_m_sel} por operador (raw)",
                        )
                        fig_m_serv.update_traces(mode="lines", line=dict(width=3))
                        st.plotly_chart(fig_m_serv, use_container_width=True)

                        st.markdown("**Últimos 12 meses por operador (detalle – RAW)**")
                        last_dates_serv = sorted(df_m_serv_agg["fecha"].unique())[-12:]
                        df_last12_serv = df_m_serv_agg[df_m_serv_agg["fecha"].isin(last_dates_serv)].copy()
                        st.dataframe(
                            df_last12_serv.pivot_table(
                                index="operador",
                                columns="fecha",
                                values=indicador_m,
                                aggfunc="sum",
                            ),
                            use_container_width=True,
                        )

                st.markdown("**Visión táctica del último mes (mercado total – RAW)**")
                df_total = (
                    df_m.groupby("fecha", as_index=False)[indicador_m]
                    .sum()
                    .sort_values("fecha")
                )
                if len(df_total) >= 2:
                    ultimo = df_total.iloc[-1]
                    penultimo = df_total.iloc[-2]
                    var_abs = ultimo[indicador_m] - penultimo[indicador_m]
                    var_pct = (
                        var_abs / penultimo[indicador_m] * 100
                        if penultimo[indicador_m] != 0
                        else np.nan
                    )
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(
                            "Último mes",
                            f"{ultimo[indicador_m]:,.0f}",
                            delta=None,
                        )
                    with c2:
                        st.metric(
                            "Variación absoluta vs mes anterior",
                            f"{var_abs:,.0f}",
                        )
                    with c3:
                        st.metric(
                            "Variación % vs mes anterior",
                            f"{var_pct:,.1f} %",
                        )


# ----------------- 4.2 TRIMESTRAL – visión estructural (RAW) -----------------
with tab_trimestral:
    st.subheader("4.2 Evolución trimestral de ingresos por servicio (datos raw CNMC)")

    if trimestral_raw.empty:
        st.warning("No se ha podido cargar el dataset trimestrales RAW (data/raw/trimestrales.csv).")
    else:
        df_t = trimestral_raw.copy()
        if "trimestre" not in df_t.columns:
            st.info("El dataset trimestrales RAW no contiene la columna 'trimestre'.")
        elif "ingresos" not in df_t.columns:
            st.info("El dataset trimestrales RAW no contiene la columna 'ingresos'.")
        else:
            if "servicio" not in df_t.columns:
                df_t_agg = (
                    df_t.groupby("trimestre", as_index=False)["ingresos"]
                    .sum()
                    .sort_values("trimestre")
                )
                fig_t = px.line(
                    df_t_agg,
                    x="trimestre",
                    y="ingresos",
                    markers=False,
                    labels={
                        "trimestre": "Trimestre",
                        "ingresos": "Ingresos (M€)",
                    },
                    title="Ingresos trimestrales – mercado total (raw)",
                )
                fig_t.update_traces(mode="lines", line=dict(width=3))
                st.plotly_chart(fig_t, use_container_width=True)

                df_t_agg["var_abs"] = df_t_agg["ingresos"].diff()
                df_t_agg["var_%"] = df_t_agg["ingresos"].pct_change() * 100
                st.markdown("**Últimos 8 trimestres (detalle – RAW)**")
                st.dataframe(
                    df_t_agg.tail(8).set_index("trimestre"),
                    use_container_width=True,
                )
            else:
                df_t_agg = (
                    df_t.groupby(["trimestre", "servicio"], as_index=False)["ingresos"]
                    .sum()
                    .sort_values("trimestre")
                )

                fig_t = px.line(
                    df_t_agg,
                    x="trimestre",
                    y="ingresos",
                    color="servicio",
                    markers=False,
                    labels={
                        "trimestre": "Trimestre",
                        "ingresos": "Ingresos (M€)",
                        "servicio": "Servicio",
                    },
                    title="Ingresos trimestrales por servicio (raw)",
                )
                fig_t.update_traces(mode="lines", line=dict(width=3))
                st.plotly_chart(fig_t, use_container_width=True)

                df_t_agg["var_abs"] = (
                    df_t_agg.groupby("servicio")["ingresos"].diff()
                )
                df_t_agg["var_%"] = (
                    df_t_agg.groupby("servicio")["ingresos"].pct_change() * 100
                )

                st.markdown("**Últimos 8 trimestres por servicio (detalle – RAW)**")
                st.dataframe(
                    df_t_agg.tail(8).set_index("trimestre"),
                    use_container_width=True,
                )

                st.markdown("**Crecimiento interanual (YoY) en el último trimestre disponible (RAW)**")
                df_t_agg["anio_trim"] = df_t_agg["trimestre"].astype(str).str.extract(r"^(\d{4})")[0]
                df_t_agg["n_trim"] = df_t_agg["trimestre"].astype(str).str.extract(r"T(\d)")[0]
                if df_t_agg["anio_trim"].notna().all() and df_t_agg["n_trim"].notna().all():
                    df_t_agg["anio_trim"] = df_t_agg["anio_trim"].astype(int)
                    df_t_agg["n_trim"] = df_t_agg["n_trim"].astype(int)

                    last_year = df_t_agg["anio_trim"].max()
                    last_quarter = df_t_agg[df_t_agg["anio_trim"] == last_year]["n_trim"].max()

                    df_last_q = df_t_agg[
                        (df_t_agg["anio_trim"] == last_year) &
                        (df_t_agg["n_trim"] == last_quarter)
                    ].copy()

                    df_prev_q = df_t_agg[
                        (df_t_agg["anio_trim"] == last_year - 1) &
                        (df_t_agg["n_trim"] == last_quarter)
                    ].copy()

                    if not df_last_q.empty and not df_prev_q.empty:
                        df_yoy = df_last_q.merge(
                            df_prev_q[["servicio", "ingresos"]],
                            on="servicio",
                            how="left",
                            suffixes=("_actual", "_previo"),
                        )
                        df_yoy["var_abs"] = df_yoy["ingresos_actual"] - df_yoy["ingresos_previo"]
                        df_yoy["var_%"] = (
                            df_yoy["var_abs"] /
                            df_yoy["ingresos_previo"].replace({0: np.nan}) * 100
                        )

                        fig_yoy = px.bar(
                            df_yoy,
                            x="servicio",
                            y="var_%",
                            labels={
                                "servicio": "Servicio",
                                "var_%": "Crecimiento interanual (%)",
                            },
                            title=f"Crecimiento interanual por servicio – T{last_quarter} {last_year} (raw)",
                        )
                        fig_yoy.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_yoy, use_container_width=True)

                        st.dataframe(df_yoy, use_container_width=True)
                    else:
                        st.info(
                            "No hay suficientes datos para calcular el crecimiento interanual "
                            "del último trimestre (raw)."
                        )
                else:
                    st.info(
                        "No se ha podido interpretar el formato de 'trimestre' para el cálculo interanual (raw)."
                    )


# =====================================================================
# 5. Panel de “sanidad” de los datasets (opcional)
# =====================================================================

with st.expander("Ver resumen rápido de calidad de datos", expanded=False):
    def quick_quality(df: pd.DataFrame, nombre: str):
        if df is None or df.empty:
            st.write(f"**{nombre}**: sin datos")
            return
        n_rows = len(df)
        n_cols = df.shape[1]
        n_nulls = int(df.isna().sum().sum())
        n_dups = int(df.duplicated().sum())
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(f"{nombre}: filas", f"{n_rows:,}")
        with c2:
            st.metric(f"{nombre}: columnas", f"{n_cols}")
        with c3:
            st.metric(f"{nombre}: nulos", f"{n_nulls:,}")
        with c4:
            st.metric(f"{nombre}: duplicados", f"{n_dups:,}")

    quick_quality(anual_dg,          "Anual – Datos generales (merged)")
    quick_quality(anual_merc_merged, "Anual – Mercados (merged)")
    quick_quality(anual_merc_raw,    "Anual – Mercados (RAW)")
    quick_quality(prov,              "Provinciales (merged)")
    quick_quality(mensual_raw,       "Mensual – RAW")
    quick_quality(trimestral_raw,    "Trimestrales – RAW")
    quick_quality(load_merged_dataset("infraestructuras"), "Infraestructuras (merged)")
