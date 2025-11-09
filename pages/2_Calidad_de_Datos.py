# pages/2_Calidad_de_Datos.py
import pandas as pd
import streamlit as st
from utils.data_quality import run_quality_suite

st.set_page_config(page_title="Calidad de Datos – DSS CNMC", layout="wide", page_icon="🧪")
st.header("Calidad de Datos (RAW / CLEAN / FINAL)")

# -----------------------------
# Selector de origen y rutas
# -----------------------------
origen = st.radio(
    "Origen de datos",
    ["RAW (data/raw)", "CLEAN (data/clean)", "FINAL (data/final)"],
    horizontal=True,
)

if origen.startswith("RAW"):
    base = "data/raw/"
    datasets = {
        "Anual – Datos generales": base + "anual_datos_generales.csv",
        "Anual – Mercados":       base + "anual_mercados.csv",
        "Mensual":                base + "mensual.csv",
        "Provinciales":           base + "provinciales.csv",
        "Trimestrales":           base + "trimestrales.csv",
        "Infraestructuras":       base + "infraestructuras.csv",
    }
elif origen.startswith("CLEAN"):
    base = "data/clean/"
    datasets = {
        "Anual – Datos generales": base + "anual_datos_generales_clean.csv",
        "Anual – Mercados":       base + "anual_mercados_clean.csv",
        "Mensual":                base + "mensual_clean.csv",
        "Provinciales":           base + "provinciales_clean.csv",
        "Trimestrales":           base + "trimestrales_clean.csv",
        "Infraestructuras":       base + "infraestructuras_clean.csv",
    }
else:  # FINAL
    base = "data/final/"
    datasets = {
        "Anual – Datos generales": base + "anual_datos_generales_final.csv",
        "Anual – Mercados":       base + "anual_mercados_final.csv",
        "Mensual":                base + "mensual_final.csv",
        "Provinciales":           base + "provinciales_final.csv",
        "Trimestrales":           base + "trimestrales_final.csv",
        "Infraestructuras":       base + "infraestructuras_final.csv",
    }

name = st.selectbox("Elige dataset", list(datasets.keys()))
path = datasets[name]

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = None
try:
    df = load_csv(path)
    st.success(f"{name}: {df.shape[0]:,} filas × {df.shape[1]} columnas")
except Exception as e:
    st.error(f"No se pudo cargar {path}: {e}")

# ----------------------------------
# Vista previa + ejecución de suite
# ----------------------------------
if df is not None:
    with st.expander(" Vista previa (primeras 50 filas)", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)

    # Pista para reforzar reglas por tipo de dataset
    hint = (
        "anual" if "Anual" in name else
        "mensual" if "Mensual" in name else
        "trimestral" if "Trimestrales" in name else
        "provincial" if "Provinciales" in name else
        "infraestructuras" if "Infraestructuras" in name else
        None
    )

    if st.button("▶️ Ejecutar suite de calidad", use_container_width=True):
        with st.spinner("Evaluando calidad…"):
            rep = run_quality_suite(df, dataset_hint=hint)

        score = rep.attrs.get("quality_score", 100.0)
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Quality Score", f"{score:.2f}/100")
            st.caption(f"Fuente: **{origen.split()[0]}** · Archivo: `{path}`")
        with c2:
            st.write("**Resumen de incidencias por prueba** (menor es mejor):")
            st.dataframe(
                rep.sort_values(["dimension", "pct_rows"], ascending=[True, False]),
                use_container_width=True,
                height=440
            )

        st.download_button(
            "⬇️ Descargar informe (CSV)",
            data=rep.to_csv(index=False).encode("utf-8"),
            file_name=f"data_quality_{origen.split()[0]}_{name.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        with st.expander(" Guía de interpretación"):
            st.markdown("""
- **Completitud**: columnas con mayor % de nulos → decidir imputación o descarte.
- **Validez**: negativos en métricas que no los admiten; rangos temporales anómalos.
- **Consistencia**: cuotas por `periodo-mercado` ≈ 100% cuando aplique.
- **Unicidad**: duplicados por claves semánticas (p. ej., `periodo-operador-servicio`).
- **Integridad**: provincias/tecnologías fuera de catálogo → normalizar.
- **Exactitud**: outliers (IQR) y saltos temporales (|z|>3) → revisar casos.
- **Actualidad**: último `periodo` disponible; lag respecto a hoy.
            """)
