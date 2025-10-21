import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from utils.mcdm import topsis_rank

st.set_page_config(page_title="DSS Telecom CNMC", layout="wide", page_icon="📶")
st.title("📶 DSS Telecomunicaciones – CNMC (prototipo)")

st.sidebar.header("Fuente de datos")
modo = st.sidebar.radio("Selecciona", ["CSV (repositorio)", "A futuro: API CNMC"], index=0)

DATASETS = {
    "Anual – Datos generales": "data/clean/anual_datos_generales_clean.csv",
    "Anual – Mercados": "data/clean/anual_mercados_clean.csv",
    "Mensual": "data/clean/mensual_clean.csv",
    "Provinciales": "data/clean/provinciales_clean.csv",
    "Trimestrales": "data/clean/trimestrales_clean.csv",
    "Infraestructuras": "data/clean/infraestructuras_clean.csv",
}

@st.cache_data(show_spinner=False)
def load_csv(path: str):
    return pd.read_csv(path)

dataset_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
path = DATASETS[dataset_name]

df = None
if modo.startswith("CSV"):
    try:
        df = load_csv(path)
        st.success(f"{dataset_name}: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    except Exception as e:
        st.error(f"No se pudo cargar {path}. Ejecuta antes scripts/download_cnmc.py y sube los CSV. Detalle: {e}")
else:
    st.info("La carga directa por API se activará en la siguiente iteración.")

if df is not None:
    with st.expander("Vista previa de datos"):
        st.dataframe(df.head(50), use_container_width=True)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    st.header("⚖️ Multicriterio (TOPSIS)")
    crit = st.multiselect("Elige criterios numéricos", num_cols, default=num_cols[:3] if len(num_cols)>=3 else num_cols)
    if crit:
        benefit_flags, weights = [], []
        st.subheader("Pesos y tipo de criterio")
        for c in crit:
            cols = st.columns([2,1,2])
            with cols[0]: st.write(f"**{c}**")
            with cols[1]: benefit_flags.append(st.toggle("Beneficio", True, key=f"b_{c}"))
            with cols[2]: weights.append(st.slider(f"Peso {c}", 0.0, 1.0, 1.0/len(crit), 0.01, key=f"w_{c}"))
        try:
            rank = topsis_rank(df, crit, weights, benefit_flags)
            st.subheader("Ranking TOPSIS")
            st.dataframe(rank.head(50), use_container_width=True)
            fig = px.histogram(rank, x="score_topsis", nbins=40, title="Distribución del score TOPSIS")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo calcular TOPSIS: {e}")

    st.header("📈 Exploración rápida")
    if len(num_cols) >= 2:
        xcol = st.selectbox("Eje X", num_cols, index=0)
        ycol = st.selectbox("Eje Y", num_cols, index=1)
        fig = px.scatter(df, x=xcol, y=ycol, title=f"{ycol} vs {xcol}")
        st.plotly_chart(fig, use_container_width=True)

st.caption("Prototipo base. Próximos pasos: mapas, unión por provincia/fecha, proyecciones e informes en páginas.")
