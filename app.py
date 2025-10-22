# app.py — DSS Telecom CNMC (ETL + persistencia + MCDM + exploración)
import io
import zipfile
import base64
import datetime as dt
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import streamlit as st

# Utils del proyecto
from utils.mcdm import topsis_rank
from utils.cnmc_ckan import fetch_resource
from utils.data_prep import (
    unify_columns_lower, clean_strings, coerce_numeric,
    build_period_column, drop_dupes_and_aggregate,
    group_small_ops, normalize_minmax
)

# ------------------------------------------------------------
# Config general
# ------------------------------------------------------------
st.set_page_config(page_title="DSS Telecomunicaciones – CNMC", layout="wide", page_icon="📶")
st.title("📶 DSS Telecomunicaciones – CNMC")

# Recursos CKAN a “congelar” como CSV limpios
RESOURCES = {
    "anual_datos_generales": "5e2d8f37-2385-4774-82ec-365cd83d65bd",
    "anual_mercados": "7afbf769-655d-4b43-b49f-95c2919ec1fe",
    "mensual": "3632297f-07d8-480c-aca5-c987dcde0ccb",
    "provinciales": "1efe6d64-72a8-4f45-a36c-691054f3e277",
    "trimestrales": "5da45f2f-e596-4940-b682-eab18e85288a",
    "infraestructuras": "baab2a5e-cc52-4704-a799-a28b19223a3b",
}

# Rutas de lectura de CSV “congelados” en el repo
CSV_PATHS = {
    "Anual – Datos generales": "data/clean/anual_datos_generales_clean.csv",
    "Anual – Mercados": "data/clean/anual_mercados_clean.csv",
    "Mensual": "data/clean/mensual_clean.csv",
    "Provinciales": "data/clean/provinciales_clean.csv",
    "Trimestrales": "data/clean/trimestrales_clean.csv",
    "Infraestructuras": "data/clean/infraestructuras_clean.csv",
}

# ------------------------------------------------------------
# ETL: limpieza robusta (misma lógica que el script)
# ------------------------------------------------------------
TEXT_COLS_CANDIDATES = [
    "operador","servicio","concepto","tipo_de_paquete","tipo_de_ingreso",
    "provincia","ccaa","tecnologia_de_acceso","tipo_de_ba_mayorista",
    "tipo_de_estaciones_base","unidades"
]

def basic_clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza columnas y texto, convierte a numérico, crea 'periodo',
       colapsa duplicados por llaves típicas y agrupa operadores pequeños en 'Otros'."""
    df = unify_columns_lower(df)
    df = clean_strings(df, [c for c in TEXT_COLS_CANDIDATES if c in df.columns])
    df = coerce_numeric(df, prefer_comma_decimal=True)
    df = build_period_column(df)

    candidate_keys = ["periodo","operador","servicio","provincia","ccaa","tecnologia_de_acceso","concepto"]
    keys = [k for k in candidate_keys if k in df.columns]
    if keys:
        df = drop_dupes_and_aggregate(df, keys=keys)

    if "operador" in df.columns:
        df = group_small_ops(df, top_n=5, col_op="operador")

    return df

def download_and_clean_all() -> dict[str, bytes]:
    """Descarga y limpia todos los recursos. Devuelve {nombre_csv: bytes_csv}."""
    out: dict[str, bytes] = {}
    for name, rid in RESOURCES.items():
        df = fetch_resource(rid)
        dfc = basic_clean_pipeline(df)
        out[f"{name}_clean.csv"] = dfc.to_csv(index=False).encode("utf-8")
    return out

def make_zip(files_dict: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, content in files_dict.items():
            zf.writestr(fname, content)
    return buf.getvalue()

# ------------------------------------------------------------
# Guardado automático en GitHub (opcional, via secrets)
# ------------------------------------------------------------
def github_put_file(owner, repo, branch, path, content_bytes, token):
    """Crea/actualiza archivo en GitHub (PUT /repos/:owner/:repo/contents/:path)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

    # comprobar si existe para obtener sha
    r = requests.get(url, params={"ref": branch}, headers={"Authorization": f"Bearer {token}"})
    sha = r.json().get("sha") if r.status_code == 200 else None

    payload = {
        "message": f"Update {path}",
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers={"Authorization": f"Bearer {token}"}, json=payload)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub PUT failed for {path}: {r.status_code} {r.text}")

def push_all_to_github(files_dict: dict[str, bytes], subdir: str = "data/clean/"):
    """Empuja todos los CSV a tu repo usando secretos en Streamlit Cloud."""
    owner_repo = st.secrets["GITHUB_REPO"]            # ej: "usuario/DSS-TFM-"
    branch     = st.secrets.get("GITHUB_BRANCH", "main")
    token      = st.secrets["GITHUB_TOKEN"]
    owner, repo = owner_repo.split("/", 1)
    for fname, content in files_dict.items():
        github_put_file(owner, repo, branch, f"{subdir}{fname}", content, token)

# ------------------------------------------------------------
# Sidebar: persistencia y fuente de datos
# ------------------------------------------------------------
st.sidebar.header("Fuente de datos")
modo = st.sidebar.radio("Selecciona", ["CSV (repositorio)", "Descargar ahora desde CNMC"], index=0)

st.sidebar.subheader("Persistencia de datos (CNMC → CSV)")
c1, c2 = st.sidebar.columns(2)
btn_fetch = c1.button("⬇️ Descargar+limpiar")
btn_zip   = c2.button("💾 Generar ZIP")
btn_push  = st.sidebar.button("⬆️ Guardar en GitHub (data/clean/)")  # requiere secrets

if "CNMC_FILES" not in st.session_state:
    st.session_state["CNMC_FILES"] = None

if btn_fetch:
    with st.spinner("Descargando y limpiando datasets CNMC…"):
        files = download_and_clean_all()
        st.session_state["CNMC_FILES"] = files
        st.success(f"Listo: {len(files)} CSV limpios preparados.")

if st.session_state["CNMC_FILES"] and btn_zip:
    zbytes = make_zip(st.session_state["CNMC_FILES"])
    st.download_button(
        "Descargar data_clean.zip",
        data=zbytes,
        file_name="data_clean.zip",
        mime="application/zip",
        use_container_width=True
    )

if st.session_state["CNMC_FILES"] and btn_push:
    try:
        push_all_to_github(st.session_state["CNMC_FILES"], subdir="data/clean/")
        st.success("CSV guardados en tu repositorio en data/clean/ ✅")
    except Exception as e:
        st.error(f"No se pudieron subir a GitHub: {e}")
        st.info("Configura secrets: GITHUB_REPO, GITHUB_TOKEN (y opcional GITHUB_BRANCH).")

# ------------------------------------------------------------
# Carga de CSV “congelados” desde el repo
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str):
    return pd.read_csv(path)

dataset_name = st.sidebar.selectbox("Dataset", list(CSV_PATHS.keys()))
path = CSV_PATHS[dataset_name]

df = None
if modo.startswith("CSV"):
    try:
        df = load_csv(path)
        st.success(f"{dataset_name}: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    except Exception as e:
        st.error(f"No se pudo cargar {path}. Sube los CSV limpios a data/clean/. Detalle: {e}")
else:
    # Descarga on-the-fly: carga el dataframe del dict ya preparado (si lo hay)
    if st.session_state["CNMC_FILES"]:
        # usa el nombre mapeado
        key_map = {
            "Anual – Datos generales": "anual_datos_generales_clean.csv",
            "Anual – Mercados": "anual_mercados_clean.csv",
            "Mensual": "mensual_clean.csv",
            "Provinciales": "provinciales_clean.csv",
            "Trimestrales": "trimestrales_clean.csv",
            "Infraestructuras": "infraestructuras_clean.csv",
        }
        k = key_map[dataset_name]
        if k in st.session_state["CNMC_FILES"]:
            df = pd.read_csv(io.BytesIO(st.session_state["CNMC_FILES"][k]))
            st.success(f"(CNMC directo) {dataset_name}: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        else:
            st.warning("Aún no has descargado ese dataset en esta sesión. Usa '⬇️ Descargar+limpiar'.")
    else:
        st.info("Pulsa '⬇️ Descargar+limpiar' para obtener datos desde CNMC en esta sesión.")

# ------------------------------------------------------------
# Opcional: agrupar “pequeños” en la UI (además del CSV)
# ------------------------------------------------------------
if df is not None and "operador" in df.columns:
    agrupar_ui = st.sidebar.toggle("Agrupar operadores pequeños en 'Otros' (UI)", value=True)
    if agrupar_ui:
        df = group_small_ops(df, top_n=5, col_op="operador")

# ------------------------------------------------------------
# Vista previa + MCDM (TOPSIS) + exploración
# ------------------------------------------------------------
if df is not None:
    with st.expander("Vista previa de datos"):
        st.dataframe(df.head(50), use_container_width=True)

    st.header("⚖️ Análisis multicriterio (TOPSIS)")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    default_crit = num_cols[:3] if len(num_cols) >= 3 else num_cols
    crit = st.multiselect("Elige criterios numéricos", num_cols, default=default_crit)

    if crit:
        benefit_flags, weights = [], []
        st.subheader("Pesos y tipo de criterio")
        for c in crit:
            cols = st.columns([2, 1, 2])
            with cols[0]: st.write(f"**{c}**")
            with cols[1]: benefit_flags.append(st.toggle("Beneficio", True, key=f"b_{c}"))
            with cols[2]: weights.append(st.slider(f"Peso {c}", 0.0, 1.0, 1.0 / max(len(crit), 1), 0.01, key=f"w_{c}"))
        try:
            rank = topsis_rank(df, crit, weights, benefit_flags)
            st.subheader("Ranking TOPSIS")
            st.dataframe(rank.head(50), use_container_width=True)
            fig = px.histogram(rank, x="score_topsis", nbins=40, title="Distribución del score TOPSIS")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo calcular TOPSIS con las columnas seleccionadas: {e}")

    st.header("📈 Exploración rápida")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 2:
        xcol = st.selectbox("Eje X", num_cols, index=0)
        ycol = st.selectbox("Eje Y", num_cols, index=1)
        fig = px.scatter(df, x=xcol, y=ycol, title=f"{ycol} vs {xcol}")
        st.plotly_chart(fig, use_container_width=True)

st.caption("Incluye: descarga/limpieza CNMC, ZIP y push opcional a GitHub; lectura de CSV persistentes; TOPSIS y exploración.")
