
import io
import zipfile
import base64
import pandas as pd
import numpy as np
import requests
import streamlit as st

from utils.cnmc_ckan import fetch_resource  # descarga CKAN
# Los limpiadores existen pero NO se usan aquí (analizamos RAW primero)

st.set_page_config(page_title="DSS Telecomunicaciones – CNMC", layout="wide", page_icon="📶")
st.title(" DSS Telecomunicaciones – CNMC")

# Recursos CKAN (RAW)
RESOURCES = {
    "anual_datos_generales": "5e2d8f37-2385-4774-82ec-365cd83d65bd",
    "anual_mercados": "7afbf769-655d-4b43-b49f-95c2919ec1fe",
    "mensual": "3632297f-07d8-480c-aca5-c987dcde0ccb",
    "provinciales": "1efe6d64-72a8-4f45-a36c-691054f3e277",
    "trimestrales": "5da45f2f-e596-4940-b682-eab18e85288a",
    "infraestructuras": "baab2a5e-cc52-4704-a799-a28b19223a3b",
}

# Rutas (RAW) ya persistidas en repo
CSV_PATHS = {
    "Anual – Datos generales": "data/raw/anual_datos_generales.csv",
    "Anual – Mercados": "data/raw/anual_mercados.csv",
    "Mensual": "data/raw/mensual.csv",
    "Provinciales": "data/raw/provinciales.csv",
    "Trimestrales": "data/raw/trimestrales.csv",
    "Infraestructuras": "data/raw/infraestructuras.csv",
}

# ---------- Utilidades ----------
def download_all_raw() -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for name, rid in RESOURCES.items():
        df = fetch_resource(rid)
        out[f"{name}.csv"] = df.to_csv(index=False).encode("utf-8")
    return out

def make_zip(files_dict: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, content in files_dict.items():
            zf.writestr(fname, content)
    return buf.getvalue()

def github_put_file(owner, repo, branch, path, content_bytes, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    r = requests.get(url, params={"ref": branch}, headers={"Authorization": f"Bearer {token}"})
    sha = r.json().get("sha") if r.status_code == 200 else None
    payload = {
        "message": f"Update {path}",
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": branch,
    }
    if sha: payload["sha"] = sha
    r = requests.put(url, headers={"Authorization": f"Bearer {token}"}, json=payload)
    if r.status_code not in (200,201):
        raise RuntimeError(f"GitHub PUT failed for {path}: {r.status_code} {r.text}")

def push_all_to_github(files_dict: dict[str, bytes], subdir: str = "data/raw/"):
    owner_repo = st.secrets["GITHUB_REPO"]
    branch     = st.secrets.get("GITHUB_BRANCH", "main")
    token      = st.secrets["GITHUB_TOKEN"]
    owner, repo = owner_repo.split("/", 1)
    for fname, content in files_dict.items():
        github_put_file(owner, repo, branch, f"{subdir}{fname}", content, token)

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def df_basic_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "columna": c,
            "dtype": str(s.dtype),
            "no_nulos": int(s.notna().sum()),
            "nulos": int(s.isna().sum()),
            "unicos": int(s.nunique(dropna=True)),
            "ejemplo": s.dropna().iloc[0] if s.notna().any() else None,
        })
    return pd.DataFrame(rows).sort_values("nulos", ascending=False)

def df_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    return num.describe(percentiles=[0.01,0.25,0.5,0.75,0.99]).T if not num.empty else pd.DataFrame()

def df_top_values(df: pd.DataFrame, max_cols: int = 8, top_n: int = 12) -> dict[str, pd.Series]:
    out = {}
    for c in [c for c in df.columns if df[c].dtype == "object"][:max_cols]:
        out[c] = df[c].value_counts(dropna=False).head(top_n)
    return out

# ---------- UI ----------
st.sidebar.header("Fuente de datos")
modo = st.sidebar.radio("Selecciona", ["CSV (repositorio)", "Descargar ahora desde CNMC"], index=0)

st.sidebar.subheader("Persistencia de datos (CNMC → CSV RAW)")
c1, c2 = st.sidebar.columns(2)
btn_fetch = c1.button("⬇️ Descargar RAW")
btn_zip   = c2.button("💾 Generar ZIP RAW")
btn_push  = st.sidebar.button("⬆️ Guardar en GitHub (data/raw/)")

if "CNMC_RAW" not in st.session_state:
    st.session_state["CNMC_RAW"] = None

if btn_fetch:
    with st.spinner("Descargando datasets RAW desde la CNMC…"):
        files = download_all_raw()
        st.session_state["CNMC_RAW"] = files
        st.success(f"Listo: {len(files)} CSV RAW en memoria.")

if st.session_state["CNMC_RAW"] and btn_zip:
    zbytes = make_zip(st.session_state["CNMC_RAW"])
    st.download_button("Descargar data_raw.zip", data=zbytes, file_name="data_raw.zip",
                       mime="application/zip", use_container_width=True)

if st.session_state["CNMC_RAW"] and btn_push:
    try:
        push_all_to_github(st.session_state["CNMC_RAW"], subdir="data/raw/")
        st.success("CSV RAW guardados en tu repositorio en data/raw/ ✅")
    except KeyError:
        st.error("Faltan secrets en Cloud: GITHUB_REPO, GITHUB_TOKEN (y opcional GITHUB_BRANCH).")
    except Exception as e:
        st.error(f"No se pudo subir a GitHub: {e}")

dataset_name = st.sidebar.selectbox("Dataset", list(CSV_PATHS.keys()))
path = CSV_PATHS[dataset_name]

df = None
if modo.startswith("CSV"):
    try:
        df = load_csv(path)
        st.success(f"{dataset_name}: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    except Exception as e:
        st.error(f"No se pudo cargar {path}. Sube los CSV RAW a data/raw/. Detalle: {e}")
else:
    if st.session_state["CNMC_RAW"]:
        key_map = {
            "Anual – Datos generales": "anual_datos_generales.csv",
            "Anual – Mercados": "anual_mercados.csv",
            "Mensual": "mensual.csv",
            "Provinciales": "provinciales.csv",
            "Trimestrales": "trimestrales.csv",
            "Infraestructuras": "infraestructuras.csv",
        }
        k = key_map[dataset_name]
        if k in st.session_state["CNMC_RAW"]:
            df = pd.read_csv(io.BytesIO(st.session_state["CNMC_RAW"][k]))
            st.success(f"(CNMC directo) {dataset_name}: {df.shape[0]:,} filas × {df.shape[1]}")
        else:
            st.warning("Aún no has descargado ese dataset en esta sesión. Usa '⬇️ Descargar RAW'.")
    else:
        st.info("Pulsa '⬇️ Descargar RAW' para obtener datos desde CNMC en esta sesión.")

if df is not None:
    with st.expander(" Vista previa (primeras 50 filas)", expanded=True):
        st.dataframe(df.head(50), use_container_width=True)

    st.header(" Análisis exploratorio (RAW)")
    cA, cB, cC, cD = st.columns([1,1,1,1])
    with cA:  st.metric("Filas", f"{len(df):,}")
    with cB:  st.metric("Columnas", f"{df.shape[1]}")
    with cC:  st.metric("Duplicados", f"{df.duplicated().sum():,}")
    with cD:  st.metric("Nulos (totales)", f"{int(df.isna().sum().sum()):,}")

    with st.expander(" Columnas: tipos, nulos y únicos"):
        st.dataframe(df_basic_overview(df), use_container_width=True, height=420)

    num_stats = df_numeric_stats(df)
    if not num_stats.empty:
        with st.expander(" Estadísticas numéricas (describe)"):
            st.dataframe(num_stats, use_container_width=True)

    with st.expander(" Categóricas: valores más frecuentes"):
        topvals = df_top_values(df, max_cols=8, top_n=12)
        if not topvals:
            st.write("No hay columnas categóricas (tipo object) que mostrar.")
        else:
            for col, ser in topvals.items():
                st.write(f"**{col}**")
                st.dataframe(ser.rename("frecuencia"), use_container_width=True)

st.caption("Flujo actual: descarga RAW → ZIP opcional → push a GitHub → análisis exploratorio. La limpieza se ejecuta con el script en scripts/make_clean_from_raw.py.")
