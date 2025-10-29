# pages/1_Limpieza_y_Normalizacion.py
import streamlit as st
import subprocess
import sys
import zipfile
import io
import pandas as pd
from pathlib import Path
import base64
import requests

st.set_page_config(page_title="Limpieza y Normalización – DSS CNMC", layout="wide", page_icon="🧹")
st.header(" Limpieza y Normalización de Datos")

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
LOGS_DIR = Path("logs")

st.markdown("""
Esta página ejecuta el proceso ETL (**Extract–Transform–Load**) para transformar los datos **RAW** descargados desde la CNMC
en versiones **limpias y normalizadas (CLEAN)**, aplicando las rutinas definidas en `utils/data_cleaners/`.

**Flujo completo:** `data/raw → limpieza ETL → data/clean → logs/clean_report.csv → subida opcional a GitHub`
""")


# Función auxiliar para subir archivos a GitHub

def github_put_file(owner, repo, branch, path, content_bytes, token):
    """Sube o actualiza un archivo en GitHub (REST API /contents)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
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
    """Sube todos los CSV CLEAN al repo usando secretos configurados en Streamlit Cloud."""
    owner_repo = st.secrets["GITHUB_REPO"]
    branch     = st.secrets.get("GITHUB_BRANCH", "main")
    token      = st.secrets["GITHUB_TOKEN"]
    owner, repo = owner_repo.split("/", 1)
    for fname, content in files_dict.items():
        github_put_file(owner, repo, branch, f"{subdir}{fname}", content, token)


# BOTONES PRINCIPALES

c1, c2, c3, c4 = st.columns([1,1,1,1])

# --- Ejecutar limpieza completa ---
if c1.button("🔄 Ejecutar limpieza completa (RAW → CLEAN)", use_container_width=True):
    with st.spinner("Ejecutando script de limpieza..."):
        try:
            result = subprocess.run(
                [sys.executable, "scripts/make_clean_from_raw.py"],
                capture_output=True, text=True, check=True
            )
            st.success("✅ Limpieza completada correctamente.")
            st.text(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error("❌ Error durante la limpieza.")
            st.text(e.stderr)

# --- Ver informe de limpieza ---
if c2.button("📊 Ver informe de limpieza (logs/clean_report.csv)", use_container_width=True):
    log_file = LOGS_DIR / "clean_report.csv"
    if log_file.exists():
        df = pd.read_csv(log_file)
        st.success("Informe de limpieza cargado.")
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.warning("No se encontró logs/clean_report.csv. Ejecuta primero la limpieza.")

# --- Descargar ZIP CLEAN ---
if c3.button("💾 Descargar ZIP CLEAN", use_container_width=True):
    if not CLEAN_DIR.exists() or not any(CLEAN_DIR.glob("*.csv")):
        st.warning("No hay archivos limpios en data/clean/. Ejecuta primero la limpieza.")
    else:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in CLEAN_DIR.glob("*.csv"):
                zf.write(f, arcname=f.name)
        st.download_button(
            "⬇️ Descargar data_clean.zip",
            data=buf.getvalue(),
            file_name="data_clean.zip",
            mime="application/zip",
            use_container_width=True
        )

# --- Subir CLEAN a GitHub ---
if c4.button("⬆️ Subir CLEAN a GitHub (data/clean/)", use_container_width=True):
    if not CLEAN_DIR.exists() or not any(CLEAN_DIR.glob("*.csv")):
        st.warning("No hay archivos limpios en data/clean/. Ejecuta primero la limpieza.")
    else:
        with st.spinner("Subiendo archivos limpios a GitHub..."):
            try:
                files_dict = {f.name: f.read_bytes() for f in CLEAN_DIR.glob("*.csv")}
                push_all_to_github(files_dict, subdir="data/clean/")
                st.success("✅ Archivos CLEAN subidos correctamente a GitHub (data/clean/).")
            except KeyError:
                st.error("❌ Faltan secretos GITHUB_REPO, GITHUB_TOKEN o GITHUB_BRANCH en Streamlit Cloud.")
            except Exception as e:
                st.error(f"❌ Error al subir archivos a GitHub: {e}")

# ------------------------------------------------------------
# Mostrar contenido actual de las carpetas
# ------------------------------------------------------------
st.divider()
st.subheader("📂 Estado actual de carpetas")

def list_csvs(folder: Path):
    if not folder.exists():
        return pd.DataFrame(columns=["archivo","tamaño (KB)"])
    files = list(folder.glob("*.csv"))
    data = [{
        "archivo": f.name,
        "tamaño (KB)": round(f.stat().st_size / 1024, 2)
    } for f in files]
    return pd.DataFrame(data)

tab1, tab2 = st.tabs(["data/raw/", "data/clean/"])
with tab1:
    st.dataframe(list_csvs(RAW_DIR), use_container_width=True)
with tab2:
    st.dataframe(list_csvs(CLEAN_DIR), use_container_width=True)

st.caption("Flujo completo: **data/raw → limpieza ETL → data/clean → logs/clean_report.csv → subida opcional a GitHub.**")
