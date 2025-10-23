# pages/1_Limpieza_y_Normalizacion.py
import streamlit as st
import subprocess
import zipfile
import io
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Limpieza y Normalización – DSS CNMC", layout="wide", page_icon="🧹")
st.header("🧹 Limpieza y Normalización de Datos")

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
LOGS_DIR = Path("logs")

st.markdown("""
Esta página ejecuta el proceso ETL (Extract–Transform–Load) para transformar los datos **RAW** descargados desde la CNMC
en versiones **limpias y normalizadas (CLEAN)**, aplicando las rutinas definidas en `utils/data_cleaners/`.
""")

# ------------------------------------------------------------
# BOTONES PRINCIPALES
# ------------------------------------------------------------
c1, c2, c3 = st.columns([1,1,1])

# --- Ejecutar limpieza completa ---
if c1.button("🔄 Ejecutar limpieza completa (RAW → CLEAN)", use_container_width=True):
    with st.spinner("Ejecutando script de limpieza..."):
        try:
            result = subprocess.run(
                ["python", "scripts/make_clean_from_raw.py"],
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

st.caption("El flujo completo queda así: **data/raw → limpieza ETL → data/clean → logs/clean_report.csv**.")
