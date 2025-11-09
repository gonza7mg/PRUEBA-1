# pages/2_Calidad_de_Datos.py
import io
import os
import pandas as pd
import streamlit as st

from utils.data_quality import (
    run_quality_suite,
    evaluate_df,
    evaluate_path,
    save_quality_report,
)

st.set_page_config(page_title="Calidad de Datos – DSS CNMC", layout="wide", page_icon="🧪")
st.title("🧪 Calidad de Datos – RAW vs CLEAN vs FINAL")

# Rutas conocidas en el proyecto
RAW_PATHS = {
    "Anual – Datos generales": "data/raw/anual_datos_generales.csv",
    "Anual – Mercados":        "data/raw/anual_mercados.csv",
    "Mensual":                 "data/raw/mensual.csv",
    "Provinciales":            "data/raw/provinciales.csv",
    "Trimestrales":            "data/raw/trimestrales.csv",
    "Infraestructuras":        "data/raw/infraestructuras.csv",
}
CLEAN_PATHS = {
    "Anual – Datos generales": "data/clean/anual_datos_generales_clean.csv",
    "Anual – Mercados":        "data/clean/anual_mercados_clean.csv",
    "Mensual":                 "data/clean/mensual_clean.csv",
    "Provinciales":            "data/clean/provinciales_clean.csv",
    "Trimestrales":            "data/clean/trimestrales_clean.csv",
    "Infraestructuras":        "data/clean/infraestructuras_clean.csv",
}
FINAL_PATHS = {
    "Anual – Datos generales": "data/final/anual_datos_generales_final.csv",
    "Anual – Mercados":        "data/final/anual_mercados_final.csv",
    "Mensual":                 "data/final/mensual_final.csv",
    "Provinciales":            "data/final/provinciales_final.csv",
    "Trimestrales":            "data/final/trimestrales_final.csv",
    "Infraestructuras":        "data/final/infraestructuras_final.csv",
}

def exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

st.sidebar.header("Modo de evaluación")
modo = st.sidebar.radio("Elige:", ["Capa única (RAW/CLEAN/FINAL)", "Comparar RAW vs CLEAN vs FINAL"], index=1)

datasets = list(RAW_PATHS.keys())

# ============= CAPA ÚNICA =============
if modo.startswith("Capa única"):
    capa = st.sidebar.selectbox("Capa", ["RAW", "CLEAN", "FINAL"], index=2)
    ds = st.selectbox("Dataset", datasets)

    st.info(
        "RAW (desde CSV o memoria si descargaste en 'Inicio'), "
        "CLEAN (data/clean/) y FINAL (data/final/)."
    )

    # Cargar según capa
    result_foto = None   # foto rápida (score simple)
    df = None            # dataframe para suite detallada

    if capa == "RAW":
        # Prioriza RAW en memoria si existe (descargado en la página Inicio)
        if "CNMC_RAW" in st.session_state and st.session_state["CNMC_RAW"]:
            key_map = {
                "Anual – Datos generales": "anual_datos_generales.csv",
                "Anual – Mercados": "anual_mercados.csv",
                "Mensual": "mensual.csv",
                "Provinciales": "provinciales.csv",
                "Trimestrales": "trimestrales.csv",
                "Infraestructuras": "infraestructuras.csv",
            }
            k = key_map[ds]
            if k in st.session_state["CNMC_RAW"]:
                try:
                    df = pd.read_csv(io.BytesIO(st.session_state["CNMC_RAW"][k]))
                    st.success("Usando RAW desde memoria (CNMC descargado en esta sesión).")
                except Exception as e:
                    st.warning(f"No se pudo leer RAW en memoria: {e}")
        # Si no hay en memoria, usa CSV de data/raw
        if df is None:
            path = RAW_PATHS[ds]
            if exists(path):
                result_foto = evaluate_path(path, ds)
                try:
                    df = pd.read_csv(path)
                except Exception as e:
                    st.error(f"No se pudo cargar {path} para la suite detallada: {e}")
            else:
                st.error(f"No existe {path}. Descarga RAW o súbelo a data/raw/.")

    elif capa == "CLEAN":
        path = CLEAN_PATHS[ds]
        if exists(path):
            result_foto = evaluate_path(path, ds)
            try:
                df = pd.read_csv(path)
            except Exception as e:
                st.error(f"No se pudo cargar {path} para la suite detallada: {e}")
        else:
            st.error(f"No existe {path}. Genera CLEAN primero.")

    else:  # FINAL
        path = FINAL_PATHS[ds]
        if exists(path):
            result_foto = evaluate_path(path, ds)
            try:
                df = pd.read_csv(path)
            except Exception as e:
                st.error(f"No se pudo cargar {path} para la suite detallada: {e}")
        else:
            st.error(f"No existe {path}. Ejecuta scripts/make_final_from_clean.py")

    # Vista previa y suite detallada
    if isinstance(df, pd.DataFrame):
        with st.expander(" Vista previa (primeras 50 filas)", expanded=False):
            st.dataframe(df.head(50), use_container_width=True)

        # Pista para reglas de unicidad y checks temporales
        hint = "anual" if "Anual" in ds else \
               "mensual" if "Mensual" in ds else \
               "trimestral" if "Trimestrales" in ds else \
               "provincial" if "Provinciales" in ds else \
               "infraestructuras" if "Infraestructuras" in ds else None

        if st.button("▶️ Ejecutar suite de calidad detallada", use_container_width=True):
            with st.spinner("Evaluando calidad…"):
                rep = run_quality_suite(df, dataset_hint=hint)

            score = rep.attrs.get("quality_score", 100.0)
            c1, c2 = st.columns([1,3])
            with c1:
                st.metric("Quality Score (suite detallada)", f"{score:.2f}/100")
            with c2:
                st.write("**Resumen de incidencias por prueba** (menor es mejor):")
                st.dataframe(
                    rep.sort_values(["dimension","pct_rows"], ascending=[True, False]),
                    use_container_width=True,
                    height=420
                )

            st.download_button(
                "⬇️ Descargar informe (CSV)",
                data=rep.to_csv(index=False).encode("utf-8"),
                file_name=f"data_quality_{capa.lower()}_{ds.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Foto rápida (completitud/unicidad/no-informativas) y guardado
    if result_foto:
        st.subheader(f"Foto rápida – {ds} [{capa}]")
        st.dataframe(pd.DataFrame([result_foto]), use_container_width=True)
        if st.button("💾 Guardar foto rápida (CSV)"):
            out = save_quality_report([result_foto], "data/quality", f"quality_{capa.lower()}_{ds.replace(' ','_')}")
            st.success(f"Guardado en {out}")

# ============= COMPARACIÓN 3 CAPAS =============
else:
    st.subheader("Comparación por dataset (RAW vs CLEAN vs FINAL)")

    rows = []
    for ds in datasets:
        # RAW: memoria > CSV
        raw_res = None
        if "CNMC_RAW" in st.session_state and st.session_state["CNMC_RAW"]:
            key_map = {
                "Anual – Datos generales": "anual_datos_generales.csv",
                "Anual – Mercados": "anual_mercados.csv",
                "Mensual": "mensual.csv",
                "Provinciales": "provinciales.csv",
                "Trimestrales": "trimestrales.csv",
                "Infraestructuras": "infraestructuras.csv",
            }
            k = key_map[ds]
            if k in st.session_state["CNMC_RAW"]:
                try:
                    df = pd.read_csv(io.BytesIO(st.session_state["CNMC_RAW"][k]))
                    raw_res = evaluate_df(df, ds)
                except Exception as e:
                    raw_res = {"dataset": ds, "score": 0.0, "error": f"RAW mem: {e}"}
        if raw_res is None:
            rp = RAW_PATHS[ds]
            raw_res = evaluate_path(rp, ds) if exists(rp) else {"dataset": ds, "score": 0.0, "error": "RAW no disponible"}

        # CLEAN
        cp = CLEAN_PATHS[ds]
        clean_res = evaluate_path(cp, ds) if exists(cp) else {"dataset": ds, "score": 0.0, "error": "CLEAN no disponible"}

        # FINAL
        fp = FINAL_PATHS[ds]
        final_res = evaluate_path(fp, ds) if exists(fp) else {"dataset": ds, "score": 0.0, "error": "FINAL no disponible"}

        rows.append({
            "dataset": ds,
            "raw_score": raw_res.get("score", 0.0),
            "clean_score": clean_res.get("score", 0.0),
            "final_score": final_res.get("score", 0.0),
            "Δ clean-raw": round((clean_res.get("score", 0.0) - raw_res.get("score", 0.0)), 2),
            "Δ final-clean": round((final_res.get("score", 0.0) - clean_res.get("score", 0.0)), 2),
            "Δ final-raw": round((final_res.get("score", 0.0) - raw_res.get("score", 0.0)), 2),
        })

    df_cmp = pd.DataFrame(rows)
    st.dataframe(df_cmp, use_container_width=True)

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("💾 Guardar comparación (CSV)"):
            out = save_quality_report(rows, "data/quality", "quality_compare_raw_clean_final")
            st.success(f"Guardado en {out}")
    with c2:
        st.download_button(
            "⬇️ Descargar comparación",
            data=df_cmp.to_csv(index=False).encode("utf-8"),
            file_name="quality_compare_raw_clean_final.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.caption("Consejo: si ves 0 en RAW y estás usando 'Descargar ahora desde CNMC', vuelve a la página Inicio y pulsa '⬇️ Descargar RAW'.")
