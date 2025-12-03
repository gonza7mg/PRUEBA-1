# 3_build_ia_trimestral_model_input.py
#
# Construye el dataset de entrenamiento para el módulo de IA
# a nivel trimestre–operador, integrando los 6 RAW de la CNMC.
#
# Salida: data/model_input/ia_trimestral_model.csv
# Columnas principales:
#   - trimestre (YYYYTQ)
#   - anno
#   - num_trim (1..4)
#   - operador
#   - valor  (ingresos minoristas por operador y trimestre)
#   - + features de infraestructuras, mensual, anual, provinciales...
#   - + variables derivadas: lags, ARPU, ratios...

from __future__ import annotations

from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd


# ==========================
# CONFIGURACIÓN DE RUTAS
# ==========================

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/model_input")
OUT_PATH = OUT_DIR / "ia_trimestral_model.csv"


def build_ia_trimestral_model_input(
    raw_dir: Path = RAW_DIR,
    out_path: Path = OUT_PATH,
) -> pd.DataFrame:
    """
    Construye un dataset de entrenamiento a nivel trimestre–operador
    usando todos los datasets RAW de la CNMC:
      - trimestrales.csv
      - infraestructuras.csv
      - mensual.csv
      - anual_datos_generales.csv
      - anual_mercados.csv
      - provinciales.csv

    Devuelve el DataFrame final y guarda un CSV en out_path.
    """

    # ---------- 1. Cargar RAW ----------
    tri_path = raw_dir / "trimestrales.csv"
    inf_path = raw_dir / "infraestructuras.csv"
    men_path = raw_dir / "mensual.csv"
    ag_path = raw_dir / "anual_datos_generales.csv"
    am_path = raw_dir / "anual_mercados.csv"
    prov_path = raw_dir / "provinciales.csv"

    for p in [tri_path, inf_path, men_path, ag_path, am_path, prov_path]:
        if not p.exists():
            raise FileNotFoundError(f"No se encuentra el fichero RAW esperado: {p}")

    tri = pd.read_csv(tri_path)
    inf = pd.read_csv(inf_path)
    men = pd.read_csv(men_path)
    ag = pd.read_csv(ag_path)
    am = pd.read_csv(am_path)
    prov = pd.read_csv(prov_path)

    # ============================================================
    # 2. BASE: ingresos minoristas por operador y trimestre
    # ============================================================

    base = tri[
        (tri["concepto"] == "Ingresos")
        & (tri["tipo_de_mercado"] == "Servicio minorista")
        & (tri["operador"].notna())
    ].copy()

    # Variable objetivo (target) para el modelo
    base["valor"] = base["ingresos_por_operador"]

    # Año y num_trim a partir de "2005T1" -> anno=2005, num_trim=1
    base["trimestre"] = base["trimestre"].astype(str)
    base["anno"] = base["trimestre"].str.slice(0, 4).astype(int)
    base["num_trim"] = base["trimestre"].str[-1].astype(int)

    # Ingresos totales del trimestre (suma de operadores)
    tot = (
        base.groupby("trimestre", as_index=False)["valor"]
        .sum()
        .rename(columns={"valor": "tri_ingresos_total_trimestre"})
    )
    base = base.merge(tot, on="trimestre", how="left")

    # Cuota de ingresos del operador en ese trimestre
    base["tri_cuota_ingresos_trimestre"] = (
        base["valor"] / base["tri_ingresos_total_trimestre"]
    )

    # ============================================================
    # 3. INFRAESTRUCTURAS: líneas BAM, tráfico datos, estaciones, nodos
    # ============================================================

    inf_feat_list: list[pd.DataFrame] = []

    # Líneas de Banda Ancha móvil
    m_lines_bam = (
        (inf["servicio"] == "Banda Ancha móvil")
        & (inf["concepto"] == "Líneas")
    )
    inf_lines_bam = (
        inf[m_lines_bam]
        .groupby(["trimestre", "operador"], as_index=False)["lineas_o_accesos"]
        .sum()
        .rename(columns={"lineas_o_accesos": "inf_bam_lineas"})
    )
    inf_feat_list.append(inf_lines_bam)

    # Tráfico de datos de Banda Ancha móvil
    m_traf_bam = (
        (inf["servicio"] == "Banda Ancha móvil")
        & (inf["concepto"] == "Tráfico - datos")
    )
    inf_traf_bam = (
        inf[m_traf_bam]
        .groupby(["trimestre", "operador"], as_index=False)["trafico_de_datos"]
        .sum()
        .rename(columns={"trafico_de_datos": "inf_bam_trafico_datos"})
    )
    inf_feat_list.append(inf_traf_bam)

    # Estaciones base (cualquier servicio, concepto = Estaciones Base)
    m_est = inf["concepto"] == "Estaciones Base"
    inf_est = (
        inf[m_est]
        .groupby(["trimestre", "operador"], as_index=False)["estaciones_base"]
        .sum()
        .rename(columns={"estaciones_base": "inf_estaciones_base"})
    )
    inf_feat_list.append(inf_est)

    # Nodos de radio
    m_nodos = inf["concepto"] == "Nodos"
    inf_nodos = (
        inf[m_nodos]
        .groupby(["trimestre", "operador"], as_index=False)["nodos_radio"]
        .sum()
        .rename(columns={"nodos_radio": "inf_nodos_radio"})
    )
    inf_feat_list.append(inf_nodos)

    if inf_feat_list:
        inf_feats = reduce(
            lambda left, right: pd.merge(
                left, right, on=["trimestre", "operador"], how="outer"
            ),
            inf_feat_list,
        )
    else:
        inf_feats = pd.DataFrame(columns=["trimestre", "operador"])

    base = base.merge(inf_feats, on=["trimestre", "operador"], how="left")

    # ============================================================
    # 4. MENSUAL -> AGREGADO TRIMESTRAL
    # ============================================================

    men2 = men.copy()
    men2["mes_dt"] = pd.to_datetime(men2["mes"])
    men2["anno"] = men2["mes_dt"].dt.year
    men2["trim"] = ((men2["mes_dt"].dt.month - 1) // 3 + 1).astype(int)
    men2["trimestre"] = men2["anno"].astype(str) + "T" + men2["trim"].astype(str)

    # Portabilidades de Telefonía móvil (suma de 3 meses)
    m_port = (
        (men2["servicio"] == "Telefonía móvil")
        & (men2["concepto"] == "Portabilidades")
    )
    men_port = (
        men2[m_port]
        .groupby(["trimestre", "operador"], as_index=False)["portabilidades"]
        .sum()
        .rename(columns={"portabilidades": "men_portab_moviles"})
    )

    # Líneas de BAF minorista (media trimestral)
    m_baf = (
        (men2["servicio"] == "Banda ancha fija minorista")
        & (men2["concepto"] == "Líneas")
    )
    men_baf = (
        men2[m_baf]
        .groupby(["trimestre", "operador"], as_index=False)["lineas"]
        .mean()
        .rename(columns={"lineas": "men_baf_lineas_media"})
    )

    men_feats = men_port.merge(men_baf, on=["trimestre", "operador"], how="outer")
    base = base.merge(men_feats, on=["trimestre", "operador"], how="left")

    # ============================================================
    # 5. ANUAL DATOS GENERALES: ingresos, empleados, inversiones
    # ============================================================

    m_ag_ing = (
        (ag["concepto"] == "Ingresos")
        & (ag["tipo_de_mercado"] == "Servicio minorista")
        & (ag["operador"].notna())
    )
    ag_ing = (
        ag[m_ag_ing]
        .groupby(["anno", "operador"], as_index=False)["ingresos_por_operador"]
        .sum()
        .rename(columns={"ingresos_por_operador": "an_gen_ingresos_minorista"})
    )

    m_ag_emp = (ag["concepto"] == "Número de empleados") & (ag["operador"].notna())
    ag_emp = (
        ag[m_ag_emp]
        .groupby(["anno", "operador"], as_index=False)["empleados_por_operador"]
        .sum()
        .rename(columns={"empleados_por_operador": "an_gen_empleados"})
    )

    m_ag_inv = (
        (ag["concepto"] == "Inversiones en infraestr. de telec. y serv. audiov.")
        & (ag["operador"].notna())
    )
    ag_inv = (
        ag[m_ag_inv]
        .groupby(["anno", "operador"], as_index=False)["inversiones_por_operador"]
        .sum()
        .rename(columns={"inversiones_por_operador": "an_gen_inversiones"})
    )

    ag_feats = ag_ing.merge(ag_emp, on=["anno", "operador"], how="outer").merge(
        ag_inv, on=["anno", "operador"], how="outer"
    )
    base = base.merge(ag_feats, on=["anno", "operador"], how="left")

    # ============================================================
    # 6. ANUAL MERCADOS: líneas, clientes e ingresos móviles (ARPU)
    # ============================================================

    def agg_lineas(servicio: str, concepto: str, col_in: str, newname: str) -> pd.DataFrame:
        m = (
            (am["servicio"] == servicio)
            & (am["concepto"] == concepto)
            & (am["operador"].notna())
        )
        return (
            am[m]
            .groupby(["anno", "operador"], as_index=False)[col_in]
            .sum()
            .rename(columns={col_in: newname})
        )

    # Líneas BAF minorista
    am_baf_lines = agg_lineas(
        "Banda ancha fija minorista",
        "Líneas",
        "lineas_o_accesos_por_operador",
        "an_merc_baf_lineas",
    )

    # Líneas móviles
    am_mov_lines = agg_lineas(
        "Telefonía móvil",
        "Líneas",
        "lineas_o_accesos_por_operador",
        "an_merc_mov_lineas",
    )

    # Líneas BAM
    am_bam_lines = agg_lineas(
        "Banda Ancha móvil",
        "Líneas",
        "lineas_o_accesos_por_operador",
        "an_merc_bam_lineas",
    )

    # Clientes móviles anuales
    m_am_mov_cli = (
        (am["servicio"] == "Telefonía móvil")
        & (am["concepto"] == "Clientes")
        & (am["operador"].notna())
    )
    am_mov_clients = (
        am[m_am_mov_cli]
        .groupby(["anno", "operador"], as_index=False)["clientes_por_operador"]
        .sum()
        .rename(columns={"clientes_por_operador": "an_merc_mov_clientes"})
    )

    # Ingresos móviles anuales (para ARPU)
    m_am_mov_ing = (
        (am["servicio"] == "Telefonía móvil")
        & (am["concepto"] == "Ingresos")
        & (am["operador"].notna())
    )
    am_mov_ing = (
        am[m_am_mov_ing]
        .groupby(["anno", "operador"], as_index=False)["ingresos_por_operador"]
        .sum()
        .rename(columns={"ingresos_por_operador": "an_merc_mov_ingresos"})
    )

    am_feats = reduce(
        lambda l, r: pd.merge(l, r, on=["anno", "operador"], how="outer"),
        [am_baf_lines, am_mov_lines, am_bam_lines, am_mov_clients, am_mov_ing],
    )

    base = base.merge(am_feats, on=["anno", "operador"], how="left")

    # ============================================================
    # 7. PROVINCIALES: BAF por operador a nivel nacional
    # ============================================================

    pr = prov.copy()
    pr_baf = pr[pr["servicio"] == "Banda ancha fija"].copy()
    pr_baf = pr_baf[pr_baf["operador"].notna()]

    prov_feats = (
        pr_baf.groupby(["anno", "operador"])
        .agg(
            prov_baf_lineas_total=("lineas_o_accesos", "sum"),
            prov_baf_pen_media=("tasa_de_penetracion", "mean"),
            prov_baf_pen_std=("tasa_de_penetracion", "std"),
        )
        .reset_index()
    )

    base = base.merge(prov_feats, on=["anno", "operador"], how="left")

    # ============================================================
    # 8. VARIABLES DERIVADAS (ARPU, ratios, lags...)
    # ============================================================

    # ARPU móvil anual = ingresos móviles anuales / clientes móviles anuales
    base["arpu_mov_anual"] = np.where(
        base["an_merc_mov_clientes"].fillna(0) > 0,
        base["an_merc_mov_ingresos"] / base["an_merc_mov_clientes"],
        np.nan,
    )

    # Tráfico de datos por línea BAM (a nivel trimestral)
    base["trafico_datos_por_linea_bam"] = np.where(
        base["inf_bam_lineas"].fillna(0) > 0,
        base["inf_bam_trafico_datos"] / base["inf_bam_lineas"],
        np.nan,
    )

    # Ordenamos por operador y trimestre para calcular lags del target
    base = base.sort_values(["operador", "anno", "num_trim"]).reset_index(drop=True)

    base["valor_lag1"] = base.groupby("operador")["valor"].shift(1)
    base["valor_lag4"] = base.groupby("operador")["valor"].shift(4)

    # ============================================================
    # 9. Selección final de columnas + imputación de NaN mejorada
    # ============================================================

    feature_cols = [
        "tri_ingresos_total_trimestre",
        "tri_cuota_ingresos_trimestre",
        "num_trim",
        "inf_bam_lineas",
        "inf_bam_trafico_datos",
        "inf_estaciones_base",
        "inf_nodos_radio",
        "men_portab_moviles",
        "men_baf_lineas_media",
        "an_gen_ingresos_minorista",
        "an_gen_empleados",
        "an_gen_inversiones",
        "an_merc_baf_lineas",
        "an_merc_mov_lineas",
        "an_merc_bam_lineas",
        "an_merc_mov_clientes",
        "prov_baf_lineas_total",
        "prov_baf_pen_media",
        "prov_baf_pen_std",
        "valor_lag1",
        "valor_lag4",
        "arpu_mov_anual",
        "trafico_datos_por_linea_bam",
    ]

    final_cols = ["trimestre", "anno", "operador", "valor"] + feature_cols
    ia_df = base[final_cols].copy()

    # Imputación:
    # 1) ffill + bfill por operador (evita saltos a 0 cuando falta un año)
    ia_df = ia_df.sort_values(["operador", "anno", "num_trim"]).reset_index(drop=True)
    ia_df[feature_cols] = (
        ia_df.groupby("operador", group_keys=False)[feature_cols]
        .apply(lambda g: g.ffill().bfill())
    )

    # 2) cualquier NaN residual (operadores sin info nunca) se rellena a 0
    ia_df[feature_cols] = ia_df[feature_cols].fillna(0.0)

    # Orden final por operador y trimestre
    ia_df = ia_df.sort_values(["operador", "trimestre"]).reset_index(drop=True)

    # ============================================================
    # 10. Guardado
    # ============================================================

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ia_df.to_csv(out_path, index=False)

    print(f"✅ Dataset de IA generado en: {out_path}")
    print(f"   Filas: {len(ia_df)}, columnas: {len(ia_df.columns)}")
    print("   Columnas:", ", ".join(ia_df.columns))

    return ia_df


if __name__ == "__main__":
    df_out = build_ia_trimestral_model_input()
