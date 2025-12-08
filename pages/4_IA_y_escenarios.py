import os
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA


# =====================================================
# CONFIGURACIÓN BÁSICA
# =====================================================

DATA_PATH = "data/model_input/ia_trimestral_model.csv"

DATE_COL = "trimestre"   # periodo tipo '2018T3'
YEAR_COL = "anno"
GROUP_COL = "operador"
TARGET_COL = "valor"

# nº de retardos para el modelo autoregresivo simple
N_LAGS = 4

# Lista de features pensada para el dataset nuevo (modelo explicable)
ML_BASE_FEATURES = [
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


# =====================================================
# FUNCIONES AUXILIARES COMUNES
# =====================================================

@st.cache_data
def load_ia_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se ha encontrado el dataset de IA en {path}. "
            f"Asegúrate de haber ejecutado 3_build_ia_trimestral_model_input.py."
        )

    df = pd.read_csv(path)

    # 1) Fusionar Orange + Grupo MASMOVIL -> MASORANGE
    df[GROUP_COL] = df[GROUP_COL].replace(
        {
            "Orange": "MASORANGE",
            "Grupo MASMOVIL": "MASORANGE",
        }
    )

    # 2) Asegurar que cada (operador, año, trimestre) es único
    #    Sumamos todas las columnas numéricas en caso de duplicados.
    key_cols = [GROUP_COL, YEAR_COL, DATE_COL]

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # no agregamos las columnas clave dentro de num_cols
    num_cols = [c for c in num_cols if c not in key_cols]

    df = (
        df.groupby(key_cols, as_index=False)[num_cols]
        .sum()
    )

    # 3) num_trim a partir del trimestre (YYYYTQ -> Q)
    df["num_trim"] = df[DATE_COL].astype(str).str[-1].astype(int)

    # 4) Orden temporal
    df = df.sort_values([GROUP_COL, YEAR_COL, "num_trim"]).reset_index(drop=True)

    # 5) Recalcular lags de ingresos (valor) por operador
    df["valor_lag1"] = (
        df.groupby(GROUP_COL)[TARGET_COL]
        .shift(1)
        .fillna(0.0)
    )
    df["valor_lag4"] = (
        df.groupby(GROUP_COL)[TARGET_COL]
        .shift(4)
        .fillna(0.0)
    )

    return df



def compute_hhi(shares: np.ndarray) -> float:
    """
    Calcula el índice HHI a partir de cuotas (en fracción, no en %).
    Devuelve HHI en puntos (0–10 000).
    """
    shares_pct = shares * 100.0
    return float(np.sum(shares_pct ** 2))


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Devuelve solo las columnas de ML_BASE_FEATURES que existen en df."""
    return [c for c in ML_BASE_FEATURES if c in df.columns]


# =====================================================
# FUNCIONES AUXILIARES PARA FORECAST ML AUTORREGRESIVO
# =====================================================

def build_ar_lag_dataset(y: np.ndarray, n_lags: int = 4):
    """
    Construye un dataset autoregresivo:
    X[i] = [y_{t-1}, ..., y_{t-n_lags}], y[i] = y_t
    """
    X, target = [], []
    for t in range(n_lags, len(y)):
        X.append(y[t - n_lags: t])
        target.append(y[t])
    return np.array(X), np.array(target)


def iterative_forecast_ar(
    history_y: np.ndarray,
    model,
    n_lags: int,
    horizon: int,
    shock_first: float | None = None,
):
    """
    Forecast iterativo autoregresivo puro (solo lags del propio ingreso).
    Si shock_first está definido → se aplica solo al primer trimestre futuro.
    """
    hist = list(history_y.astype(float))
    preds = []

    for step in range(horizon):
        window = np.array(hist[-n_lags:]).reshape(1, -1)
        y_pred = float(model.predict(window)[0])

        if step == 0 and shock_first is not None:
            y_pred *= (1.0 + shock_first)

        preds.append(y_pred)
        hist.append(y_pred)

    return preds


def generate_future_quarters(df_op: pd.DataFrame, horizon: int) -> list[str]:
    """
    Genera etiquetas de trimestre reales para el forecast:
    2024T4 → 2025T1 → 2025T2 → ...
    Se basa en la última etiqueta de trimestre del operador (columna DATE_COL).
    """
    last = df_op.iloc[-1]
    tri_str = str(last[DATE_COL])  # ej. '2024T3'
    anno = int(tri_str[:4])
    num_trim = int(tri_str[-1])

    labels: list[str] = []
    for _ in range(horizon):
        num_trim += 1
        if num_trim > 4:
            num_trim = 1
            anno += 1
        labels.append(f"{anno}T{num_trim}")
    return labels


# =====================================================
# MODELO ML POR OPERADOR (EXPLICABLE + FORECAST)
# =====================================================

def prepare_df_operator(df: pd.DataFrame, operador: str) -> pd.DataFrame:
    """Filtra y ordena el histórico de un operador."""
    sub = df[df[GROUP_COL] == operador].copy()
    if sub.empty:
        raise ValueError("No hay datos para el operador seleccionado.")
    sub = sub.sort_values([YEAR_COL, "num_trim"]).reset_index(drop=True)
    return sub


def train_rf_for_operator(
    df: pd.DataFrame,
    operador: str,
    feature_cols: List[str],
) -> Tuple[RandomForestRegressor, pd.DataFrame]:
    """Entrena un RandomForest para un operador y devuelve (modelo, df_op)."""
    df_op = prepare_df_operator(df, operador)

    X = df_op[feature_cols].values
    y = df_op[TARGET_COL].values

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(X, y)

    return model, df_op


def next_quarter(anno: int, num_trim: int) -> Tuple[int, int]:
    """Devuelve (año, num_trim) del trimestre siguiente."""
    if num_trim < 4:
        return anno, num_trim + 1
    return anno + 1, 1


def iterative_forecast(
    model: RandomForestRegressor,
    df_op: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    shock_first: float | None = None,
):
    """
    Forecast iterativo por operador usando RF y lags valor_lag1 / valor_lag4.

    - horizon: nº de trimestres a futuro.
    - shock_first: si no es None, aplica multiplicador (1 + shock_first)
      SOLO al primer paso.
    """
    df_op = df_op.sort_values([YEAR_COL, "num_trim"]).reset_index(drop=True)
    history_vals = df_op[TARGET_COL].tolist()

    last_row = df_op.iloc[-1].copy()
    anno = int(last_row[YEAR_COL])
    num_trim = int(last_row["num_trim"])

    future_tris = []
    future_vals = []

    for step in range(horizon):
        anno, num_trim = next_quarter(anno, num_trim)
        trimestre_str = f"{anno}T{num_trim}"

        new_row = last_row.copy()
        new_row[YEAR_COL] = anno
        new_row["num_trim"] = num_trim
        new_row[DATE_COL] = trimestre_str

        # actualizamos lags en función del histórico + predicciones
        new_row["valor_lag1"] = history_vals[-1]
        if len(history_vals) >= 4:
            new_row["valor_lag4"] = history_vals[-4]
        else:
            new_row["valor_lag4"] = history_vals[0]

        X_new = new_row[feature_cols].values.reshape(1, -1)
        y_pred = float(model.predict(X_new)[0])

        if step == 0 and shock_first is not None:
            y_pred = y_pred * (1.0 + shock_first)

        history_vals.append(y_pred)
        future_tris.append(trimestre_str)
        future_vals.append(y_pred)

        last_row = new_row

    return future_tris, future_vals


def train_arima(ts: pd.Series, order=(1, 1, 1)):
    model = ARIMA(ts, order=order)
    results = model.fit()
    return results


def detect_anomalies(residuals: np.ndarray, contamination: float = 0.15):
    resid = residuals.reshape(-1, 1)
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
    )
    labels = iso.fit_predict(resid)
    return labels  # 1 normal, -1 anómalo


# =====================================================
# MODELO GLOBAL PARA SIMULADOR 
# =====================================================

def get_snapshot_prefusion(df: pd.DataFrame, year_max: int = 2023) -> pd.DataFrame:
    """
    Foto de mercado PRE-FUSIÓN para el simulador global:
      - solo años <= year_max
      - excluye MASORANGE
      - toma la última observación de cada operador en ese periodo
    """
    df_pref = df.copy()
    df_pref = df_pref[df_pref[YEAR_COL] <= year_max]
    df_pref = df_pref[df_pref[GROUP_COL] != "MASORANGE"]
    df_pref = df_pref.sort_values([GROUP_COL, YEAR_COL, DATE_COL])
    snap = df_pref.groupby(GROUP_COL, as_index=False).tail(1)
    return snap.reset_index(drop=True)


@st.cache_data
def build_fused_for_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un dataset donde Orange + Grupo MASMOVIL se sustituyen por MASORANGE
    en todo el histórico. Se agregan columnas por operador, pero el tamaño total
    de mercado (tri_ingresos_total_trimestre) se mantiene coherente por trimestre.
    """
    df = df.copy()

    # 1) Fusionar Orange + Grupo MASMOVIL -> MASORANGE
    mask_om = df[GROUP_COL].isin(["Orange", "Grupo MASMOVIL"])
    sub_om = df[mask_om].copy()
    rest = df[~mask_om].copy()

    if sub_om.empty:
        df2 = df.copy()
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # agregamos SOLO numéricos para la pareja fusionada
        fused = (
            sub_om.groupby([YEAR_COL, DATE_COL], as_index=False)[num_cols]
            .sum()
        )
        fused[GROUP_COL] = "MASORANGE"
        df2 = pd.concat([rest, fused], ignore_index=True)

    # 2) Evitar sumar tri_ingresos_total_trimestre al agrupar por operador
    num_cols2 = df2.select_dtypes(include=[np.number]).columns.tolist()
    grp_cols = [GROUP_COL, YEAR_COL, DATE_COL]

    cols_sum = [c for c in num_cols2 if c != "tri_ingresos_total_trimestre"]

    df_agg = (
        df2.groupby(grp_cols, as_index=False)[cols_sum]
        .sum()
    )

    # 3) Recuperar el tamaño total de mercado original por trimestre
    tri_tot = (
        df[[DATE_COL, "tri_ingresos_total_trimestre"]]
        .drop_duplicates(subset=[DATE_COL])
    )

    df_agg = df_agg.merge(tri_tot, on=DATE_COL, how="left")

    return df_agg



@st.cache_data
def train_global_scenario_model(df_fused: pd.DataFrame):
    """
    Modelo global sencillo para el simulador (regresión lineal).
    Solo usa unas pocas variables de escenario con buena señal.
    """
    scenario_features = [
        "tri_ingresos_total_trimestre",
        "men_portab_moviles",
        "an_merc_mov_lineas",
        "an_merc_bam_lineas",
    ]
    feat_cols = [c for c in scenario_features if c in df_fused.columns]
    if not feat_cols:
        raise ValueError("No se han encontrado las columnas de escenario esperadas.")

    df_train = df_fused.dropna(subset=feat_cols + [TARGET_COL]).copy()
    X = df_train[feat_cols].values
    y = df_train[TARGET_COL].values

    model = LinearRegression()
    model.fit(X, y)

    return model, feat_cols


def get_latest_snapshot_postfusion(df_fused: pd.DataFrame) -> pd.DataFrame:
    """Última observación por operador en el dataset fusionado (mundo post-fusión)."""
    df_sorted = df_fused.sort_values([GROUP_COL, YEAR_COL, DATE_COL])
    snap = df_sorted.groupby(GROUP_COL, as_index=False).tail(1)
    return snap.reset_index(drop=True)


# =====================================================
# INTERFAZ STREAMLIT
# =====================================================

st.title("Módulo de IA: Predicción, Escenarios y Anomalías")

st.markdown(
    """
Esta página utiliza el dataset integrado **ia_trimestral_model.csv** para:

1. Entrenar un **modelo ML explicable (RandomForest, scikit-learn)** por operador.  
2. Generar un **forecast temporal autoregresivo** con escenario simple.  
3. Producir un **forecast clásico ARIMA** con bandas de confianza.  
4. Detectar **anomalías** en la evolución histórica (IsolationForest).  
5. Simular **escenarios de negocio** en un mercado post-fusión (MASORANGE).
"""
)

# ------------------ carga de datos ------------------

try:
    df = load_ia_dataset(DATA_PATH)
except Exception as e:
    st.error(f"Error cargando el dataset de IA: {e}")
    st.stop()

# Unificar Orange + Grupo MASMOVIL bajo MASORANGE en toda la página
df[GROUP_COL] = df[GROUP_COL].replace(
    {
        "Orange": "MASORANGE",
        "Grupo MASMOVIL": "MASORANGE",
    }
)

feature_cols_global = get_feature_cols(df)

# ------------------ sidebar -------------------------

st.sidebar.header("Configuración IA (modelo temporal por operador)")

operadores = sorted(df[GROUP_COL].dropna().unique().tolist())
operador_sel = st.sidebar.selectbox("Operador", operadores)

horizon = st.sidebar.slider(
    "Horizonte de predicción (trimestres)",
    min_value=4,
    max_value=12,
    value=8,
)

shock_pct = st.sidebar.slider(
    "Shock sobre el primer trimestre del forecast (%)",
    min_value=-30,
    max_value=50,
    value=10,
    step=5,
) / 100.0

contamination = st.sidebar.slider(
    "Sensibilidad detección de anomalías (%)",
    min_value=5,
    max_value=30,
    value=15,
    step=5,
) / 100.0


# ------------------ preparar serie por operador ------------------

try:
    df_op = prepare_df_operator(df, operador_sel)
except Exception as e:
    st.error(f"Error preparando la serie temporal: {e}")
    st.stop()

if len(df_op) < 8:
    st.warning(
        f"La serie tiene pocas observaciones ({len(df_op)}). "
        "Las predicciones pueden no ser muy estables."
    )

st.subheader("Serie temporal de ingresos por operador")

col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(
        df_op[[DATE_COL, YEAR_COL, TARGET_COL]].rename(
            columns={DATE_COL: "periodo", TARGET_COL: "ingresos_trimestrales"}
        ),
        use_container_width=True,
    )

with col2:
    st.metric("Observaciones", len(df_op))
    st.metric("Mínimo ingresos", f"{df_op[TARGET_COL].min():,.0f}")
    st.metric("Máximo ingresos", f"{df_op[TARGET_COL].max():,.0f}")

st.line_chart(
    df_op.set_index(DATE_COL)[TARGET_COL],
    height=260,
)

# =====================================================
# 1. MODELO ML EXPLICABLE (RandomForest por operador)
# =====================================================

st.markdown("---")
st.header("1. Modelo ML explicable (RandomForest por operador)")

st.markdown(
    """
En esta sección se entrena un **RandomForest por operador** para explicar
una variable de interés (target) que puedes elegir:

- Ingresos minoristas trimestrales (`valor`)
- Portabilidades móviles trimestrales (agregado mensual)
- Inversión anual
- Líneas móviles / BAM / BAF, etc.

El modelo se entrena solo con las **features estructurales** del dataset
(infraestructura, portabilidades, ingresos anuales, líneas, provinciales…).
"""
)

# --- 1.1 Selección de la variable objetivo (target) ---

# Posibles variables a explicar (solo se mostrarán las que existan en df_op)
candidate_targets = {
    "Ingresos minoristas (valor)": "valor",
    "Portabilidades móviles trimestrales (mensual agregado)": "men_portab_moviles",
    "Inversión anual (an_gen_inversiones)": "an_gen_inversiones",
    "Líneas móviles anuales": "an_merc_mov_lineas",
    "Líneas BAM anuales": "an_merc_bam_lineas",
    "Líneas BAF anuales": "an_merc_baf_lineas",
    "ARPU móvil anual": "arpu_mov_anual",
}

available_targets = {
    label: col
    for label, col in candidate_targets.items()
    if col in df_op.columns and np.issubdtype(df_op[col].dtype, np.number)
}

if not available_targets:
    st.error("No se han encontrado variables numéricas adecuadas para el modelo explicable.")
else:
    target_label = st.selectbox(
        "Variable a explicar con el modelo ML",
        list(available_targets.keys()),
        index=0,
    )
    target_col = available_targets[target_label]

    st.markdown(
        f"Se está explicando la variable **{target_label}** (`{target_col}`) "
        f"para el operador **{operador_sel}**."
    )

    # --- 1.2 Preparar X (features) e y (target) ---

    # Features globales definidas al principio (ML_BASE_FEATURES filtradas por columnas existentes)
    # Evitamos usar como feature la propia columna target
    feat_cols_1 = [
        c for c in feature_cols_global
        if c in df_op.columns and c != target_col
    ]

    if not feat_cols_1:
        st.error("No hay suficientes features disponibles para entrenar el modelo ML.")
    else:
        X_op = df_op[feat_cols_1].values
        y_op = df_op[target_col].values.astype(float)

        # Por si hubiera NaN en el target (p.ej. en ARPU), filtramos filas válidas
        mask_valid = np.isfinite(y_op)
        if mask_valid.sum() < 5:
            st.warning(
                "Hay muy pocas observaciones válidas para la variable seleccionada. "
                "El modelo explicable puede no ser estable."
            )

        X_train = X_op[mask_valid]
        y_train = y_op[mask_valid]

        # --- 1.3 Entrenar RandomForest para la variable seleccionada ---

        rf_model = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)

        # Predicciones in-sample (solo en filas válidas)
        y_hat_in_valid = rf_model.predict(X_train)

        # Rellenamos un vector de predicción alineado con todo df_op (NaN donde no había datos)
        y_hat_in = np.full_like(y_op, fill_value=np.nan, dtype=float)
        y_hat_in[mask_valid] = y_hat_in_valid

        # --- 1.4 Métricas de ajuste ---

        try:
            mae = mean_absolute_error(y_train, y_hat_in_valid)
            rmse = math.sqrt(mean_squared_error(y_train, y_hat_in_valid))
        except ValueError:
            mae, rmse = np.nan, np.nan

        colm1, colm2 = st.columns(2)
        with colm1:
            st.metric("MAE (in-sample)", f"{mae:,.2f}" if np.isfinite(mae) else "N/A")
        with colm2:
            st.metric("RMSE (in-sample)", f"{rmse:,.2f}" if np.isfinite(rmse) else "N/A")

        # --- 1.5 Gráfico Real vs Predicho para la variable target ---

        plot_df = pd.DataFrame({
            "periodo": df_op[DATE_COL].astype(str),
            "Real": y_op,
            "Predicho_ML": y_hat_in,
        })

        st.line_chart(
            plot_df.set_index("periodo")[["Real", "Predicho_ML"]],
            height=320,
        )

        # --- 1.6 Importancia de variables ---

        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({"feature": feat_cols_1, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False)

        st.subheader("Importancia de variables (IA explicable)")
        st.bar_chart(
            imp_df.set_index("feature")["importance"],
            height=250,
        )

        st.markdown(
            """
Aclaraciones :

- El modelo se entrena **solo con datos históricos observados**, sin hacer supuestos
  sobre el futuro.
- Puedes seleccionar distintas variables de negocio (ingresos, portabilidades, inversión…)
  y ver qué factores del dataset parecen más relevantes para explicarlas.
- En la sección 4, la **detección de anomalías** se realiza sobre la misma variable target
  que has seleccionado aquí.
"""
        )

# =====================================================
# 2. FORECAST TEMPORAL Y ESCENARIO SIMPLE (ML con exógenas reales)
# =====================================================

st.markdown("---")
st.header("2. Forecast temporal y escenario simple (modelo ML por operador)")

st.markdown(
    """
En este apartado se construye un **forecast ML con exógenas reales** por operador:

- El modelo utiliza:
  - los **últimos 4 trimestres del propio indicador** (lags), y  
  - un pequeño conjunto de **variables explicativas reales** (BAM, líneas, etc.).  
- Para los trimestres futuros, las exógenas se mantienen constantes en el valor
  del último trimestre disponible (no inventamos futuros de BAM, inversión, etc.).  
- Se muestran dos curvas:
  - **Baseline_ML**: forecast “inercial” del modelo.  
  - **Escenario_ML**: mismo forecast, pero aplicando un **shock** multiplicativo
    al primer trimestre futuro (slider lateral).

Además puedes elegir qué indicador quieres pronosticar: ingresos, portabilidades,
inversión anual…
"""
)

# -----------------------------------------------------
# Funciones auxiliares internas del apartado 2
# -----------------------------------------------------

def _build_lagged_dataset_for_target(
    df_op: pd.DataFrame,
    target_col: str,
    exog_cols: list[str],
    n_lags: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """
    Construye un dataset ML para un target concreto:
      - crea lags del target (y_{t-1}..y_{t-n})
      - devuelve:
          ts_all: serie completa con lags (incluyendo NaN iniciales)
          ts_ml:  filas válidas para entrenar (sin NaN)
          feature_cols: orden de features
          X, y: matrices para entrenar el modelo
    """
    cols = [DATE_COL, target_col] + exog_cols
    ts = df_op[cols].copy().sort_values(DATE_COL)

    # crear lags del target
    lag_cols = []
    for lag in range(1, n_lags + 1):
        cname = f"lag_{lag}"
        ts[cname] = ts[target_col].shift(lag)
        lag_cols.append(cname)

    # filas válidas (sin NaN en target ni en lags)
    ts_ml = ts.dropna(subset=[target_col] + lag_cols).reset_index(drop=True)

    feature_cols = lag_cols + exog_cols
    X = ts_ml[feature_cols].values.astype(float)
    y = ts_ml[target_col].values.astype(float)

    return ts, ts_ml, feature_cols, X, y


def _rf_forecast_with_exog(
    ts_all: pd.DataFrame,
    model: RandomForestRegressor,
    target_col: str,
    exog_cols: list[str],
    n_lags: int,
    horizon: int,
    shock_first: float | None = None,
) -> list[float]:
    """
    Forecast iterativo con RandomForest:
      - usa lags del target (histórico + predicciones)
      - mantiene las exógenas congeladas en el valor del último trimestre observado
      - shock_first (en fracción, ej. 0.1 = +10%) se aplica solo al primer paso
    """
    # serie histórica limpia
    ts_clean = ts_all.dropna(subset=[target_col]).sort_values(DATE_COL)
    hist = ts_clean[target_col].astype(float).tolist()

    if len(hist) < n_lags:
        return []

    # exógenas = último valor disponible
    if exog_cols:
        exog_vec = ts_clean.iloc[-1][exog_cols].values.astype(float)
    else:
        exog_vec = np.array([], dtype=float)

    preds: list[float] = []

    for step in range(horizon):
        # ventana de lags
        lags = np.array(hist[-n_lags:], dtype=float)

        x = np.concatenate([lags, exog_vec]).reshape(1, -1)
        y_pred = float(model.predict(x)[0])

        if step == 0 and shock_first is not None:
            y_pred *= (1.0 + shock_first)

        preds.append(y_pred)
        hist.append(y_pred)

    return preds


def _generate_future_quarters_from_last(last_period: str, horizon: int) -> list[str]:
    """
    Genera etiquetas de trimestre reales a partir del último periodo del operador:
    '2024T4' -> '2025T1', '2025T2', ...
    """
    last_period = str(last_period)
    anno = int(last_period[:4])
    num_trim = int(last_period[-1])

    labels: list[str] = []
    for _ in range(horizon):
        num_trim += 1
        if num_trim > 4:
            num_trim = 1
            anno += 1
        labels.append(f"{anno}T{num_trim}")
    return labels


# -----------------------------------------------------
# Selección del indicador a pronosticar
# -----------------------------------------------------

# Candidatos “bonitos”  (solo se mostrarán los que existan en df_op)
candidate_targets = [
    ("Ingresos minoristas trimestrales (valor)", "valor"),
    ("Portabilidades móviles (mensual agregada)", "men_portab_moviles"),
    ("Inversión anual por operador", "an_gen_inversiones"),
    ("Líneas móviles anuales", "an_merc_mov_lineas"),
]

available_targets = [
    (label, col)
    for (label, col) in candidate_targets
    if col in df_op.columns and df_op[col].notna().any()
]

if not available_targets:
    st.warning("No se han encontrado columnas numéricas adecuadas para el forecast ML.")
else:
    labels, cols = zip(*available_targets)
    target_label = st.selectbox(
        "Indicador a pronosticar",
        labels,
        index=0,
        key="forecast_target",
    )
    target_col = dict(available_targets)[target_label]

    # -------------------------------------------------
    # Definir exógenas “realistas” para el forecast
    # -------------------------------------------------
    FORECAST_EXOG_CANDIDATES = [
        "tri_ingresos_total_trimestre",
        "inf_bam_lineas",
        "inf_bam_trafico_datos",
        "inf_estaciones_base",
        "men_baf_lineas_media",
        "an_gen_empleados",
        "an_merc_bam_lineas",
        "prov_baf_pen_media",
    ]

    exog_cols = [
        c
        for c in FORECAST_EXOG_CANDIDATES
        if c in df_op.columns and df_op[c].notna().any() and c != target_col
    ]

    n_lags_forecast = 4

    # Construir dataset ML para este target
    ts_all, ts_ml, feature_cols_fc, X_fc, y_fc = _build_lagged_dataset_for_target(
        df_op,
        target_col=target_col,
        exog_cols=exog_cols,
        n_lags=n_lags_forecast,
    )

    if len(ts_ml) < 8:
        st.warning(
            "Hay pocas observaciones útiles para este indicador; el forecast puede ser inestable."
        )
    else:
        # Entrenar modelo RandomForest específico de forecast
        rf_forecast = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        rf_forecast.fit(X_fc, y_fc)

        # Forecast baseline y escenario con shock
        baseline_preds = _rf_forecast_with_exog(
            ts_all=ts_all,
            model=rf_forecast,
            target_col=target_col,
            exog_cols=exog_cols,
            n_lags=n_lags_forecast,
            horizon=horizon,
            shock_first=None,
        )
        scenario_preds = _rf_forecast_with_exog(
            ts_all=ts_all,
            model=rf_forecast,
            target_col=target_col,
            exog_cols=exog_cols,
            n_lags=n_lags_forecast,
            horizon=horizon,
            shock_first=shock_pct,
        )

        if not baseline_preds:
            st.warning("No ha sido posible generar el forecast (serie demasiado corta).")
        else:
            # Etiquetas futuras de trimestre
            last_period = df_op[DATE_COL].iloc[-1]
            future_labels = _generate_future_quarters_from_last(
                last_period, len(baseline_preds)
            )

            # Construir DataFrame para gráfico
            hist_vals = df_op[target_col].astype(float).tolist()

            fc_df = pd.DataFrame({
                "periodo": list(df_op[DATE_COL].astype(str)) + future_labels,
                "Historico": hist_vals + [np.nan] * len(baseline_preds),
                "Baseline_ML": [np.nan] * len(hist_vals) + baseline_preds,
                "Escenario_ML": [np.nan] * len(hist_vals) + scenario_preds,
            })

            st.line_chart(
                fc_df.set_index("periodo")[["Historico", "Baseline_ML", "Escenario_ML"]],
                height=340,
            )

            st.markdown(f"### Detalle numérico del forecast ML para **{target_label}**")
            st.dataframe(
                pd.DataFrame({
                    "trimestre": future_labels,
                    "Baseline_ML": baseline_preds,
                    "Escenario_ML": scenario_preds,
                }),
                use_container_width=True,
            )

            st.markdown(
                """
- **Baseline_ML**: forecast ML con lags 1–4 del indicador seleccionado y
  exógenas reales congeladas en el último trimestre.  
- **Escenario_ML**: mismo forecast, pero aplicando un *shock* multiplicativo al
  **primer trimestre futuro** (slider lateral). Ese shock se propaga porque
  alimenta los lags de los pasos siguientes.
"""
            )


# =====================================================
# 3. FORECAST CLÁSICO ARIMA (variable seleccionable)
# =====================================================

st.markdown("---")
st.header("3. Forecast clásico con ARIMA (con bandas de confianza)")

st.markdown(
    """
En este apartado se estima un modelo **ARIMA(p, d, q)** sobre una variable
temporal de tu elección para el operador seleccionado.

Puedes elegir:

- La **serie** sobre la que aplicar ARIMA (ingresos, portabilidades, líneas…).  
- Los **parámetros (p, d, q)** del modelo dentro de esta sección.

El resultado muestra:

- La serie histórica.  
- El **forecast ARIMA** a horizonte seleccionado.  
- Las **bandas de confianza del 95%**.
"""
)

# --- 3.1 Selección de la variable a modelizar con ARIMA ---

candidate_targets_arima = {
    "Ingresos minoristas (valor)": "valor",
    "Portabilidades móviles trimestrales (mensual agregado)": "men_portab_moviles",
    "Inversión anual (an_gen_inversiones)": "an_gen_inversiones",
    "Líneas móviles anuales": "an_merc_mov_lineas",
    "Líneas BAM anuales": "an_merc_bam_lineas",
    "Líneas BAF anuales": "an_merc_baf_lineas",
    "ARPU móvil anual": "arpu_mov_anual",
}

available_targets_arima = {
    label: col
    for label, col in candidate_targets_arima.items()
    if col in df_op.columns and np.issubdtype(df_op[col].dtype, np.number)
}

if not available_targets_arima:
    st.error("No se han encontrado variables numéricas adecuadas para ARIMA.")
else:
    target_label_arima = st.selectbox(
        "Serie a modelizar con ARIMA",
        list(available_targets_arima.keys()),
        index=0,
        key="arima_target",
    )
    target_col_arima = available_targets_arima[target_label_arima]

    st.markdown(
        f"Se está ajustando ARIMA sobre **{target_label_arima}** "
        f"(`{target_col_arima}`) para el operador **{operador_sel}**."
    )

    # --- 3.2 Parámetros ARIMA dentro de la sección ---

    col_p, col_d, col_q = st.columns(3)

    with col_p:
        p = st.number_input("p (autoregresivo)", min_value=0, max_value=5, value=1, step=1, key="arima_p")
    with col_d:
        d = st.number_input("d (diferencias)", min_value=0, max_value=2, value=1, step=1, key="arima_d")
    with col_q:
        q = st.number_input("q (media móvil)", min_value=0, max_value=5, value=1, step=1, key="arima_q")

    st.markdown(
        """
**Leyenda de parámetros ARIMA(p, d, q):**

- **p**: número de retardos autoregresivos. Cuántos trimestres pasados usa el modelo.  
- **d**: número de diferencias aplicadas para estabilizar la serie (tendencia).  
- **q**: número de términos de media móvil. Cuánto “peso” tienen shocks pasados en el error.
"""
    )

    # --- 3.3 Preparar la serie temporal ---

    serie = df_op[target_col_arima].astype(float)

    # Filtramos NaN por si la serie es incompleta (p.ej. ARPU)
    mask_valid_arima = np.isfinite(serie.values)
    serie_valid = serie[mask_valid_arima]

    if len(serie_valid) < (p + d + q + 4):
        st.warning(
            f"La serie válida para ARIMA tiene pocas observaciones ({len(serie_valid)}). "
            "El ajuste puede ser inestable."
        )

    # --- 3.4 Ajuste de ARIMA y forecast ---

    try:
        arima_results = train_arima(serie_valid, order=(int(p), int(d), int(q)))
    except Exception as e:
        st.error(f"No se ha podido ajustar ARIMA({p},{d},{q}): {e}")
        arima_results = None

    if arima_results is not None:
        fc_arima = arima_results.get_forecast(steps=horizon)
        mean_fc = fc_arima.predicted_mean
        conf_int = fc_arima.conf_int(alpha=0.05)

        # Etiquetas de trimestres futuros reales usando helper existente
        future_labels_arima = generate_future_quarters(df_op, len(mean_fc))

        # Construimos un DataFrame que concatena histórico + forecast
        arima_df = pd.DataFrame({
            "periodo": list(df_op[DATE_COL].astype(str)) + future_labels_arima,
            "Histórico": list(serie.values) + [np.nan] * len(mean_fc),
            "Forecast_ARIMA": [np.nan] * len(df_op) + mean_fc.tolist(),
            "Lower_95": [np.nan] * len(df_op) + conf_int.iloc[:, 0].tolist(),
            "Upper_95": [np.nan] * len(df_op) + conf_int.iloc[:, 1].tolist(),
        })

        # --- 3.5 Gráfico de forecast ARIMA ---

        st.line_chart(
            arima_df.set_index("periodo")[["Histórico", "Forecast_ARIMA"]],
            height=320,
        )

        st.markdown("### Detalle numérico del forecast ARIMA (con bandas 95%)")
        st.dataframe(
            pd.DataFrame({
                "trimestre": future_labels_arima,
                "Forecast_ARIMA": mean_fc.values,
                "Lower_95": conf_int.iloc[:, 0].values,
                "Upper_95": conf_int.iloc[:, 1].values,
            }),
            use_container_width=True,
        )

        st.markdown(
            """
Aclaraciones:

- ARIMA se aplica **operador a operador** y **variable a variable**, sin usar
  otras features del modelo ML.  
- El forecast depende de los parámetros (p, d, q): puedes justificar en la memoria
  que has probado distintas configuraciones y escogido las más estables.  
- Las **bandas de confianza** reflejan la incertidumbre estadística del modelo,
  complementando el enfoque más “caja negra” del RandomForest.
"""
        )


# =====================================================
# 4. DETECCIÓN DE ANOMALÍAS (variable seleccionable)
# =====================================================

st.markdown("---")
st.header("4. Detección de anomalías (IsolationForest sobre residuales ML)")

st.markdown(
    """
En esta sección se detectan **trimestres anómalos** para el operador seleccionado
a partir de los **residuales** de un modelo ML:

1. Se elige una **variable objetivo** (ingresos, portabilidades, inversión, líneas, ARPU…).  
2. Se entrena un RandomForest con las features estructurales.  
3. Se calculan los **residuales** (Real - Predicho).  
4. Sobre esos residuales se aplica **IsolationForest** para marcar trimestres anómalos.

Solo se muestra el gráfico marcado con los puntos anómalos,
sin tabla de detalle, para no sobrecargar la interfaz.
"""
)

# --- 4.1 Selección de la variable para anomalías ---

candidate_targets_anom = {
    "Ingresos minoristas (valor)": "valor",
    "Portabilidades móviles trimestrales (mensual agregado)": "men_portab_moviles",
    "Inversión anual (an_gen_inversiones)": "an_gen_inversiones",
    "Líneas móviles anuales": "an_merc_mov_lineas",
    "Líneas BAM anuales": "an_merc_bam_lineas",
    "Líneas BAF anuales": "an_merc_baf_lineas",
    "ARPU móvil anual": "arpu_mov_anual",
}

available_targets_anom = {
    label: col
    for label, col in candidate_targets_anom.items()
    if col in df_op.columns and np.issubdtype(df_op[col].dtype, np.number)
}

if not available_targets_anom:
    st.error("No se han encontrado variables numéricas adecuadas para detección de anomalías.")
else:
    target_label_anom = st.selectbox(
        "Variable sobre la que buscar anomalías",
        list(available_targets_anom.keys()),
        index=0,
        key="anom_target",
    )
    target_col_anom = available_targets_anom[target_label_anom]

    st.markdown(
        f"Se detectan anomalías sobre **{target_label_anom}** "
        f"(`{target_col_anom}`) para el operador **{operador_sel}**."
    )

    # --- 4.2 Preparar datos y entrenar modelo ML para esa variable ---

    # Features globales (ML_BASE_FEATURES filtradas por columnas existentes),
    # evitando usar la propia target como feature.
    feat_cols_anom = [
        c for c in feature_cols_global
        if c in df_op.columns and c != target_col_anom
    ]

    if not feat_cols_anom:
        st.error("No hay suficientes features disponibles para entrenar el modelo de anomalías.")
    else:
        y_full = df_op[target_col_anom].astype(float).values
        X_full = df_op[feat_cols_anom].values

        # Filtramos observaciones válidas (por si hay NaN en la serie objetivo)
        mask_valid_anom = np.isfinite(y_full)
        y_train_anom = y_full[mask_valid_anom]
        X_train_anom = X_full[mask_valid_anom]

        if len(y_train_anom) < 8:
            st.warning(
                f"La serie válida para anomalías tiene pocas observaciones ({len(y_train_anom)}). "
                "Los resultados pueden ser inestables."
            )

        # Modelo RandomForest para obtener residuales
        rf_anom = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        rf_anom.fit(X_train_anom, y_train_anom)

        y_hat_valid = rf_anom.predict(X_train_anom)

        # Residuales solo donde hay datos válidos
        resid_valid = y_train_anom - y_hat_valid

        # --- 4.3 IsolationForest sobre residuales ---

        labels_valid = detect_anomalies(resid_valid, contamination=contamination)
        # labels_valid: 1 normal, -1 anómalo

        # Construimos un DataFrame alineado con todos los trimestres del operador
        anom_df = pd.DataFrame({
            "periodo": df_op[DATE_COL].astype(str),
            "valor_real": y_full,
            "es_valido": mask_valid_anom,
            "estado": "No evaluado",
        })

        # Rellenamos estado solo en los puntos válidos
        estados = np.where(labels_valid == -1, "Anómalo", "Normal")
        anom_df.loc[mask_valid_anom, "estado"] = estados

        # Para el gráfico marcamos solo los puntos anómalos
        anom_df["valor_anomalo"] = np.where(
            anom_df["estado"] == "Anómalo",
            anom_df["valor_real"],
            np.nan,
        )

        # --- 4.4 Gráfico con puntos anómalos marcados ---

        st.line_chart(
            anom_df.set_index("periodo")[["valor_real", "valor_anomalo"]],
            height=340,
        )

        st.markdown(
            """
Interpretación:

- La línea continua muestra la serie histórica de la variable seleccionada.  
- Los puntos marcados corresponden a trimestres que el modelo ML no consigue
  explicar bien (residuales grandes) y que el IsolationForest clasifica como
  **anómalos**.  

Estos trimestres pueden interpretarse como periodos con shocks competitivos,
cambios regulatorios, campañas comerciales atípicas o problemas en la propia
medición de datos.
"""
        )


# =====================================================
# 5. SIMULADOR DE ESCENARIOS DE NEGOCIO (MUNDO POST-FUSIÓN)
# =====================================================

st.markdown("---")
st.header("5. Simulador de escenarios de negocio (mundo post-fusión MASORANGE)")

st.markdown(
    """
En esta sección se construye un **mercado post-fusión**, donde **Orange y Grupo MASMOVIL
se agregan como MASORANGE en todo el histórico**. Así, todos los escenarios se interpretan
ya en un contexto futuro con la fusión completada.

El motor del simulador es un **modelo global sencillo (regresión lineal)** con pocas
palancas claras: tamaño de mercado, portabilidades y base de clientes móvil/BAM.
"""
)

df_fused = build_fused_for_scenarios(df)
global_model, feat_cols_global = train_global_scenario_model(df_fused)
snapshot = get_latest_snapshot_postfusion(df_fused)

# Features base que usaremos solo para que el modelo calcule los escenarios
snapshot_features = snapshot[feat_cols_global].copy()

# ⚠️ Ingresos base = dato real del dataset fusionado, no predicho
ingresos_base = (
    snapshot[TARGET_COL]
    .fillna(0.0)
    .to_numpy()
    .astype(float)
)

# Por si hubiera algún valor raro en origen, nunca dejamos que sea negativo
ingresos_base = np.maximum(ingresos_base, 0.0)

baseline_total = ingresos_base.sum()
if baseline_total <= 0:
    baseline_shares = np.zeros_like(ingresos_base)
else:
    baseline_shares = ingresos_base / baseline_total

baseline_hhi = compute_hhi(baseline_shares)

snapshot_base = snapshot[[GROUP_COL, YEAR_COL, DATE_COL]].copy()
snapshot_base["ingresos_base"] = ingresos_base
snapshot_base["cuota_base"] = baseline_shares

# =====================================================
# FUNCION AUXILIAR: ELASTICIDADES MACRO POR OPERADOR
# =====================================================

@st.cache_data
def compute_macro_elasticities(df_fused: pd.DataFrame) -> pd.DataFrame:
    """
    Estima, para cada operador, una elasticidad β_i entre los ingresos
    del operador y el tamaño total del mercado.

    Modelo (en logaritmos por operador):
        log(ingresos_operador) = a_i + β_i * log(ingresos_totales_mercado) + error

    Devuelve un DataFrame con:
        - operador
        - beta_macro  (elasticidad estimada, acotada entre 0 y 2)
    """

    # Validación mínima
    if "tri_ingresos_total_trimestre" not in df_fused.columns:
        raise ValueError("Falta la columna tri_ingresos_total_trimestre en df_fused.")

    df_hist = df_fused.copy()

    # Eliminamos observaciones con valores no válidos
    df_hist = df_hist[
        (df_hist[TARGET_COL] > 0) &
        (df_hist["tri_ingresos_total_trimestre"] > 0)
    ].copy()

    # Logaritmos para la regresión
    df_hist["log_y"] = np.log(df_hist[TARGET_COL])
    df_hist["log_M"] = np.log(df_hist["tri_ingresos_total_trimestre"])

    rows = []

    for op, sub in df_hist.groupby(GROUP_COL):
        # Si hay pocos datos para el operador → elasticidad = 1.0
        if len(sub) < 6:
            rows.append({"operador": op, "beta_macro": 1.0})
            continue

        X = sub[["log_M"]].values
        y = sub["log_y"].values

        reg = LinearRegression()
        reg.fit(X, y)
        beta = float(reg.coef_[0])

        # Acotamos elasticidad entre 0 y 2 para evitar locuras
        beta = float(np.clip(beta, 0.0, 2.0))

        rows.append({"operador": op, "beta_macro": beta})

    elastic_df = pd.DataFrame(rows)

    return elastic_df

tabs = st.tabs([
    "Plan de inversión agresivo",
    "Guerra de portabilidades",
    "Expansión operador low-cost",
    "Recorte de inversión",
    "Fusión y HHI",
    "Shock macro mercado",
])

# ------------------ Escenario 1 ------------------ #
with tabs[0]:
    st.subheader("Escenario 1 – Plan de inversión agresivo")

    st.markdown(
        """
Se modeliza la inversión con una **regla de negocio simple**:

> Inversión +X% ⇒ ingresos operador ≈ ingresos_base × (1 + 0,5·X)

- Elasticidad positiva (0,5): si invierte más, mejora sus ingresos.  
- El resto de operadores permanece constante.  
- Aquí solo se analiza el efecto “neto” sobre ingresos y cuota del **operador seleccionado**.
"""
    )

    op_inv = st.selectbox(
        "Operador que incrementa la inversión",
        sorted(snapshot_base[GROUP_COL].unique().tolist()),
        key="esc1_operador",
    )

    delta_inv = st.slider(
        "Incremento de inversión (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
        key="esc1_inv",
    ) / 100.0

    elasticidad_ing = 0.5  # regla de negocio

    # Tabla completa (todos los operadores) para HHI y contexto
    scen_table = snapshot_base.copy()
    scen_table["ingresos_escenario"] = scen_table["ingresos_base"]
    scen_table.loc[scen_table[GROUP_COL] == op_inv, "ingresos_escenario"] *= (
        1 + elasticidad_ing * delta_inv
    )

    scen_total = scen_table["ingresos_escenario"].sum()
    scen_table["cuota_escenario"] = scen_table["ingresos_escenario"] / scen_total
    scen_shares = scen_table["cuota_escenario"].values
    scen_hhi = compute_hhi(scen_shares)

    scen_table = scen_table.sort_values("ingresos_escenario", ascending=False)

    # --- SOLO OPERADOR SELECCIONADO PARA LAS GRÁFICAS ---
    fila_op = scen_table.loc[scen_table[GROUP_COL] == op_inv].iloc[0]
    ing_base = float(fila_op["ingresos_base"])
    ing_esc = float(fila_op["ingresos_escenario"])
    delta_ing = ing_esc - ing_base

    cuota_base = float(fila_op["cuota_base"])
    cuota_esc = float(fila_op["cuota_escenario"])
    delta_cuota = cuota_esc - cuota_base

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.dataframe(scen_table, use_container_width=True)

        st.markdown(f"#### Operador seleccionado: {op_inv}")

        # -------- Gráfica apilada de ingresos (SOLO operador seleccionado) --------
        st.markdown("**Ingresos del operador: base vs escenario (barras apiladas)**")

        ingresos_chart_df = pd.DataFrame(
            [
                # Barra BASE: solo base, sin incremento
                {"tipo": "Base", "componente": "Ingresos base", "valor": ing_base},
                {"tipo": "Base", "componente": "Incremento por inversión", "valor": 0.0},
                # Barra ESCENARIO: base + incremento
                {"tipo": "Escenario", "componente": "Ingresos base", "valor": ing_base},
                {
                    "tipo": "Escenario",
                    "componente": "Incremento por inversión",
                    "valor": delta_ing,
                },
            ]
        )

        st.altair_chart(
            alt.Chart(ingresos_chart_df)
            .mark_bar()
            .encode(
                x=alt.X("tipo:N", title=""),
                y=alt.Y("valor:Q", title="Ingresos trimestrales", stack="zero"),
                color=alt.Color("componente:N", title="Componente"),
            )
            .properties(height=260),
            use_container_width=True,
        )

        # -------- Gráfica apilada de cuota (SOLO operador seleccionado) --------
        st.markdown("**Cuota del operador: base vs escenario (barras apiladas)**")

        cuota_chart_df = pd.DataFrame(
            [
                {"tipo": "Base", "componente": "Cuota base", "valor": cuota_base},
                {"tipo": "Base", "componente": "Incremento por inversión", "valor": 0.0},
                {"tipo": "Escenario", "componente": "Cuota base", "valor": cuota_base},
                {
                    "tipo": "Escenario",
                    "componente": "Incremento por inversión",
                    "valor": delta_cuota,
                },
            ]
        )

        st.altair_chart(
            alt.Chart(cuota_chart_df)
            .mark_bar()
            .encode(
                x=alt.X("tipo:N", title=""),
                y=alt.Y(
                    "valor:Q",
                    axis=alt.Axis(format=".0%"),
                    title="Cuota de mercado",
                    stack="zero",
                ),
                color=alt.Color("componente:N", title="Componente"),
            )
            .properties(height=260),
            use_container_width=True,
        )

    with col_right:
        st.metric("HHI base", f"{baseline_hhi:,.0f}")
        st.metric("HHI escenario", f"{scen_hhi:,.0f}")
        st.metric(f"Cuota base {op_inv}", f"{cuota_base*100:,.2f} %")
        st.metric(f"Cuota escenario {op_inv}", f"{cuota_esc*100:,.2f} %")
        st.metric(f"Δ cuota {op_inv}", f"{delta_cuota*100:,.2f} p.p.")

    st.markdown(
        """
**Aclaraciones**:

- El resto de operadores sirve solo como contexto de mercado (tabla y HHI).  
- Las barras apiladas se centran exclusivamente en la evolución del **operador seleccionado**.  
- El incremento de ingresos/cuota se interpreta como el efecto neto de un plan de inversión
  agresivo en un entorno post-fusión (MASORANGE ya consolidado).
"""
    )

# ------------------ Escenario 2 – Guerra de portabilidades ------------------ #
with tabs[1]:
    st.subheader("Escenario 2 – Guerra de portabilidades")

    st.markdown(
        """
Simula una **campaña agresiva de captación** donde:

- un operador aumenta sus portabilidades (`men_portab_moviles`),
- opcionalmente, un competidor concreto pierde parte de ese flujo.

Se observa cómo cambia la distribución de **ingresos** y **cuotas de mercado**,
así como el índice de concentración **HHI**.
"""
    )

    op_pro = st.selectbox(
        "Operador protagonista",
        sorted(snapshot_base[GROUP_COL].unique().tolist()),
        key="esc2_op_pro",
    )
    op_vic = st.selectbox(
        "Operador víctima (opcional)",
        ["(Ninguno)"] + sorted(snapshot_base[GROUP_COL].unique().tolist()),
        key="esc2_op_vic",
    )

    delta_pro = st.slider(
        "Incremento portabilidades protagonista (%)",
        min_value=0,
        max_value=120,
        value=40,
        step=5,
        key="esc2_delta_pro",
    ) / 100.0
    delta_vic = st.slider(
        "Reducción portabilidades víctima (%)",
        min_value=0,
        max_value=60,
        value=20,
        step=5,
        key="esc2_delta_vic",
    ) / 100.0

    # --- Construimos el escenario sobre las features de la foto base ---
    scen_feat2 = snapshot_features.copy()

    if "men_portab_moviles" in scen_feat2.columns:
        # protagonista gana portabilidades
        scen_feat2.loc[snapshot_base[GROUP_COL] == op_pro, "men_portab_moviles"] *= (
            1.0 + delta_pro
        )
        # víctima (si la hay) pierde parte del flujo
        if op_vic != "(Ninguno)":
            scen_feat2.loc[snapshot_base[GROUP_COL] == op_vic, "men_portab_moviles"] *= (
                1.0 - delta_vic
            )

    # Predicción de ingresos con el modelo global
    scen_pred2 = global_model.predict(scen_feat2.values)
    scen_total2 = scen_pred2.sum()
    scen_shares2 = scen_pred2 / scen_total2
    scen_hhi2 = compute_hhi(scen_shares2)

    # Tabla base + escenario
    scen_table2 = snapshot_base.copy()
    scen_table2["ingresos_escenario"] = scen_pred2
    scen_table2["cuota_escenario"] = scen_shares2
    scen_table2 = scen_table2.sort_values("ingresos_escenario", ascending=False)

    # ==========================
    # 1) TABLA RESUMEN
    # ==========================
    st.markdown("### Tabla de mercado: base vs escenario")
    st.dataframe(scen_table2, use_container_width=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("HHI base", f"{baseline_hhi:,.0f}")
    with col_m2:
        st.metric("HHI escenario", f"{scen_hhi2:,.0f}")

    # ==========================
    # 2) GRÁFICO: CUOTAS DE MERCADO
    # ==========================
    st.markdown("### Distribución de cuotas de mercado (base vs escenario)")

    cuotas_chart_df = scen_table2[["operador", "cuota_base", "cuota_escenario"]].copy()
    cuotas_chart_df = cuotas_chart_df.melt(
        id_vars="operador",
        value_vars=["cuota_base", "cuota_escenario"],
        var_name="estado",
        value_name="cuota",
    )

    # renombramos para la leyenda
    cuotas_chart_df["estado"] = cuotas_chart_df["estado"].replace(
        {"cuota_base": "Base", "cuota_escenario": "Escenario"}
    )

    chart_cuotas = (
        alt.Chart(cuotas_chart_df)
        .mark_bar()
        .encode(
            x=alt.X("operador:N", title="Operador"),
            y=alt.Y("cuota:Q", title="Cuota de mercado", axis=alt.Axis(format="%")),
            color=alt.Color(
                "estado:N",
                title="Situación",
                scale=alt.Scale(domain=["Base", "Escenario"]),
            ),
            column=alt.Column("estado:N", title=""),
        )
        .properties(height=320)
    )

    st.altair_chart(chart_cuotas, use_container_width=True)

    # ==========================
    # 3) GRÁFICO: HHI BASE vs ESCENARIO
    # ==========================
    st.markdown("### Índice de concentración HHI (antes y después de la campaña)")

    hhi_df = pd.DataFrame(
        {
            "estado": ["Base", "Escenario"],
            "HHI": [baseline_hhi, scen_hhi2],
        }
    )

    chart_hhi = (
        alt.Chart(hhi_df)
        .mark_bar()
        .encode(
            x=alt.X("estado:N", title="Situación"),
            y=alt.Y("HHI:Q", title="HHI"),
            color=alt.Color("estado:N", legend=None),
        )
        .properties(height=260)
    )

    st.altair_chart(chart_hhi, use_container_width=True)

    st.markdown(
        """
**Aclaraciones**:  
Este escenario permite cuantificar cómo una guerra de portabilidades modifica:

- la **distribución de cuotas** entre operadores,
- y el nivel de **concentración del mercado (HHI)**.

Si el HHI sube, la campaña tiende a concentrar más el mercado; si baja,
implica un reparto más atomizado de las cuotas.
"""
    )


# ------------------ Escenario 3 – Expansión operador low-cost ------------------ #
with tabs[2]:
    st.subheader("Escenario 3 – Expansión operador low-cost")

    st.markdown(
        """
Pensado para operadores tipo **Digi** o similares: se incrementan a la vez
portabilidades y base de líneas (móvil y BAM), simulando una fase de expansión agresiva.

Se observa cómo cambia la distribución de **ingresos** y **cuotas de mercado**
para todo el mercado y el efecto en el **HHI**.
"""
    )

    # Operadores candidatos "low-cost"
    low_candidates = [
        op
        for op in snapshot_base[GROUP_COL].unique().tolist()
        if ("Digi" in op) or ("MAS" in op) or ("Low" in op) or ("Resto" in op)
    ]
    if not low_candidates:
        low_candidates = snapshot_base[GROUP_COL].unique().tolist()

    op_low = st.selectbox(
        "Operador low-cost / retador",
        sorted(low_candidates),
        key="esc3_op_low",
    )

    delta_porta = st.slider(
        "Incremento portabilidades (%)",
        min_value=0,
        max_value=200,
        value=60,
        step=10,
        key="esc3_porta",
    ) / 100.0
    delta_lineas = st.slider(
        "Incremento base de líneas móviles / BAM (%)",
        min_value=0,
        max_value=200,
        value=60,
        step=10,
        key="esc3_lineas",
    ) / 100.0

    # --- Construimos el escenario sobre la foto base ---
    scen_feat3 = snapshot_features.copy()
    mask_low = snapshot_base[GROUP_COL] == op_low

    # Más portabilidades y más líneas para el low-cost
    if "men_portab_moviles" in scen_feat3.columns:
        scen_feat3.loc[mask_low, "men_portab_moviles"] *= (1.0 + delta_porta)
    if "an_merc_mov_lineas" in scen_feat3.columns:
        scen_feat3.loc[mask_low, "an_merc_mov_lineas"] *= (1.0 + delta_lineas)
    if "an_merc_bam_lineas" in scen_feat3.columns:
        scen_feat3.loc[mask_low, "an_merc_bam_lineas"] *= (1.0 + delta_lineas)

    # Predicción de ingresos con el modelo global
    scen_pred3 = global_model.predict(scen_feat3.values)
    scen_total3 = scen_pred3.sum()
    scen_shares3 = scen_pred3 / scen_total3
    scen_hhi3 = compute_hhi(scen_shares3)

    scen_table3 = snapshot_base.copy()
    scen_table3["ingresos_escenario"] = scen_pred3
    scen_table3["cuota_escenario"] = scen_shares3
    scen_table3 = scen_table3.sort_values("ingresos_escenario", ascending=False)

    # ==========================
    # 1) TABLA RESUMEN
    # ==========================
    st.markdown("### Tabla de mercado: base vs escenario")
    st.dataframe(scen_table3, use_container_width=True)

    col3_m1, col3_m2, col3_m3 = st.columns(3)
    with col3_m1:
        st.metric("HHI base", f"{baseline_hhi:,.0f}")
    with col3_m2:
        st.metric("HHI escenario", f"{scen_hhi3:,.0f}")
    with col3_m3:
        base_q = float(
            scen_table3.loc[scen_table3[GROUP_COL] == op_low, "cuota_base"]
        )
        scen_q = float(
            scen_table3.loc[scen_table3[GROUP_COL] == op_low, "cuota_escenario"]
        )
        st.metric(f"Δ cuota {op_low}", f"{(scen_q - base_q)*100:,.2f} p.p.")

    # ==========================
    # 2) GRÁFICO: CUOTAS DE MERCADO
    # ==========================
    st.markdown("### Distribución de cuotas de mercado (base vs escenario)")

    cuotas_chart3_df = scen_table3[[GROUP_COL, "cuota_base", "cuota_escenario"]].copy()
    cuotas_chart3_df = cuotas_chart3_df.rename(columns={GROUP_COL: "operador"})
    cuotas_chart3_df = cuotas_chart3_df.melt(
        id_vars="operador",
        value_vars=["cuota_base", "cuota_escenario"],
        var_name="estado",
        value_name="cuota",
    )
    cuotas_chart3_df["estado"] = cuotas_chart3_df["estado"].replace(
        {"cuota_base": "Base", "cuota_escenario": "Escenario"}
    )

    chart_cuotas3 = (
        alt.Chart(cuotas_chart3_df)
        .mark_bar()
        .encode(
            x=alt.X("operador:N", title="Operador"),
            y=alt.Y("cuota:Q", title="Cuota de mercado", axis=alt.Axis(format="%")),
            color=alt.Color(
                "estado:N",
                title="Situación",
                scale=alt.Scale(domain=["Base", "Escenario"]),
            ),
            column=alt.Column("estado:N", title=""),
        )
        .properties(height=320)
    )

    st.altair_chart(chart_cuotas3, use_container_width=True)

    # ==========================
    # 3) GRÁFICO: HHI BASE vs ESCENARIO
    # ==========================
    st.markdown("### Índice de concentración HHI (antes y después de la expansión low-cost)")

    hhi3_df = pd.DataFrame(
        {
            "estado": ["Base", "Escenario"],
            "HHI": [baseline_hhi, scen_hhi3],
        }
    )

    chart_hhi3 = (
        alt.Chart(hhi3_df)
        .mark_bar()
        .encode(
            x=alt.X("estado:N", title="Situación"),
            y=alt.Y("HHI:Q", title="HHI"),
            color=alt.Color("estado:N", legend=None),
        )
        .properties(height=260)
    )

    st.altair_chart(chart_hhi3, use_container_width=True)

    st.markdown(
        """
**Aclaraciones**:  
Este escenario representa una **expansión agresiva de un operador low-cost**.
Permite analizar:

- cuánto gana ese operador en **cuota de mercado**,  
- cómo se redistribuyen las cuotas entre el resto,  
- y si el mercado se vuelve **más o menos concentrado** (cambio en el HHI).
"""
    )


# ------------------ Escenario 4 – Recorte de inversión ------------------ #
with tabs[3]:
    st.subheader("Escenario 4 – Recorte de inversión / austeridad")

    st.markdown(
        """
En este escenario se modeliza un **recorte de inversión** como el caso simétrico
del Escenario 1:

- Se aplica **solo al operador seleccionado**; el resto permanece constante.  
- Usamos una regla de negocio sencilla sobre los ingresos trimestrales:  

> **Recorte −X% ⇒ ingresos operador ≈ ingresos_base × (1 − 0,5·X)**  

- La elasticidad 0,5 garantiza que:
  - si recorta inversión, **sus ingresos llevan siempre la dirección correcta** (bajan);
  - los ingresos nunca se vuelven negativos.
"""
    )

    op_rec = st.selectbox(
        "Operador que recorta inversión",
        sorted(snapshot_base[GROUP_COL].unique().tolist()),
        key="esc4_op_rec",
    )

    delta_rec = st.slider(
        "Recorte de inversión (%)",
        min_value=0,
        max_value=80,
        value=30,
        step=5,
        key="esc4_rec",
    ) / 100.0

    # Elasticidad simétrica a la del escenario 1
    elasticidad_rec = 0.5

    scen_table4 = snapshot_base.copy()
    mask_rec = scen_table4[GROUP_COL] == op_rec

    # Ingresos escenario: iguales a los base salvo para el operador que recorta
    scen_table4["ingresos_escenario"] = scen_table4["ingresos_base"]
    scen_table4.loc[mask_rec, "ingresos_escenario"] *= (1.0 - elasticidad_rec * delta_rec)

    # Nunca ingresos negativos (por seguridad numérica, aunque aquí no debería ocurrir)
    scen_table4["ingresos_escenario"] = scen_table4["ingresos_escenario"].clip(lower=0.0)

    scen_total4 = scen_table4["ingresos_escenario"].sum()
    scen_table4["cuota_escenario"] = scen_table4["ingresos_escenario"] / scen_total4

    scen_shares4 = scen_table4["cuota_escenario"].values
    scen_hhi4 = compute_hhi(scen_shares4)

    scen_table4 = scen_table4.sort_values("ingresos_escenario", ascending=False)

    # --- Tabla y métricas ---

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.dataframe(scen_table4, use_container_width=True)

    with col_right:
        st.metric("HHI base", f"{baseline_hhi:,.0f}")
        st.metric("HHI escenario", f"{scen_hhi4:,.0f}")

        base_q = float(
            scen_table4.loc[scen_table4[GROUP_COL] == op_rec, "cuota_base"]
        )
        scen_q = float(
            scen_table4.loc[scen_table4[GROUP_COL] == op_rec, "cuota_escenario"]
        )
        st.metric(f"Δ cuota {op_rec}", f"{(scen_q - base_q) * 100:,.2f} p.p.")

    # --- Gráfico para el operador seleccionado (solo él, barras apiladas) ---

    st.markdown("#### Operador seleccionado: " + str(op_rec))

    op_row = scen_table4[scen_table4[GROUP_COL] == op_rec].iloc[0]

    base_ing = float(op_row["ingresos_base"])
    esc_ing = float(op_row["ingresos_escenario"])
    delta_ing = esc_ing - base_ing

    chart_df = pd.DataFrame(
        {
            "escenario": ["Base", "Escenario"],
            "Base": [base_ing, base_ing],
            "Variación por recorte": [0.0, delta_ing],
        }
    )

    chart_df = chart_df.melt(
        id_vars="escenario",
        value_vars=["Base", "Variación por recorte"],
        var_name="componente",
        value_name="ingresos",
    )

    st.bar_chart(
        chart_df.pivot(index="escenario", columns="componente", values="ingresos"),
        height=320,
    )

    st.markdown(
        """
En el gráfico se ve cómo el recorte de inversión afecta a los **ingresos trimestrales
del operador seleccionado**, manteniendo constante el resto del mercado.
"""
    )


# ------------------ Escenario 5 – Fusión o joint-venture (cambio en HHI) ------------------ #
with tabs[4]:
    st.subheader("Escenario 5 – Fusión o joint-venture (cambio en HHI)")

    st.markdown(
        """
En este escenario se simula una **fusión futura adicional** (por ejemplo, MASORANGE + Vodafone),
partiendo de un mercado donde MASORANGE ya existe.

La mecánica es:

- Se eligen dos operadores A y B.  
- Se crea un operador combinado **A+B** cuyos ingresos son:  

> ingresos(A+B) = (ingresos_A + ingresos_B) × (1 + sinergias%)

- Se recalculan las **cuotas de mercado** y el **HHI** con el nuevo mapa de operadores.
"""
    )

    ops_pref = sorted(snapshot_base[GROUP_COL].unique().tolist())

    op_f1 = st.selectbox("Operador A", ops_pref, key="esc5_op1")
    op_f2 = st.selectbox(
        "Operador B",
        [op for op in ops_pref if op != op_f1],
        key="esc5_op2",
    )

    delta_sin = st.slider(
        "Sinergias en ingresos combinados (%)",
        min_value=-20,
        max_value=40,
        value=10,
        step=5,
        key="esc5_sin",
    ) / 100.0

    # --- Construcción mecánica del escenario de fusión sobre ingresos_base (reales) ---

    base_ops = snapshot_base[GROUP_COL].tolist()
    base_ing = snapshot_base["ingresos_base"].tolist()

    scen_ing_dict: dict[str, float] = {}
    for op, y in zip(base_ops, base_ing):
        if op in (op_f1, op_f2):
            continue
        scen_ing_dict[op] = float(y)

    y1 = float(snapshot_base.loc[snapshot_base[GROUP_COL] == op_f1, "ingresos_base"])
    y2 = float(snapshot_base.loc[snapshot_base[GROUP_COL] == op_f2, "ingresos_base"])
    y_fus = (y1 + y2) * (1.0 + delta_sin)
    op_fus = f"{op_f1}+{op_f2}"

    scen_ing_dict[op_fus] = y_fus

    scen_ops = list(scen_ing_dict.keys())
    scen_ing = np.array(list(scen_ing_dict.values()), dtype=float)
    scen_total5 = scen_ing.sum()
    scen_shares5 = scen_ing / scen_total5
    scen_hhi5 = compute_hhi(scen_shares5)

    scen_table5 = pd.DataFrame({
        "operador": scen_ops,
        "ingresos_escenario": scen_ing,
        "cuota_escenario": scen_shares5,
    }).sort_values("ingresos_escenario", ascending=False)

    # --- Tabla + métricas HHI ---

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.dataframe(scen_table5, use_container_width=True)

    with col_right:
        st.metric("HHI base (mercado actual)", f"{baseline_hhi:,.0f}")
        st.metric("HHI post-fusión", f"{scen_hhi5:,.0f}")

    # =========================
    # GRÁFICOS DE MERCADO – BURBUJAS (CUOTAS)
    # =========================

    st.markdown("### Cuotas de mercado: antes vs después de la fusión (gráficos de burbujas)")

    # Cuotas base (antes de la fusión)
    base_chart = snapshot_base[[GROUP_COL, "cuota_base"]].rename(
        columns={GROUP_COL: "operador", "cuota_base": "cuota"}
    )
    base_chart["escenario"] = "Base"

    # Cuotas escenario (después de la fusión)
    scen_chart = scen_table5[["operador", "cuota_escenario"]].rename(
        columns={"cuota_escenario": "cuota"}
    )
    scen_chart["escenario"] = "Post-fusión"

    # Para escalar bien el tamaño de las burbujas
    all_shares = pd.concat([base_chart, scen_chart], ignore_index=True)
    max_share = all_shares["cuota"].max() if len(all_shares) > 0 else 0.0

    # Gráfico de burbujas "antes"
    bubble_base = (
        alt.Chart(base_chart)
        .mark_circle()
        .encode(
            x=alt.X("operador:N", title="Operador"),
            y=alt.value(0),  # todos en una línea horizontal
            size=alt.Size(
                "cuota:Q",
                title="Cuota de mercado",
                scale=alt.Scale(domain=[0, max_share]),
            ),
            tooltip=[
                alt.Tooltip("operador:N"),
                alt.Tooltip("cuota:Q", format=".2%"),
            ],
            color=alt.value("#1f77b4"),
        )
        .properties(
            title="Cuotas de mercado ANTES de la fusión",
            height=180,
        )
    )

    # Gráfico de burbujas "después"
    bubble_scen = (
        alt.Chart(scen_chart)
        .mark_circle()
        .encode(
            x=alt.X("operador:N", title="Operador"),
            y=alt.value(0),
            size=alt.Size(
                "cuota:Q",
                title="Cuota de mercado",
                scale=alt.Scale(domain=[0, max_share]),
            ),
            tooltip=[
                alt.Tooltip("operador:N"),
                alt.Tooltip("cuota:Q", format=".2%"),
            ],
            color=alt.value("#ff7f0e"),
        )
        .properties(
            title="Cuotas de mercado DESPUÉS de la fusión",
            height=180,
        )
    )

    st.altair_chart(bubble_base, use_container_width=True)
    st.altair_chart(bubble_scen, use_container_width=True)

    st.markdown(
        """
En estos gráficos, el **tamaño de la burbuja** representa la **cuota de mercado**
de cada operador. El segundo gráfico muestra el operador combinado **A+B** tras la fusión.
"""
    )

    # =========================
    # GRÁFICOS DE INGRESOS – DOS BARRAS SEPARADAS
    # =========================

    st.markdown("### Ingresos trimestrales por operador: antes vs post-fusión")

    base_ing_chart = snapshot_base[[GROUP_COL, "ingresos_base"]].rename(
        columns={GROUP_COL: "operador", "ingresos_base": "ingresos"}
    )

    scen_ing_chart = scen_table5[["operador", "ingresos_escenario"]].rename(
        columns={"ingresos_escenario": "ingresos"}
    )

    chart_ing_base = (
        alt.Chart(base_ing_chart)
        .mark_bar()
        .encode(
            x=alt.X("operador:N", title="Operador"),
            y=alt.Y("ingresos:Q", title="Ingresos trimestrales"),
            tooltip=[
                alt.Tooltip("operador:N"),
                alt.Tooltip("ingresos:Q", format=",.0f"),
            ],
            color=alt.value("#1f77b4"),
        )
        .properties(
            title="Ingresos ANTES de la fusión",
            height=260,
        )
    )

    chart_ing_scen = (
        alt.Chart(scen_ing_chart)
        .mark_bar()
        .encode(
            x=alt.X("operador:N", title="Operador"),
            y=alt.Y("ingresos:Q", title="Ingresos trimestrales"),
            tooltip=[
                alt.Tooltip("operador:N"),
                alt.Tooltip("ingresos:Q", format=",.0f"),
            ],
            color=alt.value("#ff7f0e"),
        )
        .properties(
            title="Ingresos DESPUÉS de la fusión",
            height=260,
        )
    )

    st.altair_chart(chart_ing_base, use_container_width=True)
    st.altair_chart(chart_ing_scen, use_container_width=True)

    st.markdown(
        """
En estos dos gráficos de barras se comparan los **ingresos trimestrales**
por operador antes y después de la fusión.  

"""
    )

# =====================================================
# AUXILIAR ESCENARIO 6 – Elasticidades macro βᵢ (históricas)
# =====================================================

def compute_macro_elasticities(df_hist: pd.DataFrame) -> dict:
    """
    Calcula elasticidades macroeconómicas βᵢ por operador, basadas en
    cómo ha participado históricamente en el crecimiento del mercado.

    Idea:
      - Para cada periodo t se calcula:
            Δ mercado_t = mercado_t - mercado_{t-1}
            Δ operador_{i,t} = ingresos_{i,t} - ingresos_{i,t-1}
      - βᵢ ≈ media( Δ operador_{i,t} / Δ mercado_t )
      - Se fuerza βᵢ >= 0 y se normalizan para que sumen 1.

    Resultado:
      - Ningún operador tiene elasticidad negativa.
      - El shock macro se reparte proporcionalmente según el peso
        histórico de cada operador en el crecimiento del mercado.
    """

    # Ordenamos por tiempo para que los diff tengan sentido
    df_hist = df_hist.sort_values([YEAR_COL, DATE_COL, GROUP_COL])

    # Serie de mercado total por periodo (suma de ingresos)
    market = (
        df_hist.groupby([YEAR_COL, DATE_COL])[TARGET_COL]
        .sum()
        .sort_index()
    )
    market_diff = market.diff()
    # Quitamos NaN y periodos sin variación
    market_diff = market_diff.replace(0, np.nan).dropna()

    betas: dict[str, float] = {}

    # Serie por operador
    for op, g in df_hist.groupby(GROUP_COL):
        series = (
            g.groupby([YEAR_COL, DATE_COL])[TARGET_COL]
            .sum()
            .sort_index()
        )
        series_diff = series.diff()

        # Alineamos con el mercado
        aligned = pd.concat(
            [series_diff, market_diff],
            axis=1,
            keys=["op", "mkt"],
        ).dropna()

        # Eliminamos periodos donde el mercado no se mueve
        aligned = aligned[aligned["mkt"] != 0]

        if aligned.empty:
            betas[op] = 0.0
            continue

        ratios = aligned["op"].values / aligned["mkt"].values
        beta_raw = float(np.mean(ratios))

        # No permitimos elasticidades negativas
        betas[op] = max(beta_raw, 0.0)

    # Normalizamos para que el shock agregado cuadre con el mercado total
    total_beta = sum(betas.values())
    if total_beta == 0:
        # Si todo sale 0 (caso extremo), repartimos a partes iguales
        n = len(betas) if betas else 1
        betas = {op: 1.0 / n for op in betas}
    else:
        betas = {op: b / total_beta for op, b in betas.items()}

    return betas


# =====================================================
# AUXILIAR ESCENARIO 6 – Elasticidades βᵢ basadas en cuota
# =====================================================

def compute_share_based_elasticities(snapshot_base: pd.DataFrame) -> dict:
    """
    Calcula elasticidades macro βᵢ a partir de la **cuota base** de cada operador.

    Fórmula:
        βᵢ = cuota_baseᵢ / sum_j (cuota_baseⱼ²)

    Propiedades:
      - βᵢ > 0 para todos los operadores.
      - Operadores con mayor cuota ⇒ βᵢ más alta (más sensibles al ciclo).
      - Se cumple sum_i cuota_baseᵢ * βᵢ = 1, así que el shock agregado sobre
        el mercado coincide exactamente con el % del slider.

    Devuelve:
      dict {operador: beta_i}
    """
    if "cuota_base" not in snapshot_base.columns:
        raise ValueError("snapshot_base debe contener la columna 'cuota_base'.")

    shares = snapshot_base["cuota_base"].values.astype(float)
    denom = float(np.sum(shares ** 2))

    if denom <= 0:
        # Caso extremo: repartimos a partes iguales
        n = len(shares)
        betas_vec = np.ones(n, dtype=float) / n
    else:
        betas_vec = shares / denom

    return dict(zip(snapshot_base[GROUP_COL].values, betas_vec))


# =====================================================================
#  ESCENARIO 6 – SHOCK MACRO / REGULATORIO (tamaño de mercado)
# =====================================================================

with tabs[5]:
    st.subheader("Escenario 6 – Shock macro / regulación (tamaño de mercado)")

    st.markdown(
        """
En lugar de usar un modelo de regresión complicado (que daba resultados poco estables),
se adopta una regla **simple y explicable** para repartir un shock macro entre operadores:

1. Se parte de la foto base del simulador (*ingresos* y *cuotas post-fusión*).  
2. Se calcula una elasticidad macro \\( \\beta_i \\) para cada operador usando solo su **cuota actual**.  
3. Operadores con mayor cuota ⇒ \\( \\beta_i \\) mayor ⇒ más sensibles al ciclo.  
4. El shock macro se aplica sobre los **ingresos base** de cada operador.  
5. Con esta formulación se garantiza que el **mercado total** cambia exactamente el % indicado en el *slider*.
"""
    )

    st.markdown("**Definición de la elasticidad macro de cada operador:**")
    st.latex(r"""
\beta_i = \frac{\text{cuota\_base}_i}{\sum_j \text{cuota\_base}_j^2}
""")

    st.markdown("**Aplicación del shock macro sobre los ingresos:**")
    st.latex(r"""
\text{ingresos\_escenario}_i
= \text{ingresos\_base}_i \cdot \bigl(1 + \beta_i \cdot \text{shock}\bigr)
""")

    st.markdown(
        """
Donde `shock` es la variación del tamaño total del mercado (en fracción, por ejemplo \\(+0{,}10\\) = +10 %).  
Con esta regla, al sumar todos los operadores se cumple que el mercado total cambia exactamente ese porcentaje.
"""
    )

    # -------------------------------------------------------------
    # 1) Cálculo de elasticidades β_i a partir de las cuotas base
    # -------------------------------------------------------------
    betas_dict = compute_share_based_elasticities(snapshot_base)
    # Vector de betas en el mismo orden que snapshot_base
    betas = snapshot_base[GROUP_COL].map(betas_dict).values.astype(float)

    elastic_df = snapshot_base[[GROUP_COL]].copy()
    elastic_df["elasticidad_beta"] = betas

    st.markdown("### Elasticidades macro βᵢ derivadas de la cuota base")
    st.dataframe(elastic_df, use_container_width=True)

    # -------------------------------------------------------------
    # 2) Slider de shock macro
    # -------------------------------------------------------------
    shock_pct = st.slider(
        "Variación del tamaño total del mercado (%)",
        min_value=-40,
        max_value=40,
        value=0,
        step=5,
        key="esc6_shock",
    ) / 100.0

    # -------------------------------------------------------------
    # 3) Aplicación del shock a los ingresos base
    # -------------------------------------------------------------
    # Ingresos base (modelo global, mundo post-fusión)
    ingresos_base_6 = snapshot_base["ingresos_base"].values.astype(float)

    # Ingresos escenario aplicando la regla:
    # ingresos_esc_i = ingresos_base_i * (1 + beta_i * shock_pct)
    ingresos_esc_6 = ingresos_base_6 * (1.0 + betas * shock_pct)

    # Totales de mercado
    base_total6 = float(ingresos_base_6.sum())
    esc_total6 = float(ingresos_esc_6.sum())

    # Cuotas base y escenario (recalculadas por robustez)
    cuota_base6 = ingresos_base_6 / base_total6 if base_total6 > 0 else np.zeros_like(ingresos_base_6)
    cuota_esc6 = ingresos_esc_6 / esc_total6 if esc_total6 > 0 else np.zeros_like(ingresos_esc_6)

    # HHI base y escenario
    hhi_base6 = compute_hhi(cuota_base6)
    hhi_esc6 = compute_hhi(cuota_esc6)

    # -------------------------------------------------------------
    # 4) Tabla detallada por operador
    # -------------------------------------------------------------
    scen_table6 = snapshot_base.copy()
    scen_table6["elasticidad_beta"] = betas
    scen_table6["ingresos_escenario"] = ingresos_esc_6
    scen_table6["cuota_base_modelo"] = cuota_base6
    scen_table6["cuota_escenario"] = cuota_esc6

    # Ordenamos solo PARA MOSTRAR, pero mantenemos coherencia fila a fila
    scen_table6 = scen_table6.sort_values("ingresos_escenario", ascending=False)

    col_tab, col_metrics = st.columns([2, 1])

    with col_tab:
        st.markdown("### Tabla de mercado: base vs escenario (modelo de elasticidades)")
        st.dataframe(
            scen_table6[
                [
                    GROUP_COL,
                    YEAR_COL,
                    DATE_COL,
                    "ingresos_base",
                    "cuota_base_modelo",
                    "elasticidad_beta",
                    "ingresos_escenario",
                    "cuota_escenario",
                ]
            ],
            use_container_width=True,
        )

    with col_metrics:
        st.markdown("### Indicadores agregados (modelo)")
        st.metric("HHI base (modelo)", f"{hhi_base6:,.0f}")
        st.metric("HHI escenario (modelo)", f"{hhi_esc6:,.0f}")
        st.metric("Tamaño mercado base (modelo)", f"{base_total6:,.0f}")
        st.metric("Tamaño mercado escenario (modelo)", f"{esc_total6:,.0f}")

    # -------------------------------------------------------------
    # 5) Gráficos explicativos
    # -------------------------------------------------------------

    st.markdown("### Shock macro sobre el tamaño de mercado y el reparto por operador")

    # ---- Gráfico 1: tamaño total del mercado (base vs shock) ----
    total_df = pd.DataFrame(
        {
            "escenario": ["Base (modelo)", "Shock (modelo)"],
            "ingresos_totales": [base_total6, esc_total6],
        }
    )

    total_chart = (
        alt.Chart(total_df)
        .mark_bar()
        .encode(
            x=alt.X("escenario:N", title="Escenario"),
            y=alt.Y("ingresos_totales:Q", title="Ingresos totales del mercado"),
            color=alt.Color("escenario:N", legend=None),
            tooltip=[
                alt.Tooltip("escenario:N", title="Escenario"),
                alt.Tooltip("ingresos_totales:Q", title="Ingresos", format=",.0f"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(total_chart, use_container_width=True)

    # ---- Gráfico 2: Δ ingresos por operador (escenario – base) ----
    # IMPORTANTE: calculamos el delta usando la propia scen_table6,
    # de forma que operador y diferencia estén siempre alineados.
    delta_df = scen_table6[[GROUP_COL, "ingresos_base", "ingresos_escenario"]].copy()
    delta_df["delta_ingresos"] = delta_df["ingresos_escenario"] - delta_df["ingresos_base"]
    delta_df = delta_df[[GROUP_COL, "delta_ingresos"]].rename(columns={GROUP_COL: "operador"})
    delta_df = delta_df.sort_values("delta_ingresos", ascending=False)

    delta_chart = (
        alt.Chart(delta_df)
        .mark_bar()
        .encode(
            x=alt.X("operador:N", title="Operador"),
            y=alt.Y(
                "delta_ingresos:Q",
                title="Δ Ingresos (escenario – base, modelo)",
            ),
            color=alt.condition(
                "datum.delta_ingresos >= 0",
                alt.value("#2ca02c"),  # verde
                alt.value("#d62728"),  # rojo
            ),
            tooltip=[
                alt.Tooltip("operador:N", title="Operador"),
                alt.Tooltip("delta_ingresos:Q", title="Δ Ingresos", format=",.0f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(delta_chart, use_container_width=True)

    st.markdown(
        """
Con **shock = 0 %**, las barras de Δ ingresos deberían ser prácticamente **todas 0**  
(no hay diferencia entre base y escenario).  

Cuando el shock es positivo o negativo:

- Los operadores con mayor cuota (\\( \\beta_i \\) más alta) cambian más sus ingresos.  
- El total del mercado cambia exactamente el % indicado en el *slider*, por construcción.
"""
    )
