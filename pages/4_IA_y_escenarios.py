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

    # por si acaso, garantizamos num_trim y lags básicos si no existen
    if "num_trim" not in df.columns:
        df["num_trim"] = df[DATE_COL].astype(str).str[-1].astype(int)

    df = df.sort_values([GROUP_COL, YEAR_COL, "num_trim"]).reset_index(drop=True)

    if "valor_lag1" not in df.columns:
        df["valor_lag1"] = df.groupby(GROUP_COL)[TARGET_COL].shift(1).fillna(0.0)
    if "valor_lag4" not in df.columns:
        df["valor_lag4"] = df.groupby(GROUP_COL)[TARGET_COL].shift(4).fillna(0.0)

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
# MODELO GLOBAL PARA SIMULADOR (MUNDO POST-FUSIÓN)
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
    en todo el histórico. Se agregan todas las columnas numéricas a nivel
    (operador, anno, trimestre).
    """
    df = df.copy()

    mask_om = df[GROUP_COL].isin(["Orange", "Grupo MASMOVIL"])
    sub_om = df[mask_om].copy()
    rest = df[~mask_om].copy()

    if sub_om.empty:
        df2 = df.copy()
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        fused = (
            sub_om.groupby([YEAR_COL, DATE_COL], as_index=False)[num_cols]
            .sum()
        )
        fused[GROUP_COL] = "MASORANGE"
        df2 = pd.concat([rest, fused], ignore_index=True)

    num_cols2 = df2.select_dtypes(include=[np.number]).columns.tolist()
    grp_cols = [GROUP_COL, YEAR_COL, DATE_COL]

    df_agg = (
        df2.groupby(grp_cols, as_index=False)[num_cols2]
        .sum()
    )

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

st.sidebar.subheader("Parámetros ARIMA")
p = st.sidebar.number_input("p", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("d", min_value=0, max_value=2, value=1, step=1)
q = st.sidebar.number_input("q", min_value=0, max_value=5, value=1, step=1)

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

rf_model, df_op = train_rf_for_operator(df, operador_sel, feature_cols_global)

X_op = df_op[feature_cols_global].values
y_op = df_op[TARGET_COL].values
y_hat_in = rf_model.predict(X_op)

mae = mean_absolute_error(y_op, y_hat_in)
rmse = math.sqrt(mean_squared_error(y_op, y_hat_in))

colm1, colm2 = st.columns(2)
with colm1:
    st.metric("MAE (in-sample)", f"{mae:,.2f}")
with colm2:
    st.metric("RMSE (in-sample)", f"{rmse:,.2f}")

plot_df = pd.DataFrame({
    "periodo": df_op[DATE_COL].astype(str),
    "Real": y_op,
    "Predicho_ML": y_hat_in,
}).set_index("periodo")

st.line_chart(
    plot_df[["Real", "Predicho_ML"]],
    height=320,
)

# Importancias de variables
importances = rf_model.feature_importances_
imp_df = pd.DataFrame({"feature": feature_cols_global, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False)

st.subheader("Importancia de variables (IA explicable)")
st.bar_chart(
    imp_df.set_index("feature")["importance"],
    height=250,
)

# =====================================================
# 2. FORECAST TEMPORAL Y ESCENARIO SIMPLE (modelo ML autoregresivo)
# =====================================================

st.markdown("---")
st.header("2. Forecast temporal y escenario simple (modelo ML por operador)")

st.markdown(
    """
En este apartado se construye un **forecast autoregresivo ML** por operador:

- El modelo solo usa como entrada los **últimos 4 trimestres de ingresos** (lags).  
- No se usan exógenos futuros (BAM, portabilidades, etc.), para evitar supuestos
  fuertes que no podemos justificar.  
- Se muestran dos curvas:
  - **Baseline_ML**: forecast “inercial” del modelo.  
  - **Escenario_ML**: mismo forecast, pero aplicando un **shock** multiplicativo
    al primer trimestre futuro (el del slider lateral).
"""
)

# Serie histórica de ingresos del operador
y_hist = df_op[TARGET_COL].values.astype(float)

# Dataset autoregresivo: X = [y_{t-1}..y_{t-4}], y = y_t
X_ar, y_ar = build_ar_lag_dataset(y_hist, n_lags=N_LAGS)

if len(y_ar) < 5:
    st.warning("Hay muy pocos datos para un forecast autoregresivo estable.")
else:
    # Modelo ML autoregresivo (RandomForest pequeño)
    rf_ar = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    rf_ar.fit(X_ar, y_ar)

    # Forecast baseline (sin shock) y escenario (con shock en el primer paso)
    baseline_preds = iterative_forecast_ar(
        history_y=y_hist,
        model=rf_ar,
        n_lags=N_LAGS,
        horizon=horizon,
        shock_first=None,
    )
    scenario_preds = iterative_forecast_ar(
        history_y=y_hist,
        model=rf_ar,
        n_lags=N_LAGS,
        horizon=horizon,
        shock_first=shock_pct,
    )

    future_labels = generate_future_quarters(df_op, len(baseline_preds))

    # DF para gráfico: histórico + futuro
    fc_df = pd.DataFrame({
        "periodo": list(df_op[DATE_COL].astype(str)) + future_labels,
        "Historico": list(y_hist) + [np.nan] * len(baseline_preds),
        "Baseline_ML": [np.nan] * len(y_hist) + baseline_preds,
        "Escenario_ML": [np.nan] * len(y_hist) + scenario_preds,
    })

    st.line_chart(
        fc_df.set_index("periodo")[["Historico", "Baseline_ML", "Escenario_ML"]],
        height=340,
    )

    st.markdown("### Detalle numérico del forecast ML (autoregresivo)")
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
- **Baseline_ML**: forecast autoregresivo puro (RandomForest sobre lags 1–4).  
- **Escenario_ML**: mismo forecast, pero aplicando un *shock* multiplicativo al
  **primer trimestre futuro**, que se propaga por los lags.
"""
    )

# =====================================================
# 3. FORECAST CLÁSICO ARIMA
# =====================================================

st.markdown("---")
st.header("3. Forecast clásico con ARIMA (con bandas de confianza)")

try:
    arima_results = train_arima(df_op[TARGET_COL], order=(p, d, q))
except Exception as e:
    st.error(f"No se ha podido ajustar ARIMA({p},{d},{q}): {e}")
    arima_results = None

if arima_results is not None:
    fc_arima = arima_results.get_forecast(steps=horizon)
    mean_fc = fc_arima.predicted_mean
    conf_int = fc_arima.conf_int(alpha=0.05)

    future_labels_arima = [f"Fut_ARIMA_{i+1}" for i in range(len(mean_fc))]

    arima_df = pd.DataFrame({
        "periodo": list(df_op[DATE_COL].astype(str)) + future_labels_arima,
        "Histórico": list(df_op[TARGET_COL]) + [np.nan] * len(mean_fc),
        "Forecast_ARIMA": [np.nan] * len(df_op) + mean_fc.tolist(),
        "Lower_95": [np.nan] * len(df_op) + conf_int.iloc[:, 0].tolist(),
        "Upper_95": [np.nan] * len(df_op) + conf_int.iloc[:, 1].tolist(),
    })

    st.line_chart(
        arima_df.set_index("periodo")[["Histórico", "Forecast_ARIMA"]],
        height=320,
    )

    st.markdown("Tabla con el forecast ARIMA y bandas de confianza (95%):")
    st.dataframe(
        arima_df.tail(horizon)[["Forecast_ARIMA", "Lower_95", "Upper_95"]],
        use_container_width=True,
    )

# =====================================================
# 4. DETECCIÓN DE ANOMALÍAS
# =====================================================

st.markdown("---")
st.header("4. Detección de anomalías (IsolationForest sobre residuales ML)")

resid = y_op - y_hat_in
labels = detect_anomalies(resid, contamination=contamination)
anom_flags = np.where(labels == -1, "Anómalo", "Normal")

anom_df = pd.DataFrame({
    "periodo": df_op[DATE_COL].astype(str),
    "y": y_op,
    "residual": resid,
    "estado": anom_flags,
})

ts_anom_plot = anom_df.copy()
ts_anom_plot["y_anomalo"] = np.where(
    ts_anom_plot["estado"] == "Anómalo",
    ts_anom_plot["y"],
    np.nan,
)

st.line_chart(
    ts_anom_plot.set_index("periodo")[["y", "y_anomalo"]],
    height=340,
)

st.markdown(
    """
Los puntos marcados como **anómalos** corresponden a trimestres cuyos residuales
(no explicados por el modelo ML) resultan inusuales según IsolationForest.
"""
)

st.subheader("Detalle de observaciones anómalas")
st.dataframe(
    anom_df[anom_df["estado"] == "Anómalo"].reset_index(drop=True),
    use_container_width=True,
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

snapshot_features = snapshot[feat_cols_global].copy()

baseline_pred = global_model.predict(snapshot_features.values)
baseline_total = baseline_pred.sum()
baseline_shares = baseline_pred / baseline_total
baseline_hhi = compute_hhi(baseline_shares)

snapshot_base = snapshot[[GROUP_COL, YEAR_COL, DATE_COL]].copy()
snapshot_base["ingresos_base"] = baseline_pred
snapshot_base["cuota_base"] = baseline_shares

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
Se modeliza la inversión con una **regla de negocio**:

> Inversión +X% ⇒ ingresos operador ≈ ingresos_base × (1 + 0,5·X)

- elasticidad positiva (0,5),
- el resto de operadores permanece constante.
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

    elasticidad_ing = 0.5

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

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.dataframe(scen_table, use_container_width=True)
    with col_right:
        st.metric("HHI base", f"{baseline_hhi:,.0f}")
        st.metric("HHI escenario", f"{scen_hhi:,.0f}")
        base_q = float(scen_table.loc[scen_table[GROUP_COL] == op_inv, "cuota_base"])
        scen_q = float(scen_table.loc[scen_table[GROUP_COL] == op_inv, "cuota_escenario"])
        st.metric(f"Δ cuota {op_inv}", f"{(scen_q - base_q)*100:,.2f} p.p.")

# ------------------ Escenario 2 ------------------ #
with tabs[1]:
    st.subheader("Escenario 2 – Guerra de portabilidades")

    st.markdown(
        """
Simula una **campaña agresiva de captación**:

- un operador aumenta sus portabilidades,
- opcionalmente otro sufre una reducción.
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

    scen_feat2 = snapshot_features.copy()
    if "men_portab_moviles" in scen_feat2.columns:
        scen_feat2.loc[snapshot_base[GROUP_COL] == op_pro, "men_portab_moviles"] *= (1.0 + delta_pro)
        if op_vic != "(Ninguno)":
            scen_feat2.loc[snapshot_base[GROUP_COL] == op_vic, "men_portab_moviles"] *= (1.0 - delta_vic)

    scen_pred2 = global_model.predict(scen_feat2.values)
    scen_total2 = scen_pred2.sum()
    scen_shares2 = scen_pred2 / scen_total2
    scen_hhi2 = compute_hhi(scen_shares2)

    scen_table2 = snapshot_base.copy()
    scen_table2["ingresos_escenario"] = scen_pred2
    scen_table2["cuota_escenario"] = scen_shares2
    scen_table2 = scen_table2.sort_values("ingresos_escenario", ascending=False)

    st.dataframe(scen_table2, use_container_width=True)
    st.metric("HHI base", f"{baseline_hhi:,.0f}")
    st.metric("HHI escenario", f"{scen_hhi2:,.0f}")

# ------------------ Escenario 3 ------------------ #
with tabs[2]:
    st.subheader("Escenario 3 – Expansión operador low-cost")

    st.markdown(
        """
Pensado para operadores tipo **Digi** o similares: se incrementan portabilidades
y base de líneas (móvil y BAM), simulando una fase de expansión agresiva.
"""
    )

    low_candidates = [op for op in snapshot_base[GROUP_COL].unique().tolist()
                      if "Digi" in op or "Resto" in op or "Low" in op]
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
        "Incremento base de líneas (%)",
        min_value=0,
        max_value=200,
        value=60,
        step=10,
        key="esc3_lineas",
    ) / 100.0

    scen_feat3 = snapshot_features.copy()
    mask_low = snapshot_base[GROUP_COL] == op_low

    if "men_portab_moviles" in scen_feat3.columns:
        scen_feat3.loc[mask_low, "men_portab_moviles"] *= (1.0 + delta_porta)
    if "an_merc_mov_lineas" in scen_feat3.columns:
        scen_feat3.loc[mask_low, "an_merc_mov_lineas"] *= (1.0 + delta_lineas)
    if "an_merc_bam_lineas" in scen_feat3.columns:
        scen_feat3.loc[mask_low, "an_merc_bam_lineas"] *= (1.0 + delta_lineas)

    scen_pred3 = global_model.predict(scen_feat3.values)
    scen_total3 = scen_pred3.sum()
    scen_shares3 = scen_pred3 / scen_total3
    scen_hhi3 = compute_hhi(scen_shares3)

    scen_table3 = snapshot_base.copy()
    scen_table3["ingresos_escenario"] = scen_pred3
    scen_table3["cuota_escenario"] = scen_shares3
    scen_table3 = scen_table3.sort_values("ingresos_escenario", ascending=False)

    st.dataframe(scen_table3, use_container_width=True)
    st.metric("HHI base", f"{baseline_hhi:,.0f}")
    st.metric("HHI escenario", f"{scen_hhi3:,.0f}")

# ------------------ Escenario 4 ------------------ #
with tabs[3]:
    st.subheader("Escenario 4 – Recorte de inversión / austeridad")

    st.markdown(
        """
Recorte de inversión modelizado como el inverso del escenario 3:

- menos inversión ⇒ menos líneas ⇒ menos portabilidades.
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
        step=10,
        key="esc4_rec",
    ) / 100.0

    scen_feat4 = snapshot_features.copy()
    mask_rec = snapshot_base[GROUP_COL] == op_rec

    if "an_merc_mov_lineas" in scen_feat4.columns:
        scen_feat4.loc[mask_rec, "an_merc_mov_lineas"] *= (1.0 - 0.5 * delta_rec)
    if "an_merc_bam_lineas" in scen_feat4.columns:
        scen_feat4.loc[mask_rec, "an_merc_bam_lineas"] *= (1.0 - 0.5 * delta_rec)
    if "men_portab_moviles" in scen_feat4.columns:
        scen_feat4.loc[mask_rec, "men_portab_moviles"] *= (1.0 - 0.3 * delta_rec)

    scen_pred4 = global_model.predict(scen_feat4.values)
    scen_total4 = scen_pred4.sum()
    scen_shares4 = scen_pred4 / scen_total4
    scen_hhi4 = compute_hhi(scen_shares4)

    scen_table4 = snapshot_base.copy()
    scen_table4["ingresos_escenario"] = scen_pred4
    scen_table4["cuota_escenario"] = scen_shares4
    scen_table4 = scen_table4.sort_values("ingresos_escenario", ascending=False)

    st.dataframe(scen_table4, use_container_width=True)
    st.metric("HHI base", f"{baseline_hhi:,.0f}")
    st.metric("HHI escenario", f"{scen_hhi4:,.0f}")

# ------------------ Escenario 5 ------------------ #
with tabs[4]:
    st.subheader("Escenario 5 – Fusión o joint-venture (cambio en HHI)")

    st.markdown(
        """
Permite simular **futuras fusiones adicionales** (por ejemplo MASORANGE+Vodafone),
sumando ingresos y recalculando el HHI.
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

    base_ops = snapshot_base[GROUP_COL].tolist()
    base_ing = snapshot_base["ingresos_base"].tolist()

    scen_ing_dict = {}
    for op, y in zip(base_ops, base_ing):
        if op in (op_f1, op_f2):
            continue
        scen_ing_dict[op] = y

    y1 = float(snapshot_base.loc[snapshot_base[GROUP_COL] == op_f1, "ingresos_base"])
    y2 = float(snapshot_base.loc[snapshot_base[GROUP_COL] == op_f2, "ingresos_base"])
    y_fus = (y1 + y2) * (1.0 + delta_sin)
    op_fus = f"{op_f1}+{op_f2}"
    scen_ing_dict[op_fus] = y_fus

    scen_ops = list(scen_ing_dict.keys())
    scen_ing = np.array(list(scen_ing_dict.values()))
    scen_total5 = scen_ing.sum()
    scen_shares5 = scen_ing / scen_total5
    scen_hhi5 = compute_hhi(scen_shares5)

    scen_table5 = pd.DataFrame({
        "operador": scen_ops,
        "ingresos_escenario": scen_ing,
        "cuota_escenario": scen_shares5,
    }).sort_values("ingresos_escenario", ascending=False)

    st.metric("HHI base (post-fusión actual)", f"{baseline_hhi:,.0f}")
    st.metric("HHI tras nueva fusión", f"{scen_hhi5:,.0f}")
    st.dataframe(scen_table5, use_container_width=True)

# ------------------ Escenario 6 ------------------ #
with tabs[5]:
    st.subheader("Escenario 6 – Shock macro / regulación (tamaño de mercado)")

    st.markdown(
        """
Simula un **shock global** (crisis, regulación de precios, etc.) que cambia el
tamaño total del mercado (`tri_ingresos_total_trimestre`). El modelo global reparte
ese ajuste entre operadores.
"""
    )

    delta_macro = st.slider(
        "Variación del tamaño total del mercado (%)",
        min_value=-40,
        max_value=40,
        value=-10,
        step=5,
        key="esc6_macro",
    ) / 100.0

    scen_feat6 = snapshot_features.copy()
    if "tri_ingresos_total_trimestre" in scen_feat6.columns:
        scen_feat6["tri_ingresos_total_trimestre"] *= (1.0 + delta_macro)

    scen_pred6 = global_model.predict(scen_feat6.values)
    scen_total6 = scen_pred6.sum()
    scen_shares6 = scen_pred6 / scen_total6
    scen_hhi6 = compute_hhi(scen_shares6)

    scen_table6 = snapshot_base.copy()
    scen_table6["ingresos_escenario"] = scen_pred6
    scen_table6["cuota_escenario"] = scen_shares6
    scen_table6 = scen_table6.sort_values("ingresos_escenario", ascending=False)

    st.dataframe(scen_table6, use_container_width=True)
    st.metric("HHI base", f"{baseline_hhi:,.0f}")
    st.metric("HHI escenario", f"{scen_hhi6:,.0f}")
