import os
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


# =====================================================
# CONFIGURACIÓN BÁSICA
# =====================================================

DATA_PATH = "data/model_input/ia_trimestral_model.csv"

DATE_COL = "trimestre"   # periodo tipo '2018T3'
YEAR_COL = "anno"
GROUP_COL = "operador"
TARGET_COL = "valor"

# nº de retardos del target que usaremos como features
N_LAGS = 4


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
    return df


def get_exog_columns(df: pd.DataFrame) -> List[str]:
    """
    Devuelve las columnas de features exógenas (todas las numéricas excepto target y claves).
    """
    key_cols = {DATE_COL, YEAR_COL, GROUP_COL, TARGET_COL}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exog_cols = [c for c in num_cols if c not in key_cols]
    return exog_cols


def compute_hhi(shares: np.ndarray) -> float:
    """
    Calcula el índice HHI a partir de cuotas (en fracción, no en %).
    Devuelve HHI en puntos (0–10 000).
    """
    shares_pct = shares * 100.0
    return float(np.sum(shares_pct ** 2))


# =====================================================
# FUNCIONES PARA MODELO ML POR OPERADOR (DINÁMICO)
# =====================================================

def prepare_series(df: pd.DataFrame, operador: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Filtra por operador y ordena por trimestre.
    Devuelve un DataFrame 'ts' con:
      periodo, anno, operador, y (target) + exog_cols
    y la lista de columnas exógenas.
    """
    sub = df[df[GROUP_COL] == operador].copy()
    if sub.empty:
        raise ValueError("No hay datos para el operador seleccionado.")

    sub = sub.sort_values(DATE_COL)

    exog_cols = get_exog_columns(sub)

    cols = [DATE_COL, YEAR_COL, GROUP_COL, TARGET_COL] + exog_cols
    ts = sub[cols].copy()
    ts = ts.rename(columns={DATE_COL: "periodo", TARGET_COL: "y"})

    # Índice temporal artificial
    ts = ts.reset_index(drop=True)
    ts["t"] = np.arange(len(ts))

    return ts, exog_cols


def build_ml_dataset(
    ts: pd.DataFrame,
    exog_cols: List[str],
    n_lags: int
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Añade lags del target y devuelve:
      - ts_ml: dataset listo para ML
      - lag_cols: lista de columnas lag_*
      - feature_cols: lag_cols + exog_cols
    """
    ts_ml = ts.copy()
    for lag in range(1, n_lags + 1):
        ts_ml[f"lag_{lag}"] = ts_ml["y"].shift(lag)

    # quitar filas con NA por los lags
    ts_ml = ts_ml.dropna().reset_index(drop=True)

    lag_cols = [f"lag_{lag}" for lag in range(1, n_lags + 1)]
    feature_cols = lag_cols + exog_cols

    return ts_ml, lag_cols, feature_cols


def train_rf_model(ts_ml: pd.DataFrame, feature_cols: List[str]):
    """
    Entrena un RandomForestRegressor con las columnas indicadas.
    Devuelve el modelo y predicciones in-sample.
    """
    X = ts_ml[feature_cols].values
    y = ts_ml["y"].values

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        min_samples_leaf=2,
    )
    model.fit(X, y)

    y_hat = model.predict(X)

    return model, y_hat


def rolling_forecast_rf(
    model,
    history_y: List[float],
    exog_vector: np.ndarray,
    lag_cols: List[str],
    exog_cols: List[str],
    horizon: int,
):
    """
    Forecast iterativo con RandomForest.
    - history_y: lista con el histórico del target
    - exog_vector: valores exógenos que usaremos para todos los pasos futuros
    """
    preds = []
    hist = history_y.copy()
    n_lags = len(lag_cols)

    for _ in range(horizon):
        if len(hist) < n_lags:
            break
        lags = np.array(hist[-n_lags:])
        x = np.concatenate([lags, exog_vector]).reshape(1, -1)
        y_hat = model.predict(x)[0]
        preds.append(y_hat)
        hist.append(y_hat)

    return preds


def rolling_forecast_rf_scenario(
    model,
    history_y: List[float],
    exog_vector: np.ndarray,
    lag_cols: List[str],
    exog_cols: List[str],
    horizon: int,
    shock_pct: float,
):
    """
    Igual que rolling_forecast_rf, pero aplicando un shock al último valor
    histórico (escenario de negocio muy simple).
    """
    preds = []
    hist = history_y.copy()
    n_lags = len(lag_cols)

    if len(hist) >= 1:
        hist[-1] = hist[-1] * (1 + shock_pct)

    for _ in range(horizon):
        if len(hist) < n_lags:
            break
        lags = np.array(hist[-n_lags:])
        x = np.concatenate([lags, exog_vector]).reshape(1, -1)
        y_hat = model.predict(x)[0]
        preds.append(y_hat)
        hist.append(y_hat)

    return preds


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
# MODELO GLOBAL PARA SIMULADOR (SIN LAGS)
# =====================================================

@st.cache_data
def train_global_rf(df: pd.DataFrame):
    """
    Entrena un modelo RandomForest global (todos los operadores, todos los trimestres)
    usando SOLO features exógenas (sin lags) para escenarios estáticos.
    """
    exog_cols_glob = get_exog_columns(df)
    X = df[exog_cols_glob].values
    y = df[TARGET_COL].values

    model = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        min_samples_leaf=3,
    )
    model.fit(X, y)
    return model, exog_cols_glob


def get_latest_by_operator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene la última observación disponible por operador.
    Esto sirve como "base" para escenarios (punto de partida).
    """
    df_sorted = df.sort_values([GROUP_COL, YEAR_COL, DATE_COL])
    latest = df_sorted.groupby(GROUP_COL, as_index=False).tail(1)
    return latest.reset_index(drop=True)


# =====================================================
# INTERFAZ STREAMLIT
# =====================================================

st.title("Módulo de IA: Predicción, Escenarios y Anomalías")

st.markdown(
    """
Esta página utiliza el dataset integrado **ia_trimestral_model.csv** para:

1. Entrenar un **modelo ML explicable (RandomForest, scikit-learn)** por operador.  
2. Generar **escenarios de negocio** modificando variables clave (inversión, portabilidades, infra…).  
3. Producir un **forecast clásico ARIMA** con bandas de confianza.  
4. Detectar **anomalías** en la evolución histórica (IsolationForest).
"""
)

# ------------------ carga de datos ------------------

try:
    df = load_ia_dataset(DATA_PATH)
except Exception as e:
    st.error(f"Error cargando el dataset de IA: {e}")
    st.stop()

# ------------------ sidebar -------------------------

st.sidebar.header("Configuración IA (modelo temporal por operador)")

operadores = sorted(df[GROUP_COL].dropna().unique().tolist())
operador_sel = st.sidebar.selectbox("Operador", operadores)

horizon = st.sidebar.slider(
    "Horizonte de predicción (trimestres)",
    min_value=1,
    max_value=8,
    value=4,
)

shock_pct = st.sidebar.slider(
    "Shock sobre el último valor histórico (%)",
    min_value=-30,
    max_value=30,
    value=0,
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
    ts, exog_cols = prepare_series(df, operador_sel)
except Exception as e:
    st.error(f"Error preparando la serie temporal: {e}")
    st.stop()

if len(ts) < N_LAGS + 4:
    st.warning(
        f"La serie tiene pocas observaciones ({len(ts)}). "
        "Las predicciones pueden no ser muy estables."
    )

st.subheader("Serie temporal de ingresos por operador")

col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(
        ts[["periodo", "anno", "y"]].rename(columns={"y": "ingresos_trimestrales"}),
        use_container_width=True,
    )

with col2:
    st.metric("Observaciones", len(ts))
    st.metric("Mínimo ingresos", f"{ts['y'].min():,.0f}")
    st.metric("Máximo ingresos", f"{ts['y'].max():,.0f}")

st.line_chart(
    ts.set_index("periodo")["y"],
    height=260,
)

# =====================================================
# 1. MODELO ML EXPLICABLE (RandomForest por operador)
# =====================================================

st.markdown("---")
st.header("1. Modelo ML explicable (RandomForest por operador)")

ts_ml, lag_cols, feature_cols = build_ml_dataset(ts, exog_cols, N_LAGS)

if ts_ml.empty:
    st.error("No hay suficientes datos para construir lags para el modelo ML.")
    st.stop()

rf_model, y_hat_in = train_rf_model(ts_ml, feature_cols)

y_true = ts_ml["y"].values

mae = mean_absolute_error(y_true, y_hat_in)
rmse = math.sqrt(mean_squared_error(y_true, y_hat_in))

colm1, colm2 = st.columns(2)
with colm1:
    st.metric("MAE (in-sample)", f"{mae:,.2f}")
with colm2:
    st.metric("RMSE (in-sample)", f"{rmse:,.2f}")

plot_df = ts_ml[["periodo"]].copy()
plot_df["Real"] = y_true
plot_df["Predicho_ML"] = y_hat_in

st.line_chart(
    plot_df.set_index("periodo")[["Real", "Predicho_ML"]],
    height=320,
)

# Importancias de variables
importances = rf_model.feature_importances_
imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False)

st.subheader("Importancia de variables (IA explicable)")
st.bar_chart(
    imp_df.set_index("feature")["importance"],
    height=250,
)

# =====================================================
# 2. FORECAST Y ESCENARIOS TEMPORALES CON ML
# =====================================================

st.markdown("---")
st.header("2. Forecast temporal y escenario simple (modelo ML por operador)")

history_y = ts["y"].tolist()
# Usamos los exógenos del último trimestre como referencia futura
last_exog_vector = ts.iloc[-1][exog_cols].values.astype(float)

baseline_preds = rolling_forecast_rf(
    rf_model,
    history_y,
    last_exog_vector,
    lag_cols,
    exog_cols,
    horizon=horizon,
)

scenario_preds = rolling_forecast_rf_scenario(
    rf_model,
    history_y,
    last_exog_vector,
    lag_cols,
    exog_cols,
    horizon=horizon,
    shock_pct=shock_pct,
)

future_labels = [f"Futuro_{i+1}" for i in range(len(baseline_preds))]

fc_df = pd.DataFrame({
    "periodo": list(ts["periodo"]) + future_labels,
    "Histórico": list(ts["y"]) + [np.nan] * len(baseline_preds),
    "Baseline_ML": [np.nan] * len(ts) + baseline_preds,
    "Escenario_ML": [np.nan] * len(ts) + scenario_preds,
})

st.line_chart(
    fc_df.set_index("periodo")[["Histórico", "Baseline_ML", "Escenario_ML"]],
    height=340,
)

st.markdown(
    """
- **Baseline ML**: predicción iterativa utilizando el modelo RandomForest y los exógenos del último trimestre.  
- **Escenario ML**: mismo modelo, pero aplicando un *shock* multiplicativo al último dato histórico de ingresos.
"""
)

# =====================================================
# 3. FORECAST CLÁSICO ARIMA
# =====================================================

st.markdown("---")
st.header("3. Forecast clásico con ARIMA (con bandas de confianza)")

try:
    arima_results = train_arima(ts["y"], order=(p, d, q))
except Exception as e:
    st.error(f"No se ha podido ajustar ARIMA({p},{d},{q}): {e}")
    arima_results = None

if arima_results is not None:
    fc_arima = arima_results.get_forecast(steps=horizon)
    mean_fc = fc_arima.predicted_mean
    conf_int = fc_arima.conf_int(alpha=0.05)

    future_labels_arima = [f"Fut_ARIMA_{i+1}" for i in range(len(mean_fc))]

    arima_df = pd.DataFrame({
        "periodo": list(ts["periodo"]) + future_labels_arima,
        "Histórico": list(ts["y"]) + [np.nan] * len(mean_fc),
        "Forecast_ARIMA": [np.nan] * len(ts) + mean_fc.tolist(),
        "Lower_95": [np.nan] * len(ts) + conf_int.iloc[:, 0].tolist(),
        "Upper_95": [np.nan] * len(ts) + conf_int.iloc[:, 1].tolist(),
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

resid = y_true - y_hat_in
labels = detect_anomalies(resid, contamination=contamination)
anom_flags = np.where(labels == -1, "Anómalo", "Normal")

anom_df = ts_ml[["periodo"]].copy()
anom_df["y"] = y_true
anom_df["residual"] = resid
anom_df["estado"] = anom_flags

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
# 5. SIMULADOR DE ESCENARIOS DE NEGOCIO (MODELO GLOBAL)
# =====================================================

st.markdown("---")
st.header("5. Simulador de escenarios de negocio (RandomForest global)")

st.markdown(
    """
En esta sección se utiliza un **RandomForest global** (entrenado con todos los operadores
y todos los trimestres, solo con features exógenas) para simular distintos escenarios
de decisión de negocio:

- cambios en inversión,
- campañas agresivas de portabilidades,
- expansión de operadores low-cost,
- recortes de inversión,
- fusiones y shocks macro en el tamaño del mercado.

En todos los casos se parte de las **últimas observaciones disponibles** por operador.
"""
)

# Entrenamos el modelo global y preparamos la base de escenarios
rf_global, exog_cols_global = train_global_rf(df)
latest = get_latest_by_operator(df)

# predicción base (situación actual)
baseline_pred = rf_global.predict(latest[exog_cols_global].values)
baseline_total = baseline_pred.sum()
baseline_shares = baseline_pred / baseline_total
baseline_hhi = compute_hhi(baseline_shares)

latest_base = latest[[GROUP_COL, YEAR_COL, DATE_COL]].copy()
latest_base["ingresos_base"] = baseline_pred
latest_base["cuota_base"] = baseline_shares

tabs = st.tabs([
    "Plan de inversión agresivo",
    "Guerra de portabilidades",
    "Expansión operador low-cost",
    "Recorte de inversión",
    "Fusión y HHI",
    "Shock macro mercado",
])

# ------------------ Escenario 1: Plan de inversión agresivo ------------------ #
with tabs[0]:
    st.subheader("Escenario 1 – Plan de inversión agresivo")

    st.markdown(
        """
Simula el impacto de **aumentar la inversión** de un operador en sus ingresos
y en su cuota de mercado, manteniendo al resto constantes.
Se modifica la variable `an_gen_inversiones` (y opcionalmente infra básica).
"""
    )

    op_inv = st.selectbox(
        "Operador que incrementa la inversión",
        operadores,
        key="esc1_operador",
    )
    delta_inv = st.slider(
        "Incremento de inversión anual (%)",
        min_value=-50,
        max_value=100,
        value=20,
        step=5,
        key="esc1_inv",
    ) / 100.0
    delta_infra = st.slider(
        "Incremento en infraestructuras básicas (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        key="esc1_infra",
    ) / 100.0

    scen_df = latest.copy()

    mask_op = scen_df[GROUP_COL] == op_inv
    if "an_gen_inversiones" in scen_df.columns:
        scen_df.loc[mask_op, "an_gen_inversiones"] *= (1.0 + delta_inv)
    # subimos estaciones base y líneas BAM como proxy de despliegue
    if "inf_estaciones_base" in scen_df.columns:
        scen_df.loc[mask_op, "inf_estaciones_base"] *= (1.0 + delta_infra)
    if "inf_bam_lineas" in scen_df.columns:
        scen_df.loc[mask_op, "inf_bam_lineas"] *= (1.0 + delta_infra)

    scen_pred = rf_global.predict(scen_df[exog_cols_global].values)
    scen_total = scen_pred.sum()
    scen_shares = scen_pred / scen_total
    scen_hhi = compute_hhi(scen_shares)

    scen_table = scen_df[[GROUP_COL, YEAR_COL, DATE_COL]].copy()
    scen_table["ingresos_base"] = baseline_pred
    scen_table["cuota_base"] = baseline_shares
    scen_table["ingresos_escenario"] = scen_pred
    scen_table["cuota_escenario"] = scen_shares
    scen_table = scen_table.sort_values("ingresos_escenario", ascending=False)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.dataframe(
            scen_table,
            use_container_width=True,
        )

    with col_right:
        st.metric("HHI base", f"{baseline_hhi:,.0f}")
        st.metric("HHI escenario", f"{scen_hhi:,.0f}")
        st.metric(
            f"Δ cuota {op_inv}",
            f"{float(scen_table.loc[scen_table[GROUP_COL] == op_inv, 'cuota_escenario']) * 100 - float(scen_table.loc[scen_table[GROUP_COL] == op_inv, 'cuota_base']) * 100:,.2f} p.p."
            if op_inv in scen_table[GROUP_COL].values
            else "N/A",
        )

    st.markdown(
        """
**Lectura para el TFM**: este escenario permite analizar cómo cambios en la inversión
se traducen, según el modelo, en aumentos o reducciones de cuota de mercado y en la
concentración del mercado (HHI).
"""
    )

# ------------------ Escenario 2: Guerra de portabilidades ------------------ #
with tabs[1]:
    st.subheader("Escenario 2 – Guerra de portabilidades")

    st.markdown(
        """
Simula una **campaña agresiva de captación**:
un operador aumenta sus portabilidades (`men_portab_moviles`) y, opcionalmente,
se detrae parte de esa captación de un competidor concreto.
"""
    )

    op_pro = st.selectbox(
        "Operador que lanza la campaña",
        operadores,
        key="esc2_op_pro",
    )
    op_vic = st.selectbox(
        "Operador que pierde portabilidades (opcional)",
        ["(Ninguno)"] + operadores,
        key="esc2_op_vic",
    )
    delta_porta_pro = st.slider(
        "Incremento portabilidades operador protagonista (%)",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        key="esc2_delta_pro",
    ) / 100.0
    delta_porta_vic = st.slider(
        "Reducción portabilidades operador víctima (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        key="esc2_delta_vic",
    ) / 100.0

    scen_df2 = latest.copy()

    if "men_portab_moviles" in scen_df2.columns:
        scen_df2.loc[scen_df2[GROUP_COL] == op_pro, "men_portab_moviles"] *= (1.0 + delta_porta_pro)
        if op_vic != "(Ninguno)":
            scen_df2.loc[scen_df2[GROUP_COL] == op_vic, "men_portab_moviles"] *= (1.0 - delta_porta_vic)

    scen_pred2 = rf_global.predict(scen_df2[exog_cols_global].values)
    scen_total2 = scen_pred2.sum()
    scen_shares2 = scen_pred2 / scen_total2
    scen_hhi2 = compute_hhi(scen_shares2)

    scen_table2 = scen_df2[[GROUP_COL, YEAR_COL, DATE_COL]].copy()
    scen_table2["ingresos_base"] = baseline_pred
    scen_table2["cuota_base"] = baseline_shares
    scen_table2["ingresos_escenario"] = scen_pred2
    scen_table2["cuota_escenario"] = scen_shares2
    scen_table2 = scen_table2.sort_values("ingresos_escenario", ascending=False)

    st.dataframe(
        scen_table2,
        use_container_width=True,
    )

    st.metric("HHI base", f"{baseline_hhi:,.0f}")
    st.metric("HHI escenario", f"{scen_hhi2:,.0f}")

    st.markdown(
        """
Este escenario ilustra el efecto de una **guerra comercial de portabilidades**:
cómo cambia la distribución de ingresos y cuotas si un operador incrementa su
capacidad de captación a costa de otros.
"""
    )

# ------------------ Escenario 3: Expansión operador low-cost ------------------ #
with tabs[2]:
    st.subheader("Escenario 3 – Expansión de operador low-cost")

    st.markdown(
        """
Pensado para operadores tipo **Digi / Grupo MASMOVIL**: se incrementan simultáneamente
portabilidades, líneas de BAM e inversiones, simulando una fase de expansión agresiva.
"""
    )

    # Filtramos una lista orientativa de "low-cost" (puedes cambiar nombres según tu dataset)
    posibles_lowcost = [op for op in operadores if "Digi" in op or "MAS" in op or "Resto" in op] or operadores
    op_low = st.selectbox(
        "Operador low-cost / retador",
        posibles_lowcost,
        key="esc3_op_low",
    )

    delta_porta = st.slider(
        "Incremento portabilidades (%)",
        min_value=0,
        max_value=150,
        value=40,
        step=10,
        key="esc3_porta",
    ) / 100.0
    delta_lineas_bam = st.slider(
        "Incremento líneas BAM (%)",
        min_value=0,
        max_value=150,
        value=40,
        step=10,
        key="esc3_bam",
    ) / 100.0
    delta_inv_low = st.slider(
        "Incremento inversiones (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=10,
        key="esc3_inv",
    ) / 100.0

    scen_df3 = latest.copy()
    mask_low = scen_df3[GROUP_COL] == op_low

    if "men_portab_moviles" in scen_df3.columns:
        scen_df3.loc[mask_low, "men_portab_moviles"] *= (1.0 + delta_porta)
    if "inf_bam_lineas" in scen_df3.columns:
        scen_df3.loc[mask_low, "inf_bam_lineas"] *= (1.0 + delta_lineas_bam)
    if "an_gen_inversiones" in scen_df3.columns:
        scen_df3.loc[mask_low, "an_gen_inversiones"] *= (1.0 + delta_inv_low)

    scen_pred3 = rf_global.predict(scen_df3[exog_cols_global].values)
    scen_total3 = scen_pred3.sum()
    scen_shares3 = scen_pred3 / scen_total3
    scen_hhi3 = compute_hhi(scen_shares3)

    scen_table3 = scen_df3[[GROUP_COL, YEAR_COL, DATE_COL]].copy()
    scen_table3["ingresos_base"] = baseline_pred
    scen_table3["cuota_base"] = baseline_shares
    scen_table3["ingresos_escenario"] = scen_pred3
    scen_table3["cuota_escenario"] = scen_shares3
    scen_table3 = scen_table3.sort_values("ingresos_escenario", ascending=False)

    st.dataframe(
        scen_table3,
        use_container_width=True,
    )

    st.metric("HHI base", f"{baseline_hhi:,.0f}")
    st.metric("HHI escenario", f"{scen_hhi3:,.0f}")

    st.markdown(
        """
Escenario directamente interpretable como **expansión de un operador low-cost**,
con impacto tanto en su cuota como en la concentración global del mercado.
"""
    )

# ------------------ Escenario 4: Recorte de inversión ------------------ #
with tabs[3]:
    st.subheader("Escenario 4 – Recorte de inversión / austeridad")

    st.markdown(
        """
Simula un escenario de **austeridad** en el que un operador reduce fuertemente
sus inversiones, y posiblemente deja de ampliar infraestructuras.
"""
    )

    op_rec = st.selectbox(
        "Operador que recorta inversión",
        operadores,
        key="esc4_op_rec",
    )
    delta_rec_inv = st.slider(
        "Reducción de inversión (%)",
        min_value=0,
        max_value=80,
        value=30,
        step=10,
        key="esc4_rec_inv",
    ) / 100.0
    delta_rec_infra = st.slider(
        "Reducción en crecimiento de infra (%)",
        min_value=0,
        max_value=50,
        value=20,
        step=10,
        key="esc4_rec_infra",
    ) / 100.0

    scen_df4 = latest.copy()
    mask_rec = scen_df4[GROUP_COL] == op_rec

    if "an_gen_inversiones" in scen_df4.columns:
        scen_df4.loc[mask_rec, "an_gen_inversiones"] *= (1.0 - delta_rec_inv)
    if "inf_estaciones_base" in scen_df4.columns:
        scen_df4.loc[mask_rec, "inf_estaciones_base"] *= (1.0 - delta_rec_infra)
    if "inf_bam_lineas" in scen_df4.columns:
        scen_df4.loc[mask_rec, "inf_bam_lineas"] *= (1.0 - delta_rec_infra)

    scen_pred4 = rf_global.predict(scen_df4[exog_cols_global].values)
    scen_total4 = scen_pred4.sum()
    scen_shares4 = scen_pred4 / scen_total4
    scen_hhi4 = compute_hhi(scen_shares4)

    scen_table4 = scen_df4[[GROUP_COL, YEAR_COL, DATE_COL]].copy()
    scen_table4["ingresos_base"] = baseline_pred
    scen_table4["cuota_base"] = baseline_shares
    scen_table4["ingresos_escenario"] = scen_pred4
    scen_table4["cuota_escenario"] = scen_shares4
    scen_table4 = scen_table4.sort_values("ingresos_escenario", ascending=False)

    st.dataframe(
        scen_table4,
        use_container_width=True,
    )

    st.metric("HHI base", f"{baseline_hhi:,.0f}")
    st.metric("HHI escenario", f"{scen_hhi4:,.0f}")

    st.markdown(
        """
Útil para discutir el trade-off entre **ahorro en CAPEX** y **pérdida potencial
de cuota** e ingresos, especialmente en mercados maduros.
"""
    )

# ------------------ Escenario 5: Fusión y HHI ------------------ #
with tabs[4]:
    st.subheader("Escenario 5 – Fusión o joint-venture (cambio en HHI)")

    st.markdown(
        """
Simula una **fusión entre dos operadores** (o joint-venture) sumando sus ingresos
en un operador combinado y recalculando el HHI.
"""
    )

    op_f1 = st.selectbox(
        "Operador A",
        operadores,
        key="esc5_op1",
    )
    op_f2 = st.selectbox(
        "Operador B",
        [op for op in operadores if op != op_f1],
        key="esc5_op2",
    )
    delta_sinergias = st.slider(
        "Sinergias en ingresos combinados (%)",
        min_value=-20,
        max_value=40,
        value=10,
        step=5,
        key="esc5_sin",
    ) / 100.0

    scen_df5 = latest.copy()
    scen_pred5 = baseline_pred.copy()

    # fusionamos
    idx1 = scen_df5[GROUP_COL] == op_f1
    idx2 = scen_df5[GROUP_COL] == op_f2

    if idx1.any() and idx2.any():
        y1 = scen_pred5[idx1][0]
        y2 = scen_pred5[idx2][0]
        y_fusion = (y1 + y2) * (1.0 + delta_sinergias)

        # creamos nuevo operador "Fusionado A+B"
        nuevo_nombre = f"{op_f1}+{op_f2}"
        nueva_fila = scen_df5[idx1].copy()
        nueva_fila[GROUP_COL] = nuevo_nombre

        # reemplazamos predicciones
        nuevas_pred = []
        nuevos_ops = []
        for op, y in zip(scen_df5[GROUP_COL], scen_pred5):
            if op in (op_f1, op_f2):
                continue
            nuevos_ops.append(op)
            nuevas_pred.append(y)
        nuevos_ops.append(nuevo_nombre)
        nuevas_pred.append(y_fusion)

        scen_pred5_arr = np.array(nuevas_pred)
        scen_total5 = scen_pred5_arr.sum()
        scen_shares5 = scen_pred5_arr / scen_total5
        scen_hhi5 = compute_hhi(scen_shares5)

        resumen = pd.DataFrame({
            "operador": nuevos_ops,
            "ingresos_base_aprox": nuevos_ops,  # placeholder para no dejar vacío
        })
        # como base estamos usando baseline_pred original; para el texto del TFM basta con HHI
        st.metric("HHI base", f"{baseline_hhi:,.0f}")
        st.metric("HHI post-fusión", f"{scen_hhi5:,.0f}")
        st.write(f"Operador fusionado: **{nuevo_nombre}** (sinergias {delta_sinergias*100:.0f}%).")
    else:
        st.warning("No se han podido localizar ambos operadores en la última foto de mercado.")

    st.markdown(
        """
Escenario útil para analizar cómo una fusión altera la **concentración del mercado (HHI)**,
algo muy relevante en el contexto de la CNMC y la regulación de competencia.
"""
    )

# ------------------ Escenario 6: Shock macro del mercado ------------------ #
with tabs[5]:
    st.subheader("Escenario 6 – Shock macro / regulación (tamaño de mercado)")

    st.markdown(
        """
Simula un **shock macroeconómico o regulatorio** que afecta al tamaño total del
mercado (`tri_ingresos_total_trimestre`), manteniendo la estructura relativa
de los operadores.
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

    scen_df6 = latest.copy()

    if "tri_ingresos_total_trimestre" in scen_df6.columns:
        scen_df6["tri_ingresos_total_trimestre"] *= (1.0 + delta_macro)

    scen_pred6 = rf_global.predict(scen_df6[exog_cols_global].values)
    scen_total6 = scen_pred6.sum()
    scen_shares6 = scen_pred6 / scen_total6
    scen_hhi6 = compute_hhi(scen_shares6)

    scen_table6 = scen_df6[[GROUP_COL, YEAR_COL, DATE_COL]].copy()
    scen_table6["ingresos_base"] = baseline_pred
    scen_table6["ingresos_escenario"] = scen_pred6
    scen_table6["cuota_base"] = baseline_shares
    scen_table6["cuota_escenario"] = scen_shares6
    scen_table6 = scen_table6.sort_values("ingresos_escenario", ascending=False)

    st.dataframe(
        scen_table6,
        use_container_width=True,
    )

    st.metric("HHI base", f"{baseline_hhi:,.0f}")
    st.metric("HHI escenario", f"{scen_hhi6:,.0f}")

    st.markdown(
        """
Este escenario permite cuantificar cómo un shock global (por ejemplo, una crisis
económica o una regulación de precios) puede reducir el tamaño del mercado y cuál
es el impacto relativo para cada operador.
"""
    )

st.markdown(
    """
Con todo esto, el módulo de IA del DSS integra:

1. **Modelo ML explicable** por operador (RandomForest, scikit-learn).  
2. **Forecast temporal** y escenario simple sobre el último trimestre.  
3. **Forecast clásico ARIMA** con bandas de confianza.  
4. **Detección de anomalías** en la serie histórica.  
5. Un **simulador de escenarios de negocio** con múltiples casos de uso:
   inversión, portabilidades, expansión low-cost, austeridad, fusiones y shocks macro.

Esto encaja directamente con los objetivos de un TFM orientado a
**soporte a la decisión** en el sector telecom a partir de datos CNMC.
"""
)
