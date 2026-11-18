from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Evaluación de Riesgo Crediticio",
    page_icon="📊",
    layout="centered",
)

MODEL_PATH = Path("modelo_crediticio.joblib")
FEATURES_PATH = Path("model_features.json")

EXPECTED_FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# Valores de referencia usados cuando el usuario no abre las opciones avanzadas.
# El modelo SIEMPRE recibe las 23 variables originales.
REFERENCE_VALUES = {
    "LIMIT_BAL": 140000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 2,
    "AGE": 34,

    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,

    "BILL_AMT1": 22382,
    "BILL_AMT2": 21200,
    "BILL_AMT3": 20088,
    "BILL_AMT4": 19052,
    "BILL_AMT5": 18104,
    "BILL_AMT6": 17071,

    "PAY_AMT1": 2100,
    "PAY_AMT2": 2009,
    "PAY_AMT3": 1800,
    "PAY_AMT4": 1500,
    "PAY_AMT5": 1500,
    "PAY_AMT6": 1500,
}

PAY_STATUS_OPTIONS = {
    "Pago al día (-1)": -1,
    "Sin atraso registrado (0)": 0,
    "Atraso de 1 mes": 1,
    "Atraso de 2 meses": 2,
    "Atraso de 3 meses": 3,
    "Atraso de 4 meses": 4,
    "Atraso de 5 meses": 5,
    "Atraso de 6 meses": 6,
    "Atraso de 7 meses": 7,
    "Atraso de 8 meses": 8,
    "Atraso de 9 meses o más": 9,
    "Código especial del dataset (-2)": -2,
}

FRIENDLY_NAMES = {
    "PAY_0": "Estado de pago más reciente",
    "PAY_AMT2": "Pago realizado recientemente",
    "PAY_4": "Estado de pago hace 3 períodos",
    "LIMIT_BAL": "Límite de crédito",
    "PAY_3": "Estado de pago hace 2 períodos",
    "PAY_2": "Estado de pago hace 1 período",
}

# ============================================================
# FUNCIONES
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("No se encontró el archivo modelo_crediticio.joblib.")
        st.stop()

    model = joblib.load(MODEL_PATH)

    features = EXPECTED_FEATURES
    if FEATURES_PATH.exists():
        features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))

    return model, features


def status_select(label, key, default_value=0, help_text=None):
    reverse = {v: k for k, v in PAY_STATUS_OPTIONS.items()}
    labels = list(PAY_STATUS_OPTIONS.keys())

    default_label = reverse.get(default_value, "Sin atraso registrado (0)")
    default_index = labels.index(default_label)

    selected = st.selectbox(
        label,
        labels,
        index=default_index,
        key=key,
        help=help_text,
    )
    return PAY_STATUS_OPTIONS[selected]


def render_result(model, feature_order, values):
    row = pd.DataFrame(
        [[values[f] for f in feature_order]],
        columns=feature_order,
    )

    prediction = int(model.predict(row)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])

    st.divider()
    st.subheader("Resultado de la evaluación")

    risk_label = "ALTO" if prediction == 1 else "BAJO"

    if prediction == 1:
        st.error(f"Riesgo estimado: **{risk_label}**")
    else:
        st.success(f"Riesgo estimado: **{risk_label}**")

    if probability is not None:
        pct = probability * 100

        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Probabilidad estimada de incumplimiento",
                f"{pct:.1f}%"
            )
        with c2:
            st.metric(
                "Umbral de clasificación",
                "50%"
            )

        st.progress(min(max(probability, 0.0), 1.0))

        if 40 <= pct < 50:
            st.warning(
                "El resultado se clasifica como riesgo bajo, pero se encuentra "
                "relativamente cerca del umbral del 50%."
            )

    st.caption(
        "La clasificación corresponde a una estimación del modelo y no constituye "
        "una decisión automática de aprobación o rechazo de crédito."
    )

    if hasattr(model, "feature_importances_"):
        importance = pd.Series(
            model.feature_importances_,
            index=feature_order
        ).sort_values(ascending=False)

        top = importance.head(6).rename(index=FRIENDLY_NAMES)

        st.subheader("¿Qué variables utiliza más el modelo?")
        st.write(
            "El gráfico muestra la importancia **global** de las principales variables "
            "en el Árbol de Decisión. No representa causalidad ni explica por sí solo "
            "este caso individual."
        )
        st.bar_chart(top)

    with st.expander("Ver detalle técnico de los datos enviados al modelo"):
        st.dataframe(
            row.T.rename(columns={0: "Valor"}),
            use_container_width=True,
        )


# ============================================================
# INTERFAZ
# ============================================================

model, feature_order = load_model()

st.title("Evaluación de Riesgo Crediticio")
st.caption("Modelo de aprendizaje supervisado · Árbol de Decisión controlado")

st.info(
    "Demostración académica basada en el dataset **Default of Credit Card Clients** "
    "de UCI. La interfaz simplifica la entrada de datos, pero el modelo mantiene "
    "sus **23 variables originales**."
)

st.subheader("Datos principales")
st.write(
    "Para facilitar el uso, la evaluación se concentra en los antecedentes que "
    "tuvieron mayor importancia predictiva en el modelo."
)

advanced_mode = st.toggle(
    "Mostrar opciones avanzadas",
    value=False,
    help=(
        "Permite modificar las variables secundarias. "
        "Si se mantiene desactivado, el sistema utiliza valores de referencia."
    ),
)

values = REFERENCE_VALUES.copy()

with st.form("credit_risk_form"):

    # --------------------------------------------------------
    # 1. SITUACIÓN ACTUAL
    # --------------------------------------------------------
    st.markdown("### 1. Situación de pago actual")

    values["PAY_0"] = status_select(
        "¿Cuál es la situación de pago más reciente?",
        key="pay_0",
        default_value=REFERENCE_VALUES["PAY_0"],
        help_text=(
            "Es la variable con mayor importancia global en el árbol entrenado."
        ),
    )

    # --------------------------------------------------------
    # 2. HISTORIAL RECIENTE
    # --------------------------------------------------------
    st.markdown("### 2. Historial reciente")

    st.caption(
        "Indica si el cliente presentó atrasos en los tres períodos anteriores."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        values["PAY_2"] = status_select(
            "1 período atrás",
            key="pay_2",
            default_value=REFERENCE_VALUES["PAY_2"],
        )

    with c2:
        values["PAY_3"] = status_select(
            "2 períodos atrás",
            key="pay_3",
            default_value=REFERENCE_VALUES["PAY_3"],
        )

    with c3:
        values["PAY_4"] = status_select(
            "3 períodos atrás",
            key="pay_4",
            default_value=REFERENCE_VALUES["PAY_4"],
        )

    # --------------------------------------------------------
    # 3. INFORMACIÓN FINANCIERA
    # --------------------------------------------------------
    st.markdown("### 3. Información financiera")

    c1, c2 = st.columns(2)

    with c1:
        values["LIMIT_BAL"] = st.number_input(
            "Límite de crédito (NT$)",
            min_value=0,
            value=int(REFERENCE_VALUES["LIMIT_BAL"]),
            step=10000,
            help="Monto máximo de crédito disponible para el cliente.",
        )

    with c2:
        values["PAY_AMT2"] = st.number_input(
            "Pago realizado recientemente (NT$)",
            min_value=0,
            value=int(REFERENCE_VALUES["PAY_AMT2"]),
            step=500,
            help=(
                "Corresponde a PAY_AMT2, una de las variables de mayor importancia "
                "en el árbol entrenado."
            ),
        )

    # --------------------------------------------------------
    # 4. OPCIONES AVANZADAS
    # --------------------------------------------------------
    if advanced_mode:
        st.markdown("### 4. Opciones avanzadas")
        st.caption(
            "Estas variables forman parte del modelo original, pero tuvieron menor "
            "importancia global que los antecedentes mostrados arriba."
        )

        with st.expander("Historial de pago adicional", expanded=False):
            values["PAY_5"] = status_select(
                "4 períodos atrás",
                key="pay_5",
                default_value=REFERENCE_VALUES["PAY_5"],
            )

            values["PAY_6"] = status_select(
                "5 períodos atrás",
                key="pay_6",
                default_value=REFERENCE_VALUES["PAY_6"],
            )

        with st.expander("Montos facturados", expanded=False):
            for idx in range(1, 7):
                feature = f"BILL_AMT{idx}"
                values[feature] = st.number_input(
                    f"Facturación período {idx} (NT$)",
                    value=int(REFERENCE_VALUES[feature]),
                    step=1000,
                    key=f"bill_{idx}",
                )

        with st.expander("Otros pagos realizados", expanded=False):
            payment_features = [
                "PAY_AMT1",
                "PAY_AMT3",
                "PAY_AMT4",
                "PAY_AMT5",
                "PAY_AMT6",
            ]

            for feature in payment_features:
                values[feature] = st.number_input(
                    f"{feature} (NT$)",
                    min_value=0,
                    value=int(REFERENCE_VALUES[feature]),
                    step=500,
                    key=f"advanced_{feature}",
                )

        with st.expander("Antecedentes demográficos", expanded=False):
            c1, c2 = st.columns(2)

            with c1:
                values["AGE"] = st.number_input(
                    "Edad",
                    min_value=18,
                    max_value=100,
                    value=int(REFERENCE_VALUES["AGE"]),
                    step=1,
                )

                sex_map = {
                    "Masculino": 1,
                    "Femenino": 2,
                }
                sex_label = st.selectbox(
                    "Sexo",
                    list(sex_map.keys()),
                    index=1,
                )
                values["SEX"] = sex_map[sex_label]

            with c2:
                education_map = {
                    "Posgrado": 1,
                    "Universidad": 2,
                    "Enseñanza media": 3,
                    "Otros": 4,
                }
                education_label = st.selectbox(
                    "Nivel educacional",
                    list(education_map.keys()),
                    index=1,
                )
                values["EDUCATION"] = education_map[education_label]

                marriage_map = {
                    "Casado/a": 1,
                    "Soltero/a": 2,
                    "Otros": 3,
                }
                marriage_label = st.selectbox(
                    "Estado civil",
                    list(marriage_map.keys()),
                    index=1,
                )
                values["MARRIAGE"] = marriage_map[marriage_label]

    submitted = st.form_submit_button(
        "Evaluar riesgo",
        type="primary",
        use_container_width=True,
    )


if not advanced_mode:
    st.caption(
        "Las variables secundarias se completan automáticamente con valores de "
        "referencia para conservar la estructura original del modelo."
    )

if submitted:
    render_result(
        model=model,
        feature_order=feature_order,
        values=values,
    )

st.divider()

st.caption(
    "Uso exclusivamente académico. Esta aplicación no debe utilizarse para "
    "aprobar, rechazar o modificar condiciones de créditos reales."
)
