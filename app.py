from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

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

PAY_STATUS_OPTIONS = {
    "Código -2 (presente en el dataset)": -2,
    "Pago al día (-1)": -1,
    "Código 0 (presente en el dataset)": 0,
    "Atraso de 1 mes": 1,
    "Atraso de 2 meses": 2,
    "Atraso de 3 meses": 3,
    "Atraso de 4 meses": 4,
    "Atraso de 5 meses": 5,
    "Atraso de 6 meses": 6,
    "Atraso de 7 meses": 7,
    "Atraso de 8 meses": 8,
    "Atraso de 9 meses o más": 9,
}

MONTHS = [
    ("Septiembre 2005", "PAY_0", "BILL_AMT1", "PAY_AMT1"),
    ("Agosto 2005", "PAY_2", "BILL_AMT2", "PAY_AMT2"),
    ("Julio 2005", "PAY_3", "BILL_AMT3", "PAY_AMT3"),
    ("Junio 2005", "PAY_4", "BILL_AMT4", "PAY_AMT4"),
    ("Mayo 2005", "PAY_5", "BILL_AMT5", "PAY_AMT5"),
    ("Abril 2005", "PAY_6", "BILL_AMT6", "PAY_AMT6"),
]

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, EXPECTED_FEATURES
    model = joblib.load(MODEL_PATH)

    features = EXPECTED_FEATURES
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            features = json.load(f)

    return model, features

def status_select(label, key, default_label="Pago al día (-1)"):
    labels = list(PAY_STATUS_OPTIONS.keys())
    default_index = labels.index(default_label) if default_label in labels else 1
    selected = st.selectbox(label, labels, index=default_index, key=key)
    return PAY_STATUS_OPTIONS[selected]

model, feature_order = load_model()

st.title("Evaluación de Riesgo Crediticio")
st.caption("Modelo de aprendizaje supervisado · Árbol de Decisión controlado")

st.info(
    "Demostración académica basada en el dataset **Default of Credit Card Clients** "
    "de UCI. Los montos se expresan en **dólares de Taiwán (NT$)**, tal como fueron "
    "utilizados para entrenar el modelo."
)

if model is None:
    st.error(
        "No se encontró `modelo_crediticio.joblib`. Ejecuta primero la celda de "
        "exportación en Google Colab y copia el archivo a esta carpeta."
    )
    st.stop()

st.subheader("1. Historial de pagos")
st.write(
    "Este bloque se presenta primero porque el comportamiento de pago reciente "
    "fue el grupo de variables con mayor relevancia en el modelo entrenado."
)

payment_status = {}
with st.expander("Estado de pago de los últimos seis períodos", expanded=True):
    for month, pay_feature, _, _ in MONTHS:
        payment_status[pay_feature] = status_select(
            f"{month} — estado de pago",
            key=f"status_{pay_feature}",
        )

st.subheader("2. Información financiera")

col1, col2 = st.columns(2)
with col1:
    limit_bal = st.number_input(
        "Límite de crédito (NT$)",
        min_value=0,
        value=50000,
        step=1000,
    )
with col2:
    age = st.number_input(
        "Edad",
        min_value=18,
        max_value=100,
        value=35,
        step=1,
    )

bill_amounts = {}
pay_amounts = {}

with st.expander("Facturación mensual (NT$)", expanded=False):
    for month, _, bill_feature, _ in MONTHS:
        bill_amounts[bill_feature] = st.number_input(
            f"{month}",
            value=0,
            step=1000,
            key=f"bill_{bill_feature}",
        )

with st.expander("Pagos realizados (NT$)", expanded=True):
    for month, _, _, pay_amt_feature in MONTHS:
        pay_amounts[pay_amt_feature] = st.number_input(
            f"{month}",
            min_value=0,
            value=0,
            step=1000,
            key=f"paid_{pay_amt_feature}",
        )

st.subheader("3. Otros antecedentes")
st.caption(
    "El modelo fue entrenado con estas variables y por ello se conservan como entrada, "
    "aunque su importancia global fue menor que la del historial de pagos."
)

with st.expander("Antecedentes demográficos", expanded=False):
    sex_label = st.selectbox("Sexo", ["Masculino", "Femenino"])
    sex = 1 if sex_label == "Masculino" else 2

    education_map = {
        "Posgrado": 1,
        "Universidad": 2,
        "Enseñanza media": 3,
        "Otros": 4,
    }
    education_label = st.selectbox("Nivel educacional", list(education_map.keys()), index=1)
    education = education_map[education_label]

    marriage_map = {
        "Casado/a": 1,
        "Soltero/a": 2,
        "Otros": 3,
    }
    marriage_label = st.selectbox("Estado civil", list(marriage_map.keys()), index=1)
    marriage = marriage_map[marriage_label]

input_values = {
    "LIMIT_BAL": limit_bal,
    "SEX": sex,
    "EDUCATION": education,
    "MARRIAGE": marriage,
    "AGE": age,
    **payment_status,
    **bill_amounts,
    **pay_amounts,
}

st.divider()

if st.button("Evaluar riesgo", type="primary", use_container_width=True):
    missing = [f for f in feature_order if f not in input_values]
    if missing:
        st.error(f"Faltan variables requeridas por el modelo: {missing}")
        st.stop()

    row = pd.DataFrame(
        [[input_values[f] for f in feature_order]],
        columns=feature_order,
    )

    prediction = int(model.predict(row)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probs = model.predict_proba(row)[0]
        if 1 in classes:
            probability = float(probs[classes.index(1)])

    st.subheader("Resultado")

    if probability is not None:
        pct = probability * 100
        risk_label = "ALTO" if prediction == 1 else "BAJO"

        if prediction == 1:
            st.error(f"Riesgo estimado: **{risk_label}**")
        else:
            st.success(f"Riesgo estimado: **{risk_label}**")

        st.metric("Probabilidad estimada de incumplimiento", f"{pct:.1f}%")
        st.progress(min(max(probability, 0.0), 1.0))

        st.write(
            "La clasificación utiliza el umbral estándar del modelo: "
            "**50%**. Un valor inferior se clasifica como riesgo bajo y uno "
            "igual o superior como riesgo alto."
        )
    else:
        risk_label = "ALTO" if prediction == 1 else "BAJO"
        st.write(f"Riesgo estimado: **{risk_label}**")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(
            model.feature_importances_,
            index=feature_order,
            name="Importancia",
        ).sort_values(ascending=False)

        top = importances.head(10)
        st.subheader("Variables con mayor importancia global")
        st.caption(
            "La importancia representa cuánto utiliza el árbol cada variable para "
            "clasificar. No implica causalidad ni explica por sí sola este caso individual."
        )
        st.bar_chart(top)

    with st.expander("Ver datos enviados al modelo"):
        st.dataframe(row.T.rename(columns={0: "Valor"}), use_container_width=True)

st.divider()
st.caption(
    "Uso académico. Esta aplicación no aprueba ni rechaza créditos y no debe utilizarse "
    "para tomar decisiones financieras reales."
)
