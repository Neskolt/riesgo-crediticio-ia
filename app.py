from pathlib import Path
import json
import joblib
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
    "Sin atraso / código 0": 0,
    "Atraso de 1 mes": 1,
    "Atraso de 2 meses": 2,
    "Atraso de 3 meses": 3,
    "Atraso de 4 meses": 4,
    "Atraso de 5 meses": 5,
    "Atraso de 6 meses": 6,
    "Atraso de 7 meses": 7,
    "Atraso de 8 meses": 8,
    "Atraso de 9 meses o más": 9,
    "Código -2 (presente en el dataset)": -2,
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
    model = joblib.load(MODEL_PATH)
    features = EXPECTED_FEATURES
    if FEATURES_PATH.exists():
        features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    return model, features

def status_select(label, key, default_value=0):
    labels = list(PAY_STATUS_OPTIONS.keys())
    reverse = {v: k for k, v in PAY_STATUS_OPTIONS.items()}
    default_label = reverse.get(default_value, "Sin atraso / código 0")
    selected = st.selectbox(
        label,
        labels,
        index=labels.index(default_label),
        key=key,
    )
    return PAY_STATUS_OPTIONS[selected]

def evaluate_and_show(model, feature_order, input_values):
    row = pd.DataFrame(
        [[input_values[f] for f in feature_order]],
        columns=feature_order,
    )

    prediction = int(model.predict(row)[0])
    probs = model.predict_proba(row)[0]
    classes = list(model.classes_)
    probability = float(probs[classes.index(1)]) if 1 in classes else None

    st.divider()
    st.subheader("Resultado")

    if probability is not None:
        pct = probability * 100
        risk_label = "ALTO" if prediction == 1 else "BAJO"

        if prediction == 1:
            st.error(f"Riesgo estimado: **{risk_label}**")
        else:
            st.success(f"Riesgo estimado: **{risk_label}**")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Probabilidad de incumplimiento", f"{pct:.1f}%")
        with c2:
            st.metric("Umbral de clasificación", "50%")

        st.progress(min(max(probability, 0.0), 1.0))

        if 40 <= pct < 50:
            st.warning(
                "El caso queda clasificado como riesgo bajo, pero está relativamente "
                "cerca del umbral del 50%."
            )

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(
            model.feature_importances_,
            index=feature_order,
            name="Importancia",
        ).sort_values(ascending=False)

        top = importances.head(6)
        st.subheader("Variables más relevantes del modelo")
        st.caption(
            "Importancia global del árbol entrenado. No implica causalidad ni explica "
            "por sí sola este caso individual."
        )
        st.bar_chart(top)

    with st.expander("Ver los 23 valores enviados al modelo"):
        st.dataframe(row.T.rename(columns={0: "Valor"}), use_container_width=True)

model, feature_order = load_model()

st.title("Evaluación de Riesgo Crediticio")
st.caption("Modelo de aprendizaje supervisado · Árbol de Decisión controlado")

st.info(
    "Demostración académica basada en el dataset **Default of Credit Card Clients** "
    "de UCI. Los montos se expresan en **dólares de Taiwán (NT$)**."
)

modo = st.radio(
    "Modo de evaluación",
    ["Rápido", "Completo"],
    horizontal=True,
    help="Rápido solicita solo las variables de mayor importancia. Completo permite modificar las 23 entradas.",
)

if modo == "Rápido":
    st.subheader("Evaluación rápida")
    st.write(
        "Se solicitan únicamente **6 variables**, que en conjunto concentran "
        "aproximadamente **93,1% de la importancia global del árbol**. "
        "Las restantes se completan con valores de referencia del dataset."
    )

    input_values = REFERENCE_VALUES.copy()

    with st.form("quick_form"):
        limit_bal = st.number_input(
            "Límite de crédito (NT$)",
            min_value=0,
            value=int(REFERENCE_VALUES["LIMIT_BAL"]),
            step=10000,
        )

        st.markdown("#### Historial reciente de pagos")
        c1, c2 = st.columns(2)

        with c1:
            pay0 = status_select(
                "Septiembre 2005",
                "q_pay0",
                REFERENCE_VALUES["PAY_0"],
            )
            pay3 = status_select(
                "Julio 2005",
                "q_pay3",
                REFERENCE_VALUES["PAY_3"],
            )

        with c2:
            pay2 = status_select(
                "Agosto 2005",
                "q_pay2",
                REFERENCE_VALUES["PAY_2"],
            )
            pay4 = status_select(
                "Junio 2005",
                "q_pay4",
                REFERENCE_VALUES["PAY_4"],
            )

        pay_amt2 = st.number_input(
            "Pago realizado en agosto 2005 (NT$)",
            min_value=0,
            value=int(REFERENCE_VALUES["PAY_AMT2"]),
            step=500,
            help="Corresponde a PAY_AMT2, segunda variable más importante del árbol.",
        )

        submitted = st.form_submit_button(
            "Evaluar riesgo",
            type="primary",
            use_container_width=True,
        )

    input_values.update({
        "LIMIT_BAL": limit_bal,
        "PAY_0": pay0,
        "PAY_2": pay2,
        "PAY_3": pay3,
        "PAY_4": pay4,
        "PAY_AMT2": pay_amt2,
    })

    with st.expander("¿Qué valores completa automáticamente el modo rápido?"):
        st.write(
            "El modelo sigue recibiendo las **23 variables originales**. Las variables "
            "no solicitadas se completan con valores centrales de referencia del dataset "
            "(medianas o categorías más frecuentes). Esto simplifica la demostración "
            "sin reentrenar ni modificar el modelo."
        )
        ref_df = pd.DataFrame(
            {
                "Variable": list(REFERENCE_VALUES.keys()),
                "Valor de referencia": list(REFERENCE_VALUES.values()),
            }
        )
        st.dataframe(ref_df, hide_index=True, use_container_width=True)

    if submitted:
        evaluate_and_show(model, feature_order, input_values)

else:
    st.subheader("Evaluación completa")
    st.write(
        "Permite modificar las **23 variables** utilizadas durante el entrenamiento. "
        "Este modo es útil para revisión técnica o pruebas específicas."
    )

    input_values = REFERENCE_VALUES.copy()

    with st.form("full_form"):
        c1, c2 = st.columns(2)
        with c1:
            input_values["LIMIT_BAL"] = st.number_input(
                "Límite de crédito (NT$)",
                min_value=0,
                value=int(REFERENCE_VALUES["LIMIT_BAL"]),
                step=10000,
            )
        with c2:
            input_values["AGE"] = st.number_input(
                "Edad",
                min_value=18,
                max_value=100,
                value=int(REFERENCE_VALUES["AGE"]),
                step=1,
            )

        st.markdown("#### Historial de pagos")
        for month, pay_feature, _, _ in MONTHS:
            input_values[pay_feature] = status_select(
                month,
                f"f_{pay_feature}",
                REFERENCE_VALUES[pay_feature],
            )

        with st.expander("Facturación mensual", expanded=False):
            for month, _, bill_feature, _ in MONTHS:
                input_values[bill_feature] = st.number_input(
                    f"{month} (NT$)",
                    value=int(REFERENCE_VALUES[bill_feature]),
                    step=1000,
                    key=f"f_bill_{bill_feature}",
                )

        with st.expander("Pagos realizados", expanded=False):
            for month, _, _, pay_feature in MONTHS:
                input_values[pay_feature] = st.number_input(
                    f"{month} (NT$)",
                    min_value=0,
                    value=int(REFERENCE_VALUES[pay_feature]),
                    step=500,
                    key=f"f_amt_{pay_feature}",
                )

        with st.expander("Antecedentes demográficos", expanded=False):
            sex_map = {"Masculino": 1, "Femenino": 2}
            sex_label = st.selectbox("Sexo", list(sex_map.keys()), index=1)
            input_values["SEX"] = sex_map[sex_label]

            education_map = {
                "Posgrado": 1,
                "Universidad": 2,
                "Enseñanza media": 3,
                "Otros": 4,
            }
            edu = st.selectbox("Nivel educacional", list(education_map.keys()), index=1)
            input_values["EDUCATION"] = education_map[edu]

            marriage_map = {"Casado/a": 1, "Soltero/a": 2, "Otros": 3}
            mar = st.selectbox("Estado civil", list(marriage_map.keys()), index=1)
            input_values["MARRIAGE"] = marriage_map[mar]

        submitted_full = st.form_submit_button(
            "Evaluar riesgo",
            type="primary",
            use_container_width=True,
        )

    if submitted_full:
        evaluate_and_show(model, feature_order, input_values)

st.divider()
st.caption(
    "Uso académico. La aplicación no aprueba ni rechaza créditos y no debe utilizarse "
    "para decisiones financieras reales."
)
