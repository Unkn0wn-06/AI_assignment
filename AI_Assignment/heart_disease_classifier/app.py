"""Streamlit interface for the SVM-only heart disease classification prototype."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "heart_disease.csv"
MODEL_PATH = ROOT / "models" / "svm_pipeline.joblib"
METRICS_PATH = ROOT / "static" / "metrics.json"
CONFUSION_MATRIX_PATH = ROOT / "static" / "svm_confusion_matrix.png"
STATUS_COLUMN = "Heart Disease Status"

NUMERICAL_FEATURES = [
    "Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours",
    "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level",
]
CATEGORICAL_FEATURES = [
    "Gender", "Exercise Habits", "Smoking", "Family Heart Disease", "Diabetes",
    "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol",
    "Alcohol Consumption", "Stress Level", "Sugar Consumption",
]

st.set_page_config(page_title="Heart Disease Classification Prediction", page_icon="❤️", layout="wide")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


st.title("Heart Disease Classification Prediction")
st.caption("TARUMT Artificial Intelligence assignment — SVM-only classification prototype")
st.warning(
    "Disclaimer: This is an academic AI prototype, not a medical diagnostic system. "
    "Its output must not replace evaluation or advice from a qualified healthcare professional."
)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Exploratory Data Analysis", "SVM Model Performance", "Prediction"],
)

try:
    df = load_dataset()
except Exception as exc:
    st.error(f"Unable to load the local dataset: {exc}")
    st.stop()


if page == "Overview":
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Input attributes", "20")
    col3.metric("Outcome field", STATUS_COLUMN)
    st.write(
        "The local dataset contains 9 numerical and 11 categorical input attributes. "
        "The SVM classifies the outcome as Yes (1) or No (0)."
    )

    counts = df[STATUS_COLUMN].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=ax)
    ax.set_title("Heart Disease Status Distribution")
    ax.set_xlabel(STATUS_COLUMN)
    ax.set_ylabel("Records")
    st.pyplot(fig)

elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    preview, distributions, correlations = st.tabs(
        ["Dataset preview", "Numerical distributions", "Numerical correlations"]
    )
    with preview:
        st.dataframe(df.head(10), use_container_width=True)
        st.subheader("Missing values")
        st.dataframe(df.isna().sum().rename("Missing values").to_frame(), use_container_width=True)

    with distributions:
        feature = st.selectbox("Numerical attribute", NUMERICAL_FEATURES)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x=feature, hue=STATUS_COLUMN, kde=True, ax=ax)
        ax.set_title(f"{feature} by {STATUS_COLUMN}")
        st.pyplot(fig)

    with correlations:
        st.write("Correlation is calculated only from numerical values; categorical text is excluded.")
        numeric_for_correlation = df[NUMERICAL_FEATURES].copy()
        numeric_for_correlation[STATUS_COLUMN] = df[STATUS_COLUMN].map({"No": 0, "Yes": 1})
        fig, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(numeric_for_correlation.corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Numerical Attribute Correlations")
        st.pyplot(fig)

elif page == "SVM Model Performance":
    st.header("SVM Model Performance")
    if not METRICS_PATH.exists():
        st.info("Run `python train.py` to create the evaluation results.")
    else:
        metrics = load_metrics()
        columns = st.columns(4)
        for column, name in zip(columns, ["Accuracy", "Precision", "Recall", "F1-Score"]):
            column.metric(name, f"{metrics[name] * 100:.2f}%")
        st.caption(
            f"Best cross-validation F1: {metrics['BestCrossValidationF1']:.4f} | "
            f"Best parameters: {metrics['BestParameters']}"
        )
        st.write("Confusion matrix (rows: actual No/Yes; columns: predicted No/Yes):")
        st.dataframe(
            pd.DataFrame(metrics["ConfusionMatrix"], index=["Actual No", "Actual Yes"],
                         columns=["Predicted No", "Predicted Yes"]),
            use_container_width=True,
        )
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), caption="SVM confusion matrix")

else:
    st.header("Heart Disease Classification Prediction")
    if not MODEL_PATH.exists():
        st.info("Run `python train.py` before submitting a prediction.")
        st.stop()

    yes_no = ["No", "Yes"]
    level = ["Low", "Medium", "High"]
    with st.form("prediction_form"):
        left, middle, right = st.columns(3)
        with left:
            age = st.number_input("Age", 18.0, 120.0, 50.0)
            blood_pressure = st.number_input("Blood Pressure", 50.0, 250.0, 150.0)
            cholesterol = st.number_input("Cholesterol Level", 50.0, 500.0, 225.0)
            bmi = st.number_input("BMI", 10.0, 70.0, 29.0)
            sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
            triglycerides = st.number_input("Triglyceride Level", 20.0, 800.0, 250.0)
            fasting_sugar = st.number_input("Fasting Blood Sugar", 20.0, 500.0, 120.0)
        with middle:
            crp = st.number_input("CRP Level", 0.0, 100.0, 7.5)
            homocysteine = st.number_input("Homocysteine Level", 0.0, 100.0, 12.4)
            gender = st.selectbox("Gender", ["Female", "Male"])
            exercise = st.selectbox("Exercise Habits", level)
            smoking = st.selectbox("Smoking", yes_no)
            family_history = st.selectbox("Family Heart Disease", yes_no)
            diabetes = st.selectbox("Diabetes", yes_no)
        with right:
            high_bp = st.selectbox("High Blood Pressure", yes_no)
            low_hdl = st.selectbox("Low HDL Cholesterol", yes_no)
            high_ldl = st.selectbox("High LDL Cholesterol", yes_no)
            alcohol = st.selectbox("Alcohol Consumption", level)
            stress = st.selectbox("Stress Level", level)
            sugar = st.selectbox("Sugar Consumption", level)
        submitted = st.form_submit_button("Submit classification prediction")

    if submitted:
        raw_input = pd.DataFrame([{
            "Age": age,
            "Blood Pressure": blood_pressure,
            "Cholesterol Level": cholesterol,
            "BMI": bmi,
            "Sleep Hours": sleep,
            "Triglyceride Level": triglycerides,
            "Fasting Blood Sugar": fasting_sugar,
            "CRP Level": crp,
            "Homocysteine Level": homocysteine,
            "Gender": gender,
            "Exercise Habits": exercise,
            "Smoking": smoking,
            "Family Heart Disease": family_history,
            "Diabetes": diabetes,
            "High Blood Pressure": high_bp,
            "Low HDL Cholesterol": low_hdl,
            "High LDL Cholesterol": high_ldl,
            "Alcohol Consumption": alcohol,
            "Stress Level": stress,
            "Sugar Consumption": sugar,
        }], columns=NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
        prediction = int(load_pipeline().predict(raw_input)[0])
        if prediction == 1:
            st.error("SVM classification result: Yes")
        else:
            st.success("SVM classification result: No")
        st.caption("This classification is for academic demonstration only and is not a diagnosis.")
