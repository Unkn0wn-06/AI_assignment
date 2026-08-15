"""Dataset and SVM project overview."""

from utils import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, STATUS_COLUMN,
    banner, load_dataset, press_enter, print_table, section,
)


FEATURE_DESCRIPTIONS = {
    "Age": "Age in years",
    "Blood Pressure": "Blood pressure measurement",
    "Cholesterol Level": "Cholesterol measurement",
    "BMI": "Body mass index",
    "Sleep Hours": "Average sleep duration",
    "Triglyceride Level": "Triglyceride measurement",
    "Fasting Blood Sugar": "Fasting blood sugar measurement",
    "CRP Level": "C-reactive protein level",
    "Homocysteine Level": "Homocysteine measurement",
    "Gender": "Female or Male",
    "Exercise Habits": "Low, Medium, or High",
    "Smoking": "No or Yes",
    "Family Heart Disease": "No or Yes",
    "Diabetes": "No or Yes",
    "High Blood Pressure": "No or Yes",
    "Low HDL Cholesterol": "No or Yes",
    "High LDL Cholesterol": "No or Yes",
    "Alcohol Consumption": "Low, Medium, or High",
    "Stress Level": "Low, Medium, or High",
    "Sugar Consumption": "Low, Medium, or High",
}


def run():
    banner("OVERVIEW DASHBOARD")
    section("Project")
    print("""
  This academic project uses one Support Vector Machine classifier to predict
  Heart Disease Status from 20 raw attributes. The complete saved pipeline
  performs preprocessing and classification consistently for both interfaces.

  This is an academic AI prototype and is not a medical diagnosis.
    """)

    section("Objectives")
    for goal in [
        "1. Preprocess missing numerical and categorical input values.",
        "2. Tune SVM C, gamma, and kernel settings using F1 scoring.",
        "3. Address the imbalanced classes with balanced class weights.",
        "4. Evaluate the SVM using Accuracy, Precision, Recall and F1-score.",
    ]:
        print(f"  {goal}")

    section("Dataset Summary")
    df = load_dataset()
    if df is None:
        print("  [ERROR] The shared dataset is missing.")
    else:
        counts = df[STATUS_COLUMN].value_counts()
        print_table({
            "Dataset records": len(df),
            "Input attributes": len(df.columns) - 1,
            "Numerical attributes": len(NUMERICAL_FEATURES),
            "Categorical attributes": len(CATEGORICAL_FEATURES),
            "Target": STATUS_COLUMN,
            "No": int(counts.get("No", 0)),
            "Yes": int(counts.get("Yes", 0)),
            "Missing input values": int(df.drop(columns=STATUS_COLUMN).isna().sum().sum()),
        })

    section("Input Attribute Dictionary")
    for name, description in FEATURE_DESCRIPTIONS.items():
        kind = "Numerical" if name in NUMERICAL_FEATURES else "Categorical"
        print(f"  {name:<28} {kind:<12} {description}")
    press_enter()
