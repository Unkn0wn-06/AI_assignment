"""Interactive raw-input prediction with the shared preprocessing + SVM pipeline."""

import pandas as pd

from utils import (
    FEATURES, PIPELINE_PATH, WIDTH, banner, load_pipeline, press_enter,
    print_table, prompt_choice, prompt_float, prompt_yn, section,
)


def _choice(label: str, options: list[str]) -> str:
    print(f"\n  {label}")
    return options[prompt_choice("Select", options)]


def _collect_patient_data() -> pd.DataFrame:
    section("Enter All 20 Input Attributes")
    print("  Enter numerical measurements and select each categorical value.\n")
    data = {
        "Age": prompt_float("1. Age", 18, 80),
        "Blood Pressure": prompt_float("2. Blood Pressure", 120, 180),
        "Cholesterol Level": prompt_float("3. Cholesterol Level", 150, 300),
        "BMI": prompt_float("4. BMI", 18, 40),
        "Sleep Hours": prompt_float("5. Sleep Hours", 4, 10),
        "Triglyceride Level": prompt_float("6. Triglyceride Level", 100, 400),
        "Fasting Blood Sugar": prompt_float("7. Fasting Blood Sugar", 80, 160),
        "CRP Level": prompt_float("8. CRP Level", 0, 15),
        "Homocysteine Level": prompt_float("9. Homocysteine Level", 5, 20),
        "Gender": _choice("10. Gender", ["Female", "Male"]),
        "Exercise Habits": _choice("11. Exercise Habits", ["Low", "Medium", "High"]),
        "Smoking": _choice("12. Smoking", ["No", "Yes"]),
        "Family Heart Disease": _choice("13. Family Heart Disease", ["No", "Yes"]),
        "Diabetes": _choice("14. Diabetes", ["No", "Yes"]),
        "High Blood Pressure": _choice("15. High Blood Pressure", ["No", "Yes"]),
        "Low HDL Cholesterol": _choice("16. Low HDL Cholesterol", ["No", "Yes"]),
        "High LDL Cholesterol": _choice("17. High LDL Cholesterol", ["No", "Yes"]),
        "Alcohol Consumption": _choice("18. Alcohol Consumption", ["Low", "Medium", "High"]),
        "Stress Level": _choice("19. Stress Level", ["Low", "Medium", "High"]),
        "Sugar Consumption": _choice("20. Sugar Consumption", ["Low", "Medium", "High"]),
    }
    return pd.DataFrame([data], columns=FEATURES)


def _print_result(prediction: int, patient_df: pd.DataFrame):
    label = "Yes" if prediction == 1 else "No"
    print(f"\n{'=' * WIDTH}")
    print("  HEART DISEASE CLASSIFICATION PREDICTION")
    print(f"  Result: {label}")
    print("  This is an academic AI prototype and is not a medical diagnosis.")
    print("=" * WIDTH)
    section("Entered Attribute Summary")
    print_table(patient_df.iloc[0].to_dict())


def predict_patient(patient_df: pd.DataFrame) -> int:
    """Pass the raw, exact-schema DataFrame directly to the shared pipeline."""
    return int(load_pipeline().predict(patient_df)[0])


def run():
    banner("HEART DISEASE CLASSIFICATION PREDICTION")
    if not PIPELINE_PATH.is_file():
        print("\n  [ERROR] The fitted SVM pipeline is missing. Run Model Training first.")
        press_enter()
        return
    while True:
        patient_df = _collect_patient_data()
        _print_result(predict_patient(patient_df), patient_df)
        if not prompt_yn("Run another classification prediction?"):
            break
