"""Smoke-test raw-data inference through the fitted preprocessing + SVM pipeline."""

from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
pipeline = joblib.load(ROOT / "models" / "svm_pipeline.joblib")

sample_patient = pd.DataFrame([{
    "Age": 55.0,
    "Blood Pressure": 145.0,
    "Cholesterol Level": 245.0,
    "BMI": 28.5,
    "Sleep Hours": 7.0,
    "Triglyceride Level": 220.0,
    "Fasting Blood Sugar": 115.0,
    "CRP Level": 5.0,
    "Homocysteine Level": 11.0,
    "Gender": "Male",
    "Exercise Habits": "Medium",
    "Smoking": "No",
    "Family Heart Disease": "Yes",
    "Diabetes": "No",
    "High Blood Pressure": "Yes",
    "Low HDL Cholesterol": "No",
    "High LDL Cholesterol": "Yes",
    "Alcohol Consumption": "Low",
    "Stress Level": "Medium",
    "Sugar Consumption": "Medium",
}])

prediction = int(pipeline.predict(sample_patient)[0])
print("Inference completed successfully with all 20 raw input attributes.")
print(f"Prediction: {prediction} ({'Yes' if prediction == 1 else 'No'})")
