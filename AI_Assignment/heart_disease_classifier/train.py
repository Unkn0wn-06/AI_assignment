"""Train and evaluate the assignment's SVM-only classification pipeline."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "heart_disease.csv"
MODEL_PATH = ROOT / "models" / "svm_pipeline.joblib"
METRICS_PATH = ROOT / "static" / "metrics.json"
CONFUSION_MATRIX_PATH = ROOT / "static" / "svm_confusion_matrix.png"
TARGET = "Heart Disease Status"

NUMERICAL_FEATURES = [
    "Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours",
    "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level",
]
CATEGORICAL_FEATURES = [
    "Gender", "Exercise Habits", "Smoking", "Family Heart Disease", "Diabetes",
    "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol",
    "Alcohol Consumption", "Stress Level", "Sugar Consumption",
]
EXPECTED_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Required local dataset not found: {DATA_PATH}")

    print(f"Loading local dataset: {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH)
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    extra_columns = [column for column in df.columns if column not in EXPECTED_COLUMNS]
    if missing_columns or extra_columns:
        raise ValueError(f"Unexpected CSV schema. Missing: {missing_columns}; extra: {extra_columns}")
    if len(df) != 10_000:
        raise ValueError(f"Expected 10,000 records, found {len(df):,}")

    X = df.drop(columns=TARGET)
    y = df[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        invalid = sorted(df.loc[y.isna(), TARGET].astype(str).unique())
        raise ValueError(f"Target contains values other than Yes/No: {invalid}")
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("svm", SVC(class_weight="balanced")),
    ])

    parameter_grid = [
        {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10], "svm__gamma": ["scale"]},
        {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10], "svm__gamma": ["scale", "auto", 0.01, 0.1]},
    ]
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    print("Tuning SVC with GridSearchCV (scoring='f1', class_weight='balanced')...")
    search.fit(X_train, y_train)
    fitted_pipeline = search.best_estimator_
    predictions = fitted_pipeline.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, zero_division=0)),
        "F1-Score": float(f1_score(y_test, predictions, zero_division=0)),
        "ConfusionMatrix": matrix.tolist(),
        "BestParameters": search.best_params_,
        "BestCrossValidationF1": float(search.best_score_),
        "TrainSize": int(len(X_train)),
        "TestSize": int(len(X_test)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
    )
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted Heart Disease Status")
    plt.ylabel("Actual Heart Disease Status")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()

    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validation F1: {search.best_score_:.4f}")
    for name in ("Accuracy", "Precision", "Recall", "F1-Score"):
        print(f"{name}: {metrics[name]:.4f}")
    print(f"Confusion matrix: {matrix.tolist()}")
    print(f"Saved fitted pipeline to {MODEL_PATH}")
    print(f"Saved evaluation metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
