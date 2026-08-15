# Heart Disease Classification Prediction

This project is a TARUMT Artificial Intelligence assignment that uses a Support Vector Machine (SVM) to classify a person's heart disease status.

SVC is the **only core machine-learning classification algorithm** used in this project. No KNN, Logistic Regression, Random Forest, Decision Tree, ANN, or other classifier is trained or compared.

> **Disclaimer:** This application is an academic AI prototype, not a medical diagnostic system. Its predictions must not replace professional medical evaluation, diagnosis, or advice.

## Project Features

- Loads only the local `heart_disease.csv` dataset.
- Uses all 20 input attributes from the dataset.
- Converts `Heart Disease Status` from `Yes`/`No` to `1`/`0` for model training.
- Uses an 80/20 stratified train/test split with `random_state=42`.
- Replaces missing numerical values using median imputation.
- Standardizes numerical values with `StandardScaler`.
- Replaces missing categorical values with `Unknown`.
- One-hot encodes categorical attributes with `handle_unknown='ignore'`.
- Combines preprocessing and SVC in one scikit-learn `Pipeline`.
- Uses `class_weight='balanced'` for the imbalanced outcome classes.
- Tunes SVM `C`, `gamma`, and `kernel` values with `GridSearchCV`.
- Selects hyperparameters using F1-score rather than accuracy alone.
- Provides a Streamlit interface for EDA, model performance, and prediction.

## Dataset

The local `heart_disease.csv` contains:

- **10,000 records**
- **20 input attributes**
- **1 outcome column:** `Heart Disease Status`
- `Yes` means heart disease present (`1`)
- `No` means heart disease absent (`0`)

### Numerical input attributes

| Attribute | Description |
|---|---|
| `Age` | Age in years |
| `Blood Pressure` | Blood pressure measurement |
| `Cholesterol Level` | Cholesterol measurement |
| `BMI` | Body mass index |
| `Sleep Hours` | Average sleep duration |
| `Triglyceride Level` | Triglyceride measurement |
| `Fasting Blood Sugar` | Fasting blood sugar measurement |
| `CRP Level` | C-reactive protein level |
| `Homocysteine Level` | Homocysteine measurement |

### Categorical input attributes

| Attribute | Values in the dataset |
|---|---|
| `Gender` | Female, Male |
| `Exercise Habits` | Low, Medium, High |
| `Smoking` | No, Yes |
| `Family Heart Disease` | No, Yes |
| `Diabetes` | No, Yes |
| `High Blood Pressure` | No, Yes |
| `Low HDL Cholesterol` | No, Yes |
| `High LDL Cholesterol` | No, Yes |
| `Alcohol Consumption` | Low, Medium, High |
| `Stress Level` | Low, Medium, High |
| `Sugar Consumption` | Low, Medium, High |

## SVM Architecture

The complete fitted pipeline is saved as `models/svm_pipeline.joblib`.

```text
Raw 20-attribute DataFrame
          |
          +-- Numerical attributes
          |     +-- Median imputation
          |     +-- StandardScaler
          |
          +-- Categorical attributes
                +-- Replace missing values with Unknown
                +-- OneHotEncoder(handle_unknown="ignore")
          |
          +-- SVC(class_weight="balanced")
```

The Streamlit application sends the raw input DataFrame directly to this pipeline. Separate preprocessing by the application is not required.

## Evaluation Results

The current saved model was evaluated on the 2,000-record test set. These values were produced by `train.py` and are also stored in `static/metrics.json`.

| Metric | Result |
|---|---:|
| Accuracy | 48.50% |
| Precision | 19.77% |
| Recall | 51.50% |
| F1-score | 28.57% |
| Best cross-validation F1 | 29.65% |

Confusion matrix:

```text
                 Predicted No  Predicted Yes
Actual No              764            836
Actual Yes             194            206
```

Selected SVM hyperparameters:

```text
C = 10
gamma = scale
kernel = linear
class_weight = balanced
```

The evaluation values are generated from the dataset and are not hard-coded by the application. Running `python train.py` again will update the saved model, metrics, and confusion matrix.

## Project Structure

```text
heart_disease_classifier/
|-- app.py
|-- train.py
|-- test_inference.py
|-- heart_disease.csv
|-- requirements.txt
|-- README.md
|-- models/
|   `-- svm_pipeline.joblib
`-- static/
    |-- metrics.json
    `-- svm_confusion_matrix.png
```

## Installation

Open a terminal in the project directory and install the dependencies:

```bash
pip install -r requirements.txt
```

## Training the SVM

Run:

```bash
python train.py
```

The script will:

1. Validate the local dataset and its column names.
2. create the stratified training and test sets.
3. Tune the SVM pipeline with five-fold cross-validation.
4. Evaluate the best pipeline on the test set.
5. Save `models/svm_pipeline.joblib`.
6. Save the actual metrics to `static/metrics.json`.
7. Generate `static/svm_confusion_matrix.png`.

## Testing Inference

Run the included sample prediction:

```bash
python test_inference.py
```

The sample contains all 20 input attributes and calls `.predict()` directly on the saved pipeline.

## Running the Streamlit Application

Run:

```bash
python -m streamlit run app.py
```

Then open `http://localhost:8501` if it does not open automatically.

The application contains:

- **Dataset Overview** — dataset size and outcome distribution.
- **Exploratory Data Analysis** — preview, missing values, numerical distributions, and numerical correlations.
- **SVM Model Performance** — actual accuracy, precision, recall, F1-score, and confusion matrix.
- **Prediction** — a form containing all 20 input attributes.
