# Heart Disease Classification Prediction CLI

This terminal application is the CLI interface for the TARUMT Artificial Intelligence assignment. It uses the same complete preprocessing + Support Vector Machine pipeline as the sibling Streamlit project.

SVC is the only core classification algorithm used.

> **Disclaimer:** This is an academic AI prototype and is not a medical diagnosis. Its classifications must not replace professional medical evaluation or advice.

## Shared Dataset and Model

The CLI does not maintain its own dataset or model copies. It reads:

- `../heart_disease_classifier/heart_disease.csv`
- `../heart_disease_classifier/models/svm_pipeline.joblib`
- `../heart_disease_classifier/static/metrics.json`
- `../heart_disease_classifier/static/svm_confusion_matrix.png`

The dataset contains 10,000 records, 20 inputs, and the outcome `Heart Disease Status`. Its current class counts are 8,000 `No` and 2,000 `Yes`.

### Nine numerical inputs

- Age
- Blood Pressure
- Cholesterol Level
- BMI
- Sleep Hours
- Triglyceride Level
- Fasting Blood Sugar
- CRP Level
- Homocysteine Level

### Eleven categorical inputs

- Gender
- Exercise Habits
- Smoking
- Family Heart Disease
- Diabetes
- High Blood Pressure
- Low HDL Cholesterol
- High LDL Cholesterol
- Alcohol Consumption
- Stress Level
- Sugar Consumption

## Architecture

The prediction form constructs a raw one-row pandas DataFrame using all 20 exact CSV input names. It passes that DataFrame directly to `svm_pipeline.joblib`, which performs missing-value handling, scaling, one-hot encoding, and SVM classification.

The CLI reports only a `Yes` or `No` classification. It does not calculate or claim a probability.

## Menu

1. **Overview Dashboard** — dynamically displays dataset size, class counts, objectives, and all attributes.
2. **Exploratory Data Analysis** — dataset preview, numerical statistics, missing-value summary, feature distributions, and numerical correlation with encoded status.
3. **SVM Model Performance** — accuracy, precision, recall, F1-score, tuning results, split sizes, and the SVM confusion matrix.
4. **Heart Disease Classification Prediction** — interactive form for all 20 inputs.
5. **Model Training** — executes the sibling `heart_disease_classifier/train.py` implementation.
6. **Exit**.

## Current Saved Evaluation

The values below are read from the current `metrics.json` generated from the 2,000-record test set:

| Metric | Value |
|---|---:|
| Accuracy | 48.50% |
| Precision | 19.77% |
| Recall | 51.50% |
| F1-score | 28.57% |
| Best cross-validation F1 | 29.65% |

Selected parameters: `C=10`, `gamma=scale`, and `kernel=linear`, with balanced class weights.

Accuracy alone can be misleading because approximately 80% of records have status `No` and 20% have status `Yes`. Precision, recall, and F1-score are therefore shown alongside accuracy.

## Installation and Launch

From this directory:

```bash
pip install -r requirements.txt
python main.py
```

The CLI resolves shared paths relative to its own file location, so it can also be launched from another working directory.

## Retraining

Choose **Model Training** and then **Run shared SVM training**. The CLI executes:

```bash
python ../heart_disease_classifier/train.py
```

Training recreates only the fitted SVM pipeline, metrics JSON, and one SVM confusion-matrix image used by both interfaces.
