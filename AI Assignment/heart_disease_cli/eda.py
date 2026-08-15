"""Mixed-type exploratory data analysis for the shared 10,000-row dataset."""

from utils import (
    CATEGORICAL_FEATURES, FEATURES, NUMERICAL_FEATURES, STATUS_COLUMN,
    banner, load_dataset, press_enter, print_table, prompt_choice, section,
)


def _show_preview(df):
    section("Dataset Preview (First 10 Rows)")
    print(df.head(10).to_string(index=False))
    press_enter()


def _show_stats(df):
    section("Numerical Statistical Summary")
    print(df[NUMERICAL_FEATURES].describe().T.to_string(float_format=lambda value: f"{value:.2f}"))
    press_enter()


def _show_missing(df):
    section("Missing-Value Summary")
    missing = df.isna().sum()
    summary = {column: int(count) for column, count in missing.items()}
    print_table(summary)
    print(f"\n  Total missing values: {int(missing.sum())}")
    press_enter()


def _show_distribution(df):
    section("Feature Distribution")
    index = prompt_choice("Select feature", FEATURES + ["Back"])
    if index == len(FEATURES):
        return
    feature = FEATURES[index]
    if feature in NUMERICAL_FEATURES:
        values = df[feature]
        print_table({
            "Count": int(values.count()), "Minimum": float(values.min()),
            "Maximum": float(values.max()), "Mean": float(values.mean()),
            "Median": float(values.median()), "Standard deviation": float(values.std()),
        })
        print("\n  Split by Heart Disease Status:")
        grouped = df.groupby(STATUS_COLUMN)[feature].agg(["count", "min", "max", "mean", "median", "std"])
        print(grouped.to_string(float_format=lambda value: f"{value:.2f}"))
    else:
        print("\n  Counts split by Heart Disease Status:")
        counts = df.groupby([feature, STATUS_COLUMN], dropna=False).size().unstack(fill_value=0)
        print(counts.to_string())
    press_enter()


def _show_correlations(df):
    section("Numerical Correlation with Heart Disease Status")
    numeric = df[NUMERICAL_FEATURES].copy()
    numeric[STATUS_COLUMN] = df[STATUS_COLUMN].map({"No": 0, "Yes": 1})
    correlations = numeric.corr()[STATUS_COLUMN].drop(STATUS_COLUMN).sort_values(key=abs, ascending=False)
    for feature, value in correlations.items():
        print(f"  {feature:<28} {value: .4f}")
    print("\n  Only numerical inputs are included; categorical text is excluded.")
    press_enter()


def run():
    df = load_dataset()
    if df is None:
        banner("EXPLORATORY DATA ANALYSIS")
        print("\n  [ERROR] The shared dataset is missing.")
        press_enter()
        return
    while True:
        banner("EXPLORATORY DATA ANALYSIS")
        print(f"\n  Dataset: {len(df):,} records, {len(FEATURES)} input attributes")
        choice = prompt_choice("Choose an option", [
            "Dataset Preview", "Numerical Statistical Summary", "Missing-Value Summary",
            "Feature Distribution", "Numerical Correlation with Heart Disease Status",
            "Back to Main Menu",
        ])
        if choice == 0:
            _show_preview(df)
        elif choice == 1:
            _show_stats(df)
        elif choice == 2:
            _show_missing(df)
        elif choice == 3:
            _show_distribution(df)
        elif choice == 4:
            _show_correlations(df)
        else:
            break
