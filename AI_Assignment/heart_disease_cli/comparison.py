"""Display the saved SVM evaluation results."""

from utils import banner, load_metrics, press_enter, print_confusion_matrix, print_table, section


def show_performance(metrics: dict):
    section("Test-Set Metrics")
    print_table({
        "Accuracy": f"{metrics['Accuracy'] * 100:.2f}%",
        "Precision": f"{metrics['Precision'] * 100:.2f}%",
        "Recall": f"{metrics['Recall'] * 100:.2f}%",
        "F1-score": f"{metrics['F1-Score'] * 100:.2f}%",
        "Best cross-validation F1": f"{metrics['BestCrossValidationF1'] * 100:.2f}%",
        "Training records": metrics["TrainSize"],
        "Test records": metrics["TestSize"],
    })
    section("Best SVM Parameters")
    for name, value in metrics["BestParameters"].items():
        print(f"  {name.removeprefix('svm__'):<16} {value}")
    section("SVM Confusion Matrix")
    print_confusion_matrix(metrics["ConfusionMatrix"])
    print(
        "  Accuracy alone may be misleading because approximately 80% of records\n"
        "  have status No and 20% have status Yes. Precision, recall, and F1-score\n"
        "  provide essential context for this imbalanced classification task."
    )


def run():
    banner("SVM MODEL PERFORMANCE")
    metrics = load_metrics()
    if metrics is None:
        print("\n  [ERROR] metrics.json is missing. Run Model Training first.")
    else:
        show_performance(metrics)
    press_enter()
