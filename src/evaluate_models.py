"""Metrics, tables and plots. Can also be run alone to re-evaluate the saved model:
    python -m src.evaluate_models
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")  # draw to files only; no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from src.config import (CLASS_ORDER, FEATURES, MODEL_PATH, RANDOM_STATE, REPORTS_DIR,
                        SUMMARY_PATH, TARGET)


def compute_metrics(pipeline, X_test, y_test) -> dict:
    """All headline metrics for one fitted pipeline on the held-out test set."""
    pred = pipeline.predict(X_test)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0)
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "test_macro_precision": prec,
        "test_macro_recall": rec,
        "test_macro_f1": f1,
        "test_weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
    }


def per_class_table(pipeline, X_test, y_test) -> pd.DataFrame:
    """Precision / recall / F1 / support for every class."""
    pred = pipeline.predict(X_test)
    labels = sorted(y_test.unique())
    p, r, f, s = precision_recall_fscore_support(y_test, pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f, "support": s})


def compute_metrics_from_predictions(y_true, pred) -> dict:
    """Headline metrics when we already have predictions (e.g. out-of-fold)."""
    return {"accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0))}


def per_class_table_from_predictions(y_true, pred) -> pd.DataFrame:
    labels = [c for c in CLASS_ORDER if c in set(y_true)]
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f, "support": s})


def plot_confusion_matrix_from_predictions(y_true, pred, title, path):
    labels = CLASS_ORDER
    cm = confusion_matrix(y_true, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def timed_fit(pipeline, X_train, y_train):
    """Fit and return (fitted_pipeline, seconds)."""
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    return pipeline, time.perf_counter() - start


# ---------------------------------------------------------------- plots
def plot_target_distribution(before: dict, after: dict, path):
    order = ["Healthy", "At-risk", "Frail"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for ax, counts, title in [(axes[0], before, "Original file (7,488 rows)"),
                              (axes[1], after, "After removing exact duplicates")]:
        values = [counts.get(c, 0) for c in order]
        bars = ax.bar(order, values, color=["#4caf50", "#ffb300", "#e53935"])
        ax.set_title(title)
        ax.set_ylabel("Number of records")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center", va="bottom")
    fig.suptitle("Target class distribution (aging_outcome)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(table: pd.DataFrame, path):
    """Bar chart: cross-validated macro-F1 (training data) vs test macro-F1."""
    data = table.sort_values("cv_macro_f1")
    positions = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(data) + 2))
    ax.barh(positions - 0.2, data["cv_macro_f1"], height=0.4, label="CV macro-F1 (training data)")
    ax.barh(positions + 0.2, data["test_macro_f1"], height=0.4, label="Test macro-F1")
    ax.set_yticks(positions)
    ax.set_yticklabels(data["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1-score")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(pipeline, X_test, y_test, title, path):
    labels = ["Healthy", "At-risk", "Frail"]
    cm = confusion_matrix(y_test, pipeline.predict(X_test), labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_feature_importance(pipeline, X_test, y_test) -> pd.DataFrame:
    """Permutation importance on the ORIGINAL 13 input columns.

    Each column is shuffled and we measure how much macro-F1 drops. Because the
    pipeline contains the preprocessing, one-hot columns never need renaming.
    This works for every model type. It shows what the model USES, not what CAUSES aging.
    """
    result = permutation_importance(pipeline, X_test, y_test, scoring="f1_macro",
                                    n_repeats=20, random_state=RANDOM_STATE, n_jobs=1)
    return (pd.DataFrame({"feature": FEATURES, "importance_mean": result.importances_mean,
                          "importance_std": result.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def plot_feature_importance(importance: pd.DataFrame, model_name: str, path):
    data = importance.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(data["feature"], data["importance_mean"], xerr=data["importance_std"], color="#3f6fb5")
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Drop in test macro-F1 when the feature is shuffled")
    ax.set_title(f"Permutation feature importance - {model_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_classification_report(pipeline, X_test, y_test, model_name, path):
    pred = pipeline.predict(X_test)
    text = (f"Final model: {model_name}\nEvaluated on the held-out test set "
            f"({len(y_test)} records).\n\n"
            + classification_report(y_test, pred, zero_division=0)
            + "\nNote: the 'Healthy' class has very few independent records, so its "
              "scores are based on a handful of test cases and are unreliable.\n")
    path.write_text(text, encoding="utf-8")
    return text


def main():
    """Re-evaluate the saved model on the same reproducible test split."""
    from src.data_cleaning import prepare_dataset
    from src.data_loader import load_raw_data
    from src.preprocessing import split_data

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run: python -m src.train_models")
    pipeline = joblib.load(MODEL_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else {}
    name = summary.get("final_model", "Saved model")
    clean, _ = prepare_dataset(load_raw_data())
    _, X_test, _, y_test = split_data(clean)
    print(f"Re-evaluating {name} on {len(y_test)} test records")
    for key, value in compute_metrics(pipeline, X_test, y_test).items():
        print(f"  {key}: {value:.4f}")
    print(write_classification_report(pipeline, X_test, y_test, name,
                                      REPORTS_DIR / "classification_report.txt"))
    plot_confusion_matrix(pipeline, X_test, y_test, f"Confusion matrix - {name}",
                          REPORTS_DIR / "confusion_matrix.png")
    importance = compute_feature_importance(pipeline, X_test, y_test)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(importance, name, REPORTS_DIR / "feature_importance.png")
    print("Reports refreshed in the reports/ folder.")


if __name__ == "__main__":
    main()
