"""Train, tune, compare and save the SwarSaathi model.

Run from the project root:
    python -m src.train_models                 # full run (with tuning)
    python -m src.train_models --skip-tuning   # quicker run
"""
import argparse
import json
import platform

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.metrics import classification_report

from src.config import (CATEGORICAL_FEATURES, CLASS_ORDER, CV_FOLDS, CV_REPEATS, FEATURES,
                        METADATA_PATH, MODEL_PATH, MODELS_DIR, NUMERIC_FEATURES, OOF_CM_PATH,
                        OOF_REPORT_PATH, RANDOM_STATE, REPORTS_DIR, SUMMARY_PATH, TARGET,
                        TEST_SIZE)
from src.data_cleaning import audit_data, check_split_leakage, prepare_dataset
from src.data_loader import load_raw_data
from src.evaluate_models import (compute_feature_importance, compute_metrics,
                                 compute_metrics_from_predictions, per_class_table,
                                 per_class_table_from_predictions, plot_confusion_matrix,
                                 plot_confusion_matrix_from_predictions, plot_feature_importance,
                                 plot_model_comparison, plot_target_distribution, timed_fit,
                                 write_classification_report)
from src.preprocessing import MODEL_REASONS, build_pipeline, get_models, split_data
from src.tune_models import tune_model

SELECTION_METRIC = "cv_macro_f1"  # mean repeated 5-fold macro-F1 on TRAINING data only


def make_row(name, stage, pipeline, seconds, cv_mean, cv_std, X_test, y_test, params=None):
    """One row of the comparison table."""
    row = {"model": name, "stage": stage, "cv_macro_f1": cv_mean, "cv_macro_f1_std": cv_std}
    row.update(compute_metrics(pipeline, X_test, y_test))
    row["training_seconds"] = seconds
    row["best_params"] = json.dumps(params) if params else ""
    return row


def main(skip_tuning: bool = False, top_n: int = 3):
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # ---- 1. load + audit (original data is never modified) -------------------
    raw = load_raw_data()
    audit = audit_data(raw)
    print(f"Loaded {audit['rows']} rows; exact duplicates: {audit['exact_duplicate_rows']}; "
          f"unique rows: {audit['unique_rows']}")
    print("Target after removing duplicates:", audit["target_distribution_after_dedup"])

    # ---- 2. clean (de-duplicate BEFORE splitting) + split ---------------------
    clean, cleaning_log = prepare_dataset(raw)
    X_train, X_test, y_train, y_test = split_data(clean)
    leaked = check_split_leakage(X_train, X_test)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | "
          f"test rows identical to a training row: {leaked}")
    if leaked:
        raise RuntimeError("Data leakage: identical rows found in train and test sets.")
    print("Train classes:", y_train.value_counts().to_dict())
    print("Test classes: ", y_test.value_counts().to_dict())
    plot_target_distribution(audit["target_distribution_original"],
                             audit["target_distribution_after_dedup"],
                             REPORTS_DIR / "target_distribution.png")

    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    rows, fitted = [], {}

    # ---- 3. reference: always predicting the most common class ----------------
    dummy = build_pipeline(DummyClassifier(strategy="most_frequent"))
    scores = cross_val_score(dummy, X_train, y_train, cv=cv, scoring="f1_macro")
    dummy, secs = timed_fit(dummy, X_train, y_train)
    rows.append(make_row("Majority-class baseline", "reference", dummy, secs,
                         scores.mean(), scores.std(), X_test, y_test))

    # ---- 4. baseline models ---------------------------------------------------
    for name, model in get_models().items():
        print(f"Training baseline: {name}")
        pipeline = build_pipeline(model)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        pipeline, secs = timed_fit(pipeline, X_train, y_train)
        rows.append(make_row(name, "baseline", pipeline, secs, scores.mean(), scores.std(),
                             X_test, y_test))
        fitted[(name, "baseline")] = pipeline

    # ---- 5. tune the most promising models (training data only) ---------------
    if not skip_tuning:
        baseline_rows = sorted([r for r in rows if r["stage"] == "baseline"],
                               key=lambda r: r[SELECTION_METRIC], reverse=True)
        for base in baseline_rows[:top_n]:
            name = base["model"]
            print(f"Tuning: {name}")
            best_pipe, info = tune_model(name, get_models()[name], X_train, y_train)
            rows.append(make_row(name, "tuned", best_pipe, info["tuning_seconds"],
                                 info["cv_macro_f1"], info["cv_macro_f1_std"],
                                 X_test, y_test, info["best_params"]))
            fitted[(name, "tuned")] = best_pipe

    # ---- 6. comparison table + selection --------------------------------------
    table = pd.DataFrame(rows).sort_values(SELECTION_METRIC, ascending=False).reset_index(drop=True)
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False, float_format="%.4f")
    plot_model_comparison(table, REPORTS_DIR / "model_comparison.png")

    candidates = table[table["stage"] != "reference"]
    best = candidates.iloc[0]
    best_name, best_stage = best["model"], best["stage"]
    final_pipeline = fitted[(best_name, best_stage)]
    final_label = f"{best_name} ({best_stage})"
    print(f"\nSelected: {final_label} | CV macro-F1 {best['cv_macro_f1']:.3f} | "
          f"test macro-F1 {best['test_macro_f1']:.3f}")

    # ---- 7. detailed evaluation of the selected model -------------------------
    write_classification_report(final_pipeline, X_test, y_test, final_label,
                                REPORTS_DIR / "classification_report.txt")
    plot_confusion_matrix(final_pipeline, X_test, y_test, f"Confusion matrix - {final_label}",
                          REPORTS_DIR / "confusion_matrix.png")
    class_table = per_class_table(final_pipeline, X_test, y_test)
    class_table.to_csv(REPORTS_DIR / "per_class_metrics.csv", index=False, float_format="%.4f")
    importance = compute_feature_importance(final_pipeline, X_test, y_test)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False, float_format="%.4f")
    plot_feature_importance(importance, final_label, REPORTS_DIR / "feature_importance.png")

    # Cross-validated per-class results on training data: more Healthy cases than the test set.
    single_cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_pred = cross_val_predict(clone(final_pipeline), X_train, y_train, cv=single_cv, n_jobs=-1)
    cv_report = classification_report(y_train, cv_pred, output_dict=True, zero_division=0)

    # Out-of-fold evaluation on ALL unique records: every Healthy case (22) is scored once by a
    # model that never saw it. Far more informative than the 5 Healthy cases in the test split.
    X_all, y_all = clean[FEATURES], clean[TARGET]
    oof_pred = cross_val_predict(clone(final_pipeline), X_all, y_all, cv=single_cv, n_jobs=-1)
    oof_table = per_class_table_from_predictions(y_all, oof_pred)
    oof_table.to_csv(OOF_REPORT_PATH, index=False, float_format="%.4f")
    plot_confusion_matrix_from_predictions(
        y_all, oof_pred, f"Out-of-fold confusion matrix - all {len(y_all)} unique records", OOF_CM_PATH)
    oof_metrics = compute_metrics_from_predictions(y_all, oof_pred)
    print(f"Out-of-fold (all unique rows): accuracy {oof_metrics['accuracy']:.3f} | "
          f"macro-F1 {oof_metrics['macro_f1']:.3f} | balanced acc {oof_metrics['balanced_accuracy']:.3f}")

    # ---- 8. save model, metadata and summary ----------------------------------
    joblib.dump(final_pipeline, MODEL_PATH)
    metadata = {
        "feature_order": FEATURES,
        "class_labels": [str(c) for c in final_pipeline.classes_],
        "categories": audit["categories"],
        "observed_ranges": audit["observed_ranges"],
        "supports_probabilities": hasattr(final_pipeline, "predict_proba"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        # Typical ("average person") values: used by the app to explain which inputs push risk up.
        "reference_profile": {
            **{c: float(clean[c].median()) for c in NUMERIC_FEATURES},
            **{c: str(clean[c].mode().iloc[0]) for c in CATEGORICAL_FEATURES},
        },
        "numeric_summary": {c: {"mean": float(clean[c].mean()), "std": float(clean[c].std()),
                                "min": float(clean[c].min()), "max": float(clean[c].max())}
                            for c in NUMERIC_FEATURES},
        "training_rows": int(len(clean)),
        "class_counts": {k: int(v) for k, v in clean[TARGET].value_counts().items()},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    final_metrics = {k: float(best[k]) for k in table.columns if k.startswith("test_")}
    summary = {
        "final_model": final_label,
        "selection_metric": f"{SELECTION_METRIC} (mean {CV_FOLDS}-fold stratified macro-F1 on training data)",
        "selected_cv_macro_f1": float(best["cv_macro_f1"]),
        "selected_params": json.loads(best["best_params"]) if best["best_params"] else {},
        "test_metrics": final_metrics,
        "per_class_test_metrics": class_table.to_dict(orient="records"),
        "per_class_cv_metrics_training_data": {
            k: {m: float(v) for m, v in vals.items()}
            for k, vals in cv_report.items() if k in metadata["class_labels"]},
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE,
                  "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                  "train_class_counts": y_train.value_counts().to_dict(),
                  "test_class_counts": y_test.value_counts().to_dict(),
                  "test_rows_identical_to_training_rows": leaked},
        "data_audit": audit,
        "cleaning_log": cleaning_log,
        "out_of_fold_metrics_all_unique_rows": oof_metrics,
        "out_of_fold_per_class": oof_table.to_dict(orient="records"),
        "cv_scheme": f"{CV_REPEATS}x repeated stratified {CV_FOLDS}-fold",
        "model_reasons": MODEL_REASONS,
        "tuning_performed": not skip_tuning,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # ---- 9. prove the saved file works ----------------------------------------
    reloaded = joblib.load(MODEL_PATH)
    sample_pred = reloaded.predict(X_test.head(3))
    print("Reload check OK. Sample predictions:", list(sample_pred))
    print("\nTop of comparison table:")
    print(table[["model", "stage", "cv_macro_f1", "test_accuracy", "test_macro_f1"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nDone. Reports are in reports/, model is in models/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SwarSaathi models")
    parser.add_argument("--skip-tuning", action="store_true", help="skip GridSearchCV")
    parser.add_argument("--top-n", type=int, default=3, help="how many models to tune")
    args = parser.parse_args()
    main(skip_tuning=args.skip_tuning, top_n=args.top_n)
