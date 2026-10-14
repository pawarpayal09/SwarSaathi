"""Hyperparameter tuning with GridSearchCV (training data only)."""
import time
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from src.config import CV_FOLDS, CV_REPEATS, RANDOM_STATE
from src.preprocessing import build_pipeline

# Small grids so tuning finishes in a few minutes on a student laptop.
# Keys use the "model__" prefix because the classifier is a step called "model".
PARAM_GRIDS = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "Decision Tree": {"model__max_depth": [3, 5, 8, None],
                      "model__min_samples_leaf": [1, 5, 10]},
    "Random Forest": {"model__n_estimators": [200, 400],
                      "model__max_depth": [None, 8, 15],
                      "model__min_samples_leaf": [1, 3, 5]},
    "SVM (RBF)": {"model__C": [0.5, 1, 5, 20],
                  "model__gamma": ["scale", 0.01, 0.1]},
    "K-Nearest Neighbors": {"model__n_neighbors": [3, 5, 9, 15, 25],
                            "model__weights": ["uniform", "distance"]},
    "Gradient Boosting": {"model__learning_rate": [0.03, 0.1],
                          "model__max_depth": [3, None],
                          "model__max_iter": [100, 200]},
    "Soft Voting Ensemble": {"model__svm__C": [10, 20, 50],
                             "model__lr__C": [0.3, 1, 10]},
}


def tune_model(name, model, X_train, y_train, scoring="f1_macro"):
    """Run a grid search with stratified CV; return (best_pipeline, info dict).

    The search sees ONLY the training data. best_estimator_ is refit on all
    of the training data automatically.
    """
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    search = GridSearchCV(build_pipeline(model), PARAM_GRIDS[name], scoring=scoring,
                          cv=cv, n_jobs=-1, refit=True)
    start = time.perf_counter()
    search.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, {
        "best_params": best_params,
        "cv_macro_f1": float(search.best_score_),
        "cv_macro_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        "tuning_seconds": seconds,
    }
