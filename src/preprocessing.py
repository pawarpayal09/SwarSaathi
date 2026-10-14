"""Preprocessing pipeline, model definitions and the train/test split."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                        RANDOM_STATE, TARGET, TEST_SIZE)


def build_preprocessor() -> ColumnTransformer:
    """Numbers: fill gaps with the median, then standardise.
    Categories: fill gaps with the most frequent value, then one-hot encode.
    Nothing is fitted here - fitting happens inside the Pipeline on training data only.
    """
    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_steps, NUMERIC_FEATURES),
        ("categorical", categorical_steps, CATEGORICAL_FEATURES),
    ])


ENSEMBLE_NAME = "Soft Voting Ensemble"


def build_ensemble() -> VotingClassifier:
    """Soft-voting ensemble: averages the class probabilities of three different learners.

    SVM (smooth boundary) + Logistic Regression (linear) + Gradient Boosting (trees) make
    different kinds of mistakes, so averaging them is usually more stable than any single one.
    """
    return VotingClassifier(
        estimators=[
            ("svm", SVC(kernel="rbf", C=20, gamma=0.01, class_weight="balanced",
                        probability=True, random_state=RANDOM_STATE)),
            ("lr", LogisticRegression(max_iter=3000, C=1, class_weight="balanced",
                                      random_state=RANDOM_STATE)),
            ("gb", HistGradientBoostingClassifier(
                learning_rate=0.05, max_depth=3, class_weight="balanced",
                random_state=RANDOM_STATE)),
        ],
        voting="soft",
    )


def get_models() -> dict:
    """The six candidate classifiers (default-ish settings = 'baseline').

    class_weight='balanced' is used where supported because the cleaned data
    is very imbalanced (only 22 Healthy rows). KNN has no such option.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(
            kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE),
        ENSEMBLE_NAME: build_ensemble(),
    }


MODEL_REASONS = {
    "Logistic Regression": "Simple, fast linear baseline; coefficients are easy to explain.",
    "Decision Tree": "Learns easy if-then rules; very readable but can overfit.",
    "Random Forest": "Many trees voting together; handles mixed features and is robust to noise.",
    "SVM (RBF)": "Finds a smooth boundary between classes; works well on small, scaled data.",
    "K-Nearest Neighbors": "Classifies by similar people in the data; easy to understand.",
    "Gradient Boosting": "Builds trees one after another, each fixing earlier mistakes; strong on tables.",
    ENSEMBLE_NAME: "Averages SVM + Logistic Regression + Gradient Boosting so their different "
                   "errors partly cancel out; usually the most stable choice on small data.",
}


def build_pipeline(model) -> Pipeline:
    """Preprocessing + classifier in ONE object, so prediction repeats the same steps."""
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def split_data(df: pd.DataFrame):
    """Stratified 80/20 split with a fixed random_state (df must already be de-duplicated)."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
