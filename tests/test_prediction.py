"""Basic tests. Run from the project root:  python -m pytest"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import FEATURES, MODEL_PATH, NUMERIC_FEATURES, TARGET
from src.data_cleaning import check_split_leakage, prepare_dataset
from src.data_loader import load_raw_data
from src.predict import (build_input_frame, load_metadata, load_model, predict_outcome,
                         validate_inputs)
from src.preprocessing import build_pipeline, build_preprocessor, split_data

CLASSES = {"Healthy", "At-risk", "Frail"}
VALID_INPUT = {"age": 70, "gender": "Female", "education_years": 10, "income_level": "Low",
               "physical_activity": "Low", "smoking_status": "Former", "chronic_conditions": 3,
               "memory_score": 65.0, "bmi": 27.0, "blood_pressure": 140, "diet_quality": "Average",
               "alcohol_consumption": "None", "social_engagement": "Low"}


@pytest.fixture(scope="module")
def raw():
    return load_raw_data()


@pytest.fixture(scope="module")
def prepared(raw):
    clean, _ = prepare_dataset(raw)
    return clean


@pytest.fixture(scope="module")
def split(prepared):
    return split_data(prepared)


@pytest.fixture(scope="module")
def small_metadata(prepared):
    return {"observed_ranges": {c: [float(prepared[c].min()), float(prepared[c].max())]
                                for c in NUMERIC_FEATURES},
            "categories": {c: sorted(prepared[c].dropna().unique())
                           for c in prepared.columns if c not in NUMERIC_FEATURES + [TARGET]}}


@pytest.fixture(scope="module")
def quick_model(split):
    """A fast model trained inside the test, independent of the saved file."""
    X_train, _, y_train, _ = split
    return build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X_train, y_train)


# ---- data ---------------------------------------------------------------------
def test_dataset_loads(raw):
    assert len(raw) == 7488 and raw.shape[1] == 14


def test_target_column_exists_with_three_classes(raw):
    assert TARGET in raw.columns
    assert set(raw[TARGET].unique()) == CLASSES


def test_none_alcohol_is_a_category_not_missing(raw):
    assert raw["alcohol_consumption"].isna().sum() == 0
    assert (raw["alcohol_consumption"] == "None").sum() == 4008


def test_duplicates_removed_before_split(prepared):
    assert prepared.duplicated().sum() == 0


def test_no_train_test_leakage(split):
    X_train, X_test, _, _ = split
    assert check_split_leakage(X_train, X_test) == 0


def test_split_is_stratified_80_20(split, prepared):
    X_train, X_test, y_train, y_test = split
    assert abs(len(X_test) / len(prepared) - 0.20) < 0.01
    assert set(y_train.unique()) == CLASSES and set(y_test.unique()) == CLASSES


# ---- preprocessing / training -------------------------------------------------
def test_preprocessor_fits_on_training_data_only(split):
    X_train, X_test, _, _ = split
    preprocessor = build_preprocessor().fit(X_train)
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    assert np.isclose(scaler.mean_[0], X_train["age"].mean())  # learned from train, not test/all
    assert preprocessor.transform(X_test).shape[0] == len(X_test)


def test_model_training_completes(quick_model, split):
    _, X_test, _, _ = split
    assert set(quick_model.predict(X_test)) <= CLASSES


# ---- saved model --------------------------------------------------------------
@pytest.fixture(scope="module")
def saved_model():
    if not MODEL_PATH.exists():
        pytest.skip("Run `python -m src.train_models` first to create the saved model.")
    return load_model()


def test_saved_model_loads_and_predicts(saved_model):
    prediction = saved_model.predict(build_input_frame(VALID_INPUT))[0]
    assert prediction in CLASSES


def test_saved_pipeline_includes_preprocessing(saved_model):
    assert "preprocessor" in saved_model.named_steps


# ---- prediction function used by Streamlit -----------------------------------
def test_predict_outcome_valid_input(quick_model, small_metadata):
    result = predict_outcome(VALID_INPUT, model=quick_model, metadata=small_metadata)
    assert result["label"] in CLASSES
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6


@pytest.mark.parametrize("field,value", [
    ("age", None), ("age", "abc"), ("age", -5), ("bmi", 500), ("memory_score", 150),
    ("gender", "Unknown"), ("alcohol_consumption", ""), ("smoking_status", 7)])
def test_invalid_input_is_rejected(quick_model, small_metadata, field, value):
    bad = dict(VALID_INPUT, **{field: value})
    with pytest.raises(ValueError):
        predict_outcome(bad, model=quick_model, metadata=small_metadata)


def test_missing_field_is_rejected(small_metadata):
    incomplete = {k: v for k, v in VALID_INPUT.items() if k != "age"}
    errors, _ = validate_inputs(incomplete, small_metadata)
    assert any("age" in e for e in errors)


def test_value_outside_training_range_gives_warning_not_error(small_metadata):
    errors, warnings = validate_inputs(dict(VALID_INPUT, age=100), small_metadata)
    assert not errors and warnings


def test_input_frame_has_exact_feature_order():
    assert list(build_input_frame(VALID_INPUT).columns) == FEATURES
