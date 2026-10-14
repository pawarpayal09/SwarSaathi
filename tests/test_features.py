"""Tests for the new modules: explanations, what-if, batch prediction, PDF report."""
import pandas as pd
import pytest

from src.batch import predict_batch, read_batch_csv
from src.config import MODEL_PATH, SAMPLE_BATCH_PATH
from src.explain import (improvement_opportunities, lifestyle_scores, risk_drivers, what_if,
                         wellness_tips)
from src.predict import fmt_pct, load_metadata, load_model, predict_outcome, risk_index
from src.report import build_pdf

PROFILE = {"age": 70, "gender": "Female", "education_years": 8, "income_level": "Low",
           "physical_activity": "Low", "smoking_status": "Former", "chronic_conditions": 3,
           "memory_score": 65.0, "bmi": 27.0, "blood_pressure": 140, "diet_quality": "Poor",
           "alcohol_consumption": "None", "social_engagement": "Low"}

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="Train the model first")


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def meta():
    return load_metadata()


def test_result_has_new_fields(model, meta):
    r = predict_outcome(PROFILE, model, meta)
    assert 0 <= r["risk_index"] <= 100 and 0 < r["confidence"] <= 1 and r["confidence_note"]


def test_risk_index_extremes():
    assert risk_index({"Healthy": 1, "At-risk": 0, "Frail": 0}) == 0
    assert risk_index({"Healthy": 0, "At-risk": 0, "Frail": 1}) == 100


def test_fmt_pct_never_claims_certainty():
    assert fmt_pct(1.0) == ">99.9%" and fmt_pct(0.0) == "<0.1%" and fmt_pct(0.5) == "50.0%"


def test_drivers_cover_all_features(model, meta):
    d = risk_drivers(model, PROFILE, meta["reference_profile"])
    assert len(d) == 13 and abs(d["share"].abs().sum() - 100) < 1e-6


def test_improvement_opportunities_exist(model):
    opps = improvement_opportunities(model, PROFILE)
    assert len(opps) > 0 and (opps["score_change"] <= 1e-9).sum() >= 1


def test_what_if_no_change_is_identical(model):
    r = what_if(model, PROFILE, {})
    assert abs(r["risk_before"] - r["risk_after"]) < 1e-9


def test_lifestyle_scores_in_range():
    assert all(0 <= v <= 100 for v in lifestyle_scores(PROFILE).values())


def test_tips_have_priority():
    tips = wellness_tips(PROFILE)
    assert tips and {t["priority"] for t in tips} <= {"High", "Medium", "Good"}


def test_batch_prediction_sample_file(model, meta):
    out, problems = predict_batch(read_batch_csv(SAMPLE_BATCH_PATH), model, meta)
    assert not problems and out["prediction"].notna().all()


def test_batch_flags_invalid_rows(model, meta):
    df = read_batch_csv(SAMPLE_BATCH_PATH)
    df.loc[0, "age"] = 500
    df.loc[1, "gender"] = "Robot"
    out, problems = predict_batch(df, model, meta)
    assert len(problems) == 2 and (out["status"] == "Invalid").sum() == 2


def test_batch_missing_column_raises(model, meta):
    with pytest.raises(ValueError):
        predict_batch(pd.DataFrame({"age": [50]}), model, meta)


def test_pdf_report_is_a_pdf(model, meta):
    r = predict_outcome(PROFILE, model, meta)
    pdf = build_pdf(r, PROFILE, risk_drivers(model, PROFILE, meta["reference_profile"]),
                    wellness_tips(PROFILE), "Test model")
    assert pdf[:4] == b"%PDF"
