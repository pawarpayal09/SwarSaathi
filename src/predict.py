"""Loading the saved pipeline and predicting for one new person.
The Streamlit app only calls the functions in this file; it never trains anything.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import (CLASS_ORDER, FEATURES, METADATA_PATH, MODEL_PATH, NUMERIC_FEATURES,
                        RISK_WEIGHTS, VALID_RANGES)


class ModelNotAvailableError(Exception):
    """Raised when the trained model or its metadata cannot be found."""


def load_metadata(path: Path = METADATA_PATH) -> dict:
    if not Path(path).exists():
        raise ModelNotAvailableError("Model information not found. Train the model first.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model(path: Path = MODEL_PATH):
    """Load the complete saved pipeline (preprocessing + classifier)."""
    if not Path(path).exists():
        raise ModelNotAvailableError("Trained model not found. Run: python -m src.train_models")
    return joblib.load(path)


def validate_inputs(inputs: dict, metadata: dict):
    """Check one user record. Returns (errors, warnings).

    errors   -> the record cannot be used (missing, wrong type, impossible value, unknown category)
    warnings -> value is possible but outside the range seen in the training data
    """
    errors, warnings = [], []
    for feature in FEATURES:
        value = inputs.get(feature)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"'{feature}' is required.")
            continue
        if feature in NUMERIC_FEATURES:
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"'{feature}' must be a number.")
                continue
            if number != number:  # NaN check
                errors.append(f"'{feature}' must be a number.")
                continue
            low, high = VALID_RANGES[feature]
            if not low <= number <= high:
                errors.append(f"'{feature}' must be between {low} and {high}.")
                continue
            seen_low, seen_high = metadata["observed_ranges"][feature]
            if not seen_low <= number <= seen_high:
                warnings.append(f"'{feature}' = {number:g} is outside the range seen in the "
                                f"training data ({seen_low:g} to {seen_high:g}); "
                                "the prediction may be less reliable.")
        else:
            if value not in metadata["categories"][feature]:
                errors.append(f"'{feature}' must be one of {metadata['categories'][feature]}.")
    return errors, warnings


def build_input_frame(inputs: dict) -> pd.DataFrame:
    """One-row DataFrame with the exact column names/order the pipeline was trained on."""
    return pd.DataFrame([{feature: inputs[feature] for feature in FEATURES}], columns=FEATURES)


def predict_outcome(inputs: dict, model=None, metadata: dict = None) -> dict:
    """Validate, predict, and return a dictionary the UI can display.

    Returns: {"label", "probabilities" (dict or None), "warnings", "confidence", "risk_index",
              "confidence_note"}
    Raises ValueError with a readable message if the input is invalid.
    """
    metadata = metadata or load_metadata()
    model = model or load_model()
    errors, warnings = validate_inputs(inputs, metadata)
    if errors:
        raise ValueError(" ".join(errors))
    frame = build_input_frame(inputs)
    label = str(model.predict(frame)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(frame)[0]
        probabilities = {str(c): float(p) for c, p in zip(model.classes_, scores)}
    result = {"label": label, "probabilities": probabilities, "warnings": warnings,
              "confidence": None, "risk_index": None, "confidence_note": ""}
    if probabilities:
        result["confidence"] = max(probabilities.values())
        result["risk_index"] = risk_index(probabilities)
        result["confidence_note"] = describe_confidence(probabilities)
    return result


def risk_index(probabilities: dict) -> float:
    """0-100 'expected severity': Healthy=0, At-risk=50, Frail=100, weighted by the model's
    class probabilities. A simple, transparent summary of the three probabilities."""
    return float(100 * sum(RISK_WEIGHTS.get(c, 0) * p for c, p in probabilities.items()))


def describe_confidence(probabilities: dict) -> str:
    """Plain-language comment on how decisive the model was."""
    ranked = sorted(probabilities.values(), reverse=True)
    top, second = ranked[0], ranked[1] if len(ranked) > 1 else 0.0
    if top >= 0.80:
        return "The model is quite decisive for this profile."
    if top - second < 0.15:
        return "The top two categories are close, so this profile sits near a boundary."
    return "The model leans towards one category but is not very decisive."


def predict_probabilities(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Class probabilities for many rows at once, columns ordered Healthy/At-risk/Frail."""
    scores = pd.DataFrame(model.predict_proba(frame), columns=[str(c) for c in model.classes_])
    return scores.reindex(columns=[c for c in CLASS_ORDER if c in scores.columns])


def fmt_pct(probability: float) -> str:
    """Show a probability as text without claiming impossible certainty (never '100.0%')."""
    pct = probability * 100
    if pct > 99.9:
        return ">99.9%"
    if pct < 0.1:
        return "<0.1%"
    return f"{pct:.1f}%"
