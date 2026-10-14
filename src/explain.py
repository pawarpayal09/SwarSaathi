"""Explaining one prediction, what-if simulation and wellness tips.

Everything here works with ANY fitted pipeline (no special model internals needed):
we simply re-run the model on slightly changed copies of the person's record.

IMPORTANT: results describe how the MODEL behaves on a dataset of unverified origin. They show
associations, not medical causes, and are not medical advice.
"""
import numpy as np
import pandas as pd

from src.config import FEATURES, RISK_WEIGHTS
from src.predict import build_input_frame, predict_probabilities

FEATURE_LABELS = {
    "age": "Age", "gender": "Gender", "education_years": "Education (years)",
    "income_level": "Income level", "physical_activity": "Physical activity",
    "smoking_status": "Smoking status", "chronic_conditions": "Chronic conditions",
    "memory_score": "Memory score", "bmi": "BMI", "blood_pressure": "Blood pressure",
    "diet_quality": "Diet quality", "alcohol_consumption": "Alcohol consumption",
    "social_engagement": "Social engagement",
}

# Factors a person can realistically change, and the "best" value of each.
MODIFIABLE = {
    "physical_activity": "High",
    "diet_quality": "Good",
    "smoking_status": "Never",
    "alcohol_consumption": "None",
    "social_engagement": "High",
}


def _index(probs: pd.DataFrame) -> pd.Series:
    """Risk index (0-100) for every row of a probability table."""
    weights = pd.Series({c: RISK_WEIGHTS[c] for c in probs.columns})
    return (probs * weights).sum(axis=1) * 100


def _score(probs: pd.DataFrame) -> pd.Series:
    """Risk index on the log-odds scale.

    Probabilities saturate near 0 or 1 for very typical profiles, which would make every
    change look like 'no effect'. The log-odds scale keeps small differences visible.
    """
    p = (_index(probs) / 100).clip(1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def risk_drivers(model, inputs: dict, reference: dict) -> pd.DataFrame:
    """How much does each input move the risk index compared with a 'typical' person?

    For every feature we swap the person's value for the typical value (median / most common
    in the training data) and see how the risk index changes. Positive = this value RAISES
    the model's risk estimate; negative = it LOWERS it.
    """
    base = build_input_frame(inputs)
    frames = [base]
    for feature in FEATURES:
        altered = base.copy()
        altered[feature] = reference[feature]
        frames.append(altered)
    score = _score(predict_probabilities(model, pd.concat(frames, ignore_index=True)))
    effects = score.iloc[0] - score.iloc[1:].to_numpy()
    out = pd.DataFrame({
        "feature": FEATURES,
        "label": [FEATURE_LABELS[f] for f in FEATURES],
        "your_value": [inputs[f] for f in FEATURES],
        "typical_value": [reference[f] for f in FEATURES],
        "effect": effects,
    })
    out["abs_effect"] = out["effect"].abs()
    total = out["abs_effect"].sum()
    out["share"] = 100 * out["effect"] / total if total else 0.0   # signed % of total influence
    return out.sort_values("abs_effect", ascending=False).reset_index(drop=True)


def improvement_opportunities(model, inputs: dict) -> pd.DataFrame:
    """For each changeable lifestyle factor: how much would the risk index drop if it were at
    its best level? (One factor at a time, everything else unchanged.)"""
    base = build_input_frame(inputs)
    rows, frames = [], [base]
    for feature, best in MODIFIABLE.items():
        if inputs[feature] == best:
            continue
        altered = base.copy()
        altered[feature] = best
        frames.append(altered)
        rows.append((feature, inputs[feature], best))
    if not rows:
        return pd.DataFrame(columns=["factor", "current", "target", "risk_change", "score_change"])
    probs = predict_probabilities(model, pd.concat(frames, ignore_index=True))
    index, score = _index(probs), _score(probs)
    return (pd.DataFrame({"factor": [FEATURE_LABELS[r[0]] for r in rows],
                          "current": [r[1] for r in rows], "target": [r[2] for r in rows],
                          "risk_change": index.iloc[1:].to_numpy() - index.iloc[0],
                          "score_change": score.iloc[1:].to_numpy() - score.iloc[0]})
            .sort_values("score_change").reset_index(drop=True))


def what_if(model, inputs: dict, changes: dict) -> dict:
    """Compare the original profile with a modified one. Returns both probability dicts."""
    modified = dict(inputs, **changes)
    both = pd.concat([build_input_frame(inputs), build_input_frame(modified)], ignore_index=True)
    probs = predict_probabilities(model, both)
    index = _index(probs)
    return {"before": probs.iloc[0].to_dict(), "after": probs.iloc[1].to_dict(),
            "risk_before": float(index.iloc[0]), "risk_after": float(index.iloc[1]),
            "label_before": probs.iloc[0].idxmax(), "label_after": probs.iloc[1].idxmax()}


def lifestyle_scores(inputs: dict) -> dict:
    """Simple 0-100 scores for a radar chart (higher = healthier habit)."""
    level = {"Low": 0, "Moderate": 50, "High": 100}
    diet = {"Poor": 0, "Average": 50, "Good": 100}
    smoke = {"Current": 0, "Former": 60, "Never": 100}
    alcohol = {"Heavy": 0, "Moderate": 60, "None": 100}
    return {
        "Activity": level[inputs["physical_activity"]],
        "Diet": diet[inputs["diet_quality"]],
        "Social life": level[inputs["social_engagement"]],
        "Smoke-free": smoke[inputs["smoking_status"]],
        "Alcohol": alcohol[inputs["alcohol_consumption"]],
        "Memory": max(0.0, min(100.0, float(inputs["memory_score"]))),
    }


def wellness_tips(inputs: dict) -> list:
    """General, non-medical wellness suggestions based on the entered habits.
    Returns a list of {"area", "tip", "priority"} (priority: High / Medium / Good)."""
    tips = []

    def add(area, tip, priority):
        tips.append({"area": area, "tip": tip, "priority": priority})

    act = inputs["physical_activity"]
    if act == "Low":
        add("Physical activity", "Start with short daily walks (10-15 minutes) and build up gradually. "
            "Regular movement is widely recommended for healthy ageing.", "High")
    elif act == "Moderate":
        add("Physical activity", "Good base. Adding balance or strength exercises a couple of times a "
            "week is a common next step.", "Medium")
    else:
        add("Physical activity", "Great, keep your activity level up.", "Good")

    diet = inputs["diet_quality"]
    if diet == "Poor":
        add("Diet", "Try adding more vegetables, fruit, whole grains and pulses, and cut back on "
            "highly processed food.", "High")
    elif diet == "Average":
        add("Diet", "Small upgrades help: more fresh food, less sugar and salt.", "Medium")
    else:
        add("Diet", "Your diet quality looks good. Keep it balanced.", "Good")

    if inputs["smoking_status"] == "Current":
        add("Smoking", "Quitting is one of the most valuable health steps. Local quit-lines and "
            "your doctor can help.", "High")
    elif inputs["smoking_status"] == "Former":
        add("Smoking", "Well done for quitting. Stay smoke-free.", "Good")

    alc = inputs["alcohol_consumption"]
    if alc == "Heavy":
        add("Alcohol", "Consider reducing alcohol and speaking with a healthcare professional about "
            "safe limits.", "High")
    elif alc == "Moderate":
        add("Alcohol", "Keep within recommended limits; alcohol-free days are a good habit.", "Medium")

    soc = inputs["social_engagement"]
    if soc == "Low":
        add("Social life", "Regular contact with friends, family or community groups supports mood "
            "and mental sharpness.", "High")
    elif soc == "Moderate":
        add("Social life", "Try to add one more regular social activity or club.", "Medium")
    else:
        add("Social life", "Strong social connection. Keep it going.", "Good")

    if inputs["chronic_conditions"] >= 3:
        add("Health conditions", "With several chronic conditions, regular check-ups and following "
            "your doctor's treatment plan matter most.", "High")
    bmi = float(inputs["bmi"])
    if bmi < 18.5:
        add("Weight", "BMI is below the usual healthy range; a dietitian or doctor can advise.", "High")
    elif bmi >= 30:
        add("Weight", "BMI is in the obese range; gradual changes in food and activity help. "
            "Ask a professional for a plan.", "High")
    elif bmi >= 25:
        add("Weight", "BMI is slightly above the usual healthy range; small, steady changes help.",
            "Medium")
    if float(inputs["memory_score"]) < 65:
        add("Memory", "Mentally stimulating activities (reading, puzzles, learning) and good sleep "
            "may help; raise memory concerns with a doctor.", "High")
    if float(inputs["blood_pressure"]) >= 140:
        add("Blood pressure", "The value entered is on the high side; have your blood pressure "
            "checked by a professional.", "Medium")
    order = {"High": 0, "Medium": 1, "Good": 2}
    return sorted(tips, key=lambda t: order[t["priority"]])
