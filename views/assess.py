from datetime import datetime

import pandas as pd
import streamlit as st

from src.config import VALID_RANGES
from src.explain import (MODIFIABLE, improvement_opportunities, lifestyle_scores, risk_drivers,
                         wellness_tips)
from src.predict import fmt_pct, predict_outcome
from src.report import build_pdf
from ui import charts
from ui.common import get_summary, plot, require_model, table
from ui.theme import disclaimer, page_head, result_card, tip_card

model, meta = require_model()
cats = meta["categories"]

EXAMPLES = {
    "Custom (enter your own)": None,
    "Active adult, healthy habits": dict(age=45, gender="Female", education_years=16, income_level="High",
        physical_activity="High", smoking_status="Never", chronic_conditions=0, memory_score=92.0, bmi=22.5,
        blood_pressure=118, diet_quality="Good", alcohol_consumption="None", social_engagement="High"),
    "Middle-aged, some risk factors": dict(age=55, gender="Female", education_years=12, income_level="Medium",
        physical_activity="Moderate", smoking_status="Never", chronic_conditions=1, memory_score=85.0,
        bmi=24.0, blood_pressure=125, diet_quality="Good", alcohol_consumption="None",
        social_engagement="Moderate"),
    "Older adult, several conditions": dict(age=68, gender="Male", education_years=10, income_level="Medium",
        physical_activity="Moderate", smoking_status="Former", chronic_conditions=2, memory_score=76.0,
        bmi=26.0, blood_pressure=140, diet_quality="Average", alcohol_consumption="Moderate",
        social_engagement="Moderate"),
    "Higher-risk profile": dict(age=78, gender="Female", education_years=5, income_level="Low",
        physical_activity="Low", smoking_status="Current", chronic_conditions=4, memory_score=58.0, bmi=29.5,
        blood_pressure=165, diet_quality="Poor", alcohol_consumption="Heavy", social_engagement="Low"),
}
DEFAULTS = EXAMPLES["Middle-aged, some risk factors"]


def bounds(feature, integer):
    seen_lo, seen_hi = meta["observed_ranges"][feature]
    hard_lo, hard_hi = VALID_RANGES[feature]
    lo, hi = max(seen_lo, hard_lo), min(seen_hi, hard_hi)
    return (int(lo), int(hi)) if integer else (round(float(lo), 1), round(float(hi), 1))


for k, v in DEFAULTS.items():
    st.session_state.setdefault(f"in_{k}", v)
st.session_state.setdefault("history", [])


def load_example():
    ex = EXAMPLES[st.session_state["example_choice"]]
    if ex:
        for k, v in ex.items():
            st.session_state[f"in_{k}"] = v


def reset_form():
    for k, v in DEFAULTS.items():
        st.session_state[f"in_{k}"] = v
    st.session_state["example_choice"] = "Custom (enter your own)"


page_head("Individual assessment", "Fill in all 13 inputs below, then run the assessment. "
          "Every input is combined into one single prediction.")
st.selectbox("Quick start", list(EXAMPLES), key="example_choice", on_change=load_example,
             help="Examples are made-up illustrations, not real people.")

with st.container(border=True):
    col_p, col_l, col_h = st.columns(3, gap="large")
    with col_p:
        st.markdown("##### 👤 Personal")
        lo, hi = bounds("age", True)
        st.slider("Age (years)", lo, hi, key="in_age")
        st.selectbox("Gender", cats["gender"], key="in_gender")
        lo, hi = bounds("education_years", True)
        st.slider("Education (years)", lo, hi, key="in_education_years")
        st.selectbox("Income level", cats["income_level"], key="in_income_level")
    with col_l:
        st.markdown("##### 🏃 Lifestyle")
        st.selectbox("Physical activity", cats["physical_activity"], key="in_physical_activity")
        st.selectbox("Smoking status", cats["smoking_status"], key="in_smoking_status")
        st.selectbox("Diet quality", cats["diet_quality"], key="in_diet_quality")
        st.selectbox("Alcohol consumption", cats["alcohol_consumption"], key="in_alcohol_consumption",
                     help="'None' means no alcohol.")
        st.selectbox("Social engagement", cats["social_engagement"], key="in_social_engagement")
    with col_h:
        st.markdown("##### ❤️ Health measures")
        lo, hi = bounds("chronic_conditions", True)
        st.number_input("Number of chronic conditions", lo, hi, step=1, key="in_chronic_conditions")
        lo, hi = bounds("memory_score", False)
        st.number_input("Memory score", lo, hi, step=0.5, key="in_memory_score")
        lo, hi = bounds("bmi", False)
        st.number_input("BMI", lo, hi, step=0.1, key="in_bmi")
        lo, hi = bounds("blood_pressure", True)
        st.number_input("Blood pressure (as stored in dataset)", lo, hi, step=1, key="in_blood_pressure",
                        help="The dataset does not state which reading or unit this is.")
    b1, b2, _ = st.columns([1, 1, 3])
    run_clicked = b1.button("Run assessment", type="primary")
    b2.button("Reset form", on_click=reset_form)

if run_clicked:
    inputs = {f: st.session_state[f"in_{f}"] for f in meta["feature_order"]}
    for f in ("age", "education_years", "chronic_conditions", "blood_pressure"):
        inputs[f] = int(inputs[f])
    for f in ("memory_score", "bmi"):
        inputs[f] = float(inputs[f])
    try:
        with st.spinner("Analysing profile..."):
            result = predict_outcome(inputs, model=model, metadata=meta)
            drivers = risk_drivers(model, inputs, meta["reference_profile"])
            opps = improvement_opportunities(model, inputs)
        st.session_state["last"] = dict(inputs=inputs, result=result, drivers=drivers, opps=opps,
                                        tips=wellness_tips(inputs))
        st.session_state["history"].append({
            "Time": datetime.now().strftime("%H:%M:%S"), "Age": inputs["age"], "Gender": inputs["gender"],
            "Prediction": result["label"], "Risk index": round(result["risk_index"], 1),
            "Confidence": fmt_pct(result["confidence"])})
    except ValueError as error:
        st.error(f"Please correct the input: {error}")
    except Exception as error:
        st.error(f"Something went wrong while predicting: {error}")

last = st.session_state.get("last")
if last:
    res, inputs = last["result"], last["inputs"]
    st.divider()
    for message in res["warnings"]:
        st.warning(message)
    left, mid, right = st.columns([1.1, 1, 1.2])
    with left:
        result_card(res["label"], f"Confidence {fmt_pct(res['confidence'])}")
        st.caption(res["confidence_note"])
        if res["label"] == "Frail":
            st.caption("Note: about 64% of the training records are Frail, so many realistic profiles "
                       "(older age, several conditions, weaker habits) are classed Frail. Try the "
                       "'Quick start' examples to see the other categories.")
    with mid:
        plot(charts.risk_gauge(res["risk_index"]), key="gauge")
    with right:
        plot(charts.probability_bars(res["probabilities"]), key="probs")
    st.caption("The risk index = 0 × P(Healthy) + 50 × P(At-risk) + 100 × P(Frail). It summarises the "
               "model's three probabilities; it is not a clinical score.")

    t1, t2, t3 = st.tabs(["🔍 Why this result", "🌱 How to improve", "📄 Report"])
    with t1:
        st.write("Each input was swapped for the *typical* value in the dataset to see how much it moves "
                 "the model's estimate. Red raises the estimated risk, green lowers it.")
        plot(charts.drivers_chart(last["drivers"]), key="drivers")
        st.caption("These are associations learned from the dataset, not medical causes.")
    with t2:
        c1, c2 = st.columns([1, 1])
        with c1:
            if len(last["opps"]):
                plot(charts.opportunities_chart(last["opps"]), key="opps")
                st.caption("Each bar shows how much the model's risk score drops if that single habit "
                           "were at its best level. Try combinations in the What-if page.")
            else:
                st.success("All changeable lifestyle factors are already at their best level.")
            st.page_link("views/whatif.py", label="Open the What-if simulator", icon="🔬")
        with c2:
            plot(charts.radar(lifestyle_scores(inputs)), key="radar")
        st.subheader("General wellness suggestions")
        for t in last["tips"]:
            tip_card(t["area"], t["tip"], t["priority"])
    with t3:
        st.write("Download a one-page summary of this assessment.")
        pdf = build_pdf(res, inputs, last["drivers"], last["tips"], "Soft Voting Ensemble"
                        if meta.get("model_name") is None else meta["model_name"])
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download PDF report", pdf, "swarsaathi_report.pdf", "application/pdf")
        c2.download_button("⬇️ Download inputs + result (CSV)",
                           pd.DataFrame([{**inputs, "prediction": res["label"],
                                          "risk_index": round(res["risk_index"], 1)}]).to_csv(index=False),
                           "swarsaathi_result.csv", "text/csv")
    st.write("")
    disclaimer()

if st.session_state["history"]:
    with st.expander(f"Assessment history this session ({len(st.session_state['history'])})"):
        hist = pd.DataFrame(st.session_state["history"])
        table(hist)
        c1, c2 = st.columns(2)
        c1.download_button("Download history (CSV)", hist.to_csv(index=False), "history.csv", "text/csv")
        if c2.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()
