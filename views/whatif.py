import streamlit as st

from src.explain import MODIFIABLE, what_if
from src.predict import fmt_pct
from ui import charts
from ui.common import plot, require_model
from ui.theme import disclaimer, page_head

model, meta = require_model()
cats = meta["categories"]
page_head("What-if simulator", "Change habits in your last assessed profile and compare the model's estimate.")

last = st.session_state.get("last")
if not last:
    st.info("Run an assessment first so there is a profile to modify.")
    st.page_link("views/assess.py", label="Go to Individual assessment", icon="🩺")
    st.stop()

base = last["inputs"]
st.caption(f"Baseline: {base['age']}-year-old {base['gender'].lower()}, predicted **{last['result']['label']}**.")
c1, c2, c3 = st.columns(3)
changes = {}
labels = {"physical_activity": "Physical activity", "diet_quality": "Diet quality",
          "smoking_status": "Smoking status", "alcohol_consumption": "Alcohol consumption",
          "social_engagement": "Social engagement"}
for col, feature in zip([c1, c2, c3, c1, c2], MODIFIABLE):
    opts = cats[feature]
    changes[feature] = col.selectbox(labels[feature], opts, index=opts.index(base[feature]),
                                     key=f"wi_{feature}")
lo, hi = meta["observed_ranges"]["bmi"]
changes["bmi"] = c3.slider("BMI", float(max(lo, 10)), float(hi), float(base["bmi"]), 0.1, key="wi_bmi")

result = what_if(model, base, changes)
delta = result["risk_after"] - result["risk_before"]
st.divider()
a, b, c = st.columns(3)
a.metric("Category", result["label_after"], delta=None if result["label_after"] == result["label_before"]
         else f"was {result['label_before']}", delta_color="off")
b.metric("Risk index", f"{result['risk_after']:.1f}", f"{delta:+.1f} vs current", delta_color="inverse")
c.metric("P(Frail)", fmt_pct(result["after"].get("Frail", 0)),
         f"was {fmt_pct(result['before'].get('Frail', 0))}", delta_color="off")
plot(charts.before_after(result["before"], result["after"]), key="ba")
if abs(delta) < 0.5:
    st.info("The model's probabilities barely move. For very typical profiles the model is already very "
            "sure, so single habit changes may not shift the category. This is a limit of the data, "
            "not proof that habits do not matter.")
st.caption("Simulation of the model's behaviour only; it does not predict what would happen to a real person.")
disclaimer()
