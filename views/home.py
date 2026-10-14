import streamlit as st

from ui.common import get_metadata, get_summary
from ui.theme import card, disclaimer, hero, stat

summary, meta = None, None
try:
    summary, meta = get_summary(), get_metadata()
except Exception:
    pass

hero("Understand healthy ageing, with machine learning",
     "SwarSaathi studies demographic, lifestyle and health inputs and estimates whether a profile looks "
     "most like Healthy, At-risk or Frail, then explains why and shows what could improve it.",
     pill="MCA Project  •  Educational ML prototype")

if summary:
    m, oof = summary["test_metrics"], summary.get("out_of_fold_metrics_all_unique_rows", {})
    cols = st.columns(4)
    with cols[0]: stat(f"{summary['data_audit']['unique_rows']:,}", "unique records after cleaning")
    with cols[1]: stat("13", "input features")
    with cols[2]: stat(f"{m['test_accuracy']*100:.1f}%", "test accuracy (held-out)")
    with cols[3]: stat(f"{oof.get('macro_f1', m['test_macro_f1']):.2f}", "out-of-fold macro-F1 (all data)")
    st.write("")

st.subheader("What you can do")
c1, c2, c3 = st.columns(3)
with c1: card("Individual assessment", "Enter a profile and get a prediction, probabilities, a risk index, "
              "the main influencing factors and a downloadable PDF report.", "🩺")
with c2: card("What-if simulator", "Change lifestyle factors like activity, diet or smoking and see how "
              "the model's estimate moves.", "🔬")
with c3: card("Batch prediction", "Upload a CSV with many people, validate it automatically and "
              "download all predictions at once.", "📂")
st.write("")
c1, c2, c3 = st.columns(3)
with c1: card("Model insights", "Compare 7 algorithms, see confusion matrices, feature importance and "
              "honest out-of-fold results.", "📊")
with c2: card("Data explorer", "Interactive charts of the dataset: distributions, group comparisons "
              "and correlations.", "🔎")
with c3: card("Transparent limitations", "Clear notes on duplicates, the tiny Healthy class and why the "
              "results are a demonstration.", "⚖️")

st.write("")
st.subheader("How it works")
steps = [("Clean", "7,488 raw rows are audited; 5,049 exact duplicates are removed before splitting so the "
          "test data is never seen in training."),
         ("Compare", "Seven algorithms are scored with repeated stratified cross-validation; the top ones "
          "are tuned with grid search."),
         ("Select", "The final model is chosen by cross-validated macro-F1 on training data only, never "
          "by test score."),
         ("Explain", "Each prediction is re-run on altered copies of the profile to show which inputs "
          "matter most.")]
cols = st.columns(4)
for i, (col, (title, text)) in enumerate(zip(cols, steps), 1):
    with col:
        st.markdown(f'<div class="card"><div class="step-num">{i}</div><h4>{title}</h4><p>{text}</p></div>',
                    unsafe_allow_html=True)
st.write("")
st.page_link("views/assess.py", label="Start an assessment", icon="🩺")
st.write("")
disclaimer()
