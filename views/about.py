import streamlit as st

from ui.theme import card, disclaimer, page_head

page_head("About SwarSaathi")
st.write("SwarSaathi (\"companion for ageing\") is an MCA machine-learning project that predicts whether a "
         "profile looks **Healthy**, **At-risk** or **Frail**, and explains the result in plain language.")
c1, c2 = st.columns(2)
with c1:
    card("Technology", "Python, pandas, scikit-learn (pipelines, SVM, logistic regression, gradient "
         "boosting, soft-voting ensemble), Streamlit, Plotly, ReportLab, pytest.", "🛠️")
with c2:
    card("ML workflow", "Audit → de-duplicate → stratified split → repeated cross-validation → grid-search "
         "tuning → selection by CV macro-F1 → out-of-fold evaluation → saved pipeline.", "🔁")
st.subheader("Limitations (please read)")
st.markdown("""
- The dataset's origin is **unverified**; it must not be described as real clinical data.
- After removing duplicates only **22 Healthy** records remain, so Healthy-class results are unreliable.
- Labels look rule-generated (very narrow ranges for Healthy), so high scores may reflect those rules rather than real medicine.
- Explanations show **associations in this dataset**, not causes. The what-if tool simulates the model, not a real person.
- Not a medical device. No diagnosis. Consult a healthcare professional.
""")
st.subheader("Team")
st.write(
    """
    - MC2520 Bhakti Gharat
    - MC2546 Payal Pawar
    - MC2549 Diti Prajapati
    """
)

st.subheader("Guide")
st.write("- Ms.Shaesta Mujawar")
disclaimer()
