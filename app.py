"""SwarSaathi - Aging Risk Assessment (Streamlit app).
Run from the project root:   streamlit run app.py
The app only LOADS the trained model; it never trains.
"""
import streamlit as st

st.set_page_config(page_title="SwarSaathi - Aging Risk Assessment", page_icon="🧓", layout="wide")

from ui.theme import inject_css, sidebar_brand  # noqa: E402

inject_css()
sidebar_brand()
pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/assess.py", title="Assessment", icon="🩺"),
    st.Page("views/whatif.py", title="What-if Simulator", icon="🔬"),
    st.Page("views/insights.py", title="Model Insights", icon="📊"),
    st.Page("views/explorer.py", title="Data Explorer", icon="🔎"),
    st.Page("views/about.py", title="About", icon="ℹ️"),
]
st.navigation(pages).run()
st.sidebar.caption("Educational prototype. Not medical advice.")
