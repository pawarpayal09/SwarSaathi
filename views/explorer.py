import streamlit as st

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.data_cleaning import prepare_dataset
from src.data_loader import load_raw_data
from ui import charts
from ui.common import plot, table
from ui.theme import page_head


@st.cache_data
def load_data():
    raw = load_raw_data()
    unique, _ = prepare_dataset(raw)
    return raw, unique


page_head("Data explorer", "Explore the dataset behind the model.")
try:
    raw, unique = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

c1, c2, c3 = st.columns([1.2, 1.4, 1.4])
version = c1.radio("Dataset version", ["Unique records (used for training)", "Original file (with duplicates)"])
df = unique if version.startswith("Unique") else raw
classes = c2.multiselect("Outcome classes", ["Healthy", "At-risk", "Frail"], default=["Healthy", "At-risk", "Frail"])
age_lo, age_hi = c3.slider("Age range", int(df.age.min()), int(df.age.max()),
                           (int(df.age.min()), int(df.age.max())))
view = df[df[TARGET].isin(classes) & df.age.between(age_lo, age_hi)]
st.caption(f"Showing {len(view):,} records.")
if version.startswith("Original"):
    st.warning("The original file repeats the same records many times (7,488 rows but only 2,439 unique). "
               "Counts here are inflated.")

if view.empty:
    st.info("No records match the filters.")
    st.stop()

t1, t2, t3, t4 = st.tabs(["Numeric features", "Categorical features", "Correlation", "Data table"])
with t1:
    col = st.selectbox("Feature", NUMERIC_FEATURES)
    plot(charts.histogram(view, col, TARGET), key="hist")
    st.dataframe(view.groupby(TARGET)[col].describe().round(2))
with t2:
    col = st.selectbox("Feature", CATEGORICAL_FEATURES)
    plot(charts.stacked_categories(view, col, TARGET), key="cat")
with t3:
    plot(charts.correlation_heatmap(view, NUMERIC_FEATURES), key="corr")
with t4:
    table(view)
    st.download_button("Download filtered data (CSV)", view.to_csv(index=False), "filtered_data.csv", "text/csv")
