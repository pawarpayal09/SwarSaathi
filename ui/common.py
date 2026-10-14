"""Shared helpers for all pages: cached loading, version-safe widgets, plotting."""
import json

import pandas as pd
import streamlit as st

from src.config import SUMMARY_PATH
from src.predict import ModelNotAvailableError, load_metadata, load_model


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


@st.cache_data
def get_metadata():
    return load_metadata()


@st.cache_data
def get_summary():
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else None


def require_model():
    """Return (model, metadata) or show a friendly error and stop the page."""
    try:
        return get_model(), get_metadata()
    except ModelNotAvailableError as error:
        st.error(f"{error}  Train first:  `python -m src.train_models`")
    except Exception as error:  # corrupt file, version mismatch, ...
        st.error("The saved model could not be loaded. Please re-run training: "
                 f"`python -m src.train_models`  (details: {error})")
    st.stop()


def plot(fig, key=None):
    """Show a plotly figure full-width (works on old and new Streamlit versions)."""
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), font=dict(family="sans-serif"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df: pd.DataFrame, **kwargs):
    """Show a dataframe full-width (works on old and new Streamlit versions)."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)
