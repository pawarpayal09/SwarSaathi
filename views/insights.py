import pandas as pd
import streamlit as st

from src.config import OOF_CM_PATH, REPORTS_DIR
from ui import charts
from ui.common import get_summary, plot, table
from ui.theme import page_head

page_head("Model insights", "How the model was built, compared and evaluated.")
summary = get_summary()
if summary is None:
    st.error("No evaluation results found. Run training first:  `python -m src.train_models`")
    st.stop()

t = summary["test_metrics"]
oof = summary.get("out_of_fold_metrics_all_unique_rows", {})
st.subheader(f"Selected model: {summary['final_model']}")
st.caption(f"Selection rule: {summary['selection_metric']}. The test set was NOT used to choose the model.  "
           f"CV scheme: {summary.get('cv_scheme', '5-fold')}.")
c = st.columns(4)
c[0].metric("Test accuracy", f"{t['test_accuracy']:.3f}")
c[1].metric("Test macro-F1", f"{t['test_macro_f1']:.3f}")
c[2].metric("Test balanced acc.", f"{t['test_balanced_accuracy']:.3f}")
c[3].metric("CV macro-F1 (train)", f"{summary['selected_cv_macro_f1']:.3f}")
if oof:
    c = st.columns(4)
    c[0].metric("Out-of-fold accuracy", f"{oof['accuracy']:.3f}")
    c[1].metric("Out-of-fold macro-F1", f"{oof['macro_f1']:.3f}")
    c[2].metric("Out-of-fold balanced acc.", f"{oof['balanced_accuracy']:.3f}")
    c[3].metric("Records evaluated", f"{summary['data_audit']['unique_rows']:,}")
    st.caption("Out-of-fold = every unique record was predicted by a model that never saw it "
               "(5-fold). This scores all 22 Healthy cases instead of only 5.")

tabs = st.tabs(["Comparison", "Per-class results", "Confusion matrices", "Feature importance", "Data quality"])
with tabs[0]:
    comp = REPORTS_DIR / "model_comparison.csv"
    if comp.exists():
        df = pd.read_csv(comp)
        plot(charts.model_comparison(df), key="mc")
        show = df.drop(columns=["best_params"], errors="ignore").round(3)
        table(show)
    st.markdown("**Why these models?**")
    for name, why in summary["model_reasons"].items():
        st.markdown(f"- **{name}**: {why}")
with tabs[1]:
    st.write("**Held-out test set**")
    table(pd.DataFrame(summary["per_class_test_metrics"]).round(3))
    if summary.get("out_of_fold_per_class"):
        st.write("**Out-of-fold, all unique records**")
        table(pd.DataFrame(summary["out_of_fold_per_class"]).round(3))
    st.warning("The Healthy class has very few independent records; its scores are far less reliable "
               "than the other two classes.")
with tabs[2]:
    c1, c2 = st.columns(2)
    if (REPORTS_DIR / "confusion_matrix.png").exists():
        c1.image(str(REPORTS_DIR / "confusion_matrix.png"), caption="Held-out test set")
    if OOF_CM_PATH.exists():
        c2.image(str(OOF_CM_PATH), caption="Out-of-fold, all unique records")
with tabs[3]:
    fi = REPORTS_DIR / "feature_importance.csv"
    if fi.exists():
        plot(charts.importance_chart(pd.read_csv(fi)), key="fi")
    st.caption("Shows what the model relies on, not what causes ageing.")
with tabs[4]:
    a = summary["data_audit"]
    c = st.columns(4)
    c[0].metric("Raw rows", f"{a['rows']:,}")
    c[1].metric("Exact duplicates", f"{a['exact_duplicate_rows']:,}")
    c[2].metric("Unique rows", f"{a['unique_rows']:,}")
    c[3].metric("Unique Healthy rows", a["target_distribution_after_dedup"].get("Healthy", 0))
    img = REPORTS_DIR / "target_distribution.png"
    if img.exists():
        st.image(str(img))
    st.markdown("- Duplicates are removed **before** splitting, otherwise copies in train and test inflate scores.\n"
                "- The original file looks artificially balanced; the dataset's origin is unverified.\n"
                "- The meaning/unit of `blood_pressure` and `memory_score` is not documented.\n"
                "- `None` in alcohol consumption is a real category (no alcohol), not a missing value.")
