"""Plotly charts used by the app."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import CLASS_COLORS, CLASS_ORDER
from src.predict import fmt_pct


def risk_gauge(value: float, title: str = "Model risk index"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": "/100", "font": {"size": 36}},
        title={"text": title, "font": {"size": 15}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0f172a", "thickness": 0.22},
               "steps": [{"range": [0, 33], "color": "#bbf7d0"},
                         {"range": [33, 66], "color": "#fde68a"},
                         {"range": [66, 100], "color": "#fecaca"}]}))
    fig.update_layout(height=250)
    return fig


def probability_bars(probabilities: dict):
    labels = [c for c in CLASS_ORDER if c in probabilities]
    vals = [probabilities[c] * 100 for c in labels]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=[CLASS_COLORS[c] for c in labels],
        text=[fmt_pct(probabilities[c]) for c in labels], textposition="auto",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    fig.update_layout(height=230, xaxis=dict(range=[0, 100], title="Probability (%)"),
                      yaxis=dict(autorange="reversed"), title="Class probabilities")
    return fig


def drivers_chart(drivers: pd.DataFrame, top: int = 8):
    data = drivers.head(top).iloc[::-1]
    colours = ["#dc2626" if e > 0 else "#16a34a" for e in data["effect"]]
    text = [f"{r.your_value}  (typical: {r.typical_value})" if isinstance(r.your_value, str)
            else f"{r.your_value:g}  (typical: {float(r.typical_value):g})" for r in data.itertuples()]
    fig = go.Figure(go.Bar(x=data["share"], y=data["label"], orientation="h", marker_color=colours,
                           text=text, textposition="auto", hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(height=60 + 42 * len(data), title="Influence on the model's risk estimate",
                      xaxis_title="← lowers risk   |   raises risk →  (% of total influence)")
    return fig


def opportunities_chart(opps: pd.DataFrame):
    data = opps.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=-data["score_change"], y=data["factor"], orientation="h", marker_color="#0f766e",
        text=[f"{r.current} → {r.target}" for r in data.itertuples()], textposition="auto"))
    fig.update_layout(height=70 + 50 * len(data), title="Biggest opportunities (one change at a time)",
                      xaxis_title="Reduction in the model's risk score (log-odds units)")
    return fig


def radar(scores: dict):
    names, vals = list(scores), list(scores.values())
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=names + names[:1], fill="toself",
                                    line_color="#0f766e", fillcolor="rgba(15,118,110,0.25)"))
    fig.update_layout(height=320, polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)),
                      title="Lifestyle profile (higher = healthier habit)", showlegend=False)
    return fig


def before_after(before: dict, after: dict):
    labels = [c for c in CLASS_ORDER if c in before]
    fig = go.Figure()
    fig.add_bar(name="Current", x=labels, y=[before[c] * 100 for c in labels], marker_color="#94a3b8")
    fig.add_bar(name="After changes", x=labels, y=[after[c] * 100 for c in labels], marker_color="#0f766e")
    fig.update_layout(barmode="group", height=320, yaxis_title="Probability (%)",
                      title="Class probabilities: current vs. changed")
    return fig


def prediction_counts(series: pd.Series):
    counts = series.value_counts().reindex(CLASS_ORDER).fillna(0).reset_index()
    counts.columns = ["Category", "People"]
    fig = px.bar(counts, x="Category", y="People", color="Category", color_discrete_map=CLASS_COLORS,
                 text="People")
    fig.update_layout(height=300, showlegend=False, title="Predictions in this file")
    return fig


def model_comparison(table: pd.DataFrame):
    data = table[table["stage"] != "reference"].copy()
    data["name"] = data["model"] + " (" + data["stage"] + ")"
    data = data.sort_values("cv_macro_f1")
    fig = go.Figure()
    fig.add_bar(name="CV macro-F1 (training data)", y=data["name"], x=data["cv_macro_f1"],
                orientation="h", marker_color="#0f766e")
    fig.add_bar(name="Test macro-F1", y=data["name"], x=data["test_macro_f1"],
                orientation="h", marker_color="#38bdf8")
    fig.update_layout(barmode="group", height=120 + 38 * len(data), xaxis=dict(range=[0, 1]),
                      title="Model comparison", legend=dict(orientation="h", y=-0.12))
    return fig


def importance_chart(importance: pd.DataFrame):
    data = importance.sort_values("importance_mean")
    fig = go.Figure(go.Bar(x=data["importance_mean"], y=data["feature"], orientation="h",
                           error_x=dict(type="data", array=data["importance_std"]), marker_color="#0e7490"))
    fig.update_layout(height=120 + 30 * len(data), title="Permutation feature importance (test set)",
                      xaxis_title="Drop in macro-F1 when the feature is shuffled")
    return fig


def histogram(df: pd.DataFrame, column: str, target: str):
    fig = px.histogram(df, x=column, color=target, barmode="overlay", opacity=0.65, nbins=30,
                       color_discrete_map=CLASS_COLORS, category_orders={target: CLASS_ORDER})
    fig.update_layout(height=360, title=f"Distribution of {column} by outcome")
    return fig


def stacked_categories(df: pd.DataFrame, column: str, target: str):
    ct = pd.crosstab(df[column], df[target], normalize="index").mul(100)
    ct = ct.reindex(columns=[c for c in CLASS_ORDER if c in ct.columns]).reset_index()
    fig = px.bar(ct, x=column, y=[c for c in CLASS_ORDER if c in ct.columns],
                 color_discrete_map=CLASS_COLORS)
    fig.update_layout(height=360, yaxis_title="% of people in group", legend_title="Outcome",
                      title=f"Outcome mix within each '{column}' group")
    return fig


def correlation_heatmap(df: pd.DataFrame, columns: list):
    corr = df[columns].corr()
    fig = go.Figure(go.Heatmap(z=corr.values, x=columns, y=columns, zmin=-1, zmax=1,
                               colorscale="RdBu_r", text=np.round(corr.values, 2),
                               texttemplate="%{text}"))
    fig.update_layout(height=470, title="Correlation between numeric features")
    return fig
