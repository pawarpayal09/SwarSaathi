"""Create a downloadable PDF summary of one assessment (uses reportlab + matplotlib)."""
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from src.config import CLASS_COLORS, CLASS_ORDER
from src.explain import FEATURE_LABELS
from src.predict import fmt_pct

DISCLAIMER = ("SwarSaathi is an educational machine-learning prototype. It is NOT a medical device "
              "and does not give a diagnosis or clinical advice. Please consult a qualified "
              "healthcare professional for any health concern.")


def _probability_chart(probabilities: dict) -> io.BytesIO:
    labels = [c for c in CLASS_ORDER if c in probabilities]
    values = [probabilities[c] * 100 for c in labels]
    fig, ax = plt.subplots(figsize=(6, 2.2))
    bars = ax.barh(labels, values, color=[CLASS_COLORS[c] for c in labels])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Probability (%)")
    ax.invert_yaxis()
    for bar, v in zip(bars, values):
        inside = v > 80
        ax.text(v - 1 if inside else v + 1, bar.get_y() + bar.get_height() / 2, fmt_pct(v / 100),
                va="center", ha="right" if inside else "left", fontsize=9,
                color="white" if inside else "black")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return buf


def build_pdf(result: dict, inputs: dict, drivers, tips: list, model_name: str) -> bytes:
    """Return the PDF as bytes (so Streamlit's download button can serve it)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="SwarSaathi Assessment Report")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=colors.HexColor("#0f766e"), fontSize=22)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=colors.HexColor("#134e4a"), spaceBefore=10)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#6b7280"))

    label = result["label"]
    label_color = colors.HexColor(CLASS_COLORS.get(label, "#334155"))
    story = [Paragraph("SwarSaathi Assessment Report", h1),
             Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M} &nbsp;|&nbsp; Model: {model_name}", small),
             Spacer(1, 10)]

    headline = Table([[Paragraph(f"<font color='{label_color.hexval().replace('0x', '#')}' size=18>"
                                 f"<b>{label}</b></font><br/><font size=9>Predicted category</font>", body),
                       Paragraph(f"<font size=18><b>{result['risk_index']:.0f}/100</b></font><br/>"
                                 "<font size=9>Model risk index</font>", body)]],
                     colWidths=[8.5 * cm, 8.5 * cm])
    headline.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
                                  ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                                  ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story += [headline, Spacer(1, 6), Paragraph(result.get("confidence_note", ""), body)]

    story += [Paragraph("Class probabilities", h2),
              Image(_probability_chart(result["probabilities"]), width=14 * cm, height=5.1 * cm)]

    rows = [["Input", "Value"]] + [[FEATURE_LABELS[k], str(v)] for k, v in inputs.items()]
    table = Table(rows, colWidths=[7 * cm, 6 * cm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("FONTSIZE", (0, 0), (-1, -1), 9),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")])]))
    story += [Paragraph("Values entered", h2), table]

    if drivers is not None and len(drivers):
        story.append(Paragraph("What influenced the model most", h2))
        story.append(Paragraph("Compared with a typical person in the dataset (positive = pushes the model "
                               "towards higher risk, negative = towards lower risk):", body))
        top = drivers.head(5)
        story.append(Paragraph("<br/>".join(
            f"&bull; <b>{r.label}</b> ({r.your_value} vs typical {r.typical_value}): "
            f"{'raises' if r.effect > 0 else 'lowers'} risk ({abs(r.share):.0f}% of total influence)"
            for r in top.itertuples()), body))

    if tips:
        story.append(Paragraph("General wellness suggestions", h2))
        story.append(Paragraph("<br/>".join(f"&bull; <b>{t['area']}</b> [{t['priority']}]: {t['tip']}"
                                            for t in tips), body))
    story += [Spacer(1, 14), Paragraph("<b>Disclaimer.</b> " + DISCLAIMER, small)]
    doc.build(story)
    return buf.getvalue()
