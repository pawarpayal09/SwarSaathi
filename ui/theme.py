"""Look and feel: custom CSS plus small HTML helpers (cards, hero banner, badges)."""
import html

import streamlit as st

from src.config import CLASS_COLORS

CSS = """
<style>
:root { --teal:#0f766e; --teal-dark:#115e59; --ink:#0f172a; --muted:#64748b; --card:#ffffff; --line:#e2e8f0; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.01em; }
#MainMenu, footer { visibility: hidden; }

/* sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b3b3a 0%,#115e59 100%); }
section[data-testid="stSidebar"] * { color: #e6fffb; }
section[data-testid="stSidebar"] a { border-radius: 10px; }
section[data-testid="stSidebar"] a:hover { background: rgba(255,255,255,0.10); }
section[data-testid="stSidebar"] [aria-current="page"] { background: rgba(255,255,255,0.18) !important; }
.brand { display:flex; align-items:center; gap:.6rem; padding:.2rem .4rem 1rem .4rem; }
.brand .logo { font-size:2rem; }
.brand .name { font-size:1.35rem; font-weight:800; color:#fff; line-height:1.1; }
.brand .tag { font-size:.72rem; color:#99f6e4; }

/* hero */
.hero { background: linear-gradient(120deg,#0f766e 0%,#0e7490 55%,#1d4ed8 120%); color:#fff;
        border-radius: 22px; padding: 2.2rem 2.4rem; margin-bottom: 1.4rem;
        box-shadow: 0 12px 30px rgba(15,118,110,.25); }
.hero h1 { color:#fff; font-size: 2.4rem; margin:0 0 .4rem 0; }
.hero p { color:#d1fae5; font-size:1.05rem; margin:0; max-width: 780px; line-height:1.55; }
.hero .pill { display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3);
        padding:.18rem .7rem; border-radius:999px; font-size:.78rem; margin-bottom:.8rem; }
.page-head { margin-bottom: 1rem; }
.page-head h1 { margin:0; font-size:1.9rem; }
.page-head p { color: var(--muted); margin:.2rem 0 0 0; }

/* cards */
.card { background: var(--card); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.25rem;
        box-shadow: 0 2px 10px rgba(15,23,42,.05); height:100%; }
.card h4 { margin:0 0 .35rem 0; color: var(--teal-dark); font-size:1.02rem; }
.card p { margin:0; color:#334155; font-size:.93rem; line-height:1.5; }
.stat { background: var(--card); border:1px solid var(--line); border-radius:16px; padding:1rem 1.2rem;
        box-shadow: 0 2px 10px rgba(15,23,42,.05); }
.stat .v { font-size:1.9rem; font-weight:800; color: var(--teal); line-height:1.1; }
.stat .l { font-size:.82rem; color: var(--muted); margin-top:.15rem; }
.step-num { display:inline-flex; width:2rem; height:2rem; border-radius:50%; background:var(--teal); color:#fff;
        align-items:center; justify-content:center; font-weight:700; margin-bottom:.5rem; }

/* result */
.result { border-radius:18px; padding:1.3rem 1.5rem; color:#fff; box-shadow:0 10px 24px rgba(15,23,42,.18); }
.result .k { font-size:.8rem; opacity:.9; text-transform:uppercase; letter-spacing:.08em; }
.result .v { font-size:2.5rem; font-weight:800; line-height:1.1; }
.result .s { font-size:.95rem; opacity:.95; margin-top:.3rem; }
.badge { display:inline-block; padding:.1rem .6rem; border-radius:999px; font-size:.75rem; font-weight:700; color:#fff; }
.disclaimer { background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; border-radius:12px;
        padding:.7rem 1rem; font-size:.85rem; }
.tip { border-left:5px solid var(--line); background:#fff; border-radius:10px; padding:.6rem .9rem; margin-bottom:.5rem;
       box-shadow:0 1px 6px rgba(15,23,42,.05); }
.tip b { color: var(--ink); }

/* buttons & inputs */
.stButton > button[kind="primary"], .stDownloadButton > button { border-radius: 12px; font-weight:700; }
.stButton > button, .stDownloadButton > button { border-radius: 12px; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:.7rem 1rem;
        box-shadow:0 2px 8px rgba(15,23,42,.04); }
button[data-baseweb="tab"] { font-weight:600; }
</style>
"""

PRIORITY_COLORS = {"High": "#dc2626", "Medium": "#f59e0b", "Good": "#16a34a"}


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_brand():
    st.sidebar.markdown(
        '<div class="brand"><div class="logo">🧓</div><div><div class="name">SwarSaathi</div>'
        '<div class="tag">Aging Risk Assessment</div></div></div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, pill: str = ""):
    pill_html = f'<div class="pill">{html.escape(pill)}</div>' if pill else ""
    st.markdown(f'<div class="hero">{pill_html}<h1>{html.escape(title)}</h1>'
                f'<p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def page_head(title: str, subtitle: str = ""):
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="page-head"><h1>{html.escape(title)}</h1>{sub}</div>', unsafe_allow_html=True)


def card(title: str, body: str, icon: str = ""):
    st.markdown(f'<div class="card"><h4>{icon} {html.escape(title)}</h4><p>{html.escape(body)}</p></div>',
                unsafe_allow_html=True)


def stat(value: str, label: str):
    st.markdown(f'<div class="stat"><div class="v">{html.escape(str(value))}</div>'
                f'<div class="l">{html.escape(label)}</div></div>', unsafe_allow_html=True)


def result_card(label: str, subtitle: str):
    colour = CLASS_COLORS.get(label, "#334155")
    st.markdown(f'<div class="result" style="background:linear-gradient(135deg,{colour},{colour}cc)">'
                f'<div class="k">Predicted category</div><div class="v">{html.escape(label)}</div>'
                f'<div class="s">{html.escape(subtitle)}</div></div>', unsafe_allow_html=True)


def tip_card(area: str, tip: str, priority: str):
    colour = PRIORITY_COLORS.get(priority, "#64748b")
    st.markdown(f'<div class="tip" style="border-left-color:{colour}"><span class="badge" '
                f'style="background:{colour}">{html.escape(priority)}</span> '
                f'<b>{html.escape(area)}</b><br/>{html.escape(tip)}</div>', unsafe_allow_html=True)


def disclaimer():
    st.markdown('<div class="disclaimer"><b>Not medical advice.</b> SwarSaathi is an educational '
                'machine-learning prototype trained on a dataset of unverified origin. It gives no '
                'diagnosis. Please consult a qualified healthcare professional.</div>',
                unsafe_allow_html=True)
