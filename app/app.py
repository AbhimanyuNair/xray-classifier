import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import engine

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Chest X-ray Pneumonia Triage Support",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
MODEL_DIR = ROOT / "model"
WEIGHTS = {"B": "densenet_centeronly.pt", "A": "densenet_sizebalanced.pt"}
PROB_COL = {"A": "prob_A", "B": "prob_B", "G": "prob_geometry"}
NAME = {
    "B": "B · Center-only (headline)",
    "A": "A · Full image",
    "G": "Image size only (no pixels)",
}
C = {
    "cyan": "#22d3ee", "indigo": "#818cf8", "violet": "#a78bfa", "pink": "#f472b6",
    "amber": "#fbbf24", "green": "#34d399", "red": "#fb7185", "slate": "#64748b",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
/* Safety net: keep the dark look even if the theme file is missing */
.stApp { background: #0b1220; }
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stHeader"] { background: transparent; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp span { color: #e6edf7; }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p { color: #94a3b8; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
.hero { background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 55%, #a855f7 100%); border-radius: 24px;
  padding: 36px 40px; color: #fff; box-shadow: 0 20px 60px rgba(99,102,241,.35); margin-bottom: 22px; }
.hero h1 { font-size: 2.3rem; font-weight: 800; margin: 0 0 8px 0; letter-spacing: -0.02em; line-height: 1.15; }
.hero p { opacity: .93; font-size: 1.04rem; margin: 0; max-width: 820px; }
.chip { display: inline-block; padding: 5px 13px; margin: 16px 8px 0 0; border-radius: 999px;
  background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); font-size: .78rem; font-weight: 600; }
.kpi { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09); border-radius: 18px;
  padding: 18px 20px; transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease; height: 100%; }
.kpi:hover { transform: translateY(-4px); border-color: rgba(34,211,238,.55); box-shadow: 0 12px 30px rgba(34,211,238,.12); }
.kpi .label { font-size: .74rem; text-transform: uppercase; letter-spacing: .09em; color: #94a3b8; font-weight: 700; }
.kpi .value { font-size: 2.1rem; font-weight: 800; line-height: 1.25;
  background: linear-gradient(90deg, #22d3ee, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.kpi .sub { font-size: .8rem; color: #94a3b8; }
.card { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09); border-radius: 18px;
  padding: 20px 22px; height: 100%; transition: transform .16s ease, border-color .16s ease; }
.card:hover { transform: translateY(-3px); border-color: rgba(129,140,248,.55); }
.card h4 { margin: 0 0 4px 0; font-size: 1.02rem; font-weight: 700; }
.card .big { font-size: 1.9rem; font-weight: 800; margin: 2px 0 6px 0; }
.card p, .card li { color: #cbd5e1; font-size: .92rem; line-height: 1.5; margin: 0; }
.card ul { padding-left: 1.1rem; margin: 6px 0 0 0; }
.verdict { border-radius: 20px; padding: 22px 24px; margin-bottom: 10px; }
.verdict .t { font-size: 1.35rem; font-weight: 800; margin-bottom: 4px; }
.verdict .s { color: #cbd5e1; font-size: .92rem; }
.verdict.pos { background: linear-gradient(135deg, rgba(244,63,94,.20), rgba(251,146,60,.10)); border: 1px solid rgba(251,113,133,.55); }
.verdict.neg { background: linear-gradient(135deg, rgba(52,211,153,.18), rgba(34,211,238,.08)); border: 1px solid rgba(52,211,153,.5); }
.pill { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .76rem; font-weight: 700;
  background: rgba(129,140,248,.18); border: 1px solid rgba(129,140,248,.45); color: #c7d2fe; margin-right: 6px; }
.disclaimer { border-left: 4px solid #fbbf24; background: rgba(251,191,36,.08); border-radius: 10px;
  padding: 12px 16px; color: #fde68a; font-size: .88rem; line-height: 1.45; }
.sidebar-title { font-size: 1.25rem; font-weight: 800; margin-bottom: 2px; }
a.glink { display: inline-block; margin-top: 8px; padding: 8px 14px; border-radius: 12px; font-weight: 700; font-size: .85rem;
  color: #0b1220 !important; background: linear-gradient(90deg, #22d3ee, #818cf8); text-decoration: none !important; }
.stButton > button { border-radius: 12px; border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.04);
  font-weight: 600; transition: all .15s ease; }
.stButton > button:hover { border-color: #22d3ee; color: #22d3ee; transform: translateY(-2px); }
[data-baseweb="tab-list"] { gap: 8px; }
[data-baseweb="tab"] { padding: 10px 18px; border-radius: 12px; font-weight: 600; }
table.tbl { width: 100%; border-collapse: collapse; font-size: .88rem; }
table.tbl th { text-align: left; color: #94a3b8; font-weight: 700; font-size: .74rem; text-transform: uppercase;
  letter-spacing: .06em; padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,.12); }
table.tbl td { padding: 11px 12px; border-bottom: 1px solid rgba(255,255,255,.06); color: #e2e8f0; }
table.tbl tr:hover td { background: rgba(255,255,255,.03); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_predictions():
    return pd.read_csv(ASSETS / "test_predictions.csv")


@st.cache_data
def load_metrics():
    return pd.read_csv(ASSETS / "test_metrics.csv").set_index("model")


@st.cache_data
def load_samples():
    path = ASSETS / "samples.json"
    return json.loads(path.read_text()) if path.exists() else []


@st.cache_data
def load_config():
    path = ASSETS / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_resource
def get_model(key):
    path = MODEL_DIR / WEIGHTS[key]
    if not path.exists() or not engine.torch_available():
        return None
    try:
        return engine.load_model(path)
    except Exception:
        return None


@st.cache_data(show_spinner="Running the model...")
def run_inference(key, img_bytes):
    model = get_model(key)
    if model is None:
        return None
    img = Image.open(io.BytesIO(img_bytes))
    gray = engine.to_gray256(img)
    x = engine.preprocess(gray, center_only=(key == "B"))
    prob, cam = engine.predict_with_cam(model, x)
    return prob, cam


preds = load_predictions()
metrics = load_metrics()
samples = load_samples()
cfg = load_config()
DEFAULT_THR = {"A": float(metrics.loc["A", "threshold"]), "B": float(metrics.loc["B", "threshold"]), "G": 0.5}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def split_ci(text):
    text = str(text)
    if "[" in text:
        main, ci = text.split("[", 1)
        return main.strip(), "95% CI [" + ci.strip()
    return text, ""


def table_html(df):
    html = df.to_html(index=False, classes="tbl", border=0, escape=True)
    return "".join(line.strip() for line in html.splitlines())


def kpi(label, value, sub=""):
    return ("<div class='kpi'><div class='label'>" + label + "</div><div class='value'>" + value
            + "</div><div class='sub'>" + sub + "</div></div>")


def card(title, big, text, accent):
    return ("<div class='card' style='border-top:3px solid " + accent + "'><h4>" + title + "</h4><div class='big' style='color:"
            + accent + "'>" + big + "</div><p>" + text + "</p></div>")


def style(fig, height=340, legend=True):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=10, r=10, t=44, b=10),
        font=dict(family="Inter, sans-serif", size=13),
        legend=dict(orientation="h", y=-0.18), showlegend=legend,
    )
    return fig


def safe_div(a, b):
    return a / b if b else float("nan")


def counts(df, col, thr, weighted):
    y = df["target"].to_numpy()
    p = df[col].to_numpy()
    w = df["w"].to_numpy() if weighted else np.ones(len(df))
    keep = w > 0
    y, p, w = y[keep], p[keep], w[keep]
    pred = p >= thr
    tp = float(w[pred & (y == 1)].sum())
    fn = float(w[~pred & (y == 1)].sum())
    fp = float(w[pred & (y == 0)].sum())
    tn = float(w[~pred & (y == 0)].sum())
    return tp, fp, tn, fn, int(keep.sum())


def roc_curve_np(df, col, weighted):
    y = df["target"].to_numpy()
    p = df[col].to_numpy()
    w = df["w"].to_numpy() if weighted else np.ones(len(df))
    keep = w > 0
    y, p, w = y[keep], p[keep], w[keep]
    order = np.argsort(-p, kind="mergesort")
    y, p, w = y[order], p[order], w[order]
    tps = np.cumsum(w * (y == 1))
    fps = np.cumsum(w * (y == 0))
    last = np.r_[np.where(np.diff(p) != 0)[0], len(p) - 1]
    tpr = np.r_[0.0, tps[last] / tps[-1]]
    fpr = np.r_[0.0, fps[last] / fps[-1]]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


def set_default_threshold(state_key, model_key):
    """Reset the slider whenever the model selection changes."""
    prev_key = state_key + "_model"
    if st.session_state.get(prev_key) != model_key:
        st.session_state[state_key] = DEFAULT_THR[model_key]
        st.session_state[prev_key] = model_key


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🫁 X-ray Triage Support</div>", unsafe_allow_html=True)
    st.caption("PyTorch · DenseNet121 · Grad-CAM")
    st.markdown("**Model used in the live demo**")
    demo_model = st.radio(
        "Model", ["B", "A"], format_func=lambda k: NAME[k], key="demo_model_choice", label_visibility="collapsed"
    )
    st.caption("B sees only the central chest (periphery masked). A sees the full image.")
    repo = cfg.get("repo_url")
    if repo:
        st.markdown("<a class='glink' href='" + repo + "' target='_blank'>View code and decision log →</a>",
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<div class='disclaimer'><b>Research prototype.</b> Not a medical device and not for diagnosis or treatment decisions. "
        "Trained only on paediatric chest X-rays from one dataset.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Hero + KPIs
# ----------------------------------------------------------------------------
st.markdown(
    "<div class='hero'><h1>Chest X-ray Pneumonia Triage Support</h1>"
    "<p>A decision-support prototype built with leakage-free evaluation, shortcut controls and explainability. "
    "Every number on this page comes from patients the model never saw.</p>"
    "<span class='chip'>Patient-level split</span><span class='chip'>Shortcut-controlled evaluation</span>"
    "<span class='chip'>Grad-CAM explainability</span><span class='chip'>Sealed test set, opened once</span></div>",
    unsafe_allow_html=True,
)

rowB = metrics.loc["B"]
auc_main, auc_ci = split_ci(rowB["AUROC adj"])
rec_main, rec_ci = split_ci(rowB["recall std"])
spec_main, spec_ci = split_ci(rowB["spec std"])
n_test = len(preds)
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi("AUROC, size-adjusted", auc_main, auc_ci), unsafe_allow_html=True)
k2.markdown(kpi("Recall (pneumonia caught)", rec_main, rec_ci), unsafe_allow_html=True)
k3.markdown(kpi("Specificity (normals cleared)", spec_main, spec_ci), unsafe_allow_html=True)
k4.markdown(kpi("Held-out test set", f"{n_test:,}", "images from unseen patients"), unsafe_allow_html=True)
st.write("")

tab_over, tab_demo, tab_perf, tab_built, tab_card = st.tabs(
    ["🏠 Overview", "🔬 Live demo", "📊 Performance explorer", "🧪 How it was built", "📋 Model card"]
)

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
with tab_over:
    st.markdown("### Three findings that shaped this project")
    g_std = metrics.loc["Geometry-only", "AUROC std"].split("[")[0].strip()
    g_adj = metrics.loc["Geometry-only", "AUROC adj"].split("[")[0].strip()
    c1, c2, c3 = st.columns(3, gap="large")
    c1.markdown(card("1 · The public test split leaked", "66.7%",
                     "of the original Kaggle test images came from patients who were also in training. "
                     "I rebuilt the split at patient level and verified it in code.", C["pink"]), unsafe_allow_html=True)
    c2.markdown(card("2 · Image size alone predicted the answer", g_std + " → " + g_adj,
                     "AUROC of a model that sees only image width and height (test set: standard, then size-adjusted). "
                     "A size-matched evaluation removed the shortcut.", C["amber"]), unsafe_allow_html=True)
    c3.markdown(card("3 · The test set was opened once", "1×",
                     "The model-choice rule was fixed before looking at test data, and every metric has a "
                     "patient-level bootstrap confidence interval.", C["cyan"]), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("### What the model does")
        st.markdown(
            "Given a frontal chest X-ray of a young child, it estimates the probability of a **pneumonia pattern**. "
            "It is framed as **triage support**: flag likely cases so a clinician can look at them sooner. "
            "A clinician always makes the call."
        )
        st.markdown(
            "<span class='pill'>Primary metric: recall</span><span class='pill'>Threshold chosen on validation</span>"
            "<span class='pill'>Paediatric data only</span>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("**How to read the numbers.** *Size-adjusted* metrics compare only images of similar size and shape, "
                    "so image dimensions cannot hint at the answer. Open the **Performance explorer** to move the threshold "
                    "and watch the trade-off between missed cases and false alarms on the real test predictions.")
    with right:
        st.markdown("### Test results at a glance")
        disp = metrics.reset_index().copy()
        disp["model"] = disp["model"].map({"A": NAME["A"], "B": NAME["B"] + " ★", "Geometry-only": NAME["G"]})
        disp = disp[["model", "AUROC std", "AUROC adj"]].rename(
            columns={"model": "Model", "AUROC std": "AUROC (all images)", "AUROC adj": "AUROC (size-adjusted)"})
        st.markdown(table_html(disp), unsafe_allow_html=True)
        st.caption("Square brackets show 95% confidence intervals (patient-level bootstrap, 1,000 resamples).")

# ----------------------------------------------------------------------------
# Live demo
# ----------------------------------------------------------------------------
with tab_demo:
    st.markdown("### Try it on a held-out test image")
    st.caption("Pick a sample from the held-out test set, or upload your own image. Uploaded images are processed in memory "
               "and are not saved by this app.")

    if "src" not in st.session_state:
        st.session_state["src"] = "sample"
        st.session_state["sample_idx"] = 1 if len(samples) > 1 else 0

    if samples:
        cols = st.columns(len(samples))
        for i, (col, s) in enumerate(zip(cols, samples)):
            if col.button(s["title"], key="sample_btn_" + str(i)):
                st.session_state["src"] = "sample"
                st.session_state["sample_idx"] = i

    up = st.file_uploader("Or upload a chest X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"], key="uploader")
    if up is not None:
        sig = (up.name, up.size)
        if st.session_state.get("upload_sig") != sig:
            st.session_state["upload_sig"] = sig
            st.session_state["src"] = "upload"

    img_bytes, sample = None, None
    if st.session_state["src"] == "upload" and up is not None:
        img_bytes = up.getvalue()
    elif samples:
        sample = samples[min(st.session_state["sample_idx"], len(samples) - 1)]
        img_bytes = (ASSETS / "samples" / sample["file"]).read_bytes()

    if img_bytes is None:
        st.info("Choose a sample above or upload an image to see a prediction.")
    else:
        try:
            gray = engine.to_gray256(Image.open(io.BytesIO(img_bytes)))
        except Exception:
            st.error("Could not read that image. Please upload a PNG or JPG.")
            gray = None

        if gray is not None:
            set_default_threshold("demo_thr", demo_model)
            result = run_inference(demo_model, img_bytes)
            live = result is not None
            if live:
                prob, cam = result
            elif sample is not None:
                prob, cam = float(sample["prob_" + demo_model]), None
                st.warning("Live inference is not available in this environment, so this page shows the precomputed result "
                           "for the selected test sample.")
            else:
                prob, cam = None, None
                st.warning("Live inference is not available in this environment. Select one of the test samples instead.")

            if prob is not None:
                ctl1, ctl2, ctl3 = st.columns([1.4, 1, 1], gap="large")
                thr = ctl1.slider("Decision threshold", 0.001, 0.999, step=0.001, format="%.3f", key="demo_thr")
                ctl1.caption("Default is the operating point chosen on validation data (target: catch 95% of pneumonia).")
                show_cam = ctl2.toggle("Show Grad-CAM", value=True, disabled=cam is None)
                alpha = ctl3.slider("Heatmap opacity", 0.1, 0.9, 0.45, 0.05, disabled=cam is None)

                st.write("")
                a, b, c = st.columns(3, gap="medium")
                a.markdown("**Input (grayscale)**")
                a.image(gray.resize((330, 330)))
                b.markdown("**What the model sees**")
                b.image(engine.masked_preview(gray, demo_model == "B").resize((330, 330)))
                c.markdown("**Where it looks (Grad-CAM)**")
                if cam is not None and show_cam:
                    c.image(engine.overlay(gray, cam, alpha).resize((330, 330)))
                elif cam is None:
                    c.caption("Heatmap needs live inference.")
                else:
                    c.caption("Heatmap hidden.")

                st.write("")
                v1, v2 = st.columns([1.1, 1], gap="large")
                positive = prob >= thr
                with v1:
                    if positive:
                        st.markdown("<div class='verdict pos'><div class='t'>⚠️ Pneumonia pattern flagged</div>"
                                    "<div class='s'>Probability " + f"{prob:.1%}" + " is at or above the threshold "
                                    + f"{thr:.3f}" + ". A clinician should review this image.</div></div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='verdict neg'><div class='t'>✅ No pneumonia pattern flagged</div>"
                                    "<div class='s'>Probability " + f"{prob:.1%}" + " is below the threshold "
                                    + f"{thr:.3f}" + ". This is not a clearance: the model can miss cases.</div></div>",
                                    unsafe_allow_html=True)
                    if sample is not None and st.session_state["src"] == "sample":
                        st.markdown("<span class='pill'>Ground truth: " + sample["truth"] + "</span>"
                                    "<span class='pill'>" + sample["title"] + "</span>", unsafe_allow_html=True)
                        st.caption(sample.get("note", ""))
                    else:
                        st.caption("Uploaded image: no ground truth available. This model only knows paediatric frontal chest "
                                   "X-rays, so results on other images are not meaningful.")
                with v2:
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 38}},
                        title={"text": "Pneumonia probability"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": C["red"] if positive else C["green"]},
                               "steps": [{"range": [0, thr * 100], "color": "rgba(52,211,153,.18)"},
                                         {"range": [thr * 100, 100], "color": "rgba(251,113,133,.18)"}],
                               "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": thr * 100}},
                    ))
                    st.plotly_chart(style(gauge, 300, legend=False), key="gauge")

    st.markdown("<div class='disclaimer'>For demonstration only. Never use this tool for clinical decisions.</div>",
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Performance explorer
# ----------------------------------------------------------------------------
with tab_perf:
    st.markdown("### Explore the real test-set predictions")
    st.caption("Everything here is computed live from the saved predictions on the held-out test set.")
    e1, e2 = st.columns([1.2, 1], gap="large")
    exp_model = e1.radio("Model", ["B", "A", "G"], format_func=lambda k: NAME[k], horizontal=True, key="exp_model")
    basis = e2.radio("Evaluation basis", ["Size-adjusted (fair)", "Standard (all images)"], horizontal=True, key="exp_basis")
    weighted = basis.startswith("Size")
    col = PROB_COL[exp_model]

    set_default_threshold("exp_thr", exp_model)
    t1, t2 = st.columns([4, 1])
    thr = t1.slider("Decision threshold", 0.001, 0.999, step=0.001, format="%.3f", key="exp_thr")

    def _reset():
        st.session_state["exp_thr"] = DEFAULT_THR[exp_model]
    t2.write("")
    t2.button("Reset to validated", on_click=_reset, key="exp_reset")

    tp, fp, tn, fn, n_used = counts(preds, col, thr, weighted)
    recall, spec, prec = safe_div(tp, tp + fn), safe_div(tn, tn + fp), safe_div(tp, tp + fp)
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(kpi("Recall", f"{recall:.3f}", "pneumonia cases caught"), unsafe_allow_html=True)
    m2.markdown(kpi("Specificity", f"{spec:.3f}", "normals correctly cleared"), unsafe_allow_html=True)
    m3.markdown(kpi("Precision", f"{prec:.3f}", "flagged cases that are real"), unsafe_allow_html=True)
    m4.markdown(kpi("Images used", f"{n_used}", "of " + str(len(preds)) + " test images"), unsafe_allow_html=True)
    st.write("")

    p1, p2 = st.columns(2, gap="large")
    with p1:
        fpr, tpr, auc = roc_curve_np(preds, col, weighted)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUROC {auc:.3f})", line=dict(color=C["cyan"], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color=C["slate"], dash="dot")))
        fig.add_trace(go.Scatter(x=[safe_div(fp, fp + tn)], y=[recall], mode="markers", name="Current threshold",
                                 marker=dict(color=C["amber"], size=14, line=dict(color="white", width=2))))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style(fig, 360), key="roc")
    with p2:
        z = [[tn, fp], [fn, tp]]
        fig = go.Figure(go.Heatmap(
            z=z, x=["Predicted normal", "Predicted pneumonia"], y=["Actual normal", "Actual pneumonia"],
            text=[[f"{v:,.0f}" for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=22),
            colorscale=[[0, "#10203d"], [1, "#22d3ee"]], showscale=False, hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed")
        title = "Confusion matrix (weighted counts)" if weighted else "Confusion matrix (image counts)"
        fig.update_layout(title=title)
        st.plotly_chart(style(fig, 360, legend=False), key="cm")

    rows = []
    for name, mask, kind in [("Bacterial pneumonia", preds["subtype"] == "bacteria", "recall"),
                             ("Viral pneumonia", preds["subtype"] == "virus", "recall"),
                             ("Normal", preds["label"] == "NORMAL", "specificity")]:
        a_tp, a_fp, a_tn, a_fn, a_n = counts(preds[mask], col, thr, weighted)
        value = safe_div(a_tp, a_tp + a_fn) if kind == "recall" else safe_div(a_tn, a_tn + a_fp)
        rows.append((name + " (" + kind + ")", value, a_n))
    fig = go.Figure(go.Bar(x=[r[0] for r in rows], y=[r[1] for r in rows],
                           text=[f"{r[1]:.3f}<br>n={r[2]}" for r in rows], textposition="outside",
                           marker=dict(color=[C["indigo"], C["violet"], C["green"]])))
    fig.update_layout(title="Rate by group at the current threshold", yaxis=dict(range=[0, 1.15]))
    st.plotly_chart(style(fig, 330, legend=False), key="groups")

    excluded = int((preds["w"] == 0).sum())
    if weighted:
        st.caption(f"Size-adjusted: {excluded} images are excluded because their size/shape bin has almost no examples of one "
                   "class, so no fair comparison is possible there. Weights equalise class mix within each size/shape bin.")
    if exp_model == "G":
        st.caption("This model never sees pixels: only image width, height and aspect ratio. Its size-adjusted score falls to "
                   "chance, which shows the adjustment works.")

    st.markdown("#### Final test table (thresholds fixed on validation)")
    full = metrics.reset_index().copy()
    full["model"] = full["model"].map({"A": NAME["A"], "B": NAME["B"] + " ★", "Geometry-only": NAME["G"]})
    full.loc[full["model"] == NAME["G"], ["recall std", "spec std", "recall adj", "spec adj"]] = "-"
    full = full.rename(columns={
        "model": "Model", "threshold": "Threshold", "AUROC std": "AUROC (all)", "recall std": "Recall (all)",
        "spec std": "Specificity (all)", "AUROC adj": "AUROC (adj.)", "recall adj": "Recall (adj.)", "spec adj": "Specificity (adj.)"})
    st.markdown(table_html(full), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# How it was built
# ----------------------------------------------------------------------------
with tab_built:
    st.markdown("### The evidence behind the design")
    b1, b2 = st.columns(2, gap="large")
    with b1:
        fig = go.Figure(go.Pie(labels=["From patients also seen in training", "From unseen patients"], values=[416, 208],
                               hole=0.68, marker=dict(colors=[C["pink"], C["slate"]]), textinfo="none", sort=False))
        fig.add_annotation(text="66.7%", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=34))
        fig.update_layout(title="Original Kaggle test set: 624 images")
        st.plotly_chart(style(fig, 340), key="leak")
        st.caption("264 patients appeared in both the original train and test folders.")
    with b2:
        names = ["Image size only", "Small CNN (naive)", "Small CNN (size-balanced)", "DenseNet121 (plain)", "DenseNet121 (size-balanced)"]
        std = [0.925, 0.992, 0.976, 0.997, 0.997]
        adj = [0.556, 0.966, 0.962, 0.990, 0.994]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Standard AUROC", x=names, y=std, marker_color=C["indigo"]))
        fig.add_trace(go.Bar(name="Size-adjusted AUROC", x=names, y=adj, marker_color=C["cyan"]))
        fig.update_layout(title="Validation: how much was the size shortcut worth?", barmode="group",
                          yaxis=dict(range=[0.4, 1.02]))
        st.plotly_chart(style(fig, 340), key="shortcut")
        st.caption("A model that sees only image dimensions scores 0.925, then drops to 0.556 once size is controlled.")

    b3, b4 = st.columns(2, gap="large")
    with b3:
        labels = ["Nothing hidden", "Top corners hidden", "Bottom corners hidden", "Outer 10% border hidden",
                  "Only center visible", "Only periphery visible"]
        vals = [0.994, 0.992, 0.993, 0.991, 0.985, 0.897]
        colors = [C["cyan"], C["green"], C["green"], C["green"], C["indigo"], C["amber"]]
        fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors,
                               text=[f"{v:.3f}" for v in vals], textposition="outside"))
        fig.update_layout(title="Occlusion tests (validation, size-adjusted AUROC)", xaxis=dict(range=[0.8, 1.02]))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 340, legend=False), key="occlusion")
        st.caption("Hiding corners or borders barely matters. The central chest carries almost all the signal.")
    with b4:
        labels = ["Full model", "Trained on center only", "Trained on periphery only"]
        vals = [0.994, 0.994, 0.966]
        fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[C["cyan"], C["indigo"], C["amber"]],
                               text=[f"{v:.3f}" for v in vals], textposition="outside"))
        fig.update_layout(title="Probe models (validation, size-adjusted AUROC)", yaxis=dict(range=[0.8, 1.03]))
        st.plotly_chart(style(fig, 340, legend=False), key="probes")
        st.caption("The periphery alone still reaches 0.966: some class signal exists outside the lungs (shortcut or edge "
                   "lung tissue). That is why model B, which cannot see it, is the headline model.")

    st.markdown("### Project timeline")
    steps = [
        ("Phases 0 to 1 · Setup and problem framing",
         "Colab + Drive + GitHub workflow with a decision log. Framed as decision support, recall as primary metric, claims limited to paediatric data."),
        ("Phase 2 · Exploratory analysis",
         "11,712 files collapsed to 5,856 unique images after removing a duplicated folder. 2,790 patients. 264 patients leaked across the original train and test folders."),
        ("Phase 3 · Patient-level split and pipeline",
         "Stratified patient-level split (about 71/15/14). Cached 256 px grayscale images, mild augmentation without flips, bilinear interpolation after a visual check caught jagged artifacts."),
        ("Phases 4 to 4B · Baselines and the size shortcut",
         "A pixel-free model reached AUROC 0.925. Built size-matched weights that pushed it to 0.556 and re-scored every model on that fair basis."),
        ("Phase 5 · Transfer learning",
         "ImageNet DenseNet121 reached 0.990 to 0.994 size-adjusted AUROC on validation, versus 0.966 for the small CNN."),
        ("Phase 6 · Explainability",
         "Grad-CAM, region attention, occlusion tests and probe models showed the central chest is sufficient. Periphery carries extra signal, so a center-only model was pre-declared."),
        ("Phase 7 · Sealed test evaluation",
         "Test set opened once. Thresholds fixed on validation, patient-level bootstrap intervals, subtype breakdown, and both pre-declared models reported."),
    ]
    for title, body in steps:
        with st.expander(title):
            st.write(body)

# ----------------------------------------------------------------------------
# Model card
# ----------------------------------------------------------------------------
with tab_card:
    st.markdown("### Model card")
    m_a, m_b = st.columns(2, gap="large")
    m_a.markdown(
        "<div class='card'><h4>Intended use</h4><ul>"
        "<li>Research and portfolio demonstration of pneumonia triage support on paediatric frontal chest X-rays.</li>"
        "<li>A clinician reviews every image. The output is a probability, not a diagnosis.</li>"
        "<li><b>Not</b> for clinical use, screening programs or any patient-care decision.</li></ul></div>",
        unsafe_allow_html=True)
    m_b.markdown(
        "<div class='card'><h4>Model and data</h4><ul>"
        "<li>DenseNet121 pretrained on ImageNet, fine-tuned for one output (pneumonia probability).</li>"
        "<li>Headline model B sees only the central chest region; model A sees the full image.</li>"
        "<li>Data: Kaggle Chest X-Ray Images (Pneumonia), paediatric patients from a single source, re-split at patient level "
        "(patient IDs inferred from filenames).</li></ul></div>",
        unsafe_allow_html=True)
    st.write("")
    m_c, m_d = st.columns(2, gap="large")
    m_c.markdown(
        "<div class='card'><h4>Evaluation protocol</h4><ul>"
        "<li>Patient-level stratified split; test set opened once.</li>"
        "<li>Model-choice rule and decision threshold fixed on validation before testing.</li>"
        "<li>Size-adjusted metrics compare images of similar size and shape; 95% intervals by patient bootstrap.</li></ul></div>",
        unsafe_allow_html=True)
    m_d.markdown(
        "<div class='card'><h4>Known limitations</h4><ul>"
        "<li>One data source, young children only. No external validation yet.</li>"
        "<li>Class signal also exists outside the central chest. Without lung masks, shortcut versus real tissue cannot be separated.</li>"
        "<li>The size-adjusted evaluation covers a subset of test images, and recall on that subset was below the 95% target.</li>"
        "<li>Probabilities are not calibrated. Performance on other scanners, ages or hospitals is unknown.</li></ul></div>",
        unsafe_allow_html=True)
    st.write("")
    st.markdown(
        "<div class='disclaimer'><b>Attribution.</b> Sample images come from the public Kaggle dataset "
        "\"Chest X-Ray Images (Pneumonia)\" (Kermany et al.), listed under CC BY 4.0. Please check the license on the dataset page. "
        "This app does not save uploaded images.</div>",
        unsafe_allow_html=True)
