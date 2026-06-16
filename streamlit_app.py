"""
Streamlit application for multimodal skin lesion classification.
Final system: EfficientNet-B0 image branch + clinical metadata, late fusion,
selective weighted ensemble (localization 0.6 / sex_age 0.4) + melanoma
threshold 0.30. Explainability: Grad-CAM, SmoothGrad, metadata SHAP,
image-vs-metadata ablation, MC-Dropout uncertainty.
Dataset: HAM10000 | TFG - Maialen Blanco Ibarra, Universidad de Deusto
"""

import io
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import inference as inf
import xai_app as xai

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = inf.CLASSES
CLASS_LABELS = {
    "akiec": "Actinic Keratoses",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma ⚠",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesion",
}
MEL_IDX         = inf.MEL_IDX
MEL_THRESHOLD   = inf.MEL_THRESHOLD     # 0.30
UNCERTAINTY_ENT = 0.50                  # entropy above this -> high uncertainty

LOCATIONS = inf.LOC_CATEGORIES
SEXES     = inf.SEX_CATEGORIES

# Locations with high melanoma prevalence but underrepresented in training
# (from the shortcut-learning analysis, notebook 13).
HIGH_RISK_UNDERREPRESENTED = {
    "ear":  31.7,
    "face": 18.0,
    "neck": 16.7,
}

CLASS_DESCRIPTIONS = {
    "mel":   "Malignant. Urgent referral needed.",
    "bcc":   "Malignant. Requires treatment.",
    "akiec": "Precancerous. Monitor closely.",
    "bkl":   "Benign skin growth.",
    "nv":    "Benign mole.",
    "df":    "Benign nodule.",
    "vasc":  "Benign vascular lesion.",
}
HIGH_RISK_CLASSES = {"mel", "bcc", "akiec"}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading models from Hugging Face…")
def load_models():
    return inf.load_models(device=torch.device("cpu"))


# ══════════════════════════════════════════════════════════════════════════════
# METADATA BACKGROUNDS (for SHAP and ablation)
# ══════════════════════════════════════════════════════════════════════════════

def build_backgrounds():
    """One paired background row per localization, with mean sex/age, so SHAP and
    the ablation always have contrast. Returns (bg_loc (15,15), bg_sa (15,4))."""
    n = len(LOCATIONS)
    bg_loc = np.zeros((n, len(LOCATIONS)), dtype=np.float32)
    for i in range(n):
        bg_loc[i, i] = 1.0
    bg_sa = np.zeros((n, len(SEXES) + 1), dtype=np.float32)
    bg_sa[:, 0] = 0.56          # mean male fraction
    bg_sa[:, 1] = 0.44          # mean female fraction
    bg_sa[:, 3] = 51.9 / 90.0   # mean age normalized
    return bg_loc, bg_sa


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def render_probability_bar(label, prob, is_pred, is_mel):
    bar_color = "#e53e3e" if is_mel else ("#3182ce" if is_pred else "#a0aec0")
    bar_width = max(prob * 100, 0.5)
    st.markdown(f"""
    <div style="margin-bottom:6px">
      <div style="display:flex; justify-content:space-between;
                  font-size:13px; margin-bottom:2px;">
        <span style="font-weight:{'700' if is_pred else '400'}">{label}</span>
        <span style="font-weight:{'700' if is_pred else '400'}">{prob:.1%}</span>
      </div>
      <div style="background:#e2e8f0; border-radius:4px; height:10px;">
        <div style="background:{bar_color}; width:{bar_width}%;
                    height:10px; border-radius:4px;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_shap_plot(shap_vals, feature_names, active_feature, title):
    """Horizontal SHAP bar chart, top-10 by |value|, active feature highlighted."""
    k = min(10, len(shap_vals))
    indices = np.argsort(np.abs(shap_vals))[::-1][:k]
    features = [feature_names[i] for i in indices][::-1]
    values   = [float(shap_vals[i]) for i in indices][::-1]
    colors   = ["#e53e3e" if v > 0 else "#3182ce" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(features, values, color=colors, height=0.6)
    ax.axvline(0, color="#4a5568", linewidth=0.8)
    ax.set_xlabel("SHAP value", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    for ticklabel, feat in zip(ax.get_yticklabels(), features):
        if feat == active_feature:
            ticklabel.set_fontweight("bold")
            ticklabel.set_bbox(dict(facecolor="#fefcbf", edgecolor="none", pad=2))
    fig.tight_layout()
    return fig


def render_contrib_plot(contrib_img, contrib_meta):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    bars = ax.barh(["Metadata", "Image"], [contrib_meta, contrib_img],
                   color=["#e53e3e", "#3182ce"], height=0.5)
    ax.set_xlabel("Contribution to p(predicted class)", fontsize=10)
    ax.set_title("Image vs Metadata influence", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, [contrib_meta, contrib_img]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, max(contrib_img, contrib_meta) * 1.3 + 1e-6)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report_pdf(pil_img, overlay, saliency, probs, pred_name, confidence,
                        sex, age, location, entropy, high_unc,
                        shap_loc, shap_sa, contrib_img, contrib_meta, agree):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Image as RLImage, Table, TableStyle, PageBreak)
    from reportlab.lib.units import cm
    from datetime import datetime

    buffer = io.BytesIO()
    W, H = A4
    MARGIN = 2 * cm
    INNER = W - 2 * MARGIN

    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=MARGIN, leftMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN,
                            title="Skin Lesion Classification Report",
                            author="Maialen Blanco Ibarra — Universidad de Deusto")
    story = []
    title_style = ParagraphStyle("title", fontSize=18, fontName="Helvetica-Bold", spaceAfter=10)
    sub_style   = ParagraphStyle("sub", fontSize=8, textColor=colors.grey, spaceAfter=14)
    h2_style    = ParagraphStyle("h2", fontSize=12, fontName="Helvetica-Bold",
                                 spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#2d3748"))
    note_style  = ParagraphStyle("note", fontSize=8, textColor=colors.grey, spaceAfter=6)
    disc_style  = ParagraphStyle("disc", fontSize=7, textColor=colors.grey, spaceBefore=12)

    story.append(Paragraph("Skin Lesion Classification Report", title_style))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "EfficientNet-B0 multimodal · ensemble 0.6/0.4 · melanoma threshold 0.30 "
        "· HAM10000 · Universidad de Deusto 2026", sub_style))

    pred_color = (colors.HexColor("#e53e3e") if pred_name == "mel" else
                  colors.HexColor("#ed8936") if pred_name in {"bcc", "akiec"} else
                  colors.HexColor("#38a169"))
    result_data = [
        ["Prediction",           CLASS_LABELS[pred_name]],
        ["Description",          CLASS_DESCRIPTIONS[pred_name]],
        ["Confidence",           f"{confidence:.1%}"],
        ["Melanoma probability", f"{probs[MEL_IDX]:.1%}"],
        ["Melanoma threshold",   str(MEL_THRESHOLD)],
        ["Predictive entropy",   f"{entropy:.3f}" + ("  (HIGH)" if high_unc else "")],
        ["Patient sex",          sex],
        ["Patient age",          f"{age} years"],
        ["Anatomical location",  location],
        ["Models agreement",     "agree" if agree else "disagree (weighted ensemble)"],
        ["Analysis date",        datetime.now().strftime("%Y-%m-%d %H:%M")],
    ]
    t = Table(result_data, colWidths=[5 * cm, INNER - 5 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), pred_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    def pil_to_rl(img, max_w_cm):
        buf = io.BytesIO()
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img.save(buf, format="PNG"); buf.seek(0)
        max_w = max_w_cm * cm
        return RLImage(buf, width=max_w, height=max_w * img.height / img.width)

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Visual Explanations", h2_style))

    # Small image size so the table + all three images fit on page 1.
    SMALL_W = 5.2 * cm   # width of each small image

    # Row 1: original lesion, left-aligned, small.
    aspect = pil_img.width / pil_img.height
    orig_w = SMALL_W
    orig_h = orig_w / aspect
    buf0 = io.BytesIO(); pil_img.save(buf0, format="PNG"); buf0.seek(0)
    orig_rl = RLImage(buf0, width=orig_w, height=orig_h)
    cap_o = Paragraph("Original", ParagraphStyle("capo", fontSize=7,
                      textColor=colors.grey, alignment=0, spaceBefore=2))
    orig_tbl = Table([[orig_rl], [cap_o]], colWidths=[orig_w], hAlign="LEFT")
    orig_tbl.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(orig_tbl)

    # Row 2: Grad-CAM + SmoothGrad side by side, left-aligned, small.
    img_w_cm = SMALL_W / cm
    row_imgs = [pil_to_rl(overlay, img_w_cm)]
    row_caps = ["Grad-CAM"]
    if saliency is not None:
        row_imgs.append(pil_to_rl(saliency, img_w_cm)); row_caps.append("SmoothGrad")
    maps_tbl = Table([row_imgs, row_caps], colWidths=[SMALL_W] * len(row_imgs), hAlign="LEFT")
    maps_tbl.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("FONTSIZE", (0, 1), (-1, 1), 7),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.grey),
    ]))
    story.append(maps_tbl)

    story.append(PageBreak())
    story.append(Paragraph("Class Probabilities", h2_style))
    sorted_idx = np.argsort(probs)[::-1]
    prob_data = [["Class", "Probability"]]
    for i in sorted_idx:
        prob_data.append([CLASS_LABELS[CLASS_NAMES[i]], f"{probs[i]:.1%}"])
    pt = Table(prob_data, colWidths=[INNER * 0.6, INNER * 0.4])
    pt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(pt)

    story.append(Paragraph("Metadata Influence (SHAP)", h2_style))
    if shap_loc is None:
        story.append(Paragraph("SHAP not computed.", note_style))
    else:
        story.append(Paragraph(
            "Localization model SHAP" +
            ("" if agree else " — shown together with the sex/age model because the "
                              "two models disagreed on this case."), note_style))
        loc_order = np.argsort(np.abs(shap_loc))[::-1][:8]
        loc_data = [["Localization feature", "SHAP value"]]
        for i in loc_order:
            loc_data.append([xai.LOC_FEATURE_NAMES[i], f"{shap_loc[i]:+.5f}"])
        lt = Table(loc_data, colWidths=[INNER * 0.6, INNER * 0.4])
        lt.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        story.append(lt)
        if (not agree) and (shap_sa is not None):
            story.append(Spacer(1, 0.3 * cm))
            sa_data = [["Sex/Age feature", "SHAP value"]]
            for i in range(len(shap_sa)):
                sa_data.append([xai.SA_FEATURE_NAMES[i], f"{shap_sa[i]:+.5f}"])
            sat = Table(sa_data, colWidths=[INNER * 0.6, INNER * 0.4])
            sat.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            story.append(sat)

    if contrib_img is not None:
        story.append(Paragraph("Image vs Metadata Contribution", h2_style))
        ratio = contrib_img / (contrib_meta + 1e-8)
        if ratio > 50:
            note = "Image-driven prediction. Clinical metadata had negligible influence."
        elif ratio > 10:
            note = f"Clinical metadata contributed to this prediction (ratio {ratio:.0f}x)."
        else:
            note = "Ambiguous image with strong clinical metadata influence. Specialist review recommended."
        contrib_data = [
            ["Image contribution",    f"{contrib_img:.4f}"],
            ["Metadata contribution", f"{contrib_meta:.4f}"],
            ["Image/Metadata ratio",  f"{ratio:.0f}x"],
            ["Interpretation",        note],
        ]
        ct = Table(contrib_data, colWidths=[INNER * 0.4, INNER * 0.6])
        ct.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        story.append(ct)

    story.append(Paragraph(
        "This report is intended for research and educational purposes only. "
        "It does not constitute a medical diagnosis. "
        "Always consult a qualified dermatologist for clinical decisions.", disc_style))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬", layout="wide")

st.markdown("""
<style>
  .main-title { font-size: 2.1rem; font-weight: 800; color: #1a202c; margin-bottom: 0; }
  .subtitle { font-size: 1rem; color: #718096; margin-top: 4px; }
  .result-box { border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; }
  .result-mel { background: #fff5f5; border: 2px solid #feb2b2; }
  .result-ok { background: #f0fff4; border: 2px solid #9ae6b4; }
  .result-warning { background: #fffaf0; border: 2px solid #f6ad55; }
  .warning-box { background: #fffbeb; border: 2px solid #f6e05e; border-radius: 10px;
                 padding: 14px 18px; font-size: 14px; color: #744210; margin-top: 10px; }
  .flag-box { background: #fff5f5; border: 2px solid #fc8181; border-radius: 10px;
              padding: 14px 18px; font-size: 14px; color: #742a2a; margin-top: 10px; }
  .section-title { font-size: 1rem; font-weight: 700; color: #2d3748;
                   margin-bottom: 8px; margin-top: 16px; }
  .disclaimer { font-size: 11px; color: #a0aec0; margin-top: 24px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🔬 Skin Lesion Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Multimodal deep learning · EfficientNet-B0 + clinical metadata · '
    'ensemble 0.6/0.4 · melanoma threshold 0.30 · HAM10000 · Explainable AI</p>',
    unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Patient data")
    uploaded_file = st.file_uploader("Dermoscopic image", type=["jpg", "jpeg", "png"],
                                     help="Upload a dermoscopic image of the skin lesion.")
    sex = st.selectbox("Sex", options=SEXES, index=0)
    age = st.slider("Patient age", 1, 90, 45, 1)
    location = st.selectbox("Anatomical location", options=LOCATIONS,
                            index=LOCATIONS.index("back"))
    run_shap = st.checkbox("Compute SHAP (slower)", value=False,
                           help="KernelExplainer on the metadata branch.")
    analyze_btn = st.button("Analyze", use_container_width=True, type="primary")

    st.divider()
    with st.expander("About the model"):
        st.markdown("""
**Intended use:** clinical decision-support tool for primary-care 
physicians and other non-specialists, to assist in triaging skin 
lesions. Specialists may also use it as a complementary second 
opinion. It is not an auto-diagnostic or at-home diagnostic device, 
and does not replace clinical judgment.
                    
**System:** EfficientNet-B0 multimodal, selective weighted ensemble
(localization 0.6 / sex_age 0.4) + melanoma threshold 0.30.

**Dataset:** HAM10000 — 10,015 images, 7 classes.

| Metric | Value |
|--------|-------|
| Melanoma Recall | 0.857 |
| Melanoma ROC-AUC | 0.967 |
| Macro Recall | 0.726 |

**Inference:** TTA ×5 · threshold 0.30 (validation-selected).

**XAI:** Grad-CAM, SmoothGrad, metadata SHAP, image-vs-metadata ablation,
MC-Dropout uncertainty.

**Limitations:** underrepresented locations (ear, face, neck);
not validated outside HAM10000.

**Author:** Maialen Blanco Ibarra · Universidad de Deusto, 2026
        """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if not uploaded_file:
    st.info("👈 Upload a dermoscopic image and fill in the patient data to begin.")
    st.stop()

pil_img = Image.open(uploaded_file).convert("RGB")
image_rgb = np.array(pil_img)

img_bytes = uploaded_file.getvalue()
cache_key = f"{hashlib.md5(img_bytes).hexdigest()}_{sex}_{age}_{location}_{run_shap}"
if st.session_state.get("last_cache_key") != cache_key:
    st.session_state.pop("last_result", None)
    st.session_state["last_cache_key"] = cache_key

if analyze_btn or "last_result" in st.session_state:

    if analyze_btn:
        models = load_models()
        device = models["device"]

        progress = st.progress(0, text="Step 1/4 — Running TTA inference…")
        result = inf.predict(image_rgb, sex, age, location, models)
        probs, pred_idx = result["final_probs"], result["pred_idx"]
        pred_name, confidence = result["pred_class"], float(result["final_probs"][result["pred_idx"]])
        agree = result["agree"]

        progress.progress(30, text="Step 2/4 — Grad-CAM…")
        meta_loc = inf.encode_metadata(sex, age, location, inf.METADATA_COLS_LOC)
        meta_sa  = inf.encode_metadata(sex, age, location, inf.METADATA_COLS_SA)
        img_t = inf.VAL_TRANSFORM(image=image_rgb)["image"]
        disp = cv2.resize(image_rgb, (224, 224))
        target_loc = int(np.argmax(result["probs_loc"]))
        cam = xai.run_gradcam(models["loc"], img_t.clone(), torch.tensor(meta_loc), target_loc, device)
        overlay = xai.overlay_heatmap(disp, cam)

        progress.progress(55, text="Step 3/4 — SmoothGrad…")
        sg = xai.run_smoothgrad(models["loc"], img_t.clone(), torch.tensor(meta_loc), target_loc, device)
        saliency = xai.overlay_heatmap(disp, sg, colormap=cv2.COLORMAP_HOT)

        progress.progress(75, text="Step 4/4 — Uncertainty…")
        unc = inf.mc_dropout_uncertainty(image_rgb, sex, age, location, models,
                                         entropy_threshold=UNCERTAINTY_ENT)

        shap_loc = shap_sa = contrib_img = contrib_meta = None
        if run_shap:
            progress.progress(85, text="Computing SHAP…")
            bg_loc, bg_sa = build_backgrounds()
            shap_loc = xai.run_shap_metadata(models["loc"], img_t.clone(), meta_loc,
                                             bg_loc, target_loc, device, n_background=15)
            if not agree:
                target_sa = int(np.argmax(result["probs_sa"]))
                shap_sa = xai.run_shap_metadata(models["sa"], img_t.clone(), meta_sa,
                                                bg_sa, target_sa, device, n_background=15)
            contrib_img, contrib_meta, _ = xai.compute_image_vs_metadata_contrib(
                models, img_t.clone(), meta_loc, meta_sa, bg_loc, bg_sa, pred_idx)

        progress.progress(100, text="Analysis complete ✓")
        progress.empty()

        st.session_state["last_result"] = {
            "probs": probs, "pred_idx": pred_idx, "pred_name": pred_name,
            "confidence": confidence, "agree": agree, "overlay": overlay,
            "saliency": saliency, "entropy": unc["entropy"],
            "high_unc": unc["high_uncertainty"], "shap_loc": shap_loc, "shap_sa": shap_sa,
            "contrib_img": contrib_img, "contrib_meta": contrib_meta,
        }

    res = st.session_state["last_result"]
    probs = res["probs"]; pred_idx = res["pred_idx"]; pred_name = res["pred_name"]
    confidence = res["confidence"]; agree = res["agree"]
    overlay = res["overlay"]; saliency = res["saliency"]
    entropy = res["entropy"]; high_unc = res["high_unc"]
    shap_loc = res["shap_loc"]; shap_sa = res["shap_sa"]
    contrib_img = res["contrib_img"]; contrib_meta = res["contrib_meta"]

    col1, col2 = st.columns([1, 1], gap="large")

    # LEFT — images & explanations
    with col1:
        st.markdown('<p class="section-title">Input image</p>', unsafe_allow_html=True)
        st.image(pil_img, caption="Original", use_container_width=True)
        cam_col, sal_col = st.columns(2)
        cam_col.image(overlay, caption="Grad-CAM", use_container_width=True)
        sal_col.image(saliency, caption="SmoothGrad", use_container_width=True)
        with st.expander("What do these visualizations mean?"):
            st.markdown("""
**Grad-CAM** — regions that most influenced the decision (red = important).

**SmoothGrad** — pixel-level saliency averaged over noisy copies of the image.
            """)

        if shap_loc is not None:
            st.markdown('<p class="section-title">Metadata influence (SHAP)</p>',
                        unsafe_allow_html=True)
            if not np.all(np.abs(shap_loc) < 1e-6):
                fig = render_shap_plot(shap_loc, xai.LOC_FEATURE_NAMES,
                                       location, "Localization model — SHAP")
                st.pyplot(fig); plt.close(fig)
            if (not agree) and (shap_sa is not None):
                st.caption("Models disagreed → showing the sex/age model too:")
                active_sa = "age"
                fig2 = render_shap_plot(shap_sa, xai.SA_FEATURE_NAMES,
                                        active_sa, "Sex/Age model — SHAP")
                st.pyplot(fig2); plt.close(fig2)
            elif agree:
                st.caption("Both models agreed → localization (primary) model shown.")

        if contrib_img is not None:
            ratio = contrib_img / (contrib_meta + 1e-8)
            fig3 = render_contrib_plot(contrib_img, contrib_meta)
            st.pyplot(fig3); plt.close(fig3)
            if ratio > 50:
                st.info("**Image-driven prediction.** Clinical metadata had negligible influence.")
            elif ratio > 10:
                st.info(f"**Clinical metadata contributed** (image/metadata ratio: {ratio:.0f}×).")
            else:
                st.warning("**Strong metadata influence on an ambiguous image.** Specialist review recommended.")

    # RIGHT — results
    with col2:
        st.markdown('<div style="height: 38px;"></div>', unsafe_allow_html=True)
        is_mel = pred_idx == MEL_IDX
        is_high = pred_name in HIGH_RISK_CLASSES
        box_cls = "result-mel" if is_mel else ("result-warning" if is_high else "result-ok")
        icon = "🔴" if is_mel else ("🟠" if is_high else "🟢")
        st.markdown(
            f'<div class="result-box {box_cls}">'
            f'<div style="font-size:1.5rem; font-weight:800;">{icon} {CLASS_LABELS[pred_name]}</div>'
            f'<div style="font-size:0.9rem; color:#718096; margin-top:2px;">{CLASS_DESCRIPTIONS[pred_name]}</div>'
            f'<div style="font-size:0.9rem; color:#4a5568; margin-top:6px;">'
            f'Top class probability: <b>{confidence:.1%}</b> &nbsp;|&nbsp; '
            f'Melanoma probability: <b>{probs[MEL_IDX]:.1%}</b> (threshold: {MEL_THRESHOLD})'
            f'</div></div>', unsafe_allow_html=True)

        # Uncertainty — always shown, colored by level
        if entropy < 0.30:
            unc_bg, unc_border, unc_color, unc_label, unc_icon = "#f0fff4", "#9ae6b4", "#22543d", "Low", "🟢"
        elif entropy < UNCERTAINTY_ENT:
            unc_bg, unc_border, unc_color, unc_label, unc_icon = "#fffaf0", "#f6ad55", "#744210", "Moderate", "🟡"
        else:
            unc_bg, unc_border, unc_color, unc_label, unc_icon = "#fff5f5", "#fc8181", "#742a2a", "High", "🔴"

        unc_pct = min(entropy * 100, 100)
        st.markdown(
            f'<div style="background:{unc_bg}; border:2px solid {unc_border}; '
            f'border-radius:10px; padding:14px 18px; margin-top:10px;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<span style="font-weight:700; color:{unc_color};">{unc_icon} Uncertainty: {unc_label}</span>'
            f'<span style="font-weight:700; color:{unc_color};">entropy {entropy:.3f}</span>'
            f'</div>'
            f'<div style="background:#e2e8f0; border-radius:4px; height:8px; margin-top:8px;">'
            f'<div style="background:{unc_border}; width:{unc_pct}%; height:8px; border-radius:4px;"></div>'
            f'</div>'
            f'<div style="font-size:12px; color:{unc_color}; margin-top:6px;">'
            f'{"Prediction is stable." if entropy < UNCERTAINTY_ENT else "Low model reliability — interpret with clinical judgment and consider referral."}'
            f'</div></div>',
            unsafe_allow_html=True)

        if location in HIGH_RISK_UNDERREPRESENTED:
            prev = HIGH_RISK_UNDERREPRESENTED[location]
            st.markdown(
                f'<div class="flag-box">🚩 <b>High-risk location notice</b> — '
                f'<i>{location}</i> has a real-world melanoma prevalence of <b>{prev}%</b> '
                f'but is underrepresented in training (n&lt;100). The model may underweight it. '
                f'<b>Consider specialist referral regardless of prediction.</b></div>',
                unsafe_allow_html=True)

        st.markdown('<p class="section-title">Class probabilities</p>', unsafe_allow_html=True)
        for i in np.argsort(probs)[::-1]:
            render_probability_bar(CLASS_LABELS[CLASS_NAMES[i]], float(probs[i]),
                                   i == pred_idx, i == MEL_IDX)

        st.markdown('<p class="section-title">Patient input summary</p>', unsafe_allow_html=True)
        st.markdown(f"- **Sex:** {sex}  \n- **Age:** {age} years  \n- **Location:** {location}")
        st.caption("Models agreed." if agree else "Models disagreed — weighted ensemble used.")

        try:
            pdf_bytes = generate_report_pdf(
                pil_img, overlay, saliency, probs, pred_name, confidence,
                sex, age, location, entropy, high_unc,
                shap_loc, shap_sa, contrib_img, contrib_meta, agree)
            st.download_button("⤓ Download PDF Report", data=pdf_bytes,
                               file_name=f"skin_lesion_report_{pred_name}.pdf",
                               mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.caption(f"PDF report unavailable: {e}")

    st.markdown(
        '<p class="disclaimer">This tool is for research and educational purposes only. '
        'It does not constitute a medical diagnosis. '
        'Always consult a qualified dermatologist for clinical decisions.</p>',
        unsafe_allow_html=True)
