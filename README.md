# Multimodal Skin Lesion Classifier

Streamlit app for a multimodal (image + clinical metadata) skin-lesion
classifier built on HAM10000.

**System:** EfficientNet-B0 image branch + clinical metadata MLP, late fusion,
selective weighted ensemble (localization 0.6 / sex_age 0.4) with a melanoma
decision threshold of 0.30.

**Explainability:** Grad-CAM, SmoothGrad, per-model metadata SHAP,
image-vs-metadata ablation and MC-Dropout uncertainty. A full PDF report can
be downloaded for each case.

Models are hosted on the Hugging Face Hub (`maialenblancoo/skin-lesion-pfg`)
and downloaded at runtime.

> Research prototype. Not a medical device; not for clinical use.

**Author:** Maialen Blanco Ibarra · Universidad de Deusto, 2026
