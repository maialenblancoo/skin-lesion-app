"""
xai_app.py — Explainability helpers for the multimodal skin-lesion app.

Adapted from the research project's xai.py to the multimodal model:
  * hooks live on  model.backbone.blocks[-1]  (timm EfficientNet inside the fusion model)
  * every forward needs (image, metadata)
  * metadata SHAP is computed per-model with the real image fixed (as in notebook 13)
  * image-vs-metadata ablation is computed on the 0.6 / 0.4 ensemble

Two models are involved:
  loc  -> localization metadata (15 features)
  sa   -> sex + age metadata    (4 features)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import shap


# ── Metadata feature names ────────────────────────────────────────────────────
LOC_FEATURE_NAMES = [
    'abdomen', 'acral', 'back', 'chest', 'ear', 'face',
    'foot', 'genital', 'hand', 'lower extremity', 'neck',
    'scalp', 'trunk', 'unknown', 'upper extremity'
]
SA_FEATURE_NAMES = ['male', 'female', 'unknown', 'age']


# ── Map helpers ───────────────────────────────────────────────────────────────
def normalize_map(saliency_map):
    s_min, s_max = saliency_map.min(), saliency_map.max()
    if s_max - s_min < 1e-8:
        return np.zeros_like(saliency_map)
    return (saliency_map - s_min) / (s_max - s_min)


def overlay_heatmap(image_rgb, heatmap, alpha=0.5, colormap=None):
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    heatmap_uint8 = (normalize_map(heatmap) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return (alpha * heatmap_color + (1 - alpha) * image_rgb).astype(np.uint8)


# ── Grad-CAM (multimodal) ─────────────────────────────────────────────────────
class _GradCAMmm:
    def __init__(self, model):
        self.model = model
        self.acts = None
        self.grads = None
        target = self.model.backbone.blocks[-1]
        target.register_forward_hook(lambda m, i, o: setattr(self, 'acts', o.detach()))
        target.register_full_backward_hook(lambda m, gi, go: setattr(self, 'grads', go[0].detach()))

    def __call__(self, img_t, meta_t, class_idx, device):
        self.model.zero_grad()
        img = img_t.to(device).requires_grad_(True)
        mt = meta_t.to(device)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if mt.dim() == 1:
            mt = mt.unsqueeze(0)
        logits = self.model(img, mt)
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.acts).sum(dim=1).squeeze()
        cam = F.relu(cam).cpu().detach().numpy()
        cam = cv2.resize(cam, (img.shape[-1], img.shape[-2]))
        return normalize_map(cam)


def run_gradcam(model, img_t, meta_t, class_idx, device):
    """Grad-CAM heatmap (224x224, normalized) for the multimodal model."""
    return _GradCAMmm(model)(img_t, meta_t, class_idx, device)


# ── SmoothGrad (multimodal) ───────────────────────────────────────────────────
def run_smoothgrad(model, img_t, meta_t, class_idx, device, n_samples=20, noise_level=0.15):
    """SmoothGrad saliency (224x224, normalized) for the multimodal model."""
    img = img_t.to(device)
    mt = meta_t.to(device)
    if img.dim() == 4:
        img = img.squeeze(0)
    if mt.dim() == 1:
        mt = mt.unsqueeze(0)
    accumulated = np.zeros(img.shape[1:])
    std = noise_level * (img.max() - img.min()).item()
    for _ in range(n_samples):
        noise = torch.randn_like(img) * std
        noisy = (img + noise).unsqueeze(0).requires_grad_(True)
        model.zero_grad()
        logits = model(noisy, mt)
        logits[0, class_idx].backward()
        sal = noisy.grad.data.abs().squeeze().max(dim=0)[0].cpu().numpy()
        accumulated += sal
    return normalize_map(accumulated / n_samples)


# ── Metadata SHAP (per model, real image fixed) ───────────────────────────────
def run_shap_metadata(model, img_t, meta_vec, background_meta, target_class, device, n_background=50):
    """
    KernelExplainer SHAP for one model's metadata, with the real image fixed
    (image features precomputed once). Returns shap values (n_features,) for
    target_class. Mirrors notebook 13's per-case SHAP.
    """
    img = img_t.to(device)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        img_feat = model.backbone(img)            # (1, 1280)

    def predict_fn(meta):
        m = torch.tensor(meta, dtype=torch.float32, device=device)
        feats = img_feat.expand(m.shape[0], -1)
        z = model.metadata_branch(m)
        logits = model.classifier(torch.cat([feats, z], dim=1))
        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    bg = np.asarray(background_meta, dtype=np.float32)[:n_background]
    explainer = shap.KernelExplainer(predict_fn, bg)
    sv = explainer.shap_values(np.array([meta_vec], dtype=np.float32), silent=True)
    if isinstance(sv, list):
        return np.asarray(sv[target_class][0])
    return np.asarray(sv[0, :, target_class])


# ── Image vs metadata ablation (ensemble 0.6/0.4) ─────────────────────────────
def compute_image_vs_metadata_contrib(models, img_t, meta_loc, meta_sa,
                                       bg_loc, bg_sa, target_class, w1=0.6, w2=0.4):
    """
    2-point ablation of image vs metadata on the ENSEMBLE probability of
    target_class for a single case. Baselines: zero image, background metadata.
    bg_loc and bg_sa must have the same number of rows (paired). Returns
    (contrib_image, contrib_metadata, target_class).
    """
    device = models['device']
    img = img_t.to(device)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    zimg = torch.zeros_like(img)

    with torch.no_grad():
        fl  = models['loc'].backbone(img)    # real-image features
        fs  = models['sa'].backbone(img)
        fl0 = models['loc'].backbone(zimg)   # baseline-image features
        fs0 = models['sa'].backbone(zimg)

        def ens_prob(fl_, meta_l, fs_, meta_s):
            ml = torch.tensor(np.atleast_2d(meta_l), dtype=torch.float32, device=device)
            ms = torch.tensor(np.atleast_2d(meta_s), dtype=torch.float32, device=device)
            n = ml.shape[0]
            zl = models['loc'].metadata_branch(ml)
            pl = torch.softmax(models['loc'].classifier(torch.cat([fl_.expand(n, -1), zl], dim=1)), dim=1)
            zs = models['sa'].metadata_branch(ms)
            ps = torch.softmax(models['sa'].classifier(torch.cat([fs_.expand(n, -1), zs], dim=1)), dim=1)
            return (w1 * pl + w2 * ps).cpu().numpy()[:, target_class]

        p_real      = ens_prob(fl,  meta_loc, fs,  meta_sa)[0]
        p_img_only  = ens_prob(fl,  bg_loc,   fs,  bg_sa).mean()    # real image, bg metadata
        p_meta_only = ens_prob(fl0, meta_loc, fs0, meta_sa)[0]      # zero image, real metadata

    contrib_img  = abs(float(p_real) - float(p_meta_only))
    contrib_meta = abs(float(p_real) - float(p_img_only))
    return contrib_img, contrib_meta, target_class
