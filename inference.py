"""
inference.py — Inference engine for the multimodal skin-lesion app.

Self-contained version for Streamlit Cloud: it does NOT depend on the research
project's config.py / dataset.py. All constants are defined here, and the two
trained models are downloaded from the Hugging Face Hub at runtime.

Pipeline (final system, selected on validation):
    TTA (5 augmentations) per model
        -> selective weighted ensemble (localization 0.6 / sex_age 0.4)
        -> melanoma decision threshold 0.30

Uncertainty (MC Dropout) is a separate, optional signal — it does not change the
diagnosis. It reproduces notebook 12 (flat 0.6/0.4 ensemble, no TTA, T=30 passes).
"""

import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download

from model import MultimodalModel


# -- Classes (same order as training) --
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASSES)
CLASS_NAMES_FULL = {
    "akiec": "Actinic Keratosis",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesion",
}
MEL_IDX = CLASSES.index('mel')
BCC_IDX = CLASSES.index('bcc')

# -- Metadata encoding (must match training exactly) --
SEX_CATEGORIES = ['male', 'female', 'unknown']
LOC_CATEGORIES = [
    'abdomen', 'acral', 'back', 'chest', 'ear', 'face',
    'foot', 'genital', 'hand', 'lower extremity', 'neck',
    'scalp', 'trunk', 'unknown', 'upper extremity'
]
METADATA_COLS_LOC = ['localization']
METADATA_COLS_SA  = ['sex', 'age']

# -- Final-system hyper-parameters (selected on validation) --
W_PRIMARY     = 0.6
W_SECONDARY   = 0.4
MEL_THRESHOLD = 0.30
AGE_MEAN      = 51.9

# -- Hugging Face model repo --
HF_REPO   = "maialenblancoo/skin-lesion-pfg"
LOC_FILE  = "multimodal_b0_none_localization_fold0.pth"
SA_FILE   = "multimodal_b0_none_sex_age_fold0.pth"

# -- Image transforms --
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

TTA_TRANSFORMS = [
    A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.HorizontalFlip(p=1.0), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.VerticalFlip(p=1.0), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Rotate(limit=(90, 90), p=1.0), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Rotate(limit=(270, 270), p=1.0), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
]
VAL_TRANSFORM = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


def encode_metadata(sex, age, localization, metadata_cols, age_mean=AGE_MEAN):
    """Encode raw clinical inputs into the fixed-length float vector (training-identical)."""
    features = []
    if 'sex' in metadata_cols:
        sex_val = str(sex).lower() if sex is not None else 'unknown'
        if sex_val not in SEX_CATEGORIES:
            sex_val = 'unknown'
        features.extend([1.0 if sex_val == c else 0.0 for c in SEX_CATEGORIES])
    if 'age' in metadata_cols:
        if age is None or (isinstance(age, float) and np.isnan(age)):
            age = age_mean
        features.append(float(age) / 90.0)
    if 'localization' in metadata_cols:
        loc_val = str(localization).lower() if localization is not None else 'unknown'
        if loc_val not in LOC_CATEGORIES:
            loc_val = 'unknown'
        features.extend([1.0 if loc_val == c else 0.0 for c in LOC_CATEGORIES])
    return np.array(features, dtype=np.float32)


def load_models(device=None):
    """Download both models from HF Hub and load them. Wrap in st.cache_resource."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loc_path = hf_hub_download(repo_id=HF_REPO, filename=LOC_FILE)
    sa_path  = hf_hub_download(repo_id=HF_REPO, filename=SA_FILE)
    model_loc = MultimodalModel(metadata_dim=len(LOC_CATEGORIES),
                                efficientnet_version='b0', pretrained=False)
    model_loc.load_state_dict(torch.load(loc_path, map_location=device))
    model_loc = model_loc.to(device).eval()
    model_sa = MultimodalModel(metadata_dim=len(SEX_CATEGORIES) + 1,
                               efficientnet_version='b0', pretrained=False)
    model_sa.load_state_dict(torch.load(sa_path, map_location=device))
    model_sa = model_sa.to(device).eval()
    return {'loc': model_loc, 'sa': model_sa, 'device': device}


def _tta_probs_single(model, image_rgb, metadata_vec, device):
    """Average softmax over the 5 TTA transforms for one image. -> (7,)"""
    batch = torch.stack([t(image=image_rgb)['image'] for t in TTA_TRANSFORMS])
    meta  = torch.tensor(metadata_vec, dtype=torch.float32).unsqueeze(0).repeat(batch.shape[0], 1)
    model.eval()
    with torch.no_grad():
        logits = model(batch.to(device), meta.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    return probs.mean(axis=0)


def selective_weighted_ensemble(probs_primary, probs_secondary, w1=W_PRIMARY, w2=W_SECONDARY):
    """Selective weighted ensemble (identical to nb09). Accepts (7,) or (N,7)."""
    pp = np.atleast_2d(probs_primary).astype(np.float64)
    ps = np.atleast_2d(probs_secondary).astype(np.float64)
    agree = np.argmax(pp, axis=1) == np.argmax(ps, axis=1)
    final = w1 * pp + w2 * ps
    final[agree] = pp[agree]
    if np.ndim(probs_primary) == 1:
        return final[0], bool(agree[0])
    return final, agree


def apply_melanoma_threshold(probs, threshold=MEL_THRESHOLD):
    """Melanoma decision rule (identical to nb10). Accepts (7,) or (N,7)."""
    p2d = np.atleast_2d(probs)
    preds = []
    for i in range(len(p2d)):
        if p2d[i, MEL_IDX] >= threshold:
            preds.append(MEL_IDX)
        else:
            r = p2d[i].copy(); r[MEL_IDX] = -1.0
            preds.append(int(np.argmax(r)))
    preds = np.array(preds)
    return int(preds[0]) if np.ndim(probs) == 1 else preds


def predict(image_rgb, sex, age, localization, models,
            w1=W_PRIMARY, w2=W_SECONDARY, threshold=MEL_THRESHOLD):
    """Full deterministic prediction for one lesion (TTA + ensemble + threshold)."""
    device = models['device']
    meta_loc = encode_metadata(sex, age, localization, METADATA_COLS_LOC)
    meta_sa  = encode_metadata(sex, age, localization, METADATA_COLS_SA)
    probs_loc = _tta_probs_single(models['loc'], image_rgb, meta_loc, device)
    probs_sa  = _tta_probs_single(models['sa'],  image_rgb, meta_sa,  device)
    final_probs, agree = selective_weighted_ensemble(probs_loc, probs_sa, w1, w2)
    pred_idx = apply_melanoma_threshold(final_probs, threshold)
    return {
        'final_probs': final_probs,
        'pred_idx':    pred_idx,
        'pred_class':  CLASSES[pred_idx],
        'pred_name':   CLASS_NAMES_FULL[CLASSES[pred_idx]],
        'agree':       agree,
        'probs_loc':   probs_loc,
        'probs_sa':    probs_sa,
    }


def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_uncertainty(image_rgb, sex, age, localization, models,
                           T=30, w1=W_PRIMARY, w2=W_SECONDARY, entropy_threshold=0.5):
    """MC-Dropout uncertainty (nb12 setup: flat ensemble, no TTA, T passes through head)."""
    device = models['device']
    meta_loc = encode_metadata(sex, age, localization, METADATA_COLS_LOC)
    meta_sa  = encode_metadata(sex, age, localization, METADATA_COLS_SA)
    img = VAL_TRANSFORM(image=image_rgb)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        feat_loc = models['loc'].backbone(img)
        feat_sa  = models['sa'].backbone(img)
    ml = torch.tensor(meta_loc, dtype=torch.float32).unsqueeze(0).to(device)
    ms = torch.tensor(meta_sa,  dtype=torch.float32).unsqueeze(0).to(device)
    enable_mc_dropout(models['loc'])
    enable_mc_dropout(models['sa'])
    samples = []
    with torch.no_grad():
        for _ in range(T):
            zl = models['loc'].metadata_branch(ml)
            pl = torch.softmax(models['loc'].classifier(torch.cat([feat_loc, zl], dim=1)), dim=1)
            zs = models['sa'].metadata_branch(ms)
            ps = torch.softmax(models['sa'].classifier(torch.cat([feat_sa, zs], dim=1)), dim=1)
            samples.append((w1 * pl + w2 * ps).cpu().numpy()[0])
    models['loc'].eval(); models['sa'].eval()
    samples    = np.stack(samples, axis=0)
    mean_probs = samples.mean(axis=0)
    entropy    = float(-np.sum(mean_probs * np.log(mean_probs + 1e-12)) / np.log(NUM_CLASSES))
    return {
        'mean_probs':       mean_probs,
        'entropy':          entropy,
        'high_uncertainty': bool(entropy >= entropy_threshold),
        'pred_std':         float(samples.std(axis=0)[int(np.argmax(mean_probs))]),
    }
