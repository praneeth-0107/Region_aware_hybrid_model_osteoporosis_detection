"""
=============================================================
  BONE DENSITY PREDICTION PIPELINE
  Upload an X-ray → Segment Femur & Tibia → Ensemble Predict
  Outputs: Class + confidence % for Femur, Tibia, and Combined

  USAGE:
      python predict_bone_density.py --image path/to/xray.jpg

  OPTIONAL OVERRIDES:
      --femur_arch   efficientnet_v2_s   (default)
      --tibia_arch   efficientnet_v2_s   (default)
      --models_dir   models/             (default)
      --output_dir   predictions/        (default)
      --no_save                          skip saving crops

  REQUIREMENTS:
      pip install torch torchvision opencv-python
                  scipy pillow numpy matplotlib seaborn

  OUTPUTS:
      predictions/
        femur_crop.png        ← segmented femur region
        tibia_crop.png        ← segmented tibia region
        annotated.png         ← X-ray with joint line overlay
        prediction_report.txt ← full text report
        prediction_chart.png  ← visual confidence chart
=============================================================
"""

import os
import re
import sys
import argparse
import warnings
import numpy as np
import cv2
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import torch
import torch.nn as nn
from torchvision import transforms, models
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from PIL import Image
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURATION  (must match training scripts)
# ──────────────────────────────────────────────
CLASSES     = ["osteoporosis", "osteopenia", "normal"]
CLS2IDX     = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE    = 224
USE_AMP     = torch.cuda.is_available()
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Segmentation config (must match batch_knee_separator_v3.py)
PAD_PX          = 15
SEARCH_LO       = 0.20
SEARCH_HI       = 0.88
BONE_MASK_KSIZE = 9
SMOOTH_SIGMA    = 8
DIP_PROMINENCE  = 25
DIP_DISTANCE    = 15

# Disease risk tier labels
RISK_TIERS = {
    "osteoporosis": ("HIGH RISK",   "⚠️",  "#e74c3c"),
    "osteopenia":   ("MEDIUM RISK", "⚡",  "#f39c12"),
    "normal":       ("HEALTHY",     "✅",  "#27ae60"),
}


# ──────────────────────────────────────────────
# SEGMENTATION  (from batch_knee_separator_v3.py)
# ──────────────────────────────────────────────
def get_active_roi_bounds(img, threshold=15):
    H, W      = img.shape
    col_means = img.mean(axis=0)
    row_means = img.mean(axis=1)
    x0 = next((i for i in range(W)       if col_means[i] > threshold), 0)
    x1 = next((i for i in range(W-1,-1,-1) if col_means[i] > threshold), W-1)
    y0 = next((i for i in range(H)       if row_means[i] > threshold), 0)
    y1 = next((i for i in range(H-1,-1,-1) if row_means[i] > threshold), H-1)
    return x0, y0, x1, y1


def preprocess_seg(roi):
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(roi)
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=40, sigmaSpace=40)
    return denoised


def get_bone_mask(denoised):
    _, bone = cv2.threshold(denoised, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (BONE_MASK_KSIZE, BONE_MASK_KSIZE))
    bone    = cv2.morphologyEx(bone, cv2.MORPH_CLOSE, kernel, iterations=3)
    bone    = cv2.morphologyEx(bone, cv2.MORPH_OPEN,  kernel, iterations=1)
    return bone


def bone_width_profile(bone, margin_frac=0.10):
    rH, rW = bone.shape
    cx0    = int(rW * margin_frac)
    cx1    = int(rW * (1.0 - margin_frac))
    widths = bone[:, cx0:cx1].sum(axis=1).astype(float) / 255.0
    return gaussian_filter1d(widths, sigma=SMOOTH_SIGMA)


def find_joint_row_in_roi(roi, offset_y=0):
    rH, _    = roi.shape
    denoised = preprocess_seg(roi)
    bone     = get_bone_mask(denoised)
    profile  = bone_width_profile(bone)

    y_lo = int(rH * SEARCH_LO)
    y_hi = int(rH * SEARCH_HI)
    seg  = profile[y_lo:y_hi]

    inv_seg = -seg
    peaks, props = find_peaks(inv_seg,
                               prominence=DIP_PROMINENCE,
                               distance=DIP_DISTANCE,
                               width=2)
    if len(peaks) > 0:
        best      = np.argmax(props["prominences"])
        joint_loc = y_lo + peaks[best]
    else:
        joint_loc = y_lo + int(np.argmin(seg))

    pre  = profile[:joint_loc]
    post = profile[joint_loc:]
    fb   = int(np.argmax(pre[-max(1, int(rH*0.25)):])
               + max(0, joint_loc - int(rH*0.25)))
    tt   = joint_loc + int(np.argmax(post[:max(1, int(rH*0.25))]))

    return (joint_loc + offset_y, fb + offset_y, tt + offset_y)


def is_bilateral(img):
    H, W = img.shape
    if W / H > 1.3:
        return False
    cx_lo        = int(W * 0.40)
    cx_hi        = int(W * 0.60)
    center_strip = img[:, cx_lo:cx_hi]
    left_strip   = img[:, int(W*0.15):int(W*0.35)]
    right_strip  = img[:, int(W*0.65):int(W*0.85)]
    center_mean  = center_strip.mean()
    sides_mean   = (left_strip.mean() + right_strip.mean()) / 2.0
    return (sides_mean - center_mean) > 20


def find_bilateral_split(img):
    H, W     = img.shape
    cx_lo    = int(W * 0.35)
    cx_hi    = int(W * 0.65)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(img)
    _, bm    = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    col_dens = bm[:, cx_lo:cx_hi].mean(axis=0)
    return cx_lo + int(np.argmin(col_dens))


def segment_xray(image_path):
    """
    Segments X-ray into femur and tibia crops.
    Returns:
        femur_crop  : numpy array (grayscale)
        tibia_crop  : numpy array (grayscale)
        annotated   : numpy array (BGR colour for saving)
        joint_row   : int
        meta        : dict with mode info
    """
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Cannot read image: {image_path}")

    H, W      = raw.shape
    bilateral = is_bilateral(raw)
    meta      = {"bilateral": bilateral, "H": H, "W": W}

    if bilateral:
        split_col      = find_bilateral_split(raw)
        left_roi       = raw[:, :split_col]
        right_roi      = raw[:, split_col:]
        jl, fbl, ttl   = find_joint_row_in_roi(left_roi,  offset_y=0)
        jr, fbr, ttr   = find_joint_row_in_roi(right_roi, offset_y=0)
        gap_l          = abs(ttl - fbl)
        gap_r          = abs(ttr - fbr)
        final_row      = int((jl * gap_l + jr * gap_r) / (gap_l + gap_r)) \
                         if (gap_l + gap_r) > 0 else (jl + jr) // 2
        meta.update({"split_col": split_col, "jl": jl, "jr": jr})
        fb_global      = (fbl + fbr) // 2
        tt_global      = (ttl + ttr) // 2
    else:
        x0, y0, x1, y1 = get_active_roi_bounds(raw)
        roi             = raw[y0:y1, x0:x1]
        jrow_roi, fb_roi, tt_roi = find_joint_row_in_roi(roi, offset_y=0)
        final_row  = jrow_roi + y0
        fb_global  = fb_roi   + y0
        tt_global  = tt_roi   + y0
        meta.update({"roi": (x0, y0, x1, y1)})

    # Clamp joint row
    final_row = max(int(H * 0.20), min(int(H * 0.88), final_row))
    meta["joint_row"]  = final_row
    meta["fb_global"]  = fb_global
    meta["tt_global"]  = tt_global

    # Crops
    femur_crop = raw[:final_row + PAD_PX, :]
    tibia_crop = raw[max(0, final_row - PAD_PX):, :]

    # Annotated QC image
    vis  = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(vis, (0, final_row), (W, final_row), (0, 255, 0), 3)
    for x in range(0, W, 24):
        cv2.line(vis, (x, final_row + PAD_PX), (x+12, final_row + PAD_PX), (0, 160, 0), 1)
        cv2.line(vis, (x, final_row - PAD_PX), (x+12, final_row - PAD_PX), (0, 160, 0), 1)

    if bilateral:
        cv2.line(vis, (split_col, 0), (split_col, H), (180, 180, 180), 1)
        cv2.line(vis, (0, jl), (split_col, jl), (0, 220, 220), 2)
        cv2.line(vis, (split_col, jr), (W, jr), (0, 220, 220), 2)
    else:
        cv2.line(vis, (0, fb_global), (W, fb_global), (255, 80, 0),  2)
        cv2.line(vis, (0, tt_global), (W, tt_global), (0, 140, 255), 2)

    lbl_y = max(final_row - 12, 30)
    cv2.putText(vis, "FEMUR", (12, lbl_y),            font, 0.9, (0, 255, 0), 2)
    cv2.putText(vis, "TIBIA", (12, final_row + 38),   font, 0.9, (0, 255, 0), 2)
    cv2.putText(vis, f"joint row: {final_row}  ({final_row/H*100:.0f}%)",
                (W - 280, final_row - 8), font, 0.6, (0, 255, 0), 2)

    return femur_crop, tibia_crop, vis, final_row, meta


# ──────────────────────────────────────────────
# MODEL BUILDER  (must match training scripts)
# ──────────────────────────────────────────────
def build_model(arch, num_classes=3):
    def custom_head(in_feat):
        return nn.Sequential(
            nn.Dropout(p=0.40),
            nn.Linear(in_feat, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.30),
            nn.Linear(256, num_classes),
        )

    if arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = custom_head(m.fc.in_features)
    elif arch == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = custom_head(m.classifier.in_features)
    elif arch == "vgg16":
        m = models.vgg16(weights=None)
        m.classifier = custom_head(m.classifier[0].in_features)
    elif arch == "efficientnet_b3":
        m = models.efficientnet_b3(weights=None)
        m.classifier = custom_head(m.classifier[1].in_features)
    elif arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier = custom_head(m.classifier[1].in_features)
    elif arch == "swin_t":
        m = models.swin_t(weights=None)
        m.head = custom_head(m.head.in_features)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m


def load_model(arch, site, models_dir):
    """
    Tries to load in this priority order:
      1. v2s_{site}_best_fold.pth    (k-fold best)
      2. {arch}_{site}.pth           (standard training)
    """
    candidates = [
        os.path.join(models_dir, f"v2s_{site}_best_fold.pth"),
        os.path.join(models_dir, f"{arch}_{site}.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
            model = build_model(arch, num_classes=len(CLASSES)).to(DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            print(f"  ✅ Loaded {site} model  ← {path}")
            return model
    raise FileNotFoundError(
        f"No model found for arch='{arch}' site='{site}' in '{models_dir}'.\n"
        f"  Tried:\n" + "\n".join(f"    {p}" for p in candidates)
    )


# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────
def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict_single(model, img_array_gray):
    """
    Takes a grayscale numpy crop, converts to RGB PIL, returns softmax probs.
    Returns: probs (np.array shape [3]), pred_class (int), pred_label (str)
    """
    tf   = get_val_transform()
    pil  = Image.fromarray(img_array_gray).convert("RGB")
    t    = tf(pil).unsqueeze(0).to(DEVICE)

    with torch.amp.autocast(device_type="cuda" if USE_AMP else "cpu",
                             enabled=USE_AMP):
        out = model(t)

    probs      = torch.softmax(out, dim=1).squeeze().cpu().numpy()
    pred_class = int(np.argmax(probs))
    return probs, pred_class, CLASSES[pred_class]


def auc_weighted_ensemble(femur_probs, tibia_probs,
                           femur_auc=0.5, tibia_auc=0.5):
    """
    Weighted combination:  prob = (w_f * femur + w_t * tibia) / (w_f + w_t)
    """
    total = femur_auc + tibia_auc
    w_f   = femur_auc / total
    w_t   = tibia_auc / total
    combined = w_f * femur_probs + w_t * tibia_probs
    pred_cls = int(np.argmax(combined))
    return combined, pred_cls, CLASSES[pred_cls], w_f, w_t


# ──────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────
def build_report(image_path, femur_probs, tibia_probs,
                 ensemble_probs, ensemble_pred,
                 w_f, w_t, meta):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def prob_bar(p, width=20):
        n = int(round(p * width))
        return "█" * n + "░" * (width - n) + f"  {p*100:5.1f}%"

    lines = [
        "=" * 65,
        "  BONE DENSITY PREDICTION REPORT",
        f"  Image  : {Path(image_path).name}",
        f"  Date   : {now}",
        f"  Device : {DEVICE}",
        "=" * 65,
        "",
        "  ─────────────────────────────────────────",
        "  FEMUR  (distal femur / patella region)",
        "  ─────────────────────────────────────────",
    ]
    fp = int(np.argmax(femur_probs))
    for i, cls in enumerate(CLASSES):
        marker = " ◄ PREDICTION" if i == fp else ""
        lines.append(f"  {cls:<18} {prob_bar(femur_probs[i])}{marker}")
    tier, icon, _ = RISK_TIERS[CLASSES[fp]]
    lines += [
        f"\n  Femur Prediction : {CLASSES[fp].upper()}  [{tier}]",
        f"  Femur Confidence : {femur_probs[fp]*100:.1f}%",
        "",
        "  ─────────────────────────────────────────",
        "  TIBIA  (proximal tibia / fibula region)",
        "  ─────────────────────────────────────────",
    ]
    tp = int(np.argmax(tibia_probs))
    for i, cls in enumerate(CLASSES):
        marker = " ◄ PREDICTION" if i == tp else ""
        lines.append(f"  {cls:<18} {prob_bar(tibia_probs[i])}{marker}")
    tier, icon, _ = RISK_TIERS[CLASSES[tp]]
    lines += [
        f"\n  Tibia Prediction : {CLASSES[tp].upper()}  [{tier}]",
        f"  Tibia Confidence : {tibia_probs[tp]*100:.1f}%",
        "",
        "  ─────────────────────────────────────────",
        f"  ENSEMBLE  (femur {w_f*100:.0f}% + tibia {w_t*100:.0f}%)",
        "  ─────────────────────────────────────────",
    ]
    ep = int(np.argmax(ensemble_probs))
    for i, cls in enumerate(CLASSES):
        marker = " ◄ FINAL DIAGNOSIS" if i == ep else ""
        lines.append(f"  {cls:<18} {prob_bar(ensemble_probs[i])}{marker}")
    tier, icon, _ = RISK_TIERS[ensemble_pred]
    lines += [
        f"\n  FINAL DIAGNOSIS   : {ensemble_pred.upper()}",
        f"  RISK LEVEL        : {tier}",
        f"  OVERALL CONFIDENCE: {ensemble_probs[ep]*100:.1f}%",
        "",
        "  ─────────────────────────────────────────",
        "  DISEASE PROBABILITY BREAKDOWN",
        "  ─────────────────────────────────────────",
        f"  {'Class':<18} {'Femur %':>8} {'Tibia %':>8} {'Ensemble %':>12}",
        f"  {'─'*50}",
    ]
    for i, cls in enumerate(CLASSES):
        lines.append(
            f"  {cls:<18} {femur_probs[i]*100:>7.1f}%"
            f" {tibia_probs[i]*100:>7.1f}%"
            f" {ensemble_probs[i]*100:>11.1f}%"
        )
    lines += [
        "",
        "  ─────────────────────────────────────────",
        "  SEGMENTATION INFO",
        "  ─────────────────────────────────────────",
        f"  Mode       : {'Bilateral' if meta.get('bilateral') else 'Single knee'}",
        f"  Joint row  : {meta.get('joint_row', 'N/A')} px  "
        f"({meta.get('joint_row', 0)/meta.get('H',1)*100:.1f}% from top)",
        f"  Image size : {meta.get('W', '?')} × {meta.get('H', '?')} px",
        "",
        "  ⚠  DISCLAIMER: This is an AI-assisted tool for research",
        "     purposes only. Not a substitute for clinical diagnosis.",
        "=" * 65,
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────
# CHART
# ──────────────────────────────────────────────
def save_prediction_chart(femur_probs, tibia_probs, ensemble_probs,
                           ensemble_pred, save_path):
    colors = {
        "osteoporosis": "#e74c3c",
        "osteopenia":   "#f39c12",
        "normal":       "#27ae60",
    }
    bar_colors = [colors[c] for c in CLASSES]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    data = [
        (femur_probs,    "Femur"),
        (tibia_probs,    "Tibia"),
        (ensemble_probs, f"Ensemble\n→ {ensemble_pred.upper()}"),
    ]

    for ax, (probs, title) in zip(axes, data):
        ax.set_facecolor("#161b22")
        ax.spines[:].set_color("#30363d")
        ax.tick_params(colors="#8b949e")

        bars = ax.barh(CLASSES, probs * 100,
                       color=bar_colors, alpha=0.85, height=0.55,
                       edgecolor="#30363d", linewidth=0.5)

        # Value labels inside bars
        for bar, p in zip(bars, probs):
            w = bar.get_width()
            ax.text(min(w + 1.5, 95), bar.get_y() + bar.get_height() / 2,
                    f"{p*100:.1f}%", va="center", ha="left",
                    color="#c9d1d9", fontsize=10, fontweight="bold")

        # Highlight predicted class
        pred_idx = int(np.argmax(probs))
        bars[pred_idx].set_edgecolor("#ffffff")
        bars[pred_idx].set_linewidth(2.0)

        ax.set_xlim(0, 110)
        ax.set_xlabel("Probability (%)", color="#8b949e", fontsize=9)
        ax.set_title(title, color="#f0f6fc", fontsize=11, fontweight="bold", pad=10)
        ax.yaxis.set_tick_params(labelcolor="#c9d1d9", labelsize=10)
        ax.xaxis.set_tick_params(labelcolor="#8b949e", labelsize=8)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#30363d", linewidth=0.5, linestyle="--")

    # Legend
    patches = [mpatches.Patch(color=colors[c], label=c.capitalize())
               for c in CLASSES]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               facecolor="#1c2128", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    tier, _, col = RISK_TIERS[ensemble_pred]
    fig.suptitle(
        f"Bone Density Analysis  |  Final: {ensemble_pred.upper()}  [{tier}]",
        color="#f0f6fc", fontsize=13, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Chart saved  → {save_path}")


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Bone Density Prediction: X-ray → Segment → Ensemble → Diagnose")
    parser.add_argument("--image",       required=True,
                        help="Path to input X-ray image")
    parser.add_argument("--femur_arch",  default="efficientnet_v2_s",
                        help="Femur model architecture (default: efficientnet_v2_s)")
    parser.add_argument("--tibia_arch",  default="efficientnet_v2_s",
                        help="Tibia model architecture (default: efficientnet_v2_s)")
    parser.add_argument("--models_dir",  default="models",
                        help="Folder containing .pth model files")
    parser.add_argument("--output_dir",  default="predictions",
                        help="Folder to save outputs")
    parser.add_argument("--femur_auc",   type=float, default=None,
                        help="Manual femur AUC weight (default: 0.5 equal)")
    parser.add_argument("--tibia_auc",   type=float, default=None,
                        help="Manual tibia AUC weight (default: 0.5 equal)")
    parser.add_argument("--no_save",     action="store_true",
                        help="Do not save crop/annotation images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print("  BONE DENSITY PREDICTION PIPELINE")
    print(f"  Device  : {DEVICE}  |  AMP : {USE_AMP}")
    print(f"  Image   : {args.image}")
    print(f"  Femur   : {args.femur_arch}")
    print(f"  Tibia   : {args.tibia_arch}")
    print(f"{'='*65}\n")

    # ── STEP 1: Segment X-ray ─────────────────────────────────────
    print("  [1/4]  Segmenting X-ray...")
    try:
        femur_crop, tibia_crop, annotated, joint_row, meta = \
            segment_xray(args.image)
    except Exception as e:
        print(f"  [ERROR] Segmentation failed: {e}")
        sys.exit(1)

    print(f"  ✓  Joint line at row {joint_row}  "
          f"({joint_row/meta['H']*100:.0f}% from top)")
    print(f"  ✓  Femur crop : {femur_crop.shape[1]}×{femur_crop.shape[0]} px")
    print(f"  ✓  Tibia crop : {tibia_crop.shape[1]}×{tibia_crop.shape[0]} px")

    if not args.no_save:
        femur_path = os.path.join(args.output_dir, "femur_crop.png")
        tibia_path = os.path.join(args.output_dir, "tibia_crop.png")
        annot_path = os.path.join(args.output_dir, "annotated.png")
        cv2.imwrite(femur_path, femur_crop)
        cv2.imwrite(tibia_path, tibia_crop)
        cv2.imwrite(annot_path, annotated)
        print(f"  💾 Femur crop  → {femur_path}")
        print(f"  💾 Tibia crop  → {tibia_path}")
        print(f"  💾 Annotated   → {annot_path}")

    # ── STEP 2: Load models ───────────────────────────────────────
    print(f"\n  [2/4]  Loading models from '{args.models_dir}'...")
    try:
        femur_model = load_model(args.femur_arch, "femur", args.models_dir)
        tibia_model = load_model(args.tibia_arch, "tibia", args.models_dir)
    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}")
        sys.exit(1)

    # ── STEP 3: Run inference ─────────────────────────────────────
    print(f"\n  [3/4]  Running inference...")
    femur_probs, femur_pred_cls, femur_pred_label = predict_single(
        femur_model, femur_crop)
    tibia_probs, tibia_pred_cls, tibia_pred_label = predict_single(
        tibia_model, tibia_crop)

    # AUC weights (use provided or equal 50/50)
    f_auc = args.femur_auc if args.femur_auc is not None else 0.5
    t_auc = args.tibia_auc if args.tibia_auc is not None else 0.5

    ensemble_probs, ensemble_pred_cls, ensemble_pred_label, w_f, w_t = \
        auc_weighted_ensemble(femur_probs, tibia_probs, f_auc, t_auc)

    # ── STEP 4: Report & chart ────────────────────────────────────
    print(f"\n  [4/4]  Generating report and chart...")

    report = build_report(
        args.image,
        femur_probs, tibia_probs, ensemble_probs,
        ensemble_pred_label, w_f, w_t, meta
    )
    print("\n" + report)

    report_path = os.path.join(args.output_dir, "prediction_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  📄 Report      → {report_path}")

    chart_path = os.path.join(args.output_dir, "prediction_chart.png")
    save_prediction_chart(
        femur_probs, tibia_probs, ensemble_probs,
        ensemble_pred_label, chart_path
    )

    # ── FINAL SUMMARY ─────────────────────────────────────────────
    tier, icon, _ = RISK_TIERS[ensemble_pred_label]
    print(f"\n{'='*65}")
    print(f"  {icon}  FINAL RESULT")
    print(f"  Diagnosis : {ensemble_pred_label.upper()}")
    print(f"  Risk      : {tier}")
    print(f"  Confidence: {ensemble_probs[ensemble_pred_cls]*100:.1f}%")
    print(f"\n  Per-bone confidence for '{ensemble_pred_label}':")
    print(f"  Femur : {femur_probs[ensemble_pred_cls]*100:.1f}%")
    print(f"  Tibia : {tibia_probs[ensemble_pred_cls]*100:.1f}%")
    print(f"\n  Per-bone breakdown:")
    print(f"  {'Class':<18} {'Femur':>8} {'Tibia':>8} {'Ensemble':>10}")
    print(f"  {'─'*48}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<18} {femur_probs[i]*100:>7.1f}%"
              f" {tibia_probs[i]*100:>7.1f}%"
              f" {ensemble_probs[i]*100:>9.1f}%")
    print(f"\n  Outputs saved → {args.output_dir}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()