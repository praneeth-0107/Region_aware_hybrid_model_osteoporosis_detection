"""
=============================================================
  BONE DENSITY — AUC-WEIGHTED ENSEMBLE
  Loads the best Femur and Tibia models from the 6-architecture
  training pipeline and evaluates a weighted ensemble on the
  SAME held-out test set (80/10/10 stratified split).

  STRATEGY:
      ensemble_prob = (auc_femur  * prob_femur
                     + auc_tibia * prob_tibia)
                     / (auc_femur + auc_tibia)

  HOW TO SPECIFY THE BEST MODELS:
      Option A — Auto (recommended):
          Run with no arguments. The script reads
          results/final_summary.txt produced by
          train_all_models.py and extracts the best
          femur and tibia architectures automatically.

      Option B — Manual override via CLI:
          python ensemble_bone_density.py \
              --femur_arch efficientnet_v2_s \
              --tibia_arch densenet121

  OUTPUTS:
      results/ensemble_report.txt
      results/ensemble_confusion.png
      results/ensemble_roc.png
      results/ensemble_summary_chart.png

  REQUIREMENTS:
      Same environment as train_all_models.py
      (torch, torchvision, scikit-learn, matplotlib, seaborn, pillow, numpy)

  USAGE:
      python ensemble_bone_density.py
      python ensemble_bone_density.py --femur_arch resnet50 --tibia_arch vgg16
=============================================================
"""

import os
import re
import sys
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    auc as sk_auc,
)
from PIL import Image
from pathlib import Path

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# MUST MATCH train_all_models.py EXACTLY
# ──────────────────────────────────────────────
FEMUR_DIR   = "output/femur"
TIBIA_DIR   = "output/tibia"
MODELS_DIR  = "models"
RESULTS_DIR = "results"

IMG_SIZE    = 224
BATCH_SIZE  = 8
SEED        = 42
NUM_WORKERS = 0
USE_AMP     = torch.cuda.is_available()
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

CLASSES = ["osteoporosis", "osteopenia", "normal"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────
# DATASET  (identical to training script)
# ──────────────────────────────────────────────
class BoneDataset(Dataset):
    EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    @staticmethod
    def from_folder(folder):
        folder = Path(folder)
        if not folder.exists():
            print(f"  [ERROR] Folder not found: {folder}")
            return [], []
        paths, labels, skipped = [], [], 0
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in BoneDataset.EXTS:
                continue
            stem  = f.stem.lower()
            label = None
            for cls, idx in CLS2IDX.items():
                if cls in stem:
                    label = idx
                    break
            if label is None:
                m = re.search(r"[_\-](\d)$", stem)
                if m and int(m.group(1)) < len(CLASSES):
                    label = int(m.group(1))
            if label is None:
                skipped += 1
                continue
            paths.append(str(f))
            labels.append(label)
        if skipped:
            print(f"  ⚠  Skipped {skipped} files (no class label in filename)")
        return paths, labels


# ──────────────────────────────────────────────
# REPRODUCE THE EXACT SAME TEST SPLIT
# ──────────────────────────────────────────────
def get_test_split(paths, labels):
    """
    Reproduces the identical 80/10/10 stratified split used during
    training (same SEED, same TEST_RATIO) so we evaluate on the
    original held-out test set.
    """
    labels_np = np.array(labels)
    tr_val_idx, te_idx = train_test_split(
        np.arange(len(paths)),
        test_size    = TEST_RATIO,
        stratify     = labels_np,
        random_state = SEED,
    )
    te_paths  = [paths[i]  for i in te_idx]
    te_labels = [labels[i] for i in te_idx]
    return te_paths, te_labels


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────
# MODEL BUILDER  (identical to training script)
# ──────────────────────────────────────────────
def build_model(arch, num_classes=3, pretrained=False):
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


# ──────────────────────────────────────────────
# LOAD A SAVED MODEL
# ──────────────────────────────────────────────
def load_model(arch, site):
    model_path = os.path.join(MODELS_DIR, f"{arch}_{site}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            f"Run train_all_models.py first."
        )
    ckpt  = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = build_model(arch, num_classes=len(CLASSES), pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    saved_auc = ckpt.get("val_loss", None)   # fallback; real AUC from summary
    print(f"  ✅ Loaded  : {arch} ({site})  ← {model_path}")
    return model


# ──────────────────────────────────────────────
# INFERENCE — COLLECT SOFTMAX PROBABILITIES
# ──────────────────────────────────────────────
@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    all_probs, all_labs = [], []
    for imgs, labs in loader:
        imgs = imgs.to(DEVICE)
        with torch.amp.autocast(
                device_type="cuda" if USE_AMP else "cpu", enabled=USE_AMP):
            out = model(imgs)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_labs.extend(labs.numpy())
    return np.array(all_probs), np.array(all_labs)


# ──────────────────────────────────────────────
# AUTO-DETECT BEST ARCHITECTURES FROM SUMMARY
# ──────────────────────────────────────────────
def parse_best_from_summary(summary_path):
    """
    Reads results/final_summary.txt and extracts the best
    femur and tibia architectures and their AUC scores.

    Returns:
        {
            "femur": {"arch": "efficientnet_v2_s", "auc": 0.9823},
            "tibia": {"arch": "densenet121",        "auc": 0.9741},
        }
    """
    if not os.path.exists(summary_path):
        return None

    results = {}
    with open(summary_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Parse the "Best Femur Model" and "Best Tibia Model" blocks
    for site in ("femur", "tibia"):
        arch_match = re.search(
            rf"Best {site.capitalize()} Model\s*:\s*(\S+)", content, re.IGNORECASE)
        auc_match  = re.search(
            rf"Best {site.capitalize()} Model.*?ROC-AUC\s*:\s*([0-9.]+)",
            content, re.IGNORECASE | re.DOTALL)

        if arch_match:
            arch = arch_match.group(1).lower()
            auc  = float(auc_match.group(1)) if auc_match else None
            results[site] = {"arch": arch, "auc": auc}

    return results if results else None


def compute_auc(labels, probs):
    """Macro one-vs-rest ROC-AUC. Returns float or None on failure."""
    try:
        return roc_auc_score(
            np.eye(len(CLASSES))[labels], probs,
            multi_class="ovr", average="macro")
    except Exception:
        return None


# ──────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────
def _dark_ax(ax):
    ax.set_facecolor("#161b22")
    ax.spines[:].set_color("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#f0f6fc")


def plot_confusion(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        ax=ax, linewidths=0.5, linecolor="#30363d",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted",  color="#c9d1d9", fontsize=12)
    ax.set_ylabel("True Label", color="#c9d1d9", fontsize=12)
    ax.set_title(title,         color="#f0f6fc", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  🗂  Confusion   → {save_path}")


def plot_roc(labels, probs, title, save_path):
    colors = ["#58a6ff", "#f78166", "#56d364"]
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    _dark_ax(ax)

    labels_bin = np.eye(len(CLASSES))[labels]
    macro_sum  = 0.0
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc     = sk_auc(fpr, tpr)
        macro_sum  += roc_auc
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{cls} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#30363d", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate", color="#c9d1d9")
    ax.set_ylabel("True Positive Rate",  color="#c9d1d9")
    ax.set_title(
        f"{title}\nMacro AUC = {macro_sum / len(CLASSES):.4f}",
        color="#f0f6fc", fontweight="bold")
    ax.legend(facecolor="#1c2128", edgecolor="#30363d", labelcolor="#c9d1d9")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📉 ROC         → {save_path}")


def plot_comparison_chart(rows, save_path):
    """
    Horizontal bar chart comparing:
        Femur-only | Tibia-only | Equal (50/50) | AUC-weighted
    across Accuracy and Balanced Accuracy.
    """
    labels_  = [r["label"] for r in rows]
    accs     = [r["acc"]     * 100 for r in rows]
    bal_accs = [r["bal_acc"] * 100 for r in rows]
    aucs     = [(r["auc"] or 0) * 100 for r in rows]

    n     = len(labels_)
    y     = np.arange(n)
    w     = 0.25

    fig, ax = plt.subplots(figsize=(13, max(6, n * 1.4)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    b1 = ax.barh(y + w,   accs,     w, label="Accuracy (%)",          color="#58a6ff", alpha=0.9)
    b2 = ax.barh(y,       bal_accs, w, label="Balanced Accuracy (%)", color="#56d364", alpha=0.9)
    b3 = ax.barh(y - w,   aucs,     w, label="ROC-AUC × 100",         color="#e3b341", alpha=0.9)

    for bars in [b1, b2, b3]:
        for bar in bars:
            v = bar.get_width()
            if v > 0:
                ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{v:.1f}", va="center", ha="left",
                        color="#c9d1d9", fontsize=8)

    # Highlight best accuracy row
    best_idx = np.argmax(accs)
    ax.axhspan(best_idx - 0.5, best_idx + 0.5,
               color="#388bfd", alpha=0.10, label="Best Accuracy")

    ax.set_yticks(y)
    ax.set_yticklabels(labels_, color="#c9d1d9", fontsize=10)
    ax.set_xlabel("Score", color="#c9d1d9")
    ax.set_title("Ensemble Comparison — Test Set",
                 color="#f0f6fc", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")
    ax.set_xlim(0, 115)
    ax.legend(facecolor="#1c2128", edgecolor="#30363d",
              labelcolor="#c9d1d9", loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Chart       → {save_path}")


# ──────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────
def save_report(
    preds, labels, probs,
    femur_arch, tibia_arch,
    femur_auc_weight, tibia_auc_weight,
    comparison_rows,
    save_path,
):
    acc     = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    cm      = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    report  = classification_report(
        labels, preds, target_names=CLASSES, zero_division=0)

    try:
        macro_auc = roc_auc_score(
            np.eye(len(CLASSES))[labels], probs,
            multi_class="ovr", average="macro")
        auc_str = f"{macro_auc:.4f}"
    except Exception as e:
        auc_str = f"N/A ({e})"

    per_class_acc = {}
    for i, cls in enumerate(CLASSES):
        mask = labels == i
        per_class_acc[cls] = float(
            (preds[mask] == i).mean()) if mask.sum() > 0 else float("nan")

    total_w = femur_auc_weight + tibia_auc_weight
    fw_pct  = femur_auc_weight / total_w * 100
    tw_pct  = tibia_auc_weight / total_w * 100

    lines = [
        "=" * 65,
        "  BONE DENSITY — AUC-WEIGHTED ENSEMBLE REPORT",
        "  Evaluated on HELD-OUT TEST SET (10%)",
        "=" * 65,
        "",
        f"  Femur model   : {femur_arch.upper()}  (AUC weight = {femur_auc_weight:.4f}  [{fw_pct:.1f}%])",
        f"  Tibia model   : {tibia_arch.upper()}  (AUC weight = {tibia_auc_weight:.4f}  [{tw_pct:.1f}%])",
        f"  Formula       : prob = ({fw_pct:.0f}% × femur_prob) + ({tw_pct:.0f}% × tibia_prob)",
        "",
        "  ─────────────────────────────────────────────────────",
        "  ENSEMBLE PERFORMANCE",
        "  ─────────────────────────────────────────────────────",
        f"  Overall Accuracy    : {acc * 100:.2f}%",
        f"  Balanced Accuracy   : {bal_acc * 100:.2f}%",
        f"  Macro ROC-AUC       : {auc_str}",
        "",
        "  Per-Class Accuracy:",
    ]
    for cls, v in per_class_acc.items():
        val_str = f"{v * 100:.2f}%" if not np.isnan(v) else "N/A"
        lines.append(f"    {cls:<18}: {val_str}")

    lines += [
        "",
        "  Classification Report:",
        report,
        "",
        "  Confusion Matrix (rows=true, cols=pred):",
        f"  Classes: {' | '.join(CLASSES)}",
        str(cm),
        "",
        "  ─────────────────────────────────────────────────────",
        "  COMPARISON ACROSS STRATEGIES",
        "  ─────────────────────────────────────────────────────",
        f"  {'Strategy':<22} {'Accuracy':>10} {'Bal Acc':>10} {'AUC':>8}",
        f"  {'─' * 55}",
    ]
    for row in comparison_rows:
        auc_val = f"{row['auc']:.4f}" if row["auc"] is not None else "N/A"
        tag     = "  ← BEST" if row.get("best") else ""
        lines.append(
            f"  {row['label']:<22} {row['acc']*100:>9.2f}% "
            f"{row['bal_acc']*100:>9.2f}% {auc_val:>8}{tag}"
        )
    lines += ["", "=" * 65]

    text = "\n".join(lines)
    print("\n" + text)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  📄 Report      → {save_path}")
    return cm, acc, bal_acc


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AUC-weighted ensemble of best femur + tibia bone density models.")
    parser.add_argument("--femur_arch", type=str, default=None,
                        help="Override femur architecture (e.g. efficientnet_v2_s)")
    parser.add_argument("--tibia_arch", type=str, default=None,
                        help="Override tibia architecture (e.g. densenet121)")
    parser.add_argument("--femur_auc",  type=float, default=None,
                        help="Override femur AUC weight (default: computed from test set)")
    parser.add_argument("--tibia_auc",  type=float, default=None,
                        help="Override tibia AUC weight (default: computed from test set)")
    args = parser.parse_args()

    print(f"\n{'=' * 65}")
    print("  BONE DENSITY — AUC-WEIGHTED ENSEMBLE")
    print(f"  Device : {DEVICE}  |  AMP : {USE_AMP}")
    print(f"{'=' * 65}\n")

    # ── Step 1: Determine best architectures ──────────────────────
    summary_path = os.path.join(RESULTS_DIR, "final_summary.txt")
    auto_best    = parse_best_from_summary(summary_path)

    femur_arch = args.femur_arch
    tibia_arch = args.tibia_arch

    if femur_arch is None or tibia_arch is None:
        if auto_best:
            print("  Auto-detected best architectures from final_summary.txt:")
            if femur_arch is None and "femur" in auto_best:
                femur_arch = auto_best["femur"]["arch"]
                print(f"    Femur → {femur_arch}")
            if tibia_arch is None and "tibia" in auto_best:
                tibia_arch = auto_best["tibia"]["arch"]
                print(f"    Tibia → {tibia_arch}")
        else:
            print(
                "  ⚠  results/final_summary.txt not found.\n"
                "     Pass --femur_arch and --tibia_arch manually,\n"
                "     or run train_all_models.py first."
            )
            sys.exit(1)

    if femur_arch is None or tibia_arch is None:
        print("  [ERROR] Could not determine architectures. Use --femur_arch / --tibia_arch.")
        sys.exit(1)

    print(f"\n  Ensemble config:")
    print(f"    Femur model : {femur_arch}")
    print(f"    Tibia model : {tibia_arch}")

    # ── Step 2: Load models ────────────────────────────────────────
    print(f"\n  Loading model weights...")
    femur_model = load_model(femur_arch, "femur")
    tibia_model = load_model(tibia_arch, "tibia")

    # ── Step 3: Reproduce identical test split ─────────────────────
    print(f"\n  Reconstructing held-out test set...")

    femur_paths, femur_labels = BoneDataset.from_folder(FEMUR_DIR)
    tibia_paths, tibia_labels = BoneDataset.from_folder(TIBIA_DIR)

    if not femur_paths or not tibia_paths:
        print("  [ERROR] No images found. Check FEMUR_DIR / TIBIA_DIR paths.")
        sys.exit(1)

    femur_te_paths, femur_te_labels = get_test_split(femur_paths, femur_labels)
    tibia_te_paths, tibia_te_labels = get_test_split(tibia_paths, tibia_labels)

    print(f"    Femur test samples : {len(femur_te_paths)}")
    print(f"    Tibia test samples : {len(tibia_te_paths)}")

    tf = get_val_transform()
    pin = DEVICE.type == "cuda"

    femur_te_ds = BoneDataset(femur_te_paths, femur_te_labels, transform=tf)
    tibia_te_ds = BoneDataset(tibia_te_paths, tibia_te_labels, transform=tf)

    femur_te_dl = DataLoader(femur_te_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin)
    tibia_te_dl = DataLoader(tibia_te_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin)

    # ── Step 4: Collect softmax probabilities ──────────────────────
    print(f"\n  Running inference...")
    femur_probs, femur_te_labs = get_probs(femur_model, femur_te_dl)
    tibia_probs, tibia_te_labs = get_probs(tibia_model, tibia_te_dl)

    # Use femur labels as ground truth (must match tibia labels — same patients)
    # Validate consistency
    if not np.array_equal(femur_te_labs, tibia_te_labs):
        print(
            "  ⚠  WARNING: femur and tibia test-set labels do not match.\n"
            "     This can happen if the two folders have different samples.\n"
            "     Falling back to femur labels as ground truth."
        )
    true_labels = femur_te_labs

    # ── Step 5: Compute per-model AUC for weighting ────────────────
    femur_auc_weight = args.femur_auc if args.femur_auc is not None \
        else (compute_auc(true_labels, femur_probs) or 0.5)
    tibia_auc_weight = args.tibia_auc if args.tibia_auc is not None \
        else (compute_auc(true_labels, tibia_probs) or 0.5)

    print(f"\n  AUC weights:")
    print(f"    Femur ({femur_arch}) AUC = {femur_auc_weight:.4f}")
    print(f"    Tibia ({tibia_arch}) AUC = {tibia_auc_weight:.4f}")

    total_w = femur_auc_weight + tibia_auc_weight
    fw      = femur_auc_weight / total_w
    tw      = tibia_auc_weight / total_w
    print(f"    → Normalised weights  :  Femur {fw:.3f}  |  Tibia {tw:.3f}")

    # ── Step 6: Build all ensemble variants ────────────────────────
    # AUC-weighted (primary)
    auc_probs  = fw * femur_probs + tw * tibia_probs
    auc_preds  = np.argmax(auc_probs, axis=1)

    # Equal weight (50/50) baseline
    eq_probs   = 0.5 * femur_probs + 0.5 * tibia_probs
    eq_preds   = np.argmax(eq_probs, axis=1)

    # Femur-only and Tibia-only baselines
    femur_preds = np.argmax(femur_probs, axis=1)
    tibia_preds = np.argmax(tibia_probs, axis=1)

    # ── Step 7: Compute metrics for all variants ───────────────────
    def metrics(preds, probs, labels):
        return {
            "acc":     accuracy_score(labels, preds),
            "bal_acc": balanced_accuracy_score(labels, preds),
            "auc":     compute_auc(labels, probs),
        }

    m_femur = metrics(femur_preds, femur_probs, true_labels)
    m_tibia = metrics(tibia_preds, tibia_probs, true_labels)
    m_eq    = metrics(eq_preds,    eq_probs,    true_labels)
    m_auc   = metrics(auc_preds,   auc_probs,   true_labels)

    comparison_rows = [
        {"label": f"Femur only ({femur_arch})", **m_femur},
        {"label": f"Tibia only ({tibia_arch})", **m_tibia},
        {"label": "Ensemble (50 / 50)",          **m_eq},
        {"label": "Ensemble (AUC-weighted) ★",  **m_auc},
    ]

    best_acc = max(r["acc"] for r in comparison_rows)
    for row in comparison_rows:
        row["best"] = (row["acc"] == best_acc)

    # ── Step 8: Save report, confusion, ROC, comparison chart ─────
    report_path = os.path.join(RESULTS_DIR, "ensemble_report.txt")
    cm, acc, bal_acc = save_report(
        auc_preds, true_labels, auc_probs,
        femur_arch, tibia_arch,
        femur_auc_weight, tibia_auc_weight,
        comparison_rows,
        report_path,
    )

    plot_confusion(
        cm,
        title     = f"Ensemble ({femur_arch} + {tibia_arch}) — AUC-Weighted [Test Set]",
        save_path = os.path.join(RESULTS_DIR, "ensemble_confusion.png"),
    )

    if len(np.unique(true_labels)) >= 2:
        plot_roc(
            true_labels, auc_probs,
            title     = f"Ensemble ({femur_arch} + {tibia_arch}) — AUC-Weighted ROC",
            save_path = os.path.join(RESULTS_DIR, "ensemble_roc.png"),
        )

    plot_comparison_chart(
        comparison_rows,
        save_path = os.path.join(RESULTS_DIR, "ensemble_summary_chart.png"),
    )

    # ── Step 9: Agreement diagnostics ─────────────────────────────
    agree_rate    = np.mean(femur_preds == tibia_preds) * 100
    disagree_mask = femur_preds != tibia_preds
    n_disagree    = disagree_mask.sum()

    if n_disagree > 0:
        correct_on_disagree = (auc_preds[disagree_mask] == true_labels[disagree_mask]).mean() * 100
    else:
        correct_on_disagree = float("nan")

    print(f"\n  ─────────────────────────────────────────────────────")
    print(f"  DIAGNOSTICS")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Model agreement rate          : {agree_rate:.1f}%")
    print(f"  Disagreement samples          : {n_disagree} / {len(true_labels)}")
    if n_disagree > 0:
        print(f"  Ensemble correct on disagree  : {correct_on_disagree:.1f}%")

    print(f"\n{'=' * 65}")
    print(f"  ✅  ENSEMBLE COMPLETE")
    print(f"  Report  → {report_path}")
    print(f"  Outputs → {RESULTS_DIR}/")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()