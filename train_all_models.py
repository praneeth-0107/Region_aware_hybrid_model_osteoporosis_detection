"""
=============================================================
  BONE DENSITY — 6 MODEL TRAINING PIPELINE
  Trains 6 architectures separately on Femur and Tibia data
  using a 80% Train / 10% Val / 10% Test stratified split.

  MODELS:
      1. ResNet50
      2. DenseNet121
      3. VGG16
      4. EfficientNet-B3
      5. EfficientNet-V2-S
      6. Swin-T

  CLASSES:
      osteoporosis (0) | osteopenia (1) | normal (2)

  LABEL FORMAT (class name must appear in filename):
      patient01_osteoporosis.png
      scan_normal_003.png
      case_osteopenia_42.png

  OUTPUTS:
      models/<arch>_femur.pth          saved model per architecture
      models/<arch>_tibia.pth
      results/<arch>_femur_report.txt  per-model test set report
      results/<arch>_tibia_report.txt
      results/<arch>_femur_confusion.png
      results/<arch>_tibia_confusion.png
      results/<arch>_femur_curves.png
      results/<arch>_tibia_curves.png
      results/final_summary.txt        ranked comparison of all 12 runs

  SKIP LOGIC:
      If models/<arch>_<site>.pth already exists → skip training,
      load saved weights and evaluate on test set directly.
      Delete the .pth file to force retraining that model.

  REQUIREMENTS:
      pip install torch torchvision scikit-learn
                  matplotlib seaborn pillow numpy

  USAGE:
      python train_all_models.py
=============================================================
"""

import os
import re
import sys
import copy
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
# CONFIGURATION
# ──────────────────────────────────────────────
FEMUR_DIR   = "output/femur"
TIBIA_DIR   = "output/tibia"
MODELS_DIR  = "models"
RESULTS_DIR = "results"

IMG_SIZE    = 224
BATCH_SIZE  = 8
EPOCHS      = 25
LR          = 1e-4
WD          = 1e-4
SEED        = 42
NUM_WORKERS = 0
USE_AMP     = torch.cuda.is_available()
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

CLASSES = ["osteoporosis", "osteopenia", "normal"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

# All 6 architectures to train
ARCHITECTURES = [
    "resnet50",
    "densenet121",
    "vgg16",
    "efficientnet_b3",
    "efficientnet_v2_s",
    "swin_t",
]

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device  : {DEVICE}")
print(f"AMP     : {USE_AMP}")
print(f"Classes : {CLASSES}")
print(f"Split   : Train {TRAIN_RATIO:.0%} / Val {VAL_RATIO:.0%} / Test {TEST_RATIO:.0%}")


# ──────────────────────────────────────────────
# DATASET
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
# DATASET SPLIT  80 / 10 / 10
# ──────────────────────────────────────────────
def split_dataset(paths, labels):
    labels_np = np.array(labels)

    counts = np.bincount(labels_np, minlength=len(CLASSES))
    for i, cls in enumerate(CLASSES):
        if counts[i] < 3:
            raise ValueError(
                f"Class '{cls}' has only {counts[i]} samples. "
                f"Need at least 3 for a 3-way stratified split."
            )

    tr_val_idx, te_idx = train_test_split(
        np.arange(len(paths)),
        test_size    = TEST_RATIO,
        stratify     = labels_np,
        random_state = SEED,
    )

    val_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    tr_idx, val_idx = train_test_split(
        tr_val_idx,
        test_size    = val_adjusted,
        stratify     = labels_np[tr_val_idx],
        random_state = SEED,
    )

    def gather(idx_list):
        return [paths[i] for i in idx_list], [labels[i] for i in idx_list]

    tr_data  = gather(tr_idx)
    val_data = gather(val_idx)
    te_data  = gather(te_idx)

    print(f"\n  Split summary:")
    print(f"    Train : {len(tr_data[0]):>5}  ({len(tr_data[0])/len(paths)*100:.1f}%)")
    print(f"    Val   : {len(val_data[0]):>5}  ({len(val_data[0])/len(paths)*100:.1f}%)")
    print(f"    Test  : {len(te_data[0]):>5}  ({len(te_data[0])/len(paths)*100:.1f}%)  ← held-out")

    for split_name, (_, split_labels) in [("Train", tr_data),
                                           ("Val",   val_data),
                                           ("Test",  te_data)]:
        c = np.bincount(split_labels, minlength=len(CLASSES))
        print(f"    {split_name} classes: " +
              " | ".join(f"{CLASSES[i]}:{c[i]}" for i in range(len(CLASSES))))

    return tr_data, val_data, te_data


# ──────────────────────────────────────────────
# TRANSFORMS
# ──────────────────────────────────────────────
def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.4),
        transforms.RandomAffine(degrees=0, shear=8),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.10)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# ──────────────────────────────────────────────
# MODEL BUILDER
# ──────────────────────────────────────────────
def build_model(arch, num_classes=3, pretrained=True):

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
        m = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = custom_head(m.fc.in_features)

    elif arch == "densenet121":
        m = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        m.classifier = custom_head(m.classifier.in_features)

    elif arch == "vgg16":
        m = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        m.classifier = custom_head(m.classifier[0].in_features)

    elif arch == "efficientnet_b3":
        m = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        m.classifier = custom_head(m.classifier[1].in_features)

    elif arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        m.classifier = custom_head(m.classifier[1].in_features)

    elif arch == "swin_t":
        m = models.swin_t(
            weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
        m.head = custom_head(m.head.in_features)

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return m


# ──────────────────────────────────────────────
# WEIGHTED SAMPLER
# ──────────────────────────────────────────────
def make_sampler(labels):
    counts   = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    counts   = np.where(counts == 0, 1.0, counts)
    w_class  = 1.0 / counts
    w_sample = np.array([w_class[l] for l in labels], dtype=np.float32)
    return WeightedRandomSampler(
        weights     = torch.from_numpy(w_sample),
        num_samples = len(labels),
        replacement = True,
    )


# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = correct = total = 0
    for imgs, labs in loader:
        imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(
                device_type="cuda" if USE_AMP else "cpu", enabled=USE_AMP):
            out  = model(imgs)
            loss = criterion(out, labs)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        correct      += (out.argmax(1) == labs).sum().item()
        total        += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    running_loss = correct = total = 0
    all_preds, all_labs, all_probs = [], [], []
    for imgs, labs in loader:
        imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
        with torch.amp.autocast(
                device_type="cuda" if USE_AMP else "cpu", enabled=USE_AMP):
            out = model(imgs)
            if criterion is not None:
                running_loss += criterion(out, labs).item() * imgs.size(0)
        probs  = torch.softmax(out, dim=1)
        preds  = out.argmax(1)
        correct += (preds == labs).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labs.extend(labs.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    loss = running_loss / total if (criterion is not None and total > 0) else 0.0
    return (loss, correct / total,
            np.array(all_preds), np.array(all_labs), np.array(all_probs))


# ──────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────
def plot_curves(history, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.spines[:].set_color("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#f0f6fc")

    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], color="#58a6ff", lw=2, label="Train")
    axes[0].plot(ep, history["val_loss"],   color="#f78166", lw=2, label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(facecolor="#1c2128", edgecolor="#30363d", labelcolor="#c9d1d9")

    axes[1].plot(ep, history["train_acc"], color="#56d364", lw=2, label="Train")
    axes[1].plot(ep, history["val_acc"],   color="#e3b341", lw=2, label="Val")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(facecolor="#1c2128", edgecolor="#30363d", labelcolor="#c9d1d9")

    fig.suptitle(title, color="#f0f6fc", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    📈 Curves → {save_path}")


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
    print(f"    🗂  Confusion → {save_path}")


def plot_roc(labels, probs, title, save_path):
    colors = ["#58a6ff", "#f78166", "#56d364"]
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.spines[:].set_color("#30363d")
    ax.tick_params(colors="#8b949e")

    labels_bin  = np.eye(len(CLASSES))[labels]
    macro_sum   = 0.0
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc     = sk_auc(fpr, tpr)
        macro_sum  += roc_auc
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{cls} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#30363d", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate", color="#c9d1d9")
    ax.set_ylabel("True Positive Rate",  color="#c9d1d9")
    ax.set_title(f"{title}\nMacro AUC = {macro_sum/len(CLASSES):.4f}",
                 color="#f0f6fc", fontweight="bold")
    ax.legend(facecolor="#1c2128", edgecolor="#30363d", labelcolor="#c9d1d9")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    📉 ROC     → {save_path}")


# ──────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────
def save_report(preds, labels, probs, arch, site, save_path):
    acc     = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    cm      = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    report  = classification_report(
        labels, preds, target_names=CLASSES, zero_division=0)

    per_class_acc = {}
    for i, cls in enumerate(CLASSES):
        mask = labels == i
        per_class_acc[cls] = float(
            (preds[mask] == i).mean()) if mask.sum() > 0 else float("nan")

    try:
        macro_auc = roc_auc_score(
            np.eye(len(CLASSES))[labels], probs,
            multi_class="ovr", average="macro")
        auc_str = f"{macro_auc:.4f}"
    except Exception as e:
        auc_str   = f"N/A ({e})"
        macro_auc = 0.0

    lines = [
        "=" * 60,
        f"  ARCH : {arch.upper()}  |  SITE : {site.upper()}",
        f"  Evaluated on HELD-OUT TEST SET ({int(TEST_RATIO*100)}%)",
        "=" * 60,
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
        "=" * 60,
    ]

    text = "\n".join(lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"    📄 Report  → {save_path}")
    return cm, acc, bal_acc, macro_auc


# ──────────────────────────────────────────────
# TRAIN ONE ARCHITECTURE ON ONE SITE
# ──────────────────────────────────────────────
def train_one_model(arch, site, tr_data, val_data, te_data):
    """
    Full training + evaluation pipeline for one architecture on one site.

    SKIP LOGIC:
        If models/<arch>_<site>.pth already exists → skip training entirely,
        load the saved weights, evaluate on test set, and return results.
        Delete the .pth file to force retraining.
    """
    tr_paths,  tr_labels  = tr_data
    val_paths, val_labels = val_data
    te_paths,  te_labels  = te_data

    model_path = os.path.join(MODELS_DIR, f"{arch}_{site}.pth")
    _, val_tf  = get_transforms()
    pin        = DEVICE.type == "cuda"

    # Build test loader — always needed whether training or loading
    te_ds = BoneDataset(te_paths, te_labels, transform=val_tf)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=pin)

    # ── SKIP: model already trained ─────────────────────────────
    if os.path.exists(model_path):
        print(f"\n  {'─'*60}")
        print(f"  ⏭  SKIPPING  : {arch.upper()}  |  SITE : {site.upper()}")
        print(f"     Saved model found → {model_path}")
        print(f"     Loading weights and evaluating on test set...")
        print(f"     To retrain: delete {model_path}")
        print(f"  {'─'*60}")

        try:
            ckpt  = torch.load(model_path, map_location=DEVICE, weights_only=False)
            model = build_model(arch, num_classes=len(CLASSES), pretrained=False).to(DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            _, _, te_preds, te_labs, te_probs = evaluate(model, te_dl)

            report_path = os.path.join(RESULTS_DIR, f"{arch}_{site}_report.txt")
            cm, acc, bal_acc, macro_auc = save_report(
                te_preds, te_labs, te_probs, arch, site, report_path)

            plot_confusion(
                cm,
                title     = f"{arch} ({site}) — Confusion Matrix [Test Set]",
                save_path = os.path.join(RESULTS_DIR, f"{arch}_{site}_confusion.png"),
            )

            if len(np.unique(te_labs)) >= 2:
                plot_roc(
                    te_labs, te_probs,
                    title     = f"{arch} ({site}) — ROC Curves [Test Set]",
                    save_path = os.path.join(RESULTS_DIR, f"{arch}_{site}_roc.png"),
                )

            print(f"\n  ✅ {arch} ({site}) [LOADED]: "
                  f"Acc={acc*100:.2f}%  BalAcc={bal_acc*100:.2f}%  "
                  f"AUC={macro_auc:.4f}")

            return {
                "arch":       arch,
                "site":       site,
                "accuracy":   acc * 100,
                "bal_acc":    bal_acc * 100,
                "auc":        macro_auc,
                "time_min":   0.0,
                "model_path": model_path,
            }

        except Exception as e:
            print(f"  [ERROR] Could not load saved model: {e}")
            print(f"  Falling back to retraining from scratch...")

    # ── TRAIN FROM SCRATCH ───────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  TRAINING : {arch.upper()}  |  SITE : {site.upper()}")
    print(f"  Train:{len(tr_paths)}  Val:{len(val_paths)}  Test:{len(te_paths)}")
    print(f"  {'─'*60}")

    train_tf, _ = get_transforms()
    tr_ds = BoneDataset(tr_paths,  tr_labels,  transform=train_tf)
    vl_ds = BoneDataset(val_paths, val_labels, transform=val_tf)

    sampler = make_sampler(tr_labels)

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler,
                       num_workers=NUM_WORKERS, pin_memory=pin)
    vl_dl = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=pin)

    try:
        model = build_model(arch, num_classes=len(CLASSES), pretrained=True).to(DEVICE)
    except Exception as e:
        print(f"  [ERROR] Could not build {arch}: {e}")
        return None

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler(enabled=USE_AMP)

    history    = {"train_loss": [], "val_loss": [],
                  "train_acc":  [], "val_acc":  []}
    best_val   = float("inf")
    best_state = None
    t_start    = time.time()

    print(f"\n  {'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'ValLoss':>9}  {'ValAcc':>7}  {'LR':>10}")
    print(f"  {'─'*54}")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc     = train_one_epoch(
            model, tr_dl, criterion, optimizer, scaler)
        vl_loss, vl_acc, *_ = evaluate(model, vl_dl, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        lr_now = scheduler.get_last_lr()[0]
        flag   = " ✓" if vl_loss < best_val else ""
        print(f"  {epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>6.2%}  "
              f"{vl_loss:>9.4f}  {vl_acc:>6.2%}  {lr_now:>10.2e}{flag}")

        if vl_loss < best_val:
            best_val   = vl_loss
            best_state = copy.deepcopy(model.state_dict())

    elapsed = (time.time() - t_start) / 60

    if best_state is None:
        print(f"  [ERROR] No valid checkpoint for {arch} {site}")
        return None

    # Restore best weights (lowest val loss across all 25 epochs)
    model.load_state_dict(best_state)

    # Save model
    torch.save({
        "model_state_dict": best_state,
        "arch":             arch,
        "site":             site,
        "classes":          CLASSES,
        "img_size":         IMG_SIZE,
        "val_loss":         best_val,
    }, model_path)
    print(f"\n  💾 Model saved → {model_path}")

    # Evaluate on TEST SET
    print(f"\n  Evaluating on held-out test set...")
    _, _, te_preds, te_labs, te_probs = evaluate(model, te_dl)

    report_path = os.path.join(RESULTS_DIR, f"{arch}_{site}_report.txt")
    cm, acc, bal_acc, macro_auc = save_report(
        te_preds, te_labs, te_probs, arch, site, report_path)

    plot_confusion(
        cm,
        title     = f"{arch} ({site}) — Confusion Matrix [Test Set]",
        save_path = os.path.join(RESULTS_DIR, f"{arch}_{site}_confusion.png"),
    )

    plot_curves(
        history,
        title     = f"{arch} ({site}) — Training Curves",
        save_path = os.path.join(RESULTS_DIR, f"{arch}_{site}_curves.png"),
    )

    if len(np.unique(te_labs)) >= 2:
        plot_roc(
            te_labs, te_probs,
            title     = f"{arch} ({site}) — ROC Curves [Test Set]",
            save_path = os.path.join(RESULTS_DIR, f"{arch}_{site}_roc.png"),
        )

    print(f"\n  ✅ {arch} ({site}): "
          f"Acc={acc*100:.2f}%  BalAcc={bal_acc*100:.2f}%  "
          f"AUC={macro_auc:.4f}  Time={elapsed:.1f}min")

    return {
        "arch":      arch,
        "site":      site,
        "accuracy":  acc * 100,
        "bal_acc":   bal_acc * 100,
        "auc":       macro_auc,
        "time_min":  elapsed,
        "model_path": model_path,
    }


# ──────────────────────────────────────────────
# FINAL SUMMARY REPORT
# ──────────────────────────────────────────────
def save_final_summary(all_results):
    femur_res = [r for r in all_results if r and r["site"] == "femur"]
    tibia_res = [r for r in all_results if r and r["site"] == "tibia"]

    femur_sorted = sorted(femur_res, key=lambda x: x["accuracy"], reverse=True)
    tibia_sorted = sorted(tibia_res, key=lambda x: x["accuracy"], reverse=True)

    lines = [
        "=" * 72,
        "  FINAL SUMMARY — ALL MODELS — TEST SET RESULTS",
        "=" * 72,
        "",
        f"  {'─'*68}",
        f"  FEMUR RESULTS (ranked by accuracy)",
        f"  {'─'*68}",
        f"  {'Rank':<5} {'Model':<22} {'Accuracy':>10} {'Bal Acc':>10} "
        f"{'AUC':>8} {'Time':>8}",
        f"  {'─'*68}",
    ]
    for rank, r in enumerate(femur_sorted, 1):
        tag = "  ← BEST" if rank == 1 else ""
        lines.append(
            f"  {rank:<5} {r['arch']:<22} {r['accuracy']:>9.2f}% "
            f"{r['bal_acc']:>9.2f}% {r['auc']:>8.4f} "
            f"{r['time_min']:>6.1f}m{tag}"
        )

    lines += [
        "",
        f"  {'─'*68}",
        f"  TIBIA RESULTS (ranked by accuracy)",
        f"  {'─'*68}",
        f"  {'Rank':<5} {'Model':<22} {'Accuracy':>10} {'Bal Acc':>10} "
        f"{'AUC':>8} {'Time':>8}",
        f"  {'─'*68}",
    ]
    for rank, r in enumerate(tibia_sorted, 1):
        tag = "  ← BEST" if rank == 1 else ""
        lines.append(
            f"  {rank:<5} {r['arch']:<22} {r['accuracy']:>9.2f}% "
            f"{r['bal_acc']:>9.2f}% {r['auc']:>8.4f} "
            f"{r['time_min']:>6.1f}m{tag}"
        )

    best_femur = femur_sorted[0] if femur_sorted else None
    best_tibia = tibia_sorted[0] if tibia_sorted else None

    lines += [
        "",
        "=" * 72,
        "  BEST MODEL RECOMMENDATION FOR ENSEMBLE",
        "=" * 72,
    ]

    if best_femur:
        lines += [
            f"  Best Femur Model : {best_femur['arch'].upper()}",
            f"    Accuracy       : {best_femur['accuracy']:.2f}%",
            f"    Balanced Acc   : {best_femur['bal_acc']:.2f}%",
            f"    ROC-AUC        : {best_femur['auc']:.4f}",
            f"    Saved at       : {best_femur['model_path']}",
        ]
    if best_tibia:
        lines += [
            "",
            f"  Best Tibia Model : {best_tibia['arch'].upper()}",
            f"    Accuracy       : {best_tibia['accuracy']:.2f}%",
            f"    Balanced Acc   : {best_tibia['bal_acc']:.2f}%",
            f"    ROC-AUC        : {best_tibia['auc']:.4f}",
            f"    Saved at       : {best_tibia['model_path']}",
        ]

    lines += [
        "",
        "  Next step:",
        "    Use these two best models to build your ensemble.",
        "    Combine their softmax probabilities (weighted by AUC)",
        "    and evaluate on the same held-out test set.",
        "=" * 72,
    ]

    text = "\n".join(lines)
    print("\n" + text)

    save_path = os.path.join(RESULTS_DIR, "final_summary.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  📄 Final summary → {save_path}")

    return best_femur, best_tibia


# ──────────────────────────────────────────────
# COMPARISON BAR CHART
# ──────────────────────────────────────────────
def plot_summary_chart(all_results):
    femur_res = {r["arch"]: r for r in all_results if r and r["site"] == "femur"}
    tibia_res = {r["arch"]: r for r in all_results if r and r["site"] == "tibia"}

    archs     = ARCHITECTURES
    femur_acc = [femur_res.get(a, {}).get("accuracy", 0) for a in archs]
    tibia_acc = [tibia_res.get(a, {}).get("accuracy", 0) for a in archs]
    femur_auc = [femur_res.get(a, {}).get("auc", 0) * 100 for a in archs]
    tibia_auc = [tibia_res.get(a, {}).get("auc", 0) * 100 for a in archs]

    n     = len(archs)
    y     = np.arange(n)
    width = 0.20

    fig, ax = plt.subplots(figsize=(14, max(8, n * 1.2)))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    b1 = ax.barh(y + width * 1.5, femur_acc, width,
                 label="Femur Accuracy",  color="#58a6ff", alpha=0.9)
    b2 = ax.barh(y + width * 0.5, tibia_acc, width,
                 label="Tibia Accuracy",  color="#56d364", alpha=0.9)
    b3 = ax.barh(y - width * 0.5, femur_auc, width,
                 label="Femur AUC×100",   color="#e3b341", alpha=0.9)
    b4 = ax.barh(y - width * 1.5, tibia_auc, width,
                 label="Tibia AUC×100",   color="#f78166", alpha=0.9)

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            w = bar.get_width()
            if w > 0:
                ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{w:.1f}", va="center", ha="left",
                        color="#c9d1d9", fontsize=7)

    femur_best_idx = np.argmax(femur_acc)
    tibia_best_idx = np.argmax(tibia_acc)
    ax.axhspan(femur_best_idx - 0.5, femur_best_idx + 0.5,
               color="#388bfd", alpha=0.12, label="Best Femur")
    if tibia_best_idx != femur_best_idx:
        ax.axhspan(tibia_best_idx - 0.5, tibia_best_idx + 0.5,
                   color="#3fb950", alpha=0.12, label="Best Tibia")

    ax.set_yticks(y)
    ax.set_yticklabels(archs, color="#c9d1d9", fontsize=10)
    ax.set_xlabel("Score (%)", color="#c9d1d9")
    ax.set_title(
        "All Models — Femur & Tibia — Accuracy and AUC (Test Set)",
        color="#f0f6fc", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    ax.spines[:].set_color("#30363d")
    ax.set_xlim(0, 112)
    ax.legend(facecolor="#1c2128", edgecolor="#30363d",
              labelcolor="#c9d1d9", loc="lower right", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "summary_chart.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Summary chart → {save_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print("  BONE DENSITY — 6 MODEL TRAINING PIPELINE")
    print(f"  Device     : {DEVICE}  |  AMP : {USE_AMP}")
    print(f"  Split      : Train {TRAIN_RATIO:.0%} / Val {VAL_RATIO:.0%} / Test {TEST_RATIO:.0%}")
    print(f"  Epochs     : {EPOCHS}  (no early stopping — full run)")
    print(f"  Batch size : {BATCH_SIZE}  |  IMG size : {IMG_SIZE}")
    print(f"  Models     : {ARCHITECTURES}")
    print(f"  Skip logic : Already trained .pth files will be loaded, not retrained.")
    print(f"               Delete a .pth file from models/ to force retraining it.")
    print(f"{'='*65}\n")

    if DEVICE.type == "cpu":
        est = len(ARCHITECTURES) * 2 * EPOCHS * 8 / 60
        print(f"  ⚠  CPU detected.")
        print(f"     Estimated total time : {est:.0f}–{est*1.5:.0f} hours")
        print(f"     Tip: reduce IMG_SIZE=112 and EPOCHS=10 for faster run\n")

    all_results = []

    for site, data_dir in [("femur", FEMUR_DIR), ("tibia", TIBIA_DIR)]:

        print(f"\n{'='*65}")
        print(f"  SITE : {site.upper()}  |  DIR : {data_dir}")
        print(f"{'='*65}")

        paths, labels = BoneDataset.from_folder(data_dir)
        if not paths:
            print(f"  [SKIP] No data found in {data_dir}")
            continue

        print(f"\n  Total samples : {len(paths)}")
        counts = np.bincount(np.array(labels), minlength=len(CLASSES))
        for i, cls in enumerate(CLASSES):
            print(f"    {cls:<18}: {counts[i]}")

        try:
            tr_data, val_data, te_data = split_dataset(paths, labels)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            continue

        for arch in ARCHITECTURES:
            result = train_one_model(arch, site, tr_data, val_data, te_data)
            if result:
                all_results.append(result)

    if all_results:
        print(f"\n{'='*65}")
        print("  ALL TRAINING COMPLETE — GENERATING SUMMARY")
        print(f"{'='*65}")
        best_femur, best_tibia = save_final_summary(all_results)
        plot_summary_chart(all_results)
    else:
        print("\n  [ERROR] No models trained successfully.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print("  ✅  PIPELINE COMPLETE")
    print(f"  Models   → {MODELS_DIR}/")
    print(f"  Reports  → {RESULTS_DIR}/")
    print(f"  Summary  → {RESULTS_DIR}/final_summary.txt")
    print(f"  Chart    → {RESULTS_DIR}/summary_chart.png")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main() 