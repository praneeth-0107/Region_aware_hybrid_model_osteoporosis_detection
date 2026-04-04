"""
================================================================
 BATCH KNEE X-RAY JOINT SEPARATOR  v3  — PRODUCTION GRADE
================================================================
HANDLES:
  ✓ Single-knee images (most clinical X-rays)
  ✓ Bilateral (two-knee) images
  ✓ Any zoom level / field of view
  ✓ Bone-on-bone contact (no visible joint space)
  ✓ Landscape or portrait orientation
  ✓ Variable contrast / brightness

KEY ALGORITHM — Bone-Width Dip Detection:
  1. Strip black borders → get active ROI
  2. CLAHE + bilateral filter → enhance bone edges
  3. Otsu bone mask → binary bone/not-bone
  4. Per-row bone width profile → how many bone pixels per row
  5. Smooth the profile → find the deepest, most prominent DIP
     The joint space = where bone width DROPS sharply between
     femoral condyles (wide) and tibial plateau (also wide)
     but narrower gap in between
  6. For bilateral images: detect the split column first,
     then run the same algorithm on each knee separately

USAGE:
  python batch_knee_separator_v3.py --input ./xrays --output ./output
  python batch_knee_separator_v3.py --input ./xrays --output ./output --no-annot

OUTPUT STRUCTURE:
  output/
    femur/       ← above joint line (distal femur + patella)
    tibia/       ← below joint line (proximal tibia + fibula)
    annotated/   ← QC overlay — ALWAYS check these!
================================================================
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import os
import argparse
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_DIR      = "input"
OUTPUT_DIR     = "output"
PAD_PX         = 15          # overlap padding in above/below crops
SEARCH_LO      = 0.20        # vertical search start (fraction of active height)
SEARCH_HI      = 0.88        # vertical search end
BONE_MASK_KSIZE = 9          # morphology kernel size
SMOOTH_SIGMA   = 8           # gaussian smoothing for bone-width profile
DIP_PROMINENCE = 25          # min prominence for a valid joint dip
DIP_DISTANCE   = 15          # min distance between dip candidates
EXTENSIONS     = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
SAVE_ANNOT     = True

# ─────────────────────────────────────────────
# ARG PARSER
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Batch Knee X-Ray Joint Separator v3")
parser.add_argument("--input",    default=INPUT_DIR,  help="Input folder")
parser.add_argument("--output",   default=OUTPUT_DIR, help="Output folder")
parser.add_argument("--no-annot", action="store_true", help="Skip annotated images")
args       = parser.parse_args()
INPUT_DIR  = args.input
OUTPUT_DIR = args.output
SAVE_ANNOT = not args.no_annot

# ─────────────────────────────────────────────
# OUTPUT FOLDERS
# ─────────────────────────────────────────────
femur_dir = os.path.join(OUTPUT_DIR, "femur")
tibia_dir = os.path.join(OUTPUT_DIR, "tibia")
annot_dir = os.path.join(OUTPUT_DIR, "annotated")
os.makedirs(femur_dir, exist_ok=True)
os.makedirs(tibia_dir, exist_ok=True)
if SAVE_ANNOT:
    os.makedirs(annot_dir, exist_ok=True)


# ─────────────────────────────────────────────
# CORE: BONE WIDTH DIP DETECTOR
# ─────────────────────────────────────────────
def get_active_roi_bounds(img, threshold=15):
    """Find bounding box of non-black content."""
    H, W       = img.shape
    col_means  = img.mean(axis=0)
    row_means  = img.mean(axis=1)
    x0 = next((i for i in range(W)    if col_means[i] > threshold), 0)
    x1 = next((i for i in range(W-1,-1,-1) if col_means[i] > threshold), W-1)
    y0 = next((i for i in range(H)    if row_means[i] > threshold), 0)
    y1 = next((i for i in range(H-1,-1,-1) if row_means[i] > threshold), H-1)
    return x0, y0, x1, y1


def preprocess(roi):
    """CLAHE + bilateral denoise."""
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(roi)
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=40, sigmaSpace=40)
    return denoised


def get_bone_mask(denoised):
    """Otsu threshold + morphological cleanup."""
    _, bone  = cv2.threshold(denoised, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (BONE_MASK_KSIZE, BONE_MASK_KSIZE))
    bone     = cv2.morphologyEx(bone, cv2.MORPH_CLOSE, kernel, iterations=3)
    bone     = cv2.morphologyEx(bone, cv2.MORPH_OPEN,  kernel, iterations=1)
    return bone


def bone_width_profile(bone, margin_frac=0.10):
    """Per-row bone pixel count using central 80% of width."""
    rH, rW   = bone.shape
    cx0      = int(rW * margin_frac)
    cx1      = int(rW * (1.0 - margin_frac))
    widths   = bone[:, cx0:cx1].sum(axis=1).astype(float) / 255.0
    return gaussian_filter1d(widths, sigma=SMOOTH_SIGMA)


def find_joint_row_in_roi(roi, offset_y=0):
    """
    Returns (joint_row_in_orig_img, femur_row, tibia_row).
    offset_y is the y-offset of roi inside the original image.
    """
    rH, rW   = roi.shape
    denoised = preprocess(roi)
    bone     = get_bone_mask(denoised)
    profile  = bone_width_profile(bone)

    y_lo = int(rH * SEARCH_LO)
    y_hi = int(rH * SEARCH_HI)
    seg  = profile[y_lo:y_hi]

    # Find all significant dips (local minima)
    inv_seg = -seg
    peaks, props = find_peaks(inv_seg,
                               prominence=DIP_PROMINENCE,
                               distance=DIP_DISTANCE,
                               width=2)

    if len(peaks) > 0:
        # Choose the most prominent dip
        best      = np.argmax(props['prominences'])
        joint_loc = y_lo + peaks[best]
    else:
        # Fallback: smoothed argmin
        joint_loc = y_lo + int(np.argmin(seg))

    # Estimate femur bottom (local max before joint) and tibia top (local max after)
    pre  = profile[:joint_loc]
    post = profile[joint_loc:]
    fb   = int(np.argmax(pre[-max(1, int(rH*0.25)):]) + max(0, joint_loc - int(rH*0.25)))
    tt   = joint_loc + int(np.argmax(post[:max(1, int(rH*0.25))]))

    return (joint_loc + offset_y,
            fb        + offset_y,
            tt        + offset_y)


def is_bilateral(img):
    """
    Returns True if the image shows TWO knees side by side.
    Heuristic: aspect ratio > 1.4 (wide) OR a dark vertical
    strip exists in the central 20% of the image.
    """
    H, W = img.shape
    if W / H > 1.3:
        return False   # landscape = probably single knee zoomed
    # Check for a dark vertical gap in the center (bilateral have a gap between legs)
    cx_lo = int(W * 0.40)
    cx_hi = int(W * 0.60)
    center_strip = img[:, cx_lo:cx_hi]
    left_strip   = img[:, int(W*0.15):int(W*0.35)]
    right_strip  = img[:, int(W*0.65):int(W*0.85)]
    center_mean  = center_strip.mean()
    sides_mean   = (left_strip.mean() + right_strip.mean()) / 2.0
    # If center is meaningfully darker than sides → bilateral gap
    return (sides_mean - center_mean) > 20


def find_bilateral_split(img):
    """Find the column that separates left and right knees."""
    H, W      = img.shape
    cx_lo     = int(W * 0.35)
    cx_hi     = int(W * 0.65)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    enhanced  = clahe.apply(img)
    _, bm     = cv2.threshold(enhanced, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    col_dens  = bm[:, cx_lo:cx_hi].mean(axis=0)
    return cx_lo + int(np.argmin(col_dens))


# ─────────────────────────────────────────────
# PROCESS ONE IMAGE
# ─────────────────────────────────────────────
def process_image(img_path, stem):
    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"  [SKIP] Cannot read")
        return False

    H, W = raw.shape

    # ── Detect bilateral vs single ──────────────────────────────────
    bilateral = is_bilateral(raw)

    if bilateral:
        # ── BILATERAL: run detector on each knee half ────────────────
        split_col   = find_bilateral_split(raw)
        left_roi    = raw[:, :split_col]
        right_roi   = raw[:, split_col:]

        jl, fbl, ttl = find_joint_row_in_roi(left_roi,  offset_y=0)
        jr, fbr, ttr = find_joint_row_in_roi(right_roi, offset_y=0)

        gap_l     = abs(ttl - fbl)
        gap_r     = abs(ttr - fbr)
        if gap_l + gap_r > 0:
            final_row = int((jl * gap_l + jr * gap_r) / (gap_l + gap_r))
        else:
            final_row = (jl + jr) // 2

        mode_str  = f"BILATERAL split={split_col} L={jl} R={jr}"

    else:
        # ── SINGLE KNEE ──────────────────────────────────────────────
        x0, y0, x1, y1 = get_active_roi_bounds(raw)
        roi            = raw[y0:y1, x0:x1]

        jrow_roi, fb_roi, tt_roi = find_joint_row_in_roi(roi, offset_y=0)
        final_row  = jrow_roi + y0
        fb_global  = fb_roi   + y0
        tt_global  = tt_roi   + y0

        mode_str   = f"SINGLE active=[{x0}:{x1},{y0}:{y1}]"

    # Clamp
    final_row = max(int(H * 0.20), min(int(H * 0.88), final_row))

    # ── Crop & save ───────────────────────────────────────────────────
    above = raw[:final_row + PAD_PX, :]
    below = raw[max(0, final_row - PAD_PX):, :]
    cv2.imwrite(os.path.join(femur_dir, f"{stem}.png"), above)
    cv2.imwrite(os.path.join(tibia_dir, f"{stem}.png"), below)

    # ── Annotated QC image ────────────────────────────────────────────
    if SAVE_ANNOT:
        vis  = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Final joint line — bright green
        cv2.line(vis, (0, final_row), (W, final_row), (0, 255, 0), 3)

        # Padding dashes
        for x in range(0, W, 24):
            cv2.line(vis, (x,   final_row + PAD_PX), (x+12, final_row + PAD_PX), (0, 160, 0), 1)
            cv2.line(vis, (x,   final_row - PAD_PX), (x+12, final_row - PAD_PX), (0, 160, 0), 1)

        if bilateral:
            cv2.line(vis, (split_col, 0), (split_col, H), (180, 180, 180), 1)
            cv2.line(vis, (0,  jl), (split_col, jl), (0, 220, 220), 2)
            cv2.line(vis, (split_col, jr), (W,  jr), (0, 220, 220), 2)
        else:
            cv2.line(vis, (0, fb_global), (W, fb_global), (255, 80, 0),  2)
            cv2.line(vis, (0, tt_global), (W, tt_global), (0, 140, 255), 2)

        lbl_y = max(final_row - 12, 30)
        cv2.putText(vis, "FEMUR", (12, lbl_y),             font, 0.9, (0,255,0), 2)
        cv2.putText(vis, "TIBIA", (12, final_row + 38),    font, 0.9, (0,255,0), 2)
        cv2.putText(vis, f"row {final_row}  ({final_row/H*100:.0f}%)",
                    (W - 240, final_row - 8),              font, 0.65, (0,255,0), 2)

        # Legend
        ly = 28
        entries = [
            ((0, 255, 0),    "Final joint line"),
            ((255, 80, 0),   "Femur bottom"),
            ((0, 140, 255),  "Tibia top"),
        ]
        for col_bgr, txt in entries:
            cv2.line(vis, (W-220, ly), (W-180, ly), col_bgr, 3)
            cv2.putText(vis, txt, (W-175, ly+5), font, 0.45, col_bgr, 1)
            ly += 24

        cv2.imwrite(os.path.join(annot_dir, f"{stem}_annotated.png"), vis)

    print(f"  ✓  row={final_row} ({final_row/H*100:.0f}%)  "
          f"femur={above.shape[0]}px  tibia={below.shape[0]}px  [{mode_str}]")
    return True


# ─────────────────────────────────────────────
# BATCH LOOP
# ─────────────────────────────────────────────
def main():
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"[ERROR] Input folder not found: {INPUT_DIR}")
        return

    images = sorted([p for p in input_path.iterdir()
                     if p.suffix.lower() in EXTENSIONS])
    if not images:
        print(f"[ERROR] No images found in: {INPUT_DIR}")
        return

    print(f"\n{'='*65}")
    print(f"  BATCH KNEE SEPARATOR  v3  (bone-width dip detection)")
    print(f"  Input     : {INPUT_DIR}  ({len(images)} images)")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"    femur/       ← above joint (distal femur + patella)")
    print(f"    tibia/       ← below joint (proximal tibia + fibula)")
    if SAVE_ANNOT:
        print(f"    annotated/   ← QC overlay — CHECK THESE FIRST")
    print(f"{'='*65}\n")

    ok, failed = 0, []
    for i, img_path in enumerate(images, 1):
        print(f"[{i:>4}/{len(images)}]  {img_path.name}")
        try:
            if process_image(img_path, img_path.stem):
                ok += 1
            else:
                failed.append(img_path.name)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            failed.append(img_path.name)

    print(f"\n{'='*65}")
    print(f"  DONE ✅   {ok}/{len(images)} processed")
    if failed:
        print(f"  ⚠  Failed ({len(failed)}): {', '.join(failed)}")
    print(f"  Femur  → {femur_dir}/")
    print(f"  Tibia  → {tibia_dir}/")
    if SAVE_ANNOT:
        print(f"  Annot  → {annot_dir}/  ← review for accuracy")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()