"""
Flask Backend — Bone Health AI Diagnostic Platform
"""

import os, sys, json, uuid, base64, traceback, io
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

PARENT_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PARENT_DIR)

from predict_bone_denisty import (
    segment_xray, build_model, load_model, predict_single,
    auc_weighted_ensemble, CLASSES, RISK_TIERS, DEVICE,
)

app = Flask(__name__, static_folder="static")
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")
MODELS_DIR = os.path.join(PARENT_DIR, "models")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("\n Loading AI models...")
try:
    FEMUR_MODEL = load_model("efficientnet_v2_s", "femur", MODELS_DIR)
    TIBIA_MODEL = load_model("efficientnet_v2_s", "tibia", MODELS_DIR)
    print(" Models loaded successfully!\n")
except Exception as e:
    print(f" Model loading failed: {e}")
    FEMUR_MODEL = None
    TIBIA_MODEL = None

def img_to_b64(path):
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "models_loaded": FEMUR_MODEL is not None})

@app.route("/api/predict", methods=["POST"])
def predict():
    if FEMUR_MODEL is None:
        return jsonify({"error": "Models not loaded."}), 500
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    run_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or ".png"
    upload_path = os.path.join(UPLOAD_FOLDER, f"{run_id}{ext}")
    file.save(upload_path)
    run_dir = os.path.join(RESULTS_FOLDER, run_id)
    os.makedirs(run_dir, exist_ok=True)

    try:
        import cv2, numpy as np
        femur_crop, tibia_crop, annotated, joint_row, meta = segment_xray(upload_path)
        femur_path = os.path.join(run_dir, "femur_crop.png")
        tibia_path = os.path.join(run_dir, "tibia_crop.png")
        annot_path = os.path.join(run_dir, "annotated.png")
        cv2.imwrite(femur_path, femur_crop)
        cv2.imwrite(tibia_path, tibia_crop)
        cv2.imwrite(annot_path, annotated)

        femur_probs, femur_cls, femur_lbl = predict_single(FEMUR_MODEL, femur_crop)
        tibia_probs, tibia_cls, tibia_lbl = predict_single(TIBIA_MODEL, tibia_crop)
        ens_probs, ens_cls, ens_lbl, w_f, w_t = auc_weighted_ensemble(femur_probs, tibia_probs, 0.5, 0.5)

        def pd(probs):
            return {c: round(float(probs[i])*100, 2) for i, c in enumerate(CLASSES)}

        tier, _, color = RISK_TIERS[ens_lbl]
        ft, _, fc = RISK_TIERS[femur_lbl]
        tt, _, tc = RISK_TIERS[tibia_lbl]

        return jsonify({
            "run_id": run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "images": {
                "original": img_to_b64(upload_path),
                "annotated": img_to_b64(annot_path),
                "femur_crop": img_to_b64(femur_path),
                "tibia_crop": img_to_b64(tibia_path),
            },
            "femur": {"prediction": femur_lbl, "confidence": round(float(femur_probs[femur_cls])*100,2),
                      "probabilities": pd(femur_probs), "risk_level": ft, "risk_color": fc},
            "tibia": {"prediction": tibia_lbl, "confidence": round(float(tibia_probs[tibia_cls])*100,2),
                      "probabilities": pd(tibia_probs), "risk_level": tt, "risk_color": tc},
            "ensemble": {"prediction": ens_lbl, "confidence": round(float(ens_probs[ens_cls])*100,2),
                         "probabilities": pd(ens_probs), "risk_level": tier, "risk_color": color,
                         "femur_weight": round(w_f*100,1), "tibia_weight": round(w_t*100,1)},
            "segmentation": {"mode": "Bilateral" if meta.get("bilateral") else "Single Knee",
                             "joint_row": meta.get("joint_row",0), "image_height": meta.get("H",0),
                             "image_width": meta.get("W",0),
                             "joint_position_pct": round(meta.get("joint_row",0)/max(meta.get("H",1),1)*100,1)},
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Bone Health AI Diagnostic Platform")
    print(f"  Device: {DEVICE}")
    print("  Open http://localhost:5000")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
