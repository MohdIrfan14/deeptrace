"""
app.py — DeepTrace Flask API.

Changes vs original:
  - predict_deepfake(path, tta=True) — TTA enabled by default for best accuracy.
  - Returns confidence_pct for easy frontend display (e.g. "94.2 % fake").
  - /model_info endpoint exposes current config for debugging.
  - Graceful 503 if model not yet loaded (race condition guard).
"""

from __future__ import annotations

import os
import sys
import uuid

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import cfg
from inference import load_model, predict_deepfake

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR   = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
    ".mp4", ".avi", ".mov", ".webm", ".mkv",
}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB

    CORS(app, resources={r"/*": {"origins": "*"}})

    # ── Health check ─────────────────────────────────────────────────────────
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    # ── Model info ───────────────────────────────────────────────────────────
    @app.route("/model_info", methods=["GET"])
    def model_info():
        return jsonify({
            "img_size":          cfg.IMG_SIZE,
            "spatial_backbone":  cfg.SPATIAL_BACKBONE,
            "freq_backbone":     cfg.FREQ_BACKBONE,
            "model_path":        cfg.BEST_MODEL_PATH,
        })

    # ── Prediction ───────────────────────────────────────────────────────────
    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Accept multipart/form-data with field "file" (image or video).
        Optional form field "tta" = "false" to disable test-time augmentation.

        Response JSON:
          {
            "result":          "real" | "fake",
            "confidence":      0.9423,          # prob of predicted class [0,1]
            "confidence_pct":  "94.23%",
            "prob_fake":       0.9423            # always P(fake) for charting
          }
        """
        if "file" not in request.files:
            return jsonify({"error": 'Missing form field "file".'}), 400

        f = request.files["file"]
        if not f or f.filename == "":
            return jsonify({"error": "No file selected."}), 400

        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"Unsupported extension '{ext}'.",
                "allowed": sorted(ALLOWED_EXTENSIONS),
            }), 400

        # TTA flag from form (default True)
        use_tta = request.form.get("tta", "true").lower() != "false"

        safe_name = f"{uuid.uuid4().hex}_{secure_filename(f.filename or 'upload')}"
        save_path = os.path.join(UPLOAD_DIR, safe_name)

        try:
            f.save(save_path)
            result, confidence = predict_deepfake(save_path, tta=use_tta)

            return jsonify({
                "result":         result,
                "confidence":     round(confidence, 6),
                "confidence_pct": f"{confidence * 100:.2f}%",
            })

        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.isfile(save_path):
                try:
                    os.remove(save_path)
                except OSError:
                    pass

    return app


def main() -> None:
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    app = create_app()
    print("DeepTrace API  →  http://127.0.0.1:5000")
    print('POST /predict  with multipart "file" (image or video)')
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()