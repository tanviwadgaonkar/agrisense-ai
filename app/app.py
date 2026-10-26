"""AgriSense Flask app - soil analysis + crop recommendation UI and JSON API.

Run from the repo root:  python -m app.app   (then open http://localhost:5000)
Debug mode is off unless FLASK_DEBUG=1 (the Werkzeug debugger executes code).
"""
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from src.predict import recommend
from src.preprocess import FEATURES

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

FIELD_META = {
    "N": ("Nitrogen (N)", "kg/ha", 78),
    "P": ("Phosphorus (P)", "kg/ha", 42),
    "K": ("Potassium (K)", "kg/ha", 40),
    "temperature": ("Temperature", "°C", 24),
    "humidity": ("Humidity", "%", 82),
    "ph": ("Soil pH", "", 6.4),
    "rainfall": ("Rainfall", "mm", 220),
    "soil_moisture": ("Soil moisture", "%", 68),
}


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, values = None, None, {f: FIELD_META[f][2] for f in FEATURES}
    if request.method == "POST":
        values = {f: request.form.get(f, "") for f in FEATURES}
        try:
            result = recommend(values)
        except ValueError as exc:
            error = f"Please check your input: {exc}."
    return render_template("index.html", fields=FIELD_META, values=values,
                           result=result, error=error)


@app.post("/api/recommend")
def api_recommend():
    """JSON in: the 8 features. JSON out: soil report + top-3 crops."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    try:
        return jsonify(recommend(payload))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict-disease")
def predict_disease():
    """Validates a leaf image upload. No disease model is trained yet, so a
    valid image gets 501 rather than a made-up prediction."""
    if "image" not in request.files:
        return jsonify({"error": "image is not found"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "file is empty"}), 400
    try:
        image = Image.open(file.stream)
        if image.format not in ("JPEG", "PNG"):
            return jsonify({"error": "Unsupported Image Format"}), 400
        image.verify()
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Image Invalid"}), 400

    return jsonify({
        "error": "disease detection model not available yet",
        "image": {"format": image.format, "width": image.width, "height": image.height},
    }), 501


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"error": "file larger than 5 MB"}), 413


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
