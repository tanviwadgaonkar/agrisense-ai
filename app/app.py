"""AgriSense Flask app - soil analysis + crop recommendation UI.

Run from the repo root:  python -m app.app   (then open http://localhost:5000)
"""
from flask import Flask, render_template, request
from PIL import Image,UnidentifiedImageError
from src.predict import recommend
from src.preprocess import FEATURES
from flask import jsonify
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"]=5*1024*1024

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
        try:
            values = {f: float(request.form[f]) for f in FEATURES}
            result = recommend(values)
        except (KeyError, ValueError):
            error = "Please enter a valid number for every field."
    return render_template("index.html", fields=FIELD_META, values=values,
                           result=result, error=error)

@app.route("/predict-disease",methods=["POST"])
def predict_disease():
    if "image" not in request.files:
        return jsonify({"error":"image is not found"}),400
    file=request.files["image"]
    if file.filename=='':
            return jsonify({"error":"file is empty"}),400
    try:
        image=Image.open(file.stream)
        if image.format not in ("JPEG","PNG"):
            return jsonify({"error":"Unsupported Image Format"}),400
        image.verify()
    except(UnidentifiedImageError,OSError):
        return jsonify({"error":"Image Invalid"}),400
    
    return jsonify({"message":"File found"}),200
if __name__ == "__main__":
    app.run(debug=True)
