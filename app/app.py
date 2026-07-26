"""AgriSense Flask app - soil analysis + crop recommendation UI.

Run from the repo root:  python -m app.app   (then open http://localhost:5000)
"""
from flask import Flask, render_template, request

from src.predict import recommend
from src.preprocess import FEATURES

app = Flask(__name__)

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


if __name__ == "__main__":
    app.run(debug=True)
