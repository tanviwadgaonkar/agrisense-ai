"""Rule-based soil condition classification.

The ML model recommends a crop; these agronomy rules classify the soil itself
(pH class, nutrient levels, moisture state) so the app can explain *why* a
soil is or is not in good condition. Thresholds follow common soil-testing
guidance for Indian field soils.
"""

def classify_ph(ph: float) -> str:
    if ph < 5.5:
        return "strongly acidic"
    if ph < 6.5:
        return "slightly acidic"
    if ph <= 7.5:
        return "neutral"
    return "alkaline"


def classify_nutrient(name: str, value: float) -> str:
    thresholds = {"N": (50, 100), "P": (30, 60), "K": (25, 60)}
    low, high = thresholds[name]
    if value < low:
        return "low"
    if value <= high:
        return "adequate"
    return "high"


def classify_moisture(moisture: float) -> str:
    if moisture < 30:
        return "dry"
    if moisture <= 60:
        return "moist"
    return "wet"


def soil_report(sample: dict) -> dict:
    """Full soil-condition report for one sample (dict of feature -> value)."""
    report = {
        "ph_class": classify_ph(sample["ph"]),
        "nitrogen": classify_nutrient("N", sample["N"]),
        "phosphorus": classify_nutrient("P", sample["P"]),
        "potassium": classify_nutrient("K", sample["K"]),
        "moisture_state": classify_moisture(sample["soil_moisture"]),
    }
    issues = []
    if report["ph_class"] in ("strongly acidic", "alkaline"):
        issues.append(f"pH is {report['ph_class']} - consider amendment (lime/gypsum)")
    for nutrient, key in (("N", "nitrogen"), ("P", "phosphorus"), ("K", "potassium")):
        if report[key] == "low":
            issues.append(f"{nutrient} is low - fertilizer recommended")
    if report["moisture_state"] == "dry":
        issues.append("soil moisture is low - irrigation recommended")
    report["issues"] = issues
    report["overall"] = "good" if not issues else "needs attention"
    return report
