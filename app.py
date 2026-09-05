from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / "house_price_model.sav", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)
NUMERIC_FEATURES = ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]
CATEGORICAL_FEATURES = ["City", "House direction", "Balcony direction", "Legal status", "Furniture state"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
PAGE = """<!doctype html>
<title>Vietnam House Price</title>
<h1>Vietnam House Price Prediction</h1>
<form method="post">
{% for feature in features %}<label>{{ feature }}<br><input name="{{ feature }}" type="text" required></label><br><br>{% endfor %}
<button type="submit">Predict price</button>
</form>
{% if result %}<h2>{{ result }}</h2>{% endif %}
"""


def predict(payload):
    values = {}
    for feature in NUMERIC_FEATURES:
        values[feature] = float(payload[feature])
    for feature in CATEGORICAL_FEATURES:
        values[feature] = str(payload[feature])
    sample = pd.DataFrame([values], columns=FEATURES)
    predicted_price = float(model.predict(sample)[0])
    return {"predicted_price": round(predicted_price, 3), "unit": "billion VND"}


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        try:
            result = f"Estimated price: {predict(request.form)['predicted_price']} billion VND"
        except (KeyError, TypeError, ValueError):
            result = "Please enter valid values for every field."
    return render_template_string(PAGE, features=FEATURES, result=result)


@app.post("/predict")
def api_predict():
    try:
        return jsonify(predict(request.get_json(force=True)))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "JSON must contain all housing feature names."}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5002)
