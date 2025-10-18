"""
Flask app to serve MNIST classification model.
Loads model.pkl and provides a simple web + API interface for predictions.
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os
import logging

# ----------------------------- Logging -----------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------- Flask App -----------------------------
app = Flask(__name__)

# ----------------------------- Load Model -----------------------------
MODEL_PATH = "model/model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
else:
    logging.error("Model file not found. Please run DVC pipeline first.")

# ----------------------------- Routes -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a list of pixel values via form or JSON, predicts the class.
    """
    try:
        if request.is_json:
            data = request.get_json()
            features = np.array(data["features"]).reshape(1, -1)
        else:
            # From form input
            features = [float(x) for x in request.form.values()]
            features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)
        logging.info(f"Prediction successful: {prediction.tolist()}")
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": str(e)}), 400

# ----------------------------- Run App -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
