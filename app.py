import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)


# ============================================================
# Load CHD Model
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chd.pickle")

try:
    with open(MODEL_PATH, "rb") as file:
        app.model = pickle.load(file)

    print("CHD model loaded successfully!")
    print("Model path:", MODEL_PATH)

except Exception as e:
    app.model = None
    print("ERROR: Could not load CHD model")
    print("Error:", str(e))


# ============================================================
# Home Route
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "CHD Prediction API is running",
        "model": "chd.pickle",
        "model_status": (
            "loaded"
            if app.model is not None
            else "not loaded"
        )
    })


# ============================================================
# Prediction Route
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check whether model is loaded
        if app.model is None:
            return jsonify({
                "error": "CHD model is not loaded"
            }), 500

        # Get JSON input
        data = request.get_json(force=True)

        if data is None:
            return jsonify({
                "error": "No JSON body received"
            }), 400

        print("Received data:", data)

        # Convert JSON to DataFrame
        chd_df = pd.DataFrame([data])

        print("Input DataFrame:")
        print(chd_df)

        # ====================================================
        # Predict CHD Probability
        # ====================================================

        pred_prob = app.model.predict_proba(chd_df)[0][1]

        # Round probability to 2 decimal places
        pred_prob = float(np.round(pred_prob, 2))

        print(f"Predicted probability of CHD: {pred_prob}")

        # ====================================================
        # Return Prediction
        # ====================================================

        return jsonify({
            "probability_of_CHD": pred_prob
        })

    except Exception as e:

        print("Error during prediction:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# Run Flask
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5002))

    app.run(
        host="0.0.0.0",
        port=port
    )