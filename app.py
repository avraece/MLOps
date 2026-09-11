import os
import pickle
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
        "message": "CHD ML Model API is running",
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

    if app.model is None:
        return jsonify({
            "error": "CHD model is not loaded"
        }), 500

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No JSON data provided"
        }), 400

    try:

        # Convert JSON input to DataFrame
        data_df = pd.DataFrame([data])

        # Prediction
        prediction = app.model.predict(data_df)

        return jsonify({
            "model": "chd.pickle",
            "prediction": prediction.tolist()
        })

    except Exception as e:

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