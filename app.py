import os
import mlflow
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# ============================================================
# MLflow Configuration
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://avramlops.onrender.com"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("MLflow Tracking URI:", MLFLOW_TRACKING_URI)


# ============================================================
# Model Configuration
# ============================================================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "logistic"
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "1"
)

print("Model Name:", MODEL_NAME)
print("Model Version:", MODEL_VERSION)


# ============================================================
# Load Model from MLflow
# ============================================================

try:

    app.model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )

    print("Model loaded successfully!")

except Exception as e:

    print("ERROR: Could not load MLflow model")
    print("Error:", str(e))

    app.model = None


# ============================================================
# Home Route
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "message": "ML Model Server is running",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
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
            "error": "Model is not loaded"
        }), 500

    data = request.get_json()

    if not data:

        return jsonify({
            "error": "No JSON data provided"
        }), 400

    try:

        # Convert input JSON to DataFrame
        data_df = pd.DataFrame([data])

        # Generate prediction
        prediction = app.model.predict(data_df)

        return jsonify({
            "model": MODEL_NAME,
            "version": MODEL_VERSION,
            "prediction": prediction.tolist()
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# Run Flask Application
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5002))

    app.run(
        host="0.0.0.0",
        port=port
    )