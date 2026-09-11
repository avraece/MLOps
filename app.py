import os
import joblib
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

    # Model was saved using joblib.dump()
    app.model = joblib.load(MODEL_PATH)

    print("==========================================")
    print("CHD MODEL LOADED SUCCESSFULLY")
    print("==========================================")
    print("Model path:", MODEL_PATH)
    print("Model type:", type(app.model))
    print("Has predict:", hasattr(app.model, "predict"))
    print("Has predict_proba:", hasattr(app.model, "predict_proba"))
    print("==========================================")

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
        ),
        "model_type": (
            str(type(app.model))
            if app.model is not None
            else None
        ),
        "predict_proba_available": (
            hasattr(app.model, "predict_proba")
            if app.model is not None
            else False
        )
    })


# ============================================================
# Prediction Route
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if app.model is None:

            return jsonify({
                "error": "CHD model is not loaded"
            }), 500


        # ----------------------------------------------------
        # Get JSON input
        # ----------------------------------------------------

        data = request.get_json(force=True)

        if data is None:

            return jsonify({
                "error": "No JSON body received"
            }), 400

        print("Received data:")
        print(data)


        # ----------------------------------------------------
        # Required features
        # Based on training pipeline
        # ----------------------------------------------------

        required_features = [
            "sbp",
            "tobacco",
            "ldl",
            "adiposity",
            "famhist",
            "typea",
            "obesity",
            "alcohol",
            "age"
        ]


        # ----------------------------------------------------
        # Check missing features
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in required_features
            if feature not in data
        ]

        if missing_features:

            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400


        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        chd_df = pd.DataFrame(
            [[data[feature] for feature in required_features]],
            columns=required_features
        )

        print("Input DataFrame:")
        print(chd_df)


        # ----------------------------------------------------
        # Check predict_proba
        # ----------------------------------------------------

        if not hasattr(app.model, "predict_proba"):

            return jsonify({
                "error": "Loaded model does not support predict_proba",
                "model_type": str(type(app.model))
            }), 500


        # ----------------------------------------------------
        # Predict CHD Probability
        # ----------------------------------------------------

        pred_prob = app.model.predict_proba(chd_df)[0][1]

        pred_prob = float(np.round(pred_prob, 2))


        # ----------------------------------------------------
        # Predict Class
        # ----------------------------------------------------

        prediction = app.model.predict(chd_df)[0]

        prediction = int(prediction)


        print("Predicted class:", prediction)
        print("Predicted probability:", pred_prob)


        # ----------------------------------------------------
        # Return Response
        # ----------------------------------------------------

        return jsonify({
            "model": "chd.pickle",
            "prediction": prediction,
            "probability_of_CHD": pred_prob
        })


    except Exception as e:

        print("Error during prediction:")
        print(str(e))

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