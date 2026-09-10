import os
import mlflow
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# MLflow server URL
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:8080"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Model information
model_name = "logistic"
model_version = "1"

# Load model from MLflow
app.model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)


@app.route("/")
def home():
    return jsonify({
        "message": "ML Model Server is running",
        "model": model_name,
        "version": model_version
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    data_df = pd.DataFrame([data])

    prediction = app.model.predict(data_df)

    return jsonify({
        "prediction": prediction.tolist()
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port)