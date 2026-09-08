# MLOps Chapter 12

This workspace contains the ML training, MLflow tracking, and Flask inference example from the MLOps chapter.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

   python -m pip install -r requirements.txt

3. Start the MLflow tracking server from the workspace root:

   mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:///%cd%/mlruns

4. Run the notebooks in order:
   - Chapter_12_MLOps.ipynb
   - Chapter_12_MLServer.ipynb
   - Chapter_12_MLClient.ipynb

## Notes

- The dataset file is expected to be in the workspace root as `SAheart.data`.
- The MLflow model name used in the notebooks is `logistic`.
- The Flask app loads the model from `models:/logistic`.
