"""
ChurnGuard — Model Loader
Loads the XGBoost model from MLflow registry once at startup.
The loaded model is reused for every prediction request — not reloaded each time.
"""

from pathlib import Path

import mlflow
import mlflow.xgboost


def load_model(model_name: str = "churnguard-xgboost", version: int = 1):
    """
    Load a registered model from the local MLflow registry.
    
    Args:
        model_name: Name of the registered model in MLflow
        version: Model version to load
    
    Returns:
        Loaded XGBoost model ready for inference
    """
    # Point to our local MLflow tracking server
    # In production this would be a remote URI (e.g. hosted MLflow or S3)
    mlruns_path = Path(__file__).parent.parent / "mlruns"
    mlflow.set_tracking_uri(mlruns_path.as_uri())

    model_uri = f"models:/{model_name}/{version}"
    print(f"Loading model from MLflow: {model_uri}")

    model = mlflow.xgboost.load_model(model_uri)
    print("Model loaded successfully")
    return model


def get_feature_columns() -> list[str]:
    """
    Returns the exact list of feature columns the model expects — in order.
    This must match what training/train.py produces after preprocessing.
    Hardcoded here so the API never has to run preprocessing logic.
    """
    return [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]