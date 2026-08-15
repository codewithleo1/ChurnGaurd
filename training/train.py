"""
ChurnGuard — Training Script
Loads data, preprocesses, trains XGBoost, logs everything to MLflow.
Run with: uv run python training/train.py
"""

import os
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

# ── 1. CONFIG ────────────────────────────────────────────────────────────────
# All hardcoded values live here — easy to change without hunting through code
DATA_PATH = "data/raw/telco_churn.csv"
MLFLOW_EXPERIMENT = "churnguard-training"
TEST_SIZE = 0.2       # 80% train, 20% test
RANDOM_STATE = 42     # fixed seed → reproducible splits every run

# XGBoost hyperparameters — we'll tune these in a later phase
PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "scale_pos_weight": 2.7,  # handles class imbalance (ratio of No:Yes ≈ 73:27)
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
}


# ── 2. DATA LOADING ──────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


# ── 3. PREPROCESSING ─────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Clean and encode the raw DataFrame.
    Returns X (features) and y (target).
    
    Key decisions made here (informed by EDA):
    - Drop customerID: it's an identifier, not a feature
    - TotalCharges: convert to float, fill 11 blank rows with 0
    - All string columns: label-encode to integers (XGBoost needs numbers)
    - Target: Churn Yes→1, No→0
    """
    df = df.copy()

    # Drop identifier column — carries no predictive signal
    df.drop(columns=["customerID"], inplace=True)

    # Fix TotalCharges: blank strings → 0, then cast to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Encode target: Yes → 1, No → 0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Label-encode all remaining string columns
    # (One-hot encoding would also work — we'll revisit in feature engineering)
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    # Split into features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print(f"Features: {X.shape[1]} columns")
    print(f"Target distribution:\n{y.value_counts()}")
    return X, y


# ── 4. TRAINING ──────────────────────────────────────────────────────────────
def train(X, y):
    """
    Split data, train XGBoost, log everything to MLflow.
    Returns the trained model and test metrics.
    """
    # Split — stratify ensures class ratio is preserved in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Start MLflow run — everything inside this block gets logged
    with mlflow.start_run():

        # Log hyperparameters so we can reproduce this exact run later
        mlflow.log_params(PARAMS)

        # Train the model
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Compute metrics — we care most about recall and AUC for churn
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),    # did we catch churners?
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),  # overall discrimination
        }

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Log the model itself to MLflow model registry
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="churnguard-xgboost",
        )

        # Print results
        print("\n── Metrics ──────────────────────────────")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("─────────────────────────────────────────")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    return model, metrics


# ── 5. ENTRYPOINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Point MLflow at a local tracking server (file-based for now)
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("── Step 1: Load data ────────────────────")
    df = load_data(DATA_PATH)

    print("\n── Step 2: Preprocess ───────────────────")
    X, y = preprocess(df)

    print("\n── Step 3: Train & log to MLflow ────────")
    model, metrics = train(X, y)

    print("\nDone. Run `uv run mlflow ui` to explore results.")