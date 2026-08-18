"""
ChurnGuard — FastAPI Serving Layer
Exposes a REST API for churn prediction.

Endpoints:
    GET  /health          — health check
    POST /predict         — churn prediction for a customer

Run with: uv run uvicorn serving.main:app --reload
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

from serving.explain import compute_shap_values, shap_to_text
from serving.llm_explain import get_llm_explanation
from serving.model_loader import get_feature_columns, load_model

# ── STARTUP / SHUTDOWN ────────────────────────────────────────────────────────
# FastAPI lifespan: code before yield runs at startup, after yield at shutdown.
# We load the model and feature store ONCE here — not on every request.

app_state = {}  # shared state across requests

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — load everything into memory
    print("Loading model from MLflow...")
    app_state["model"] = load_model()

    print("Connecting to Feast feature store...")
    app_state["store"] = FeatureStore(repo_path="feature_repo")

    print("API ready.")
    yield
    # Shutdown — cleanup if needed
    app_state.clear()


# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ChurnGuard API",
    description="Predicts customer churn probability with SHAP explanations.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── SCHEMAS ───────────────────────────────────────────────────────────────────
# Pydantic models define the shape of requests and responses.
# FastAPI validates inputs automatically — wrong types = 422 error, not a crash.

class PredictRequest(BaseModel):
    customer_id: str  # e.g. "7590-VHVEG"

class PredictResponse(BaseModel):
    customer_id: str
    churn_probability: float          # 0.0 to 1.0
    churn_prediction: bool            # True = likely to churn
    top_shap_features: dict           # {feature: shap_value} top 5
    shap_explanation: str             # human-readable SHAP summary
    llm_explanation: str              # LMM summerizes the prediction with SHAP values


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — returns OK if the API is running."""
    return {"status": "ok", "model_loaded": "model" in app_state}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict churn probability for a customer.

    Steps:
    1. Fetch features from Feast online store
    2. Encode string features to integers (model expects numbers)
    3. Run XGBoost inference
    4. Compute SHAP values
    5. Return prediction + explanation
    """
    model = app_state["model"]
    store = app_state["store"]
    feature_cols = get_feature_columns()

    # ── Step 1: Fetch features from Feast online store ────────────────────────
    try:
        feast_features = [f"customer_features:{col}" for col in [
            "tenure", "MonthlyCharges", "TotalCharges", "Contract",
            "InternetService", "TechSupport", "PaymentMethod",
            "SeniorCitizen", "Partner", "Dependents"
        ]]
        online_features = store.get_online_features(
            features=feast_features,
            entity_rows=[{"customerID": request.customer_id}],
        ).to_dict()
    except Exception as e:      # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"Customer not found: {e}")

    # Check we got data back
    if online_features["tenure"][0] is None:
        raise HTTPException(
            status_code=404,
            detail=f"No features found for customer {request.customer_id}"
        )

    # ── Step 2: Build feature DataFrame ──────────────────────────────────────
    # Map Feast feature names to what the model expects
    raw = {
        "SeniorCitizen": online_features["SeniorCitizen"][0],
        "Partner": online_features["Partner"][0],
        "Dependents": online_features["Dependents"][0],
        "tenure": online_features["tenure"][0],
        "InternetService": online_features["InternetService"][0],
        "OnlineSecurity": "No",       # not in Feast — use default
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": online_features["TechSupport"][0],
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": online_features["Contract"][0],
        "PaperlessBilling": "Yes",
        "PaymentMethod": online_features["PaymentMethod"][0],
        "MonthlyCharges": online_features["MonthlyCharges"][0],
        "TotalCharges": online_features["TotalCharges"][0],
        "gender": "Male",             # not in Feast — use default
        "PhoneService": "Yes",
        "MultipleLines": "No",
    }

    feature_df = pd.DataFrame([raw])[feature_cols]

    # ── Step 3: Encode string columns to integers ─────────────────────────────
    # The model was trained on label-encoded data — same encoding needed here
    le = LabelEncoder()
    for col in feature_df.select_dtypes(include="object").columns:
        feature_df[col] = le.fit_transform(feature_df[col])

    # ── Step 4: Run inference ─────────────────────────────────────────────────
    churn_prob = float(model.predict_proba(feature_df)[0][1])
    churn_pred = churn_prob >= 0.5

    # ── Step 5: Compute SHAP values ───────────────────────────────────────────
    top_shap = compute_shap_values(model, feature_df)
    shap_text = shap_to_text(top_shap)

    llm_text = get_llm_explanation(
        customer_id=request.customer_id,
        churn_probability=churn_prob,
        shap_explanation=shap_text,
    )

    return PredictResponse(
        customer_id=request.customer_id,
        churn_probability=round(churn_prob, 4),
        churn_prediction=churn_pred,
        top_shap_features=top_shap,
        shap_explanation=shap_text,
        llm_explanation=llm_text,
    )