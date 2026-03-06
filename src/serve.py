"""
FastAPI endpoint for serving churn predictions.

Usage:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
"""

import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a telecom customer is likely to churn based on their account and service features.",
    version="1.0.0",
)

# ─── Load Model ──────────────────────────────────────────────────────────────

MODEL_PATH = Path("models/gradient_boosting_pipeline.joblib")

model = None

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        print(f"WARNING: Model not found at {MODEL_PATH}. Train first with: python src/train.py")
        return
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")


# ─── Request / Response Schemas ──────────────────────────────────────────────

class CustomerData(BaseModel):
    """Input schema for a single customer prediction."""
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0, ge=0, le=1)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12, ge=0)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.35)
    TotalCharges: float = Field(..., example=844.2)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.20,
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str


# ─── Feature Engineering (must match training) ──────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as during training."""
    df = df.copy()
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "12-24m", "24-48m", "48-72m"]
    ).astype(str)
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    df["NumServices"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )
    return df


def get_risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    return "LOW"


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Customer Churn Prediction API",
        "status": "running",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """Predict churn probability for a single customer."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train first with: python src/train.py"
        )

    # Convert to DataFrame
    input_df = pd.DataFrame([customer.model_dump()])

    # Apply feature engineering
    input_df = engineer_features(input_df)

    # Predict
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    risk_level = get_risk_level(probability)

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        risk_level=risk_level,
    )


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerData]):
    """Predict churn for multiple customers at once."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train first with: python src/train.py"
        )

    input_df = pd.DataFrame([c.model_dump() for c in customers])
    input_df = engineer_features(input_df)

    predictions = model.predict(input_df).tolist()
    probabilities = model.predict_proba(input_df)[:, 1].tolist()

    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append({
            "churn_prediction": int(pred),
            "churn_probability": round(prob, 4),
            "risk_level": get_risk_level(prob),
        })

    return {"predictions": results, "count": len(results)}
