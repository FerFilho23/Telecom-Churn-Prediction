from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd


VERSION = "v1"
BEST_THRESHOLD = 0.35
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / f"{VERSION}_thrs_{BEST_THRESHOLD}.joblib"

app = FastAPI(title="customer-churn-classifier")
model = None

try:
    model = joblib.load(MODEL_FILE)
    print("Loaded model from", MODEL_FILE)
except FileNotFoundError:
    # Let container start, but return 503 on predict until the model is present
    print(f"Model not found at {MODEL_FILE}. Train first or mount the file.")


# ---------- Schemas ----------
yes_no = Literal["Yes", "No"]
contract = Literal["Month-to-month", "One year", "Two year"]
payment_method = Literal[
    "Mailed check",
    "Electronic check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
internet_service = Literal["DSL", "Fiber optic", "No"]


class Customer(BaseModel):
    # Categorical
    senior_citizen: yes_no
    dependents: yes_no
    internet_service: internet_service
    online_security: yes_no | Literal["No internet service"]
    online_backup: yes_no | Literal["No internet service"]
    device_protection: yes_no | Literal["No internet service"]
    tech_support: yes_no | Literal["No internet service"]
    streaming_tv: yes_no | Literal["No internet service"]
    streaming_movies: yes_no | Literal["No internet service"]
    contract: contract
    paperless_billing: yes_no
    payment_method: payment_method

    # Numeric
    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)


@app.get("/health")
def health():
    ok = model is not None
    return {
        "status": "ok",
        "model": VERSION,
        "threshold": BEST_THRESHOLD if ok else "model_missing",
    }


@app.post("/predict")
def predict(customer: Customer):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = pd.DataFrame([customer.dict()])
        proba = float(model.predict_proba(X)[:, 1][0])
        will_churn = "Churn" if proba >= BEST_THRESHOLD else "No Churn"
        return {"churn_probability": proba, "will_churn": will_churn}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
