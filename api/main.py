"""
FastAPI inference endpoint for the construction anomaly detector.

POST /predict      — score a single line item
POST /predict/batch — score multiple line items

Usage:
    uvicorn api.main:app --reload
"""

import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.network import AnomalyDetector
from data.generate import TRADE_INDEX, SCOPE_INDEX
from model.dataset import NUMERIC_FEATURES, INPUT_DIM

# ── Load model at startup ──────────────────────────────────────────────────────
CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_model.pt")
THRESHOLD  = 0.5

app = FastAPI(
    title="Construction Cost Anomaly Detector",
    description="Scores change order line items for anomalous cost patterns.",
    version="1.0.0",
)

model    = None
scaler   = None
device   = torch.device("cpu")


@app.on_event("startup")
def load_model():
    global model, scaler
    if not os.path.exists(CHECKPOINT):
        print("WARNING: No checkpoint found. Run model/train.py first.")
        return

    ckpt   = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model  = AnomalyDetector().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = ckpt["scaler"]
    print(f"Model loaded (trained to epoch {ckpt['epoch']}, val AUC: {ckpt['val_auc']:.4f})")


# ── Request / Response schemas ─────────────────────────────────────────────────

class LineItem(BaseModel):
    trade: Literal["electrical", "mechanical", "concrete", "steel", "plumbing", "drywall", "hvac", "general"]
    scope_category: Literal["owner_directive", "unforeseen_condition", "design_error", "code_compliance", "value_engineering"]
    labor_hours: float = Field(..., gt=0, description="Total labor hours")
    labor_rate: float  = Field(..., gt=0, description="Hourly labor rate ($/hr)")
    labor_cost: float  = Field(..., gt=0, description="Total labor cost ($)")
    labor_burden_pct: float = Field(..., gt=0, le=2.0, description="Labor burden as decimal (e.g. 0.32)")
    material_cost: float    = Field(..., ge=0, description="Total material cost ($)")
    markup_pct: float       = Field(..., ge=0, le=2.0, description="Markup as decimal (e.g. 0.10)")
    total_cost: float       = Field(..., gt=0, description="Total line item cost ($)")

    class Config:
        json_schema_extra = {
            "example": {
                "trade": "electrical",
                "scope_category": "unforeseen_condition",
                "labor_hours": 12.0,
                "labor_rate": 82.50,
                "labor_cost": 990.0,
                "labor_burden_pct": 0.32,
                "material_cost": 540.0,
                "markup_pct": 0.10,
                "total_cost": 1683.0,
            }
        }


class PredictionResult(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    risk_level: Literal["low", "medium", "high"]


def encode_item(item: LineItem) -> np.ndarray:
    """Encode a LineItem into the feature vector expected by the model."""
    numeric = np.array([[
        item.labor_hours,
        item.labor_rate,
        item.labor_cost,
        item.labor_burden_pct,
        item.material_cost,
        item.markup_pct,
        item.total_cost,
    ]], dtype=np.float32)

    numeric_scaled = scaler.transform(numeric)

    trade_enc = np.zeros((1, len(TRADE_INDEX)), dtype=np.float32)
    trade_enc[0, TRADE_INDEX[item.trade]] = 1.0

    scope_enc = np.zeros((1, len(SCOPE_INDEX)), dtype=np.float32)
    scope_enc[0, SCOPE_INDEX[item.scope_category]] = 1.0

    return np.concatenate([numeric_scaled, trade_enc, scope_enc], axis=1)


def score_to_risk(score: float) -> str:
    if score < 0.35:
        return "low"
    elif score < 0.65:
        return "medium"
    return "high"


def predict_items(items: List[LineItem]) -> List[PredictionResult]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model/train.py first.")

    features = np.concatenate([encode_item(item) for item in items], axis=0)
    tensor   = torch.tensor(features, dtype=torch.float32).to(device)

    with torch.no_grad():
        scores = model(tensor).cpu().numpy()

    return [
        PredictionResult(
            anomaly_score=round(float(score), 4),
            is_anomaly=float(score) >= THRESHOLD,
            risk_level=score_to_risk(float(score)),
        )
        for score in scores
    ]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResult)
def predict(item: LineItem):
    """Score a single change order line item."""
    return predict_items([item])[0]


@app.post("/predict/batch", response_model=List[PredictionResult])
def predict_batch(items: List[LineItem]):
    """Score a batch of change order line items."""
    if len(items) > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100.")
    return predict_items(items)
