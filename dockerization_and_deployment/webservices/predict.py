"""
predict.py
==========
Standalone FastAPI prediction webservice used by the CI/CD pipeline.

Independent of training_pipeline/app.py — that one still serves your
local development with Postgres logging. This one is the leaner
version that gets containerized in CD: it loads the model from the
MLflow registry and serves /predict using prediction_core.
"""

import os
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prediction_core import predict as _predict_core

# ─────────────────────────────────────────────────────────
# Load model from MLflow Model Registry
# ─────────────────────────────────────────────────────────
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
model = mlflow.pyfunc.load_model("models:/retail_demand_model@Staging")


def predict(input_data: dict):
    """Module-level convenience wrapper — what test_predict.py calls."""
    return _predict_core(input_data, model)


# ─────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Retail Demand Forecasting API",
    description="Predicts daily Units Sold using a model from MLflow Registry.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    date: str
    store_id: str
    product_id: str
    category: str
    region: str
    inventory_level: float
    demand_forecast: float
    price: float
    discount: float
    weather_condition: str
    holiday_promotion: int
    competitor_pricing: float
    seasonality: str

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-03-15",
                "store_id": "S001",
                "product_id": "P0005",
                "category": "Electronics",
                "region": "North",
                "inventory_level": 200,
                "demand_forecast": 150.5,
                "price": 49.99,
                "discount": 10,
                "weather_condition": "Sunny",
                "holiday_promotion": 1,
                "competitor_pricing": 52.00,
                "seasonality": "Winter"
            }
        }


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        data = req.model_dump()
        return {"prediction": predict(data)}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
