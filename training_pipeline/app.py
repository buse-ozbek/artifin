"""
app.py
======
Retail Store Inventory — FastAPI Prediction Service

Exposes your trained ML model as a REST API so anyone
(or any system) can send store/product data and get a
demand prediction back.

Usage:
    pip install fastapi uvicorn
    uvicorn app:app --reload

Then open:
    http://127.0.0.1:8000          → welcome message
    http://127.0.0.1:8000/docs     → interactive API docs (try it live!)
    http://127.0.0.1:8000/predict  → POST endpoint for predictions
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
MODEL_DIR   = "models"
DEFAULT_REF = os.path.join(MODEL_DIR, "best_model_path.txt")

FEATURE_COLS = [
    "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Demand Forecast", "Price", "Discount",
    "Weather Condition", "Holiday/Promotion", "Competitor Pricing",
    "Seasonality", "Year", "Month", "DayOfWeek",
]
CAT_COLS = [
    "Store ID", "Product ID", "Category", "Region",
    "Weather Condition", "Seasonality",
]


# ─────────────────────────────────────────────────────────
# LOAD MODEL AT STARTUP
# ─────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(DEFAULT_REF):
        raise FileNotFoundError(
            "No model found. Please run pipeline.py first."
        )
    with open(DEFAULT_REF) as f:
        model_path = f.read().strip()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    print(f"✓ Model loaded: {model_path}")
    return model, model_path


# Load the model once when the server starts
model, model_path = load_model()


# ─────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "Predicts daily Units Sold for a retail store product "
        "based on store context, pricing, weather, and more."
    ),
    version="1.0.0",
)

#YOU CAN DELETE THIS PART, HOCA SAID IT'S NOT NECESSARY BECAUSE AS THE DATA SCIENTIST YOU WILL CHOOSE THE MODEL, NOT THE USER

# ─────────────────────────────────────────────────────────
# INPUT SCHEMA
# This defines exactly what data the API expects to receive
# ─────────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    date: str               # e.g. "2024-03-15"
    store_id: str           # e.g. "S001"
    product_id: str         # e.g. "P0005"
    category: str           # e.g. "Electronics"
    region: str             # e.g. "North"
    inventory_level: float  # e.g. 200
    demand_forecast: float  # e.g. 150.5
    price: float            # e.g. 49.99
    discount: float         # e.g. 10  (percentage)
    weather_condition: str  # e.g. "Sunny"
    holiday_promotion: int  # 0 or 1
    competitor_pricing: float  # e.g. 52.00
    seasonality: str        # e.g. "Winter"

    # Example shown in the /docs page
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


# ─────────────────────────────────────────────────────────
# OUTPUT SCHEMA
# This defines what the API sends back
# ─────────────────────────────────────────────────────────
class PredictionOutput(BaseModel):
    predicted_units_sold: int
    model_used: str
    input_summary: dict


# ─────────────────────────────────────────────────────────
# PREPROCESSING  (mirrors pipeline.py exactly)
# ─────────────────────────────────────────────────────────
def preprocess_input(data: PredictionInput) -> pd.DataFrame:
    # Parse date
    date = pd.to_datetime(data.date)

    # Build a one-row dataframe with the same column names as training
    row = {
        "Store ID":          data.store_id,
        "Product ID":        data.product_id,
        "Category":          data.category,
        "Region":            data.region,
        "Inventory Level":   data.inventory_level,
        "Demand Forecast":   data.demand_forecast,
        "Price":             data.price,
        "Discount":          data.discount,
        "Weather Condition": data.weather_condition,
        "Holiday/Promotion": data.holiday_promotion,
        "Competitor Pricing":data.competitor_pricing,
        "Seasonality":       data.seasonality,
        "Year":              date.year,
        "Month":             date.month,
        "DayOfWeek":         date.dayofweek,
    }
    df = pd.DataFrame([row])

    # Encode categoricals (same as training)
    le = LabelEncoder()
    for col in CAT_COLS:
        df[col] = le.fit_transform(df[col].astype(str))

    return df[FEATURE_COLS]


# ─────────────────────────────────────────────────────────
# ROUTES (API Endpoints)
# ─────────────────────────────────────────────────────────

# GET / → welcome message
@app.get("/")
def root():
    return {
        "message": "🛒 Retail Demand Forecasting API is running!",
        "docs": "Visit /docs to try the API interactively",
        "model": model_path,
    }


# GET /health → check if the API is alive
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": model_path,
    }


# POST /predict → main prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Preprocess the incoming data
        X = preprocess_input(input_data)

        # Run prediction
        prediction = model.predict(X)[0]
        prediction = max(0, int(round(prediction)))  # no negatives, whole number

        return PredictionOutput(
            predicted_units_sold=prediction,
            model_used=os.path.basename(model_path),
            input_summary={
                "store":    input_data.store_id,
                "product":  input_data.product_id,
                "category": input_data.category,
                "date":     input_data.date,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# POST /predict/batch → predict multiple rows at once
@app.post("/predict/batch")
def predict_batch(inputs: list[PredictionInput]):
    if len(inputs) > 500:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 500 rows per request."
        )
    try:
        results = []
        for item in inputs:
            X = preprocess_input(item)
            pred = model.predict(X)[0]
            pred = max(0, int(round(pred)))
            results.append({
                "store_id":            item.store_id,
                "product_id":          item.product_id,
                "date":                item.date,
                "predicted_units_sold": pred,
            })
        return {"predictions": results, "count": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
