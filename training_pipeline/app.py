"""
app.py
======
Retail Store Inventory — FastAPI Prediction Service

Loads the best trained model directly from the MLflow Model Registry
(just like the professor's version) and serves predictions via FastAPI.

Usage:
    # Make sure mlflow ui is running first:
    mlflow ui --port 5001

    # Then start the API:
    uvicorn app:app --reload --port 8001

Then open:
    http://127.0.0.1:8001/docs  → interactive API docs
"""

import mlflow
import mlflow.pyfunc
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from sklearn.preprocessing import LabelEncoder
import sys

# ─────────────────────────────────────────────────────────
# MLFLOW SETUP
# Connect to your running MLflow server
# ─────────────────────────────────────────────────────────
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))

# ─────────────────────────────────────────────────────────
# LOAD MODEL FROM MLFLOW MODEL REGISTRY
# This loads the model that pipeline.py registered as "Staging"
# ─────────────────────────────────────────────────────────
loaded_model = mlflow.pyfunc.load_model(
    "models:/retail_demand_model@Staging"
)

# ─────────────────────────────────────────────────────────
# CONFIG — must match pipeline.py exactly
# ─────────────────────────────────────────────────────────
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
# PREPROCESSING — mirrors pipeline.py exactly
# ─────────────────────────────────────────────────────────
def preprocess(input_data: dict) -> pd.DataFrame:
    date = pd.to_datetime(input_data["date"])

    row = {
        "Store ID":           input_data["store_id"],
        "Product ID":         input_data["product_id"],
        "Category":           input_data["category"],
        "Region":             input_data["region"],
        "Inventory Level":    input_data["inventory_level"],
        "Demand Forecast":    input_data["demand_forecast"],
        "Price":              input_data["price"],
        "Discount":           input_data["discount"],
        "Weather Condition":  input_data["weather_condition"],
        "Holiday/Promotion":  input_data["holiday_promotion"],
        "Competitor Pricing": input_data["competitor_pricing"],
        "Seasonality":        input_data["seasonality"],
        "Year":               date.year,
        "Month":              date.month,
        "DayOfWeek":          date.dayofweek,
    }

    df = pd.DataFrame([row])

    # Hardcoded maps — same order as pipeline.py's LabelEncoder saw during training
    category_map    = {"Clothing": 0, "Electronics": 1, "Furniture": 2, "Groceries": 3, "Toys": 4}
    region_map      = {"East": 0, "North": 1, "South": 2, "West": 3}
    weather_map     = {"Cloudy": 0, "Rainy": 1, "Snowy": 2, "Sunny": 3}
    seasonality_map = {"Autumn": 0, "Spring": 1, "Summer": 2, "Winter": 3}

    df["Category"]          = df["Category"].map(category_map)
    df["Region"]            = df["Region"].map(region_map)
    df["Weather Condition"] = df["Weather Condition"].map(weather_map)
    df["Seasonality"]       = df["Seasonality"].map(seasonality_map)

    # Store ID and Product ID — extract the number part
    df["Store ID"]   = df["Store ID"].str.extract(r'(\d+)').astype(int) - 1
    df["Product ID"] = df["Product ID"].str.extract(r'(\d+)').astype(int) - 1

    return df[FEATURE_COLS]


# ─────────────────────────────────────────────────────────
# PREDICT FUNCTION
# ─────────────────────────────────────────────────────────
def predict(input_data: dict):
    df = preprocess(input_data)
    df = np.array(df)

    # Ensure shape is (1, n_features)
    if df.ndim == 1:
        df = df.reshape(1, -1)

    prediction = loaded_model.predict(df)
    prediction = max(0, int(round(float(prediction[0]))))
    return prediction


# ─────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Retail Demand Forecasting API",
    description="Predicts daily Units Sold using a model loaded from MLflow Model Registry.",
    version="1.0.0",
)


# ─────────────────────────────────────────────────────────
# INPUT / OUTPUT SCHEMAS
# ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    date: str                       # e.g. "2024-03-15"
    store_id: str                   # e.g. "S001"
    product_id: str                 # e.g. "P0005"
    category: str                   # e.g. "Electronics"
    region: str                     # e.g. "North"
    inventory_level: float          # e.g. 200
    demand_forecast: float          # e.g. 150.5
    price: float                    # e.g. 49.99
    discount: float                 # e.g. 10
    weather_condition: str          # e.g. "Sunny"
    holiday_promotion: int          # 0 or 1
    competitor_pricing: float       # e.g. 52.00
    seasonality: str                # e.g. "Winter"

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
# ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        prediction = predict(req.dict())
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────
# COMMAND LINE SUPPORT (for quick testing without the server)
# Usage: python app.py S001 P0005 Electronics North 200 150.5 49.99 10 Sunny 1 52.00 Winter 2024-03-15
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 15:
        print("Usage: python app.py store_id product_id category region "
              "inventory_level demand_forecast price discount "
              "weather_condition holiday_promotion competitor_pricing "
              "seasonality date")
        sys.exit(1)

    sample = {
        "store_id":          sys.argv[1],
        "product_id":        sys.argv[2],
        "category":          sys.argv[3],
        "region":            sys.argv[4],
        "inventory_level":   float(sys.argv[5]),
        "demand_forecast":   float(sys.argv[6]),
        "price":             float(sys.argv[7]),
        "discount":          float(sys.argv[8]),
        "weather_condition": sys.argv[9],
        "holiday_promotion": int(sys.argv[10]),
        "competitor_pricing":float(sys.argv[11]),
        "seasonality":       sys.argv[12],
        "date":              sys.argv[13],
    }
    result = predict(sample)
    print("Prediction:", result)
