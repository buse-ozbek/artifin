"""
prepare_reference.py
====================
Creates the reference (baseline) dataset for monitoring.
Mirrors the teacher's Iris example but adapted for the retail dataset.

Usage:
    python monitoring/scripts/prepare_reference.py
"""

import os
import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Original column names the model was trained with
MODEL_FEATURES = [
    "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Demand Forecast", "Price", "Discount",
    "Weather Condition", "Holiday/Promotion", "Competitor Pricing",
    "Seasonality", "Year", "Month", "DayOfWeek",
]

# Cleaner names for monitoring and dashboards
CLEAN_FEATURES = [
    "store_id", "product_id", "category", "region",
    "inventory_level", "demand_forecast", "price", "discount",
    "weather_condition", "holiday_promotion", "competitor_pricing",
    "seasonality", "year", "month", "day_of_week",
]

CAT_COLS = ["Store ID", "Product ID", "Category", "Region",
            "Weather Condition", "Seasonality"]


def main():
    # create folder if it does not exist
    os.makedirs("data", exist_ok=True)

    # connect to MLflow and load model
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    model = mlflow.pyfunc.load_model("models:/retail_demand_model@Staging")

    # load retail data
    df = pd.read_csv("../retail_store_inventory.csv")

    # date feature engineering (same as training)
    df["Date"]      = pd.to_datetime(df["Date"])
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # encode categoricals (same as training)
    le = LabelEncoder()
    for col in CAT_COLS:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[MODEL_FEATURES].copy()
    y = df["Units Sold"].copy()

    # generate predictions (model handles scaling internally via sklearn Pipeline)
    predictions = np.clip(np.round(model.predict(X)), 0, None).astype(int)

    # save raw features + target + prediction as reference dataset
    reference = X.copy()
    reference.columns = CLEAN_FEATURES
    reference["target"]     = y
    reference["prediction"] = predictions

    reference.to_csv("data/reference.csv", index=False)
    print("reference.csv created")


if __name__ == "__main__":
    main()