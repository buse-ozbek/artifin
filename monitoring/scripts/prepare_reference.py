"""
prepare_reference.py
====================
Creates the reference (baseline) dataset for monitoring.
Mirrors the teacher's Iris example but adapted for the retail dataset.

NOTE: Filters to REFERENCE_YEAR (default: 2022) so the reference
distribution matches what the model was trained on. New batches
sampled from later years will then show real temporal drift.

Usage:
    python monitoring/scripts/prepare_reference.py
"""

import os
import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

REFERENCE_YEAR = 2022     # ← must match TRAIN_YEAR in pipeline.py

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

    # ── TEMPORAL SPLIT: keep only REFERENCE_YEAR rows ────
    before = len(df)
    df = df[df["Date"].dt.year == REFERENCE_YEAR].copy()
    print(f"  filtered reference to year {REFERENCE_YEAR}: {before} → {len(df)} rows")

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
    reference["target"]     = y.values
    reference["prediction"] = predictions

    reference.to_csv("data/reference.csv", index=False)
    print(f"reference.csv created with {len(reference)} rows (year={REFERENCE_YEAR})")


if __name__ == "__main__":
    main()