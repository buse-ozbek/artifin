"""
generate_batch.py
=================
Generates a "batch" of new incoming data — simulates production traffic.
Mirrors the teacher's Iris example but adapted for the retail dataset.

Usage:
    python monitoring/scripts/generate_batch.py
"""

import argparse
import os
import random
import uuid
import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

MODEL_FEATURES = [
    "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Demand Forecast", "Price", "Discount",
    "Weather Condition", "Holiday/Promotion", "Competitor Pricing",
    "Seasonality", "Year", "Month", "DayOfWeek",
]

CLEAN_FEATURES = [
    "store_id", "product_id", "category", "region",
    "inventory_level", "demand_forecast", "price", "discount",
    "weather_condition", "holiday_promotion", "competitor_pricing",
    "seasonality", "year", "month", "day_of_week",
]

CAT_COLS = ["Store ID", "Product ID", "Category", "Region",
            "Weather Condition", "Seasonality"]


def main():
    parser = argparse.ArgumentParser(description="Generate a fake batch of retail data.")
    parser.add_argument("--size", type=int, default=200, help="Number of rows in the batch")
    args = parser.parse_args()

    os.makedirs("data/current_batches", exist_ok=True)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    model = mlflow.pyfunc.load_model("models:/retail_demand_model@Staging")

    df = pd.read_csv("../retail_store_inventory.csv")
    df["Date"]      = pd.to_datetime(df["Date"])
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    le = LabelEncoder()
    for col in CAT_COLS:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[MODEL_FEATURES].copy()
    y = df["Units Sold"].copy()

    batch = X.copy()
    batch["target"] = y
    batch = batch.sample(n=args.size, replace=True).reset_index(drop=True)

    # inject random drift on a random subset of features
    drift_options = {
        "Price":              lambda s: s * random.uniform(0.90, 1.25),
        "Discount":           lambda s: (s + random.randint(0, 15)).clip(upper=80),
        "Inventory Level":    lambda s: (s * random.uniform(0.5, 1.0)).clip(lower=0),
        "Competitor Pricing": lambda s: s * random.uniform(0.90, 1.20),
    }

    n_drifted = random.randint(0, len(drift_options))
    drifted_columns = random.sample(list(drift_options.keys()), n_drifted)

    for col in drifted_columns:
        batch[col] = drift_options[col](batch[col])

    print(f"  drifted columns this run: {drifted_columns or 'none'}")

    X_batch = batch[MODEL_FEATURES]
    predictions = np.clip(np.round(model.predict(X_batch)), 0, None).astype(int)

    batch = batch[MODEL_FEATURES + ["target"]].copy()
    batch.columns = CLEAN_FEATURES + ["target"]
    batch["prediction"] = predictions

    batch_id = str(uuid.uuid4())[:8]
    batch.to_csv(f"data/current_batches/{batch_id}.csv", index=False)
    print(f"batch saved: {batch_id}.csv  ({args.size} rows)")


if __name__ == "__main__":
    main()
