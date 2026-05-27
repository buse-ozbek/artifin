"""
generate_batch.py
=================
Generates a "batch" of new incoming data — simulates production traffic.

Two layers of drift:
  1. TEMPORAL: samples only from BATCH_YEAR_MIN onwards (default 2023),
     while the model was trained on 2022. This is proper MLOps practice
     (never train on future data).
  2. ARTIFICIAL: applies controlled feature shifts to simulate realistic
     production phenomena — supplier price changes, forecast accuracy
     degradation, competitor pricing shifts, inventory volatility.
     The synthetic Kaggle dataset has near-identical distributions across
     years, so without this second layer no real drift would be detected.

Usage:
    python monitoring/scripts/generate_batch.py
    python monitoring/scripts/generate_batch.py --size 500
    python monitoring/scripts/generate_batch.py --severity mild
    python monitoring/scripts/generate_batch.py --severity severe
"""

import argparse
import os
import uuid
import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BATCH_YEAR_MIN = 2023     # ← sample batches strictly AFTER TRAIN_YEAR (2022)

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


# Drift profiles — each maps to a set of (column, transform) pairs.
# Numbers are intentionally aggressive enough to show up in MAE/RMSE.
DRIFT_PROFILES = {
    "mild": {
        "Price":              lambda s: s * np.random.uniform(0.85, 1.20, size=len(s)),
        "Discount":           lambda s: np.clip(s + np.random.randint(0, 20, size=len(s)), 0, 80),
        "Demand Forecast":    lambda s: s * np.random.uniform(0.80, 1.20, size=len(s)),
    },
    "medium": {
        "Price":              lambda s: s * np.random.uniform(0.70, 1.40, size=len(s)),
        "Discount":           lambda s: np.clip(s + np.random.randint(5, 35, size=len(s)), 0, 80),
        "Inventory Level":    lambda s: np.clip(s * np.random.uniform(0.40, 1.10, size=len(s)), 0, None),
        "Demand Forecast":    lambda s: s * np.random.uniform(0.60, 1.50, size=len(s)) + np.random.normal(0, 20, size=len(s)),
        "Competitor Pricing": lambda s: s * np.random.uniform(0.75, 1.30, size=len(s)),
    },
    "severe": {
        "Price":              lambda s: s * np.random.uniform(0.50, 1.80, size=len(s)),
        "Discount":           lambda s: np.clip(s + np.random.randint(10, 50, size=len(s)), 0, 80),
        "Inventory Level":    lambda s: np.clip(s * np.random.uniform(0.20, 1.20, size=len(s)), 0, None),
        "Demand Forecast":    lambda s: s * np.random.uniform(0.30, 2.00, size=len(s)) + np.random.normal(0, 50, size=len(s)),
        "Competitor Pricing": lambda s: s * np.random.uniform(0.50, 1.60, size=len(s)),
        "Holiday/Promotion":  lambda s: np.random.randint(0, 2, size=len(s)),  # full randomization
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate a fake batch of retail data.")
    parser.add_argument("--size", type=int, default=200, help="Number of rows in the batch")
    parser.add_argument("--severity", choices=["mild", "medium", "severe"], default="medium",
                        help="Drift severity profile (default: medium)")
    args = parser.parse_args()

    os.makedirs("data/current_batches", exist_ok=True)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    model = mlflow.pyfunc.load_model("models:/retail_demand_model@Staging")

    df = pd.read_csv("../retail_store_inventory.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # ── LAYER 1: TEMPORAL DRIFT ──────────────────────────
    before = len(df)
    df = df[df["Date"].dt.year >= BATCH_YEAR_MIN].copy()
    print(f"  layer 1 (temporal): year ≥ {BATCH_YEAR_MIN}: {before} → {len(df)} rows")

    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    le = LabelEncoder()
    for col in CAT_COLS:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[MODEL_FEATURES].copy()
    y = df["Units Sold"].copy()

    batch = X.copy()
    batch["target"] = y.values
    batch = batch.sample(n=args.size, replace=True).reset_index(drop=True)

    # ── LAYER 2: ARTIFICIAL DRIFT (always applied) ──────
    profile = DRIFT_PROFILES[args.severity]
    print(f"  layer 2 (artificial): severity = {args.severity}, drifting {list(profile.keys())}")

    for col, transform in profile.items():
        batch[col] = transform(batch[col])

    # ── Score the drifted batch ──────────────────────────
    X_batch = batch[MODEL_FEATURES]
    predictions = np.clip(np.round(model.predict(X_batch)), 0, None).astype(int)

    batch = batch[MODEL_FEATURES + ["target"]].copy()
    batch.columns = CLEAN_FEATURES + ["target"]
    batch["prediction"] = predictions

    batch_id = str(uuid.uuid4())[:8]
    batch.to_csv(f"data/current_batches/{batch_id}.csv", index=False)
    print(f"batch saved: {batch_id}.csv  ({args.size} rows, severity={args.severity})")


if __name__ == "__main__":
    main()