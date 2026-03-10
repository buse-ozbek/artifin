"""
predict_pipeline.py
===================
Retail Store Inventory — Demand Prediction

Loads the best trained model saved by pipeline.py and generates
demand predictions for new / unseen records.

Usage:
    # Predict on 1000 sample rows from the original dataset (default)
    python predict_pipeline.py

    # Predict on all rows in a custom CSV
    python predict_pipeline.py --input new_data.csv --output predictions.csv --sample 0

    # Use a specific model file
    python predict_pipeline.py --model models/Exp2_GradientBoosting.pkl
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────
# CONFIG  (must match pipeline.py)
# ─────────────────────────────────────────────────────────
DATA_PATH    = "retail_store_inventory.csv"
MODEL_DIR    = "models"
DEFAULT_REF  = os.path.join(MODEL_DIR, "best_model_path.txt")

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
# PREPROCESSING  (mirrors pipeline.py)
# ─────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"]      = pd.to_datetime(df["Date"])
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    le = LabelEncoder()
    for col in CAT_COLS:
        df[col] = le.fit_transform(df[col].astype(str))

    return df[FEATURE_COLS]


# ─────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────
def load_model(model_path=None):
    if model_path is None:
        if not os.path.exists(DEFAULT_REF):
            raise FileNotFoundError(
                f"No model reference found at '{DEFAULT_REF}'.\n"
                "Please run pipeline.py first."
            )
        with open(DEFAULT_REF) as f:
            model_path = f.read().strip()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    print(f"  Model loaded : {model_path}")
    return model


# ─────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────
def predict(input_path, model_path=None, output_path="predictions.csv",
            sample_size=None):
    print("=" * 52)
    print("  Retail Demand — Prediction Pipeline")
    print("=" * 52)

    print("\n[1/3] Loading model...")
    model = load_model(model_path)

    print("\n[2/3] Loading input data...")
    df_raw = pd.read_csv(input_path)
    if sample_size:
        df_raw = df_raw.sample(n=min(sample_size, len(df_raw)), random_state=42)
    print(f"      Rows to predict: {len(df_raw)}")

    X    = preprocess(df_raw)
    preds = model.predict(X)
    preds = np.clip(preds, 0, None)  # demand cannot be negative

    print("\n[3/3] Generating predictions...")
    result = df_raw[["Date", "Store ID", "Product ID",
                     "Category", "Region"]].copy().reset_index(drop=True)
    result["Predicted_Units_Sold"] = np.round(preds).astype(int)

    if "Units Sold" in df_raw.columns:
        actuals = df_raw["Units Sold"].reset_index(drop=True)
        result["Actual_Units_Sold"] = actuals
        result["Absolute_Error"]    = np.abs(
            result["Predicted_Units_Sold"] - result["Actual_Units_Sold"]
        )
        mae  = result["Absolute_Error"].mean()
        rmse = float(np.sqrt(
            ((result["Predicted_Units_Sold"] - result["Actual_Units_Sold"]) ** 2).mean()
        ))
        print(f"\n  Evaluation vs actuals:")
        print(f"    MAE  : {mae:.4f}")
        print(f"    RMSE : {rmse:.4f}")

    result.to_csv(output_path, index=False)
    print(f"\n  Predictions saved  : {output_path}")
    print("=" * 52)
    print("\n  Sample predictions:")
    print(result.head(10).to_string(index=False))
    return result


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Predict retail demand")
    p.add_argument("--input",  default=DATA_PATH,
                   help="Input CSV path")
    p.add_argument("--output", default="predictions.csv",
                   help="Output predictions CSV path")
    p.add_argument("--model",  default=None,
                   help="Model .pkl path (default: best model from pipeline.py)")
    p.add_argument("--sample", type=int, default=1000,
                   help="Rows to predict; 0 = all rows (default: 1000)")
    return p.parse_args()


def main():
    args   = parse_args()
    sample = args.sample if args.sample > 0 else None
    predict(
        input_path  = args.input,
        model_path  = args.model,
        output_path = args.output,
        sample_size = sample,
    )


if __name__ == "__main__":
    main()
