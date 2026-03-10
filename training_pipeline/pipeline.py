"""
pipeline.py
===========
Retail Store Inventory — Demand Forecasting
Goal: Predict Units Sold (demand) using store/product/context features.

Three MLflow experiments are tracked and compared:
  - Experiment 1: Linear Regression        (simple baseline)
  - Experiment 2: Random Forest Regressor  (ensemble baseline)
  - Experiment 3: Gradient Boosting Regressor (tuned)

Usage:
    pip install mlflow scikit-learn pandas numpy
    python pipeline.py

View results:
    mlflow ui          → http://localhost:5000
"""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
DATA_PATH  = "retail_store_inventory.csv"
TARGET     = "Units Sold"
EXPERIMENT = "Retail_Demand_Forecasting"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────
def load_and_preprocess(path: str):
    df = pd.read_csv(path)

    # Date-based features
    df["Date"]       = pd.to_datetime(df["Date"])
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["DayOfWeek"]  = df["Date"].dt.dayofweek

    # Encode categoricals
    cat_cols = ["Store ID", "Product ID", "Category", "Region",
                "Weather Condition", "Seasonality"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    feature_cols = [
        "Store ID", "Product ID", "Category", "Region",
        "Inventory Level", "Demand Forecast", "Price", "Discount",
        "Weather Condition", "Holiday/Promotion", "Competitor Pricing",
        "Seasonality", "Year", "Month", "DayOfWeek",
    ]

    X = df[feature_cols]
    y = df[TARGET]
    return X, y


# ─────────────────────────────────────────────────────────
# 2. METRICS HELPER
# ─────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2":   round(r2_score(y_true, y_pred), 4),
    }


# ─────────────────────────────────────────────────────────
# 3. TRAINING + MLFLOW LOGGING
# ─────────────────────────────────────────────────────────
def train_and_log(run_name, model, X_train, X_test, y_train, y_test,
                  params, experiment_id):
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        mlflow.log_params(params)

        # Build sklearn Pipeline (scaler + estimator)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  model),
        ])
        pipe.fit(X_train, y_train)

        y_pred  = pipe.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        # Also save locally so predict_pipeline.py can load it
        save_path = os.path.join(MODEL_DIR, f"{run_name}.pkl")
        joblib.dump(pipe, save_path)
        mlflow.log_artifact(save_path)

        print(f"\n{'─'*52}")
        print(f"  Run   : {run_name}")
        print(f"  MAE   : {metrics['MAE']}")
        print(f"  RMSE  : {metrics['RMSE']}")
        print(f"  R²    : {metrics['R2']}")
        print(f"  Saved : {save_path}")

        return metrics, save_path


# ─────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 52)
    print("  Retail Demand Forecasting — Training Pipeline")
    print("=" * 52)

    print("\n[1/5] Loading & preprocessing data...")
    X, y = load_and_preprocess(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Train rows : {len(X_train)}")
    print(f"      Test  rows : {len(X_test)}")

    mlflow.set_experiment(EXPERIMENT)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id

    # ── EXPERIMENT 1 : Linear Regression (baseline) ──────
    print("\n[2/5] Running Experiment 1 — Linear Regression (baseline)...")
    lr_params = {
        "model_type": "LinearRegression",
        "fit_intercept": True,
        "note": "Simple baseline — straight line through the data",
    }
    lr_model = LinearRegression(fit_intercept=True)
    metrics_lr, path_lr = train_and_log(
        run_name      = "Exp1_LinearRegression",
        model         = lr_model,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params        = lr_params,
        experiment_id = exp_id,
    )

    # ── EXPERIMENT 2 : Random Forest ─────────────────────
    print("\n[3/5] Running Experiment 2 — Random Forest...")
    rf_params = {
        "model_type":        "RandomForestRegressor",
        "n_estimators":      100,
        "max_depth":         10,
        "min_samples_split": 5,
        "random_state":      42,
    }
    rf_model = RandomForestRegressor(
        n_estimators      = rf_params["n_estimators"],
        max_depth         = rf_params["max_depth"],
        min_samples_split = rf_params["min_samples_split"],
        random_state      = rf_params["random_state"],
        n_jobs            = -1,
    )
    metrics_rf, path_rf = train_and_log(
        run_name      = "Exp2_RandomForest",
        model         = rf_model,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params        = rf_params,
        experiment_id = exp_id,
    )

    # ── EXPERIMENT 3 : Gradient Boosting ─────────────────
    print("\n[4/5] Running Experiment 3 — Gradient Boosting...")
    gb_params = {
        "model_type":    "GradientBoostingRegressor",
        "n_estimators":  200,
        "learning_rate": 0.05,
        "max_depth":     5,
        "subsample":     0.8,
        "random_state":  42,
    }
    gb_model = GradientBoostingRegressor(
        n_estimators  = gb_params["n_estimators"],
        learning_rate = gb_params["learning_rate"],
        max_depth     = gb_params["max_depth"],
        subsample     = gb_params["subsample"],
        random_state  = gb_params["random_state"],
    )
    metrics_gb, path_gb = train_and_log(
        run_name      = "Exp3_GradientBoosting",
        model         = gb_model,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params        = gb_params,
        experiment_id = exp_id,
    )

    # ── PICK BEST — voting across all 3 metrics ───────────
    print("\n[5/5] Selecting best model...")

    all_models = {
        "Exp1_LinearRegression": (metrics_lr, path_lr),
        "Exp2_RandomForest":     (metrics_rf, path_rf),
        "Exp3_GradientBoosting": (metrics_gb, path_gb),
    }

    # Score each model: +1 for best MAE, +1 for best RMSE, +1 for best R²
    scores = {name: 0 for name in all_models}
    best_mae  = min(all_models, key=lambda n: all_models[n][0]["MAE"])
    best_rmse = min(all_models, key=lambda n: all_models[n][0]["RMSE"])
    best_r2   = max(all_models, key=lambda n: all_models[n][0]["R2"])
    scores[best_mae]  += 1
    scores[best_rmse] += 1
    scores[best_r2]   += 1

    best_name = max(scores, key=scores.get)
    best_path = all_models[best_name][1]

    ref_file = os.path.join(MODEL_DIR, "best_model_path.txt")
    with open(ref_file, "w") as f:
        f.write(best_path)

    print(f"\n{'='*52}")
    print(f"  RESULTS SUMMARY")
    print(f"  Exp1 LinearRegression  → R²={metrics_lr['R2']}  MAE={metrics_lr['MAE']}  RMSE={metrics_lr['RMSE']}")
    print(f"  Exp2 RandomForest      → R²={metrics_rf['R2']}  MAE={metrics_rf['MAE']}  RMSE={metrics_rf['RMSE']}")
    print(f"  Exp3 GradientBoosting  → R²={metrics_gb['R2']}  MAE={metrics_gb['MAE']}  RMSE={metrics_gb['RMSE']}")
    print(f"\n  Voting scores: {scores}")
    print(f"\n  ✓ Best model : {best_name}  (score: {scores[best_name]}/3)")
    print(f"  ✓ Saved to   : {best_path}")
    print(f"\n  Run 'mlflow ui' → http://localhost:5001 to compare experiments.")
    print("=" * 52)


if __name__ == "__main__":
    main()