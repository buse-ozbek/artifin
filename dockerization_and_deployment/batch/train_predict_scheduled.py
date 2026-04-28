"""
train_predict_scheduled.py
==========================
Retail Store Inventory — Prefect Flow
Wraps pipeline.py logic into Prefect tasks and a top-level flow.

Usage (direct test):
    python train_predict_scheduled.py

Then use deploy.py to schedule it via Prefect.
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

from prefect import flow, task
from prefect.logging import get_run_logger

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
DATA_PATH  = "retail_store_inventory.csv"
TARGET     = "Units Sold"
EXPERIMENT = "Retail_Demand_Forecasting"
MODEL_DIR  = "models"


# ─────────────────────────────────────────────────────────
# TASK 1 — Load & preprocess
# ─────────────────────────────────────────────────────────
@task(name="load-and-preprocess", retries=2, retry_delay_seconds=10)
def load_and_preprocess(path: str):
    logger = get_run_logger()
    logger.info(f"Loading data from: {path}")

    df = pd.read_csv(path)

    df["Date"]      = pd.to_datetime(df["Date"])
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

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

    logger.info(f"Loaded {len(df)} rows — {len(X.columns)} features")
    return X, y


# ─────────────────────────────────────────────────────────
# TASK 2 — Split data
# ─────────────────────────────────────────────────────────
@task(name="split-data")
def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    logger = get_run_logger()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────
# HELPER — metrics
# ─────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2":   round(r2_score(y_true, y_pred), 4),
    }


# ─────────────────────────────────────────────────────────
# TASK 3 — Train & log one model
# ─────────────────────────────────────────────────────────
@task(name="train-and-log-model", retries=1, retry_delay_seconds=5)
def train_and_log(run_name, model, X_train, X_test, y_train, y_test,
                  params, experiment_id):
    logger = get_run_logger()
    logger.info(f"Training: {run_name}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        mlflow.log_params(params)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  model),
        ])
        pipe.fit(X_train, y_train)

        y_pred  = pipe.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        save_path = os.path.join(MODEL_DIR, f"{run_name}.pkl")
        joblib.dump(pipe, save_path)
        mlflow.log_artifact(save_path)

        logger.info(f"{run_name} → MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  R²={metrics['R2']}")

    return metrics, save_path


# ─────────────────────────────────────────────────────────
# TASK 4 — Pick best model by voting
# ─────────────────────────────────────────────────────────
@task(name="select-best-model")
def select_best_model(all_models: dict) -> str:
    logger = get_run_logger()

    scores = {name: 0 for name in all_models}
    best_mae  = min(all_models, key=lambda n: all_models[n][0]["MAE"])
    best_rmse = min(all_models, key=lambda n: all_models[n][0]["RMSE"])
    best_r2   = max(all_models, key=lambda n: all_models[n][0]["R2"])
    scores[best_mae]  += 1
    scores[best_rmse] += 1
    scores[best_r2]   += 1

    best_name = max(scores, key=scores.get)
    best_path = all_models[best_name][1]

    logger.info(f"Voting scores: {scores}")
    logger.info(f"Best model: {best_name}  (score: {scores[best_name]}/3)")
    logger.info(f"Saved at: {best_path}")

    ref_file = os.path.join(MODEL_DIR, "best_model_path.txt")
    with open(ref_file, "w") as f:
        f.write(best_path)

    return best_name, best_path


# ─────────────────────────────────────────────────────────
# TASK 5 — Register best model in MLflow Model Registry
# ─────────────────────────────────────────────────────────
@task(name="register-model", retries=2, retry_delay_seconds=10)
def register_model(registered_model_name: str = "retail_demand_model"):
    logger = get_run_logger()

    model_uri = f"runs:/{mlflow.last_active_run().info.run_id}/model"
    logger.info(f"Registering model URI: {model_uri}")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="Staging",
        version=result.version,
    )

    logger.info(f"Registered: models:/{registered_model_name}@Staging  (v{result.version})")
    return result.version


# ─────────────────────────────────────────────────────────
# MAIN FLOW
# ─────────────────────────────────────────────────────────
@flow(
    name="retail-demand-training-pipeline",
    description="Trains 3 models, picks the best, registers it in MLflow Staging.",
)
def retail_demand_training_pipeline(
    data_path: str = DATA_PATH,
    mlflow_tracking_uri: str = "http://127.0.0.1:5001",
):
    logger = get_run_logger()
    logger.info("=" * 52)
    logger.info("  Retail Demand Forecasting — Training Pipeline")
    logger.info("=" * 52)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id

    # Step 1 — Load & split
    X, y = load_and_preprocess(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 2 — Train all three models
    lr_params = {
        "model_type": "LinearRegression",
        "fit_intercept": True,
        "note": "Simple baseline",
    }
    metrics_lr, path_lr = train_and_log(
        run_name="Exp1_LinearRegression",
        model=LinearRegression(fit_intercept=True),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params=lr_params, experiment_id=exp_id,
    )

    rf_params = {
        "model_type": "RandomForestRegressor",
        "n_estimators": 100, "max_depth": 10,
        "min_samples_split": 5, "random_state": 42,
    }
    metrics_rf, path_rf = train_and_log(
        run_name="Exp2_RandomForest",
        model=RandomForestRegressor(
            n_estimators=100, max_depth=10,
            min_samples_split=5, random_state=42, n_jobs=-1,
        ),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params=rf_params, experiment_id=exp_id,
    )

    gb_params = {
        "model_type": "GradientBoostingRegressor",
        "n_estimators": 200, "learning_rate": 0.05,
        "max_depth": 5, "subsample": 0.8, "random_state": 42,
    }
    metrics_gb, path_gb = train_and_log(
        run_name="Exp3_GradientBoosting",
        model=GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42,
        ),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        params=gb_params, experiment_id=exp_id,
    )

    # Step 3 — Pick best & register
    all_models = {
        "Exp1_LinearRegression": (metrics_lr, path_lr),
        "Exp2_RandomForest":     (metrics_rf, path_rf),
        "Exp3_GradientBoosting": (metrics_gb, path_gb),
    }
    best_name, best_path = select_best_model(all_models)
    version = register_model()

    logger.info("=" * 52)
    logger.info(f"  Best model : {best_name}")
    logger.info(f"  Saved to   : {best_path}")
    logger.info(f"  Registry   : models:/retail_demand_model@Staging  (v{version})")
    logger.info("=" * 52)

    return best_name, best_path


# ─────────────────────────────────────────────────────────
# Direct run (for testing without Prefect server)
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    retail_demand_training_pipeline()
