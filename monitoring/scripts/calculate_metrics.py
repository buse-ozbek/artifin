"""
calculate_metrics.py
====================
Compares the latest batch in data/current_batches/ against the reference
dataset, computes drift + regression performance metrics, and writes
one row to the Postgres `metrics` table.

Usage:
    python monitoring/scripts/calculate_metrics.py
"""

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Numeric features we'll check for drift (all 15 features)
FEATURES = [
    "store_id", "product_id", "category", "region",
    "inventory_level", "demand_forecast", "price", "discount",
    "weather_condition", "holiday_promotion", "competitor_pricing",
    "seasonality", "year", "month", "day_of_week",
]


def main():
    # load reference data
    reference = pd.read_csv("data/reference.csv")

    # load latest batch
    batch_files = glob.glob("data/current_batches/*.csv")
    latest_file = max(batch_files, key=os.path.getmtime)
    current = pd.read_csv(latest_file)
    batch_id = os.path.basename(latest_file).replace(".csv", "")

    # compute drift on each feature using Kolmogorov-Smirnov test
    drifted_features = 0
    for feature in FEATURES:
        _, p_value = ks_2samp(reference[feature], current[feature])
        if p_value < 0.05:  # p < 0.05 → distributions are statistically different
            drifted_features += 1

    share_drifted = drifted_features / len(FEATURES)

    # compute regression performance on the batch
    y_true = current["target"]
    y_pred = current["prediction"]
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))

    # compute prediction distribution stats (regression equivalent of class shares)
    pred_mean = float(current["prediction"].mean())
    pred_min  = float(current["prediction"].min())
    pred_max  = float(current["prediction"].max())

    # connect to Postgres (matches docker-compose.yml credentials)
    conn = psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "monitoring"),
        user=os.getenv("POSTGRES_USER", "retail"),
        password=os.getenv("POSTGRES_PASSWORD", "retail123"),
    )

    with conn.cursor() as cur:
        # create table if it does not exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp TIMESTAMP,
                batch_id TEXT,
                batch_size INT,
                num_drifted_features INT,
                share_drifted_features FLOAT,
                mae FLOAT,
                rmse FLOAT,
                r2 FLOAT,
                pred_mean FLOAT,
                pred_min FLOAT,
                pred_max FLOAT
            );
        """)

        # insert one row for this batch
        cur.execute("""
            INSERT INTO metrics (
                timestamp,
                batch_id,
                batch_size,
                num_drifted_features,
                share_drifted_features,
                mae,
                rmse,
                r2,
                pred_mean,
                pred_min,
                pred_max
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            datetime.utcnow(),
            batch_id,
            len(current),
            drifted_features,
            share_drifted,
            mae,
            rmse,
            r2,
            pred_mean,
            pred_min,
            pred_max,
        ))

    conn.commit()
    conn.close()

    print("metrics saved to database")


if __name__ == "__main__":
    main()