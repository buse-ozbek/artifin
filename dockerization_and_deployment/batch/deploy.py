"""
deploy.py
=========
Creates and serves the Prefect deployment for the retail demand training pipeline.

Before running:
    1. Start the Prefect server in another terminal:
           conda activate iris-prefect
           prefect server start

    2. Set the API URL in THIS terminal:
           export PREFECT_API_URL="http://127.0.0.1:4200/api"

    3. Make sure MLflow is running:
           mlflow ui --port 5001

Then run:
    python deploy.py

The flow is scheduled daily at 14:35. To trigger manually:
    prefect deployment run "retail-demand-training-pipeline/retail-daily-training"
"""

from dockerization_and_deployment.batch.train_predict_scheduled import retail_demand_training_pipeline

if __name__ == "__main__":
    retail_demand_training_pipeline.serve(
        name="retail-daily-training",
        cron="35 14 * * *",          # every day at 14:35
        parameters={
            "data_path": "retail_store_inventory.csv",
            "mlflow_tracking_uri": "http://127.0.0.1:5001",
        },
        tags=["retail", "ml", "demand-forecasting"],
        description="Daily retraining of the retail demand model. Picks the best of 3 models and registers it in MLflow Staging.",
    )
