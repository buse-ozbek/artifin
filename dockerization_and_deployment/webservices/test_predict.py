"""
test_predict.py
===============
End-to-end sanity tests for the standalone predict.py webservice.

NOTE: These tests import predict.py, which loads the real model from
MLflow at import time. They therefore require:
  - MLflow server running on http://127.0.0.1:5001
  - 'retail_demand_model' registered with the 'Staging' alias

CI does NOT run this file (no MLflow in the GitHub Actions runner) —
it's here for parallelism with the teacher's structure and for manual
local verification before pushing.

To run locally:
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
    pytest dockerization_and_deployment/webservices/test_predict.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


SAMPLE_INPUT = {
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
    "seasonality": "Winter",
}


def test_predict_returns_non_negative_int():
    """End-to-end: the real MLflow model produces valid output for a
    realistic input. If this fails, something is wrong with model
    loading or the prediction pipeline as a whole."""
    import predict
    result = predict.predict(SAMPLE_INPUT)
    assert isinstance(result, int)
    assert result >= 0


def test_app_has_predict_endpoint():
    """The FastAPI app must expose POST /predict — otherwise the
    deployed container can't serve any requests."""
    import predict
    paths_and_methods = [
        (route.path, route.methods) for route in predict.app.routes
        if hasattr(route, "methods")
    ]
    assert any(
        path == "/predict" and "POST" in methods
        for path, methods in paths_and_methods
    )
