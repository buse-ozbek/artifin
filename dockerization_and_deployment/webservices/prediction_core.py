"""
prediction_core.py
==================
Standalone core prediction logic for the retail demand webservice.

Lives independently of training_pipeline/app.py — exists so the
prediction logic can be unit-tested in CI without a running MLflow
server or real model artifact. Tests inject a DummyModel.
"""

import pandas as pd

# ─────────────────────────────────────────────────────────
# CONFIG — must match pipeline.py
# ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Demand Forecast", "Price", "Discount",
    "Weather Condition", "Holiday/Promotion", "Competitor Pricing",
    "Seasonality", "Year", "Month", "DayOfWeek",
]

# Hardcoded encoder maps — same order LabelEncoder saw during training
CATEGORY_MAP    = {"Clothing": 0, "Electronics": 1, "Furniture": 2, "Groceries": 3, "Toys": 4}
REGION_MAP      = {"East": 0, "North": 1, "South": 2, "West": 3}
WEATHER_MAP     = {"Cloudy": 0, "Rainy": 1, "Snowy": 2, "Sunny": 3}
SEASONALITY_MAP = {"Autumn": 0, "Spring": 1, "Summer": 2, "Winter": 3}

REQUIRED_FIELDS = [
    "date", "store_id", "product_id", "category", "region",
    "inventory_level", "demand_forecast", "price", "discount",
    "weather_condition", "holiday_promotion", "competitor_pricing",
    "seasonality",
]


def validate_input(input_data: dict):
    """Raise ValueError on malformed input. Catches the failure modes
    that would otherwise produce silent NaNs or ugly 500s."""
    missing = [f for f in REQUIRED_FIELDS if f not in input_data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if input_data["category"] not in CATEGORY_MAP:
        raise ValueError(f"Unknown category: {input_data['category']}")
    if input_data["region"] not in REGION_MAP:
        raise ValueError(f"Unknown region: {input_data['region']}")
    if input_data["weather_condition"] not in WEATHER_MAP:
        raise ValueError(f"Unknown weather_condition: {input_data['weather_condition']}")
    if input_data["seasonality"] not in SEASONALITY_MAP:
        raise ValueError(f"Unknown seasonality: {input_data['seasonality']}")

    try:
        pd.to_datetime(input_data["date"])
    except Exception:
        raise ValueError(f"Invalid date: {input_data['date']}")


def preprocess(input_data: dict) -> pd.DataFrame:
    validate_input(input_data)
    date = pd.to_datetime(input_data["date"])

    row = {
        "Store ID":           input_data["store_id"],
        "Product ID":         input_data["product_id"],
        "Category":           CATEGORY_MAP[input_data["category"]],
        "Region":             REGION_MAP[input_data["region"]],
        "Inventory Level":    input_data["inventory_level"],
        "Demand Forecast":    input_data["demand_forecast"],
        "Price":              input_data["price"],
        "Discount":           input_data["discount"],
        "Weather Condition":  WEATHER_MAP[input_data["weather_condition"]],
        "Holiday/Promotion":  input_data["holiday_promotion"],
        "Competitor Pricing": input_data["competitor_pricing"],
        "Seasonality":        SEASONALITY_MAP[input_data["seasonality"]],
        "Year":               date.year,
        "Month":              date.month,
        "DayOfWeek":          date.dayofweek,
    }
    df = pd.DataFrame([row])

    df["Store ID"]   = df["Store ID"].astype(str).str.extract(r"(\d+)").astype(float)
    df["Product ID"] = df["Product ID"].astype(str).str.extract(r"(\d+)").astype(float)
    if df["Store ID"].isna().any() or df["Product ID"].isna().any():
        raise ValueError("store_id and product_id must contain a number, e.g. 'S001'")
    df["Store ID"]   = df["Store ID"].astype(int) - 1
    df["Product ID"] = df["Product ID"].astype(int) - 1

    return df[FEATURE_COLS]


def predict(input_data: dict, model):
    """Run a prediction. Model is injected so tests can use a DummyModel."""
    df = preprocess(input_data)
    raw = model.predict(df)
    return max(0, int(round(float(raw[0]))))
