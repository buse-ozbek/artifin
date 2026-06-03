"""
test_prediction_core.py
=======================
Unit tests for the retail demand prediction service.

Uses a DummyModel so no MLflow server or real model artifact is needed —
runs fully in CI.

Each test checks that an encoded value falls within the valid range
for that field — either a fixed set of allowed values (categories,
regions, etc.) or a logical constraint (non-negative price, valid
percentage range, etc.).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from prediction_core import predict, preprocess, FEATURE_COLS


class DummyModel:
    """Stand-in for the real MLflow model."""
    def __init__(self, output=42.0):
        self.output = output
    def predict(self, X):
        return [self.output]


VALID_INPUT = {
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


# ─────────────────────────────────────────────────────────
# PREPROCESSING — output shape
# ─────────────────────────────────────────────────────────

def test_preprocess_produces_15_features_in_order():
    """The model expects exactly 15 features in a fixed order."""
    df = preprocess(VALID_INPUT)
    assert list(df.columns) == FEATURE_COLS
    assert df.shape == (1, 15)


# ─────────────────────────────────────────────────────────
# DATE — values must be within real calendar ranges
# ─────────────────────────────────────────────────────────

def test_month_is_between_1_and_12():
    """Month must be a real month number, 1 through 12."""
    df = preprocess(VALID_INPUT)
    assert 1 <= df["Month"].iloc[0] <= 12


def test_dayofweek_is_between_0_and_6():
    """Day of week follows pandas convention: Monday=0, Sunday=6."""
    df = preprocess(VALID_INPUT)
    assert 0 <= df["DayOfWeek"].iloc[0] <= 6


# ─────────────────────────────────────────────────────────
# CATEGORICAL — encoded value must be one from my dataset
# ─────────────────────────────────────────────────────────

def test_category_is_one_of_my_5_categories():
    """My data has 5 categories: Clothing, Electronics, Furniture,
    Groceries, Toys. Encoded as 0-4."""
    df = preprocess(VALID_INPUT)
    assert df["Category"].iloc[0] in [0, 1, 2, 3, 4]


def test_region_is_one_of_my_4_regions():
    """My data has 4 regions: East, North, South, West. Encoded as 0-3."""
    df = preprocess(VALID_INPUT)
    assert df["Region"].iloc[0] in [0, 1, 2, 3]


def test_weather_is_one_of_my_4_weather_conditions():
    """My data has 4 weather conditions: Cloudy, Rainy, Snowy, Sunny.
    Encoded as 0-3."""
    df = preprocess(VALID_INPUT)
    assert df["Weather Condition"].iloc[0] in [0, 1, 2, 3]


def test_seasonality_is_one_of_my_4_seasons():
    """My data has 4 seasons: Autumn, Spring, Summer, Winter. Encoded as 0-3."""
    df = preprocess(VALID_INPUT)
    assert df["Seasonality"].iloc[0] in [0, 1, 2, 3]


# ─────────────────────────────────────────────────────────
# STORE / PRODUCT IDs — must match my dataset's count
# ─────────────────────────────────────────────────────────

def test_store_id_is_one_of_my_5_stores():
    """My data has 5 stores: S001-S005. After zero-indexing: 0-4."""
    df = preprocess(VALID_INPUT)
    assert df["Store ID"].iloc[0] in [0, 1, 2, 3, 4]


def test_product_id_is_one_of_my_20_products():
    """My data has 20 products: P0001-P0020. After zero-indexing: 0-19."""
    df = preprocess(VALID_INPUT)
    assert 0 <= df["Product ID"].iloc[0] <= 19


# ─────────────────────────────────────────────────────────
# NUMERIC FIELDS — logical constraints
# ─────────────────────────────────────────────────────────

def test_price_is_non_negative():
    """Price can't be negative — it doesn't make sense for a product
    to cost less than zero."""
    df = preprocess(VALID_INPUT)
    assert df["Price"].iloc[0] >= 0


def test_discount_is_between_0_and_100_percent():
    """Discount is a percentage. It must be between 0% (no discount)
    and 100% (free), nothing outside that range is meaningful."""
    df = preprocess(VALID_INPUT)
    assert 0 <= df["Discount"].iloc[0] <= 100


def test_inventory_level_is_non_negative():
    """You can't have negative inventory."""
    df = preprocess(VALID_INPUT)
    assert df["Inventory Level"].iloc[0] >= 0


def test_demand_forecast_is_non_negative():
    """A forecast for units sold can't be negative."""
    df = preprocess(VALID_INPUT)
    assert df["Demand Forecast"].iloc[0] >= 0


def test_competitor_pricing_is_non_negative():
    """Competitor's price can't be negative either."""
    df = preprocess(VALID_INPUT)
    assert df["Competitor Pricing"].iloc[0] >= 0


# ─────────────────────────────────────────────────────────
# BINARY FLAG
# ─────────────────────────────────────────────────────────

def test_holiday_promotion_is_0_or_1():
    """Holiday/Promotion is a binary flag — either yes (1) or no (0)."""
    df = preprocess(VALID_INPUT)
    assert df["Holiday/Promotion"].iloc[0] in [0, 1]


# ─────────────────────────────────────────────────────────
# PREDICTION OUTPUT
# ─────────────────────────────────────────────────────────

def test_prediction_is_non_negative_integer():
    """Predicted demand must be a non-negative whole number of units."""
    result = predict(VALID_INPUT, DummyModel(output=42.0))
    assert isinstance(result, int)
    assert result >= 0


def test_negative_model_output_clipped_to_zero():
    """Even if the model predicts a negative number, demand can't be
    negative — must clip to 0."""
    result = predict(VALID_INPUT, DummyModel(output=-5.0))
    assert result == 0