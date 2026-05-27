"""
test_prediction_core.py
=======================
Unit tests for the retail demand prediction service.

Uses a DummyModel so no MLflow server or real model artifact is needed —
runs fully in CI. Each test targets one real failure mode in the
preprocessing or validation pipeline.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import pytest
from prediction_core import (
    predict, preprocess, validate_input,
    FEATURE_COLS,
)


# ─────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────

class DummyModel:
    """Stand-in for the real MLflow model. Returns a configurable output."""
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
# PREPROCESSING TESTS
# ─────────────────────────────────────────────────────────

def test_preprocess_produces_15_features_in_order():
    """The model expects exactly 15 features in a fixed order.
    Reordering or adding/removing a feature breaks the model."""
    df = preprocess(VALID_INPUT)
    assert list(df.columns) == FEATURE_COLS
    assert df.shape == (1, 15)


def test_preprocess_encodes_categoricals_correctly():
    """Categorical strings must map to the same integer codes that
    LabelEncoder produced during training — otherwise the model sees
    the wrong feature values."""
    df = preprocess(VALID_INPUT)
    assert df["Category"].iloc[0] == 1            # Electronics
    assert df["Region"].iloc[0] == 1              # North
    assert df["Weather Condition"].iloc[0] == 3   # Sunny
    assert df["Seasonality"].iloc[0] == 3         # Winter


def test_preprocess_extracts_date_features():
    """Year, Month, DayOfWeek must be derived from the date string."""
    df = preprocess(VALID_INPUT)
    assert df["Year"].iloc[0] == 2024
    assert df["Month"].iloc[0] == 3
    assert df["DayOfWeek"].iloc[0] == 4   # 2024-03-15 is a Friday (Mon=0)


def test_preprocess_parses_store_and_product_ids():
    """Store/Product IDs like 'S001' / 'P0005' must be parsed to
    zero-indexed integers (S001 → 0, P0005 → 4) to match training."""
    df = preprocess(VALID_INPUT)
    assert df["Store ID"].iloc[0] == 0
    assert df["Product ID"].iloc[0] == 4


# ─────────────────────────────────────────────────────────
# PREDICTION OUTPUT TESTS
# ─────────────────────────────────────────────────────────

def test_valid_prediction_returns_non_negative_int():
    """Demand is always a non-negative whole number of units."""
    result = predict(VALID_INPUT, DummyModel(output=42.0))
    assert isinstance(result, int)
    assert result >= 0


def test_prediction_rounds_model_output():
    """The model returns floats; predictions must be rounded to int."""
    assert predict(VALID_INPUT, DummyModel(output=42.7)) == 43
    assert predict(VALID_INPUT, DummyModel(output=42.3)) == 42


def test_negative_prediction_clipped_to_zero():
    """The model can predict negative numbers (regression has no
    lower bound), but demand can't be negative — must clip to 0."""
    assert predict(VALID_INPUT, DummyModel(output=-5.0)) == 0


# ─────────────────────────────────────────────────────────
# INPUT VALIDATION — UNKNOWN CATEGORICAL VALUES
# ─────────────────────────────────────────────────────────
# Without these checks, an unknown value silently becomes NaN
# and the model returns garbage. This is the most dangerous bug.

def test_unknown_category_raises():
    bad = dict(VALID_INPUT, category="Books")
    with pytest.raises(ValueError):
        validate_input(bad)


def test_unknown_region_raises():
    bad = dict(VALID_INPUT, region="Mars")
    with pytest.raises(ValueError):
        validate_input(bad)


def test_unknown_weather_raises():
    bad = dict(VALID_INPUT, weather_condition="Foggy")
    with pytest.raises(ValueError):
        validate_input(bad)


def test_unknown_seasonality_raises():
    bad = dict(VALID_INPUT, seasonality="Monsoon")
    with pytest.raises(ValueError):
        validate_input(bad)


# ─────────────────────────────────────────────────────────
# INPUT VALIDATION — MISSING / MALFORMED FIELDS
# ─────────────────────────────────────────────────────────

def test_missing_date_raises():
    bad = dict(VALID_INPUT)
    del bad["date"]
    with pytest.raises(ValueError):
        validate_input(bad)


def test_missing_price_raises():
    bad = dict(VALID_INPUT)
    del bad["price"]
    with pytest.raises(ValueError):
        validate_input(bad)


def test_invalid_date_raises():
    """Unparseable date strings must be caught at validation, not
    crash deep inside pd.to_datetime."""
    bad = dict(VALID_INPUT, date="not-a-date")
    with pytest.raises(ValueError):
        validate_input(bad)


def test_bad_store_id_raises():
    """A store_id with no digits ('Store') would silently become NaN
    in str.extract, then crash on astype(int). Must be caught."""
    bad = dict(VALID_INPUT, store_id="StoreNoNumber")
    with pytest.raises(ValueError):
        preprocess(bad)


def test_bad_product_id_raises():
    """Same risk as store_id — product_id must contain a number."""
    bad = dict(VALID_INPUT, product_id="ProductNoNumber")
    with pytest.raises(ValueError):
        preprocess(bad)
