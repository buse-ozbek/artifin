"""
test_calculate_metrics.py
=========================
Tests for the monitoring pipeline's metrics script.

The main risks in calculate_metrics.py are:
  1. FEATURES list gets out of sync with reference.csv column names
     (silent: KS test silently skips, drift is under-reported)
  2. FEATURES count drifts from the model's actual feature count
  3. A feature name accidentally gets typed in PascalCase or with
     spaces, breaking the column lookup
  4. The main() entry point gets refactored away

These 5 tests pin the contract down.
"""

import sys
from pathlib import Path

# make monitoring/scripts importable regardless of where pytest runs from
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def test_features_defined():
    """FEATURES is the loop variable for drift detection. If it's
    missing, the whole monitoring script breaks at runtime."""
    import calculate_metrics as cm
    assert hasattr(cm, "FEATURES"), "FEATURES is not defined in calculate_metrics.py"


def test_features_has_15_items():
    """The model is trained on 15 features; monitoring must check all 15.
    If someone adds a feature to the model and forgets here, drift on
    that feature would be silently invisible."""
    from calculate_metrics import FEATURES
    assert isinstance(FEATURES, list)
    assert len(FEATURES) == 15, f"expected 15 features, got {len(FEATURES)}"


def test_features_match_expected_names():
    """FEATURES names must exactly match the columns in reference.csv
    and current_batches/*.csv — otherwise the KS-test lookup fails or
    silently skips features."""
    from calculate_metrics import FEATURES
    expected = [
        "store_id", "product_id", "category", "region",
        "inventory_level", "demand_forecast", "price", "discount",
        "weather_condition", "holiday_promotion", "competitor_pricing",
        "seasonality", "year", "month", "day_of_week",
    ]
    assert FEATURES == expected


def test_features_are_snake_case():
    """Feature names must be lowercase snake_case to match the cleaned
    column names in the monitoring CSVs. A stray 'Store_ID' or 'price '
    would silently miss the column."""
    from calculate_metrics import FEATURES
    for f in FEATURES:
        assert f.islower(), f"Feature {f!r} is not lowercase"
        assert " " not in f, f"Feature {f!r} contains a space"
        assert f.replace("_", "").isalnum(), f"Feature {f!r} has weird characters"


def test_main_function_exists():
    """The script entry point must be present — Prefect / cron jobs
    that call calculate_metrics.main() would break otherwise."""
    import calculate_metrics
    assert hasattr(calculate_metrics, "main")
    assert callable(calculate_metrics.main)
