import pytest
import pandas as pd
import numpy as np
import sys, os
from data.encoders.cyclical_encoder import CyclicalEncoder

@pytest.fixture
def sample_cyclical_df():
    """Fixture: small dataframe with cyclical features."""
    return pd.DataFrame({
        "arrival_month": [1, 6, 12],
        "arrival_day": [1, 15, 31],
        "checkout_month": [1, 6, 12],
        "checkout_day": [1, 15, 31],
        "booking_month": [1, 6, 12],
        "value": [10, 20, 30],            # non-cyclical feature
    })

@pytest.fixture
def cyclical_encoder():
    """Fixture: initialize encoder with periods."""

    cyclical_features = {
            "arrival_month": 12,
            "checkout_month": 12,
            "booking_month": 12,
            "arrival_day": 31,
            "checkout_day": 31
    }
    return CyclicalEncoder(columns_period_map=cyclical_features)


def test_fit_sets_feature_names(cyclical_encoder, sample_cyclical_df):
    """Ensure .fit() stores original feature names."""

    cyclical_encoder.fit(sample_cyclical_df)
    assert cyclical_encoder.feature_names_in_ == ["arrival_month", "arrival_day", "checkout_month", "checkout_day", "booking_month", "value"]
    assert set(cyclical_encoder.added_features_) == {
        "arrival_month_sin", "arrival_month_cos", "arrival_day_sin", "arrival_day_cos",
        "checkout_month_sin", "checkout_month_cos", "checkout_day_sin", "checkout_day_cos",
        "booking_month_sin", "booking_month_cos"
    }


def test_transform_adds_columns(cyclical_encoder, sample_cyclical_df):
    """Check that transform adds sin/cos columns and keeps originals."""

    cyclical_encoder.fit(sample_cyclical_df)
    transformed = cyclical_encoder.transform(sample_cyclical_df)

    # Ensure all expected columns are present
    expected_cols = [
        "arrival_month_sin", "arrival_month_cos",
        "arrival_day_sin", "arrival_day_cos",
        "checkout_month_sin", "checkout_month_cos",
        "checkout_day_sin", "checkout_day_cos",
        "booking_month_sin", "booking_month_cos",
        "value", "arrival_month", "arrival_day", "checkout_month", "checkout_day", "booking_month"
    ]
    assert all(col in transformed.columns for col in expected_cols)
    assert transformed.shape[1] == len(expected_cols), transformed.columns.tolist()


def test_cyclical_values_in_range(cyclical_encoder, sample_cyclical_df):
    """Ensure sine/cosine values are within [-1, 1]."""

    cyclical_encoder.fit(sample_cyclical_df)
    transformed = cyclical_encoder.transform(sample_cyclical_df)

    for col in ["arrival_month_sin", "arrival_month_cos", "arrival_day_sin", "arrival_day_cos",
                "checkout_month_sin", "checkout_month_cos", "checkout_day_sin", "checkout_day_cos",
                "booking_month_sin", "booking_month_cos"]:
        assert transformed[col].between(-1, 1).all(), f"{col} out of range"


def test_original_features_unchanged(cyclical_encoder, sample_cyclical_df):
    """Ensure original numerical values are unchanged."""

    cyclical_encoder.fit(sample_cyclical_df)
    transformed = cyclical_encoder.transform(sample_cyclical_df)
    pd.testing.assert_series_equal(sample_cyclical_df["value"], transformed["value"])


def test_get_feature_names_out(cyclical_encoder, sample_cyclical_df):
    """Ensure feature names output includes added features."""

    cyclical_encoder.fit(sample_cyclical_df)
    out_names = cyclical_encoder.get_feature_names_out()
    assert "arrival_month_sin" in out_names
    assert "arrival_day_cos" in out_names
    assert "value" in out_names


def test_handles_missing_column_gracefully(sample_cyclical_df):
    """If missing column in data, skip transformation without error."""

    enc = CyclicalEncoder(columns_period_map={"arrival_month": 12, "nonexistent": 7})
    enc.fit(sample_cyclical_df)
    transformed = enc.transform(sample_cyclical_df)

    # Should not add nonexistent_sin/cos columns
    assert not any(col.startswith("nonexistent") for col in transformed.columns)
