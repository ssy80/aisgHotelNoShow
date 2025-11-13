import pytest
import pandas as pd
import numpy as np
from data.transformers.log1p_transformer import Log1pTransformer


# ---------- Fixtures ----------

@pytest.fixture
def sample_df():
    """Sample dataframe with positive, zero, and negative values."""
    return pd.DataFrame({
        "price": [10, 0, -5, 20],
        "quantity": [1, 0, 3, -1],
        "other": [5, 6, 7, 8]  # untouched column
    })


@pytest.fixture
def transformer():
    """Fixture for Log1pTransformer applied to 'price' and 'quantity'."""
    return Log1pTransformer(columns=["price", "quantity"])


# ---------- Tests ----------

def test_log1p_transformation(sample_df, transformer):
    """Check that transform applies log1p correctly after clipping at 0."""
    transformer.fit(sample_df)
    transformed = transformer.transform(sample_df)

    # Values must be >= 0 after clipping
    assert (transformed[["price", "quantity"]] >= 0).all().all()

    # Check that log1p was applied manually
    expected_price = np.log1p(np.clip(sample_df["price"], 0, None))
    expected_quantity = np.log1p(np.clip(sample_df["quantity"], 0, None))

    pd.testing.assert_series_equal(transformed["price"], expected_price)
    pd.testing.assert_series_equal(transformed["quantity"], expected_quantity)

    # Other column should remain unchanged
    pd.testing.assert_series_equal(transformed["other"], sample_df["other"])


def test_fit_raises_error_for_missing_column(sample_df):
    """Ensure ValueError is raised if a specified column doesn't exist."""
    
    transformer = Log1pTransformer(columns=["missing_col"])
    with pytest.raises(ValueError, match="column not found"):
        transformer.fit(sample_df)


def test_fit_raises_error_if_columns_is_none(sample_df):
    """Ensure ValueError is raised if columns=None."""

    transformer = Log1pTransformer(columns=None)
    with pytest.raises(ValueError, match="column not found"):
        transformer.fit(sample_df)


def test_get_feature_names_out(sample_df, transformer):
    """Ensure get_feature_names_out returns correct feature names."""

    transformer.fit(sample_df)
    feature_names = transformer.get_feature_names_out()
    assert feature_names == sample_df.columns.tolist()


def test_transform_does_not_modify_original_df(sample_df, transformer):
    """Ensure the original dataframe is not modified."""

    transformer.fit(sample_df)
    original_copy = sample_df.copy()
    transformer.transform(sample_df)
    pd.testing.assert_frame_equal(sample_df, original_copy)
