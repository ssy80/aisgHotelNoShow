import pytest
import pandas as pd
from data.transformers.checkout_day_transformer import CheckoutDayTransformer

# ---------- Fixtures ----------

@pytest.fixture
def sample_df():
    """Fixture for a simple dataframe with positive and negative checkout_day values."""
    return pd.DataFrame({
        "checkout_day": [-5, 1, 3, -10, 7],
        "other_col": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def transformer():
    """Fixture to create a CheckoutDayTransformer instance."""
    return CheckoutDayTransformer(column_name="checkout_day")


# ---------- Tests ----------

def test_transform_converts_negatives_to_positive(sample_df, transformer):
    """Ensure negative checkout_day values are converted to positive."""
    transformed = transformer.fit_transform(sample_df)

    # All values should now be >= 0
    assert all(transformed["checkout_day"] >= 0)

    # Ensure original non-negative values remain unchanged
    assert transformed.loc[transformed["checkout_day"] == 1].shape[0] == 1


def test_transform_does_not_change_other_columns(sample_df, transformer):
    """Ensure non-target columns are unaffected."""

    transformed = transformer.fit_transform(sample_df)

    # 'other_col' must be identical to input
    pd.testing.assert_series_equal(
        sample_df["other_col"],
        transformed["other_col"],
        check_dtype=False
    )


def test_raises_error_when_column_missing(sample_df):
    """Ensure ValueError is raised when target column not found."""
    
    bad_df = sample_df.drop(columns=["checkout_day"])
    transformer = CheckoutDayTransformer(column_name="checkout_day")

    with pytest.raises(ValueError, match="column not found"):
        transformer.transform(bad_df)


def test_feature_names_out_matches_input(sample_df, transformer):
    """Ensure get_feature_names_out returns original column names."""
    transformer.fit(sample_df)
    feature_names = transformer.get_feature_names_out()
    assert feature_names == sample_df.columns.tolist()
