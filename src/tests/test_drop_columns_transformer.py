import pytest
import pandas as pd
from data.transformers.drop_columns_transformer import DropColumnsTransformer


# ---------- Fixtures ----------

@pytest.fixture
def sample_df():
    """Fixture for a simple sample DataFrame."""
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9],
    })


@pytest.fixture
def transformer():
    """Fixture for a DropColumnsTransformer that drops column 'b'."""
    return DropColumnsTransformer(columns=["b"])


# ---------- Tests ----------

def test_drop_specified_columns(sample_df, transformer):
    """Ensure specified columns are dropped correctly."""
    transformed = transformer.fit_transform(sample_df)

    assert "b" not in transformed.columns, "Column 'b' was not dropped"
    assert list(transformed.columns) == ["a", "c"], "Unexpected remaining columns"


def test_other_columns_unchanged(sample_df, transformer):
    """Ensure non-dropped columns are unchanged."""
    transformed = transformer.fit_transform(sample_df)

    pd.testing.assert_series_equal(sample_df["a"], transformed["a"])
    pd.testing.assert_series_equal(sample_df["c"], transformed["c"])


def test_get_feature_names_out(sample_df, transformer):
    """Ensure get_feature_names_out excludes dropped columns."""
    transformer.fit(sample_df)
    output_features = transformer.get_feature_names_out()
    assert output_features == ["a", "c"], "Feature names not correctly updated"


def test_dropping_multiple_columns(sample_df):
    """Test that multiple columns can be dropped."""
    transformer = DropColumnsTransformer(columns=["a", "b"])
    transformed = transformer.fit_transform(sample_df)
    assert list(transformed.columns) == ["c"]


def test_raises_keyerror_for_missing_column(sample_df):
    """Ensure KeyError is raised when trying to drop a non-existent column."""
    transformer = DropColumnsTransformer(columns=["z"])
    with pytest.raises(KeyError):
        transformer.fit_transform(sample_df)
