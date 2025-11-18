from sklearn.base import BaseEstimator, TransformerMixin

class CheckoutDayTransformer(BaseEstimator, TransformerMixin):
    """Convert negative checkout_day values to positive."""

    def __init__(self, column_name='checkout_day'):
        """
        Initialize the transformer.

        Args:
            column_name (str): Name of the checkout day column.
        """
        self.column_name = column_name

    def fit(self, X, y=None):
        """Record feature names."""
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """Replace negative values in checkout_day with their absolute values."""
        X_copy = X.copy()
        if self.column_name in X_copy.columns:
            X_copy[self.column_name] = X_copy[self.column_name].apply(
                lambda x: -x if x < 0 else x
            )
        else:
            raise ValueError(f"CheckoutDayTransformer - column not found: {self.column_name}")
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names after transformation."""
        return list(getattr(self, "feature_names_in_", []))
