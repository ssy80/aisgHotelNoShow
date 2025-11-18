from sklearn.base import BaseEstimator, TransformerMixin

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drop specified columns from a DataFrame."""

    def __init__(self, columns):
        """
        Initialize transformer.

        Args:
            columns (list): Columns to drop.
        """
        self.columns = columns

    def fit(self, X, y=None):
        """Record feature names."""
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """Return DataFrame without the specified columns."""
        return X.drop(columns=self.columns)

    def get_feature_names_out(self, input_features=None):
        """Return feature names after dropping columns."""
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        return [f for f in input_features if f not in self.columns]
