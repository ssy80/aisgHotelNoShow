from sklearn.base import BaseEstimator, TransformerMixin

class DropNaTransformer(BaseEstimator, TransformerMixin):
    """Drop rows with too many missing values."""

    def __init__(self, n=3):
        """
        Initialize transformer.

        Args:
            n (int): Minimum number of non-null values required to keep a row.
        """
        self.n = n

    def fit(self, X, y=None):
        """Record feature names."""
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """Drop rows that have fewer than n non-null values."""
        X_copy = X.copy()
        thresh = len(X_copy.columns) - self.n + 1
        X_copy = X_copy.dropna(thresh=thresh)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names (rows were dropped, columns unchanged)."""
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        return list(input_features)
