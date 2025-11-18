from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class IntTransformer(BaseEstimator, TransformerMixin):
    """Convert specified columns to integer dtype."""

    def __init__(self, columns=None):
        """
        Initialize transformer.

        Args:
            columns (list): Columns to convert to int.
        """
        self.columns = columns

    def fit(self, X, y=None):
        """Record feature names and validate columns exist and have correct dtype."""
        self.feature_names_in_ = X.columns.tolist()
        
        if self.columns is None:
            raise ValueError(f"IntTransformer - columns not provided")
        missing_columns = [col for col in self.columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"IntTransformer - columns not found: {missing_columns}")
        invalid_columns = [col for col in self.columns if X[col].dtype not in ['object', 'float64']]
        if invalid_columns:
            raise ValueError(f"IntTransformer - invalid column dtype: {invalid_columns}")
        return self

    def transform(self, X):
        """Convert specified columns to int."""
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = pd.to_numeric(X_copy[col]).astype(int)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        return getattr(self, "feature_names_in_", input_features)
