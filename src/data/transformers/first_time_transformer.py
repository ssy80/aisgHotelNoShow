from sklearn.base import BaseEstimator, TransformerMixin

class FirstTimeTransformer(BaseEstimator, TransformerMixin):
    """Convert 'first_time' column values from 'yes'/'no' to 1/0."""

    def __init__(self, column_name='first_time'):
        """
        Initialize transformer.

        Args:
            column_name (str): Name of the column to transform.
        """
        self.column_name = column_name

    def fit(self, X, y=None):
        """Record feature names and check column exists."""
        self.feature_names_in_ = X.columns.tolist()
        if self.column_name not in X.columns:
            raise ValueError(f"FirstTimeTransformer - column not found: {self.column_name}")
        return self

    def transform(self, X):
        """Convert 'yes' to 1 and 'no' to 0."""
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].apply(lambda x: 1 if x == "yes" else 0)
        self._validate_data(X_copy)
        return X_copy

    def _validate_data(self, X_copy):
        """Ensure transformed values are 0 or 1."""
        for val in X_copy[self.column_name]:
            if val not in (0, 1):
                raise ValueError(f"FirstTimeTransformer - invalid data: {self.column_name} contains {val}")

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        return list(getattr(self, "feature_names_in_", []))
