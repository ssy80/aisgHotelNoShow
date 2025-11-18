from sklearn.base import BaseEstimator, TransformerMixin

class HasChildrenTransformer(BaseEstimator, TransformerMixin):
    """Create a binary 'has_children' feature from the number of children column."""

    def __init__(self, column_name=None):
        """
        Initialize transformer.

        Args:
            column_name (str): Name of the column containing number of children.
        """
        self.column_name = column_name
        self.added_features_ = ["has_children"]

    def fit(self, X, y=None):
        """Record feature names and check column exists."""
        self.feature_names_in_ = X.columns.tolist()
        if self.column_name is None or self.column_name not in X.columns:
            raise ValueError(f"HasChildrenTransformer - column not found: {self.column_name}")
        return self

    def transform(self, X):
        """Create 'has_children' column: 1 if number of children > 0, else 0."""
        X_copy = X.copy()
        X_copy["has_children"] = (X_copy[self.column_name] > 0).astype(int)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names after adding 'has_children'."""
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        return list(input_features) + self.added_features_
