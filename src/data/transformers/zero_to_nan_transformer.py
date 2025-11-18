from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ZeroToNaNTransformer(BaseEstimator, TransformerMixin):
    """
    Converts 0 values in a column to NaN.
    """

    def __init__(self, column_name='price'):
        self.column_name = column_name

    def fit(self, X, y=None):
        """
        Confirms the column exists and stores feature names.
        """
        self.feature_names_in_ = X.columns.tolist()

        if self.column_name not in X.columns:
            raise ValueError(
                f"ZeroToNaNTransformer - column not found: {self.column_name}"
            )
        return self

    def transform(self, X):
        """
        Replaces all 0 values in the column with NaN.
        """
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """
        Returns the original feature names.
        """
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return list(input_features)
