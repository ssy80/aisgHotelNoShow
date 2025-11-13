from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CyclicalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns_period_map=None):
        self.columns_period_map = columns_period_map
        self.added_features_ = []  # must initialize
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        self.added_features_ = [f"{col}_{suffix}" for col in self.columns_period_map for suffix in ['sin', 'cos']]
        return self
    
    def transform(self, X):
        X_copy = X.copy()

        for col, period in self.columns_period_map.items():
            if col in X_copy.columns:
                X_copy[f"{col}_sin"] = np.sin(2 * np.pi * X_copy[col] / period)
                X_copy[f"{col}_cos"] = np.cos(2 * np.pi * X_copy[col] / period)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        return list(input_features) + self.added_features_
