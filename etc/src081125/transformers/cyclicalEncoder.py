from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns_period_map=None):
        self.columns_period_map = columns_period_map
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, period in self.columns_period_map.items():
            if col in X_copy.columns:
                X_copy[f"{col}_sin"] = np.sin(2 * np.pi * X_copy[col] / period)
                X_copy[f"{col}_cos"] = np.cos(2 * np.pi * X_copy[col] / period)
                X_copy = X_copy.drop(columns=[col], axis=1)#, inplace=True)  # optional: drop original
        return X_copy