from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogPriceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name="price"):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if self.column_name in X_copy.columns:
            # Handle negative or zero values gracefully
            X_copy[self.column_name] = X_copy[self.column_name].clip(lower=0)
            X_copy[self.column_name] = np.log1p(X_copy[self.column_name])
        return X_copy