from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ZeroToNaNTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert price values 0 to NaN"""
    
    def __init__(self, column_name='price'):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        X_copy = X.copy()
        if self.column_name in X_copy.columns:
            X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)
        return X_copy