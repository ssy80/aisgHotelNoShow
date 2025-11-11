from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()

        if self.columns is None:
            raise ValueError(f"Log1pTransformer - column not found: {self.columns}")
        missing_columns = [col for col in self.columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"Log1pTransformer - column not found: {missing_columns}")
        return self
    
    def transform(self, X):
        """Ensure numeric values are non-negative by clipping at 0, then do the log1p transform"""

        X_copy = X.copy()
        #if self.column_name in X_copy.columns:
        for col in self.columns:
            # Handle negative or zero values gracefully
            X_copy[col] = X_copy[col].clip(lower=0)
            X_copy[col] = np.log1p(X_copy[col])
        return X_copy

    '''def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged (no columns added or removed)"""
        return getattr(self, "feature_names_in_", input_features)'''

    def get_feature_names_out(self, input_features=None):
        return list(getattr(self, "feature_names_in_", []))