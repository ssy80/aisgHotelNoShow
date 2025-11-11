from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class IntTransformer(BaseEstimator, TransformerMixin):
    """Convert columns dtype float or object to int dtype"""

    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        
        if self.columns is None:
            raise ValueError(f"IntTransformer - column not found: {self.columns}")
        missing_columns = [col for col in self.columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"IntTransformer - column not found: {missing_columns}")
        invalid_columns = [col for col in self.columns if X[col].dtype != 'object' and X[col].dtype != 'float64']
        if invalid_columns:
            raise ValueError(f"IntTransformer - invalid column dtype: {invalid_columns}")
        return self
    
    def transform(self, X):
        """Convert values of non-numeric type to int64 type"""

        X_copy = X.copy()
        
        for col in self.columns:
            X_copy[col] = pd.to_numeric(X_copy[col]).astype(int)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged (no columns added/dropped)"""
        return getattr(self, "feature_names_in_", input_features)

    '''def _validate_dtype(self, X):
        """Validate columns to convert is of dtype 'object' or 'float'"""

        invalid_columns = [col for col in self.columns if X[col].dtype != 'object' and X[col].dtype != 'float64']
        if invalid_columns:
            raise ValueError(f"IntTransformer - invalid column dtype: {invalid_columns}")'''
