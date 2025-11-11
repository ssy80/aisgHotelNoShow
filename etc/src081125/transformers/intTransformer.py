from sklearn.base import BaseEstimator, TransformerMixin

class IntTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if self.columns is None:
            # Apply to all object/string columns
            convert_cols = X_copy.select_dtypes(include=['object', 'float64']).columns
        else:
            # Apply only to specified columns
            convert_cols = [col for col in self.columns if col in X_copy.columns]
        
        for col in convert_cols:
            X_copy[col] = X_copy[col].astype(int)
        
        return X_copy