from sklearn.base import BaseEstimator, TransformerMixin

class LowercaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if self.columns is None:
            # Apply to all object/string columns
            string_cols = X_copy.select_dtypes(include=['object']).columns
        else:
            # Apply only to specified columns
            string_cols = [col for col in self.columns if col in X_copy.columns]
        
        for col in string_cols:
            X_copy[col] = X_copy[col].str.lower()
        
        return X_copy

# Usage
#lowercase_transformer = LowercaseTransformer()
# Or for specific columns:
#lowercase_transformer = LowercaseTransformer(columns=['branch', 'country', 'platform'])