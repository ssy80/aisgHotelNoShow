from sklearn.base import BaseEstimator, TransformerMixin

class FirstTimeTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert first_time values of "yes" to 1, "no" to 2"""
    
    def __init__(self, column_name='first_time'):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        X_copy = X.copy()
        if self.column_name in X_copy.columns:
            X_copy[self.column_name] = X_copy[self.column_name].apply(
                lambda x: 1 if x == "yes" else 0
            )
        return X_copy