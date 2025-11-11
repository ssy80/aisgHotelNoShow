from sklearn.base import BaseEstimator, TransformerMixin

class CheckoutDayTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert negative checkout_day values to positive"""
    
    def __init__(self, column_name='checkout_day'):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        X_copy = X.copy()
        if self.column_name in X_copy.columns:
            X_copy[self.column_name] = X_copy[self.column_name].apply(
                lambda x: (x * -1) if x < 0 else x
            )
        return X_copy