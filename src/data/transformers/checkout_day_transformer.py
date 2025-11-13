from sklearn.base import BaseEstimator, TransformerMixin

class CheckoutDayTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert negative checkout_day values to positive"""
    
    def __init__(self, column_name='checkout_day'):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        return self
    
    def transform(self, X):
        """Convert those days with negative numbers to positive numbers by multiplying by -1"""

        X_copy = X.copy()
        if self.column_name in X_copy.columns:
            X_copy[self.column_name] = X_copy[self.column_name].apply(
                lambda x: (x * -1) if x < 0 else x
            )
        else:
            raise ValueError(f"CheckoutDayTransformer - column not found: {self.column_name}")
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return list(getattr(self, "feature_names_in_", []))
