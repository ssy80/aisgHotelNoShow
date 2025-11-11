from sklearn.base import BaseEstimator, TransformerMixin

class HasChildrenTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert negative checkout_day values to positive"""
    
    def __init__(self, column_name='num_children'):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        X_copy = X.copy()
        if self.column_name in X_copy.columns:
            X_copy["has_children"] = X_copy["num_children"].map({0:0, 1:1, 2:1, 3:1})
        return X_copy