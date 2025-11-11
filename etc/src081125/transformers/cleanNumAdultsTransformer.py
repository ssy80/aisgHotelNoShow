from sklearn.base import BaseEstimator, TransformerMixin

class NumAdultsTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert num_adults values of "one" to 1, "two" to 2"""
    
    def __init__(self, column_name='num_adults'):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        X_copy = X.copy()
        if self.column_name in X_copy.columns:
            X_copy[self.column_name] = X_copy[self.column_name].apply(
                lambda x: 1 if x == "one" else (2 if x == "two" else x)
            )
        return X_copy