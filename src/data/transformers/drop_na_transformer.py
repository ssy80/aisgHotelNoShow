from sklearn.base import BaseEstimator, TransformerMixin

class DropNaTransformer(BaseEstimator, TransformerMixin):
    """"Drop rows that have n or more non-null values"""

    def __init__(self, n=3):
        self.n = n
        
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        return self
    
    def transform(self, X):
        X_copy = X.copy()

        #X_copy.dropna(thresh=len(X.columns) - self.n + 1)
        # Drop rows that have too many NaNs (less than threshold non-nulls)
        thresh = len(X_copy.columns) - self.n + 1
        X_copy = X_copy.dropna(thresh=thresh)
        return X_copy

    '''def get_feature_names_out(self, input_features=None):
        """Return unchanged feature names (rows were dropped, not columns)."""
        return getattr(self, "feature_names_in_", input_features)'''

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [])
        return list(input_features)
