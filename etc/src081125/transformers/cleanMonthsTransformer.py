from sklearn.base import BaseEstimator, TransformerMixin

class MonthsTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert months values "january" to int 1"""
    
    def __init__(self, column_name=None):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        X_copy = X.copy()

        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
            }
        
        if self.column_name in X_copy.columns:
            X_copy[self.column_name] = X_copy[self.column_name].map(months)
        return X_copy