from sklearn.base import BaseEstimator, TransformerMixin

class MonthTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert months values to int (1-12) like "january" to int 1"""
    
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()

        if self.columns is None:
            raise ValueError(f"MonthTransformer - column not found: {self.columns}")
        self._validate_months(X)
        return self
    
    def transform(self, X):
        """Convert month to int value (1-12)"""

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
        
        for mth in self.columns:
            X_copy[mth] = X_copy[mth].map(months)
        self._validate_data(X_copy)
        return X_copy

    def _validate_data(self, X_copy):
        """Validate all months in columns is converted correctly"""

        valid_int_months = [1,2,3,4,5,6,7,8,9,10,11,12]
        for mth in self.columns:
            invalid_mask = ~X_copy[mth].isin(valid_int_months)
            if invalid_mask.any():
                invalid_values = X_copy[mth][invalid_mask].unique()
                raise ValueError(f"MonthTransformer - invalid months(int) in '{mth}': {invalid_values}")

    def _validate_months(self, X):
        """
        Validate all months in columns is valid
            e.g "jaNuary" is not valid 
        """

        months = [
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
        ]

        for mth in self.columns:
            invalid_mask = ~X[mth].isin(months)
            if invalid_mask.any():
                invalid_values = X[mth][invalid_mask].unique()
                raise ValueError(f"MonthTransformer - invalid months in '{mth}': {invalid_values}")

    '''def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return list(input_features)'''

    def get_feature_names_out(self, input_features=None):
        return list(getattr(self, "feature_names_in_", []))