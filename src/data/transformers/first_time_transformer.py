from sklearn.base import BaseEstimator, TransformerMixin

class FirstTimeTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert first_time values of "yes" to 1, "no" to 2"""
    
    def __init__(self, column_name='first_time'):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()

        if self.column_name not in X.columns:
            raise ValueError(f"FirstTimeTransformer - column not found: {self.column_name}")
        return self
    
    def transform(self, X):
        """
        Convert first_time values to boolean int values of 0 or 1.
            "yes" -> 1
            "no"  -> 0 
        """

        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].apply(
            lambda x: 1 if x == "yes" else 0
        )
        self._validate_data(X_copy)
        return X_copy

    def _validate_data(self, X_copy):
        """Validate converted data is 0 or 1"""

        for i in X_copy[self.column_name]:
            if i != 0 and i != 1:
                raise ValueError(f"FirstTimeTransformer - invalid data: {self.column_name} is not 0 or 1")

    '''def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged (no columns added or dropped)"""
        return getattr(self, "feature_names_in_", input_features)'''

    def get_feature_names_out(self, input_features=None):
        return list(getattr(self, "feature_names_in_", []))