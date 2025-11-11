from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class NumAdultsTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert num_adults values of "one" to 1, "two" to 2"""
    
    def __init__(self, column_name='num_adults'):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()

        if self.column_name not in X.columns:
            raise ValueError(f"NumAdultsTransformer - column not found: {self.column_name}")
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.column_name in X_copy.columns:
            X_copy[self.column_name] = X_copy[self.column_name].apply(
                lambda x: 1 if x == "one" else (2 if x == "two" else x)
            )
        X_copy[self.column_name] = pd.to_numeric(X_copy[self.column_name], errors='coerce')
        self._validate_data(X_copy)
        return X_copy

    def _validate_data(self, X_copy):
        """Validate converted values is 1 or 2"""

        valid_values = [1, 2]
        invalid_mask = ~X_copy[self.column_name].isin(valid_values)
        if invalid_mask.any():
            invalid_values = X_copy[invalid_mask][self.column_name].unique()
            raise ValueError(f"NumAdultsTransformer - invalid values: {invalid_values}")

    '''def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return list(input_features)'''

    def get_feature_names_out(self, input_features=None):
        return list(getattr(self, "feature_names_in_", []))

    """
    Todo: create another validate method to ensure init data only contains - 1, 2, "one", "two"
    """
