from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class NumAdultsTransformer(BaseEstimator, TransformerMixin):
    """
    Converts entries in the num_adults column:
    - "one" -> 1
    - "two" -> 2
    Ensures all final values are valid (1 or 2).
    """

    def __init__(self, column_name='num_adults'):
        self.column_name = column_name

    def fit(self, X, y=None):
        """
        Checks that the column exists and stores feature names.
        """
        self.feature_names_in_ = X.columns.tolist()

        if self.column_name not in X.columns:
            raise ValueError(f"NumAdultsTransformer - column not found: {self.column_name}")

        # Optional: validate original values
        self._validate_initial_values(X)

        return self

    def transform(self, X):
        """
        Converts string values ("one", "two") to integers.
        Ensures the output contains only valid values.
        """
        X_copy = X.copy()

        if self.column_name in X_copy.columns:
            # Map valid strings → numbers
            X_copy[self.column_name] = X_copy[self.column_name].apply(
                lambda x: 1 if x == "one" else (2 if x == "two" else x)
            )

        # Convert numeric-like strings to numbers, others → NaN
        X_copy[self.column_name] = pd.to_numeric(X_copy[self.column_name], errors='coerce')

        # Validate final clean dataset
        self._validate_final_values(X_copy)

        return X_copy

    def _validate_initial_values(self, X):
        """
        Ensures the original column contains only:
        - '1', '2', "one", "two"
        Raises an error if other values appear.
        """
        allowed = {"one", "two", '1', '2'}
        invalid = X[~X[self.column_name].isin(allowed)][self.column_name].unique()

        if len(invalid) > 0:
            raise ValueError(
                f"NumAdultsTransformer - invalid initial values: {invalid}. "
                f"Allowed: {allowed}"
            )

    def _validate_final_values(self, X_copy):
        """
        Ensures the cleaned data contains only 1 or 2.
        """
        valid_values = [1, 2]
        invalid_mask = ~X_copy[self.column_name].isin(valid_values)

        if invalid_mask.any():
            invalid_values = X_copy.loc[invalid_mask, self.column_name].unique()
            raise ValueError(f"NumAdultsTransformer - invalid values after transform: {invalid_values}")

    def get_feature_names_out(self, input_features=None):
        """
        Returns the original feature names.
        """
        return list(getattr(self, "feature_names_in_", []))
