from sklearn.base import BaseEstimator, TransformerMixin

class MonthTransformer(BaseEstimator, TransformerMixin):
    """Convert month names in specified columns to integers (1-12)."""

    def __init__(self, columns=None):
        """
        Initialize transformer.

        Args:
            columns (list): Columns containing month names to convert.
        """
        self.columns = columns

    def fit(self, X, y=None):
        """Validate columns exist and contain valid month names."""
        self.feature_names_in_ = X.columns.tolist()

        if self.columns is None:
            raise ValueError("MonthTransformer - columns not provided")
        self._validate_months(X)
        return self

    def transform(self, X):
        """Map month names to integers (1-12)."""
        X_copy = X.copy()

        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }

        for mth in self.columns:
            X_copy[mth] = X_copy[mth].map(months)
        self._validate_data(X_copy)
        return X_copy

    def _validate_data(self, X_copy):
        """Ensure all transformed month values are valid integers 1-12."""
        valid_int_months = list(range(1, 13))
        for mth in self.columns:
            invalid_mask = ~X_copy[mth].isin(valid_int_months)
            if invalid_mask.any():
                invalid_values = X_copy[mth][invalid_mask].unique()
                raise ValueError(f"MonthTransformer - invalid months(int) in '{mth}': {invalid_values}")

    def _validate_months(self, X):
        """Ensure all values in columns are valid month names."""
        months = [
            "january","february","march","april","may","june",
            "july","august","september","october","november","december"
        ]
        for mth in self.columns:
            invalid_mask = ~X[mth].isin(months)
            if invalid_mask.any():
                invalid_values = X[mth][invalid_mask].unique()
                raise ValueError(f"MonthTransformer - invalid months in '{mth}': {invalid_values}")

    def get_feature_names_out(self, input_features=None):
        """Return feature names unchanged."""
        return list(getattr(self, "feature_names_in_", []))
