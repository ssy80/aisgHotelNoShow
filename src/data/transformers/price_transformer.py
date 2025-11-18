from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class PriceTransformer(BaseEstimator, TransformerMixin):
    """
    Converts price strings into SGD values.
    Examples:
    - "sgd100" -> 100
    - "usd100" -> 130
    """

    def __init__(self, column_name='price'):
        self.column_name = column_name

    def fit(self, X, y=None):
        """
        Checks the price column exists and stores feature names.
        """
        self.feature_names_in_ = X.columns.tolist()

        if self.column_name not in X.columns:
            raise ValueError(
                f"PriceTransformer - column not found: {self.column_name}"
            )
        return self

    def transform(self, X):
        """
        Converts prices to numeric SGD values.
        Steps:
        - Fill missing values with "0"
        - Detect USD entries
        - Remove non-numeric characters
        - Convert numbers to float
        - Multiply USD amounts by 1.3
        """
        X_copy = X.copy()

        # Ensure string format
        X_copy[self.column_name] = X_copy[self.column_name].fillna("0").astype(str)

        # Identify USD values
        usd_mask = X_copy[self.column_name].str.contains("usd", case=False, na=False)

        # Extract numeric part
        numeric = X_copy[self.column_name].str.replace(r"[^\d.]", "", regex=True)
        numeric = pd.to_numeric(numeric, errors="coerce").fillna(0)

        # Convert USD → SGD
        numeric.loc[usd_mask] *= 1.3

        X_copy[self.column_name] = numeric
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """
        Returns the original feature names.
        """
        return list(getattr(self, "feature_names_in_", []))
