from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class PriceImputer(BaseEstimator, TransformerMixin):
    """Impute zero prices using mean, median, or mode per category group."""

    def __init__(self, column_name='price', depend_features=None, strategy='mean'):
        """
        Initialize the imputer.

        Args:
            column_name (str): Name of the price column.
            depend_features (list): Features used for grouping.
            strategy (str): Imputation method ('mean', 'median', or 'mode').
        """
        self.column_name = column_name
        self.depend_features = depend_features
        self.fill_values_ = None
        self.strategy = strategy

    def fit(self, X, y=None):
        """Compute fill values based on the chosen strategy."""
        self.feature_names_in_ = X.columns.tolist()
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)

        if self.strategy == 'mean':
            agg_func = 'mean'
        elif self.strategy == 'median':
            agg_func = 'median'
        elif self.strategy == 'mode':
            agg_func = lambda x: x.mode()[0] if not x.mode().empty else np.nan
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")

        self.fill_values_ = (
            X_copy.groupby(self.depend_features)[self.column_name]
            .agg(agg_func)
            .to_dict()
        )
        return self

    def transform(self, X):
        """Replace zero prices with imputed values."""
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)
        X_copy[self.column_name] = X_copy.apply(
            lambda row: self._fill_price(row) if pd.isna(row[self.column_name]) else row[self.column_name],
            axis=1
        )
        return X_copy

    def _fill_price(self, row):
        """Helper to get fill value based on category combination."""
        key = tuple(row[dep] for dep in self.depend_features)
        return self.fill_values_.get(key, row[self.column_name])

    def get_feature_names_out(self, input_features=None):
        """Return feature names after transformation."""
        return list(getattr(self, "feature_names_in_", []))


        
