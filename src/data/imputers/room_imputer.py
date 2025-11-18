from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class RoomImputer(BaseEstimator, TransformerMixin):
    """Impute missing room types using branch and price similarity."""

    def __init__(self, column_name='room', price_column='price', branch_column='branch', strategy='mean'):
        """
        Initialize the imputer.

        Args:
            column_name (str): Name of the room column.
            price_column (str): Name of the price column.
            branch_column (str): Name of the branch column.
            strategy (str): Imputation method ('mean', 'median', or 'mode').
        """
        self.column_name = column_name
        self.price_column = price_column
        self.branch_column = branch_column
        self.branch_room_mean_ = None
        self.global_room_mean_ = None
        self.strategy = strategy

    def fit(self, X, y=None):
        """Compute mean price per branch and room."""
        self.feature_names_in_ = X.columns.tolist()
        X_copy = X.copy()
        non_missing = X_copy[(X_copy[self.column_name].notna()) & (X_copy[self.price_column] > 0)]

        if self.strategy == 'mean':
            agg_func = 'mean'
        elif self.strategy == 'median':
            agg_func = 'median'
        elif self.strategy == 'mode':
            agg_func = lambda x: x.mode()[0] if not x.mode().empty else np.nan
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")

        self.branch_room_mean_ = (
            non_missing.groupby([self.branch_column, self.column_name])[self.price_column]
            .agg(agg_func)
            .to_dict()
        )

        self.global_room_mean_ = (
            non_missing.groupby(self.column_name)[self.price_column]
            .agg(agg_func)
            .to_dict()
        )

        return self

    def transform(self, X):
        """Fill missing rooms with the nearest price match in the same branch."""
        X_copy = X.copy()
        mask = X_copy[self.column_name].isna()
        X_copy.loc[mask, self.column_name] = X_copy.loc[mask].apply(self._nearest_room, axis=1)
        return X_copy

    def _nearest_room(self, row):
        """Find the room with price closest to the booking price."""
        if pd.notna(row[self.column_name]) or row[self.price_column] <= 0:
            return row[self.column_name]

        branch = row[self.branch_column]
        price = row[self.price_column]

        branch_rooms = {room: mean_price for (b, room), mean_price in self.branch_room_mean_.items() if b == branch}

        if branch_rooms:
            nearest = min(branch_rooms.items(), key=lambda x: abs(x[1] - price))
            return nearest[0]
        # Optional fallback (commented out):
        # nearest = min(self.global_room_mean_.items(), key=lambda x: abs(x[1] - price))
        # return nearest[0]

    def get_feature_names_out(self, input_features=None):
        """Return feature names after transformation."""
        return list(getattr(self, "feature_names_in_", []))
