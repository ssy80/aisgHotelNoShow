from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class ModeRoomImputer(BaseEstimator, TransformerMixin):
    """Impute missing room types using the most frequent room type for a given price."""
    
    def __init__(self, column_name='room', price_column='price'):
        self.column_name = column_name      # room
        self.price_column = price_column    # price
        self.fill_values_ = None            # mapping: price -> most common room

    def fit(self, X, y=None):
        X_copy = X.copy()

        # Only consider rows where room is not missing
        non_missing = X_copy[X_copy[self.column_name].notna()]

        # Compute mode room for each price
        self.fill_values_ = (
            non_missing.groupby(self.price_column)[self.column_name]
                       .agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                       .to_dict()
        )
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Fill missing rooms using the mode room for that price
        def fill_room(row):
            if pd.notna(row[self.column_name]):
                return row[self.column_name]
            return self.fill_values_.get(row[self.price_column], row[self.column_name])

        X_copy[self.column_name] = X_copy.apply(fill_room, axis=1)
        return X_copy
