from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class NearestMeanRoomImputerBranch(BaseEstimator, TransformerMixin):
    """
    Impute missing room types using branch + price.
    For each missing room, choose the room in the same branch whose mean price
    is closest to the booking price. Falls back to overall nearest room if needed.
    """
    
    def __init__(self, column_name='room', price_column='price', branch_column='branch'):
        self.column_name = column_name
        self.price_column = price_column
        self.branch_column = branch_column
        self.branch_room_mean_ = None  # dict: (branch, room) -> mean price
        self.global_room_mean_ = None  # dict: room -> mean price for fallback

    def fit(self, X, y=None):
        X_copy = X.copy()
        # Consider rows where room is known and price > 0
        non_missing = X_copy[(X_copy[self.column_name].notna()) & (X_copy[self.price_column] > 0)]

        # Compute mean price per branch + room
        self.branch_room_mean_ = (
            non_missing.groupby([self.branch_column, self.column_name])[self.price_column]
                       .mean()
                       .to_dict()
        )

        # Compute overall mean per room as fallback
        self.global_room_mean_ = non_missing.groupby(self.column_name)[self.price_column].mean().to_dict()
        return self

    def transform(self, X):
        X_copy = X.copy()

        def nearest_room(row):
            if pd.notna(row[self.column_name]) or row[self.price_column] <= 0:
                return row[self.column_name]

            # Filter rooms in same branch
            branch = row[self.branch_column]
            price = row[self.price_column]
            
            # Rooms in the branch
            branch_rooms = {room: mean_price for (b, room), mean_price in self.branch_room_mean_.items() if b == branch}

            if branch_rooms:
                # Pick room with mean price closest to booking price
                nearest = min(branch_rooms.items(), key=lambda x: abs(x[1] - price))
                return nearest[0]
            else:
                # Fallback: overall nearest-to-mean room
                nearest = min(self.global_room_mean_.items(), key=lambda x: abs(x[1] - price))
                return nearest[0]

        # Apply to rows with missing room
        mask = X_copy[self.column_name].isna()
        X_copy.loc[mask, self.column_name] = X_copy.loc[mask].apply(nearest_room, axis=1)
        return X_copy
