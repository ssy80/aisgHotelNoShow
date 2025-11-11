from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class RoomImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing room types using branch + price.
    For each missing room, choose the room in the same branch whose mean price
    is closest to the booking price. Falls back to overall nearest room if needed.
    """
    
    def __init__(self, column_name='room', price_column='price', branch_column='branch', strategy='mean'):
        self.column_name = column_name
        self.price_column = price_column
        self.branch_column = branch_column
        self.branch_room_mean_ = None  # dict: (branch, room) -> mean price
        self.global_room_mean_ = None  # dict: room -> mean price for fallback
        self.strategy = strategy

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        X_copy = X.copy()

        # Consider rows where room is known and price > 0
        non_missing = X_copy[(X_copy[self.column_name].notna()) & (X_copy[self.price_column] > 0)]

        # Choose aggregation method based on strategy
        if self.strategy == 'mean':
            agg_func = 'mean'
        elif self.strategy == 'median':
            agg_func = 'median'
        elif self.strategy == 'mode':
            agg_func = lambda x: x.mode()[0] if not x.mode().empty else np.nan
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")

        # Compute mean price per branch + room
        self.branch_room_mean_ = (
            non_missing.groupby([self.branch_column, self.column_name])[self.price_column]
                #.mean()
                .agg(agg_func)
                .to_dict()
        )

        # Compute overall mean per room as fallback
        self.global_room_mean_ = (
            non_missing.groupby(self.column_name)[self.price_column] 
                #.mean()
                .agg(agg_func)
                .to_dict()
        )
        
        #print(self.branch_room_mean_)
        #print(self.global_room_mean_)
        
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Apply to rows with missing room
        mask = X_copy[self.column_name].isna()
        X_copy.loc[mask, self.column_name] = X_copy.loc[mask].apply(self._nearest_room, axis=1)
        return X_copy


    def _nearest_room(self, row):

        if pd.notna(row[self.column_name]) or row[self.price_column] <= 0:
            return row[self.column_name]

        # Filter rooms in same branch
        branch = row[self.branch_column]  # get branch
        price = row[self.price_column]    # get price for room
        
        # Rooms in the branch
        branch_rooms = {room: mean_price for (b, room), mean_price in self.branch_room_mean_.items() if b == branch}
        #print(f"branch:{branch}, {branch_rooms}")

        if branch_rooms:
            # Pick room with mean price closest to booking price
            nearest = min(branch_rooms.items(), key=lambda x: abs(x[1] - price))
            #print(nearest)
            return nearest[0]
        #else:
            # Fallback: overall nearest-to-mean room
        #    nearest = min(self.global_room_mean_.items(), key=lambda x: abs(x[1] - price))
            #print(nearest)
        #    return nearest[0]

    '''def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return list(input_features)'''

    def get_feature_names_out(self, input_features=None):
        return list(getattr(self, "feature_names_in_", []))