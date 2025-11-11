from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class StayedDaysTransformer(BaseEstimator, TransformerMixin):
    """Transformer to calculate number of stays stayed"""
    
    def __init__(self,
                column_arrival_month="arrival_month",
                column_arrival_day="arrival_day",
                column_checkout_month="checkout_month",
                column_checkout_day="checkout_day"
                ):
        self.column_arrival_month = column_arrival_month
        self.column_arrival_day = column_arrival_day
        self.column_checkout_month = column_checkout_month
        self.column_checkout_day = column_checkout_day
        self.added_features_ = ["stayed_days"]
        
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()

        return self
    
    def transform(self, X):
        """
        Convert using with year(2024), month and day into a date.
            using checkout_date - arrival_date to get number of days stayed.
            if num of days stayed is negative, use 366 + (-days) to get number of days stayed.
        """
        X_copy = X.copy()
        year = 2024
        X_copy['arrival_date'] = pd.to_datetime({
            'year': year,
            'month': X_copy[self.column_arrival_month],
            'day': X_copy[self.column_arrival_day]
        }, errors='coerce')

        X_copy['checkout_date'] = pd.to_datetime({
            'year': year,
            'month': X_copy[self.column_checkout_month],
            'day': X_copy[self.column_checkout_day],
        }, errors='coerce')

        X_copy['stayed_days'] = (X_copy['checkout_date'] - X_copy['arrival_date']).dt.days
        X_copy['stayed_days'] = X_copy['stayed_days'].apply(lambda x: 366 + x if x < 0 else x)
        X_copy = X_copy.drop(['arrival_date', 'checkout_date'], axis=1) #, inplace=True)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return list(input_features) + self.added_features_

