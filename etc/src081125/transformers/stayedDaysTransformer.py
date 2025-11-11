from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class StayedDaysTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert negative checkout_day values to positive"""
    
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
        
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
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
        X_copy.drop(['arrival_date', 'checkout_date'], axis=1, inplace=True)
        return X_copy