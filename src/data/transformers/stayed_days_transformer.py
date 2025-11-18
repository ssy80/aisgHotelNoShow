from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class StayedDaysTransformer(BaseEstimator, TransformerMixin):
    """
    Calculates how many days a guest stayed using arrival and checkout dates.
    """

    def __init__(self,
                 column_arrival_month="arrival_month",
                 column_arrival_day="arrival_day",
                 column_checkout_month="checkout_month",
                 column_checkout_day="checkout_day"):
        self.column_arrival_month = column_arrival_month
        self.column_arrival_day = column_arrival_day
        self.column_checkout_month = column_checkout_month
        self.column_checkout_day = column_checkout_day
        self.added_features_ = ["stayed_days"]

    def fit(self, X, y=None):
        """
        Stores input feature names.
        """
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        """
        Calculates the number of days stayed.
        Steps:
        - Build arrival and checkout dates using year 2024
        - Compute day difference
        - If negative (checkout earlier than arrival), wrap around with 366 days
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

        # Compute days stayed
        X_copy['stayed_days'] = (X_copy['checkout_date'] - X_copy['arrival_date']).dt.days
        X_copy['stayed_days'] = X_copy['stayed_days'].apply(
            lambda x: 366 + x if x < 0 else x
        )

        # Remove temporary columns
        X_copy = X_copy.drop(['arrival_date', 'checkout_date'], axis=1)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """
        Returns original feature names plus stayed_days.
        """
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return list(input_features) + self.added_features_
