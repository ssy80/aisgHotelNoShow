from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class ModePriceImputer(BaseEstimator, TransformerMixin):
    """Impute 0 prices using the most frequent price per category combination."""
    
    def __init__(self, column_name='price', cat_features=None):
        self.column_name = column_name
        self.cat_features = cat_features if cat_features else ["room", "branch"]
        self.fill_values_ = None  # To store mode per category combination

    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)

        # Compute mode price per combination of categorical features
        self.fill_values_ = (
            X_copy.groupby(self.cat_features)[self.column_name]
                  .agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                  .to_dict()
        )
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)

        # Fill missing prices using mode per category combination
        def fill_price(row):
            key = tuple(row[cat] for cat in self.cat_features)
            return self.fill_values_.get(key, row[self.column_name])

        X_copy[self.column_name] = X_copy.apply(
            lambda row: fill_price(row) if pd.isna(row[self.column_name]) else row[self.column_name],
            axis=1
        )

        return X_copy
