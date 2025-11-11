from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class PriceTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert price values of "sgd100.0" to 100.0, "usd100.0" to 130.0 (sgd). """
    
    def __init__(self, column_name='first_time'):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        X_copy = X.copy()
        if self.column_name in X_copy.columns:

            X_copy[self.column_name] = X_copy[self.column_name].fillna(0)
            usd_mask = X_copy[self.column_name].str.contains("usd", na=False)
            temp_price = X_copy[self.column_name].replace(r'[^\d.]', '', regex=True)#.astype(float)
            temp_price = pd.to_numeric(temp_price, errors='coerce')
            temp_price[usd_mask] = temp_price[usd_mask] * 1.3     #(USD 1 = SGD 1.3)
            X_copy[self.column_name] = temp_price

        return X_copy