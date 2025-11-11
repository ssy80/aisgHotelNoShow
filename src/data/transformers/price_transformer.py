from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class PriceTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to convert price values to SGD value 
        e.g "sgd100.0" to 100.0
            "usd100.0" to 130.0 (sgd) 
    """
    
    def __init__(self, column_name='price'):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()

        if self.column_name not in X.columns:
            raise ValueError(f"PriceTransformer - column not found: {self.column_name}")
        return self

    def transform(self, X):
        """
        Transformer to convert price values to SGD value 
            e.g "sgd100.0" to 100.0
                "usd100.0" to 130.0 (sgd) 
        """
        X_copy = X.copy()

        #if self.column_name in X_copy.columns:
        X_copy[self.column_name] = X_copy[self.column_name].fillna(0)
        #X_copy[self.column_name] = X_copy[self.column_name].fillna(np.nan)
        
        usd_mask = X_copy[self.column_name].str.contains("usd", na=False)
        temp_price = X_copy[self.column_name].replace(r'[^\d.]', '', regex=True)
        temp_price = pd.to_numeric(temp_price, errors='coerce')
        temp_price[usd_mask] = temp_price[usd_mask] * 1.3     #(USD 1 = SGD 1.3)
        X_copy[self.column_name] = temp_price

        return X_copy

    '''def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return list(input_features)'''

    def get_feature_names_out(self, input_features=None):
        return list(getattr(self, "feature_names_in_", []))