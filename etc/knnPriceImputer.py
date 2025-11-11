from sklearn.base import BaseEstimator, TransformerMixin
from zeroToNaNTransformer import ZeroToNaNTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd

"""
class KNNPriceImputer(BaseEstimator, TransformerMixin):
    '''Transformer to convert price values with 0 to knn price'''
    
    def __init__(self, column_name='price'):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        X_copy = X.copy()

        price_imputation_ct = ColumnTransformer([
            ('categorical', OneHotEncoder(handle_unknown='ignore'), ['room', 'branch']),
            #('numerical', 'passthrough', ['arrival_month', 'checkout_month', 'num_adults', 'num_children'])
        ])
        price_imputation_pipeline = Pipeline([
            ('select_features', price_imputation_ct),
            ('knn_imputer', KNNImputer(n_neighbors=5))
        ])

        knn_processor = Pipeline([
            ('zero_to_nan_price', ZeroToNaNTransformer(column_name="price")),
            ('price_imputation', price_imputation_pipeline)
        ])

        if self.column_name in X_copy.columns:
            #X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)

            imputed = knn_processor.fit_transform(X_copy)
            # how to return the imputed price tp X_copy["price"] ??

            imputed_price = imputed[:, -1]

            # Replace in original DataFrame
            X_copy[self.column_name] = imputed_price


        return X_copy
"""
'''class KNNPriceImputer(BaseEstimator, TransformerMixin):
    """Custom transformer to impute 0.0 prices using KNN on correlated features."""
    
    def __init__(self, column_name='price'):
        self.column_name = column_name
        self.price_imputation_pipeline = None  # Will hold fitted pipeline
    
    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)

        # Define which features KNN will use
        price_features = ["room", "branch"]#, "arrival_month", "checkout_month", "num_adults", "num_children"]
        cat_features = ["room", "branch"]
        #num_features = ["arrival_month", "checkout_month", "num_adults", "num_children"]

        price_imputation_ct = ColumnTransformer([
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
            #('numerical', 'passthrough', num_features)
        ])

        # KNN pipeline
        self.price_imputation_pipeline = Pipeline([
            ('preprocess', price_imputation_ct),
            ('knn_imputer', KNNImputer(n_neighbors=5))
        ])

        # Fit KNN on subset (include price)
        self.price_imputation_pipeline.fit(X_copy[price_features + [self.column_name]])
        #self.price_imputation_pipeline.fit(X_copy[price_features + [self.column_name]])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)

        # Recreate feature subset
        price_features = ["room", "branch"]#, "arrival_month", "checkout_month", "num_adults", "num_children"]

        # Combine features + target for imputation
        to_impute = X_copy[price_features + [self.column_name]]

        # Apply fitted KNN imputer pipeline
        imputed_array = self.price_imputation_pipeline.transform(to_impute)
        

        # Extract imputed price (last column)
        imputed_price = imputed_array[:, -1]

        # Replace in original DataFrame
        X_copy[self.column_name] = imputed_price

        return X_copy
'''

class KNNPriceImputer(BaseEstimator, TransformerMixin):
    """Impute 0.0 prices using KNN based on categorical features only."""

    def __init__(self, column_name='price', n_neighbors=5, cat_features=None):
        self.column_name = column_name
        self.n_neighbors = n_neighbors
        self.cat_features = cat_features if cat_features else ["room", "branch"]
        self.imputer = None

    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)

        # Encode categorical features as integer codes
        for col in self.cat_features:
            X_copy[col + "_code"] = X_copy[col].astype("category").cat.codes

        # KNN features: only categorical codes + price
        knn_features = [c + "_code" for c in self.cat_features] + [self.column_name]

        # Fit KNN
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer.fit(X_copy[knn_features])

        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)

        # Encode categorical features as integer codes
        for col in self.cat_features:
            X_copy[col + "_code"] = X_copy[col].astype("category").cat.codes

        # KNN features: categorical codes + price
        knn_features = [c + "_code" for c in self.cat_features] + [self.column_name]

        # Impute
        imputed_array = self.imputer.transform(X_copy[knn_features])

        # Assign imputed price
        X_copy[self.column_name] = imputed_array[:, -1]

        # Drop temporary code columns
        X_copy.drop(columns=[c + "_code" for c in self.cat_features], inplace=True)

        return X_copy