from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class PriceImputer(BaseEstimator, TransformerMixin):
    """Impute 0 prices using the mean price per category combination."""
    
    def __init__(self, column_name='price', depend_features=None, strategy='mean'):
        self.column_name = column_name
        self.depend_features = depend_features
        self.fill_values_ = None  # To store mean per category combination
        self.strategy = strategy

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)

        # Choose aggregation method based on strategy
        if self.strategy == 'mean':
            agg_func = 'mean'
        elif self.strategy == 'median':
            agg_func = 'median'
        elif self.strategy == 'mode':
            agg_func = lambda x: x.mode()[0] if not x.mode().empty else np.nan
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")
    
        # Compute mean price per combination of categorical features
        self.fill_values_ = (
            X_copy.groupby(self.depend_features)[self.column_name]
                    #.mean()  # Use mean instead of mode
                    .agg(agg_func)
                    .to_dict()
        )
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.column_name] = X_copy[self.column_name].replace(0, np.nan)

        X_copy[self.column_name] = X_copy.apply(
            lambda row: self._fill_price(row) if pd.isna(row[self.column_name]) else row[self.column_name],
            axis=1
        )
        return X_copy

    # Fill missing prices using mean per category combination
    def _fill_price(self, row):

        key = tuple(row[dep] for dep in self.depend_features)
        return self.fill_values_.get(key, row[self.column_name])


    '''def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return list(input_features)'''

    def get_feature_names_out(self, input_features=None):
        return list(getattr(self, "feature_names_in_", []))

        
