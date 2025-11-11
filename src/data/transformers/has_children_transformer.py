from sklearn.base import BaseEstimator, TransformerMixin

class HasChildrenTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert negative checkout_day values to positive"""
    
    def __init__(self, column_name=None):
        self.column_name = column_name
        self.added_features_ = ["has_children"]
        #print(self.column_name)
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()

        if self.column_name is None:
            raise ValueError(f"HasChildrenTransformer - column not found: {self.column_name}")
        if not self.column_name in X.columns:
            raise ValueError(f"HasChildrenTransformer - column not found: {self.column_name}")
        return self
    
    def transform(self, X):
        X_copy = X.copy()

        #X_copy["has_children"] = X_copy[self.column_name].map({0:0, 1:1, 2:1, 3:1})
        X_copy["has_children"] = (X_copy[self.column_name] > 0).astype(int)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return list(input_features) + self.added_features_

