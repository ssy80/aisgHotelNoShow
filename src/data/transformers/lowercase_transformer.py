from sklearn.base import BaseEstimator, TransformerMixin

class LowercaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        
        if self.columns is None:
            raise ValueError(f"LowercaseTransformer - column not found: {self.columns}")
        missing_columns = [col for col in self.columns if col not in X.columns]
        if missing_columns:
            raise ValueError(f"LowercaseTransformer - column not found: {missing_columns}")
        invalid_columns = [col for col in self.columns if X[col].dtype != 'object']
        if invalid_columns:
            raise ValueError(f"LowercaseTransformer - invalid column dtype: {invalid_columns}")
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        #if self.columns is None:
            # Apply to all object/string columns
        #    string_cols = X_copy.select_dtypes(include=['object']).columns
        #else:
            # Apply only to specified columns
        #string_cols = [col for col in self.columns if col in X_copy.columns]
        #for col in string_cols:
        #    X_copy[col] = X_copy[col].str.lower()
        for col in self.columns:
            X_copy[col] = X_copy[col].str.lower()

        return X_copy

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return list(input_features)

# Usage
#lowercase_transformer = LowercaseTransformer()
# Or for specific columns:
#lowercase_transformer = LowercaseTransformer(columns=['branch', 'country', 'platform'])