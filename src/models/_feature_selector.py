# feature_selection.py
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel
import pandas as pd

class FeatureSelector:
    def __init__(self, config):
        self.config = config
        min_features = config['feature']['feature_selection_k']
        self.selector = SelectKBest(score_func=f_classif, k=min_features)
    
    '''def select_features(self, X_train, y_train, X_test):

        X_train_sel = self.selector.fit_transform(X_train, y_train)
        X_test_sel = self.selector.transform(X_test)
        
        return X_train_sel, X_test_sel, self.selector'''


    def select_features(self, X_train, y_train, X_test):

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_processed, y_train)

        selector = SelectFromModel(model, threshold='median')  # selects top 50% important features
        X_train_sel = selector.transform(X_train_processed)
        X_test_sel  = selector.transform(X_test_processed)


        # Fit selector
        self.selector.fit(X_train, y_train)
        
        # Get boolean mask of selected features
        mask = self.selector.get_support()
        
        # Selected columns
        selected_columns = X_train.columns[mask].tolist()
        
        # Transform and wrap as DataFrame
        X_train_sel = pd.DataFrame(self.selector.transform(X_train), columns=selected_columns, index=X_train.index)
        X_test_sel  = pd.DataFrame(self.selector.transform(X_test),  columns=selected_columns, index=X_test.index)
        
        return X_train_sel, X_test_sel, self.selector
