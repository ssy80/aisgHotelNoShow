from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class FeatureSelector:
    """
    Model-based feature selector using feature importances from RandomForestClassifier.
    Automatically selects top features based on a threshold (median importance by default).
    """

    def __init__(self, config, model=None):
        self.config = config
        self.threshold = config.get('feature', {}).get('feature_selection_threshold', 'median') #mean
        '''self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=config.get('seed', 42),
            n_jobs=-1
        )'''
        self.model = model
        self.selector = None
        self.selected_columns_ = None

    def select_features(self, X_train, y_train, X_test):

        print(self.model)


        # 2. Create SelectFromModel using feature importances
        #self.selector = SelectFromModel(self.model, threshold=self.threshold, prefit=True)
        self.selector = SelectFromModel(self.model, threshold=self.threshold)

        # 1. Fit on training data
        #self.model.fit(X_train, y_train)
        # 2. Fit on training data
        self.selector.fit(X_train, y_train)


        # 3. Transform datasets
        X_train_sel = self.selector.transform(X_train)
        X_test_sel  = self.selector.transform(X_test)

        # 4. Get selected feature names
        mask = self.selector.get_support()
        self.selected_columns_ = X_train.columns[mask].tolist()

        # 5. Wrap as DataFrame to preserve column names
        X_train_sel = pd.DataFrame(X_train_sel, columns=self.selected_columns_, index=X_train.index)
        X_test_sel  = pd.DataFrame(X_test_sel,  columns=self.selected_columns_, index=X_test.index)

        print(f"Selected features ({len(self.selected_columns_)}): {self.selected_columns_}")

        return X_train_sel, X_test_sel, self.selector
