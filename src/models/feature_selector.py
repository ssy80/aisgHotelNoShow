from sklearn.feature_selection import SelectFromModel
import pandas as pd
from utils.helper import safe_get

class FeatureSelector:
    """?"""

    def __init__(self, config: dict, model=None):
        """?"""

        if config is None:
            raise ValueError(f"FeatureSelector __init__: config cannot be None")

        self.config = config
        self.model = model
        threshold = safe_get(self.config, 'feature_selection', 'feature_selection_threshold', required=True)
        self.threshold = threshold
        self.selector = None
        self.selected_columns = None

        if model is None or threshold is None:
            raise ValueError(f"FeatureSelector __init__ model, threshold cannot be None")

    def select_features(self, X_train, y_train, X_test):
        """?"""

        # Init selector
        self.selector = SelectFromModel(self.model, threshold=self.threshold)

        # Fit on training data
        self.selector.fit(X_train, y_train)

        # Transform datasets
        X_train_sel = self.selector.transform(X_train)
        X_test_sel  = self.selector.transform(X_test)

        # Get selected feature names
        mask = self.selector.get_support()
        self.selected_columns = X_train.columns[mask].tolist()

        # Wrap as DataFrame to preserve column names
        X_train_sel = pd.DataFrame(X_train_sel, columns=self.selected_columns, index=X_train.index)
        X_test_sel  = pd.DataFrame(X_test_sel,  columns=self.selected_columns, index=X_test.index)

        return X_train_sel, X_test_sel, self.selector
