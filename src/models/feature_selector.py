from sklearn.feature_selection import SelectFromModel
import pandas as pd
from utils.helper import safe_get
from utils.model_helper import get_base_model_class

class FeatureSelector:
    """?"""

    def __init__(self, config: dict):
        """?"""

        if config is None:
            raise ValueError(f"FeatureSelector __init__: config cannot be None")

        self.config = config
        threshold = safe_get(self.config, 'feature_selection', 'feature_selection_threshold', required=True)
        self.threshold = threshold
        self.selector = None
        self.selected_columns = None

        if self.threshold is None:
            raise ValueError(f"FeatureSelector __init__ threshold cannot be None")

    def select_features(self, X_train, y_train, X_test):
        """?"""

        model = self.get_model()
        # Init selector
        self.selector = SelectFromModel(model, threshold=self.threshold)

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

    def get_model(self):
        """?"""

        algorithm = safe_get(self.config, 'model', 'algorithm', required=True)
        hyperparams = safe_get(self.config, 'model', 'hyperparameters', algorithm, required=True)
            
        base_model_class = get_base_model_class(algorithm)
        model = base_model_class(**hyperparams)
        return model
