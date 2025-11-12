from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from utils.model_helper import get_base_model_class
from utils.helper import safe_get


class TunningModel:

    def __init__(self, config):

        if config is None:
            raise ValueError(f"TunningModel __init__: config cannot be None")
        self.config = config

    def get_model(self):
        """?"""

        algorithm = safe_get(self.config, 'tunning', 'algorithm', required=True)
        model_feature_selection_params = safe_get(self.config, 'feature_selection', 'hyperparameters', algorithm, required=True)

        model_base_class = get_base_model_class(algorithm)
        model = model_base_class(**model_feature_selection_params)

        return model
