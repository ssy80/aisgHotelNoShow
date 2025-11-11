from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier



class TunningModel:

    def __init__(self, config):
        self.config = config

    def get_model(self):

        model_type = self.config['tunning']['algorithm']
        model_base_class = self._get_tunning_model(model_type)
        model_feature_selection_params = self.config['tunning']['feature_selection'][model_type]
        model = model_base_class(**model_feature_selection_params)

        return model


    def _get_tunning_model(self, model_type):
        """Get base model instance based on config"""

        #model_type = self.config['tunning']['algorithm']
        #random_state = self.config['tunning']['random_state']

        if model_type == 'random_forest':
            #return RandomForestClassifier(random_state=random_state)
            return RandomForestClassifier
        elif model_type == 'xgboost':
            #return XGBClassifier(random_state=random_state)
            return XGBClassifier
        elif model_type == 'logistic_regression':
            return LogisticRegression
        elif model_type == 'svm':
            #return SVC(random_state=random_state)
            return SVC
        else:
            raise ValueError(f"Unsupported model type: {model_type}")