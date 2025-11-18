from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_base_model_class(algorithm: str) -> object:
    """Return the model class for a given algorithm name."""
    
    model_map = {
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegression,
        'xgboost': XGBClassifier,
        'svm': SVC
    }

    if algorithm not in model_map:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return model_map[algorithm]