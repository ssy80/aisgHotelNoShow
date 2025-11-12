import joblib
import os
import logging
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from utils.helper import setup_logging, safe_get
from utils.model_helper import get_base_model_class


class ModelTrainer:
    def __init__(self, config: dict):
        """?"""

        if config is None:
            raise ValueError(f"ModelTrainer __init__: config cannot be None")

        self.config = config

        setup_logging()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def get_model(self):
        """Get model instance based on configuration"""
        
        algorithm = safe_get(self.config, 'model', 'algorithm', required=True)
        hyperparams = safe_get(self.config, 'model', 'hyperparameters', algorithm, required=True)
            
        base_model_class = get_base_model_class(algorithm)
        self.model = base_model_class(**hyperparams)
        return self.model
    
    def train_model(self, X_train, y_train):
        """Train the model with cross-validation"""

        self.model = self.get_model()
        
        cv = safe_get(self.config, 'training', 'cv_folds', required=True)
        scoring = safe_get(self.config, 'training', 'scoring_metric', required=True)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, 
            X_train,
            y_train,
            cv = cv,
            scoring = scoring
        )
        
        self.logger.info(f"Cross-validation scores: {cv_scores}")
        self.logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")
        
        return self.model, cv_scores
    
    def train_final_model(self, X_train, y_train, best_params):
        """Train final model using best parameters"""

        # Initialize model with best params
        algorithm = safe_get(self.config, 'tunning', 'algorithm', required=True)
        base_model_class = get_base_model_class(algorithm)
        self.model = base_model_class(**best_params)
    
        # Evaluate with cross-validation
        cv = safe_get(self.config, 'training', 'cv_folds', required=True)
        scoring = safe_get(self.config, 'training', 'scoring_metric', required=True)

        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv = cv,
            scoring = scoring
        )
        
        # Fit model on full training data
        self.model.fit(X_train, y_train)
        
        return self.model, cv_scores

    def find_best_parameters(self, X_train, y_train):
        """Find optimal hyperparameters using GridSearchCV or RandomizedSearchCV"""
        self.logger.info("Starting hyperparameter optimization")
        
        # Get base model and parameter grid
        algorithm = safe_get(self.config, 'tunning', 'algorithm', required=True)
        base_model_class = get_base_model_class(algorithm)
        param_grid = safe_get(self.config, 'tunning', 'hyperparameters', algorithm, required=True)
        search_strategy = safe_get(self.config, 'tunning', 'search_strategy', required=True)        

        model = base_model_class()
        
        # Choose search strategy random or grid
        if search_strategy == 'random':
            best_params = self._randomized_search(X_train, y_train, model, param_grid)
        elif search_strategy == 'grid':
            best_params = self._grid_search(X_train, y_train, model, param_grid)
        else:
            raise ValueError(f"Unrecognized best params search strategy")
        
        self.logger.info(f"Best parameters found: {best_params}")
        return best_params

    def _randomized_search(self, X_train, y_train, base_model, param_distributions):
        """Perform randomized search for faster optimization"""
        self.logger.info("Performing RandomizedSearchCV")

        scoring_metric = safe_get(self.config, 'training', 'scoring_metric', required=True)
        cv = safe_get(self.config, 'training', 'cv_folds', required=True)
        n_iter = safe_get(self.config, 'tunning', 'n_iter', required=True)
        n_jobs = safe_get(self.config, 'tunning', 'n_jobs', required=True)
        random_state = safe_get(self.config, 'tunning', 'random_state', required=True)

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring_metric,
            n_jobs=n_jobs,
            random_state=random_state,
            return_train_score=True
        )

        random_search.fit(X_train, y_train)
        return random_search.best_params_

    def _grid_search(self, X_train, y_train, base_model, param_grid):
        """Perform exhaustive grid search"""
        self.logger.info("Performing GridSearchCV")
        
        scoring_metric = safe_get(self.config, 'training', 'scoring_metric', required=True)
        cv = safe_get(self.config, 'training', 'cv_folds', required=True)
        n_jobs = safe_get(self.config, 'tunning', 'n_jobs', required=True)
        random_state = safe_get(self.config, 'tunning', 'random_state', required=True)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring_metric,
            n_jobs=n_jobs,
            random_state=random_state,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)
        return grid_search.best_params_

    def save_model(self, preprocessor=None):
        """Save trained model and preprocessor"""
        
        save_model = safe_get(self.config, 'training', 'save_model', required=True)
        output_path = safe_get(self.config, 'training', 'model_output_path', required=True)
        if save_model:
            output_path = output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save model and preprocessor together
            pipeline_obj = {
                'model': self.model,
                'preprocessor': preprocessor
            }
            
            joblib.dump(pipeline_obj, output_path)
            self.logger.info(f"Model saved to {output_path}")
