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
        """Initialize ModelTrainer with configuration."""

        if config is None:
            raise ValueError(f"ModelTrainer __init__: config cannot be None")

        self.config = config

        setup_logging()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
    
    def train_model(self, X_train, y_train, best_params=None):
        """Train the model with cross-validation"""

        cross_validate = safe_get(self.config, 'training', 'cross_validate', required=True)

        if best_params:
            # Initialize model with best params
            algorithm = safe_get(self.config, 'model', 'algorithm', required=True)
            base_model_class = get_base_model_class(algorithm)
            self.model = base_model_class(**best_params)
        else:
            # Get model from config
            algorithm = safe_get(self.config, 'model', 'algorithm', required=True)
            hyperparams = safe_get(self.config, 'model', 'hyperparameters', algorithm, required=True)  
            base_model_class = get_base_model_class(algorithm)
            self.model = base_model_class(**hyperparams)

        self.logger.info(f"Model for training: {self.model}")
        self.logger.info(f"Cross-validation: {cross_validate}")

        cv_scores = None
        if cross_validate:
            cv = safe_get(self.config, 'training', 'cv_folds', required=True)
            scoring = safe_get(self.config, 'training', 'scoring_metric', required=True)
            self.logger.info(f"Perform cross-validation cv: {cv}, scoring: {scoring}")
            # Perform cross-validation
            cv_scores = cross_val_score(
                self.model, 
                X_train,
                y_train,
                cv = cv,
                scoring = scoring
            )
            self.logger.info(f"Cross-validation scores ({scoring}): {cv_scores}")
            self.logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train model
        self.logger.info(f"Perform training")
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")
        
        return self.model, cv_scores

    def find_best_parameters(self, X_train, y_train):
        """Find optimal hyperparameters using GridSearchCV or RandomizedSearchCV"""
        self.logger.info("Starting hyperparameter optimization")
        
        # Get base model and parameter grid
        algorithm = safe_get(self.config, 'model', 'algorithm', required=True)
        base_model_class = get_base_model_class(algorithm)
        param_grid = safe_get(self.config, 'tuning', 'hyperparameters', algorithm, required=True)
        search_strategy = safe_get(self.config, 'tuning', 'search_strategy', required=True)
        self.logger.info(f"Model ({base_model_class}), Strategy ({search_strategy}), Param Grid: {param_grid}")
        
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
        n_iter = safe_get(self.config, 'tuning', 'n_iter', required=True)
        n_jobs = safe_get(self.config, 'tuning', 'n_jobs', required=True)
        random_state = safe_get(self.config, 'tuning', 'random_state', required=True)

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
        n_jobs = safe_get(self.config, 'tuning', 'n_jobs', required=True)
        random_state = safe_get(self.config, 'tuning', 'random_state', required=True)

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

    #trainer.save_model(preprocessor=fitted_preprocessor, features_to_drop=features_to_drop, selector=fitted_selector, model=model)
    def save_model(self, preprocessor=None, features_to_drop=None, selector=None, model=None):
        """Save trained model and preprocessor"""
        
        save_model = safe_get(self.config, 'training', 'save_model', required=True)
        output_path = safe_get(self.config, 'training', 'model_output_path', required=True)
        if save_model:
            output_path = output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save model and preprocessor together
            pipeline_obj = {
                'model': model,
                'preprocessor': preprocessor,
                'features_to_drop': features_to_drop,
                'selector': selector
            }
            print(pipeline_obj)
            joblib.dump(pipeline_obj, output_path)
            self.logger.info(f"Model saved to {output_path}")
