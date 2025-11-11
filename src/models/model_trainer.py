import joblib
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from utils.helpers import setup_logging

class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logging()
        self.model = None
        
    def get_model(self):
        """Get model instance based on configuration"""
        algorithm = self.config['model']['algorithm']
        hyperparams = self.config['model']['hyperparameters'].get(algorithm, {})
        
        model_map = {
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            'xgboost': XGBClassifier,
            'svm': SVC
        }
        
        if algorithm not in model_map:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        base_model_class = model_map[algorithm]
        self.model = base_model_class(**hyperparams) # RandomForestClassifier(n_estimator=100, max_depth=5,....)
        self.logger.info(f"Initialized {algorithm} with hyperparameters: {hyperparams}")

        return self.model
    
    def train_model(self, X_train, y_train):
        """Train the model with cross-validation"""
        if self.model is None:
            self.get_model()
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=self.config['training']['cv_folds'],
            scoring=self.config['training']['scoring_metric']
        )
        
        self.logger.info(f"Cross-validation scores: {cv_scores}")
        self.logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")
        
        return self.model, cv_scores
    
    def train_final_model(self, X_train, y_train, best_params):
        """Train final model using best parameters"""
        # 1️⃣ Initialize model with best params
        base_model_class = self.get_tunning_model()
        model = base_model_class(**best_params)
    
        # 3️⃣ Evaluate with cross-validation
        ##cv_scores = cross_val_score(
        #    model, X_train, y_train, cv=self.cv, scoring=self.scoring, n_jobs=-1
        #)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self.config['training']['cv_folds'],
            scoring=self.config['training']['scoring_metric'],
            n_jobs=-1
        )
        
        # 4️⃣ Save trained model for later use
        #self.best_estimator_ = model
        # 2️⃣ Fit model on full training data
        model.fit(X_train, y_train)
        
        return model, cv_scores

    def find_best_parameters(self, X_train, y_train):
        """Find optimal hyperparameters using GridSearchCV or RandomizedSearchCV"""
        self.logger.info("Starting hyperparameter optimization")
        
        # Get base model and parameter grid
        base_model_class = self.get_tunning_model()
        param_grid = self._get_parameter_grid()
        print(base_model_class)
        print(param_grid)

        random_state = self.config['tunning']['random_state']
        model = base_model_class(random_state=random_state)
        
        # Choose search strategy
        #search_strategy = self.config.get('search_strategy', 'grid')
        search_strategy = self.config['tunning']['search_strategy'] # random, grid #.get('search_strategy', 'grid')
        
        if search_strategy == 'random':
            best_params = self._randomized_search(X_train, y_train, model, param_grid)
        else:
            best_params = self._grid_search(X_train, y_train, model, param_grid)
        
        self.logger.info(f"Best parameters found: {best_params}")
        return best_params

    def get_tunning_model(self):
        """Get base model instance based on config"""

        model_type = self.config['tunning']['algorithm']
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
    
    def _get_parameter_grid(self):
        """Get parameter grid based on model type and config"""

        model_type = self.config['tunning']['algorithm']
        #config = self.config.get('tunning')
        #hyperparameters_config = config.get('hyperparameters', {})
        #model_hyperparams = hyperparameters_config.get(model_type)
        config = self.config['tunning']
        hyperparameters_config = config['hyperparameters']#.get('hyperparameters', {})
        model_hyperparams = hyperparameters_config[model_type]#.get(model_type)
        return model_hyperparams


    def _randomized_search(self, X_train, y_train, base_model, param_distributions):
        """Perform randomized search for faster optimization"""
        self.logger.info("Performing RandomizedSearchCV")

        scoring_metric = self.config['training']['scoring_metric']
        cv = self.config['training']['cv_folds']
        n_iter = self.config['tunning']['n_iter']
        n_jobs = self.config['tunning']['n_jobs']
        random_state = self.config['tunning']['random_state']

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter, #self.config.get('n_iter', 20),            # With n_iter=20: Only try 20 random combinations (much faster!) 
            cv=cv,                                                   #self.config.get('cv_folds', 5),
            scoring=scoring_metric,
            n_jobs=n_jobs,                                           #self.config.get('n_jobs', -1), # Use ALL available CPU cores
            verbose=1,                                               #self.config.get('verbose', 1),
            random_state=42,
            return_train_score=True
        )
        
        random_search.fit(X_train, y_train)
        
        # Log detailed results
        #self._log_search_results(random_search)
        
        return random_search.best_params_

    def _grid_search(self, X_train, y_train, base_model, param_grid):
        """Perform exhaustive grid search"""
        self.logger.info("Performing GridSearchCV")
        
        scoring_metric = self.config['training']['scoring_metric']
        cv = self.config['training']['cv_folds']
        #n_iter = self.config['tunning']['n_iter']
        n_jobs = self.config['tunning']['n_jobs']
        random_state = self.config['tunning']['random_state']

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,                                         #self.config.get('cv_folds', 5),
            scoring=scoring_metric,                         #self.config.get('scoring_metric', 'neg_mean_squared_error'),
            n_jobs=n_jobs,                                   #self.config.get('n_jobs', -1),
            verbose=1,                                     #self.config.get('verbose', 1),
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        # Log detailed results
        #self._log_search_results(grid_search)
        
        return grid_search.best_params_

    def save_model(self, preprocessor=None):
        """Save trained model and preprocessor"""
        
        if self.config['training']['save_model']:
            output_path = self.config['training']['model_output_path']
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save model and preprocessor together
            pipeline_obj = {
                'model': self.model,
                'preprocessor': preprocessor
            }
            
            joblib.dump(pipeline_obj, output_path)
            self.logger.info(f"Model saved to {output_path}")

