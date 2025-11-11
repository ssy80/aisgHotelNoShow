import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from utils import setup_logging

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
            
        model_class = model_map[algorithm]
        self.model = model_class(**hyperparams)
        
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