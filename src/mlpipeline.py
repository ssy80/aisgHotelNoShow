import sys
import os
import logging
from utils.helper import load_config, setup_logging, safe_get
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from models.model_trainer import ModelTrainer
from models.model_evaluator import ModelEvaluator
from models.feature_selector import FeatureSelector


class MlPipeline:
    """?"""

    def __init__(self, config_path: str):

        self.config = load_config(config_path)
        if self.config is None:
            raise ValueError(f"MlPipeline __init__: config cannot be None")

        # Init logging
        setup_logging()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        
    def run(self):
        """Execute the complete ML pipeline"""
        self.logger.info("Starting ML Pipeline")
        
        try:
            # 1. Data Loading
            self.logger.info("Step 1: Data Loading")
            data_loader = DataLoader(self.config)
            data_df = data_loader.load_data()
        
            if not data_loader.validate_data(data_df):
                raise ValueError("Data validation failed")
    
            # 2. Data Preprocessing, Split into training and test set
            self.logger.info("Step 2: Data Preprocessing, Split into training and test set")
            data_preprocessor = DataPreprocessor(self.config)
            X_train, X_test, y_train, y_test, fitted_preprocessor = data_preprocessor.preprocess_data(data_df)
            self.logger.info(f"After Preprocessing Features: ({len(X_train.columns)}), {X_train.columns}")

            # 3. Feature Selection, Hyperparameter Tuning, and Model Training
            self.logger.info("Step 3: Feature Selection, Hyperparameter Tuning, and Model Training")

            # Hyperparameter Tuning needed if True
            tuning = safe_get(self.config, 'training', 'tuning', required=True)

            # Manually drop features if True
            drop_feature = safe_get(self.config, 'training', 'drop_feature', required=True)
            self.logger.info(f"Drop features: {drop_feature}")
            features_to_drop = None
            if drop_feature:
                features_to_drop = safe_get(self.config, 'feature_selection', 'features_to_drop', required=True)
                X_train, X_test = self.drop_features(X_train, X_test, features_to_drop)
                self.logger.info(f"After Drop Features: ({len(X_train.columns)}), {X_train.columns}")

            # Feature selection needed if True
            select_feature = safe_get(self.config, 'training', 'select_feature', required=True)
            self.logger.info(f"Feature selection: {select_feature}")
            fitted_selector = None
            if select_feature:
                selector = FeatureSelector(self.config)
                X_train, X_test, fitted_selector = selector.select_features(X_train, y_train, X_test)
                self.logger.info(f"Selected features: {X_train.columns}")

            # Init Model Trainer
            trainer = ModelTrainer(self.config)

            self.logger.info(f"Hyperparameter tuning: {tuning}")
            best_params = None
            if tuning:
                self.logger.info("Hyperparameter Tuning")
                best_params = trainer.find_best_parameters(X_train, y_train)

            # Train model
            self.logger.info("Model Training")
            model, cv_scores = trainer.train_model(X_train, y_train, best_params)

            # 4. Model Evaluation
            self.logger.info("Step 4: Model Evaluation")
            evaluator = ModelEvaluator(self.config)
            metrics, cm, class_report = evaluator.evaluate_model(model, X_test, y_test)
            scoring = safe_get(self.config, 'training', 'scoring_metric', required=True)
            self.logger.info(f"{scoring}: {metrics.get(scoring, 'N/A'):.4f}")
            
            # 5. Save Model
            self.logger.info("Step 5: Saving Model")
            #trainer.save_model(fitted_preprocessor)
            trainer.save_model(preprocessor=fitted_preprocessor, features_to_drop=features_to_drop, selector=fitted_selector, model=model)

            self.logger.info("ML Pipeline completed successfully!")
            
            return {
                'model': model,
                #'preprocessor': fitted_preprocessor,
                #'selector': fitted_selector,
                #'features_to_drop': features_to_drop,
                'metrics': metrics,
                'cv_scores': cv_scores,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def drop_features(self, X_train, X_test, features_to_drop: list):
        """?"""
        X_train = X_train.drop(columns=features_to_drop)
        X_test = X_test.drop(columns=features_to_drop)

        self.logger.info(f"Drop features: {features_to_drop}")
        return X_train, X_test
