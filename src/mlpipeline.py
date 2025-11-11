import sys
import os
import logging
from utils.helpers import load_config, setup_logging
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from models.model_trainer import ModelTrainer
from models.model_evaluator import ModelEvaluator
from models.feature_selector import FeatureSelector
#import inspect
#import warnings
from models.tunning_model import TunningModel


class MlPipeline:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)

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
            #print(X_train.columns)

            # Hyperparameter Tuning needed if True
            tunning = self.config['training']['tunning']
            select_feature = self.config['training']['select_feature']
            trainer = ModelTrainer(self.config)

            # 3A. Feature Selection by select from model, tuning not needed, model training
            if tunning == False:
                self.logger.info("Step 3A: Feature Selection by select from model, tuning not needed, model training")
                
                model = trainer.get_model()
                if select_feature:
                    self.logger.info("Feature selection is True")
                    selector = FeatureSelector(self.config, model)
                    X_train, X_test, fitted_selector = selector.select_features(X_train, y_train, X_test)
                #print("Selected features:", X_train_sel.columns)
                #self.logger.info("Step 3A: Model Training")
                model, cv_scores = trainer.train_model(X_train, y_train)
            
            # 3B. Feature Selection by select from model, hyperparameter tuning, model training 
            if tunning == True:
                self.logger.info("Step 3B: Feature Selection by select from model, hyperparameter tuning, model training")
                tunning_model = TunningModel(self.config)
                model = tunning_model.get_model()
                #print(f"model_params: {model}")
                if select_feature:
                    selector = FeatureSelector(self.config, model)
                    X_train, X_test, fitted_selector = selector.select_features(X_train, y_train, X_test)
                #print("Selected features:", X_train_sel.columns)
                #self.logger.info("Step 3A: Hyperparameter Tuning")
                best_params = trainer.find_best_parameters(X_train, y_train)
                #print(best_params)
                # 3B. Train Final Model with Best Parameters
                #self.logger.info("Step 3B: Training Final Model")
                model, cv_scores = trainer.train_final_model(X_train, y_train, best_params)

            # 4. Model Evaluation
            self.logger.info("Step 4: Model Evaluation")
            evaluator = ModelEvaluator(self.config)
            metrics, cm, class_report = evaluator.evaluate_model(model, X_test, y_test)
            
            # 5. Save Model
            self.logger.info("Step 5: Saving Model")
            trainer.save_model(fitted_preprocessor)
            
            self.logger.info("ML Training Pipeline completed successfully!")
            
            return {
                'model': model,
                'preprocessor': fitted_preprocessor,
                'metrics': metrics,
                'cv_scores': cv_scores,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
