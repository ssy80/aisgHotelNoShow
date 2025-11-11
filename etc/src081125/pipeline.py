import sys
import os
from utils import load_config, setup_logging
from ingestion import Ingestor
from preprocessing import Preprocessor
from modelTrainer import ModelTrainer
from modelEvaluator import ModelEvaluator



class MlPipeline:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = setup_logging()
        
    def run(self):
        """Execute the complete ML pipeline"""
        self.logger.info("Starting ML Pipeline")
        
        try:
            # 1. Data Ingestion
            self.logger.info("Step 1: Data Ingestion")
            data_ingestor = Ingestor(self.config)
            data_df = data_ingestor.load_data()
            
            if not data_ingestor.validate_data(data_df):
                raise ValueError("Data validation failed")
            
            print(data_df)

            # 2. Data Preprocessing
            self.logger.info("Step 2: Data Preprocessing")
            preprocessor = Preprocessor(self.config)
            #preprocessor.preprocess_data(data_df)
            X_train, X_test, y_train, y_test, fitted_preprocessor = preprocessor.preprocess_data(data_df)
            
            print(X_train)
            print(y_train.dtypes)
            
            # 3. Model Training
            self.logger.info("Step 3: Model Training")
            trainer = ModelTrainer(self.config)
            model, cv_scores = trainer.train_model(X_train, y_train)
            
            # 4. Model Evaluation
            self.logger.info("Step 4: Model Evaluation")
            evaluator = ModelEvaluator(self.config)
            metrics, cm, class_report = evaluator.evaluate_model(model, X_test, y_test)
            
            # 5. Save Model
            #self.logger.info("Step 5: Saving Model")
            #trainer.save_model(fitted_preprocessor)
            
            #self.logger.info("ML Training Pipeline completed successfully!")
            
            '''return {
                'model': model,
                'preprocessor': fitted_preprocessor,
                'metrics': metrics,
                'cv_scores': cv_scores,
                'confusion_matrix': cm
            }'''
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise