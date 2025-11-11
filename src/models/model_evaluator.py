import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import os
from utils.helpers import setup_logging

class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logging()
        
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {}
        evaluation_metrics = self.config['evaluation']['metrics']
        
        if 'accuracy' in evaluation_metrics:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
        if 'precision' in evaluation_metrics:
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
        if 'recall' in evaluation_metrics:
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        if 'f1' in evaluation_metrics:
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
        if 'f1_macro' in evaluation_metrics:
            metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        if 'roc_auc' in evaluation_metrics and y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1], multi_class='ovr')
        
        # Generate reports
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        self.logger.info("Model evaluation completed")
        self.logger.info(f"Test Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        
        # Save reports if configured
        if self.config['evaluation']['save_reports']:
            self.save_evaluation_reports(metrics, cm, class_report, y_test, y_pred)
        
        return metrics, cm, class_report
    
    def save_evaluation_reports(self, metrics, cm, class_report, y_test, y_pred):
        """Save evaluation reports and plots"""
        reports_path = self.config['evaluation']['reports_path']
        os.makedirs(reports_path, exist_ok=True)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{reports_path}/metrics.csv", index=False)
        
        # Save classification report
        class_report_df = pd.DataFrame(class_report).transpose()
        class_report_df.to_csv(f"{reports_path}/classification_report.csv")
        
        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{reports_path}/confusion_matrix.png")
        plt.close()
        
        self.logger.info(f"Evaluation reports saved to {reports_path}")