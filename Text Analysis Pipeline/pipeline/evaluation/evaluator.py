"""
Model Evaluation Module
Comprehensive evaluation system for NLP models and pipelines
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and reporting system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator with configuration
        
        Args:
            config: Configuration dictionary containing evaluation settings
        """
        self.config = config.get('evaluation', {})
        self.metrics = self.config.get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
        self.cv_folds = self.config.get('cross_validation_folds', 5)
        
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       y_prob: Optional[np.ndarray] = None,
                                       average: str = 'weighted') -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for AUC)
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Basic metrics
        if 'accuracy' in self.metrics:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'precision' in self.metrics:
            metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        
        if 'recall' in self.metrics:
            metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        
        if 'f1' in self.metrics:
            metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # AUC metrics (if probabilities available)
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    if 'auc' in self.metrics:
                        metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                    if 'average_precision' in self.metrics:
                        metrics['average_precision'] = average_precision_score(y_true, y_prob[:, 1])
                else:
                    # Multi-class classification
                    if 'auc' in self.metrics:
                        metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {str(e)}")
        
        return metrics
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    target_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes
            
        Returns:
            Detailed classification report
        """
        return classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    def analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                               target_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes
            
        Returns:
            Confusion matrix analysis
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        n_classes = cm.shape[0]
        per_class_metrics = {}
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_name = target_names[i] if target_names else f"Class_{i}"
            
            per_class_metrics[class_name] = {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return {
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics,
            'total_samples': int(cm.sum())
        }
    
    def evaluate_sentiment_analysis(self, results_df: pd.DataFrame,
                                  true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate sentiment analysis results
        
        Args:
            results_df: DataFrame with sentiment analysis results
            true_labels: Optional true sentiment labels for evaluation
            
        Returns:
            Sentiment analysis evaluation report
        """
        evaluation = {
            'summary_statistics': {},
            'distribution_analysis': {},
            'method_comparison': {}
        }
        
        # Summary statistics for each method
        sentiment_columns = [col for col in results_df.columns if col.endswith('_sentiment')]
        
        for col in sentiment_columns:
            sentiment_counts = results_df[col].value_counts()
            sentiment_percentages = (sentiment_counts / len(results_df) * 100).round(2)
            
            evaluation['summary_statistics'][col] = {
                'counts': sentiment_counts.to_dict(),
                'percentages': sentiment_percentages.to_dict()
            }
        
        # Numeric statistics
        numeric_columns = [col for col in results_df.columns if any(x in col for x in ['compound', 'polarity', 'confidence'])]
        
        for col in numeric_columns:
            if col in results_df.columns:
                evaluation['summary_statistics'][col] = {
                    'mean': float(results_df[col].mean()),
                    'std': float(results_df[col].std()),
                    'min': float(results_df[col].min()),
                    'max': float(results_df[col].max()),
                    'median': float(results_df[col].median()),
                    'q25': float(results_df[col].quantile(0.25)),
                    'q75': float(results_df[col].quantile(0.75))
                }
        
        # Method comparison (if ensemble available)
        if 'ensemble_sentiment' in results_df.columns:
            ensemble_col = 'ensemble_sentiment'
            other_methods = [col for col in sentiment_columns if col != ensemble_col]
            
            for method in other_methods:
                agreement = (results_df[method] == results_df[ensemble_col]).sum() / len(results_df) * 100
                evaluation['method_comparison'][f"{method}_vs_ensemble"] = {
                    'agreement_percentage': round(agreement, 2)
                }
        
        # External evaluation if true labels provided
        if true_labels is not None and len(true_labels) == len(results_df):
            for col in sentiment_columns:
                if col in results_df.columns:
                    metrics = self.calculate_classification_metrics(
                        true_labels, results_df[col].values
                    )
                    evaluation['method_comparison'][f"{col}_external_eval"] = metrics
        
        return evaluation
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results
        
        Args:
            model_results: Dictionary of model evaluation results
            
        Returns:
            Model comparison report
        """
        comparison = {
            'model_rankings': {},
            'best_models': {},
            'detailed_comparison': {}
        }
        
        # Extract metrics for comparison
        metrics_data = {}
        
        for model_name, results in model_results.items():
            if 'accuracy' in results:
                metrics_data[model_name] = {
                    'accuracy': results['accuracy'],
                    'precision': results.get('precision', 0),
                    'recall': results.get('recall', 0),
                    'f1': results.get('f1', 0),
                    'cv_mean': results.get('cv_scores', {}).get('mean', 0)
                }
        
        if not metrics_data:
            logger.warning("No comparable metrics found in model results")
            return comparison
        
        # Create rankings for each metric
        metrics_df = pd.DataFrame(metrics_data).T
        
        for metric in metrics_df.columns:
            ranking = metrics_df[metric].rank(ascending=False).sort_values()
            comparison['model_rankings'][metric] = ranking.to_dict()
            
            # Best model for this metric
            best_model = ranking.index[0]
            comparison['best_models'][f"best_{metric}"] = {
                'model': best_model,
                'score': float(metrics_df.loc[best_model, metric])
            }
        
        # Overall ranking (average of all metrics)
        metrics_df['overall_score'] = metrics_df.mean(axis=1)
        overall_ranking = metrics_df['overall_score'].rank(ascending=False).sort_values()
        comparison['model_rankings']['overall'] = overall_ranking.to_dict()
        
        comparison['best_models']['best_overall'] = {
            'model': overall_ranking.index[0],
            'score': float(metrics_df.loc[overall_ranking.index[0], 'overall_score'])
        }
        
        comparison['detailed_comparison'] = metrics_df.round(4).to_dict()
        
        return comparison
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any],
                                 output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report in markdown format
        
        Args:
            evaluation_results: Dictionary containing all evaluation results
            output_path: Optional path to save the report
            
        Returns:
            Markdown report string
        """
        report = []
        report.append("# NLP Pipeline Evaluation Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model comparison section
        if 'model_comparison' in evaluation_results:
            report.append("## Model Performance Comparison")
            report.append("")
            
            comparison = evaluation_results['model_comparison']
            
            if 'best_models' in comparison:
                report.append("### Best Models by Metric")
                report.append("")
                for metric, best in comparison['best_models'].items():
                    report.append(f"- **{metric}**: {best['model']} (Score: {best['score']:.4f})")
                report.append("")
            
            if 'detailed_comparison' in comparison:
                report.append("### Detailed Performance Metrics")
                report.append("")
                df = pd.DataFrame(comparison['detailed_comparison']).T
                report.append(df.to_markdown(floatfmt='.4f'))
                report.append("")
        
        # Sentiment analysis section
        if 'sentiment_evaluation' in evaluation_results:
            report.append("## Sentiment Analysis Results")
            report.append("")
            
            sentiment_eval = evaluation_results['sentiment_evaluation']
            
            if 'summary_statistics' in sentiment_eval:
                report.append("### Summary Statistics")
                report.append("")
                for method, stats in sentiment_eval['summary_statistics'].items():
                    report.append(f"#### {method}")
                    if 'percentages' in stats:
                        for sentiment, percentage in stats['percentages'].items():
                            report.append(f"- {sentiment}: {percentage}%")
                    report.append("")
        
        # Classification results section
        if 'classification_results' in evaluation_results:
            report.append("## Classification Results")
            report.append("")
            
            for model_name, results in evaluation_results['classification_results'].items():
                report.append(f"### {model_name}")
                report.append(f"- **Accuracy**: {results.get('accuracy', 0):.4f}")
                report.append(f"- **Precision**: {results.get('precision', 0):.4f}")
                report.append(f"- **Recall**: {results.get('recall', 0):.4f}")
                report.append(f"- **F1-Score**: {results.get('f1', 0):.4f}")
                report.append("")
        
        report_text = "\n".join(report)
        
        # Save report if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report_text
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to JSON file
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_numpy(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            target_names: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                       model_name: str = "Model",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve plot saved to {save_path}")
        
        return plt.gcf()
