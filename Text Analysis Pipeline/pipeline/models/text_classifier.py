"""
Text Classification Module
Implements classical ML classifiers for text categorization
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

logger = logging.getLogger(__name__)


class TextClassifier:
    """
    Multi-algorithm text classification system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize text classifier with configuration
        
        Args:
            config: Configuration dictionary containing classification settings
        """
        self.config = config.get('models', {}).get('classification', {})
        self.models = {}
        self.trained_models = {}
        self.feature_extractors = {}
        
    def prepare_data(self, texts: List[str], labels: List[str], 
                    test_size: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and testing data
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = self.config.get('test_size', 0.2)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, 
            test_size=test_size, 
            random_state=self.config.get('random_state', 42),
            stratify=labels
        )
        
        logger.info(f"Data split: {len(X_train)} training, {len(X_test)} testing samples")
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray, 
                                param_grid: Optional[Dict] = None) -> LogisticRegression:
        """
        Train Logistic Regression classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for hyperparameter tuning
            
        Returns:
            Trained Logistic Regression model
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            }
        
        # Base model
        lr = LogisticRegression(random_state=self.config.get('random_state', 42))
        
        # Grid search for best parameters
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best Logistic Regression params: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return best_model
    
    def train_naive_bayes(self, X_train: np.ndarray, y_train: np.ndarray,
                         param_grid: Optional[Dict] = None) -> MultinomialNB:
        """
        Train Multinomial Naive Bayes classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for hyperparameter tuning
            
        Returns:
            Trained Naive Bayes model
        """
        if param_grid is None:
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 2.0],
                'fit_prior': [True, False]
            }
        
        # Base model
        nb = MultinomialNB()
        
        # Grid search for best parameters
        grid_search = GridSearchCV(
            nb, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best Naive Bayes params: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return best_model
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                 param_grid: Optional[Dict] = None) -> SVC:
        """
        Train Support Vector Machine classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for hyperparameter tuning
            
        Returns:
            Trained SVM model
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        
        # Base model
        svm = SVC(random_state=self.config.get('random_state', 42))
        
        # Grid search for best parameters
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best SVM params: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return best_model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           param_grid: Optional[Dict] = None) -> RandomForestClassifier:
        """
        Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for hyperparameter tuning
            
        Returns:
            Trained Random Forest model
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        # Base model
        rf = RandomForestClassifier(random_state=self.config.get('random_state', 42))
        
        # Grid search for best parameters
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best Random Forest params: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return best_model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all configured classification models
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary containing trained models
        """
        algorithms = self.config.get('algorithms', ['logistic_regression', 'naive_bayes'])
        trained_models = {}
        
        for algorithm in algorithms:
            logger.info(f"Training {algorithm}...")
            
            try:
                if algorithm == 'logistic_regression':
                    model = self.train_logistic_regression(X_train, y_train)
                elif algorithm == 'naive_bayes':
                    model = self.train_naive_bayes(X_train, y_train)
                elif algorithm == 'svm':
                    model = self.train_svm(X_train, y_train)
                elif algorithm == 'random_forest':
                    model = self.train_random_forest(X_train, y_train)
                else:
                    logger.warning(f"Unknown algorithm: {algorithm}")
                    continue
                
                trained_models[algorithm] = model
                logger.info(f"Successfully trained {algorithm}")
                
            except Exception as e:
                logger.error(f"Error training {algorithm}: {str(e)}")
        
        self.trained_models = trained_models
        return trained_models
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Detailed classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_test, y_test, cv=5)
        
        evaluation = {
            'model_name': model_name,
            'accuracy': accuracy,
            'cv_scores': {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores.tolist()
            },
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist()
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return evaluation
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation results for all models
        """
        evaluations = {}
        
        for model_name, model in self.trained_models.items():
            evaluations[model_name] = self.evaluate_model(
                model, X_test, y_test, model_name
            )
        
        return evaluations
    
    def predict(self, texts: List[str], model_name: str = None) -> np.ndarray:
        """
        Make predictions on new texts
        
        Args:
            texts: List of texts to classify
            model_name: Name of model to use (best model if None)
            
        Returns:
            Predicted labels
        """
        if not self.trained_models:
            raise ValueError("No trained models available")
        
        # Select model
        if model_name is None:
            # Use the first available model (could be enhanced to use best performing)
            model_name = list(self.trained_models.keys())[0]
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.trained_models[model_name]
        
        # Note: This assumes features are already extracted
        # In practice, you'd need to extract features first
        predictions = model.predict(texts)
        
        return predictions
    
    def get_feature_importance(self, model_name: str, feature_names: List[str], 
                             top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance for models that support it
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and importance scores
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'coef_'):
            # Linear models (Logistic Regression, Linear SVM)
            if len(model.coef_) == 1:
                # Binary classification
                importance_scores = model.coef_[0]
            else:
                # Multi-class - use average importance
                importance_scores = np.mean(np.abs(model.coef_), axis=0)
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models (Random Forest)
            importance_scores = model.feature_importances_
        else:
            logger.warning(f"Model {model_name} does not support feature importance")
            return {}
        
        # Get top features
        top_indices = np.argsort(np.abs(importance_scores))[-top_n:][::-1]
        
        return {
            feature_names[i]: float(importance_scores[i])
            for i in top_indices
        }
    
    def save_models(self, output_dir: str) -> None:
        """
        Save trained models to files
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_path = os.path.join(output_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {model_name} model to {model_path}")
    
    def load_models(self, model_dir: str) -> Dict[str, Any]:
        """
        Load trained models from files
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            Dictionary of loaded models
        """
        loaded_models = {}
        
        for file in os.listdir(model_dir):
            if file.endswith('.pkl'):
                model_name = file.replace('.pkl', '')
                model_path = os.path.join(model_dir, file)
                
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    loaded_models[model_name] = model
                    logger.info(f"Loaded {model_name} model")
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {str(e)}")
        
        self.trained_models = loaded_models
        return loaded_models
