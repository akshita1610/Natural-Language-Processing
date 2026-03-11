"""
Sentiment Classification Model
Supports multiple ML algorithms and embedding techniques
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, SpatialDropout1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class SentimentClassifier:
    """
    Sentiment classification model with multiple algorithm support
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the sentiment classifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config["sentiment_model"]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model components
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        
        # Model paths
        self.models_dir = Path(config["paths"]["models_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = Path(config["paths"]["model_save"])
        
        # Supported models
        self.supported_models = {
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'random_forest': RandomForestClassifier,
            'naive_bayes': MultinomialNB
        }
        
        # Supported embeddings
        self.supported_embeddings = {
            'tfidf': TfidfVectorizer,
            'count': CountVectorizer
        }
    
    def prepare_data(self, texts: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            texts: List of preprocessed text strings
            labels: List of sentiment labels
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Convert labels to numeric
        unique_labels = list(set(labels))
        self.label_mapping = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([self.label_mapping[label] for label in labels])
        
        # Convert texts to features
        embedding_type = self.model_config["embedding_type"]
        
        if embedding_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.model_config["max_features"],
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
        elif embedding_type == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.model_config["max_features"],
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
        
        X = self.vectorizer.fit_transform(texts)
        
        self.logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_model(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Train the sentiment classification model
        
        Args:
            texts: List of preprocessed text strings
            labels: List of sentiment labels
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Starting model training...")
        
        # Prepare data
        X, y = self.prepare_data(texts, labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.model_config["test_size"],
            random_state=self.model_config["random_state"],
            stratify=y
        )
        
        # Initialize model
        model_type = self.model_config["model_type"]
        
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=self.model_config["random_state"],
                max_iter=1000
            )
        elif model_type == 'svm':
            self.model = SVC(
                random_state=self.model_config["random_state"],
                probability=True
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=self.model_config["random_state"],
                n_estimators=100
            )
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X.shape[1]
        }
        
        self.logger.info(f"Training completed. Accuracy: {metrics['accuracy']:.4f}")
        
        # Print detailed classification report
        self.logger.info("\nClassification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred, target_names=list(self.label_mapping.keys())))
        
        return metrics
    
    def predict(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict sentiment for new texts
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            List of dictionaries with predictions
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Transform texts
        X = self.vectorizer.transform(texts)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
        
        # Convert results
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'predicted_label': list(self.label_mapping.keys())[pred],
                'predicted_index': int(pred)
            }
            
            if probabilities is not None:
                prob_dict = {}
                for j, label in enumerate(self.label_mapping.keys()):
                    prob_dict[label] = float(probabilities[i][j])
                result['probabilities'] = prob_dict
                result['confidence'] = float(max(probabilities[i]))
            
            results.append(result)
        
        return results
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model
        
        Args:
            filepath: Path to save model (optional)
            
        Returns:
            Path to saved model
        """
        if filepath is None:
            filepath = self.model_path
        
        # Create model data
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_mapping': self.label_mapping,
            'config': self.model_config
        }
        
        # Save model
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
        
        return str(filepath)
    
    def load_model(self, filepath: Optional[str] = None) -> bool:
        """
        Load a trained model
        
        Args:
            filepath: Path to model file (optional)
            
        Returns:
            True if model loaded successfully
        """
        if filepath is None:
            filepath = self.model_path
        
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.label_mapping = model_data['label_mapping']
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def hyperparameter_tuning(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            texts: List of preprocessed text strings
            labels: List of sentiment labels
            
        Returns:
            Dictionary with best parameters and score
        """
        self.logger.info("Starting hyperparameter tuning...")
        
        # Prepare data
        X, y = self.prepare_data(texts, labels)
        
        # Define parameter grids
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        model_type = self.model_config["model_type"]
        
        if model_type not in param_grids:
            self.logger.warning(f"No parameter grid defined for {model_type}")
            return {}
        
        # Initialize model
        model = self.supported_models[model_type]()
        
        # Grid search
        grid_search = GridSearchCV(
            model,
            param_grids[model_type],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        results = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_
        }
        
        self.logger.info(f"Best score: {results['best_score']:.4f}")
        self.logger.info(f"Best parameters: {results['best_params']}")
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance for interpretable models
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importance per class
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        if hasattr(self.model, 'coef_'):
            # Linear models
            coef = self.model.coef_
            if len(coef.shape) == 1:
                coef = coef.reshape(1, -1)
            
            importance_dict = {}
            for i, label in enumerate(self.label_mapping.keys()):
                coef_for_class = coef[i]
                top_indices = np.argsort(np.abs(coef_for_class))[-top_n:][::-1]
                top_features = [
                    (feature_names[idx], coef_for_class[idx]) 
                    for idx in top_indices
                ]
                importance_dict[label] = top_features
            
            return importance_dict
        
        elif hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]
            top_features = [
                (feature_names[idx], importances[idx]) 
                for idx in top_indices
            ]
            
            return {'feature_importance': top_features}
        
        else:
            self.logger.warning("Feature importance not available for this model type")
            return {}

class DeepLearningSentimentModel:
    """
    Deep learning sentiment analysis models (LSTM, BERT)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize deep learning model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config["sentiment_model"]
        self.logger = logging.getLogger(__name__)
        
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available. Deep learning models disabled.")
    
    def build_lstm_model(self, vocab_size: int, max_length: int, num_classes: int) -> tf.keras.Model:
        """
        Build LSTM model for sentiment analysis
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            num_classes: Number of sentiment classes
            
        Returns:
            Compiled Keras model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        
        model = Sequential([
            Embedding(vocab_size, 128, input_length=max_length),
            SpatialDropout1D(0.3),
            LSTM(128, dropout=0.3, recurrent_dropout=0.3),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_sequences(self, texts: List[str], max_length: int = 100) -> Tuple[np.ndarray, Tokenizer]:
        """
        Prepare text sequences for LSTM
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (sequences, tokenizer)
        """
        tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        
        return sequences, tokenizer

def main():
    """
    Example usage of SentimentClassifier
    """
    import json
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize classifier
    classifier = SentimentClassifier(config)
    
    # Example data (you would use real Instagram data)
    sample_texts = [
        "I love this amazing product! It's fantastic and works perfectly.",
        "This is terrible. I hate it and it doesn't work at all.",
        "It's okay, nothing special but not bad either.",
        "Absolutely wonderful! Best purchase I've made this year.",
        "Poor quality, waste of money, very disappointed."
    ]
    
    sample_labels = ['positive', 'negative', 'neutral', 'positive', 'negative']
    
    # Train model
    metrics = classifier.train_model(sample_texts, sample_labels)
    print(f"Training metrics: {metrics}")
    
    # Save model
    classifier.save_model()
    
    # Test predictions
    test_texts = [
        "This is great!",
        "I don't like this product."
    ]
    
    predictions = classifier.predict(test_texts)
    for text, pred in zip(test_texts, predictions):
        print(f"Text: '{text}' -> Sentiment: {pred['predicted_label']} (Confidence: {pred.get('confidence', 'N/A')})")

if __name__ == "__main__":
    main()
