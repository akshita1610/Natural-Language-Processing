"""
Feature Extraction Module
Implements TF-IDF, word embeddings, and other feature extraction methods
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.models import Word2Vec, KeyedVectors
import pickle
import os

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Comprehensive feature extraction for NLP tasks
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature extractor with configuration
        
        Args:
            config: Configuration dictionary containing feature extraction settings
        """
        self.config = config.get('features', {})
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.embeddings_matrix = None
        
    def extract_tfidf_features(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Extract TF-IDF features from texts
        
        Args:
            texts: List of preprocessed texts
            fit: Whether to fit the vectorizer (True for training, False for inference)
            
        Returns:
            TF-IDF feature matrix
        """
        tfidf_config = self.config.get('tfidf', {})
        
        if fit or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=tfidf_config.get('max_features', 5000),
                ngram_range=tuple(tfidf_config.get('ngram_range', [1, 2])),
                min_df=tfidf_config.get('min_df', 2),
                max_df=tfidf_config.get('max_df', 0.95),
                stop_words='english'
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            
        return tfidf_matrix.toarray()
    
    def extract_bag_of_words(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Extract Bag-of-Words features
        
        Args:
            texts: List of preprocessed texts
            fit: Whether to fit the vectorizer
            
        Returns:
            BoW feature matrix
        """
        if fit or self.bow_vectorizer is None:
            self.bow_vectorizer = CountVectorizer(
                max_features=self.config.get('max_features', 5000),
                ngram_range=tuple(self.config.get('ngram_range', [1, 1])),
                min_df=self.config.get('min_df', 2),
                max_df=self.config.get('max_df', 0.95),
                stop_words='english'
            )
            
            bow_matrix = self.bow_vectorizer.fit_transform(texts)
        else:
            bow_matrix = self.bow_vectorizer.transform(texts)
            
        return bow_matrix.toarray()
    
    def train_word2vec(self, texts: List[str]) -> Word2Vec:
        """
        Train Word2Vec embeddings
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Trained Word2Vec model
        """
        word2vec_config = self.config.get('word_embeddings', {})
        
        # Tokenize texts for Word2Vec
        tokenized_texts = [text.split() for text in texts]
        
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=word2vec_config.get('vector_size', 100),
            window=word2vec_config.get('window', 5),
            min_count=word2vec_config.get('min_count', 2),
            workers=4,
            sg=1  # Skip-gram model
        )
        
        logger.info(f"Word2Vec model trained. Vocabulary size: {len(self.word2vec_model.wv)}")
        return self.word2vec_model
    
    def get_document_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get document-level embeddings by averaging word vectors
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Document embedding matrix
        """
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained. Call train_word2vec() first.")
        
        embeddings = []
        
        for text in texts:
            words = text.split()
            word_vectors = []
            
            for word in words:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            
            if word_vectors:
                # Average of word vectors
                doc_embedding = np.mean(word_vectors, axis=0)
            else:
                # Zero vector if no words found in vocabulary
                doc_embedding = np.zeros(self.word2vec_model.vector_size)
            
            embeddings.append(doc_embedding)
        
        return np.array(embeddings)
    
    def extract_topic_features(self, texts: List[str], n_topics: int = 10, method: str = 'lda') -> Tuple[np.ndarray, Any]:
        """
        Extract topic modeling features
        
        Args:
            texts: List of preprocessed texts
            n_topics: Number of topics to extract
            method: Topic modeling method ('lda' or 'nmf')
            
        Returns:
            Topic feature matrix and trained model
        """
        # First get TF-IDF features
        tfidf_matrix = self.extract_tfidf_features(texts, fit=True)
        
        if method.lower() == 'lda':
            model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )
        elif method.lower() == 'nmf':
            model = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=200
            )
        else:
            raise ValueError("Method must be 'lda' or 'nmf'")
        
        topic_features = model.fit_transform(tfidf_matrix)
        
        logger.info(f"Topic modeling complete. Features shape: {topic_features.shape}")
        return topic_features, model
    
    def get_top_words_for_topics(self, model: Any, n_top_words: int = 10) -> List[List[str]]:
        """
        Get top words for each topic
        
        Args:
            model: Trained topic model
            n_top_words: Number of top words to return per topic
            
        Returns:
            List of top words for each topic
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted")
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        top_words = []
        
        for topic_idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[-n_top_words:][::-1]
            top_words.append([feature_names[i] for i in top_indices])
        
        return top_words
    
    def create_feature_matrix(self, texts: List[str], feature_types: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Create comprehensive feature matrix with multiple feature types
        
        Args:
            texts: List of preprocessed texts
            feature_types: List of feature types to extract
            
        Returns:
            Dictionary containing different feature matrices
        """
        if feature_types is None:
            feature_types = ['tfidf']
        
        features = {}
        
        if 'tfidf' in feature_types:
            logger.info("Extracting TF-IDF features...")
            features['tfidf'] = self.extract_tfidf_features(texts, fit=True)
        
        if 'bow' in feature_types:
            logger.info("Extracting Bag-of-Words features...")
            features['bow'] = self.extract_bag_of_words(texts, fit=True)
        
        if 'word2vec' in feature_types:
            logger.info("Training Word2Vec model...")
            self.train_word2vec(texts)
            features['word2vec'] = self.get_document_embeddings(texts)
        
        if 'topics' in feature_types:
            logger.info("Extracting topic features...")
            topic_features, topic_model = self.extract_topic_features(texts)
            features['topics'] = topic_features
            features['topic_model'] = topic_model
        
        return features
    
    def save_features(self, features: Dict[str, np.ndarray], output_dir: str) -> None:
        """
        Save extracted features to files
        
        Args:
            features: Dictionary of feature matrices
            output_dir: Directory to save features
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for feature_name, feature_matrix in features.items():
            if isinstance(feature_matrix, np.ndarray):
                output_path = os.path.join(output_dir, f"{feature_name}_features.npz")
                np.savez_compressed(output_path, features=feature_matrix)
                logger.info(f"Saved {feature_name} features to {output_path}")
        
        # Save vectorizers and models
        if self.tfidf_vectorizer is not None:
            with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        if self.word2vec_model is not None:
            self.word2vec_model.save(os.path.join(output_dir, "word2vec_model.bin"))
            logger.info("Saved Word2Vec model")
    
    def load_features(self, feature_dir: str) -> Dict[str, np.ndarray]:
        """
        Load saved features
        
        Args:
            feature_dir: Directory containing saved features
            
        Returns:
            Dictionary of loaded feature matrices
        """
        features = {}
        
        # Load feature matrices
        for file in os.listdir(feature_dir):
            if file.endswith('_features.npz'):
                feature_name = file.replace('_features.npz', '')
                data = np.load(os.path.join(feature_dir, file))
                features[feature_name] = data['features']
                logger.info(f"Loaded {feature_name} features")
        
        # Load vectorizers
        if os.path.exists(os.path.join(feature_dir, "tfidf_vectorizer.pkl")):
            with open(os.path.join(feature_dir, "tfidf_vectorizer.pkl"), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        
        if os.path.exists(os.path.join(feature_dir, "word2vec_model.bin")):
            self.word2vec_model = KeyedVectors.load(os.path.join(feature_dir, "word2vec_model.bin"))
        
        return features
    
    def get_feature_importance(self, feature_type: str = 'tfidf', top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Args:
            feature_type: Type of features ('tfidf')
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and importance scores
        """
        if feature_type == 'tfidf' and self.tfidf_vectorizer is not None:
            # Get average TF-IDF scores across all documents
            if hasattr(self.tfidf_vectorizer, 'idf_'):
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                importance_scores = self.tfidf_vectorizer.idf_
                
                # Sort by importance (lower IDF = more important)
                top_indices = np.argsort(importance_scores)[:top_n]
                
                return {
                    feature_names[i]: importance_scores[i] 
                    for i in top_indices
                }
        
        return {}
