"""
Utility functions and helper classes for Instagram Sentiment Analyzer
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import re
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Data validation and quality checks
    """
    
    @staticmethod
    def validate_instagram_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate Instagram data structure and quality
        
        Args:
            df: DataFrame with Instagram data
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Required columns
        required_columns = ['caption', 'likes', 'comments_count', 'timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            results['is_valid'] = False
            results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Data quality checks
        if 'caption' in df.columns:
            empty_captions = df['caption'].isna().sum()
            if empty_captions > 0:
                results['warnings'].append(f"{empty_captions} posts have empty captions")
        
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'])
            except:
                results['is_valid'] = False
                results['errors'].append("Invalid timestamp format")
        
        # Statistics
        results['stats'] = {
            'total_posts': len(df),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict()
        }
        
        return results
    
    @staticmethod
    def clean_text_data(df: pd.DataFrame, text_column: str = 'caption') -> pd.DataFrame:
        """
        Clean and standardize text data
        
        Args:
            df: Input DataFrame
            text_column: Name of text column to clean
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove rows with empty text
        df_clean = df_clean[df_clean[text_column].notna()]
        df_clean = df_clean[df_clean[text_column].str.strip() != '']
        
        # Remove duplicates based on text
        df_clean = df_clean.drop_duplicates(subset=[text_column])
        
        logger.info(f"Cleaned data: {len(df)} -> {len(df_clean)} rows")
        return df_clean

class FileManager:
    """
    File management utilities
    """
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
        """
        Save data to JSON file
        
        Args:
            data: Data to save
            filepath: Output file path
            indent: JSON indentation
        """
        filepath = Path(filepath)
        FileManager.ensure_directory(filepath.parent)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        logger.info(f"Data saved to {filepath}")
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Optional[Dict]:
        """
        Load data from JSON file
        
        Args:
            filepath: Input file path
            
        Returns:
            Loaded data or None if error
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Data loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON from {filepath}: {e}")
            return None
    
    @staticmethod
    def get_file_hash(filepath: Union[str, Path]) -> str:
        """
        Get file hash for change detection
        
        Args:
            filepath: File path
            
        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

class ConfigManager:
    """
    Configuration management utilities
    """
    
    @staticmethod
    def load_config(config_path: Union[str, Path] = "config.json") -> Dict:
        """
        Load configuration from file
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        config = FileManager.load_json(config_path)
        if config is None:
            logger.warning("Using default configuration")
            config = ConfigManager.get_default_config()
        return config
    
    @staticmethod
    def save_config(config: Dict, config_path: Union[str, Path] = "config.json") -> None:
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary
            config_path: Path to save config
        """
        FileManager.save_json(config, config_path)
    
    @staticmethod
    def get_default_config() -> Dict:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            "data_collection": {
                "max_posts": 100,
                "max_comments_per_post": 50,
                "download_videos": False,
                "download_pictures": False,
                "save_metadata": True
            },
            "preprocessing": {
                "remove_stopwords": True,
                "lemmatize": True,
                "remove_urls": True,
                "remove_mentions": True,
                "convert_emojis": True,
                "min_word_length": 2,
                "language": "english"
            },
            "sentiment_model": {
                "model_type": "logistic_regression",
                "embedding_type": "tfidf",
                "max_features": 10000,
                "test_size": 0.2,
                "random_state": 42
            },
            "visualization": {
                "color_scheme": {
                    "positive": "#2E8B57",
                    "negative": "#DC143C",
                    "neutral": "#708090"
                },
                "figure_size": [12, 8],
                "dpi": 300
            },
            "paths": {
                "data_dir": "data",
                "models_dir": "models",
                "raw_data": "data/raw",
                "processed_data": "data/processed",
                "model_save": "models/sentiment_model.pkl"
            }
        }

class TextAnalyzer:
    """
    Text analysis utilities
    """
    
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        """
        Calculate simple readability score
        
        Args:
            text: Input text
            
        Returns:
            Readability score (0-1, higher is easier to read)
        """
        if not text:
            return 0.0
        
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        
        if sentences == 0:
            return 0.0
        
        avg_sentence_length = words / sentences
        
        # Simple readability formula (inverse of average sentence length)
        readability = min(1.0, 10.0 / avg_sentence_length)
        
        return readability
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        from collections import Counter
        import nltk
        from nltk.corpus import stopwords
        
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [word for word in words if word not in stop_words]
        
        word_freq = Counter(words)
        keywords = [word for word, freq in word_freq.most_common(top_n)]
        
        return keywords
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Simple language detection
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        # Simple heuristic based on common words
        english_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with'}
        spanish_words = {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no'}
        french_words = {'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'}
        
        text_lower = text.lower()
        words = set(text.split())
        
        english_count = len(words.intersection(english_words))
        spanish_count = len(words.intersection(spanish_words))
        french_count = len(words.intersection(french_words))
        
        if english_count > spanish_count and english_count > french_count:
            return 'en'
        elif spanish_count > english_count and spanish_count > french_count:
            return 'es'
        elif french_count > english_count and french_count > spanish_count:
            return 'fr'
        else:
            return 'unknown'

class PerformanceMonitor:
    """
    Performance monitoring utilities
    """
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_time = datetime.now()
        self.metrics[operation] = {'start': self.start_time}
    
    def end_timer(self, operation: str):
        """End timing an operation"""
        if self.start_time:
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            if operation in self.metrics:
                self.metrics[operation]['end'] = end_time
                self.metrics[operation]['duration'] = duration
            
            logger.info(f"Operation '{operation}' completed in {duration:.2f} seconds")
            return duration
        return None
    
    def get_metrics(self) -> Dict:
        """Get all performance metrics"""
        return self.metrics

class DataAugmenter:
    """
    Data augmentation utilities for training
    """
    
    @staticmethod
    def augment_text_data(texts: List[str], labels: List[str], 
                        augment_factor: int = 2) -> tuple:
        """
        Augment text data using simple techniques
        
        Args:
            texts: List of original texts
            labels: List of corresponding labels
            augment_factor: Factor to augment data
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        for text, label in zip(texts, labels):
            # Simple augmentation: random word shuffling
            words = text.split()
            if len(words) > 3:
                for _ in range(augment_factor - 1):
                    np.random.shuffle(words)
                    augmented_text = ' '.join(words)
                    augmented_texts.append(augmented_text)
                    augmented_labels.append(label)
        
        return augmented_texts, augmented_labels

class CacheManager:
    """
    Simple caching utilities
    """
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, data: Any) -> str:
        """Generate cache key from data"""
        import pickle
        data_str = pickle.dumps(data)
        return hashlib.md5(data_str).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data"""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None
    
    def set(self, key: str, data: Any) -> None:
        """Set cached data"""
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error caching data: {e}")

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    log_config = {
        'level': getattr(logging, log_level.upper()),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if log_file:
        log_config['filename'] = log_file
    
    logging.basicConfig(**log_config)

def validate_input(text: str, min_length: int = 3) -> bool:
    """
    Validate input text
    
    Args:
        text: Input text
        min_length: Minimum required length
        
    Returns:
        True if valid
    """
    if not isinstance(text, str):
        return False
    
    if len(text.strip()) < min_length:
        return False
    
    return True

def format_timestamp(timestamp: Union[str, datetime]) -> str:
    """
    Format timestamp for display
    
    Args:
        timestamp: Input timestamp
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def calculate_engagement_rate(likes: int, comments: int, followers: int) -> float:
    """
    Calculate engagement rate
    
    Args:
        likes: Number of likes
        comments: Number of comments
        followers: Number of followers
        
    Returns:
        Engagement rate as percentage
    """
    if followers == 0:
        return 0.0
    
    engagement = (likes + comments) / followers * 100
    return round(engagement, 2)

def main():
    """
    Example usage of utility functions
    """
    # Setup logging
    setup_logging("INFO")
    
    # Test configuration
    config = ConfigManager.load_config()
    logger.info("Configuration loaded successfully")
    
    # Test data validation
    sample_data = {
        'caption': ['Test post 1', 'Test post 2'],
        'likes': [100, 200],
        'comments_count': [10, 20],
        'timestamp': ['2024-01-01', '2024-01-02']
    }
    
    df = pd.DataFrame(sample_data)
    validation_results = DataValidator.validate_instagram_data(df)
    logger.info(f"Data validation: {validation_results}")
    
    # Test text analysis
    sample_text = "This is a test post with some content and hashtags #test #sample"
    keywords = TextAnalyzer.extract_keywords(sample_text)
    readability = TextAnalyzer.calculate_readability_score(sample_text)
    
    logger.info(f"Keywords: {keywords}")
    logger.info(f"Readability score: {readability:.2f}")

if __name__ == "__main__":
    main()
