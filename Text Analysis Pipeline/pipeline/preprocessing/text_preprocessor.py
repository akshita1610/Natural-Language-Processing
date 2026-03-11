"""
Text Preprocessor Module
Implements comprehensive text cleaning and normalization pipeline
"""

import pandas as pd
import re
import string
from typing import List, Dict, Any, Optional
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import spacy

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration dictionary containing preprocessing settings
        """
        self.config = config.get('preprocessing', {})
        self._download_nltk_data()
        self._initialize_nlp_components()
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading NLTK wordnet...")
            nltk.download('wordnet', quiet=True)
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            logger.info("Downloading NLTK POS tagger...")
            nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def _initialize_nlp_components(self):
        """Initialize NLP components"""
        # Stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Stemmer (alternative to lemmatization)
        self.stemmer = PorterStemmer()
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            self.has_spacy = True
        except OSError:
            logger.warning("spaCy English model not found. Some features will be limited.")
            self.nlp = None
            self.has_spacy = False
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Apply preprocessing to entire DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            
        Returns:
            DataFrame with preprocessed text
        """
        logger.info(f"Preprocessing {len(df)} text documents...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Apply preprocessing pipeline
        processed_df['processed_text'] = processed_df[text_column].apply(self._preprocess_single_text)
        
        # Add additional metadata
        processed_df['original_length'] = processed_df[text_column].str.len()
        processed_df['processed_length'] = processed_df['processed_text'].str.len()
        processed_df['token_count'] = processed_df['processed_text'].apply(lambda x: len(x.split()) if x else 0)
        
        # Filter out empty processed texts
        processed_df = processed_df[processed_df['processed_text'].str.len() > 0]
        
        logger.info(f"Preprocessing complete. {len(processed_df)} documents remain.")
        return processed_df.reset_index(drop=True)
    
    def _preprocess_single_text(self, text: str) -> str:
        """
        Apply complete preprocessing pipeline to single text
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # 1. Basic cleaning
        text = self._basic_cleaning(text)
        
        # 2. Lowercase
        if self.config.get('lowercase', True):
            text = text.lower()
        
        # 3. Tokenization
        tokens = word_tokenize(text)
        
        # 4. Filter tokens
        tokens = self._filter_tokens(tokens)
        
        # 5. Normalization (lemmatization or stemming)
        if self.config.get('lemmatize', True):
            tokens = self._lemmatize_tokens(tokens)
        else:
            tokens = self._stem_tokens(tokens)
        
        # 6. Join tokens back to text
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def _basic_cleaning(self, text: str) -> str:
        """Apply basic text cleaning operations"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle punctuation
        if self.config.get('remove_punctuation', True):
            # Keep basic punctuation that might be important for sentiment
            text = re.sub(r'[^\w\s\!\?\.\,\;\:]', '', text)
        
        # Remove numbers if configured
        if self.config.get('remove_numbers', False):
            text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens based on various criteria"""
        min_length = self.config.get('min_word_length', 2)
        max_length = self.config.get('max_word_length', 20)
        
        filtered_tokens = []
        
        for token in tokens:
            # Length filter
            if len(token) < min_length or len(token) > max_length:
                continue
            
            # Stopword filter
            if self.config.get('remove_stopwords', True) and token.lower() in self.stop_words:
                continue
            
            # Filter out tokens that are only punctuation
            if token in string.punctuation:
                continue
            
            # Filter out tokens that are just numbers or special characters
            if re.match(r'^[^\w]+$', token):
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        if self.has_spacy:
            # Use spaCy for better lemmatization
            text = ' '.join(tokens)
            doc = self.nlp(text)
            return [token.lemma_ for token in doc if not token.is_space]
        else:
            # Fallback to NLTK lemmatizer
            return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def _stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def generate_ngrams(self, texts: List[str], n: int = 2) -> List[str]:
        """
        Generate n-grams from preprocessed texts
        
        Args:
            texts: List of preprocessed texts
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        all_ngrams = []
        
        for text in texts:
            tokens = text.split()
            if len(tokens) >= n:
                for i in range(len(tokens) - n + 1):
                    ngram = ' '.join(tokens[i:i+n])
                    all_ngrams.append(ngram)
        
        return all_ngrams
    
    def get_preprocessing_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate statistics about preprocessing results
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Dictionary with preprocessing statistics
        """
        stats = {
            'total_documents': len(df),
            'avg_original_length': df['original_length'].mean(),
            'avg_processed_length': df['processed_length'].mean(),
            'avg_token_count': df['token_count'].mean(),
            'total_tokens': df['token_count'].sum(),
            'length_reduction_percent': ((df['original_length'] - df['processed_length']).sum() / df['original_length'].sum()) * 100
        }
        
        return stats
