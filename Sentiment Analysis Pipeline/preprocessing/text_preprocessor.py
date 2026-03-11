"""
Text Preprocessing Module

Handles text cleaning and preprocessing for sentiment analysis.
Includes tokenization, stopword removal, emoji handling, etc.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import logging
from typing import List, Dict, Union, Optional
import emoji
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing class for cleaning and normalizing text data.
    
    Features:
    - Text cleaning (HTML tags, URLs, mentions, hashtags)
    - Tokenization and normalization
    - Stopword removal
    - Emoji handling
    - Lemmatization
    """
    
    def __init__(self, config: Dict):
        """
        Initialize text preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config
        self.lowercase = config.get('lowercase', True)
        self.remove_stopwords = config.get('remove_stopwords', True)
        self.remove_punctuation = config.get('remove_punctuation', True)
        self.remove_numbers = config.get('remove_numbers', False)
        self.handle_emojis = config.get('handle_emojis', True)
        self.min_word_length = config.get('min_word_length', 2)
        self.max_word_length = config.get('max_word_length', 20)
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add custom stopwords
        custom_stopwords = {'rt', 'via', 'amp', 'gt', 'lt', 'quot', 'apos'}
        self.stop_words.update(custom_stopwords)
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
            for data in nltk_data:
                try:
                    nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'corpora/{data}' if data in ['stopwords', 'wordnet'] else f'taggers/{data}')
                except LookupError:
                    nltk.download(data, quiet=True)
            logger.info("NLTK data downloaded successfully")
        except Exception as e:
            logger.warning(f"Error downloading NLTK data: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = self._remove_html_tags(text)
        
        # Remove URLs
        text = self._remove_urls(text)
        
        # Remove mentions and hashtags (keep the text)
        text = self._clean_mentions_hashtags(text)
        
        # Remove extra whitespace
        text = self._clean_whitespace(text)
        
        return text.strip()
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        try:
            return BeautifulSoup(text, 'html.parser').get_text()
        except:
            return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def _clean_mentions_hashtags(self, text: str) -> str:
        """Clean mentions and hashtags, keeping the text content."""
        # Keep the text after @ and # but remove the symbols
        text = re.sub(r'@(\w+)', r'\1', text)  # @username -> username
        text = re.sub(r'#(\w+)', r'\1', text)  # #hashtag -> hashtag
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
    
    def handle_emojis_text(self, text: str) -> str:
        """
        Handle emojis in text based on configuration.
        
        Args:
            text: Input text
            
        Returns:
            Text with emojis handled
        """
        if not self.handle_emojis:
            return text
        
        try:
            # Convert emojis to text descriptions
            text = emoji.demojize(text, delimiters=(' ', ' '))
            
            # Remove the colons from emoji descriptions
            text = re.sub(r':([a-z_]+):', r'\1', text)
            
        except Exception as e:
            logger.warning(f"Error handling emojis: {e}")
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text (lowercase, expand contractions, etc.).
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if self.lowercase:
            text = text.lower()
        
        # Expand common contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception as e:
            logger.warning(f"Error tokenizing text: {e}")
            return text.split()
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens based on various criteria.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered list of tokens
        """
        filtered_tokens = []
        
        for token in tokens:
            # Skip empty tokens
            if not token.strip():
                continue
            
            # Length filtering
            if len(token) < self.min_word_length or len(token) > self.max_word_length:
                continue
            
            # Remove punctuation
            if self.remove_punctuation and token in string.punctuation:
                continue
            
            # Remove numbers
            if self.remove_numbers and token.isdigit():
                continue
            
            # Remove stopwords
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception as e:
            logger.warning(f"Error lemmatizing tokens: {e}")
            return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Step 1: Basic cleaning
        text = self.clean_text(text)
        
        # Step 2: Handle emojis
        text = self.handle_emojis_text(text)
        
        # Step 3: Normalize
        text = self.normalize_text(text)
        
        # Step 4: Tokenize
        tokens = self.tokenize(text)
        
        # Step 5: Filter tokens
        tokens = self.filter_tokens(tokens)
        
        # Step 6: Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Step 7: Rejoin tokens
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        processed_texts = []
        
        for i, text in enumerate(texts):
            try:
                processed = self.preprocess_text(text)
                processed_texts.append(processed)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                logger.warning(f"Error processing text {i}: {e}")
                processed_texts.append("")
        
        logger.info(f"Batch preprocessing completed: {len(processed_texts)} texts")
        
        return processed_texts
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Preprocess text column in a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column to preprocess
            
        Returns:
            DataFrame with preprocessed text
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        logger.info(f"Preprocessing text column '{text_column}' in DataFrame with {len(df)} rows")
        
        # Create a copy to avoid SettingWithCopyWarning
        df_processed = df.copy()
        
        # Apply preprocessing
        df_processed[f'{text_column}_processed'] = df_processed[text_column].apply(self.preprocess_text)
        
        # Add metadata
        df_processed['original_length'] = df_processed[text_column].str.len()
        df_processed['processed_length'] = df_processed[f'{text_column}_processed'].str.len()
        df_processed['length_reduction'] = df_processed['original_length'] - df_processed['processed_length']
        
        logger.info("DataFrame preprocessing completed")
        
        return df_processed
    
    def get_text_statistics(self, texts: List[str]) -> Dict:
        """
        Get statistics about the text data.
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary with text statistics
        """
        if not texts:
            return {}
        
        original_lengths = [len(text) for text in texts if isinstance(text, str)]
        processed_texts = [self.preprocess_text(text) for text in texts if isinstance(text, str)]
        processed_lengths = [len(text) for text in processed_texts]
        
        stats = {
            'total_texts': len(texts),
            'avg_original_length': sum(original_lengths) / len(original_lengths) if original_lengths else 0,
            'avg_processed_length': sum(processed_lengths) / len(processed_lengths) if processed_lengths else 0,
            'total_tokens': sum(len(text.split()) for text in processed_texts),
            'unique_tokens': len(set(' '.join(processed_texts).split())),
            'texts_with_urls': sum(1 for text in texts if 'http' in text),
            'texts_with_mentions': sum(1 for text in texts if '@' in text),
            'texts_with_hashtags': sum(1 for text in texts if '#' in text),
            'texts_with_emojis': sum(1 for text in texts if emoji.emoji_count(text) > 0)
        }
        
        return stats
    
    def extract_features(self, text: str) -> Dict:
        """
        Extract additional features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text features
        """
        if not isinstance(text, str):
            return {}
        
        features = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0,
            'punctuation_count': sum(1 for char in text if char in string.punctuation),
            'number_count': sum(1 for word in text.split() if word.isdigit()),
            'url_count': len(re.findall(r'http[s]?://\S+', text)),
            'mention_count': len(re.findall(r'@\w+', text)),
            'hashtag_count': len(re.findall(r'#\w+', text)),
            'emoji_count': emoji.emoji_count(text),
            'uppercase_count': sum(1 for char in text if char.isupper()),
            'lowercase_count': sum(1 for char in text if char.islower())
        }
        
        return features
