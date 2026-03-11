"""
NLP Preprocessing Pipeline
Handles text cleaning, tokenization, and preprocessing for sentiment analysis
"""

import re
import string
import emoji
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from typing import List, Dict, Union
import logging
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for Instagram data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the preprocessor with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocess_config = config["preprocessing"]
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words(self.preprocess_config["language"]))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Custom Instagram-specific patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        if self.preprocess_config["remove_urls"]:
            text = self.url_pattern.sub('', text)
        
        # Remove mentions
        if self.preprocess_config["remove_mentions"]:
            text = self.mention_pattern.sub('', text)
        
        # Convert emojis to text if enabled
        if self.preprocess_config["convert_emojis"]:
            text = self._convert_emojis_to_text(text)
        else:
            # Remove emojis
            text = self.emoji_pattern.sub('', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation (except for emoji text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers (optional - you might want to keep them)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _convert_emojis_to_text(self, text: str) -> str:
        """
        Convert emojis to their text descriptions
        
        Args:
            text: Input text with emojis
            
        Returns:
            Text with emojis converted to descriptions
        """
        # Convert emojis to text
        text = emoji.demojize(text)
        
        # Replace colons and underscores with spaces
        text = text.replace(':', ' ').replace('_', ' ')
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Use NLTK tokenizer
        tokens = word_tokenize(text)
        
        # Filter tokens
        min_length = self.preprocess_config.get("min_word_length", 2)
        tokens = [token for token in tokens 
                 if len(token) >= min_length 
                 and token.isalpha()]
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens without stopwords
        """
        if not self.preprocess_config["remove_stopwords"]:
            return tokens
        
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        if not self.preprocess_config["lemmatize"]:
            return tokens
        
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def extract_hashtags(self, text: str) -> List[str]:
        """
        Extract hashtags from text
        
        Args:
            text: Input text string
            
        Returns:
            List of hashtags (without # symbol)
        """
        hashtags = self.hashtag_pattern.findall(text)
        return [tag.replace('#', '') for tag in hashtags]
    
    def extract_mentions(self, text: str) -> List[str]:
        """
        Extract mentions from text
        
        Args:
            text: Input text string
            
        Returns:
            List of mentions (without @ symbol)
        """
        mentions = self.mention_pattern.findall(text)
        return [mention.replace('@', '') for mention in mentions]
    
    def preprocess_text(self, text: str) -> Dict[str, Union[str, List[str]]]:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with original and processed text data
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Extract features before cleaning
        hashtags = self.extract_hashtags(text)
        mentions = self.extract_mentions(text)
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back to string
        processed_text = ' '.join(tokens)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'processed_text': processed_text,
            'tokens': tokens,
            'hashtags': hashtags,
            'mentions': mentions,
            'token_count': len(tokens),
            'hashtag_count': len(hashtags),
            'mention_count': len(mentions)
        }
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'caption') -> pd.DataFrame:
        """
        Preprocess text column in a DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column to preprocess
            
        Returns:
            DataFrame with preprocessed text
        """
        self.logger.info(f"Preprocessing {len(df)} texts...")
        
        # Apply preprocessing to each text
        processed_data = df[text_column].apply(self.preprocess_text)
        
        # Extract processed data into separate columns
        df['cleaned_text'] = processed_data.apply(lambda x: x['cleaned_text'])
        df['processed_text'] = processed_data.apply(lambda x: x['processed_text'])
        df['tokens'] = processed_data.apply(lambda x: x['tokens'])
        df['token_count'] = processed_data.apply(lambda x: x['token_count'])
        
        # Handle hashtags and mentions if they don't exist
        if 'hashtags' not in df.columns:
            df['hashtags'] = processed_data.apply(lambda x: x['hashtags'])
        if 'mentions' not in df.columns:
            df['mentions'] = processed_data.apply(lambda x: x['mentions'])
        
        self.logger.info("Preprocessing completed")
        return df
    
    def preprocess_comments(self, comments_data: List[Dict]) -> List[Dict]:
        """
        Preprocess comments from posts
        
        Args:
            comments_data: List of comment dictionaries
            
        Returns:
            List of preprocessed comment dictionaries
        """
        processed_comments = []
        
        for comment in comments_data:
            if 'text' in comment:
                processed = self.preprocess_text(comment['text'])
                comment.update(processed)
                processed_comments.append(comment)
        
        return processed_comments
    
    def get_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from processed texts
        
        Args:
            texts: List of processed text strings
            
        Returns:
            Dictionary mapping words to their frequency
        """
        from collections import Counter
        
        all_tokens = []
        for text in texts:
            tokens = self.tokenize_text(self.clean_text(text))
            tokens = self.remove_stopwords(tokens)
            tokens = self.lemmatize_tokens(tokens)
            all_tokens.extend(tokens)
        
        vocabulary = Counter(all_tokens)
        return dict(vocabulary)
    
    def filter_vocabulary(self, vocabulary: Dict[str, int], 
                         min_freq: int = 5, 
                         max_freq_ratio: float = 0.95) -> Dict[str, int]:
        """
        Filter vocabulary by frequency
        
        Args:
            vocabulary: Word frequency dictionary
            min_freq: Minimum frequency threshold
            max_freq_ratio: Maximum frequency ratio (for removing too common words)
            
        Returns:
            Filtered vocabulary
        """
        total_words = sum(vocabulary.values())
        max_freq = int(total_words * max_freq_ratio)
        
        filtered_vocab = {
            word: freq for word, freq in vocabulary.items()
            if min_freq <= freq <= max_freq
        }
        
        return filtered_vocab

def main():
    """
    Example usage of TextPreprocessor
    """
    import json
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(config)
    
    # Example text
    example_text = "Amazing sunset! 🌅 Feeling so blessed and grateful for this beautiful day! #sunset #blessed @nature"
    
    # Preprocess text
    result = preprocessor.preprocess_text(example_text)
    
    print("Original Text:", result['original_text'])
    print("Cleaned Text:", result['cleaned_text'])
    print("Processed Text:", result['processed_text'])
    print("Tokens:", result['tokens'])
    print("Hashtags:", result['hashtags'])
    print("Mentions:", result['mentions'])

if __name__ == "__main__":
    main()
