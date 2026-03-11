"""
Sentiment Analysis Module

Handles sentiment analysis using various models and approaches.
Supports VADER, TextBlob, and transformer-based sentiment analysis.
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import logging
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analysis class supporting multiple models.
    
    Features:
    - VADER sentiment analysis (rule-based)
    - TextBlob sentiment analysis (pattern-based)
    - Transformer models (deep learning)
    - Batch processing
    - Confidence scoring
    """
    
    def __init__(self, config: Dict):
        """
        Initialize sentiment analyzer with configuration.
        
        Args:
            config: Sentiment analysis configuration dictionary
        """
        self.config = config
        self.model_type = config.get('model', 'vader')
        self.threshold_positive = config.get('threshold_positive', 0.05)
        self.threshold_negative = config.get('threshold_negative', -0.05)
        self.transformer_model_name = config.get('transformer_model', 'cardiffnlp/twitter-roberta-base-sentiment')
        
        # Initialize models
        self.vader_analyzer = None
        self.transformer_model = None
        self.transformer_tokenizer = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize sentiment analysis models."""
        try:
            if self.model_type in ['vader', 'all']:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                logger.info("VADER sentiment analyzer initialized")
            
            if self.model_type in ['transformer', 'all']:
                try:
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    from transformers import pipeline
                    
                    logger.info(f"Loading transformer model: {self.transformer_model_name}")
                    
                    self.transformer_tokenizer = AutoTokenizer.from_pretrained(self.transformer_model_name)
                    self.transformer_model = AutoModelForSequenceClassification.from_pretrained(self.transformer_model_name)
                    
                    self.transformer_pipeline = pipeline(
                        "sentiment-analysis",
                        model=self.transformer_model,
                        tokenizer=self.transformer_tokenizer,
                        return_all_scores=True
                    )
                    
                    logger.info("Transformer model loaded successfully")
                    
                except ImportError:
                    logger.warning("Transformers library not installed. Transformer models unavailable.")
                except Exception as e:
                    logger.warning(f"Failed to load transformer model: {e}")
            
            # TextBlob doesn't require initialization
            if self.model_type in ['textblob', 'all']:
                logger.info("TextBlob sentiment analyzer ready")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
            raise
    
    def analyze_vader(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with VADER sentiment scores
        """
        if not self.vader_analyzer:
            raise ValueError("VADER analyzer not initialized")
        
        if not isinstance(text, str) or not text.strip():
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neg': 0.0,
                'neu': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0
            }
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine sentiment based on compound score
            compound = scores['compound']
            if compound >= self.threshold_positive:
                sentiment = 'positive'
            elif compound <= self.threshold_negative:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Calculate confidence based on the dominant score
            confidence = max(scores['pos'], scores['neg'], scores['neu'])
            
            result = {
                'compound': compound,
                'pos': scores['pos'],
                'neg': scores['neg'],
                'neu': scores['neu'],
                'sentiment': sentiment,
                'confidence': confidence,
                'model': 'vader'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in VADER analysis: {e}")
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neg': 0.0,
                'neu': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'model': 'vader',
                'error': str(e)
            }
    
    def analyze_textblob(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with TextBlob sentiment scores
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'model': 'textblob'
            }
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine sentiment based on polarity
            if polarity >= self.threshold_positive:
                sentiment = 'positive'
            elif polarity <= self.threshold_negative:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Calculate confidence based on absolute polarity and subjectivity
            confidence = abs(polarity) * (1 - subjectivity * 0.5)
            
            result = {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment,
                'confidence': confidence,
                'model': 'textblob'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'model': 'textblob',
                'error': str(e)
            }
    
    def analyze_transformer(self, text: str) -> Dict:
        """
        Analyze sentiment using transformer model.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with transformer sentiment scores
        """
        if not self.transformer_pipeline:
            raise ValueError("Transformer model not initialized")
        
        if not isinstance(text, str) or not text.strip():
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'model': 'transformer'
            }
        
        try:
            # Truncate text if too long (most models have max length limits)
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            results = self.transformer_pipeline(text)
            
            # Extract scores
            scores = {result['label'].lower(): result['score'] for result in results[0]}
            
            # Find the dominant sentiment
            dominant_sentiment = max(scores.keys(), key=lambda k: scores[k])
            confidence = scores[dominant_sentiment]
            
            result = {
                'positive': scores.get('positive', 0.0),
                'negative': scores.get('negative', 0.0),
                'neutral': scores.get('neutral', 0.0),
                'sentiment': dominant_sentiment,
                'confidence': confidence,
                'model': 'transformer'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in transformer analysis: {e}")
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'model': 'transformer',
                'error': str(e)
            }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using the configured model.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if self.model_type == 'vader':
            return self.analyze_vader(text)
        elif self.model_type == 'textblob':
            return self.analyze_textblob(text)
        elif self.model_type == 'transformer':
            return self.analyze_transformer(text)
        elif self.model_type == 'all':
            # Return results from all available models
            results = {}
            if self.vader_analyzer:
                results['vader'] = self.analyze_vader(text)
            
            results['textblob'] = self.analyze_textblob(text)
            
            if self.transformer_pipeline:
                results['transformer'] = self.analyze_transformer(text)
            
            # Add ensemble result
            results['ensemble'] = self._ensemble_sentiment(results)
            
            return results
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _ensemble_sentiment(self, results: Dict) -> Dict:
        """
        Create ensemble sentiment from multiple models.
        
        Args:
            results: Dictionary of results from different models
            
        Returns:
            Ensemble sentiment result
        """
        try:
            sentiments = []
            confidences = []
            
            for model_name, result in results.items():
                if model_name == 'ensemble':
                    continue
                
                if 'sentiment' in result and 'confidence' in result:
                    sentiments.append(result['sentiment'])
                    confidences.append(result['confidence'])
            
            if not sentiments:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'model': 'ensemble'
                }
            
            # Weighted voting based on confidence
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            for sentiment, confidence in zip(sentiments, confidences):
                sentiment_counts[sentiment] += confidence
            
            dominant_sentiment = max(sentiment_counts.keys(), key=lambda k: sentiment_counts[k])
            confidence = sentiment_counts[dominant_sentiment] / sum(confidences) if confidences else 0.0
            
            return {
                'sentiment': dominant_sentiment,
                'confidence': confidence,
                'model': 'ensemble',
                'votes': sentiment_counts
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble sentiment: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'model': 'ensemble',
                'error': str(e)
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.analyze_sentiment(text)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Analyzed {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                logger.warning(f"Error analyzing text {i}: {e}")
                results.append({
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'model': self.model_type,
                    'error': str(e)
                })
        
        logger.info(f"Batch sentiment analysis completed: {len(results)} texts")
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Analyze sentiment for text column in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column to analyze
            
        Returns:
            DataFrame with sentiment analysis results
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        logger.info(f"Analyzing sentiment for column '{text_column}' in DataFrame with {len(df)} rows")
        
        # Create a copy to avoid SettingWithCopyWarning
        df_analyzed = df.copy()
        
        # Apply sentiment analysis
        results = self.analyze_batch(df_analyzed[text_column].tolist())
        
        # Add results to DataFrame
        if self.model_type == 'all':
            # Handle multiple models
            for model in ['vader', 'textblob', 'transformer', 'ensemble']:
                if model in results[0]:
                    df_analyzed[f'sentiment_{model}'] = [r[model]['sentiment'] for r in results]
                    df_analyzed[f'confidence_{model}'] = [r[model]['confidence'] for r in results]
                    
                    # Add model-specific scores
                    if model == 'vader':
                        df_analyzed['compound'] = [r[model].get('compound', 0.0) for r in results]
                    elif model == 'textblob':
                        df_analyzed['polarity'] = [r[model].get('polarity', 0.0) for r in results]
                        df_analyzed['subjectivity'] = [r[model].get('subjectivity', 0.0) for r in results]
        else:
            # Single model
            df_analyzed['sentiment'] = [r['sentiment'] for r in results]
            df_analyzed['confidence'] = [r['confidence'] for r in results]
            
            # Add model-specific scores
            if self.model_type == 'vader':
                df_analyzed['compound'] = [r.get('compound', 0.0) for r in results]
                df_analyzed['pos'] = [r.get('pos', 0.0) for r in results]
                df_analyzed['neg'] = [r.get('neg', 0.0) for r in results]
                df_analyzed['neu'] = [r.get('neu', 0.0) for r in results]
            elif self.model_type == 'textblob':
                df_analyzed['polarity'] = [r.get('polarity', 0.0) for r in results]
                df_analyzed['subjectivity'] = [r.get('subjectivity', 0.0) for r in results]
            elif self.model_type == 'transformer':
                df_analyzed['positive_score'] = [r.get('positive', 0.0) for r in results]
                df_analyzed['negative_score'] = [r.get('negative', 0.0) for r in results]
                df_analyzed['neutral_score'] = [r.get('neutral', 0.0) for r in results]
        
        # Add analysis timestamp
        df_analyzed['analyzed_at'] = datetime.now()
        
        logger.info("DataFrame sentiment analysis completed")
        
        return df_analyzed
    
    def get_sentiment_statistics(self, results: List[Dict]) -> Dict:
        """
        Get statistics about sentiment analysis results.
        
        Args:
            results: List of sentiment analysis results
            
        Returns:
            Dictionary with sentiment statistics
        """
        if not results:
            return {}
        
        # Count sentiments
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        confidences = []
        
        for result in results:
            sentiment = result.get('sentiment', 'neutral')
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            
            confidence = result.get('confidence', 0.0)
            confidences.append(confidence)
        
        # Calculate statistics
        total = len(results)
        stats = {
            'total_analyzed': total,
            'sentiment_distribution': {k: v for k, v in sentiment_counts.items()},
            'sentiment_percentages': {k: (v / total) * 100 for k, v in sentiment_counts.items()},
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'min_confidence': min(confidences) if confidences else 0.0,
            'max_confidence': max(confidences) if confidences else 0.0
        }
        
        # Add model-specific statistics
        if self.model_type == 'vader' and results:
            compounds = [r.get('compound', 0.0) for r in results]
            stats['average_compound'] = sum(compounds) / len(compounds)
            stats['sentiment_range'] = {'min': min(compounds), 'max': max(compounds)}
        
        return stats
    
    def compare_models(self, texts: List[str]) -> Dict:
        """
        Compare sentiment analysis results across different models.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary with model comparison results
        """
        if self.model_type != 'all':
            logger.warning("Model comparison requires 'all' model type")
            return {}
        
        # Store original model type
        original_model = self.model_type
        
        comparison_results = {}
        
        # Test each available model
        for model_name in ['vader', 'textblob', 'transformer']:
            if model_name == 'transformer' and not self.transformer_pipeline:
                continue
            
            self.model_type = model_name
            results = self.analyze_batch(texts)
            comparison_results[model_name] = self.get_sentiment_statistics(results)
        
        # Restore original model type
        self.model_type = original_model
        
        return comparison_results
