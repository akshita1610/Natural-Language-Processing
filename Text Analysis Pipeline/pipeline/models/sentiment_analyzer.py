"""
Sentiment Analysis Module
Implements VADER sentiment analysis and optional transformer-based models
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re

# Optional transformer imports
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Only VADER/TextBlob will be used.")

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis with multiple methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize sentiment analyzer with configuration
        
        Args:
            config: Configuration dictionary containing sentiment analysis settings
        """
        self.config = config.get('models', {}).get('sentiment', {})
        self.vader_analyzer = None
        self.transformer_pipeline = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        # Initialize VADER
        if self.config.get('vader', True):
            self.vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        
        # Initialize transformer model if enabled and available
        if (self.config.get('transformer', {}).get('enabled', False) and 
            TRANSFORMERS_AVAILABLE):
            try:
                model_name = self.config.get('transformer', {}).get(
                    'model_name', 'distilbert-base-uncased-finetuned-sst-2-english'
                )
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name
                )
                logger.info(f"Transformer model '{model_name}' loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load transformer model: {str(e)}")
                self.transformer_pipeline = None
    
    def analyze_sentiment_vader(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment using VADER
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with VADER sentiment scores
        """
        if self.vader_analyzer is None:
            raise ValueError("VADER analyzer not initialized")
        
        results = []
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                # Handle empty/invalid texts
                results.append({
                    'text': text,
                    'vader_positive': 0.0,
                    'vader_negative': 0.0,
                    'vader_neutral': 1.0,
                    'vader_compound': 0.0,
                    'vader_sentiment': 'neutral'
                })
                continue
            
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine sentiment label based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            results.append({
                'text': text,
                'vader_positive': scores['pos'],
                'vader_negative': scores['neg'],
                'vader_neutral': scores['neu'],
                'vader_compound': compound,
                'vader_sentiment': sentiment
            })
        
        return pd.DataFrame(results)
    
    def analyze_sentiment_textblob(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment using TextBlob
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with TextBlob sentiment scores
        """
        results = []
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                results.append({
                    'text': text,
                    'textblob_polarity': 0.0,
                    'textblob_subjectivity': 0.0,
                    'textblob_sentiment': 'neutral'
                })
                continue
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine sentiment label
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            results.append({
                'text': text,
                'textblob_polarity': polarity,
                'textblob_subjectivity': subjectivity,
                'textblob_sentiment': sentiment
            })
        
        return pd.DataFrame(results)
    
    def analyze_sentiment_transformer(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment using transformer model
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with transformer sentiment predictions
        """
        if self.transformer_pipeline is None:
            raise ValueError("Transformer pipeline not initialized")
        
        results = []
        
        # Process texts in batches to handle memory constraints
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Filter out empty texts
            valid_texts = [text for text in batch_texts if isinstance(text, str) and text.strip()]
            
            if not valid_texts:
                # Handle batch with no valid texts
                for text in batch_texts:
                    results.append({
                        'text': text,
                        'transformer_sentiment': 'neutral',
                        'transformer_confidence': 0.0
                    })
                continue
            
            try:
                # Get predictions
                predictions = self.transformer_pipeline(valid_texts)
                
                # Map results back to original texts
                pred_idx = 0
                for text in batch_texts:
                    if isinstance(text, str) and text.strip():
                        pred = predictions[pred_idx]
                        sentiment = pred['label'].lower()
                        confidence = pred['score']
                        
                        # Normalize sentiment labels
                        if sentiment in ['positive', 'pos']:
                            sentiment = 'positive'
                        elif sentiment in ['negative', 'neg']:
                            sentiment = 'negative'
                        else:
                            sentiment = 'neutral'
                        
                        results.append({
                            'text': text,
                            'transformer_sentiment': sentiment,
                            'transformer_confidence': confidence
                        })
                        pred_idx += 1
                    else:
                        results.append({
                            'text': text,
                            'transformer_sentiment': 'neutral',
                            'transformer_confidence': 0.0
                        })
                        
            except Exception as e:
                logger.warning(f"Error in transformer batch processing: {str(e)}")
                # Fallback to neutral for this batch
                for text in batch_texts:
                    results.append({
                        'text': text,
                        'transformer_sentiment': 'neutral',
                        'transformer_confidence': 0.0
                    })
        
        return pd.DataFrame(results)
    
    def analyze_sentiment_ensemble(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment using ensemble of available methods
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with ensemble sentiment predictions
        """
        all_results = []
        
        # Collect results from all available methods
        if self.vader_analyzer is not None:
            vader_results = self.analyze_sentiment_vader(texts)
            all_results.append(vader_results)
        
        # TextBlob (always available)
        textblob_results = self.analyze_sentiment_textblob(texts)
        all_results.append(textblob_results)
        
        if self.transformer_pipeline is not None:
            transformer_results = self.analyze_sentiment_transformer(texts)
            all_results.append(transformer_results)
        
        # Merge all results
        merged_results = all_results[0].copy()
        
        for result_df in all_results[1:]:
            for col in result_df.columns:
                if col != 'text' and col not in merged_results.columns:
                    merged_results[col] = result_df[col]
        
        # Create ensemble prediction
        sentiment_columns = [col for col in merged_results.columns if col.endswith('_sentiment')]
        
        def get_ensemble_sentiment(row):
            sentiments = [row[col] for col in sentiment_columns if pd.notna(row[col])]
            if not sentiments:
                return 'neutral'
            
            # Count sentiment votes
            sentiment_counts = {}
            for sentiment in sentiments:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            # Return the sentiment with most votes
            return max(sentiment_counts, key=sentiment_counts.get)
        
        merged_results['ensemble_sentiment'] = merged_results.apply(get_ensemble_sentiment, axis=1)
        
        return merged_results
    
    def get_sentiment_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for sentiment analysis results
        
        Args:
            results_df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        # Analyze each sentiment column
        sentiment_columns = [col for col in results_df.columns if col.endswith('_sentiment')]
        
        for col in sentiment_columns:
            sentiment_counts = results_df[col].value_counts()
            sentiment_percentages = (sentiment_counts / len(results_df) * 100).round(2)
            
            summary[col] = {
                'counts': sentiment_counts.to_dict(),
                'percentages': sentiment_percentages.to_dict()
            }
        
        # Add statistics for numeric columns
        numeric_columns = [col for col in results_df.columns if any(x in col for x in ['compound', 'polarity', 'confidence'])]
        
        for col in numeric_columns:
            if col in results_df.columns:
                summary[col] = {
                    'mean': float(results_df[col].mean()),
                    'std': float(results_df[col].std()),
                    'min': float(results_df[col].min()),
                    'max': float(results_df[col].max())
                }
        
        return summary
    
    def analyze_emotions(self, texts: List[str]) -> pd.DataFrame:
        """
        Basic emotion analysis using keyword-based approach
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            DataFrame with emotion scores
        """
        # Simple emotion keyword dictionaries
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'delighted', 'pleased', 'glad', 'cheerful'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated'],
            'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'disappointed'],
            'fear': ['afraid', 'scared', 'frightened', 'terrified', 'worried', 'anxious'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned']
        }
        
        results = []
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                results.append({
                    'text': text,
                    **{emotion: 0.0 for emotion in emotion_keywords.keys()},
                    'dominant_emotion': 'neutral'
                })
                continue
            
            text_lower = text.lower()
            emotion_scores = {}
            
            for emotion, keywords in emotion_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                emotion_scores[emotion] = score / len(keywords)  # Normalize by number of keywords
            
            # Find dominant emotion
            if max(emotion_scores.values()) > 0:
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            else:
                dominant_emotion = 'neutral'
            
            results.append({
                'text': text,
                **emotion_scores,
                'dominant_emotion': dominant_emotion
            })
        
        return pd.DataFrame(results)
