"""
Sentiment Analysis Module for Sentiment Analysis Pipeline

This module handles sentiment analysis using various models:
- VADER sentiment analysis
- TextBlob sentiment analysis
- Transformer-based models
"""

from .sentiment_analyzer import SentimentAnalyzer

__all__ = ['SentimentAnalyzer']
