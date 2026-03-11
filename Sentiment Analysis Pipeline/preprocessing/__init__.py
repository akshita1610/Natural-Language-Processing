"""
Preprocessing Module for Sentiment Analysis Pipeline

This module handles text cleaning and preprocessing:
- Tokenization and normalization
- Stopword removal
- Emoji handling
- Text cleaning utilities
"""

from .text_preprocessor import TextPreprocessor

__all__ = ['TextPreprocessor']
