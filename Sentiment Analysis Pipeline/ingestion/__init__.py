"""
Data Ingestion Module for Sentiment Analysis Pipeline

This module handles data collection from various sources:
- Twitter/X API
- Reddit API (PRAW)
- Audio → text transcripts (using Whisper or any ASR tool)
- YouTube API (video comments and metadata)
"""

from .twitter_ingestion import TwitterIngestion
from .reddit_ingestion import RedditIngestion
from .youtube_ingestion import YouTubeIngestion

# Audio ingestion requires special setup - import conditionally
try:
    from .audio_ingestion import AudioIngestion
    AUDIO_AVAILABLE = True
except ImportError:
    AudioIngestion = None
    AUDIO_AVAILABLE = False

__all__ = ['TwitterIngestion', 'RedditIngestion', 'AudioIngestion', 'YouTubeIngestion']
