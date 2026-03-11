"""
Twitter/X Data Ingestion Module

Handles data collection from Twitter/X API for sentiment analysis.
Supports both batch and real-time streaming modes.
"""

import tweepy
import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class TwitterIngestion:
    """
    Twitter/X data ingestion class for collecting tweets.
    
    Supports:
    - Batch collection using search API
    - Real-time streaming using streaming API
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Twitter ingestion with configuration.
        
        Args:
            config: Twitter configuration dictionary
        """
        self.config = config
        self.client = None
        self.api = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Twitter API client."""
        try:
            # For Twitter API v2
            self.client = tweepy.Client(
                bearer_token=self.config.get('bearer_token'),
                consumer_key=self.config.get('api_key'),
                consumer_secret=self.config.get('api_secret'),
                access_token=self.config.get('access_token'),
                access_token_secret=self.config.get('access_token_secret'),
                wait_on_rate_limit=True
            )
            
            # For Twitter API v1.1 (if needed)
            auth = tweepy.OAuth1UserHandler(
                self.config.get('api_key'),
                self.config.get('api_secret'),
                self.config.get('access_token'),
                self.config.get('access_token_secret')
            )
            self.api = tweepy.API(auth, wait_on_rate_limit=True)
            
            logger.info("Twitter API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
            raise
    
    def collect_batch_tweets(self) -> List[Dict]:
        """
        Collect tweets in batch mode using search API.
        
        Returns:
            List of tweet dictionaries
        """
        tweets = []
        try:
            query = self.config.get('query', 'sentiment')
            max_tweets = self.config.get('max_tweets', 100)
            language = self.config.get('language', 'en')
            
            logger.info(f"Collecting {max_tweets} tweets for query: {query}")
            
            # Use Twitter API v2 for recent search
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_tweets, 100),  # API limit is 100 per request
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang'],
                lang=language,
                exclude=['retweets']
            )
            
            if response.data:
                for tweet in response.data:
                    tweet_data = {
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'author_id': tweet.author_id,
                        'lang': tweet.lang,
                        'source': 'twitter',
                        'public_metrics': tweet.public_metrics if hasattr(tweet, 'public_metrics') else {}
                    }
                    tweets.append(tweet_data)
            
            logger.info(f"Collected {len(tweets)} tweets")
            
        except Exception as e:
            logger.error(f"Error collecting batch tweets: {e}")
            raise
        
        return tweets
    
    def stream_tweets(self) -> Iterator[Dict]:
        """
        Stream tweets in real-time using streaming API.
        
        Yields:
            Tweet dictionaries one at a time
        """
        class TweetStreamListener(tweepy.StreamingClient):
            def __init__(self, parent_ingestion):
                super().__init__(parent_ingestion.config.get('bearer_token'))
                self.parent = parent_ingestion
                self.tweet_queue = []
            
            def on_tweet(self, tweet):
                tweet_data = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'author_id': tweet.author_id,
                    'source': 'twitter_stream'
                }
                self.tweet_queue.append(tweet_data)
            
            def on_error(self, status_code):
                logger.error(f"Twitter stream error: {status_code}")
                return False
        
        try:
            query = self.config.get('query', 'sentiment')
            listener = TweetStreamListener(self)
            
            # Filter rules for streaming
            rule = tweepy.StreamRule(query)
            
            logger.info(f"Starting Twitter stream for query: {query}")
            
            # Get existing rules and delete them
            rules = listener.get_rules()
            if rules.data:
                listener.delete_rules(rules.data)
            
            # Add new rule
            listener.add_rules(rule)
            
            # Start streaming
            listener.filter(threaded=True)
            
            # Yield tweets as they come in
            while True:
                if listener.tweet_queue:
                    tweet = listener.tweet_queue.pop(0)
                    yield tweet
                    
        except Exception as e:
            logger.error(f"Error in Twitter streaming: {e}")
            raise
    
    def get_user_tweets(self, username: str, count: int = 50) -> List[Dict]:
        """
        Get tweets from a specific user.
        
        Args:
            username: Twitter username
            count: Number of tweets to collect
            
        Returns:
            List of tweet dictionaries
        """
        tweets = []
        try:
            user = self.client.get_user(username=username)
            if user.data:
                response = self.client.get_users_tweets(
                    id=user.data.id,
                    max_results=min(count, 100),
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                if response.data:
                    for tweet in response.data:
                        tweet_data = {
                            'id': tweet.id,
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'author_id': tweet.author_id,
                            'source': 'twitter_user',
                            'username': username,
                            'public_metrics': tweet.public_metrics if hasattr(tweet, 'public_metrics') else {}
                        }
                        tweets.append(tweet_data)
            
            logger.info(f"Collected {len(tweets)} tweets from @{username}")
            
        except Exception as e:
            logger.error(f"Error getting user tweets: {e}")
            raise
        
        return tweets
    
    def tweets_to_dataframe(self, tweets: List[Dict]) -> pd.DataFrame:
        """
        Convert list of tweet dictionaries to pandas DataFrame.
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            Pandas DataFrame with tweet data
        """
        if not tweets:
            return pd.DataFrame()
        
        df = pd.DataFrame(tweets)
        
        # Extract metrics if present
        if 'public_metrics' in df.columns:
            metrics_df = pd.json_normalize(df['public_metrics'])
            df = pd.concat([df.drop('public_metrics', axis=1), metrics_df], axis=1)
        
        # Convert created_at to datetime
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        return df
