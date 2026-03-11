"""
Reddit Data Ingestion Module

Handles data collection from Reddit API using PRAW for sentiment analysis.
Supports both batch collection and real-time monitoring.
"""

import praw
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class RedditIngestion:
    """
    Reddit data ingestion class for collecting posts and comments.
    
    Supports:
    - Batch collection from subreddits
    - Real-time monitoring of new posts
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Reddit ingestion with configuration.
        
        Args:
            config: Reddit configuration dictionary
        """
        self.config = config
        self.reddit = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Reddit API client."""
        try:
            self.reddit = praw.Reddit(
                client_id=self.config.get('client_id'),
                client_secret=self.config.get('client_secret'),
                user_agent=self.config.get('user_agent', 'Sentiment Analysis Pipeline v1.0'),
                read_only=True
            )
            
            # Test connection
            self.reddit.user.me()
            logger.info("Reddit API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            raise
    
    def collect_posts(self, subreddit: str = None, limit: int = None, sort: str = None) -> List[Dict]:
        """
        Collect posts from subreddit(s).
        
        Args:
            subreddit: Subreddit name (overrides config)
            limit: Number of posts to collect (overrides config)
            sort: Sort method (hot, new, top, rising)
            
        Returns:
            List of post dictionaries
        """
        posts = []
        try:
            subreddit = subreddit or self.config.get('subreddit', 'all')
            limit = limit or self.config.get('limit', 100)
            sort = sort or self.config.get('sort', 'hot')
            
            logger.info(f"Collecting {limit} posts from r/{subreddit} sorted by {sort}")
            
            # Get subreddit object
            if subreddit == 'all':
                subreddit_obj = self.reddit.subreddit('all')
            else:
                subreddit_obj = self.reddit.subreddit(subreddit)
            
            # Get posts based on sort method
            if sort == 'hot':
                submissions = subreddit_obj.hot(limit=limit)
            elif sort == 'new':
                submissions = subreddit_obj.new(limit=limit)
            elif sort == 'top':
                submissions = subreddit_obj.top(limit=limit)
            elif sort == 'rising':
                submissions = subreddit_obj.rising(limit=limit)
            else:
                submissions = subreddit_obj.hot(limit=limit)
            
            for submission in submissions:
                post_data = {
                    'id': submission.id,
                    'title': submission.title,
                    'text': submission.selftext,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'created_at': datetime.fromtimestamp(submission.created_utc),
                    'author': str(submission.author) if submission.author else '[deleted]',
                    'subreddit': str(submission.subreddit),
                    'url': submission.url,
                    'upvote_ratio': submission.upvote_ratio,
                    'source': 'reddit_post',
                    'combined_text': f"{submission.title} {submission.selftext}"
                }
                posts.append(post_data)
            
            logger.info(f"Collected {len(posts)} posts from r/{subreddit}")
            
        except Exception as e:
            logger.error(f"Error collecting Reddit posts: {e}")
            raise
        
        return posts
    
    def collect_comments(self, subreddit: str = None, limit: int = None, sort: str = 'hot') -> List[Dict]:
        """
        Collect comments from subreddit posts.
        
        Args:
            subreddit: Subreddit name
            limit: Number of posts to get comments from
            sort: Sort method for posts
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        try:
            # First collect posts
            posts = self.collect_posts(subreddit, limit, sort)
            
            logger.info(f"Collecting comments from {len(posts)} posts")
            
            for post in posts:
                try:
                    submission = self.reddit.submission(id=post['id'])
                    submission.comments.replace_more(limit=0)  # Remove "more comments" placeholders
                    
                    for comment in submission.comments.list():
                        if hasattr(comment, 'body'):  # Skip MoreComments objects
                            comment_data = {
                                'id': comment.id,
                                'text': comment.body,
                                'score': comment.score,
                                'created_at': datetime.fromtimestamp(comment.created_utc),
                                'author': str(comment.author) if comment.author else '[deleted]',
                                'subreddit': str(comment.subreddit),
                                'post_id': post['id'],
                                'post_title': post['title'],
                                'source': 'reddit_comment',
                                'parent_id': comment.parent_id
                            }
                            comments.append(comment_data)
                
                except Exception as e:
                    logger.warning(f"Error collecting comments for post {post['id']}: {e}")
                    continue
            
            logger.info(f"Collected {len(comments)} comments")
            
        except Exception as e:
            logger.error(f"Error collecting Reddit comments: {e}")
            raise
        
        return comments
    
    def collect_combined(self, subreddit: str = None, limit: int = None, sort: str = None) -> List[Dict]:
        """
        Collect both posts and comments, combining them into a single dataset.
        
        Args:
            subreddit: Subreddit name
            limit: Number of posts to collect
            sort: Sort method for posts
            
        Returns:
            List of dictionaries containing posts and comments
        """
        posts = self.collect_posts(subreddit, limit, sort)
        comments = self.collect_comments(subreddit, limit, sort)
        
        # Combine posts and comments
        combined = posts + comments
        logger.info(f"Combined dataset: {len(posts)} posts + {len(comments)} comments = {len(combined)} items")
        
        return combined
    
    def monitor_new_posts(self, subreddit: str = None, check_interval: int = 60) -> Iterator[Dict]:
        """
        Monitor new posts in real-time.
        
        Args:
            subreddit: Subreddit to monitor
            check_interval: Seconds between checks
            
        Yields:
            New post dictionaries as they appear
        """
        subreddit = subreddit or self.config.get('subreddit', 'all')
        subreddit_obj = self.reddit.subreddit(subreddit)
        
        logger.info(f"Starting real-time monitoring of r/{subreddit}")
        
        # Track seen posts
        seen_posts = set()
        
        try:
            while True:
                for submission in subreddit_obj.new(limit=10):
                    if submission.id not in seen_posts:
                        seen_posts.add(submission.id)
                        
                        post_data = {
                            'id': submission.id,
                            'title': submission.title,
                            'text': submission.selftext,
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'created_at': datetime.fromtimestamp(submission.created_utc),
                            'author': str(submission.author) if submission.author else '[deleted]',
                            'subreddit': str(submission.subreddit),
                            'url': submission.url,
                            'source': 'reddit_stream',
                            'combined_text': f"{submission.title} {submission.selftext}"
                        }
                        
                        yield post_data
                
                # Wait before next check
                import time
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("Stopping Reddit monitoring")
        except Exception as e:
            logger.error(f"Error in Reddit monitoring: {e}")
            raise
    
    def search_by_keyword(self, keyword: str, subreddit: str = None, limit: int = 50) -> List[Dict]:
        """
        Search for posts containing specific keywords.
        
        Args:
            keyword: Search keyword
            subreddit: Subreddit to search in
            limit: Maximum number of results
            
        Returns:
            List of matching post dictionaries
        """
        results = []
        try:
            subreddit = subreddit or self.config.get('subreddit', 'all')
            subreddit_obj = self.reddit.subreddit(subreddit)
            
            logger.info(f"Searching for '{keyword}' in r/{subreddit}")
            
            for submission in subreddit_obj.search(keyword, limit=limit):
                post_data = {
                    'id': submission.id,
                    'title': submission.title,
                    'text': submission.selftext,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'created_at': datetime.fromtimestamp(submission.created_utc),
                    'author': str(submission.author) if submission.author else '[deleted]',
                    'subreddit': str(submission.subreddit),
                    'url': submission.url,
                    'source': 'reddit_search',
                    'search_keyword': keyword,
                    'combined_text': f"{submission.title} {submission.selftext}"
                }
                results.append(post_data)
            
            logger.info(f"Found {len(results)} posts matching '{keyword}'")
            
        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
            raise
        
        return results
    
    def to_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """
        Convert list of Reddit dictionaries to pandas DataFrame.
        
        Args:
            data: List of post/comment dictionaries
            
        Returns:
            Pandas DataFrame with Reddit data
        """
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Ensure created_at is datetime
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        return df
