"""
Instagram Data Collection Module
Uses Instaloader to scrape Instagram posts, comments, and metadata
"""

import instaloader
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import logging

class InstagramScraper:
    """
    Instagram data scraper using Instaloader
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the scraper with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.L = instaloader.Instaloader(
            download_pictures=config["data_collection"]["download_pictures"],
            download_videos=config["data_collection"]["download_videos"],
            download_geotags=config["data_collection"]["save_metadata"],
            download_comments=True,
            save_metadata=config["data_collection"]["save_metadata"]
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create data directories
        self.data_dir = Path(config["paths"]["data_dir"])
        self.raw_data_dir = Path(config["paths"]["raw_data"])
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
    def login(self, username: str, password: str) -> bool:
        """
        Login to Instagram
        
        Args:
            username: Instagram username
            password: Instagram password
            
        Returns:
            bool: True if login successful
        """
        try:
            self.L.login(username, password)
            self.logger.info(f"Successfully logged in as {username}")
            return True
        except Exception as e:
            self.logger.error(f"Login failed: {e}")
            return False
    
    def get_profile_posts(self, profile_name: str, max_posts: int = None) -> List[Dict]:
        """
        Get posts from a specific profile
        
        Args:
            profile_name: Instagram profile name
            max_posts: Maximum number of posts to scrape
            
        Returns:
            List of dictionaries containing post data
        """
        if max_posts is None:
            max_posts = self.config["data_collection"]["max_posts"]
            
        try:
            profile = instaloader.Profile.from_username(self.L.context, profile_name)
            posts_data = []
            
            self.logger.info(f"Scraping {max_posts} posts from @{profile_name}")
            
            for i, post in enumerate(profile.get_posts()):
                if i >= max_posts:
                    break
                    
                try:
                    post_data = self._extract_post_data(post)
                    posts_data.append(post_data)
                    
                    # Rate limiting
                    time.sleep(1)
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Scraped {i + 1} posts")
                        
                except Exception as e:
                    self.logger.warning(f"Error scraping post {i}: {e}")
                    continue
                    
            self.logger.info(f"Successfully scraped {len(posts_data)} posts")
            return posts_data
            
        except Exception as e:
            self.logger.error(f"Error scraping profile {profile_name}: {e}")
            return []
    
    def _extract_post_data(self, post) -> Dict:
        """
        Extract data from a single post
        
        Args:
            post: Instaloader Post object
            
        Returns:
            Dictionary containing post data
        """
        # Extract hashtags
        hashtags = [tag for tag in post.caption_hashtags]
        
        # Extract mentions
        mentions = [mention for mention in post.caption_mentions]
        
        # Get comments (limited)
        comments_data = []
        max_comments = self.config["data_collection"]["max_comments_per_post"]
        
        try:
            comments = post.get_comments()
            for i, comment in enumerate(comments):
                if i >= max_comments:
                    break
                comments_data.append({
                    'text': comment.text,
                    'owner': comment.owner.username if comment.owner else 'unknown',
                    'likes': comment.likes_count,
                    'timestamp': comment.date.isoformat(),
                    'replies': comment.answers_count
                })
        except Exception as e:
            self.logger.warning(f"Could not fetch comments: {e}")
        
        post_data = {
            'shortcode': post.shortcode,
            'url': post.url,
            'caption': post.caption or '',
            'hashtags': hashtags,
            'mentions': mentions,
            'likes': post.likes,
            'comments_count': post.comments,
            'timestamp': post.date.isoformat(),
            'is_video': post.is_video,
            'location': post.location.name if post.location else None,
            'tagged_users': [user.username for user in post.tagged_users],
            'comments': comments_data
        }
        
        return post_data
    
    def get_hashtag_posts(self, hashtag: str, max_posts: int = None) -> List[Dict]:
        """
        Get posts from a specific hashtag
        
        Args:
            hashtag: Hashtag to search for
            max_posts: Maximum number of posts to scrape
            
        Returns:
            List of dictionaries containing post data
        """
        if max_posts is None:
            max_posts = self.config["data_collection"]["max_posts"]
            
        try:
            posts = instaloader.Hashtag.from_name(self.L.context, hashtag).get_posts()
            posts_data = []
            
            self.logger.info(f"Scraping {max_posts} posts from #{hashtag}")
            
            for i, post in enumerate(posts):
                if i >= max_posts:
                    break
                    
                try:
                    post_data = self._extract_post_data(post)
                    posts_data.append(post_data)
                    time.sleep(1)
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Scraped {i + 1} posts")
                        
                except Exception as e:
                    self.logger.warning(f"Error scraping post {i}: {e}")
                    continue
                    
            self.logger.info(f"Successfully scraped {len(posts_data)} posts")
            return posts_data
            
        except Exception as e:
            self.logger.error(f"Error scraping hashtag {hashtag}: {e}")
            return []
    
    def save_data(self, posts_data: List[Dict], filename: str) -> str:
        """
        Save scraped data to file
        
        Args:
            posts_data: List of post dictionaries
            filename: Filename to save data
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.raw_data_dir / f"{filename}_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Data saved to {filepath}")
        return str(filepath)
    
    def load_data(self, filepath: str) -> List[Dict]:
        """
        Load scraped data from file
        
        Args:
            filepath: Path to data file
            
        Returns:
            List of post dictionaries
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data)} posts from {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return []
    
    def to_dataframe(self, posts_data: List[Dict]) -> pd.DataFrame:
        """
        Convert posts data to pandas DataFrame
        
        Args:
            posts_data: List of post dictionaries
            
        Returns:
            DataFrame with posts data
        """
        df = pd.DataFrame(posts_data)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract additional features
        df['caption_length'] = df['caption'].str.len()
        df['hashtag_count'] = df['hashtags'].str.len()
        df['mention_count'] = df['mentions'].str.len()
        df['engagement_rate'] = (df['likes'] + df['comments_count']) / df['likes'].replace(0, 1)
        
        return df

def main():
    """
    Example usage of InstagramScraper
    """
    import json
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize scraper
    scraper = InstagramScraper(config)
    
    # Example: Scrape posts from a profile
    # Note: You'll need to provide actual Instagram credentials
    # scraper.login('your_username', 'your_password')
    
    # posts_data = scraper.get_profile_posts('example_profile', max_posts=10)
    # scraper.save_data(posts_data, 'example_profile')
    
    print("Instagram scraper initialized. Configure credentials and run scraping functions.")

if __name__ == "__main__":
    main()
