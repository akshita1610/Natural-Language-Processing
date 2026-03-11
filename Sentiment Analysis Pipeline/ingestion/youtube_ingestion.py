"""
YouTube Data Ingestion Module

Handles data collection from YouTube API for sentiment analysis.
Supports collecting video comments, titles, and descriptions.
"""

import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class YouTubeIngestion:
    """
    YouTube data ingestion class for collecting video comments and metadata.
    
    Supports:
    - Video comment collection
    - Video metadata (title, description)
    - Comment thread analysis
    """
    
    def __init__(self, config: Dict):
        """
        Initialize YouTube ingestion with configuration.
        
        Args:
            config: YouTube configuration dictionary
        """
        self.config = config
        self.api_key = config.get('api_key')
        self.youtube = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize YouTube API client."""
        try:
            self.youtube = googleapiclient.discovery.build(
                'youtube', 'v3', developerKey=self.api_key
            )
            logger.info("YouTube API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize YouTube client: {e}")
            raise
    
    def get_video_comments(self, video_id: str, max_comments: int = 100) -> List[Dict]:
        """
        Get comments from a specific YouTube video.
        
        Args:
            video_id: YouTube video ID
            max_comments: Maximum number of comments to collect
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        
        try:
            logger.info(f"Collecting comments from video: {video_id}")
            
            # Get top-level comments
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(max_comments, 100),
                order="relevance",
                textFormat="plainText"
            )
            
            while request and len(comments) < max_comments:
                response = request.execute()
                
                for item in response.get('items', []):
                    comment_data = self._process_comment_thread(item, video_id)
                    if comment_data:
                        comments.append(comment_data)
                    
                    if len(comments) >= max_comments:
                        break
                
                request = self.youtube.commentThreads().list_next(request, response)
            
            logger.info(f"Collected {len(comments)} comments from video {video_id}")
            
        except googleapiclient.errors.HttpError as e:
            if e.resp.status == 403:
                logger.error("YouTube API quota exceeded or invalid API key")
            else:
                logger.error(f"YouTube API error: {e}")
            raise
        
        return comments
    
    def _process_comment_thread(self, item: Dict, video_id: str) -> Optional[Dict]:
        """Process a comment thread and extract relevant data."""
        try:
            comment = item['snippet']['topLevelComment']['snippet']
            
            comment_data = {
                'comment_id': item['id'],
                'video_id': video_id,
                'text': comment['textDisplay'],
                'author': comment['authorDisplayName'],
                'like_count': comment.get('likeCount', 0),
                'reply_count': item['snippet'].get('totalReplyCount', 0),
                'published_at': comment['publishedAt'],
                'updated_at': comment.get('updatedAt', comment['publishedAt']),
                'source': 'youtube_comment'
            }
            
            return comment_data
            
        except Exception as e:
            logger.warning(f"Error processing comment thread: {e}")
            return None
    
    def get_video_metadata(self, video_id: str) -> Dict:
        """
        Get metadata for a specific YouTube video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with video metadata
        """
        try:
            logger.info(f"Getting metadata for video: {video_id}")
            
            request = self.youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            )
            
            response = request.execute()
            
            if not response.get('items'):
                logger.warning(f"No video found with ID: {video_id}")
                return {}
            
            video = response['items'][0]
            snippet = video['snippet']
            stats = video.get('statistics', {})
            
            metadata = {
                'video_id': video_id,
                'title': snippet['title'],
                'description': snippet.get('description', ''),
                'channel_title': snippet['channelTitle'],
                'published_at': snippet['publishedAt'],
                'view_count': int(stats.get('viewCount', 0)),
                'like_count': int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0)),
                'duration': video.get('contentDetails', {}).get('duration', ''),
                'tags': snippet.get('tags', []),
                'category_id': snippet.get('categoryId', ''),
                'source': 'youtube_video'
            }
            
            logger.info(f"Retrieved metadata for video: {snippet['title']}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting video metadata: {e}")
            return {}
    
    def search_videos(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for YouTube videos based on a query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of video dictionaries
        """
        videos = []
        
        try:
            logger.info(f"Searching YouTube for: {query}")
            
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=min(max_results, 50),
                order="relevance",
                videoDefinition="any",
                videoDuration="any"
            )
            
            response = request.execute()
            
            for item in response.get('items', []):
                video_data = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', ''),
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': item['snippet']['thumbnails']['default']['url'],
                    'source': 'youtube_search'
                }
                videos.append(video_data)
            
            logger.info(f"Found {len(videos)} videos for query: {query}")
            
        except Exception as e:
            logger.error(f"Error searching videos: {e}")
            raise
        
        return videos
    
    def collect_from_videos(self, video_ids: List[str], max_comments_per_video: int = 50) -> List[Dict]:
        """
        Collect comments from multiple videos.
        
        Args:
            video_ids: List of YouTube video IDs
            max_comments_per_video: Maximum comments per video
            
        Returns:
            List of all comments and metadata
        """
        all_data = []
        
        for video_id in video_ids:
            try:
                # Get video metadata
                metadata = self.get_video_metadata(video_id)
                if metadata:
                    all_data.append(metadata)
                
                # Get video comments
                comments = self.get_video_comments(video_id, max_comments_per_video)
                all_data.extend(comments)
                
                logger.info(f"Processed video {video_id}: {len(comments)} comments")
                
            except Exception as e:
                logger.warning(f"Error processing video {video_id}: {e}")
                continue
        
        logger.info(f"Collected total of {len(all_data)} items from {len(video_ids)} videos")
        
        return all_data
    
    def collect_from_search(self, query: str, max_videos: int = 5, max_comments_per_video: int = 50) -> List[Dict]:
        """
        Search for videos and collect their comments.
        
        Args:
            query: Search query
            max_videos: Maximum number of videos to process
            max_comments_per_video: Maximum comments per video
            
        Returns:
            List of comments and metadata
        """
        # Search for videos
        videos = self.search_videos(query, max_videos)
        
        if not videos:
            logger.warning("No videos found for search query")
            return []
        
        # Extract video IDs
        video_ids = [video['video_id'] for video in videos]
        
        # Collect data from videos
        return self.collect_from_videos(video_ids, max_comments_per_video)
    
    def get_trending_videos(self, region_code: str = "US", category_id: str = None, max_results: int = 10) -> List[Dict]:
        """
        Get trending YouTube videos.
        
        Args:
            region_code: Country code (US, GB, etc.)
            category_id: Optional category ID
            max_results: Maximum number of results
            
        Returns:
            List of trending video dictionaries
        """
        try:
            logger.info(f"Getting trending videos for region: {region_code}")
            
            request = self.youtube.videos().list(
                part="snippet,statistics",
                chart="mostPopular",
                regionCode=region_code,
                maxResults=min(max_results, 50),
                categoryId=category_id or "0"
            )
            
            response = request.execute()
            
            trending_videos = []
            
            for item in response.get('items', []):
                video_data = {
                    'video_id': item['id'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', ''),
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'view_count': int(item.get('statistics', {}).get('viewCount', 0)),
                    'like_count': int(item.get('statistics', {}).get('likeCount', 0)),
                    'comment_count': int(item.get('statistics', {}).get('commentCount', 0)),
                    'source': 'youtube_trending'
                }
                trending_videos.append(video_data)
            
            logger.info(f"Retrieved {len(trending_videos)} trending videos")
            
            return trending_videos
            
        except Exception as e:
            logger.error(f"Error getting trending videos: {e}")
            return []
    
    def to_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """
        Convert list of YouTube data to pandas DataFrame.
        
        Args:
            data: List of YouTube dictionaries
            
        Returns:
            Pandas DataFrame with YouTube data
        """
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert datetime columns
        datetime_columns = ['published_at', 'updated_at']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    
    def extract_video_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID or None if not found
        """
        import re
        
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
