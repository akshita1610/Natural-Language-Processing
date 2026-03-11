"""
YouTube Sentiment Analysis Pipeline - Simplified Working Version

This script collects YouTube comments and performs sentiment analysis.
Perfect for testing and demonstration purposes.
"""

import os
import sys
import yaml
import pandas as pd
import googleapiclient.discovery
import googleapiclient.errors
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import re

class YouTubeSentimentPipeline:
    """Simplified YouTube sentiment analysis pipeline."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize the pipeline."""
        self.config = self._load_config(config_path)
        self.youtube = self._initialize_youtube()
        self.analyzer = SentimentIntensityAnalyzer()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _initialize_youtube(self):
        """Initialize YouTube API client."""
        api_key = self.config.get('data_source', {}).get('youtube', {}).get('api_key')
        
        if not api_key or api_key == "YOUR_YOUTUBE_API_KEY":
            raise ValueError("Please add your YouTube API key to config.yaml")
        
        return googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)
    
    def search_videos(self, query, max_results=10):
        """Search for YouTube videos."""
        request = self.youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results,
            order="relevance"
        )
        
        response = request.execute()
        return response.get('items', [])
    
    def get_video_comments(self, video_id, max_comments=50):
        """Get comments from a specific video."""
        comments = []
        
        try:
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
                    comment = item['snippet']['topLevelComment']['snippet']
                    comment_data = {
                        'comment_id': item['id'],
                        'video_id': video_id,
                        'text': comment['textDisplay'],
                        'author': comment['authorDisplayName'],
                        'like_count': comment.get('likeCount', 0),
                        'published_at': comment['publishedAt'],
                        'updated_at': comment.get('updatedAt', comment['publishedAt'])
                    }
                    comments.append(comment_data)
                
                request = self.youtube.commentThreads().list_next(request, response)
        
        except googleapiclient.errors.HttpError as e:
            print(f"Error getting comments for video {video_id}: {e}")
        
        return comments
    
    def clean_text(self, text):
        """Basic text cleaning."""
        if not isinstance(text, str):
            return ""
        
        try:
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove mentions and hashtags but keep the text
            text = re.sub(r'@(\w+)', r'\1', text)
            text = re.sub(r'#(\w+)', r'\1', text)
            
            # Remove emojis and non-ASCII characters
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception:
            # If cleaning fails, return empty string
            return ""
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using VADER."""
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return {
                'text': text,
                'cleaned_text': '',
                'sentiment': 'neutral',
                'compound': 0.0,
                'pos': 0.0,
                'neg': 0.0,
                'neu': 0.0,
                'confidence': 0.0
            }
        
        scores = self.analyzer.polarity_scores(cleaned_text)
        
        # Determine sentiment
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        confidence = max(scores['pos'], scores['neg'], scores['neu'])
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment,
            'compound': compound,
            'pos': scores['pos'],
            'neg': scores['neg'],
            'neu': scores['neu'],
            'confidence': confidence
        }
    
    def run_pipeline(self):
        """Run the complete YouTube sentiment analysis pipeline."""
        print("Starting YouTube Sentiment Analysis Pipeline")
        print("=" * 60)
        
        # Get configuration
        youtube_config = self.config.get('data_source', {}).get('youtube', {})
        search_query = youtube_config.get('search_query', 'sentiment analysis OR machine learning OR AI')
        max_videos = youtube_config.get('max_videos', 5)
        max_comments_per_video = youtube_config.get('max_comments_per_video', 20)
        
        print(f"Searching for videos: {search_query}")
        print(f"Max videos: {max_videos}")
        print(f"Max comments per video: {max_comments_per_video}")
        print()
        
        # Search for videos
        videos = self.search_videos(search_query, max_videos)
        
        if not videos:
            print("No videos found for the search query")
            return
        
        print(f"Found {len(videos)} videos")
        
        all_data = []
        
        # Process each video
        for i, video in enumerate(videos, 1):
            video_id = video['id']['videoId']
            title = video['snippet']['title']
            channel = video['snippet']['channelTitle']
            
            print(f"\nProcessing video {i}/{len(videos)}: {title}")
            print(f"   Channel: {channel}")
            print(f"   ID: {video_id}")
            
            # Get video metadata
            video_data = {
                'video_id': video_id,
                'title': title,
                'channel': channel,
                'published_at': video['snippet']['publishedAt'],
                'description': video['snippet'].get('description', '')[:200] + "..." if len(video['snippet'].get('description', '')) > 200 else video['snippet'].get('description', ''),
                'source': 'youtube_video'
            }
            all_data.append(video_data)
            
            # Get comments
            print(f"   Getting comments...")
            comments = self.get_video_comments(video_id, max_comments_per_video)
            
            print(f"   Found {len(comments)} comments")
            
            # Analyze sentiment for each comment
            for j, comment in enumerate(comments, 1):
                try:
                    if j % 5 == 0:
                        print(f"     Analyzed {j}/{len(comments)} comments...")
                    
                    sentiment_result = self.analyze_sentiment(comment['text'])
                    
                    # Combine comment data with sentiment
                    combined_data = {
                        **comment,
                        **sentiment_result,
                        'source': 'youtube_comment'
                    }
                    all_data.append(combined_data)
                except Exception as e:
                    print(f"     Error processing comment {j}: {e}")
                    continue
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Convert datetime columns
        datetime_columns = ['published_at', 'updated_at']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        print(f"\nTotal items collected: {len(df)}")
        print(f"Videos: {len(df[df['source'] == 'youtube_video'])}")
        print(f"Comments: {len(df[df['source'] == 'youtube_comment'])}")
        
        # Sentiment analysis results
        comments_df = df[df['source'] == 'youtube_comment']
        if not comments_df.empty:
            sentiment_counts = comments_df['sentiment'].value_counts()
            total_comments = len(comments_df)
            
            print(f"\nSentiment Analysis Results:")
            print(f"   Positive: {sentiment_counts.get('positive', 0)} ({sentiment_counts.get('positive', 0)/total_comments*100:.1f}%)")
            print(f"   Negative: {sentiment_counts.get('negative', 0)} ({sentiment_counts.get('negative', 0)/total_comments*100:.1f}%)")
            print(f"   Neutral: {sentiment_counts.get('neutral', 0)} ({sentiment_counts.get('neutral', 0)/total_comments*100:.1f}%)")
            
            avg_compound = comments_df['compound'].mean()
            print(f"   Average Compound Score: {avg_compound:.3f}")
        
        # Save results
        output_file = f"youtube_sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Show some examples
        print(f"\nSample Results:")
        positive_comments = comments_df[comments_df['sentiment'] == 'positive'].head(2)
        negative_comments = comments_df[comments_df['sentiment'] == 'negative'].head(2)
        
        if not positive_comments.empty:
            print(f"\nPositive Comments:")
            for _, comment in positive_comments.iterrows():
                text = comment['text'][:100] + "..." if len(comment['text']) > 100 else comment['text']
                print(f"   • {text} (Score: {comment['compound']:.3f})")
        
        if not negative_comments.empty:
            print(f"\nNegative Comments:")
            for _, comment in negative_comments.iterrows():
                text = comment['text'][:100] + "..." if len(comment['text']) > 100 else comment['text']
                print(f"   • {text} (Score: {comment['compound']:.3f})")
        
        print(f"\nPipeline completed successfully!")
        return df

if __name__ == "__main__":
    try:
        pipeline = YouTubeSentimentPipeline()
        results = pipeline.run_pipeline()
        
        print(f"\nSummary:")
        print(f"   - Total data points: {len(results)}")
        print(f"   - CSV file created with all results")
        print(f"   - Ready for further analysis and visualization!")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Please check your configuration and API credentials.")
