"""
Simple YouTube Test Script
Tests YouTube API integration without the full pipeline
"""

import os
import sys
import yaml
import googleapiclient.discovery
import googleapiclient.errors

def test_youtube_api():
    """Test YouTube API connection and data collection."""
    
    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    youtube_config = config.get('data_source', {}).get('youtube', {})
    api_key = youtube_config.get('api_key')
    
    if not api_key or api_key == "YOUR_YOUTUBE_API_KEY":
        print("Please add your YouTube API key to config.yaml")
        return False
    
    print("Testing YouTube API connection...")
    
    try:
        # Initialize YouTube API client
        youtube = googleapiclient.discovery.build(
            'youtube', 'v3', developerKey=api_key
        )
        print("YouTube API client initialized successfully")
        
        # Test search
        search_query = youtube_config.get('search_query', 'sentiment analysis')
        print(f"Searching for videos: {search_query}")
        
        request = youtube.search().list(
            part="snippet",
            q=search_query,
            type="video",
            maxResults=5,
            order="relevance"
        )
        
        response = request.execute()
        videos = response.get('items', [])
        
        print(f"Found {len(videos)} videos:")
        
        for i, video in enumerate(videos, 1):
            title = video['snippet']['title']
            channel = video['snippet']['channelTitle']
            video_id = video['id']['videoId']
            print(f"  {i}. {title}")
            print(f"     Channel: {channel}")
            print(f"     ID: {video_id}")
            print()
        
        # Test getting comments from first video
        if videos:
            first_video_id = videos[0]['id']['videoId']
            print(f"Getting comments from first video...")
            
            comments_request = youtube.commentThreads().list(
                part="snippet",
                videoId=first_video_id,
                maxResults=10,
                order="relevance",
                textFormat="plainText"
            )
            
            comments_response = comments_request.execute()
            comments = comments_response.get('items', [])
            
            print(f"Found {len(comments)} comments:")
            
            for i, comment in enumerate(comments, 1):
                comment_text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                author = comment['snippet']['topLevelComment']['snippet']['authorDisplayName']
                likes = comment['snippet']['topLevelComment']['snippet'].get('likeCount', 0)
                
                # Truncate long comments
                if len(comment_text) > 100:
                    comment_text = comment_text[:100] + "..."
                
                print(f"  {i}. {author}: {comment_text} ({likes} likes)")
        
        print("\nYouTube API test completed successfully!")
        return True
        
    except googleapiclient.errors.HttpError as e:
        if e.resp.status == 403:
            print("YouTube API quota exceeded or invalid API key")
            print("   Please check your API key and ensure YouTube Data API v3 is enabled")
        else:
            print(f"YouTube API error: {e}")
        return False
    
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("YouTube API Test")
    print("=" * 50)
    
    success = test_youtube_api()
    
    if success:
        print("\nTest passed! YouTube integration is working.")
        print("You can now run the full sentiment analysis pipeline.")
    else:
        print("\nTest failed. Please check the error messages above.")
