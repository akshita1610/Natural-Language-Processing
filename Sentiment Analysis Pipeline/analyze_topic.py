"""
Analyze sentiment for a specific topic
"""

from youtube_sentiment_pipeline import YouTubeSentimentPipeline

def analyze_topic(search_query, max_videos=5, max_comments=20):
    """Analyze sentiment for a specific topic."""
    
    print(f"Analyzing topic: '{search_query}'")
    print(f"Max videos: {max_videos}, Max comments per video: {max_comments}")
    print("=" * 60)
    
    # Create pipeline
    pipeline = YouTubeSentimentPipeline()
    
    # Update config for this analysis
    pipeline.config['data_source']['youtube']['search_query'] = search_query
    pipeline.config['data_source']['youtube']['max_videos'] = max_videos
    pipeline.config['data_source']['youtube']['max_comments_per_video'] = max_comments
    
    # Run analysis
    results = pipeline.run_pipeline()
    
    return results

if __name__ == "__main__":
    # Example topics to analyze
    topics = [
        "ChatGPT tutorial",
        "machine learning course review", 
        "Python programming help",
        "data science career",
        "AI ethics debate"
    ]
    
    print("Available topics to analyze:")
    for i, topic in enumerate(topics, 1):
        print(f"{i}. {topic}")
    
    print("\nEnter a topic or choose from above (1-5):")
    choice = input().strip()
    
    if choice.isdigit() and 1 <= int(choice) <= 5:
        search_query = topics[int(choice) - 1]
    else:
        search_query = choice
    
    if search_query:
        analyze_topic(search_query)
    else:
        print("No topic provided.")
