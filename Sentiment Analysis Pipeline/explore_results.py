"""
YouTube Sentiment Analysis Results Explorer

This script helps you explore and visualize the sentiment analysis results
from the YouTube pipeline.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import glob

class ResultsExplorer:
    """Explore and visualize YouTube sentiment analysis results."""
    
    def __init__(self):
        """Initialize the explorer."""
        # Set up matplotlib for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Find the most recent results file
        self.find_latest_results()
        
    def find_latest_results(self):
        """Find the most recent YouTube sentiment results file."""
        csv_files = glob.glob("youtube_sentiment_results_*.csv")
        
        if not csv_files:
            print("No YouTube sentiment results files found!")
            print("Please run the pipeline first: python youtube_sentiment_pipeline.py")
            return None
        
        # Get the most recent file
        latest_file = max(csv_files, key=os.path.getctime)
        print(f"Loading results from: {latest_file}")
        
        self.df = pd.read_csv(latest_file)
        self.results_file = latest_file
        
        # Convert datetime columns
        datetime_columns = ['published_at', 'updated_at']
        for col in datetime_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
        
        print(f"Loaded {len(self.df)} records")
        return self.df
    
    def basic_statistics(self):
        """Show basic statistics of the dataset."""
        if self.df is None:
            return
        
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        
        # Overall dataset info
        videos_df = self.df[self.df['source'] == 'youtube_video']
        comments_df = self.df[self.df['source'] == 'youtube_comment']
        
        print(f"Total Records: {len(self.df)}")
        print(f"Videos: {len(videos_df)}")
        print(f"Comments: {len(comments_df)}")
        
        if not comments_df.empty:
            print(f"\nSentiment Distribution:")
            sentiment_counts = comments_df['sentiment'].value_counts()
            total_comments = len(comments_df)
            
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total_comments) * 100
                print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
            
            print(f"\nSentiment Scores:")
            print(f"  Average Compound Score: {comments_df['compound'].mean():.3f}")
            print(f"  Average Positive Score: {comments_df['pos'].mean():.3f}")
            print(f"  Average Negative Score: {comments_df['neg'].mean():.3f}")
            print(f"  Average Neutral Score: {comments_df['neu'].mean():.3f}")
            print(f"  Average Confidence: {comments_df['confidence'].mean():.3f}")
    
    def analyze_by_video(self):
        """Analyze sentiment by video."""
        if self.df is None:
            return
        
        print("\n" + "="*60)
        print("ANALYSIS BY VIDEO")
        print("="*60)
        
        # Get only comment data
        comments_df = self.df[self.df['source'] == 'youtube_comment'].copy()
        
        if comments_df.empty:
            print("No comment data available for analysis")
            return
        
        # Group by video
        video_analysis = comments_df.groupby('video_id').agg({
            'sentiment': ['count', lambda x: (x == 'positive').sum(), 
                         lambda x: (x == 'negative').sum(), lambda x: (x == 'neutral').sum()],
            'compound': ['mean', 'std'],
            'confidence': 'mean'
        }).round(3)
        
        # Flatten column names
        video_analysis.columns = ['total_comments', 'positive_count', 'negative_count', 
                                 'neutral_count', 'avg_compound', 'std_compound', 'avg_confidence']
        
        # Add percentages
        video_analysis['positive_pct'] = (video_analysis['positive_count'] / video_analysis['total_comments'] * 100).round(1)
        video_analysis['negative_pct'] = (video_analysis['negative_count'] / video_analysis['total_comments'] * 100).round(1)
        video_analysis['neutral_pct'] = (video_analysis['neutral_count'] / video_analysis['total_comments'] * 100).round(1)
        
        # Sort by average compound score
        video_analysis = video_analysis.sort_values('avg_compound', ascending=False)
        
        print("Top Videos by Sentiment:")
        for video_id, row in video_analysis.head(5).iterrows():
            # Get video title
            video_info = self.df[self.df['video_id'] == video_id].iloc[0]
            title = video_info.get('title', 'Unknown')
            channel = video_info.get('channel', 'Unknown')
            
            print(f"\n  {title[:60]}...")
            print(f"  Channel: {channel}")
            print(f"  Comments: {row['total_comments']} (👍{row['positive_pct']}%, 👎{row['negative_pct']}%, 😐{row['neutral_pct']}%)")
            print(f"  Avg Score: {row['avg_compound']:.3f}")
        
        print(f"\nMost Negative Videos:")
        for video_id, row in video_analysis.tail(3).iterrows():
            video_info = self.df[self.df['video_id'] == video_id].iloc[0]
            title = video_info.get('title', 'Unknown')
            channel = video_info.get('channel', 'Unknown')
            
            print(f"\n  {title[:60]}...")
            print(f"  Channel: {channel}")
            print(f"  Comments: {row['total_comments']} (👍{row['positive_pct']}%, 👎{row['negative_pct']}%, 😐{row['neutral_pct']}%)")
            print(f"  Avg Score: {row['avg_compound']:.3f}")
    
    def create_visualizations(self):
        """Create visualizations of the sentiment analysis results."""
        if self.df is None:
            return
        
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        comments_df = self.df[self.df['source'] == 'youtube_comment']
        
        if comments_df.empty:
            print("No comment data available for visualization")
            return
        
        # Create output directory
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Overall Sentiment Distribution
        plt.figure(figsize=(10, 6))
        sentiment_counts = comments_df['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        plt.subplot(1, 2, 1)
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                colors=colors[:len(sentiment_counts)], startangle=90)
        plt.title('Sentiment Distribution')
        
        plt.subplot(1, 2, 2)
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors[:len(sentiment_counts)])
        plt.title('Sentiment Counts')
        plt.ylabel('Number of Comments')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('visualizations/sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Sentiment Scores Distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(comments_df['compound'], bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(comments_df['compound'].mean(), color='red', linestyle='--', label=f"Mean: {comments_df['compound'].mean():.3f}")
        plt.title('Compound Score Distribution')
        plt.xlabel('Compound Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.hist(comments_df['pos'], bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.title('Positive Score Distribution')
        plt.xlabel('Positive Score')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 3)
        plt.hist(comments_df['neg'], bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.title('Negative Score Distribution')
        plt.xlabel('Negative Score')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 4)
        plt.scatter(comments_df['pos'], comments_df['neg'], alpha=0.6)
        plt.xlabel('Positive Score')
        plt.ylabel('Negative Score')
        plt.title('Positive vs Negative Scores')
        
        plt.tight_layout()
        plt.savefig('visualizations/sentiment_scores.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Video Comparison
        video_sentiment = comments_df.groupby('video_id')['compound'].mean().sort_values()
        
        plt.figure(figsize=(12, 8))
        
        # Get video titles for labels
        video_titles = []
        for video_id in video_sentiment.index:
            video_info = self.df[self.df['video_id'] == video_id].iloc[0]
            title = video_info.get('title', 'Unknown')
            # Truncate long titles
            if len(title) > 40:
                title = title[:40] + "..."
            video_titles.append(title)
        
        # Create horizontal bar chart
        y_pos = range(len(video_sentiment))
        colors = ['red' if x < 0 else 'green' if x > 0.05 else 'gray' for x in video_sentiment.values]
        
        bars = plt.barh(y_pos, video_sentiment.values, color=colors, alpha=0.7)
        plt.yticks(y_pos, video_titles)
        plt.xlabel('Average Compound Score')
        plt.title('Sentiment by Video (Average Compound Score)')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0.05, color='green', linestyle='--', alpha=0.5, label='Positive Threshold')
        plt.axvline(x=-0.05, color='red', linestyle='--', alpha=0.5, label='Negative Threshold')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01 if width >= 0 else width - 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/video_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Confidence Analysis
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        confidence_by_sentiment = comments_df.groupby('sentiment')['confidence'].mean()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        bars = plt.bar(confidence_by_sentiment.index, confidence_by_sentiment.values, color=colors)
        plt.title('Average Confidence by Sentiment')
        plt.ylabel('Average Confidence')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.subplot(1, 2, 2)
        plt.scatter(comments_df['confidence'], comments_df['compound'], alpha=0.6)
        plt.xlabel('Confidence')
        plt.ylabel('Compound Score')
        plt.title('Confidence vs Compound Score')
        
        plt.tight_layout()
        plt.savefig('visualizations/confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved to 'visualizations' folder:")
        print("  - sentiment_distribution.png")
        print("  - sentiment_scores.png")
        print("  - video_comparison.png")
        print("  - confidence_analysis.png")
    
    def show_sample_comments(self, n=5):
        """Show sample comments for each sentiment category."""
        if self.df is None:
            return
        
        print("\n" + "="*60)
        print(f"SAMPLE COMMENTS (Top {n} per category)")
        print("="*60)
        
        comments_df = self.df[self.df['source'] == 'youtube_comment']
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_comments = comments_df[comments_df['sentiment'] == sentiment]
            
            if sentiment_comments.empty:
                continue
            
            print(f"\n{sentiment.upper()} COMMENTS:")
            print("-" * 40)
            
            sample_comments = sentiment_comments.nlargest(n, 'confidence')
            
            for _, comment in sample_comments.iterrows():
                text = comment['text']
                if len(text) > 150:
                    text = text[:150] + "..."
                
                print(f"  • {text}")
                print(f"    Score: {comment['compound']:.3f}, Confidence: {comment['confidence']:.3f}")
                print(f"    Likes: {comment.get('like_count', 0)}")
                print()
    
    def analyze_engagement(self):
        """Analyze engagement vs sentiment."""
        if self.df is None:
            return
        
        print("\n" + "="*60)
        print("ENGAGEMENT ANALYSIS")
        print("="*60)
        
        comments_df = self.df[self.df['source'] == 'youtube_comment']
        
        if 'like_count' not in comments_df.columns:
            print("No like count data available")
            return
        
        # Remove comments with no likes for better analysis
        comments_with_likes = comments_df[comments_df['like_count'] > 0]
        
        if comments_with_likes.empty:
            print("No comments with likes found")
            return
        
        print(f"Analyzing {len(comments_with_likes)} comments with likes")
        
        # Engagement by sentiment
        engagement_by_sentiment = comments_with_likes.groupby('sentiment').agg({
            'like_count': ['mean', 'median', 'sum', 'count']
        }).round(2)
        
        engagement_by_sentiment.columns = ['avg_likes', 'median_likes', 'total_likes', 'comment_count']
        
        print("\nEngagement by Sentiment:")
        for sentiment, row in engagement_by_sentiment.iterrows():
            print(f"  {sentiment.capitalize()}:")
            print(f"    Average likes: {row['avg_likes']:.1f}")
            print(f"    Median likes: {row['median_likes']:.1f}")
            print(f"    Total likes: {row['total_likes']}")
            print(f"    Comment count: {row['comment_count']}")
        
        # Correlation between sentiment and likes
        correlation = comments_with_likes['compound'].corr(comments_with_likes['like_count'])
        print(f"\nCorrelation between sentiment score and likes: {correlation:.3f}")
        
        if abs(correlation) > 0.1:
            direction = "positive" if correlation > 0 else "negative"
            strength = "strong" if abs(correlation) > 0.3 else "moderate" if abs(correlation) > 0.2 else "weak"
            print(f"This suggests a {strength} {direction} correlation between sentiment and engagement.")
        else:
            print("There appears to be little correlation between sentiment and likes.")
    
    def explore_different_topic(self, search_query, max_videos=5, max_comments=30):
        """Run the pipeline with a different search query."""
        print(f"\n" + "="*60)
        print(f"EXPLORING NEW TOPIC: {search_query}")
        print("="*60)
        
        # Import the pipeline
        try:
            from youtube_sentiment_pipeline import YouTubeSentimentPipeline
            
            # Temporarily update config
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            original_query = config['data_source']['youtube']['search_query']
            original_max_videos = config['data_source']['youtube']['max_videos']
            original_max_comments = config['data_source']['youtube']['max_comments_per_video']
            
            # Update config for new analysis
            config['data_source']['youtube']['search_query'] = search_query
            config['data_source']['youtube']['max_videos'] = max_videos
            config['data_source']['youtube']['max_comments_per_video'] = max_comments
            
            # Save temporary config
            with open('config_temp.yaml', 'w') as f:
                yaml.dump(config, f)
            
            # Run pipeline with new config
            pipeline = YouTubeSentimentPipeline('config_temp.yaml')
            results = pipeline.run_pipeline()
            
            # Restore original config
            config['data_source']['youtube']['search_query'] = original_query
            config['data_source']['youtube']['max_videos'] = original_max_videos
            config['data_source']['youtube']['max_comments_per_video'] = original_max_comments
            
            with open('config.yaml', 'w') as f:
                yaml.dump(config, f)
            
            # Clean up temp file
            if os.path.exists('config_temp.yaml'):
                os.remove('config_temp.yaml')
            
            return results
            
        except Exception as e:
            print(f"Error exploring new topic: {e}")
            return None

def main():
    """Main function to run the exploration."""
    explorer = ResultsExplorer()
    
    if explorer.df is None:
        return
    
    while True:
        print("\n" + "="*60)
        print("YOUTUBE SENTIMENT ANALYSIS EXPLORER")
        print("="*60)
        print("1. Basic Statistics")
        print("2. Analyze by Video")
        print("3. Create Visualizations")
        print("4. Show Sample Comments")
        print("5. Engagement Analysis")
        print("6. Explore Different Topic")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            explorer.basic_statistics()
        elif choice == '2':
            explorer.analyze_by_video()
        elif choice == '3':
            explorer.create_visualizations()
        elif choice == '4':
            n = input("Number of sample comments per category (default 5): ").strip()
            n = int(n) if n.isdigit() else 5
            explorer.show_sample_comments(n)
        elif choice == '5':
            explorer.analyze_engagement()
        elif choice == '6':
            query = input("Enter search query: ").strip()
            if query:
                max_videos = input("Max videos (default 5): ").strip()
                max_videos = int(max_videos) if max_videos.isdigit() else 5
                max_comments = input("Max comments per video (default 30): ").strip()
                max_comments = int(max_comments) if max_comments.isdigit() else 30
                
                explorer.explore_different_topic(query, max_videos, max_comments)
        elif choice == '7':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
