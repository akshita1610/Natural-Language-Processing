"""
Analytics and Visualization Module
Creates comprehensive visualizations for Instagram sentiment analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from collections import Counter
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SentimentVisualizer:
    """
    Comprehensive visualization toolkit for sentiment analysis
    """
    
    def __init__(self, config: Dict):
        """
        Initialize visualizer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.viz_config = config["visualization"]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Color scheme
        self.colors = self.viz_config["color_scheme"]
        
        # Create output directory
        self.output_dir = Path("data/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib figure size and DPI
        plt.rcParams['figure.figsize'] = self.viz_config["figure_size"]
        plt.rcParams['figure.dpi'] = self.viz_config["dpi"]
    
    def sentiment_distribution_pie(self, df: pd.DataFrame, sentiment_column: str = 'sentiment', 
                                 save_path: Optional[str] = None) -> go.Figure:
        """
        Create pie chart of sentiment distribution
        
        Args:
            df: DataFrame with sentiment data
            sentiment_column: Name of sentiment column
            save_path: Path to save visualization
            
        Returns:
            Plotly figure object
        """
        sentiment_counts = df[sentiment_column].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.3,
            marker_colors=[self.colors.get(label.lower(), '#808080') for label in sentiment_counts.index],
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig.update_layout(
            title="Sentiment Distribution",
            font=dict(size=14),
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Sentiment pie chart saved to {save_path}")
        
        return fig
    
    def sentiment_timeseries(self, df: pd.DataFrame, date_column: str = 'timestamp',
                           sentiment_column: str = 'sentiment', save_path: Optional[str] = None) -> go.Figure:
        """
        Create time series plot of sentiment over time
        
        Args:
            df: DataFrame with timestamp and sentiment data
            date_column: Name of date column
            sentiment_column: Name of sentiment column
            save_path: Path to save visualization
            
        Returns:
            Plotly figure object
        """
        # Ensure timestamp is datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Group by date and sentiment
        daily_sentiment = df.groupby([df[date_column].dt.date, sentiment_column]).size().unstack(fill_value=0)
        
        # Create cumulative plot
        fig = go.Figure()
        
        for sentiment in daily_sentiment.columns:
            fig.add_trace(go.Scatter(
                x=daily_sentiment.index,
                y=daily_sentiment[sentiment],
                mode='lines+markers',
                name=sentiment.title(),
                line=dict(color=self.colors.get(sentiment.lower(), '#808080')),
                stackgroup='one'
            ))
        
        fig.update_layout(
            title="Sentiment Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Posts",
            hovermode='x unified',
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Time series plot saved to {save_path}")
        
        return fig
    
    def engagement_sentiment_correlation(self, df: pd.DataFrame, 
                                       engagement_column: str = 'engagement_rate',
                                       sentiment_column: str = 'sentiment',
                                       save_path: Optional[str] = None) -> go.Figure:
        """
        Analyze correlation between engagement and sentiment
        
        Args:
            df: DataFrame with engagement and sentiment data
            engagement_column: Name of engagement column
            sentiment_column: Name of sentiment column
            save_path: Path to save visualization
            
        Returns:
            Plotly figure object
        """
        # Create box plot
        fig = go.Figure()
        
        for sentiment in df[sentiment_column].unique():
            sentiment_data = df[df[sentiment_column] == sentiment][engagement_column]
            
            fig.add_trace(go.Box(
                y=sentiment_data,
                name=sentiment.title(),
                marker_color=self.colors.get(sentiment.lower(), '#808080'),
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title="Engagement Rate by Sentiment",
            xaxis_title="Sentiment",
            yaxis_title="Engagement Rate",
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Engagement correlation plot saved to {save_path}")
        
        return fig
    
    def word_cloud_by_sentiment(self, df: pd.DataFrame, text_column: str = 'processed_text',
                              sentiment_column: str = 'sentiment', 
                              save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create word clouds for each sentiment category
        
        Args:
            df: DataFrame with text and sentiment data
            text_column: Name of text column
            sentiment_column: Name of sentiment column
            save_path: Base path to save visualizations
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        for sentiment in df[sentiment_column].unique():
            # Filter texts for this sentiment
            sentiment_texts = df[df[sentiment_column] == sentiment][text_column]
            
            # Combine all texts
            combined_text = ' '.join(sentiment_texts.astype(str))
            
            if not combined_text.strip():
                continue
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100,
                relative_scaling=0.5,
                random_state=42
            ).generate(combined_text)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'{sentiment.title()} Sentiment Word Cloud', fontsize=16, pad=20)
            
            figures[sentiment] = fig
            
            # Save if path provided
            if save_path:
                sentiment_save_path = f"{save_path}_{sentiment.lower()}.png"
                fig.savefig(sentiment_save_path, bbox_inches='tight', dpi=300)
                plt.close(fig)
                self.logger.info(f"Word cloud for {sentiment} saved to {sentiment_save_path}")
        
        return figures
    
    def hashtag_sentiment_heatmap(self, df: pd.DataFrame, 
                                sentiment_column: str = 'sentiment',
                                top_n: int = 20, save_path: Optional[str] = None) -> go.Figure:
        """
        Create heatmap of sentiment distribution by hashtag
        
        Args:
            df: DataFrame with hashtag and sentiment data
            sentiment_column: Name of sentiment column
            top_n: Number of top hashtags to include
            save_path: Path to save visualization
            
        Returns:
            Plotly figure object
        """
        # Extract hashtags and create sentiment mapping
        hashtag_sentiment_data = []
        
        for _, row in df.iterrows():
            hashtags = row.get('hashtags', [])
            if isinstance(hashtags, list):
                for hashtag in hashtags:
                    hashtag_sentiment_data.append({
                        'hashtag': hashtag.lower(),
                        'sentiment': row[sentiment_column]
                    })
        
        if not hashtag_sentiment_data:
            self.logger.warning("No hashtag data found")
            return go.Figure()
        
        hashtag_df = pd.DataFrame(hashtag_sentiment_data)
        
        # Get top hashtags
        top_hashtags = hashtag_df['hashtag'].value_counts().head(top_n).index
        filtered_df = hashtag_df[hashtag_df['hashtag'].isin(top_hashtags)]
        
        # Create contingency table
        heatmap_data = pd.crosstab(filtered_df['hashtag'], filtered_df['sentiment'], normalize='index')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlBu',
            showscale=True,
            text=np.round(heatmap_data.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Sentiment Distribution by Hashtag",
            xaxis_title="Sentiment",
            yaxis_title="Hashtag",
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Hashtag heatmap saved to {save_path}")
        
        return fig
    
    def sentiment_by_hour(self, df: pd.DataFrame, 
                         timestamp_column: str = 'timestamp',
                         sentiment_column: str = 'sentiment',
                         save_path: Optional[str] = None) -> go.Figure:
        """
        Analyze sentiment patterns by hour of day
        
        Args:
            df: DataFrame with timestamp and sentiment data
            timestamp_column: Name of timestamp column
            sentiment_column: Name of sentiment column
            save_path: Path to save visualization
            
        Returns:
            Plotly figure object
        """
        # Extract hour from timestamp
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df['hour'] = df[timestamp_column].dt.hour
        
        # Group by hour and sentiment
        hourly_sentiment = df.groupby(['hour', sentiment_column]).size().unstack(fill_value=0)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        for sentiment in hourly_sentiment.columns:
            fig.add_trace(go.Bar(
                x=hourly_sentiment.index,
                y=hourly_sentiment[sentiment],
                name=sentiment.title(),
                marker_color=self.colors.get(sentiment.lower(), '#808080')
            ))
        
        fig.update_layout(
            title="Sentiment Distribution by Hour of Day",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Posts",
            barmode='stack',
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Hourly sentiment plot saved to {save_path}")
        
        return fig
    
    def top_influential_posts(self, df: pd.DataFrame, 
                            engagement_column: str = 'engagement_rate',
                            sentiment_column: str = 'sentiment',
                            text_column: str = 'caption',
                            top_n: int = 10) -> pd.DataFrame:
        """
        Identify top influential posts by sentiment and engagement
        
        Args:
            df: DataFrame with post data
            engagement_column: Name of engagement column
            sentiment_column: Name of sentiment column
            text_column: Name of text column
            top_n: Number of top posts to return
            
        Returns:
            DataFrame with top posts
        """
        # Sort by engagement rate
        top_posts = df.nlargest(top_n, engagement_column)[
            [text_column, sentiment_column, engagement_column, 'likes', 'comments_count']
        ]
        
        # Truncate text for display
        top_posts[text_column] = top_posts[text_column].str[:100] + '...'
        
        return top_posts
    
    def create_dashboard_summary(self, df: pd.DataFrame) -> Dict:
        """
        Create summary statistics for dashboard
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_posts': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'avg_engagement': df['engagement_rate'].mean(),
            'top_hashtags': self._get_top_hashtags(df),
            'posting_frequency': self._calculate_posting_frequency(df),
            'peak_hours': self._get_peak_posting_hours(df)
        }
        
        return summary
    
    def _get_top_hashtags(self, df: pd.DataFrame, top_n: int = 10) -> List[Dict]:
        """Get top hashtags with sentiment breakdown"""
        hashtag_data = []
        
        for _, row in df.iterrows():
            hashtags = row.get('hashtags', [])
            if isinstance(hashtags, list):
                for hashtag in hashtags:
                    hashtag_data.append({
                        'hashtag': hashtag.lower(),
                        'sentiment': row['sentiment']
                    })
        
        if not hashtag_data:
            return []
        
        hashtag_df = pd.DataFrame(hashtag_data)
        top_hashtags = hashtag_df['hashtag'].value_counts().head(top_n)
        
        result = []
        for hashtag, count in top_hashtags.items():
            hashtag_sentiments = hashtag_df[hashtag_df['hashtag'] == hashtag]['sentiment'].value_counts()
            result.append({
                'hashtag': hashtag,
                'count': int(count),
                'sentiments': hashtag_sentiments.to_dict()
            })
        
        return result
    
    def _calculate_posting_frequency(self, df: pd.DataFrame) -> Dict:
        """Calculate posting frequency metrics"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        date_range = (df['timestamp'].max() - df['timestamp'].min()).days + 1
        
        return {
            'posts_per_day': len(df) / max(date_range, 1),
            'most_active_day': df['timestamp'].dt.day_name().mode().iloc[0] if not df.empty else None
        }
    
    def _get_peak_posting_hours(self, df: pd.DataFrame) -> List[int]:
        """Get peak posting hours"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        hourly_counts = df['hour'].value_counts()
        peak_hours = hourly_counts.nlargest(3).index.tolist()
        
        return peak_hours
    
    def export_visualizations(self, df: pd.DataFrame, output_dir: str = "data/visualizations") -> List[str]:
        """
        Export all visualizations to files
        
        Args:
            df: DataFrame with sentiment analysis results
            output_dir: Directory to save visualizations
            
        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sentiment distribution pie chart
        fig1 = self.sentiment_distribution_pie(df)
        file1 = output_path / f"sentiment_distribution_{timestamp}.html"
        fig1.write_html(file1)
        saved_files.append(str(file1))
        
        # Time series plot
        fig2 = self.sentiment_timeseries(df)
        file2 = output_path / f"sentiment_timeseries_{timestamp}.html"
        fig2.write_html(file2)
        saved_files.append(str(file2))
        
        # Engagement correlation
        fig3 = self.engagement_sentiment_correlation(df)
        file3 = output_path / f"engagement_correlation_{timestamp}.html"
        fig3.write_html(file3)
        saved_files.append(str(file3))
        
        # Hashtag heatmap
        fig4 = self.hashtag_sentiment_heatmap(df)
        file4 = output_path / f"hashtag_heatmap_{timestamp}.html"
        fig4.write_html(file4)
        saved_files.append(str(file4))
        
        # Hourly sentiment
        fig5 = self.sentiment_by_hour(df)
        file5 = output_path / f"hourly_sentiment_{timestamp}.html"
        fig5.write_html(file5)
        saved_files.append(str(file5))
        
        # Word clouds
        wordcloud_base = output_path / f"wordcloud_{timestamp}"
        self.word_cloud_by_sentiment(df, save_path=str(wordcloud_base))
        
        self.logger.info(f"All visualizations exported to {output_path}")
        return saved_files

def main():
    """
    Example usage of SentimentVisualizer
    """
    import json
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize visualizer
    visualizer = SentimentVisualizer(config)
    
    # Create sample data
    sample_data = {
        'caption': [
            "I love this amazing product! #awesome #love",
            "Terrible experience, hate it #bad #disappointed",
            "It's okay, nothing special #neutral #average",
            "Absolutely wonderful! #great #amazing",
            "Poor quality, waste of money #terrible #disappointed"
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative'],
        'engagement_rate': [5.2, 2.1, 3.5, 8.7, 1.8],
        'likes': [120, 45, 78, 200, 30],
        'comments_count': [15, 8, 12, 25, 5],
        'hashtags': [['awesome', 'love'], ['bad', 'disappointed'], ['neutral', 'average'], 
                    ['great', 'amazing'], ['terrible', 'disappointed']],
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='D')
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create visualizations
    print("Creating sample visualizations...")
    
    # Sentiment distribution
    fig1 = visualizer.sentiment_distribution_pie(df)
    fig1.show()
    
    # Time series
    fig2 = visualizer.sentiment_timeseries(df)
    fig2.show()
    
    # Engagement correlation
    fig3 = visualizer.engagement_sentiment_correlation(df)
    fig3.show()
    
    # Word clouds
    wordclouds = visualizer.word_cloud_by_sentiment(df)
    for sentiment, fig in wordclouds.items():
        plt.show()
    
    print("Sample visualizations created successfully!")

if __name__ == "__main__":
    main()
