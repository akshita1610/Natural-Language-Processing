"""
Visualization Module

Handles data visualization for sentiment analysis results.
Supports matplotlib, seaborn, and plotly for creating charts and dashboards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import os

# Optional Plotly support for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class SentimentVisualizer:
    """
    Visualization class for sentiment analysis results.
    
    Features:
    - Sentiment over time plots
    - Distribution charts
    - Volume analysis
    - Interactive dashboards (with Plotly)
    - Keyword sentiment visualization
    """
    
    def __init__(self, config: Dict):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Visualization configuration dictionary
        """
        self.config = config
        self.output_dir = config.get('output_dir', 'visualizations')
        self.chart_style = config.get('chart_style', 'seaborn')
        self.figure_size = config.get('figure_size', [12, 8])
        self.save_format = config.get('save_format', 'png')
        self.dpi = config.get('dpi', 300)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        self._set_style()
        
        logger.info(f"Initialized visualizer with output directory: {self.output_dir}")
    
    def _set_style(self):
        """Set matplotlib/seaborn style."""
        try:
            if self.chart_style == 'seaborn':
                sns.set_style("whitegrid")
                sns.set_palette("husl")
            elif self.chart_style == 'ggplot':
                plt.style.use('ggplot')
            elif self.chart_style == 'classic':
                plt.style.use('classic')
            else:
                plt.style.use('default')
        except Exception as e:
            logger.warning(f"Could not set chart style: {e}")
    
    def plot_sentiment_over_time(self, df: pd.DataFrame, time_column: str = 'created_at',
                               sentiment_column: str = 'sentiment', score_column: str = 'compound',
                               title: str = None, save: bool = True) -> str:
        """
        Plot sentiment trends over time.
        
        Args:
            df: DataFrame with sentiment data
            time_column: Column containing timestamps
            sentiment_column: Column containing sentiment labels
            score_column: Column containing sentiment scores
            title: Chart title
            save: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided")
                return ""
            
            # Ensure time column is datetime
            df_copy = df.copy()
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size)
            
            # Plot 1: Sentiment distribution over time
            sentiment_counts = df_copy.groupby([df_copy[time_column].dt.date, sentiment_column]).size().unstack(fill_value=0)
            sentiment_counts.plot(kind='area', stacked=True, ax=ax1, alpha=0.7)
            ax1.set_title('Sentiment Distribution Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Number of Posts')
            ax1.legend(title='Sentiment')
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Sentiment score over time
            if score_column in df_copy.columns:
                # Calculate rolling average
                df_sorted = df_copy.sort_values(time_column)
                df_sorted['rolling_score'] = df_sorted[score_column].rolling(window=50, min_periods=1).mean()
                
                ax2.plot(df_sorted[time_column], df_sorted['rolling_score'], alpha=0.7, linewidth=2)
                ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax2.set_title('Sentiment Score Over Time (Rolling Average)')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Compound Score')
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if title:
                fig.suptitle(title, fontsize=16, y=1.02)
            
            if save:
                filename = f"sentiment_over_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.save_format}"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved sentiment over time plot: {filepath}")
                plt.close()
                return filepath
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting sentiment over time: {e}")
            return ""
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, sentiment_column: str = 'sentiment',
                                  title: str = None, save: bool = True) -> str:
        """
        Plot sentiment distribution pie chart and bar chart.
        
        Args:
            df: DataFrame with sentiment data
            sentiment_column: Column containing sentiment labels
            title: Chart title
            save: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        try:
            if df.empty or sentiment_column not in df.columns:
                logger.warning("Invalid DataFrame or missing sentiment column")
                return ""
            
            # Get sentiment counts
            sentiment_counts = df[sentiment_column].value_counts()
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
            
            # Plot 1: Pie chart
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # green, red, gray
            wedges, texts, autotexts = ax1.pie(sentiment_counts.values, labels=sentiment_counts.index,
                                             autopct='%1.1f%%', colors=colors[:len(sentiment_counts)],
                                             startangle=90)
            ax1.set_title('Sentiment Distribution')
            
            # Plot 2: Bar chart
            bars = ax2.bar(sentiment_counts.index, sentiment_counts.values, color=colors[:len(sentiment_counts)])
            ax2.set_title('Sentiment Counts')
            ax2.set_xlabel('Sentiment')
            ax2.set_ylabel('Count')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if title:
                fig.suptitle(title, fontsize=16, y=1.02)
            
            if save:
                filename = f"sentiment_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.save_format}"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved sentiment distribution plot: {filepath}")
                plt.close()
                return filepath
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting sentiment distribution: {e}")
            return ""
    
    def plot_volume_analysis(self, df: pd.DataFrame, time_column: str = 'created_at',
                           title: str = None, save: bool = True) -> str:
        """
        Plot volume analysis over time.
        
        Args:
            df: DataFrame with timestamp data
            time_column: Column containing timestamps
            title: Chart title
            save: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        try:
            if df.empty or time_column not in df.columns:
                logger.warning("Invalid DataFrame or missing time column")
                return ""
            
            # Ensure time column is datetime
            df_copy = df.copy()
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Group by hour and count posts
            volume_by_hour = df_copy.groupby(df_copy[time_column].dt.hour).size()
            
            # Plot volume
            bars = ax.bar(volume_by_hour.index, volume_by_hour.values, alpha=0.7)
            ax.set_title('Post Volume by Hour of Day')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Number of Posts')
            ax.set_xticks(range(24))
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if title:
                fig.suptitle(title, fontsize=16, y=1.02)
            
            if save:
                filename = f"volume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.save_format}"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved volume analysis plot: {filepath}")
                plt.close()
                return filepath
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting volume analysis: {e}")
            return ""
    
    def plot_keyword_sentiment(self, keyword_df: pd.DataFrame, title: str = None, save: bool = True) -> str:
        """
        Plot keyword sentiment analysis.
        
        Args:
            keyword_df: DataFrame with keyword sentiment data
            title: Chart title
            save: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        try:
            if keyword_df.empty:
                logger.warning("Empty keyword DataFrame provided")
                return ""
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Sort by total mentions
            keyword_df_sorted = keyword_df.sort_values('total_mentions', ascending=True)
            
            # Create horizontal bar chart
            y_pos = np.arange(len(keyword_df_sorted))
            
            # Plot positive mentions
            ax.barh(y_pos, keyword_df_sorted['positive_count'], 
                   color='#2ecc71', alpha=0.7, label='Positive')
            
            # Plot negative mentions (stacked)
            ax.barh(y_pos, keyword_df_sorted['negative_count'], 
                   left=keyword_df_sorted['positive_count'],
                   color='#e74c3c', alpha=0.7, label='Negative')
            
            # Plot neutral mentions (stacked)
            ax.barh(y_pos, keyword_df_sorted['neutral_count'],
                   left=keyword_df_sorted['positive_count'] + keyword_df_sorted['negative_count'],
                   color='#95a5a6', alpha=0.7, label='Neutral')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(keyword_df_sorted['keyword'])
            ax.set_xlabel('Number of Mentions')
            ax.set_title('Keyword Sentiment Analysis')
            ax.legend()
            
            plt.tight_layout()
            
            if title:
                fig.suptitle(title, fontsize=16, y=1.02)
            
            if save:
                filename = f"keyword_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.save_format}"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved keyword sentiment plot: {filepath}")
                plt.close()
                return filepath
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting keyword sentiment: {e}")
            return ""
    
    def plot_sentiment_heatmap(self, df: pd.DataFrame, time_column: str = 'created_at',
                             sentiment_column: str = 'sentiment', title: str = None, save: bool = True) -> str:
        """
        Plot sentiment heatmap by hour and day.
        
        Args:
            df: DataFrame with sentiment data
            time_column: Column containing timestamps
            sentiment_column: Column containing sentiment labels
            title: Chart title
            save: Whether to save the plot
            
        Returns:
            Path to saved plot file
        """
        try:
            if df.empty or time_column not in df.columns:
                logger.warning("Invalid DataFrame or missing columns")
                return ""
            
            # Ensure time column is datetime
            df_copy = df.copy()
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])
            
            # Create hour and day columns
            df_copy['hour'] = df_copy[time_column].dt.hour
            df_copy['day'] = df_copy[time_column].dt.day_name()
            
            # Create pivot table for heatmap
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            df_copy['sentiment_score'] = df_copy[sentiment_column].map(sentiment_map)
            
            # Group by day and hour
            heatmap_data = df_copy.groupby(['day', 'hour'])['sentiment_score'].mean().unstack(fill_value=0)
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(day_order, fill_value=0)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=self.figure_size)
            sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0, 
                       ax=ax, fmt='.2f', cbar_kws={'label': 'Average Sentiment'})
            ax.set_title('Sentiment Heatmap by Day and Hour')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Day of Week')
            
            plt.tight_layout()
            
            if title:
                fig.suptitle(title, fontsize=16, y=1.02)
            
            if save:
                filename = f"sentiment_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.save_format}"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved sentiment heatmap: {filepath}")
                plt.close()
                return filepath
            else:
                plt.show()
                return ""
                
        except Exception as e:
            logger.error(f"Error plotting sentiment heatmap: {e}")
            return ""
    
    def create_dashboard(self, df: pd.DataFrame, trends_df: pd.DataFrame = None,
                       keyword_df: pd.DataFrame = None, interactive: bool = False) -> str:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            df: Main DataFrame with sentiment data
            trends_df: DataFrame with trend analysis data
            keyword_df: DataFrame with keyword sentiment data
            interactive: Whether to create interactive dashboard (requires Plotly)
            
        Returns:
            Path to saved dashboard file
        """
        try:
            if interactive and PLOTLY_AVAILABLE:
                return self._create_interactive_dashboard(df, trends_df, keyword_df)
            else:
                return self._create_static_dashboard(df, trends_df, keyword_df)
                
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return ""
    
    def _create_static_dashboard(self, df: pd.DataFrame, trends_df: pd.DataFrame = None,
                               keyword_df: pd.DataFrame = None) -> str:
        """Create static dashboard with matplotlib."""
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            
            # Define subplot layout
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Sentiment distribution (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts()
                colors = ['#2ecc71', '#e74c3c', '#95a5a6']
                ax1.pie(sentiment_counts.values, labels=sentiment_counts.index,
                       autopct='%1.1f%%', colors=colors[:len(sentiment_counts)])
                ax1.set_title('Sentiment Distribution')
            
            # 2. Sentiment over time (top middle and right)
            ax2 = fig.add_subplot(gs[0, 1:])
            if 'created_at' in df.columns and 'sentiment' in df.columns:
                df_copy = df.copy()
                df_copy['created_at'] = pd.to_datetime(df_copy['created_at'])
                sentiment_counts = df_copy.groupby([df_copy['created_at'].dt.date, 'sentiment']).size().unstack(fill_value=0)
                sentiment_counts.plot(kind='area', stacked=True, ax=ax2, alpha=0.7)
                ax2.set_title('Sentiment Over Time')
                ax2.legend(title='Sentiment')
            
            # 3. Volume by hour (middle left)
            ax3 = fig.add_subplot(gs[1, 0])
            if 'created_at' in df.columns:
                df_copy = df.copy()
                df_copy['created_at'] = pd.to_datetime(df_copy['created_at'])
                volume_by_hour = df_copy.groupby(df_copy['created_at'].dt.hour).size()
                ax3.bar(volume_by_hour.index, volume_by_hour.values, alpha=0.7)
                ax3.set_title('Volume by Hour')
                ax3.set_xlabel('Hour')
                ax3.set_ylabel('Count')
            
            # 4. Sentiment scores (middle middle)
            ax4 = fig.add_subplot(gs[1, 1])
            if 'compound' in df.columns:
                ax4.hist(df['compound'], bins=30, alpha=0.7, edgecolor='black')
                ax4.axvline(df['compound'].mean(), color='red', linestyle='--', label='Mean')
                ax4.set_title('Compound Score Distribution')
                ax4.set_xlabel('Compound Score')
                ax4.set_ylabel('Frequency')
                ax4.legend()
            
            # 5. Keyword sentiment (middle right)
            ax5 = fig.add_subplot(gs[1, 2])
            if keyword_df is not None and not keyword_df.empty:
                keyword_df_sorted = keyword_df.sort_values('total_mentions', ascending=True).head(10)
                y_pos = np.arange(len(keyword_df_sorted))
                ax5.barh(y_pos, keyword_df_sorted['total_mentions'], alpha=0.7)
                ax5.set_yticks(y_pos)
                ax5.set_yticklabels(keyword_df_sorted['keyword'])
                ax5.set_title('Top Keywords')
                ax5.set_xlabel('Mentions')
            
            # 6. Trend analysis (bottom row)
            ax6 = fig.add_subplot(gs[2, :])
            if trends_df is not None and not trends_df.empty and 'time_period' in trends_df.columns:
                ax6.plot(trends_df['time_period'], trends_df['positive_percentage'], 
                        label='Positive %', color='green', alpha=0.7)
                ax6.plot(trends_df['time_period'], trends_df['negative_percentage'], 
                        label='Negative %', color='red', alpha=0.7)
                ax6.plot(trends_df['time_period'], trends_df['neutral_percentage'], 
                        label='Neutral %', color='gray', alpha=0.7)
                ax6.set_title('Sentiment Trends Over Time')
                ax6.set_xlabel('Time Period')
                ax6.set_ylabel('Percentage')
                ax6.legend()
                ax6.tick_params(axis='x', rotation=45)
            
            plt.suptitle('Sentiment Analysis Dashboard', fontsize=20, y=0.98)
            
            # Save dashboard
            filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.save_format}"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created static dashboard: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creating static dashboard: {e}")
            return ""
    
    def _create_interactive_dashboard(self, df: pd.DataFrame, trends_df: pd.DataFrame = None,
                                   keyword_df: pd.DataFrame = None) -> str:
        """Create interactive dashboard with Plotly."""
        try:
            if not PLOTLY_AVAILABLE:
                logger.warning("Plotly not available, falling back to static dashboard")
                return self._create_static_dashboard(df, trends_df, keyword_df)
            
            # Create subplot figure
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=('Sentiment Distribution', 'Sentiment Over Time', 'Volume by Hour',
                              'Compound Scores', 'Keywords', 'Trends'),
                specs=[[{"type": "pie"}, {"type": "scatter", "colspan": 2}, None],
                       [{"type": "bar"}, {"type": "histogram"}, {"type": "bar"}],
                       [{"type": "scatter", "colspan": 3}, None, None]]
            )
            
            # 1. Sentiment distribution (pie chart)
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts()
                fig.add_trace(
                    go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                          name="Sentiment"),
                    row=1, col=1
                )
            
            # 2. Sentiment over time (area chart)
            if 'created_at' in df.columns and 'sentiment' in df.columns:
                df_copy = df.copy()
                df_copy['created_at'] = pd.to_datetime(df_copy['created_at'])
                sentiment_counts = df_copy.groupby([df_copy['created_at'].dt.date, 'sentiment']).size().unstack(fill_value=0)
                
                for sentiment in sentiment_counts.columns:
                    fig.add_trace(
                        go.Scatter(x=sentiment_counts.index, y=sentiment_counts[sentiment],
                                  mode='lines', stackgroup='one', name=sentiment),
                        row=1, col=2
                    )
            
            # 3. Volume by hour (bar chart)
            if 'created_at' in df.columns:
                df_copy = df.copy()
                df_copy['created_at'] = pd.to_datetime(df_copy['created_at'])
                volume_by_hour = df_copy.groupby(df_copy['created_at'].dt.hour).size()
                
                fig.add_trace(
                    go.Bar(x=volume_by_hour.index, y=volume_by_hour.values, name='Volume'),
                    row=2, col=1
                )
            
            # 4. Compound scores (histogram)
            if 'compound' in df.columns:
                fig.add_trace(
                    go.Histogram(x=df['compound'], name='Compound Scores'),
                    row=2, col=2
                )
            
            # 5. Keywords (bar chart)
            if keyword_df is not None and not keyword_df.empty:
                keyword_df_sorted = keyword_df.sort_values('total_mentions', ascending=True).head(10)
                
                fig.add_trace(
                    go.Bar(x=keyword_df_sorted['total_mentions'], y=keyword_df_sorted['keyword'],
                          orientation='h', name='Keywords'),
                    row=2, col=3
                )
            
            # 6. Trends (line chart)
            if trends_df is not None and not trends_df.empty and 'time_period' in trends_df.columns:
                fig.add_trace(
                    go.Scatter(x=trends_df['time_period'], y=trends_df['positive_percentage'],
                              mode='lines', name='Positive %'),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=trends_df['time_period'], y=trends_df['negative_percentage'],
                              mode='lines', name='Negative %'),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=trends_df['time_period'], y=trends_df['neutral_percentage'],
                              mode='lines', name='Neutral %'),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                title_text="Sentiment Analysis Dashboard",
                showlegend=True,
                height=1200
            )
            
            # Save interactive dashboard
            filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(filepath)
            
            logger.info(f"Created interactive dashboard: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
            return ""
    
    def generate_report(self, df: pd.DataFrame, trends_df: pd.DataFrame = None,
                       keyword_df: pd.DataFrame = None, output_path: str = None) -> str:
        """
        Generate a comprehensive report with visualizations and statistics.
        
        Args:
            df: Main DataFrame with sentiment data
            trends_df: DataFrame with trend analysis data
            keyword_df: DataFrame with keyword sentiment data
            output_path: Path to save the report
            
        Returns:
            Path to saved report file
        """
        try:
            if output_path is None:
                output_path = os.path.join(self.output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            
            # Generate visualizations
            dashboard_path = self.create_dashboard(df, trends_df, keyword_df, interactive=True)
            
            # Calculate statistics
            stats = self._calculate_statistics(df)
            
            # Create HTML report
            html_content = self._generate_html_report(stats, dashboard_path)
            
            # Save report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Generated comprehensive report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics for the report."""
        try:
            stats = {}
            
            if not df.empty:
                # Basic statistics
                stats['total_posts'] = len(df)
                
                # Sentiment distribution
                if 'sentiment' in df.columns:
                    sentiment_counts = df['sentiment'].value_counts()
                    stats['sentiment_distribution'] = sentiment_counts.to_dict()
                    stats['sentiment_percentages'] = (sentiment_counts / len(df) * 100).to_dict()
                
                # Score statistics
                if 'compound' in df.columns:
                    stats['compound_stats'] = {
                        'mean': df['compound'].mean(),
                        'median': df['compound'].median(),
                        'std': df['compound'].std(),
                        'min': df['compound'].min(),
                        'max': df['compound'].max()
                    }
                
                # Time range
                if 'created_at' in df.columns:
                    df_copy = df.copy()
                    df_copy['created_at'] = pd.to_datetime(df_copy['created_at'])
                    stats['time_range'] = {
                        'start': df_copy['created_at'].min(),
                        'end': df_copy['created_at'].max()
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}
    
    def _generate_html_report(self, stats: Dict, dashboard_path: str) -> str:
        """Generate HTML content for the report."""
        try:
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Sentiment Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; color: #2c3e50; }}
                    .stats {{ margin: 20px 0; }}
                    .stat-item {{ margin: 10px 0; }}
                    .dashboard {{ text-align: center; margin: 30px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Sentiment Analysis Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="stats">
                    <h2>Key Statistics</h2>
                    <div class="stat-item"><strong>Total Posts:</strong> {stats.get('total_posts', 'N/A')}</div>
                    
                    <h3>Sentiment Distribution</h3>
                    {self._format_sentiment_stats(stats.get('sentiment_percentages', {}))}
                    
                    <h3>Score Statistics</h3>
                    {self._format_score_stats(stats.get('compound_stats', {}))}
                    
                    <h3>Time Range</h3>
                    {self._format_time_stats(stats.get('time_range', {}))}
                </div>
                
                <div class="dashboard">
                    <h2>Interactive Dashboard</h2>
                    <iframe src="{os.path.basename(dashboard_path)}" width="100%" height="800px"></iframe>
                </div>
            </body>
            </html>
            """
            
            return html_template
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return "<html><body><h1>Error generating report</h1></body></html>"
    
    def _format_sentiment_stats(self, sentiment_pct: Dict) -> str:
        """Format sentiment statistics for HTML."""
        if not sentiment_pct:
            return "<p>No sentiment data available</p>"
        
        html = "<ul>"
        for sentiment, percentage in sentiment_pct.items():
            html += f"<li><strong>{sentiment.capitalize()}:</strong> {percentage:.1f}%</li>"
        html += "</ul>"
        
        return html
    
    def _format_score_stats(self, score_stats: Dict) -> str:
        """Format score statistics for HTML."""
        if not score_stats:
            return "<p>No score data available</p>"
        
        html = "<ul>"
        for stat, value in score_stats.items():
            html += f"<li><strong>{stat.replace('_', ' ').title()}:</strong> {value:.3f}</li>"
        html += "</ul>"
        
        return html
    
    def _format_time_stats(self, time_stats: Dict) -> str:
        """Format time statistics for HTML."""
        if not time_stats:
            return "<p>No time data available</p>"
        
        html = "<ul>"
        html += f"<li><strong>Start:</strong> {time_stats.get('start', 'N/A')}</li>"
        html += f"<li><strong>End:</strong> {time_stats.get('end', 'N/A')}</li>"
        html += "</ul>"
        
        return html
