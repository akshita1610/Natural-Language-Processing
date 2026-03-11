"""
Trend Analysis Module

Handles trend tracking and sentiment analysis over time.
Computes rolling averages, sentiment trends, and keyword-level analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
from typing import List, Dict, Optional, Tuple, Union
import re

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Trend analysis class for sentiment tracking over time.
    
    Features:
    - Rolling averages
    - Time-based sentiment trends
    - Keyword-level sentiment tracking
    - Volume analysis
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trend analyzer with configuration.
        
        Args:
            config: Trend analysis configuration dictionary
        """
        self.config = config
        self.window_size = config.get('window_size', 100)
        self.time_interval = config.get('time_interval', 'hour')
        self.keywords = config.get('keywords', [])
        
        logger.info(f"Initialized trend analyzer with window_size={self.window_size}, interval={self.time_interval}")
    
    def compute_rolling_sentiment(self, df: pd.DataFrame, sentiment_column: str = 'sentiment', 
                                 score_column: str = 'compound', time_column: str = 'created_at') -> pd.DataFrame:
        """
        Compute rolling sentiment averages.
        
        Args:
            df: DataFrame with sentiment data
            sentiment_column: Column containing sentiment labels
            score_column: Column containing sentiment scores
            time_column: Column containing timestamps
            
        Returns:
            DataFrame with rolling sentiment statistics
        """
        try:
            # Ensure time column is datetime
            if time_column not in df.columns:
                logger.warning(f"Time column '{time_column}' not found")
                return pd.DataFrame()
            
            df_copy = df.copy()
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])
            
            # Sort by time
            df_copy = df_copy.sort_values(time_column)
            
            # Initialize rolling statistics
            rolling_stats = []
            
            for i in range(len(df_copy)):
                # Get window of data
                start_idx = max(0, i - self.window_size + 1)
                window_data = df_copy.iloc[start_idx:i+1]
                
                if len(window_data) == 0:
                    continue
                
                # Calculate sentiment distribution
                sentiment_counts = window_data[sentiment_column].value_counts()
                total_items = len(window_data)
                
                # Calculate percentages
                positive_pct = (sentiment_counts.get('positive', 0) / total_items) * 100
                negative_pct = (sentiment_counts.get('negative', 0) / total_items) * 100
                neutral_pct = (sentiment_counts.get('neutral', 0) / total_items) * 100
                
                # Calculate average scores
                avg_score = window_data[score_column].mean() if score_column in window_data.columns else 0.0
                
                # Create rolling statistics record
                stats = {
                    'timestamp': df_copy.iloc[i][time_column],
                    'window_size': len(window_data),
                    'positive_count': sentiment_counts.get('positive', 0),
                    'negative_count': sentiment_counts.get('negative', 0),
                    'neutral_count': sentiment_counts.get('neutral', 0),
                    'positive_percentage': positive_pct,
                    'negative_percentage': negative_pct,
                    'neutral_percentage': neutral_pct,
                    'average_score': avg_score,
                    'total_items': total_items
                }
                
                rolling_stats.append(stats)
            
            result_df = pd.DataFrame(rolling_stats)
            logger.info(f"Computed rolling sentiment for {len(result_df)} windows")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error computing rolling sentiment: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment_over_time(self, df: pd.DataFrame, sentiment_column: str = 'sentiment',
                                   time_column: str = 'created_at', interval: str = None) -> pd.DataFrame:
        """
        Analyze sentiment trends over time intervals.
        
        Args:
            df: DataFrame with sentiment data
            sentiment_column: Column containing sentiment labels
            time_column: Column containing timestamps
            interval: Time interval (hour, day, week, month)
            
        Returns:
            DataFrame with time-based sentiment statistics
        """
        try:
            interval = interval or self.time_interval
            
            # Ensure time column is datetime
            if time_column not in df.columns:
                logger.warning(f"Time column '{time_column}' not found")
                return pd.DataFrame()
            
            df_copy = df.copy()
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])
            
            # Set time column as index for resampling
            df_copy = df_copy.set_index(time_column)
            
            # Define resampling frequency
            freq_map = {
                'minute': '1T',
                'hour': '1H',
                'day': '1D',
                'week': '1W',
                'month': '1M'
            }
            
            freq = freq_map.get(interval, '1H')
            
            # Resample and calculate statistics
            time_stats = []
            
            for time_period, group in df_copy.resample(freq):
                if len(group) == 0:
                    continue
                
                # Calculate sentiment distribution
                sentiment_counts = group[sentiment_column].value_counts()
                total_items = len(group)
                
                # Calculate percentages
                positive_pct = (sentiment_counts.get('positive', 0) / total_items) * 100
                negative_pct = (sentiment_counts.get('negative', 0) / total_items) * 100
                neutral_pct = (sentiment_counts.get('neutral', 0) / total_items) * 100
                
                # Calculate average compound score if available
                avg_compound = group['compound'].mean() if 'compound' in group.columns else 0.0
                
                # Create time period record
                stats = {
                    'time_period': time_period,
                    'total_items': total_items,
                    'positive_count': sentiment_counts.get('positive', 0),
                    'negative_count': sentiment_counts.get('negative', 0),
                    'neutral_count': sentiment_counts.get('neutral', 0),
                    'positive_percentage': positive_pct,
                    'negative_percentage': negative_pct,
                    'neutral_percentage': neutral_pct,
                    'average_compound': avg_compound,
                    'interval': interval
                }
                
                time_stats.append(stats)
            
            result_df = pd.DataFrame(time_stats)
            logger.info(f"Analyzed sentiment over {len(result_df)} {interval} intervals")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment over time: {e}")
            return pd.DataFrame()
    
    def analyze_keyword_sentiment(self, df: pd.DataFrame, text_column: str = 'text',
                                 sentiment_column: str = 'sentiment', keywords: List[str] = None) -> pd.DataFrame:
        """
        Analyze sentiment for specific keywords.
        
        Args:
            df: DataFrame with text and sentiment data
            text_column: Column containing text data
            sentiment_column: Column containing sentiment labels
            keywords: List of keywords to analyze (uses config if None)
            
        Returns:
            DataFrame with keyword sentiment statistics
        """
        try:
            keywords = keywords or self.keywords
            
            if not keywords:
                logger.warning("No keywords provided for analysis")
                return pd.DataFrame()
            
            if text_column not in df.columns or sentiment_column not in df.columns:
                logger.warning(f"Required columns not found: {text_column}, {sentiment_column}")
                return pd.DataFrame()
            
            keyword_stats = []
            
            for keyword in keywords:
                # Find texts containing the keyword (case-insensitive)
                pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
                matching_texts = df[df[text_column].str.contains(pattern, na=False)]
                
                if len(matching_texts) == 0:
                    continue
                
                # Calculate sentiment distribution for this keyword
                sentiment_counts = matching_texts[sentiment_column].value_counts()
                total_mentions = len(matching_texts)
                
                # Calculate percentages
                positive_pct = (sentiment_counts.get('positive', 0) / total_mentions) * 100
                negative_pct = (sentiment_counts.get('negative', 0) / total_mentions) * 100
                neutral_pct = (sentiment_counts.get('neutral', 0) / total_mentions) * 100
                
                # Calculate average compound score
                avg_compound = matching_texts['compound'].mean() if 'compound' in matching_texts.columns else 0.0
                
                # Create keyword record
                stats = {
                    'keyword': keyword,
                    'total_mentions': total_mentions,
                    'positive_count': sentiment_counts.get('positive', 0),
                    'negative_count': sentiment_counts.get('negative', 0),
                    'neutral_count': sentiment_counts.get('neutral', 0),
                    'positive_percentage': positive_pct,
                    'negative_percentage': negative_pct,
                    'neutral_percentage': neutral_pct,
                    'average_compound': avg_compound
                }
                
                keyword_stats.append(stats)
            
            result_df = pd.DataFrame(keyword_stats)
            logger.info(f"Analyzed sentiment for {len(result_df)} keywords")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error analyzing keyword sentiment: {e}")
            return pd.DataFrame()
    
    def analyze_volume_trends(self, df: pd.DataFrame, time_column: str = 'created_at',
                            interval: str = None) -> pd.DataFrame:
        """
        Analyze volume trends over time.
        
        Args:
            df: DataFrame with timestamp data
            time_column: Column containing timestamps
            interval: Time interval for analysis
            
        Returns:
            DataFrame with volume statistics
        """
        try:
            interval = interval or self.time_interval
            
            # Ensure time column is datetime
            if time_column not in df.columns:
                logger.warning(f"Time column '{time_column}' not found")
                return pd.DataFrame()
            
            df_copy = df.copy()
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])
            
            # Set time column as index for resampling
            df_copy = df_copy.set_index(time_column)
            
            # Define resampling frequency
            freq_map = {
                'minute': '1T',
                'hour': '1H',
                'day': '1D',
                'week': '1W',
                'month': '1M'
            }
            
            freq = freq_map.get(interval, '1H')
            
            # Resample and calculate volume statistics
            volume_stats = []
            
            for time_period, group in df_copy.resample(freq):
                if len(group) == 0:
                    continue
                
                # Calculate volume metrics
                total_items = len(group)
                
                # Calculate rolling average (comparing to previous periods)
                volume_stats.append({
                    'time_period': time_period,
                    'volume': total_items,
                    'interval': interval
                })
            
            result_df = pd.DataFrame(volume_stats)
            
            # Add rolling average of volume
            result_df['volume_rolling_avg'] = result_df['volume'].rolling(window=5, min_periods=1).mean()
            
            # Add volume change percentage
            result_df['volume_change_pct'] = result_df['volume'].pct_change() * 100
            
            logger.info(f"Analyzed volume trends for {len(result_df)} {interval} intervals")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error analyzing volume trends: {e}")
            return pd.DataFrame()
    
    def detect_sentiment_shifts(self, df: pd.DataFrame, sentiment_column: str = 'sentiment',
                              score_column: str = 'compound', time_column: str = 'created_at',
                              threshold: float = 0.1) -> List[Dict]:
        """
        Detect significant shifts in sentiment over time.
        
        Args:
            df: DataFrame with sentiment data
            sentiment_column: Column containing sentiment labels
            score_column: Column containing sentiment scores
            time_column: Column containing timestamps
            threshold: Threshold for detecting significant shifts
            
        Returns:
            List of dictionaries describing sentiment shifts
        """
        try:
            # Get sentiment over time
            time_analysis = self.analyze_sentiment_over_time(df, sentiment_column, time_column)
            
            if len(time_analysis) < 2:
                return []
            
            shifts = []
            
            for i in range(1, len(time_analysis)):
                current = time_analysis.iloc[i]
                previous = time_analysis.iloc[i-1]
                
                # Calculate change in compound score
                score_change = current['average_compound'] - previous['average_compound']
                
                # Check if change exceeds threshold
                if abs(score_change) >= threshold:
                    shift_type = 'positive' if score_change > 0 else 'negative'
                    
                    shift_info = {
                        'timestamp': current['time_period'],
                        'previous_score': previous['average_compound'],
                        'current_score': current['average_compound'],
                        'score_change': score_change,
                        'shift_type': shift_type,
                        'magnitude': abs(score_change),
                        'previous_period': previous['time_period'],
                        'volume_change': current['total_items'] - previous['total_items']
                    }
                    
                    shifts.append(shift_info)
            
            logger.info(f"Detected {len(shifts)} sentiment shifts")
            
            return shifts
            
        except Exception as e:
            logger.error(f"Error detecting sentiment shifts: {e}")
            return []
    
    def calculate_sentiment_velocity(self, df: pd.DataFrame, score_column: str = 'compound',
                                  time_column: str = 'created_at') -> pd.DataFrame:
        """
        Calculate sentiment velocity (rate of change).
        
        Args:
            df: DataFrame with sentiment data
            score_column: Column containing sentiment scores
            time_column: Column containing timestamps
            
        Returns:
            DataFrame with sentiment velocity metrics
        """
        try:
            # Ensure time column is datetime
            if time_column not in df.columns:
                logger.warning(f"Time column '{time_column}' not found")
                return pd.DataFrame()
            
            df_copy = df.copy()
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])
            
            # Sort by time
            df_copy = df_copy.sort_values(time_column)
            
            # Calculate time differences in hours
            df_copy['time_diff'] = df_copy[time_column].diff().dt.total_seconds() / 3600
            df_copy['time_diff'] = df_copy['time_diff'].fillna(0)
            
            # Calculate score differences
            df_copy['score_diff'] = df_copy[score_column].diff()
            df_copy['score_diff'] = df_copy['score_diff'].fillna(0)
            
            # Calculate velocity (score change per hour)
            df_copy['sentiment_velocity'] = np.where(
                df_copy['time_diff'] > 0,
                df_copy['score_diff'] / df_copy['time_diff'],
                0
            )
            
            # Calculate acceleration (change in velocity)
            df_copy['sentiment_acceleration'] = df_copy['sentiment_velocity'].diff()
            df_copy['sentiment_acceleration'] = df_copy['sentiment_acceleration'].fillna(0)
            
            # Select relevant columns
            result_df = df_copy[[time_column, score_column, 'sentiment_velocity', 'sentiment_acceleration']]
            
            logger.info(f"Calculated sentiment velocity for {len(result_df)} records")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating sentiment velocity: {e}")
            return pd.DataFrame()
    
    def get_trend_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get a comprehensive summary of sentiment trends.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Dictionary with trend summary statistics
        """
        try:
            if df.empty:
                return {}
            
            summary = {}
            
            # Overall sentiment distribution
            if 'sentiment' in df.columns:
                sentiment_dist = df['sentiment'].value_counts()
                summary['overall_sentiment'] = {
                    'distribution': sentiment_dist.to_dict(),
                    'percentages': (sentiment_dist / len(df) * 100).to_dict()
                }
            
            # Score statistics
            if 'compound' in df.columns:
                summary['score_statistics'] = {
                    'mean': df['compound'].mean(),
                    'median': df['compound'].median(),
                    'std': df['compound'].std(),
                    'min': df['compound'].min(),
                    'max': df['compound'].max()
                }
            
            # Time-based trends
            if 'created_at' in df.columns:
                time_trends = self.analyze_sentiment_over_time(df)
                if not time_trends.empty:
                    latest_trend = time_trends.iloc[-1]
                    summary['latest_trend'] = {
                        'positive_percentage': latest_trend['positive_percentage'],
                        'negative_percentage': latest_trend['negative_percentage'],
                        'neutral_percentage': latest_trend['neutral_percentage'],
                        'average_compound': latest_trend['average_compound']
                    }
            
            # Volume trends
            if 'created_at' in df.columns:
                volume_trends = self.analyze_volume_trends(df)
                if not volume_trends.empty:
                    summary['volume_trends'] = {
                        'average_volume': volume_trends['volume'].mean(),
                        'latest_volume': volume_trends.iloc[-1]['volume'],
                        'volume_trend': 'increasing' if volume_trends.iloc[-1]['volume'] > volume_trends['volume'].mean() else 'decreasing'
                    }
            
            # Keyword analysis (if keywords are configured)
            if self.keywords and 'text' in df.columns:
                keyword_analysis = self.analyze_keyword_sentiment(df)
                if not keyword_analysis.empty:
                    summary['keyword_sentiment'] = keyword_analysis.to_dict('records')
            
            # Sentiment shifts
            if 'created_at' in df.columns and 'sentiment' in df.columns:
                shifts = self.detect_sentiment_shifts(df)
                summary['sentiment_shifts'] = {
                    'total_shifts': len(shifts),
                    'positive_shifts': len([s for s in shifts if s['shift_type'] == 'positive']),
                    'negative_shifts': len([s for s in shifts if s['shift_type'] == 'negative']),
                    'recent_shifts': shifts[-5:] if shifts else []
                }
            
            logger.info("Generated trend summary")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating trend summary: {e}")
            return {'error': str(e)}
    
    def export_trends(self, df: pd.DataFrame, output_path: str, format: str = 'csv') -> bool:
        """
        Export trend analysis results.
        
        Args:
            df: DataFrame with sentiment data
            output_path: Path to save the results
            format: Export format (csv, json, excel)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate all trend analyses
            rolling_sentiment = self.compute_rolling_sentiment(df)
            time_sentiment = self.analyze_sentiment_over_time(df)
            volume_trends = self.analyze_volume_trends(df)
            keyword_sentiment = self.analyze_keyword_sentiment(df) if self.keywords else pd.DataFrame()
            sentiment_shifts = pd.DataFrame(self.detect_sentiment_shifts(df))
            
            # Create a dictionary with all analyses
            export_data = {
                'rolling_sentiment': rolling_sentiment,
                'time_sentiment': time_sentiment,
                'volume_trends': volume_trends,
                'keyword_sentiment': keyword_sentiment,
                'sentiment_shifts': sentiment_shifts,
                'summary': self.get_trend_summary(df)
            }
            
            if format == 'csv':
                # Save each analysis as separate CSV file
                base_path = output_path.replace('.csv', '')
                
                for name, data in export_data.items():
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        data.to_csv(f"{base_path}_{name}.csv", index=False)
                    elif isinstance(data, dict):
                        pd.Series(data).to_json(f"{base_path}_{name}.json")
            
            elif format == 'json':
                import json
                # Convert DataFrames to dictionaries for JSON export
                json_data = {}
                for name, data in export_data.items():
                    if isinstance(data, pd.DataFrame):
                        json_data[name] = data.to_dict('records')
                    else:
                        json_data[name] = data
                
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)
            
            elif format == 'excel':
                # Save all analyses to separate sheets in Excel file
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    for name, data in export_data.items():
                        if isinstance(data, pd.DataFrame) and not data.empty:
                            data.to_excel(writer, sheet_name=name[:31], index=False)  # Excel sheet names max 31 chars
            
            logger.info(f"Exported trend analysis to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting trends: {e}")
            return False
