"""
Data Visualization Module
Creates comprehensive visualizations for NLP pipeline results
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DataVisualizer:
    """
    Comprehensive visualization system for NLP pipeline results
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualizer with configuration
        
        Args:
            config: Configuration dictionary containing visualization settings
        """
        self.config = config.get('visualization', {})
        self.save_plots = self.config.get('save_plots', True)
        self.plot_format = self.config.get('plot_format', 'png')
        self.dpi = self.config.get('dpi', 300)
        
        # Create output directory if saving plots
        if self.save_plots:
            self.output_dir = "output/visualizations"
            os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_word_frequency(self, texts: List[str], top_n: int = 20,
                           title: str = "Top Word Frequencies") -> plt.Figure:
        """
        Plot word frequency distribution
        
        Args:
            texts: List of texts to analyze
            top_n: Number of top words to display
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Combine all texts and count words
        all_text = ' '.join(texts)
        words = all_text.split()
        word_counts = pd.Series(words).value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        word_counts.plot(kind='bar')
        plt.title(title)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/word_frequency.{self.plot_format}", 
                       dpi=self.dpi, bbox_inches='tight')
        
        return plt.gcf()
    
    def create_wordcloud(self, texts: List[str], title: str = "Word Cloud",
                        max_words: int = 100) -> plt.Figure:
        """
        Create word cloud visualization
        
        Args:
            texts: List of texts to analyze
            title: Word cloud title
            max_words: Maximum number of words to display
            
        Returns:
            Matplotlib figure
        """
        # Combine all texts
        all_text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            max_words=max_words,
            background_color='white',
            colormap='viridis'
        ).generate(all_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/wordcloud.{self.plot_format}", 
                       dpi=self.dpi, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_sentiment_distribution(self, sentiment_df: pd.DataFrame,
                                   sentiment_column: str = 'vader_sentiment') -> plt.Figure:
        """
        Plot sentiment distribution
        
        Args:
            sentiment_df: DataFrame with sentiment results
            sentiment_column: Column name for sentiment labels
            
        Returns:
            Matplotlib figure
        """
        sentiment_counts = sentiment_df[sentiment_column].value_counts()
        
        plt.figure(figsize=(10, 6))
        
        # Create subplot for bar chart and pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        sentiment_counts.plot(kind='bar', ax=ax1, color=['#ff9999', '#66b3ff', '#99ff99'])
        ax1.set_title('Sentiment Distribution (Bar Chart)')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title('Sentiment Distribution (Pie Chart)')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/sentiment_distribution.{self.plot_format}", 
                       dpi=self.dpi, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_sentiment_scores(self, sentiment_df: pd.DataFrame) -> plt.Figure:
        """
        Plot sentiment score distributions
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            
        Returns:
            Matplotlib figure
        """
        score_columns = [col for col in sentiment_df.columns 
                        if any(x in col for x in ['compound', 'polarity', 'confidence'])]
        
        if not score_columns:
            logger.warning("No sentiment score columns found")
            return None
        
        fig, axes = plt.subplots(len(score_columns), 2, figsize=(15, 5 * len(score_columns)))
        if len(score_columns) == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(score_columns):
            # Histogram
            axes[i, 0].hist(sentiment_df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i, 0].set_title(f'{col} Distribution')
            axes[i, 0].set_xlabel('Score')
            axes[i, 0].set_ylabel('Frequency')
            
            # Box plot
            axes[i, 1].boxplot(sentiment_df[col].dropna())
            axes[i, 1].set_title(f'{col} Box Plot')
            axes[i, 1].set_ylabel('Score')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/sentiment_scores.{self.plot_format}", 
                       dpi=self.dpi, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_model_performance(self, model_results: Dict[str, Dict[str, Any]]) -> plt.Figure:
        """
        Plot model performance comparison
        
        Args:
            model_results: Dictionary of model evaluation results
            
        Returns:
            Matplotlib figure
        """
        # Extract metrics for comparison
        models = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for model_name, results in model_results.items():
            models.append(model_name)
            accuracies.append(results.get('accuracy', 0))
            precisions.append(results.get('precision', 0))
            recalls.append(results.get('recall', 0))
            f1_scores.append(results.get('f1', 0))
        
        # Create performance comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        ax1.bar(models, accuracies, color='skyblue')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1)
        
        # Precision
        ax2.bar(models, precisions, color='lightgreen')
        ax2.set_title('Model Precision Comparison')
        ax2.set_ylabel('Precision')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        # Recall
        ax3.bar(models, recalls, color='salmon')
        ax3.set_title('Model Recall Comparison')
        ax3.set_ylabel('Recall')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1)
        
        # F1-Score
        ax4.bar(models, f1_scores, color='gold')
        ax4.set_title('Model F1-Score Comparison')
        ax4.set_ylabel('F1-Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/model_performance.{self.plot_format}", 
                       dpi=self.dpi, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_confusion_matrix_heatmap(self, cm: np.ndarray, 
                                     class_names: Optional[List[str]] = None,
                                     title: str = "Confusion Matrix") -> plt.Figure:
        """
        Plot confusion matrix as heatmap
        
        Args:
            cm: Confusion matrix
            class_names: Names of classes
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/confusion_matrix.{self.plot_format}", 
                       dpi=self.dpi, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_topic_distribution(self, topic_features: np.ndarray, 
                               top_words: List[List[str]]) -> plt.Figure:
        """
        Plot topic distribution and top words
        
        Args:
            topic_features: Topic feature matrix
            top_words: List of top words for each topic
            
        Returns:
            Matplotlib figure
        """
        n_topics = topic_features.shape[1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Topic distribution
        topic_sums = topic_features.sum(axis=0)
        ax1.bar(range(n_topics), topic_sums)
        ax1.set_title('Topic Distribution Across Documents')
        ax1.set_xlabel('Topic Number')
        ax1.set_ylabel('Total Weight')
        
        # Top words for top 5 topics
        top_n_topics = min(5, n_topics)
        for i in range(top_n_topics):
            words_str = ', '.join(top_words[i][:5])  # Top 5 words
            ax2.text(0.1, 0.9 - i*0.15, f"Topic {i}: {words_str}", 
                    transform=ax2.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Top Words by Topic')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/topic_analysis.{self.plot_format}", 
                       dpi=self.dpi, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_text_length_distribution(self, texts: List[str]) -> plt.Figure:
        """
        Plot text length distribution
        
        Args:
            texts: List of texts
            
        Returns:
            Matplotlib figure
        """
        text_lengths = [len(text.split()) for text in texts]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(text_lengths, bins=30, alpha=0.7, edgecolor='black', color='purple')
        ax1.set_title('Text Length Distribution')
        ax1.set_xlabel('Number of Words')
        ax1.set_ylabel('Frequency')
        
        # Box plot
        ax2.boxplot(text_lengths)
        ax2.set_title('Text Length Box Plot')
        ax2.set_ylabel('Number of Words')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/text_length_distribution.{self.plot_format}", 
                       dpi=self.dpi, bbox_inches='tight')
        
        return plt.gcf()
    
    def create_interactive_sentiment_plot(self, sentiment_df: pd.DataFrame) -> go.Figure:
        """
        Create interactive sentiment plot using Plotly
        
        Args:
            sentiment_df: DataFrame with sentiment results
            
        Returns:
            Plotly figure
        """
        # Create interactive scatter plot of sentiment scores
        if 'vader_compound' in sentiment_df.columns and 'textblob_polarity' in sentiment_df.columns:
            fig = px.scatter(
                sentiment_df, 
                x='vader_compound', 
                y='textblob_polarity',
                color='vader_sentiment',
                hover_data=['text'],
                title='VADER vs TextBlob Sentiment Scores',
                labels={
                    'vader_compound': 'VADER Compound Score',
                    'textblob_polarity': 'TextBlob Polarity',
                    'vader_sentiment': 'VADER Sentiment'
                }
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
        else:
            # Fallback to simple distribution plot
            sentiment_counts = sentiment_df['vader_sentiment'].value_counts()
            fig = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title='Sentiment Distribution',
                labels={'x': 'Sentiment', 'y': 'Count'}
            )
        
        if self.save_plots:
            fig.write_html(f"{self.output_dir}/interactive_sentiment.html")
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                              title: str = "Feature Importance") -> plt.Figure:
        """
        Plot feature importance
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        features, importance = zip(*sorted_features[:20])  # Top 20 features
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if x < 0 else 'green' for x in importance]
        plt.barh(range(len(features)), importance, color=colors, alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()  # Highest importance at top
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/feature_importance.{self.plot_format}", 
                       dpi=self.dpi, bbox_inches='tight')
        
        return plt.gcf()
    
    def create_dashboard_summary(self, evaluation_results: Dict[str, Any]) -> go.Figure:
        """
        Create dashboard summary with multiple plots
        
        Args:
            evaluation_results: Dictionary containing evaluation results
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy', 'Sentiment Distribution', 
                          'Text Length Distribution', 'Topic Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Model accuracy
        if 'model_comparison' in evaluation_results:
            comparison = evaluation_results['model_comparison']
            if 'detailed_comparison' in comparison:
                df = pd.DataFrame(comparison['detailed_comparison']).T
                fig.add_trace(
                    go.Bar(x=df.index, y=df['accuracy'], name='Accuracy'),
                    row=1, col=1
                )
        
        # Add other plots based on available data
        # This is a placeholder - implement based on specific data structure
        
        fig.update_layout(
            title_text="NLP Pipeline Dashboard",
            showlegend=False,
            height=800
        )
        
        if self.save_plots:
            fig.write_html(f"{self.output_dir}/dashboard.html")
        
        return fig
