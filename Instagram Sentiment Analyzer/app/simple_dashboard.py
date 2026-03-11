"""
Simple Instagram Sentiment Analyzer Dashboard
Works with basic dependencies without ML library issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import sys
from pathlib import Path
from datetime import datetime
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Instagram Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #2E4057;
        color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 2px solid #4A90E2;
    }
    .metric-card h3 {
        color: #4A90E2;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        background-color: #4A90E2;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #357ABD;
    }
</style>
""", unsafe_allow_html=True)

class SimpleSentimentAnalyzer:
    """Simple sentiment analyzer for dashboard"""
    
    def __init__(self):
        self.positive_words = {
            'good', 'great', 'amazing', 'love', 'excellent', 'wonderful', 'fantastic',
            'beautiful', 'awesome', 'perfect', 'best', 'happy', 'blessed', 'grateful',
            'incredible', 'energized', 'motivation', 'perfect', 'love', 'great'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'hate', 'worst', 'disappointed', 'frustrated',
            'boring', 'predictable', 'waste', 'stuck', 'traffic', 'cold', 'disappointed',
            'terrible', 'bad', 'awful', 'boring', 'predictable', 'waste', 'stuck'
        }
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if not isinstance(text, str):
            text = str(text)
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[@#]', '', text)
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return sentiment, confidence, positive_count, negative_count

class SimpleDashboard:
    """Simple dashboard implementation"""
    
    def __init__(self):
        self.analyzer = SimpleSentimentAnalyzer()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">📊 Instagram Sentiment Analyzer</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.title("🎛️ Control Panel")
        
        page = st.sidebar.selectbox(
            "Select Page",
            ["🏠 Home", "📥 Sample Data", "🔧 Analysis", "📊 Dashboard"]
        )
        
        return page
    
    def render_home_page(self):
        """Render home page"""
        st.markdown("## Welcome to Instagram Sentiment Analyzer!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📥 Data Analysis</h3>
                <p>Analyze Instagram posts and comments for sentiment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🔧 Text Processing</h3>
                <p>Clean and preprocess text for accurate analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Visualizations</h3>
                <p>Interactive charts and sentiment insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Features:")
        features = [
            "✅ Real-time sentiment analysis",
            "✅ Interactive visualizations",
            "✅ Text preprocessing",
            "✅ Engagement analysis",
            "✅ Export capabilities"
        ]
        
        for feature in features:
            st.markdown(feature)
        
        st.markdown("### Getting Started:")
        st.markdown("1. Navigate to **Sample Data** to load example data")
        st.markdown("2. Go to **Analysis** to process the text")
        st.markdown("3. View results in **Dashboard**")
    
    def create_sample_data(self):
        """Create sample Instagram data"""
        sample_data = [
            {
                'caption': 'Amazing sunset today! Feeling so blessed and grateful for this beautiful view! #sunset #blessed',
                'likes': 150,
                'comments_count': 25,
                'timestamp': '2024-01-15T18:30:00'
            },
            {
                'caption': 'Terrible service at this restaurant. Waited 2 hours for cold food. Very disappointed!',
                'likes': 45,
                'comments_count': 12,
                'timestamp': '2024-01-14T20:15:00'
            },
            {
                'caption': 'Just an ordinary day at the office. Nothing exciting to report. #work #office',
                'likes': 78,
                'comments_count': 8,
                'timestamp': '2024-01-13T09:00:00'
            },
            {
                'caption': 'Absolutely love my new phone! The camera is incredible and battery life is amazing! #newphone #tech',
                'likes': 230,
                'comments_count': 35,
                'timestamp': '2024-01-12T14:20:00'
            },
            {
                'caption': 'This movie was so boring and predictable. Waste of time and money. #badmovie',
                'likes': 32,
                'comments_count': 6,
                'timestamp': '2024-01-11T21:45:00'
            },
            {
                'caption': 'Great workout session today! Feeling energized and ready to conquer the week! #fitness #motivation',
                'likes': 180,
                'comments_count': 28,
                'timestamp': '2024-01-10T07:30:00'
            },
            {
                'caption': 'Coffee and books - perfect combination for a lazy Sunday morning. #coffeetime #reading',
                'likes': 95,
                'comments_count': 15,
                'timestamp': '2024-01-09T10:15:00'
            },
            {
                'caption': 'Stuck in traffic again. This commute is getting worse every day. #traffic #frustrated',
                'likes': 28,
                'comments_count': 4,
                'timestamp': '2024-01-08T17:45:00'
            }
        ]
        
        return pd.DataFrame(sample_data)
    
    def render_sample_data_page(self):
        """Render sample data page"""
        st.markdown("## 📥 Sample Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Load Sample Instagram Data")
            
            if st.button("🔄 Load Sample Data"):
                with st.spinner("Loading sample data..."):
                    st.session_state.data = self.create_sample_data()
                    st.success("Sample data loaded successfully!")
                    st.dataframe(st.session_state.data)
        
        with col2:
            st.markdown("### ℹ️ Info")
            st.markdown("""
            - Sample Instagram posts with captions
            - Includes likes and comments data
            - Ready for sentiment analysis
            - 8 example posts included
            """)
        
        if st.session_state.data is not None:
            st.markdown("### Data Preview")
            st.dataframe(st.session_state.data)
            
            # Basic statistics
            st.markdown("### Basic Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Posts", len(st.session_state.data))
            with col2:
                st.metric("Total Likes", st.session_state.data['likes'].sum())
            with col3:
                st.metric("Total Comments", st.session_state.data['comments_count'].sum())
    
    def render_analysis_page(self):
        """Render analysis page"""
        st.markdown("## 🔧 Sentiment Analysis")
        
        if st.session_state.data is None:
            st.warning("Please load sample data first!")
            return
        
        st.markdown("### Analyzing Sentiment...")
        
        if st.button("🚀 Run Sentiment Analysis"):
            with st.spinner("Analyzing sentiment..."):
                results = []
                
                for _, row in st.session_state.data.iterrows():
                    sentiment, confidence, pos_count, neg_count = self.analyzer.analyze_sentiment(row['caption'])
                    
                    result = {
                        'caption': row['caption'],
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'positive_words': pos_count,
                        'negative_words': neg_count,
                        'likes': row['likes'],
                        'comments_count': row['comments_count'],
                        'engagement': row['likes'] + row['comments_count']
                    }
                    results.append(result)
                
                st.session_state.analysis_results = pd.DataFrame(results)
                st.success("Sentiment analysis completed!")
                
                # Show results
                st.markdown("### Analysis Results")
                st.dataframe(st.session_state.analysis_results)
                
                # Summary statistics
                sentiment_counts = st.session_state.analysis_results['sentiment'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive Posts", sentiment_counts.get('positive', 0))
                with col2:
                    st.metric("Negative Posts", sentiment_counts.get('negative', 0))
                with col3:
                    st.metric("Neutral Posts", sentiment_counts.get('neutral', 0))
    
    def render_dashboard_page(self):
        """Render comprehensive dashboard"""
        st.markdown("## 📊 Analysis Dashboard")
        
        if st.session_state.analysis_results is None:
            st.warning("Please run sentiment analysis first!")
            return
        
        df = st.session_state.analysis_results
        
        # Key metrics
        st.markdown("### 📈 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Posts", len(df))
        with col2:
            avg_confidence = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        with col3:
            avg_engagement = df['engagement'].mean()
            st.metric("Avg Engagement", f"{avg_engagement:.1f}")
        with col4:
            total_positive = df['positive_words'].sum()
            st.metric("Positive Words", total_positive)
        
        # Sentiment distribution
        st.markdown("### 🎯 Sentiment Distribution")
        
        sentiment_counts = df['sentiment'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.3,
            marker_colors=['#2E8B57', '#DC143C', '#708090']
        )])
        
        fig_pie.update_layout(
            title="Sentiment Distribution",
            font=dict(size=14)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Engagement vs Sentiment
        st.markdown("### 💬 Engagement by Sentiment")
        
        engagement_by_sentiment = df.groupby('sentiment')['engagement'].mean().reset_index()
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=engagement_by_sentiment['sentiment'],
                y=engagement_by_sentiment['engagement'],
                marker_color=['#2E8B57', '#DC143C', '#708090']
            )
        ])
        
        fig_bar.update_layout(
            title="Average Engagement by Sentiment",
            xaxis_title="Sentiment",
            yaxis_title="Average Engagement"
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed results table
        st.markdown("### 📋 Detailed Results")
        
        # Format the dataframe for display
        display_df = df.copy()
        display_df['short_caption'] = display_df['caption'].str[:50] + '...'
        display_df = display_df[['sentiment', 'confidence', 'positive_words', 'negative_words', 'engagement', 'short_caption']]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Export options
        st.markdown("### 📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="sentiment_analysis_results.json",
                mime="application/json"
            )
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        page = self.render_sidebar()
        
        if page == "🏠 Home":
            self.render_home_page()
        elif page == "📥 Sample Data":
            self.render_sample_data_page()
        elif page == "🔧 Analysis":
            self.render_analysis_page()
        elif page == "📊 Dashboard":
            self.render_dashboard_page()

def main():
    """Main function"""
    dashboard = SimpleDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
