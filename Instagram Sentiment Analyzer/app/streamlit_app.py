"""
Instagram Sentiment Analyzer Streamlit Dashboard
Interactive web interface for sentiment analysis of Instagram data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import sys
from pathlib import Path
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scraper import InstagramScraper
from preprocess import TextPreprocessor
from sentiment_model import SentimentClassifier
from visualize import SentimentVisualizer

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
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDashboard:
    """
    Streamlit dashboard for Instagram sentiment analysis
    """
    
    def __init__(self):
        """
        Initialize the dashboard
        """
        self.load_config()
        self.initialize_session_state()
        
    def load_config(self):
        """Load configuration from file"""
        try:
            config_path = Path(__file__).parent.parent / "config.json"
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            self.config = {}
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'scraper' not in st.session_state:
            st.session_state.scraper = None
        if 'preprocessor' not in st.session_state:
            st.session_state.preprocessor = None
        if 'classifier' not in st.session_state:
            st.session_state.classifier = None
        if 'visualizer' not in st.session_state:
            st.session_state.visualizer = None
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">📊 Instagram Sentiment Analyzer</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.title("🎛️ Control Panel")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Select Page",
            ["🏠 Home", "📥 Data Collection", "🔧 Preprocessing", 
             "🤖 Model Training", "📈 Analysis", "📊 Dashboard"]
        )
        
        return page
    
    def render_home_page(self):
        """Render home page"""
        st.markdown("## Welcome to Instagram Sentiment Analyzer!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📥 Data Collection</h3>
                <p>Scrape Instagram posts, comments, and metadata using Instaloader</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🔧 Text Processing</h3>
                <p>Clean, tokenize, and preprocess text for sentiment analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 ML Models</h3>
                <p>Train sentiment classifiers using various algorithms</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Features:")
        features = [
            "✅ Automated Instagram data scraping",
            "✅ Advanced NLP preprocessing pipeline",
            "✅ Multiple ML algorithms (Logistic Regression, SVM, Random Forest)",
            "✅ Interactive visualizations and analytics",
            "✅ Real-time sentiment prediction",
            "✅ Export capabilities (CSV, JSON)"
        ]
        
        for feature in features:
            st.markdown(feature)
        
        st.markdown("### Getting Started:")
        st.markdown("1. Navigate to **Data Collection** to scrape Instagram data")
        st.markdown("2. Go to **Preprocessing** to clean and process text")
        st.markdown("3. Train your sentiment model in **Model Training**")
        st.markdown("4. View comprehensive analysis in **Dashboard**")
    
    def render_data_collection_page(self):
        """Render data collection page"""
        st.markdown("## 📥 Data Collection")
        
        # Initialize scraper
        if st.session_state.scraper is None:
            st.session_state.scraper = InstagramScraper(self.config)
        
        # Collection method
        method = st.radio(
            "Collection Method",
            ["Profile Posts", "Hashtag Posts", "Upload Data"]
        )
        
        if method == "Profile Posts":
            self.render_profile_collection()
        elif method == "Hashtag Posts":
            self.render_hashtag_collection()
        else:
            self.render_upload_section()
    
    def render_profile_collection(self):
        """Render profile-based data collection"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            profile_name = st.text_input("Instagram Profile Name", placeholder="e.g., nasa")
            max_posts = st.slider("Maximum Posts", 10, 500, 100)
            
            if st.button("🔍 Scrape Profile Posts"):
                if profile_name:
                    with st.spinner(f"Scraping posts from @{profile_name}..."):
                        try:
                            posts_data = st.session_state.scraper.get_profile_posts(
                                profile_name, max_posts
                            )
                            
                            if posts_data:
                                st.session_state.data = pd.DataFrame(posts_data)
                                st.success(f"Successfully scraped {len(posts_data)} posts!")
                                st.dataframe(st.session_state.data.head())
                            else:
                                st.error("No posts found or error occurred")
                        except Exception as e:
                            st.error(f"Error scraping profile: {e}")
                else:
                    st.warning("Please enter a profile name")
        
        with col2:
            st.markdown("### ℹ️ Info")
            st.markdown("""
            - Enter the Instagram username without @
            - Maximum 500 posts per request
            - Includes captions, comments, likes, and metadata
            - Rate limiting applied automatically
            """)
    
    def render_hashtag_collection(self):
        """Render hashtag-based data collection"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            hashtag = st.text_input("Hashtag", placeholder="e.g., travel")
            max_posts = st.slider("Maximum Posts", 10, 500, 100)
            
            if st.button("🔍 Scrape Hashtag Posts"):
                if hashtag:
                    with st.spinner(f"Scraping posts from #{hashtag}..."):
                        try:
                            posts_data = st.session_state.scraper.get_hashtag_posts(
                                hashtag, max_posts
                            )
                            
                            if posts_data:
                                st.session_state.data = pd.DataFrame(posts_data)
                                st.success(f"Successfully scraped {len(posts_data)} posts!")
                                st.dataframe(st.session_state.data.head())
                            else:
                                st.error("No posts found or error occurred")
                        except Exception as e:
                            st.error(f"Error scraping hashtag: {e}")
                else:
                    st.warning("Please enter a hashtag")
        
        with col2:
            st.markdown("### ℹ️ Info")
            st.markdown("""
            - Enter hashtag without # symbol
            - Scrapes recent posts with this hashtag
            - Includes engagement metrics
            - Limited by Instagram's API restrictions
            """)
    
    def render_upload_section(self):
        """Render data upload section"""
        st.markdown("### Upload Existing Data")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'json'],
            help="Upload CSV or JSON file with Instagram data"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.data = pd.read_csv(uploaded_file)
                else:
                    with open(uploaded_file.name, 'r') as f:
                        data = json.load(f)
                    st.session_state.data = pd.DataFrame(data)
                
                st.success(f"Data loaded successfully! {len(st.session_state.data)} rows")
                st.dataframe(st.session_state.data.head())
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    def render_preprocessing_page(self):
        """Render preprocessing page"""
        st.markdown("## 🔧 Text Preprocessing")
        
        if st.session_state.data is None:
            st.warning("Please collect data first!")
            return
        
        # Initialize preprocessor
        if st.session_state.preprocessor is None:
            st.session_state.preprocessor = TextPreprocessor(self.config)
        
        # Preprocessing options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Preprocessing Options")
            remove_stopwords = st.checkbox("Remove Stopwords", value=True)
            lemmatize = st.checkbox("Lemmatize", value=True)
            remove_urls = st.checkbox("Remove URLs", value=True)
            remove_mentions = st.checkbox("Remove Mentions", value=True)
            convert_emojis = st.checkbox("Convert Emojis to Text", value=True)
        
        with col2:
            st.markdown("### Data Preview")
            if st.session_state.data is not None:
                st.write(f"Total posts: {len(st.session_state.data)}")
                text_column = st.selectbox(
                    "Text Column",
                    st.session_state.data.columns,
                    index=list(st.session_state.data.columns).index('caption') if 'caption' in st.session_state.data.columns else 0
                )
        
        if st.button("🔧 Preprocess Text"):
            with st.spinner("Preprocessing text..."):
                try:
                    # Update config with user choices
                    self.config["preprocessing"].update({
                        "remove_stopwords": remove_stopwords,
                        "lemmatize": lemmatize,
                        "remove_urls": remove_urls,
                        "remove_mentions": remove_mentions,
                        "convert_emojis": convert_emojis
                    })
                    
                    # Reinitialize preprocessor with new config
                    st.session_state.preprocessor = TextPreprocessor(self.config)
                    
                    # Preprocess data
                    processed_data = st.session_state.preprocessor.preprocess_dataframe(
                        st.session_state.data, text_column
                    )
                    
                    st.session_state.processed_data = processed_data
                    st.success("Preprocessing completed!")
                    
                    # Show results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Original Text")
                        st.text(st.session_state.data[text_column].iloc[0])
                    
                    with col2:
                        st.markdown("### Processed Text")
                        st.text(processed_data['processed_text'].iloc[0])
                    
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
    
    def render_model_training_page(self):
        """Render model training page"""
        st.markdown("## 🤖 Model Training")
        
        if st.session_state.processed_data is None:
            st.warning("Please preprocess data first!")
            return
        
        # Initialize classifier
        if st.session_state.classifier is None:
            st.session_state.classifier = SentimentClassifier(self.config)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Model Configuration")
            
            # Model selection
            model_type = st.selectbox(
                "Model Type",
                ["logistic_regression", "svm", "random_forest", "naive_bayes"]
            )
            
            # Embedding type
            embedding_type = st.selectbox(
                "Embedding Type",
                ["tfidf", "count"]
            )
            
            # Training data
            st.markdown("### Training Data")
            st.write("For demo purposes, we'll use simulated sentiment labels.")
            st.info("In production, you would need labeled data or use a pre-trained model.")
            
            if st.button("🚀 Train Model"):
                with st.spinner("Training model..."):
                    try:
                        # Update config
                        self.config["sentiment_model"].update({
                            "model_type": model_type,
                            "embedding_type": embedding_type
                        })
                        
                        # Reinitialize classifier
                        st.session_state.classifier = SentimentClassifier(self.config)
                        
                        # Create sample labels for demo
                        texts = st.session_state.processed_data['processed_text'].tolist()
                        
                        # Generate sample sentiment labels (in real scenario, you'd have actual labels)
                        np.random.seed(42)
                        sample_labels = np.random.choice(
                            ['positive', 'negative', 'neutral'], 
                            size=len(texts), 
                            p=[0.4, 0.3, 0.3]
                        ).tolist()
                        
                        # Train model
                        metrics = st.session_state.classifier.train_model(texts, sample_labels)
                        
                        st.session_state.model_trained = True
                        
                        # Display results
                        st.success("Model trained successfully!")
                        st.json(metrics)
                        
                        # Test predictions
                        test_texts = texts[:5]
                        predictions = st.session_state.classifier.predict(test_texts)
                        
                        st.markdown("### Sample Predictions")
                        for i, (text, pred) in enumerate(zip(test_texts, predictions)):
                            st.write(f"**Text {i+1}:** {text[:100]}...")
                            st.write(f"**Sentiment:** {pred['predicted_label']} (Confidence: {pred.get('confidence', 'N/A')})")
                            st.write("---")
                        
                    except Exception as e:
                        st.error(f"Error training model: {e}")
        
        with col2:
            st.markdown("### ℹ️ Model Info")
            model_info = {
                "logistic_regression": "Fast, interpretable baseline model",
                "svm": "Good for high-dimensional data",
                "random_forest": "Robust ensemble method",
                "naive_bayes": "Simple probabilistic model"
            }
            
            st.write(model_info.get(model_type, ""))
            
            st.markdown("### Embedding Info")
            embedding_info = {
                "tfidf": "Term frequency-inverse document frequency",
                "count": "Simple word count vectors"
            }
            
            st.write(embedding_info.get(embedding_type, ""))
    
    def render_analysis_page(self):
        """Render analysis page"""
        st.markdown("## 📈 Sentiment Analysis")
        
        if not st.session_state.model_trained:
            st.warning("Please train a model first!")
            return
        
        # Initialize visualizer
        if st.session_state.visualizer is None:
            st.session_state.visualizer = SentimentVisualizer(self.config)
        
        # Add sentiment predictions to data
        if st.session_state.processed_data is not None:
            texts = st.session_state.processed_data['processed_text'].tolist()
            predictions = st.session_state.classifier.predict(texts)
            
            # Add predictions to dataframe
            analysis_data = st.session_state.processed_data.copy()
            analysis_data['sentiment'] = [pred['predicted_label'] for pred in predictions]
            analysis_data['confidence'] = [pred.get('confidence', 0) for pred in predictions]
            
            # Display summary statistics
            st.markdown("### 📊 Summary Statistics")
            summary = st.session_state.visualizer.create_dashboard_summary(analysis_data)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Posts", summary['total_posts'])
            with col2:
                st.metric("Avg Engagement", f"{summary['avg_engagement']:.2f}")
            with col3:
                st.metric("Posts/Day", f"{summary['posting_frequency']['posts_per_day']:.1f}")
            with col4:
                st.metric("Peak Hour", f"{summary['peak_hours'][0] if summary['peak_hours'] else 'N/A'}:00")
            
            # Sentiment distribution
            st.markdown("### 🎯 Sentiment Distribution")
            fig1 = st.session_state.visualizer.sentiment_distribution_pie(analysis_data)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Time series
            if 'timestamp' in analysis_data.columns:
                st.markdown("### 📅 Sentiment Trends")
                fig2 = st.session_state.visualizer.sentiment_timeseries(analysis_data)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Engagement correlation
            if 'engagement_rate' in analysis_data.columns:
                st.markdown("### 💬 Engagement vs Sentiment")
                fig3 = st.session_state.visualizer.engagement_sentiment_correlation(analysis_data)
                st.plotly_chart(fig3, use_container_width=True)
            
            # Top hashtags
            if summary['top_hashtags']:
                st.markdown("### 🔝 Top Hashtags")
                hashtag_df = pd.DataFrame(summary['top_hashtags'])
                st.dataframe(hashtag_df)
    
    def render_dashboard_page(self):
        """Render comprehensive dashboard"""
        st.markdown("## 📊 Comprehensive Dashboard")
        
        if not st.session_state.model_trained or st.session_state.processed_data is None:
            st.warning("Please complete data collection, preprocessing, and model training first!")
            return
        
        # Get analysis data
        texts = st.session_state.processed_data['processed_text'].tolist()
        predictions = st.session_state.classifier.predict(texts)
        
        analysis_data = st.session_state.processed_data.copy()
        analysis_data['sentiment'] = [pred['predicted_label'] for pred in predictions]
        analysis_data['confidence'] = [pred.get('confidence', 0) for pred in predictions]
        
        # Initialize visualizer
        if st.session_state.visualizer is None:
            st.session_state.visualizer = SentimentVisualizer(self.config)
        
        # Create dashboard layout
        st.markdown("### 📈 Real-time Analytics")
        
        # Key metrics
        summary = st.session_state.visualizer.create_dashboard_summary(analysis_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive Posts", summary['sentiment_distribution'].get('positive', 0))
        with col2:
            st.metric("Negative Posts", summary['sentiment_distribution'].get('negative', 0))
        with col3:
            st.metric("Neutral Posts", summary['sentiment_distribution'].get('neutral', 0))
        
        # Interactive charts
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "📅 Trends", "🔥 Hashtags", "⏰ Time Analysis"])
        
        with tab1:
            fig1 = st.session_state.visualizer.sentiment_distribution_pie(analysis_data)
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            if 'timestamp' in analysis_data.columns:
                fig2 = st.session_state.visualizer.sentiment_timeseries(analysis_data)
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            fig3 = st.session_state.visualizer.hashtag_sentiment_heatmap(analysis_data)
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab4:
            fig4 = st.session_state.visualizer.sentiment_by_hour(analysis_data)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Export options
        st.markdown("### 📤 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download CSV"):
                csv_data = analysis_data.to_csv(index=False)
                st.download_button(
                    label="Download analysis_results.csv",
                    data=csv_data,
                    file_name="instagram_sentiment_analysis.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📥 Download JSON"):
                json_data = analysis_data.to_json(orient='records')
                st.download_button(
                    label="Download analysis_results.json",
                    data=json_data,
                    file_name="instagram_sentiment_analysis.json",
                    mime="application/json"
                )
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        page = self.render_sidebar()
        
        if page == "🏠 Home":
            self.render_home_page()
        elif page == "📥 Data Collection":
            self.render_data_collection_page()
        elif page == "🔧 Preprocessing":
            self.render_preprocessing_page()
        elif page == "🤖 Model Training":
            self.render_model_training_page()
        elif page == "📈 Analysis":
            self.render_analysis_page()
        elif page == "📊 Dashboard":
            self.render_dashboard_page()

def main():
    """Main function to run the Streamlit app"""
    dashboard = StreamlitDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
