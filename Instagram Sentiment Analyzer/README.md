# 📊 Instagram Sentiment Analyzer

A comprehensive AI-powered system for analyzing sentiment in Instagram posts, comments, and engagement data. This project combines web scraping, NLP preprocessing, machine learning, and interactive visualizations to provide deep insights into Instagram content sentiment.

## 🎯 Features

- **📥 Automated Data Collection**: Scrape Instagram posts, comments, and metadata using Instaloader
- **🔧 Advanced NLP Pipeline**: Text cleaning, tokenization, stopword removal, lemmatization, and emoji handling
- **🤖 Multiple ML Models**: Support for Logistic Regression, SVM, Random Forest, and Naive Bayes classifiers
- **📈 Rich Visualizations**: Interactive charts, word clouds, heatmaps, and time-series analysis
- **🌐 Interactive Dashboard**: Streamlit-based web interface for real-time analysis
- **📤 Export Capabilities**: Download results in CSV, JSON, and HTML formats
- **⚙️ Configurable Architecture**: JSON-driven configuration for easy customization

## 🏗️ Project Structure

```
Instagram-Sentiment-Analyzer/
│
├── 📁 data/                      # Data storage
│   ├── raw/                      # Raw scraped data
│   ├── processed/                # Preprocessed data
│   └── visualizations/           # Generated charts
│
├── 📁 src/                       # Core modules
│   ├── scraper.py               # Instagram data scraper
│   ├── preprocess.py            # NLP preprocessing pipeline
│   ├── sentiment_model.py       # ML classification models
│   ├── visualize.py             # Analytics and visualizations
│   └── utils.py                 # Utility functions
│
├── 📁 models/                    # Trained model storage
├── 📁 notebooks/                 # Jupyter notebooks
│   └── example_analysis.ipynb   # Complete workflow example
│
├── 📁 app/                       # Web application
│   └── streamlit_app.py         # Interactive dashboard
│
├── 📄 requirements.txt           # Python dependencies
├── 📄 config.json               # Configuration file
├── 📄 main.py                   # CLI entry point
└── 📄 README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Instagram Sentiment Analyzer"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### 2. Configuration

Edit `config.json` to customize settings:

```json
{
  "data_collection": {
    "max_posts": 100,
    "max_comments_per_post": 50
  },
  "sentiment_model": {
    "model_type": "logistic_regression",
    "embedding_type": "tfidf"
  }
}
```

### 3. Run the Dashboard

```bash
# Launch interactive dashboard
streamlit run app/streamlit_app.py

# Access at http://localhost:8501
```

### 4. Command Line Usage

```bash
# Scrape Instagram profile
python main.py --scrape-profile nasa --max-posts 50

# Scrape hashtag
python main.py --scrape-hashtag travel --model svm

# Use existing data
python main.py --input data.csv --visualize-only

# Train model only
python main.py --input processed_data.csv --train-only --model random_forest
```

## 📖 Usage Examples

### Python API

```python
from src.scraper import InstagramScraper
from src.preprocess import TextPreprocessor
from src.sentiment_model import SentimentClassifier
from src.visualize import SentimentVisualizer
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# 1. Scrape data
scraper = InstagramScraper(config)
posts_data = scraper.get_profile_posts('example_profile', max_posts=100)

# 2. Preprocess text
preprocessor = TextPreprocessor(config)
df = preprocessor.preprocess_dataframe(pd.DataFrame(posts_data), 'caption')

# 3. Train model
classifier = SentimentClassifier(config)
texts = df['processed_text'].tolist()
labels = ['positive', 'negative', 'neutral'] * (len(texts) // 3)  # Example labels
classifier.train_model(texts, labels)

# 4. Analyze sentiment
predictions = classifier.predict(texts)
df['sentiment'] = [pred['predicted_label'] for pred in predictions]

# 5. Visualize results
visualizer = SentimentVisualizer(config)
fig = visualizer.sentiment_distribution_pie(df)
fig.show()
```

### Jupyter Notebook

Open `notebooks/example_analysis.ipynb` for a complete step-by-step analysis workflow.

## 🔧 Configuration Options

### Data Collection
- `max_posts`: Maximum posts to scrape (default: 100)
- `max_comments_per_post`: Comments per post (default: 50)
- `download_videos/pictures`: Media download options
- `save_metadata`: Include metadata in results

### Preprocessing
- `remove_stopwords`: Remove common words (default: true)
- `lemmatize`: Convert words to base form (default: true)
- `remove_urls`: Clean URLs from text (default: true)
- `convert_emojis`: Convert emojis to text (default: true)
- `min_word_length`: Minimum word length (default: 2)

### Model Configuration
- `model_type`: Algorithm choice
  - `logistic_regression`: Fast, interpretable baseline
  - `svm`: Support vector machine
  - `random_forest`: Ensemble method
  - `naive_bayes`: Probabilistic classifier
- `embedding_type`: Text representation
  - `tfidf`: Term frequency-inverse document frequency
  - `count`: Simple word count vectors
- `max_features`: Maximum vocabulary size (default: 10,000)

### Visualization
- `color_scheme`: Custom colors for sentiments
- `figure_size`: Chart dimensions
- `dpi`: Image resolution

## 📊 Data Collection Methods

### 1. Profile-based Scraping
```python
scraper = InstagramScraper(config)
posts = scraper.get_profile_posts('username', max_posts=100)
```

### 2. Hashtag-based Scraping
```python
posts = scraper.get_hashtag_posts('travel', max_posts=100)
```

### 3. Upload Existing Data
- CSV format with columns: `caption`, `likes`, `comments_count`, `timestamp`
- JSON format with post objects

## 🤖 Model Training

### Supported Algorithms

1. **Logistic Regression** (Recommended for beginners)
   - Fast training
   - Interpretable coefficients
   - Good baseline performance

2. **Support Vector Machine**
   - Effective in high-dimensional spaces
   - Works well with TF-IDF features

3. **Random Forest**
   - Handles non-linear relationships
   - Feature importance available
   - Robust to overfitting

4. **Naive Bayes**
   - Simple probabilistic model
   - Fast training and prediction
   - Good for text classification

### Training Process

1. **Text Preprocessing**: Clean and normalize text
2. **Feature Extraction**: Convert text to numerical vectors
3. **Model Training**: Fit classifier on labeled data
4. **Evaluation**: Assess performance with accuracy metrics
5. **Prediction**: Apply model to new data

## 📈 Visualization Types

### 1. Sentiment Distribution
- Pie charts showing sentiment breakdown
- Bar charts for categorical comparison

### 2. Time Series Analysis
- Sentiment trends over time
- Posting patterns and frequency
- Hourly and daily analysis

### 3. Engagement Analysis
- Correlation between sentiment and engagement
- Box plots by sentiment category
- Statistical summaries

### 4. Word Clouds
- Sentiment-specific word clouds
- Hashtag analysis
- Topic visualization

### 5. Heatmaps
- Hashtag sentiment matrix
- Time-of-day patterns
- Correlation matrices

## 🌐 Streamlit Dashboard Features

### Pages
- **🏠 Home**: Overview and getting started
- **📥 Data Collection**: Scrape or upload data
- **🔧 Preprocessing**: Configure text processing
- **🤖 Model Training**: Train and evaluate models
- **📈 Analysis**: View results and insights
- **📊 Dashboard**: Comprehensive analytics

### Interactive Features
- Real-time data processing
- Dynamic chart updates
- Export functionality
- Configuration management

## 🔍 Advanced Features

### 1. Hyperparameter Tuning
```python
results = classifier.hyperparameter_tuning(texts, labels)
print(f"Best accuracy: {results['best_score']:.4f}")
print(f"Best parameters: {results['best_params']}")
```

### 2. Feature Importance
```python
importance = classifier.get_feature_importance(top_n=20)
for sentiment, features in importance.items():
    print(f"{sentiment}: {features[:5]}")
```

### 3. Custom Preprocessing
```python
# Extend TextPreprocessor for custom cleaning
class CustomPreprocessor(TextPreprocessor):
    def clean_text(self, text):
        text = super().clean_text(text)
        # Add custom cleaning steps
        return text
```

## 📝 API Reference

### InstagramScraper
- `get_profile_posts(username, max_posts)`: Scrape user posts
- `get_hashtag_posts(hashtag, max_posts)`: Scrape hashtag posts
- `save_data(data, filename)`: Save scraped data
- `load_data(filepath)`: Load saved data

### TextPreprocessor
- `preprocess_text(text)`: Process single text
- `preprocess_dataframe(df, text_column)`: Process DataFrame
- `extract_hashtags(text)`: Extract hashtags
- `extract_mentions(text)`: Extract mentions

### SentimentClassifier
- `train_model(texts, labels)`: Train classifier
- `predict(texts)`: Make predictions
- `save_model(filepath)`: Save trained model
- `load_model(filepath)`: Load saved model

### SentimentVisualizer
- `sentiment_distribution_pie(df)`: Create pie chart
- `sentiment_timeseries(df)`: Time series plot
- `word_cloud_by_sentiment(df)`: Generate word clouds
- `export_visualizations(df, output_dir)`: Export all charts

## 🛠️ Development

### Adding New Models
```python
# Extend SentimentClassifier
class CustomSentimentClassifier(SentimentClassifier):
    def __init__(self, config):
        super().__init__(config)
        self.supported_models['custom_model'] = CustomModel
```

### Custom Visualizations
```python
# Extend SentimentVisualizer
class CustomVisualizer(SentimentVisualizer):
    def custom_chart(self, df):
        # Create custom visualization
        pass
```

## 🐛 Troubleshooting

### Common Issues

1. **Instagram Login Issues**
   - Instagram may block automated access
   - Use session files for persistent login
   - Respect rate limits

2. **Memory Issues**
   - Reduce `max_posts` for large datasets
   - Use batch processing for big data
   - Clear cache regularly

3. **Model Performance**
   - Increase training data size
   - Try different algorithms
   - Tune hyperparameters

4. **Installation Problems**
   - Update pip: `pip install --upgrade pip`
   - Use virtual environment
   - Check Python version compatibility

### Error Messages

- `Login failed`: Check Instagram credentials
- `No posts found`: Verify profile/hashtag exists
- `Model not trained`: Train model before prediction
- `Configuration error`: Validate JSON syntax

## 📋 Requirements

### Python Version
- Python 3.8 or higher

### Key Dependencies
- `instaloader`: Instagram scraping
- `nltk`, `spacy`: NLP processing
- `scikit-learn`: Machine learning
- `streamlit`: Web dashboard
- `plotly`, `matplotlib`: Visualization
- `pandas`, `numpy`: Data manipulation

### Optional Dependencies
- `tensorflow`, `torch`: Deep learning models
- `cupy`: GPU acceleration

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push branch: `git push origin feature-name`
5. Submit pull request

### Development Guidelines
- Follow PEP 8 style
- Add docstrings to functions
- Include unit tests
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

- Respect Instagram's Terms of Service
- Use scraping responsibly
- Don't overload Instagram servers
- Consider privacy implications
- Educational use only

## 📞 Support

- **Issues**: Report bugs via GitHub Issues
- **Documentation**: Check notebooks and examples
- **Community**: Join discussions in Discussions tab

## 🔄 Updates

### Version History
- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Added Streamlit dashboard
- **v1.2.0**: Enhanced visualization options
- **v1.3.0**: Performance improvements and bug fixes

### Planned Features
- [ ] Real-time sentiment monitoring
- [ ] Advanced topic modeling
- [ ] Multi-language support
- [ ] API endpoint deployment
- [ ] Mobile application

## 🎉 Acknowledgments

- Instagram data via Instaloader
- NLP libraries: NLTK, spaCy
- ML frameworks: scikit-learn
- Visualization: Plotly, Matplotlib
- Web framework: Streamlit

---

**Built with ❤️ for data science and social media analysis**
