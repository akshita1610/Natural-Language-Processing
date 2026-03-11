# 🌟 Sentiment Analysis Pipeline

A modular, production-ready sentiment analysis pipeline designed for second-year Computer Science students but structured like real graduate-level data engineering projects. This pipeline can process text from social media streams, audio transcripts, or custom text sources to detect sentiment trends in real-time or batch mode.

## 🎯 Features

### **Data Ingestion**
- **Twitter/X API**: Stream or search tweets with advanced filtering
- **Reddit API**: Collect posts and comments from subreddits using PRAW
- **Audio Processing**: Transcribe audio files using OpenAI's Whisper
- **Configurable Sources**: Easy switching between data sources via configuration

### **Text Preprocessing**
- Tokenization and normalization
- Stopword removal and punctuation cleaning
- Emoji handling and conversion
- Lemmatization for better analysis
- Feature extraction (character count, word count, etc.)

### **Sentiment Analysis**
- **VADER**: Rule-based sentiment analysis (great for social media)
- **TextBlob**: Pattern-based sentiment analysis
- **Transformers**: Deep learning models (optional, requires additional setup)
- **Ensemble Methods**: Combine multiple models for better accuracy
- **Confidence Scoring**: Reliability metrics for each prediction

### **Storage Layer**
- **CSV**: Simple file-based storage
- **SQLite**: Local database for intermediate complexity
- **PostgreSQL**: Production-ready database (optional)
- **Metadata Tracking**: Timestamps, sources, and analysis metadata

### **Trend Analysis**
- Rolling averages and moving windows
- Time-based sentiment tracking
- Volume analysis and pattern detection
- Keyword-level sentiment tracking
- Sentiment shift detection
- Velocity and acceleration metrics

### **Visualization**
- **Static Charts**: Matplotlib/Seaborn for publication-quality plots
- **Interactive Dashboards**: Plotly for web-based interactive visualizations
- **Comprehensive Reports**: HTML reports with embedded charts
- **Multiple Chart Types**: Time series, heatmaps, distribution plots

## 🏗️ Architecture

```
/pipeline
    /ingestion          # Data collection modules
        ├── __init__.py
        ├── twitter_ingestion.py
        ├── reddit_ingestion.py
        └── audio_ingestion.py
    /preprocessing      # Text cleaning and normalization
        ├── __init__.py
        └── text_preprocessor.py
    /sentiment          # Sentiment analysis models
        ├── __init__.py
        └── sentiment_analyzer.py
    /storage            # Data persistence layer
        ├── __init__.py
        └── data_storage.py
    /trends             # Trend analysis and tracking
        ├── __init__.py
        └── trend_analyzer.py
    /visualization      # Charts and dashboards
        ├── __init__.py
        └── visualizer.py
    /trends             # Trend analysis module
        ├── __init__.py
        └── trend_analyzer.py
config.yaml            # Main configuration file
main.py                # Pipeline entry point
requirements.txt       # Python dependencies
README.md             # This file
```

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone or download the project
cd "Sentiment Analysis Pipeline"

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (automatic on first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### 2. **Configuration**

Edit `config.yaml` to set up your data sources and preferences:

```yaml
# Example: Twitter configuration
data_source:
  type: "twitter"
  twitter:
    bearer_token: "YOUR_TWITTER_BEARER_TOKEN"
    query: "sentiment OR emotions OR mood -is:retweet"
    max_tweets: 100

# Example: Sentiment analysis
sentiment:
  model: "vader"  # Options: vader, textblob, transformer, all
  threshold_positive: 0.05
  threshold_negative: -0.05

# Example: Storage
storage:
  type: "sqlite"  # Options: csv, sqlite, postgresql
  sqlite:
    database_file: "sentiment_analysis.db"
```

### 3. **Run the Pipeline**

```bash
# Batch mode (default)
python main.py

# Streaming mode
python main.py --mode stream

# Specific data source
python main.py --source reddit

# Check pipeline status
python main.py --status

# Custom configuration
python main.py --config my_config.yaml
```

## 📊 Usage Examples

### **Twitter Analysis**
```bash
# Configure your Twitter API keys in config.yaml
python main.py --source twitter --mode batch
```

### **Reddit Analysis**
```bash
# Set up Reddit API credentials in config.yaml
python main.py --source reddit
```

### **Audio Transcription**
```bash
# Place audio files in the configured directory
python main.py --source audio
```

### **Real-time Monitoring**
```bash
# Start streaming analysis
python main.py --mode stream --source twitter
```

## 🔧 Configuration Guide

### **Data Sources**

#### Twitter/X
```yaml
data_source:
  type: "twitter"
  twitter:
    bearer_token: "YOUR_BEARER_TOKEN"
    api_key: "YOUR_API_KEY"
    api_secret: "YOUR_API_SECRET"
    access_token: "YOUR_ACCESS_TOKEN"
    access_token_secret: "YOUR_ACCESS_TOKEN_SECRET"
    query: "machine learning OR AI -is:retweet"
    max_tweets: 1000
    language: "en"
```

#### Reddit
```yaml
data_source:
  type: "reddit"
  reddit:
    client_id: "YOUR_CLIENT_ID"
    client_secret: "YOUR_CLIENT_SECRET"
    user_agent: "Sentiment Analysis Pipeline v1.0"
    subreddit: "MachineLearning"
    limit: 500
    sort: "hot"
```

#### Audio
```yaml
data_source:
  type: "audio"
  audio:
    input_dir: "audio_files"
    output_dir: "transcripts"
    model_size: "base"  # tiny, base, small, medium, large
    file_format: "wav"
```

### **Sentiment Models**

#### VADER (Recommended for Social Media)
```yaml
sentiment:
  model: "vader"
  threshold_positive: 0.05
  threshold_negative: -0.05
```

#### TextBlob
```yaml
sentiment:
  model: "textblob"
  threshold_positive: 0.1
  threshold_negative: -0.1
```

#### Transformers (Advanced)
```yaml
sentiment:
  model: "transformer"
  transformer_model: "cardiffnlp/twitter-roberta-base-sentiment"
```

#### Ensemble (All Models)
```yaml
sentiment:
  model: "all"
  threshold_positive: 0.05
  threshold_negative: -0.05
```

### **Storage Options**

#### CSV (Simple)
```yaml
storage:
  type: "csv"
  csv:
    output_file: "sentiment_results.csv"
```

#### SQLite (Intermediate)
```yaml
storage:
  type: "sqlite"
  sqlite:
    database_file: "sentiment_analysis.db"
```

#### PostgreSQL (Production)
```yaml
storage:
  type: "postgresql"
  postgresql:
    host: "localhost"
    port: 5432
    database: "sentiment_analysis"
    username: "postgres"
    password: "password"
```

## 📈 Output and Results

### **Generated Files**
- `sentiment_results.csv` or `.db`: Processed sentiment data
- `visualizations/`: Charts and plots
- `dashboard_*.html`: Interactive dashboard
- `report_*.html`: Comprehensive analysis report
- `pipeline.log`: Detailed execution logs

### **Visualizations**
- Sentiment distribution pie charts
- Sentiment trends over time
- Volume analysis by hour/day
- Keyword sentiment analysis
- Interactive heatmaps
- Comprehensive dashboards

### **Trend Analysis**
- Rolling sentiment averages
- Sentiment velocity and acceleration
- Significant sentiment shifts
- Keyword-level tracking
- Volume trend correlations

## 🎓 Educational Value

This project demonstrates understanding of:

- **Data Engineering**: Pipeline architecture and modular design
- **NLP Fundamentals**: Text preprocessing and sentiment analysis
- **Real-time Processing**: Streaming data handling
- **Database Design**: Multiple storage backends
- **Visualization**: Both static and interactive charts
- **API Integration**: Twitter, Reddit, and ML model APIs
- **Configuration Management**: YAML-based configuration
- **Error Handling**: Robust error management and logging
- **Software Architecture**: Modular, extensible design

Perfect for graduate scheme applications showcasing:
- Junior data engineer capabilities
- Data analysis skills
- Software development best practices
- Understanding of production systems

## 🔍 API Setup Guides

### **Twitter API**
1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
2. Create a new project and app
3. Generate API keys and bearer token
4. Add credentials to `config.yaml`

### **Reddit API**
1. Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps)
2. Click "Create App" → "script"
3. Note client ID and secret
4. Add credentials to `config.yaml`

### **Optional: Transformers Setup**
```bash
# For advanced transformer models
pip install transformers torch
# Models will be downloaded automatically on first use
```

## 🐛 Troubleshooting

### **Common Issues**

#### **Twitter API Errors**
- Check API credentials in config.yaml
- Verify rate limits and permissions
- Ensure bearer token is correctly set

#### **NLTK Data Missing**
```bash
python -c "import nltk; nltk.download('all')"
```

#### **Audio Processing Issues**
- Install ffmpeg for audio processing
- Check audio file formats are supported
- Verify whisper model size availability

#### **Database Connection Errors**
- Check database file permissions
- Verify PostgreSQL connection details
- Ensure database server is running

#### **Visualization Errors**
- Install missing GUI libraries on Linux: `sudo apt-get install python3-tk`
- Check output directory permissions
- Verify matplotlib backend

### **Performance Tips**

#### **Large Datasets**
- Use SQLite or PostgreSQL instead of CSV
- Enable batch processing
- Consider streaming mode for real-time analysis

#### **Memory Usage**
- Process data in smaller batches
- Use streaming mode for continuous analysis
- Clear intermediate results

## 🤝 Contributing

This project is designed to be educational and extensible. Feel free to:

- Add new data sources
- Implement additional preprocessing steps
- Experiment with different sentiment models
- Create new visualization types
- Optimize performance

## 📄 License

This project is provided for educational purposes. Please respect the terms of service of all APIs used (Twitter, Reddit, etc.).

## 🙏 Acknowledgments

- **VADER Sentiment**: Hutto, C.J. & Gilbert, E.E. (2014)
- **TextBlob**: Loria, S. (2018)
- **Whisper**: OpenAI (2023)
- **NLTK**: Bird, S. (2009)
- **PRAW**: Bryce Boe (2016)

---

**Built with ❤️ for educational purposes and graduate scheme applications**
