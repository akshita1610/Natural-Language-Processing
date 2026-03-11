# 🚀 NLP Text Analysis Pipeline

A comprehensive, modular NLP pipeline designed for second-year Computer Science students. This pipeline demonstrates industry-level text processing capabilities while maintaining beginner-friendly implementation.

## 📋 Features

### 🔄 Data Ingestion
- **Multiple Sources**: Text files (.txt, .md), CSV datasets
- **Smart Filtering**: Duplicate removal, length validation
- **Flexible Storage**: CSV/JSON output formats

### 🔧 Text Preprocessing
- **Complete Cleaning**: Lowercasing, punctuation/number handling
- **Tokenization**: NLTK-based with configurable options
- **Normalization**: Stopword removal, lemmatization/stemming
- **Quality Control**: Word length filtering, empty text handling

### 📊 Feature Extraction
- **Classical Methods**: TF-IDF, Bag-of-Words
- **Word Embeddings**: Word2Vec training and document vectors
- **Topic Modeling**: LDA and NMF for topic discovery
- **Flexible Architecture**: Easy to extend with new methods

### 🎯 NLP Tasks
- **Sentiment Analysis**: 
  - VADER (rule-based)
  - TextBlob (machine learning)
  - Optional transformer models (DistilBERT)
  - Ensemble voting system
- **Text Classification**:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machines
  - Random Forest
  - Automatic hyperparameter tuning

### 📈 Evaluation & Metrics
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Cross-Validation**: Robust performance estimation
- **Confusion Matrices**: Detailed error analysis
- **Model Comparison**: Automatic best model selection

### 📊 Visualization
- **Text Analysis**: Word clouds, frequency distributions
- **Sentiment Analysis**: Score distributions, method comparisons
- **Model Performance**: Comparative bar charts, metrics plots
- **Interactive Dashboards**: Plotly-based visualizations

### 📝 Reporting
- **Markdown Reports**: Comprehensive analysis summaries
- **JSON Exports**: Machine-readable results
- **CSV Outputs**: Spreadsheet-friendly data

## 🏗️ Architecture

```
pipeline/
├── ingestion/          # Data loading from multiple sources
├── preprocessing/      # Text cleaning and normalization
├── features/          # Feature extraction and embedding
├── models/            # Sentiment analysis and classification
├── evaluation/        # Metrics and performance evaluation
└── visualization/     # Plotting and interactive charts
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Text Analysis Pipeline"

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your text data in the `data/raw/` directory:
- Text files: `.txt`, `.md` files
- CSV files: with text column (configurable in `config.yaml`)

### 3. Run Pipeline

```bash
python main.py
```

### 4. View Results

- **Reports**: `output/reports/pipeline_report.md`
- **Visualizations**: `output/visualizations/`
- **Data**: `output/processed_data.csv`, `output/sentiment_results.csv`
- **Models**: `output/models/`

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Data sources
data:
  input_sources:
    - type: "text_files"
      path: "data/raw/"
    - type: "csv"
      path: "data/raw/"
      text_column: "text"

# Preprocessing settings
preprocessing:
  lowercase: true
  remove_stopwords: true
  lemmatize: true

# Feature extraction
features:
  tfidf:
    max_features: 5000
    ngram_range: [1, 2]

# Models to use
models:
  sentiment:
    vader: true
    transformer:
      enabled: false
  classification:
    algorithms: ["logistic_regression", "naive_bayes"]
```

## 📚 Usage Examples

### Basic Pipeline Execution

```python
from main import NLPPipeline

# Initialize and run full pipeline
pipeline = NLPPipeline()
results = pipeline.run_full_pipeline()
```

### Sentiment Analysis Only

```python
texts = ["I love this product!", "This is terrible."]
results = pipeline.run_sentiment_analysis_only(texts)
print(results[['text', 'vader_sentiment', 'textblob_sentiment']])
```

### Text Classification

```python
texts = ["positive text 1", "negative text 1", "positive text 2"]
labels = ["positive", "negative", "positive"]
results = pipeline.run_text_classification(texts, labels)
```

## 🎓 Educational Value

This pipeline demonstrates:

✅ **Software Engineering Principles**
- Modular architecture
- Configuration-driven design
- Error handling and logging
- Unit testing structure

✅ **NLP Fundamentals**
- Text preprocessing pipeline
- Feature extraction techniques
- Classical vs. modern approaches
- Evaluation methodologies

✅ **Machine Learning Concepts**
- Supervised learning workflows
- Cross-validation
- Hyperparameter tuning
- Model comparison

✅ **Data Science Skills**
- Data ingestion and cleaning
- Statistical analysis
- Visualization techniques
- Report generation

## 📊 Sample Output

### Sentiment Analysis Results
| Text | VADER_Sentiment | TextBlob_Sentiment | Ensemble_Sentiment |
|------|-----------------|-------------------|-------------------|
| "I love this!" | positive | positive | positive |
| "This is awful" | negative | negative | negative |

### Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.85 | 0.84 | 0.85 | 0.84 |
| Naive Bayes | 0.82 | 0.81 | 0.82 | 0.81 |

## 🔧 Advanced Features

### Custom Feature Extraction
```python
# Add custom features
feature_extractor = FeatureExtractor(config)
features = feature_extractor.create_feature_matrix(
    texts, 
    feature_types=['tfidf', 'word2vec', 'topics']
)
```

### Custom Models
```python
# Extend with new classification algorithms
class CustomClassifier(TextClassifier):
    def train_custom_model(self, X, y):
        # Your custom implementation
        pass
```

### Interactive Visualizations
```python
# Create interactive dashboard
fig = visualizer.create_dashboard_summary(results)
fig.show()
```

## 🛠️ Development

### Adding New Components

1. **Create module in appropriate directory**
2. **Follow existing patterns**
3. **Update configuration schema**
4. **Add to main pipeline orchestrator**

### Testing
```bash
# Run tests (when implemented)
python -m pytest tests/
```

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Include error handling

## 📈 Performance Considerations

- **Memory Management**: Batch processing for large datasets
- **Computational Efficiency**: Vectorized operations where possible
- **Scalability**: Modular design allows component replacement
- **Caching**: Save intermediate results for faster re-runs

## 🔍 Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```bash
   python -c "import nltk; nltk.download('all')"
   ```

2. **spaCy Model Missing**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Memory Issues**
   - Reduce `max_features` in TF-IDF
   - Process data in smaller batches
   - Use streaming for large files

4. **Transformers Not Loading**
   - Check internet connection
   - Verify model name in config
   - Consider using VADER/TextBlob only

## 📄 License

This project is educational and may be used under MIT License.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📞 Support

For questions or issues:
- Check troubleshooting section
- Review configuration options
- Examine log files (`pipeline.log`)

---

**Perfect for**: CS students learning NLP, portfolio projects, data science interviews, and understanding production ML pipelines! 🎯
