# Installation Guide

## Quick Start (No Dependencies Required)

The **minimal demo** works with just Python standard library:

```bash
python minimal_demo.py
```

This demonstrates the core sentiment analysis functionality without any external dependencies.

## Full Installation

### Option 1: Automated Setup (Recommended)

```bash
python setup.py
```

This script handles all dependencies and configuration automatically.

### Option 2: Manual Installation

#### Step 1: Install Core Dependencies

```bash
pip install scikit-learn>=1.3.0 pandas>=2.1.0 numpy>=1.24.0
pip install nltk>=3.8.1 spacy>=3.7.0 matplotlib>=3.7.0
pip install seaborn>=0.12.0 plotly>=5.17.0 wordcloud>=1.9.0
pip install streamlit>=1.28.0 emoji>=2.8.0 regex>=2023.10.3
pip install textblob>=0.17.1 tqdm>=4.66.0 python-dotenv>=1.0.0
pip install joblib>=1.3.0 openpyxl>=3.1.0
```

#### Step 2: Download Language Models

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

#### Step 3: Install Optional Packages

```bash
# For Instagram scraping
pip install instaloader>=4.10.0 selenium>=4.15.0 beautifulsoup4>=4.12.0 requests>=2.31.0

# For web framework
pip install flask>=2.3.0

# For deep learning (optional)
pip install tensorflow>=2.14.0 torch>=2.1.0 transformers>=4.35.0
```

## Troubleshooting

### Windows Path Issues

If you encounter path-related errors during installation:

1. **Enable Long Path Support**:
   - Open Group Policy Editor
   - Navigate to: Computer Configuration → Administrative Templates → System → Filesystem
   - Enable "Enable Win32 long paths"
   - Restart your computer

2. **Use PowerShell as Administrator**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### TensorFlow Installation Issues

TensorFlow may fail on Windows due to path length limitations. Solutions:

1. **Skip TensorFlow initially**:
   ```bash
   # Comment out tensorflow line in requirements.txt
   pip install -r requirements.txt --no-deps tensorflow
   ```

2. **Install TensorFlow separately**:
   ```bash
   pip install tensorflow-cpu
   ```

### NumPy/Scikit-learn Compatibility

If you get import errors:

```bash
# Reinstall compatible versions
pip uninstall numpy scikit-learn
pip install numpy==1.24.3 scikit-learn==1.3.0
```

## Verification

### Test Basic Functionality

```bash
python minimal_demo.py
```

### Test ML Components

```bash
python quick_start.py
```

### Test Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Project Structure After Installation

```
Instagram Sentiment Analyzer/
├── data/                    # Created automatically
│   ├── raw/                # Raw scraped data
│   ├── processed/           # Processed data
│   └── visualizations/     # Generated charts
├── models/                  # Trained models
├── cache/                   # Temporary files
├── src/                     # Core modules
├── app/                     # Web interface
├── notebooks/               # Jupyter examples
├── config.json             # Configuration
├── requirements.txt        # Dependencies
├── setup.py               # Setup script
├── minimal_demo.py        # Quick demo
├── quick_start.py         # ML demo
└── main.py               # CLI interface
```

## Next Steps

1. **Run the minimal demo** to verify basic functionality
2. **Install full dependencies** for advanced features
3. **Launch the dashboard** for interactive analysis
4. **Check the README** for detailed usage instructions
5. **Explore notebooks** for step-by-step examples

## Requirements Summary

### Minimum Requirements
- Python 3.8+
- No external dependencies (for minimal demo)

### Recommended Requirements
- Python 3.8+
- 8GB+ RAM
- 2GB+ disk space
- Internet connection (for scraping)

### Optional Requirements
- NVIDIA GPU (for deep learning)
- Instagram account (for scraping)

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Try the minimal demo first
3. Run setup.py for automated installation
4. Check README.md for detailed instructions
5. Review notebooks for examples

## Performance Tips

1. **Use virtual environment** to avoid conflicts
2. **Install specific versions** if compatibility issues arise
3. **Clear cache** periodically: `rm -rf cache/`
4. **Use SSD storage** for better I/O performance
5. **Limit concurrent requests** when scraping to avoid rate limiting
