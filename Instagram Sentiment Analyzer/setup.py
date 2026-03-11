"""
Instagram Sentiment Analyzer Setup Script
Handles installation and initial setup
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 Instagram Sentiment Analyzer Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Install core requirements
    print("\n📦 Installing core requirements...")
    
    # Install basic requirements first
    basic_packages = [
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0", 
        "numpy>=1.24.0",
        "nltk>=3.8.1",
        "spacy>=3.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "wordcloud>=1.9.0",
        "streamlit>=1.28.0",
        "emoji>=2.8.0",
        "regex>=2023.10.3",
        "textblob>=0.17.1",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
        "joblib>=1.3.0",
        "openpyxl>=3.1.0"
    ]
    
    for package in basic_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Install spaCy model
    run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model")
    
    # Download NLTK data
    print("\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for data in nltk_data:
            try:
                nltk.download(data, quiet=True)
                print(f"✅ Downloaded {data}")
            except:
                print(f"⚠️  Failed to download {data}")
    except ImportError:
        print("⚠️  NLTK not installed properly")
    
    # Install optional packages (with error handling)
    print("\n🔧 Installing optional packages...")
    
    optional_packages = [
        ("instaloader>=4.10.0", "Instagram scraper"),
        ("selenium>=4.15.0", "Web automation"),
        ("beautifulsoup4>=4.12.0", "HTML parsing"),
        ("requests>=2.31.0", "HTTP requests"),
        ("flask>=2.3.0", "Web framework")
    ]
    
    for package, description in optional_packages:
        if not run_command(f"pip install {package}", f"Installing {description} ({package})"):
            print(f"⚠️  {description} installation failed - some features may not work")
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = [
        "data",
        "data/raw", 
        "data/processed",
        "data/visualizations",
        "models",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")
    
    # Test installation
    print("\n🧪 Testing installation...")
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        import streamlit
        import nltk
        import spacy
        
        print("✅ Core packages imported successfully")
        
        # Test spaCy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except:
            print("⚠️  spaCy model not available - run: python -m spacy download en_core_web_sm")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Some packages may not have installed correctly")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Run the dashboard: streamlit run app/streamlit_app.py")
    print("2. Or use CLI: python main.py --help")
    print("3. Check the README.md for detailed instructions")
    
    # Optional deep learning setup
    print("\n🤖 For deep learning features (optional):")
    print("pip install tensorflow>=2.14.0 torch>=2.1.0 transformers>=4.35.0")

if __name__ == "__main__":
    main()
