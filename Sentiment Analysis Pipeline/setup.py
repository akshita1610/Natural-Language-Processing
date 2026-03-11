"""
Quick Setup Script for Sentiment Analysis Pipeline
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data."""
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for data in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'corpora/{data}' if data in ['stopwords', 'wordnet'] else f'taggers/{data}')
            except LookupError:
                nltk.download(data, quiet=True)
        
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ['logs', 'visualizations', 'audio_files', 'transcripts']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
    
    print("✅ Directories created successfully!")
    return True

def test_youtube_api():
    """Test YouTube API connection."""
    print("🧪 Testing YouTube API...")
    try:
        subprocess.check_call([sys.executable, "test_youtube.py"])
        print("✅ YouTube API test passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ YouTube API test failed: {e}")
        print("Please check your API key in config.yaml")
        return False

def main():
    """Run complete setup."""
    print("🚀 Sentiment Analysis Pipeline Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Download NLTK data
    if not download_nltk_data():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Test YouTube API (optional)
    test_choice = input("\n🧪 Test YouTube API connection? (y/n): ").strip().lower()
    if test_choice == 'y':
        test_youtube_api()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Add your API keys to config.yaml")
    print("2. Run: python youtube_sentiment_pipeline.py")
    print("3. Explore results: python run_analysis.py")
    print("4. Read GITHUB_SETUP.md for GitHub upload instructions")
    
    return True

if __name__ == "__main__":
    main()
