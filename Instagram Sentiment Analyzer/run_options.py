"""
Instagram Sentiment Analyzer - All Run Options
Shows what's available and what works on your system
"""

import subprocess
import sys
import importlib

def test_import(module_name, friendly_name):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True, friendly_name
    except ImportError as e:
        return False, f"{friendly_name} ({str(e)[:50]}...)"

def check_available_options():
    """Check what options are available"""
    print("=" * 70)
    print("INSTAGRAM SENTIMENT ANALYZER - AVAILABLE OPTIONS")
    print("=" * 70)
    
    # Test core packages
    core_packages = [
        ("json", "JSON (built-in)"),
        ("re", "Regex (built-in)"),
        ("collections", "Collections (built-in)"),
        ("pathlib", "Pathlib (built-in)"),
        ("math", "Math (built-in)")
    ]
    
    print("\nCORE PYTHON (Always Available):")
    print("-" * 40)
    for module, name in core_packages:
        available, _ = test_import(module, name)
        status = "[OK]" if available else "[FAIL]"
        print(f"{status} {name}")
    
    # Test ML packages
    ml_packages = [
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn")
    ]
    
    print("\nMACHINE LEARNING PACKAGES:")
    print("-" * 40)
    ml_available = 0
    for module, name in ml_packages:
        available, status = test_import(module, name)
        if available:
            ml_available += 1
        print(f"[{'OK' if available else 'FAIL'}] {status}")
    
    # Test visualization packages
    viz_packages = [
        ("plotly", "Plotly"),
        ("wordcloud", "WordCloud"),
        ("streamlit", "Streamlit")
    ]
    
    print("\nVISUALIZATION & WEB:")
    print("-" * 40)
    viz_available = 0
    for module, name in viz_packages:
        available, status = test_import(module, name)
        if available:
            viz_available += 1
        print(f"[{'OK' if available else 'FAIL'}] {status}")
    
    # Test NLP packages
    nlp_packages = [
        ("nltk", "NLTK"),
        ("spacy", "spaCy"),
        ("emoji", "Emoji"),
        ("textblob", "TextBlob")
    ]
    
    print("\nNLP & TEXT PROCESSING:")
    print("-" * 40)
    nlp_available = 0
    for module, name in nlp_packages:
        available, status = test_import(module, name)
        if available:
            nlp_available += 1
        print(f"[{'OK' if available else 'FAIL'}] {status}")
    
    # Test scraping packages
    scrape_packages = [
        ("instaloader", "Instaloader"),
        ("selenium", "Selenium"),
        ("bs4", "BeautifulSoup"),
        ("requests", "Requests")
    ]
    
    print("\nWEB SCRAPING:")
    print("-" * 40)
    scrape_available = 0
    for module, name in scrape_packages:
        available, status = test_import(module, name)
        if available:
            scrape_available += 1
        print(f"[{'OK' if available else 'FAIL'}] {status}")
    
    # Determine available options
    print("\n" + "=" * 70)
    print("AVAILABLE RUN OPTIONS:")
    print("=" * 70)
    
    print("\n1. BASIC OPTIONS (Always Work):")
    print("   python minimal_demo.py")
    print("   -> Rule-based sentiment analysis")
    print("   -> No dependencies required")
    
    if ml_available >= 2:
        print("\n2. ML OPTIONS (Available):")
        print("   python working_ml_demo.py")
        print("   -> Custom ML without scikit-learn")
        print("   -> Feature-based classification")
    
    if ml_available >= 4:
        print("\n3. ADVANCED ML (Available):")
        print("   python quick_start.py")
        print("   -> Scikit-learn based ML")
        print("   -> TF-IDF + Logistic Regression")
    
    if viz_available >= 2:
        print("\n4. DASHBOARD (Available):")
        print("   streamlit run app/streamlit_app.py")
        print("   -> Interactive web interface")
        print("   -> Real-time analysis")
    else:
        print("\n4. DASHBOARD (Limited):")
        print("   streamlit run app/streamlit_app.py")
        print("   -> May have limited features")
    
    if scrape_available >= 3:
        print("\n5. DATA COLLECTION (Available):")
        print("   python main.py --scrape-profile username")
        print("   python main.py --scrape-hashtag travel")
    else:
        print("\n5. DATA COLLECTION (Limited):")
        print("   -> Install: pip install instaloader selenium beautifulsoup4 requests")
    
    print("\n6. SETUP & UTILITIES:")
    print("   python setup.py")
    print("   -> Automated dependency installation")
    print("   python simple_fix.py")
    print("   -> Fix compatibility issues")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    
    if ml_available >= 4 and viz_available >= 2:
        print("-> FULL SYSTEM AVAILABLE: All features should work!")
        print("-> Recommended: streamlit run app/streamlit_app.py")
    elif ml_available >= 2:
        print("-> BASIC ML AVAILABLE: Use working_ml_demo.py")
        print("-> Try dashboard with limited features")
    else:
        print("-> MINIMAL SETUP: Use minimal_demo.py")
        print("-> Run setup.py for more features")
    
    print(f"\nPackage Summary:")
    print(f"- ML Packages: {ml_available}/5 available")
    print(f"- Visualization: {viz_available}/3 available")
    print(f"- NLP: {nlp_available}/4 available")
    print(f"- Scraping: {scrape_available}/4 available")
    
    print(f"\nSystem Info:")
    print(f"- Python: {sys.version}")
    print(f"- Platform: {sys.platform}")

def run_demo(choice):
    """Run the selected demo"""
    demos = {
        "1": ("Minimal Demo", "minimal_demo.py"),
        "2": ("Working ML Demo", "working_ml_demo.py"),
        "3": ("Quick Start (ML)", "quick_start.py"),
        "4": ("Setup Script", "setup.py"),
        "5": ("Compatibility Fix", "simple_fix.py")
    }
    
    if choice in demos:
        name, script = demos[choice]
        print(f"\nRunning: {name}")
        print("=" * 50)
        
        try:
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
        except FileNotFoundError:
            print(f"Script {script} not found")
    else:
        print("Invalid choice")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] in ["1", "2", "3", "4", "5"]:
        run_demo(sys.argv[1])
    else:
        check_available_options()
        
        print(f"\n" + "=" * 70)
        print("QUICK RUN:")
        print("=" * 70)
        print("python run_options.py 1  # Minimal demo")
        print("python run_options.py 2  # Working ML demo")
        print("python run_options.py 3  # Quick start (if ML available)")
        print("python run_options.py 4  # Setup script")
        print("python run_options.py 5  # Compatibility fix")

if __name__ == "__main__":
    main()
