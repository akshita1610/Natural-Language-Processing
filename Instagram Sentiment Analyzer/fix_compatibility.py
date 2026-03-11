"""
Fix compatibility issues between NumPy and Scikit-learn
"""

import subprocess
import sys

def run_command(command, description):
    """Run command and handle errors"""
    print(f"Processing: {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"SUCCESS: {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def fix_numpy_sklearn_compatibility():
    """Fix NumPy and Scikit-learn compatibility"""
    print("🔧 Fixing NumPy/Scikit-learn compatibility...")
    
    # Uninstall conflicting packages
    commands = [
        ("pip uninstall numpy scikit-learn -y", "Uninstalling NumPy and Scikit-learn"),
        ("pip install numpy==1.24.3", "Installing compatible NumPy"),
        ("pip install scikit-learn==1.3.0", "Installing compatible Scikit-learn")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"⚠️  Failed to {description.lower()}")
            return False
    
    return True

def test_imports():
    """Test critical imports"""
    print("\n🧪 Testing imports...")
    
    test_modules = [
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("plotly", "Plotly"),
        ("streamlit", "Streamlit"),
        ("nltk", "NLTK"),
        ("spacy", "spaCy")
    ]
    
    failed_imports = []
    
    for module, name in test_modules:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0, failed_imports

def main():
    """Main fix function"""
    print("Instagram Sentiment Analyzer - Compatibility Fix")
    print("=" * 60)
    
    # Test current imports
    success, failed = test_imports()
    
    if success:
        print("\n🎉 All imports successful! No fixes needed.")
        print("\n📋 Next steps:")
        print("1. Test ML demo: python quick_start.py")
        print("2. Launch dashboard: streamlit run app/streamlit_app.py")
        return
    
    print(f"\n⚠️  Failed imports: {', '.join(failed)}")
    
    # Attempt to fix compatibility
    if "NumPy" in failed or "Scikit-learn" in failed:
        print("\n🔧 Attempting to fix NumPy/Scikit-learn compatibility...")
        
        if fix_numpy_sklearn_compatibility():
            print("\n✅ Compatibility fix applied")
            
            # Test again
            success, failed = test_imports()
            
            if success:
                print("\n🎉 Fixes successful! All imports working.")
            else:
                print(f"\n⚠️  Still failing: {', '.join(failed)}")
                print("\n🔄 Alternative solutions:")
                print("1. Create new virtual environment")
                print("2. Use different Python version (3.9-3.10)")
                print("3. Use conda environment")
        else:
            print("\n❌ Compatibility fix failed")
    
    print(f"\n📚 Manual fix options:")
    print(f"1. Fresh virtual environment:")
    print(f"   python -m venv fresh_env")
    print(f"   fresh_env\\Scripts\\activate")
    print(f"   pip install numpy==1.24.3 scikit-learn==1.3.0")
    
    print(f"\n2. Conda environment:")
    print(f"   conda create -n sentiment python=3.9")
    print(f"   conda activate sentiment")
    print(f"   pip install -r requirements.txt")
    
    print(f"\n3. Minimal version (always works):")
    print(f"   python minimal_demo.py")

if __name__ == "__main__":
    main()
