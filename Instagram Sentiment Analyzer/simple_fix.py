"""
Simple compatibility fix for NumPy and Scikit-learn
"""

import subprocess
import sys

def run_command(command, description):
    """Run command and handle errors"""
    print(f"Processing: {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"SUCCESS: {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description}")
        print(f"Error: {e.stderr}")
        return False

def test_imports():
    """Test critical imports"""
    print("\nTesting imports...")
    
    test_modules = [
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("plotly", "Plotly"),
        ("streamlit", "Streamlit")
    ]
    
    failed_imports = []
    
    for module, name in test_modules:
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError as e:
            print(f"[FAIL] {name}: {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0, failed_imports

def fix_compatibility():
    """Fix compatibility issues"""
    print("Fixing NumPy/Scikit-learn compatibility...")
    
    commands = [
        ("pip uninstall numpy scikit-learn -y", "Uninstall conflicting packages"),
        ("pip install numpy==1.24.3", "Install compatible NumPy"),
        ("pip install scikit-learn==1.3.0", "Install compatible Scikit-learn")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("Instagram Sentiment Analyzer - Compatibility Fix")
    print("=" * 50)
    
    # Test current state
    success, failed = test_imports()
    
    if success:
        print("\nAll imports working! No fixes needed.")
        print("\nNext steps:")
        print("1. python quick_start.py")
        print("2. streamlit run app/streamlit_app.py")
        return
    
    print(f"\nFailed imports: {', '.join(failed)}")
    
    # Fix compatibility
    print("\nAttempting compatibility fix...")
    if fix_compatibility():
        print("\nFix applied successfully!")
        
        # Test again
        success, failed = test_imports()
        
        if success:
            print("\nAll imports now working!")
            print("\nReady for advanced features:")
            print("1. python quick_start.py")
            print("2. streamlit run app/streamlit_app.py")
        else:
            print(f"\nStill failing: {', '.join(failed)}")
    else:
        print("\nFix failed. Manual solutions:")
        print("1. Create new virtual environment")
        print("2. Use Python 3.9 or 3.10")
        print("3. Use conda environment")

if __name__ == "__main__":
    main()
