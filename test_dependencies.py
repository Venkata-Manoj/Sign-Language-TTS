"""
Dependency Verification Script
Tests all imports required for Sign Language TTS Application
"""

import sys

def test_imports():
    """Test all required imports and display results."""
    
    print("=" * 70)
    print("SIGN LANGUAGE TTS - DEPENDENCY VERIFICATION")
    print("=" * 70)
    print()
    
    results = []
    
    # Test each dependency
    tests = [
        ("opencv-python", "import cv2", "cv2.__version__"),
        ("mediapipe", "import mediapipe as mp", "mp.__version__"),
        ("numpy", "import numpy as np", "np.__version__"),
        ("torch", "import torch", "torch.__version__"),
        ("pyttsx3", "import pyttsx3", "pyttsx3.version"),
        ("gTTS", "from gtts import gTTS", "'2.x.x'"),
        ("pygame", "import pygame", "pygame.version.ver"),
        ("Pillow", "from PIL import Image, ImageTk", "Image.__version__ if hasattr(Image, '__version__') else 'Pillow installed'"),
        ("language-tool-python", "import language_tool_python", "'installed'"),
        ("scikit-learn", "from sklearn.ensemble import RandomForestClassifier", "'installed'"),
        ("tqdm", "from tqdm import tqdm", "'installed'"),
    ]
    
    for package_name, import_stmt, version_expr in tests:
        try:
            # Execute import
            exec(import_stmt, globals())
            
            # Get version
            try:
                version = eval(version_expr)
            except:
                version = "✓"
            
            print(f"✅ {package_name:<25} {version}")
            results.append((package_name, True, version))
            
        except ImportError as e:
            print(f"❌ {package_name:<25} NOT INSTALLED")
            results.append((package_name, False, str(e)))
    
    print()
    print("=" * 70)
    
    # Test standard library
    print("STANDARD LIBRARY MODULES (Built-in)")
    print("=" * 70)
    print()
    
    stdlib_modules = [
        "pickle", "tkinter", "threading", "time", "os", "warnings", "collections"
    ]
    
    for module in stdlib_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} (should be built-in)")
    
    print()
    print("=" * 70)
    
    # Summary
    total = len(results)
    successful = sum(1 for _, success, _ in results if success)
    
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Dependencies: {total}")
    print(f"Successfully Installed: {successful}")
    print(f"Failed: {total - successful}")
    print()
    
    if successful == total:
        print("🎉 All dependencies installed successfully!")
        print("✅ Ready to run Sign Language TTS Application")
        return 0
    else:
        print("⚠️  Some dependencies are missing")
        print("Run: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    exit_code = test_imports()
    sys.exit(exit_code)
