#!/usr/bin/env python3
"""
Test script to verify that all Streamlit pages can be imported without errors.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all pages can be imported."""
    
    print("Testing imports...")
    
    try:
        print("1. Testing main app.py...")
        import src.app as main_app
        print("   ✅ Main app imports successfully")
    except Exception as e:
        print(f"   ❌ Main app import failed: {e}")
        return False
    
    try:
        print("2. Testing pages/upload.py...")
        import src.pages.upload as upload_page
        print("   ✅ Upload page imports successfully")
    except Exception as e:
        print(f"   ❌ Upload page import failed: {e}")
        return False
    
    try:
        print("3. Testing pages/search.py...")
        import src.pages.search as search_page
        print("   ✅ Search page imports successfully")
    except Exception as e:
        print(f"   ❌ Search page import failed: {e}")
        return False
    
    try:
        print("4. Testing pages/chat.py...")
        import src.pages.chat as chat_page
        print("   ✅ Chat page imports successfully")
    except Exception as e:
        print(f"   ❌ Chat page import failed: {e}")
        return False
    
    return True

def test_page_structure():
    """Test that page files have the expected structure."""
    
    print("\nTesting page structure...")
    
    required_files = [
        "src/app.py",
        "src/pages/upload.py", 
        "src/pages/search.py",
        "src/pages/chat.py"
    ]
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path} exists")
        else:
            print(f"   ❌ {file_path} missing")
            return False
    
    return True

def main():
    """Run all tests."""
    
    print("=" * 60)
    print("SDS-RAG Streamlit Pages Test")
    print("=" * 60)
    
    # Test file structure
    if not test_page_structure():
        print("\n❌ Page structure test failed!")
        return 1
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return 1
    
    print("\n🎉 All tests passed!")
    print("\nPage navigation should work correctly. The multi-page Streamlit app is ready.")
    print("\nTo run the application:")
    print("  poetry run streamlit run src/app.py")
    print("  OR")
    print("  python scripts/run_app.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())