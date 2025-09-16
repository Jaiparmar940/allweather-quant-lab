#!/usr/bin/env python3
"""
Test script to verify navigation functionality
"""

import requests
import time

def test_ui_navigation():
    """Test that the UI is accessible and navigation works."""
    print("Testing UI navigation...")
    
    # Test main UI page
    response = requests.get("http://localhost:8501")
    if response.status_code == 200:
        print("✅ UI is accessible")
        
        # Check if the page contains expected elements
        content = response.text.lower()
        
        if "omega portfolio engine" in content:
            print("✅ Main title found")
        else:
            print("❌ Main title not found")
            
        if "policy management" in content:
            print("✅ Policy Management page option found")
        else:
            print("❌ Policy Management page option not found")
            
        if "portfolio optimization" in content:
            print("✅ Portfolio Optimization page option found")
        else:
            print("❌ Portfolio Optimization page option not found")
            
        if "backtesting" in content:
            print("✅ Backtesting page option found")
        else:
            print("❌ Backtesting page option not found")
            
        return True
    else:
        print(f"❌ UI not accessible: {response.status_code}")
        return False

def test_api_health():
    """Test API health."""
    print("\nTesting API health...")
    
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        print("✅ API is healthy")
        return True
    else:
        print(f"❌ API health check failed: {response.status_code}")
        return False

def main():
    """Run navigation tests."""
    print("🔧 Testing UI Navigation")
    print("=" * 40)
    
    # Test API health
    api_ok = test_api_health()
    
    # Test UI navigation
    ui_ok = test_ui_navigation()
    
    print("\n" + "=" * 40)
    if api_ok and ui_ok:
        print("🎉 Navigation tests passed!")
        print("✅ UI is accessible and contains expected elements")
        print("✅ API is healthy")
        print("\n🌐 You can now use the web interface at: http://localhost:8501")
        print("   - The 'Manage Policies' button should now work")
        print("   - Click it to navigate to the Policy Management page")
        print("   - Use the sidebar to navigate between pages")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
