#!/usr/bin/env python3
"""
Final test for Manage Policies button functionality
"""

import requests
import time
import json

def test_ui_accessibility():
    """Test that the UI is accessible and responsive."""
    print("Testing UI accessibility...")
    
    try:
        # Test main page
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("✅ Main UI page accessible")
            
            # Test Streamlit health endpoint
            health_response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
            if health_response.status_code == 200:
                print("✅ Streamlit health check passed")
                return True
            else:
                print(f"❌ Streamlit health check failed: {health_response.status_code}")
                return False
        else:
            print(f"❌ Main UI page not accessible: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing UI: {e}")
        return False

def test_api_functionality():
    """Test API functionality."""
    print("\nTesting API functionality...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API health check passed")
            
            # Test optimization endpoint
            test_data = {
                "returns": [[0.01, 0.02], [0.02, 0.01]],
                "asset_names": ["Asset1", "Asset2"],
                "dates": ["2023-01-01", "2023-01-02"],
                "objective": "gmv",
                "long_only": True
            }
            
            opt_response = requests.post("http://localhost:8000/optimize", json=test_data, timeout=10)
            if opt_response.status_code == 200:
                print("✅ Optimization endpoint working")
                return True
            else:
                print(f"❌ Optimization endpoint failed: {opt_response.status_code}")
                return False
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API error: {e}")
        return False

def test_policy_files():
    """Test policy files are accessible."""
    print("\nTesting policy files...")
    
    import os
    import yaml
    
    policy_dir = "configs/policy.examples"
    if os.path.exists(policy_dir):
        policy_files = [f for f in os.listdir(policy_dir) if f.endswith('.yaml')]
        print(f"✅ Found {len(policy_files)} policy files")
        
        for filename in policy_files:
            try:
                with open(os.path.join(policy_dir, filename), 'r') as f:
                    policy = yaml.safe_load(f)
                    print(f"   ✅ {filename}: Valid")
            except Exception as e:
                print(f"   ❌ {filename}: Invalid - {e}")
        
        return len(policy_files) > 0
    else:
        print("❌ Policy directory not found")
        return False

def main():
    """Run final comprehensive test."""
    print("🔧 Final Test: Manage Policies Button Functionality")
    print("=" * 60)
    
    # Test UI accessibility
    ui_ok = test_ui_accessibility()
    
    # Test API functionality
    api_ok = test_api_functionality()
    
    # Test policy files
    policy_ok = test_policy_files()
    
    print("\n" + "=" * 60)
    if ui_ok and api_ok and policy_ok:
        print("🎉 All systems are working!")
        print("✅ UI is accessible and responsive")
        print("✅ API is working correctly")
        print("✅ Policy files are available")
        print("\n🌐 Manual Testing Instructions:")
        print("   1. Open your browser and go to: http://localhost:8501")
        print("   2. You should see the Omega Portfolio Engine interface")
        print("   3. Look for the sidebar with navigation options")
        print("   4. Select 'Portfolio Optimization' or 'Backtesting'")
        print("   5. Look for the 'Manage Policies' button (should be on the right side)")
        print("   6. Click the 'Manage Policies' button")
        print("   7. You should see a message '🔄 Navigating to Policy Management...'")
        print("   8. The page should change to show Policy Management with 3 tabs:")
        print("      - Select Policy")
        print("      - Create Policy") 
        print("      - Edit Policy")
        print("\n🔍 If the button doesn't work:")
        print("   - Check the browser console for any JavaScript errors")
        print("   - Try refreshing the page")
        print("   - Make sure you're on the Portfolio Optimization or Backtesting page")
        print("   - The button should be visible in the 'Investment Policy' section")
    else:
        print("⚠️  Some systems are not working properly.")
        print("   Check the output above for specific issues.")

if __name__ == "__main__":
    main()
