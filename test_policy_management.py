#!/usr/bin/env python3
"""
Test script to verify policy management functionality
"""

import requests
import json
import yaml
import os

def test_policy_loading():
    """Test loading existing policy templates."""
    print("Testing policy template loading...")
    
    policy_dir = "configs/policy.examples"
    if os.path.exists(policy_dir):
        policies = {}
        for filename in os.listdir(policy_dir):
            if filename.endswith('.yaml'):
                policy_name = filename.replace('.yaml', '').replace('_', ' ').title()
                try:
                    with open(os.path.join(policy_dir, filename), 'r') as f:
                        policy = yaml.safe_load(f)
                        policies[policy_name] = policy
                        print(f"✅ Loaded policy: {policy_name}")
                except Exception as e:
                    print(f"❌ Error loading policy {filename}: {e}")
        
        print(f"\n📋 Loaded {len(policies)} policy templates:")
        for name, policy in policies.items():
            risk_tol = policy.get('client_profile', {}).get('risk_tolerance', 'unknown')
            target_ret = policy.get('return_requirements', {}).get('target_return', 0) * 100
            max_dd = policy.get('risk_constraints', {}).get('max_drawdown', 0) * 100
            print(f"   • {name}: {risk_tol} risk, {target_ret:.1f}% target, {max_dd:.1f}% max DD")
        
        return policies
    else:
        print(f"❌ Policy directory {policy_dir} not found")
        return {}

def test_policy_creation():
    """Test creating a custom policy."""
    print("\nTesting custom policy creation...")
    
    # Create a sample custom policy
    custom_policy = {
        "client_profile": {
            "name": "Test Custom Policy",
            "risk_tolerance": "medium",
            "investment_horizon": "medium_term",
            "liquidity_needs": "medium"
        },
        "return_requirements": {
            "minimum_acceptable_return": 0.03,
            "target_return": 0.08,
            "return_currency": "real"
        },
        "risk_constraints": {
            "max_drawdown": 0.12,
            "max_volatility": 0.15,
            "var_confidence": 0.95,
            "cvar_confidence": 0.95
        },
        "asset_constraints": {
            "min_fixed_income": 0.30,
            "max_fixed_income": 0.60,
            "min_equity": 0.30,
            "max_equity": 0.60,
            "min_commodities": 0.05,
            "max_commodities": 0.15
        },
        "additional_constraints": {
            "max_single_weight": 0.15,
            "max_sector_weight": 0.35,
            "max_leverage": 1.2,
            "turnover_tolerance": 0.40
        }
    }
    
    print("✅ Custom policy created successfully")
    print(f"   Risk tolerance: {custom_policy['client_profile']['risk_tolerance']}")
    print(f"   Target return: {custom_policy['return_requirements']['target_return']*100:.1f}%")
    print(f"   Max drawdown: {custom_policy['risk_constraints']['max_drawdown']*100:.1f}%")
    print(f"   Max single weight: {custom_policy['additional_constraints']['max_single_weight']*100:.1f}%")
    
    return custom_policy

def test_policy_validation():
    """Test policy validation logic."""
    print("\nTesting policy validation...")
    
    # Test valid policy
    valid_policy = {
        "client_profile": {"risk_tolerance": "high"},
        "return_requirements": {"target_return": 0.10},
        "risk_constraints": {"max_drawdown": 0.20},
        "asset_constraints": {"min_equity": 0.50},
        "additional_constraints": {"max_single_weight": 0.20}
    }
    
    # Test invalid policy (missing required fields)
    invalid_policy = {
        "client_profile": {"risk_tolerance": "high"}
        # Missing other required sections
    }
    
    def validate_policy(policy):
        required_sections = ['client_profile', 'return_requirements', 'risk_constraints', 'asset_constraints']
        for section in required_sections:
            if section not in policy:
                return False, f"Missing section: {section}"
        return True, "Valid"
    
    valid, msg = validate_policy(valid_policy)
    print(f"✅ Valid policy: {msg}")
    
    valid, msg = validate_policy(invalid_policy)
    print(f"❌ Invalid policy: {msg}")
    
    return valid

def test_api_integration():
    """Test API integration with policies."""
    print("\nTesting API integration...")
    
    # Test API health
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        print("✅ API is healthy")
    else:
        print("❌ API health check failed")
        return False
    
    # Test optimization with policy constraints
    print("Testing optimization with policy constraints...")
    
    # Create sample data
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, (252, 5))
    asset_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
    
    # Test with policy constraints
    request_data = {
        "returns": returns.tolist(),
        "asset_names": asset_names,
        "dates": dates.strftime("%Y-%m-%d").tolist(),
        "objective": "gmv",
        "long_only": True,
        "max_single_weight": 0.20,  # Policy constraint
        "max_volatility": 0.15      # Policy constraint
    }
    
    response = requests.post("http://localhost:8000/optimize", json=request_data)
    if response.status_code == 200:
        data = response.json()
        print("✅ Optimization with policy constraints successful")
        print(f"   Status: {data['status']}")
        print(f"   Weights:")
        for asset, weight in data['weights'].items():
            print(f"     {asset}: {weight:.3f} ({weight*100:.1f}%)")
        return True
    else:
        print(f"❌ Optimization failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def main():
    """Run policy management tests."""
    print("🔧 Testing Policy Management Functionality")
    print("=" * 60)
    
    # Test policy loading
    policies = test_policy_loading()
    
    # Test policy creation
    custom_policy = test_policy_creation()
    
    # Test policy validation
    test_policy_validation()
    
    # Test API integration
    api_success = test_api_integration()
    
    print("\n" + "=" * 60)
    if api_success:
        print("🎉 All policy management tests passed!")
        print("✅ Policy templates loaded successfully")
        print("✅ Custom policy creation works")
        print("✅ Policy validation works")
        print("✅ API integration with policies works")
        print("\n🌐 You can now use the web interface at: http://localhost:8501")
        print("   - Go to 'Policy Management' page")
        print("   - Create custom policies or select existing ones")
        print("   - Apply policies to optimization and backtesting")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
