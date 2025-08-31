#!/usr/bin/env python3
"""
Test the new wrapper approach - using actual working agents AS-IS
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_state import PipelineState
from agents_wrapper import preprocessing_agent, feature_selection_agent, model_building_agent

def test_wrapper_imports():
    """Test that all wrapper imports work"""
    print("🧪 Testing Wrapper Imports")
    print("=" * 50)
    
    print(f"✅ Preprocessing agent available: {preprocessing_agent.available}")
    print(f"✅ Feature selection agent available: {feature_selection_agent.available}")
    print(f"✅ Model building agent available: {model_building_agent.available}")
    
    return True

def test_preprocessing_wrapper():
    """Test preprocessing agent wrapper"""
    print("\n🧪 Testing Preprocessing Wrapper")
    print("=" * 50)
    
    if not preprocessing_agent.available:
        print("❌ Preprocessing agent not available - skipping test")
        return False
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Create test state
    state = PipelineState(
        session_id="test_preprocessing",
        chat_session="test_preprocessing",
        user_query="clean my data interactively",
        raw_data=sample_data,
        target_column="target"
    )
    
    print(f"📊 Input data shape: {sample_data.shape}")
    print("🚀 Calling preprocessing wrapper...")
    
    try:
        # This should call your actual working preprocessing agent
        result_state = preprocessing_agent.run(state)
        
        print("✅ Preprocessing wrapper completed")
        if result_state.cleaned_data is not None:
            print(f"📊 Output data shape: {result_state.cleaned_data.shape}")
            return True
        else:
            print("⚠️ No cleaned data returned")
            return False
            
    except Exception as e:
        print(f"❌ Preprocessing wrapper failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_selection_wrapper():
    """Test feature selection agent wrapper"""
    print("\n🧪 Testing Feature Selection Wrapper")
    print("=" * 50)
    
    if not feature_selection_agent.available:
        print("❌ Feature selection agent not available - skipping test")
        return False
    
    # Create sample cleaned data
    cleaned_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'feature4': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Create test state
    state = PipelineState(
        session_id="test_feature_selection",
        chat_session="test_feature_selection",
        user_query="help me select the best features",
        cleaned_data=cleaned_data,
        target_column="target"
    )
    
    print(f"📊 Input data shape: {cleaned_data.shape}")
    print("🚀 Calling feature selection wrapper...")
    
    try:
        # This should call your actual working feature selection agent
        result_state = feature_selection_agent.run(state)
        
        print("✅ Feature selection wrapper completed")
        print(f"📊 Session active: {result_state.feature_selection_state.get('session_active', False)}")
        return True
        
    except Exception as e:
        print(f"❌ Feature selection wrapper failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all wrapper tests"""
    print("🚀 Testing New Wrapper Approach")
    print("Using actual working agents AS-IS")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_wrapper_imports():
        tests_passed += 1
    
    # Test preprocessing wrapper
    if test_preprocessing_wrapper():
        tests_passed += 1
    
    # Test feature selection wrapper  
    if test_feature_selection_wrapper():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All wrapper tests passed! Ready to use actual working agents.")
    else:
        print("⚠️ Some tests failed. Check the wrapper implementation.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
