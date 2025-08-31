#!/usr/bin/env python3
"""
Test script to verify interactive agent integration works correctly
"""

import sys
import os
import pandas as pd
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_state import PipelineState
from agents_integrated import preprocessing_agent, feature_selection_agent

def test_interactive_preprocessing():
    """Test that preprocessing agent launches interactive workflow"""
    print("🧪 Testing Interactive Preprocessing Agent")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Create test state
    state = PipelineState(
        session_id="test_session",
        chat_session="test_session",
        user_query="I want to clean my data interactively",
        raw_data=sample_data,
        target_column="target"
    )
    
    # Run preprocessing agent
    result_state = preprocessing_agent.run(state)
    
    # Check if interactive session was started
    if hasattr(result_state, 'interactive_session') and result_state.interactive_session:
        print("✅ Interactive preprocessing session started successfully")
        print(f"   Agent type: {result_state.interactive_session['agent_type']}")
        print(f"   Session active: {result_state.interactive_session.get('active', False)}")
        
        # Test continuing the session with user input
        continued_state = preprocessing_agent.run_interactive_workflow(result_state, "proceed")
        print("✅ Interactive session continuation works")
        
        return True
    else:
        print("❌ Interactive session was not started")
        return False

def test_interactive_feature_selection():
    """Test that feature selection agent launches interactive workflow"""
    print("\n🧪 Testing Interactive Feature Selection Agent")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'feature3': [100, 200, 300, 400, 500],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Create test state
    state = PipelineState(
        session_id="test_session",
        chat_session="test_session",
        user_query="what are the different feature selection techniques you can implement?",
        cleaned_data=sample_data,
        target_column="target"
    )
    
    # Run feature selection agent
    result_state = feature_selection_agent.run(state)
    
    # Check if interactive session was started or general query was handled
    if hasattr(result_state, 'interactive_session') and result_state.interactive_session:
        print("✅ Interactive feature selection session started successfully")
        print(f"   Agent type: {result_state.interactive_session['agent_type']}")
        print(f"   Session active: {result_state.interactive_session.get('active', False)}")
        return True
    elif result_state.last_response and "Feature Selection Overview" in result_state.last_response:
        print("✅ General query about feature selection handled correctly")
        print(f"   Response length: {len(result_state.last_response)} characters")
        return True
    else:
        print("❌ Neither interactive session nor general query response was generated")
        return False

def test_general_query_handling():
    """Test that general queries are handled correctly by agents"""
    print("\n🧪 Testing General Query Handling")
    
    # Test preprocessing general query
    state = PipelineState(
        session_id="test_session",
        chat_session="test_session",
        user_query="what is data preprocessing?",
        raw_data=pd.DataFrame({'a': [1, 2, 3]})
    )
    
    result_state = preprocessing_agent.run(state)
    
    if result_state.last_response and len(result_state.last_response) > 50:
        print("✅ Preprocessing general query handled")
    else:
        print("❌ Preprocessing general query not handled properly")
        return False
    
    # Test feature selection general query
    state = PipelineState(
        session_id="test_session",
        chat_session="test_session",
        user_query="explain feature selection techniques",
        cleaned_data=pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    )
    
    result_state = feature_selection_agent.run(state)
    
    if result_state.last_response and len(result_state.last_response) > 50:
        print("✅ Feature selection general query handled")
        return True
    else:
        print("❌ Feature selection general query not handled properly")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Interactive Agent Integration")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    try:
        if test_interactive_preprocessing():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
    
    try:
        if test_interactive_feature_selection():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Feature selection test failed: {e}")
    
    try:
        if test_general_query_handling():
            tests_passed += 1
    except Exception as e:
        print(f"❌ General query test failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Interactive integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
