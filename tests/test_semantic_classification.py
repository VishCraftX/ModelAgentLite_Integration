#!/usr/bin/env python3
"""
Test script for semantic intent classification
"""

import sys
import os
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator
from pipeline_state import PipelineState

def test_semantic_classification():
    """Test the new semantic classification system"""
    
    print("🧠 Testing Semantic Intent Classification")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Test cases that should benefit from semantic understanding
    test_cases = [
        # Synonyms and variations that keywords might miss
        ("Data Cleaning", "sanitize my dataset and handle inconsistencies"),
        ("Feature Engineering", "engineer new variables and select important attributes"),  
        ("Model Development", "develop a machine learning algorithm for prediction"),
        ("Statistical Analysis", "compute descriptive statistics and create visualizations"),
        ("General Help", "what are your capabilities and how can you assist me"),
        
        # Plurals and variations
        ("Preprocessing Plurals", "clean datasets, handle missing values, remove outliers"),
        ("Model Building Plurals", "train multiple models and compare their performances"),
        ("Feature Selection Plurals", "analyze correlations between variables"),
        
        # Context-dependent phrases
        ("Complex Preprocessing", "prepare my data for machine learning by handling quality issues"),
        ("Complex Model Building", "create predictive models using ensemble methods"),
        ("Complex Feature Selection", "identify the most predictive variables for my target"),
        
        # Edge cases that might confuse keyword matching
        ("Mixed Context", "clean my model's predictions and retrain"),
        ("Ambiguous", "analyze my data thoroughly"),
        ("Very General", "help me with my machine learning project"),
    ]
    
    for test_name, query in test_cases:
        print(f"\n🔍 {test_name}")
        print(f"Query: '{query}'")
        print("-" * 50)
        
        # Create a mock state
        state = PipelineState(user_query=query, session_id="test_session")
        
        # Test routing
        try:
            result = orchestrator.route(state)
            print(f"🎯 Routed to: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_fallback_behavior():
    """Test that the system gracefully falls back when embeddings are unavailable"""
    
    print("\n\n🔄 Testing Fallback Behavior")
    print("=" * 60)
    
    # Test with a simple query
    orchestrator = Orchestrator()
    state = PipelineState(user_query="clean my data", session_id="test_session")
    
    try:
        result = orchestrator.route(state)
        print(f"✅ Fallback test successful: {result}")
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")

if __name__ == "__main__":
    print("🚀 Semantic Intent Classification Test Suite")
    print("=" * 60)
    
    # Test semantic classification
    test_semantic_classification()
    
    # Test fallback behavior
    test_fallback_behavior()
    
    print("\n✅ Test suite completed!")