#!/usr/bin/env python3
"""
Test script for the hybrid orchestrator approach
Demonstrates fast keyword scoring with LLM fallback for ambiguous cases
"""

import pandas as pd
import numpy as np
from pipeline_state import PipelineState
from orchestrator import orchestrator

def test_hybrid_orchestrator():
    """Test the hybrid orchestrator with various confidence scenarios"""
    
    print("🧪 Testing Hybrid Orchestrator (Keyword + LLM Fallback)")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'credit_score': np.random.randint(300, 850, 100),
        'target': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    })
    
    # Test queries with different confidence levels
    test_queries = [
        # High confidence cases (should use keyword scoring)
        ("High confidence - preprocessing", "clean data missing values outliers"),
        ("High confidence - feature selection", "select features correlation analysis PCA"),
        ("High confidence - model building", "train model random forest classifier"),
        ("High confidence - full pipeline", "build complete ML pipeline end to end"),
        
        # Low confidence cases (should trigger LLM fallback)
        ("Low confidence - ambiguous", "help me with this"),
        ("Low confidence - unclear", "do something"),
        ("Low confidence - vague", "analyze"),
        
        # Ambiguous cases (close scores, should trigger LLM fallback)
        ("Ambiguous - mixed keywords", "clean model data features"),
        ("Ambiguous - general terms", "process analyze build"),
        ("Ambiguous - context dependent", "use this for analysis"),
        
        # Edge cases
        ("Edge case - greeting", "hello"),
        ("Edge case - capability", "what can you do"),
        ("Edge case - status", "current status"),
    ]
    
    for test_name, query in test_queries:
        print(f"\n🔍 {test_name}")
        print(f"Query: '{query}'")
        print("-" * 50)
        
        # Create state
        state = PipelineState(
            user_query=query,
            session_id="test_session",
            raw_data=sample_data
        )
        
        # Test routing
        try:
            routing_decision = orchestrator.route(state)
            explanation = orchestrator.get_routing_explanation(state, routing_decision)
            
            print(f"✅ Final Routing: {routing_decision}")
            print(f"📝 Explanation: {explanation}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print("🎉 Hybrid Orchestrator Test Complete!")
    print("\n📊 Expected Behavior:")
    print("⚡ High confidence queries → Fast keyword classification")
    print("🤖 Low confidence/ambiguous queries → LLM fallback")
    print("🎯 Best of both worlds: Speed + Intelligence")

def test_keyword_scoring_details():
    """Test the keyword scoring mechanism in detail"""
    
    print("\n" + "=" * 60)
    print("🔍 DETAILED KEYWORD SCORING ANALYSIS")
    print("=" * 60)
    
    test_queries = [
        "clean data and remove missing values",
        "select important features using correlation",
        "train random forest model",
        "help me analyze this",
        "clean model features data"  # Ambiguous case
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 30)
        
        intent, confidence_info = orchestrator._classify_with_keyword_scoring(query)
        
        print(f"🎯 Intent: {intent}")
        print(f"📊 Max Score: {confidence_info['max_score']:.3f}")
        print(f"📏 Score Diff: {confidence_info['score_diff']:.3f}")
        print(f"🔢 Raw Scores: {confidence_info['raw_scores']}")
        print(f"📈 Normalized: {confidence_info['scores']}")
        
        # Determine if LLM fallback would trigger
        needs_llm = (
            confidence_info["max_score"] < 0.25 or 
            confidence_info["score_diff"] < 0.1
        )
        
        if needs_llm:
            print("🤖 → Would trigger LLM fallback")
        else:
            print("⚡ → High confidence, use keyword result")

if __name__ == "__main__":
    test_hybrid_orchestrator()
    test_keyword_scoring_details()
