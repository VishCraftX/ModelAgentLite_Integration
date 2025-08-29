#!/usr/bin/env python3
"""
Test script for the enhanced orchestrator system
"""

import pandas as pd
import numpy as np
from pipeline_state import PipelineState
from orchestrator import orchestrator

def test_enhanced_orchestrator():
    """Test the enhanced orchestrator with various queries"""
    
    print("🧪 Testing Enhanced Orchestrator System (Single Flow)")
    print("=" * 55)
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'credit_score': np.random.randint(300, 850, 100),
        'target': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    })
    
    # Test queries
    test_queries = [
        ("General greeting", "Hello, what can you do?"),
        ("Capability question", "What are your capabilities?"),
        ("Status question", "What's the current status?"),
        ("Preprocessing request", "Clean this data and handle missing values"),
        ("Feature selection request", "Select the most important features"),
        ("New model request", "Train a machine learning model"),
        ("Existing model request", "Use existing model for predictions"),
        ("Model visualization", "Show model performance plots"),
        ("Full pipeline request", "Build a complete ML pipeline"),
        ("Code execution request", "Calculate correlation between features"),
    ]
    
    for test_name, query in test_queries:
        print(f"\n🔍 {test_name}")
        print(f"Query: '{query}'")
        print("-" * 40)
        
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
            
            print(f"✅ Routing: {routing_decision}")
            print(f"📝 Explanation: {explanation}")
            
            # If it's a general response, show the generated response
            if hasattr(state, 'progress') and state.progress and routing_decision == "END":
                print(f"🤖 Response: {state.progress[:200]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'=' * 50}")
    print("🎉 Enhanced Orchestrator Test Complete!")

if __name__ == "__main__":
    test_enhanced_orchestrator()
