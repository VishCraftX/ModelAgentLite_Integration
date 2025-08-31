#!/usr/bin/env python3
"""
Test Suite for Session Conflict Resolution
Tests how the system handles conflicting signals in user queries
"""

import sys
import os
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_pipeline import MultiAgentMLPipeline
from pipeline_state import PipelineState

def test_session_conflict_resolution():
    """
    Test conflict resolution when user query contains both continuation and switching signals
    This is critical for robust session management
    """
    
    print("⚖️  SESSION CONFLICT RESOLUTION TEST")
    print("=" * 80)
    print("🎯 Testing how system handles conflicting signals in user queries")
    print("🔍 Focus: Queries with both continuation AND switching indicators")
    print("=" * 80)
    
    # Test scenarios with different session contexts
    conflict_scenarios = [
        {
            "session_context": {
                "agent_type": "preprocessing",
                "phase": "outliers_detection",
                "data_state": "analyzing_outliers"
            },
            "conflict_cases": [
                # PRIORITY TESTS: What should win when both signals present?
                {
                    "query": "skip outliers and select features",
                    "continuation_signal": "skip outliers",
                    "switching_signal": "select features", 
                    "expected_winner": "SWITCHING",
                    "reasoning": "Feature selection is a new agent request, should override skip"
                },
                {
                    "query": "select features but skip this step first",
                    "continuation_signal": "skip this step",
                    "switching_signal": "select features",
                    "expected_winner": "SWITCHING", 
                    "reasoning": "Select features at start of query should dominate"
                },
                {
                    "query": "proceed with feature selection",
                    "continuation_signal": "proceed",
                    "switching_signal": "feature selection",
                    "expected_winner": "SWITCHING",
                    "reasoning": "Proceed + different agent = switch to that agent"
                },
                {
                    "query": "skip this and continue preprocessing",
                    "continuation_signal": "skip this, continue preprocessing",
                    "switching_signal": None,
                    "expected_winner": "CONTINUATION",
                    "reasoning": "Both signals are for same agent (preprocessing)"
                },
                
                # POSITION-BASED PRIORITY TESTS
                {
                    "query": "train a model but first skip outliers",
                    "continuation_signal": "skip outliers",
                    "switching_signal": "train a model",
                    "expected_winner": "SWITCHING",
                    "reasoning": "Train model at start should win despite 'but first'"
                },
                {
                    "query": "skip outliers then maybe train a model",
                    "continuation_signal": "skip outliers", 
                    "switching_signal": "train a model",
                    "expected_winner": "CONTINUATION",
                    "reasoning": "Skip at start with weak 'maybe' should continue"
                },
                
                # SEMANTIC CONTEXT TESTS
                {
                    "query": "I want to select features after skipping outliers",
                    "continuation_signal": "skipping outliers",
                    "switching_signal": "select features",
                    "expected_winner": "SWITCHING",
                    "reasoning": "Clear intent to switch after current step"
                },
                {
                    "query": "continue with target column selection for features",
                    "continuation_signal": "continue, target column",
                    "switching_signal": "for features",
                    "expected_winner": "CONTINUATION",
                    "reasoning": "Target column is preprocessing task, 'for features' is purpose"
                },
                
                # AMBIGUOUS CASES (SYSTEM SHOULD CHOOSE CONSISTENTLY)
                {
                    "query": "skip this phase and analyze feature importance",
                    "continuation_signal": "skip this phase",
                    "switching_signal": "analyze feature importance",
                    "expected_winner": "SWITCHING",
                    "reasoning": "Feature importance analysis = feature selection agent"
                },
                {
                    "query": "proceed to next step which is feature selection",
                    "continuation_signal": "proceed to next step",
                    "switching_signal": "feature selection",
                    "expected_winner": "SWITCHING",
                    "reasoning": "Explicit mention of different agent"
                }
            ]
        },
        
        {
            "session_context": {
                "agent_type": "feature_selection", 
                "phase": "correlation_analysis",
                "data_state": "analyzing_correlations"
            },
            "conflict_cases": [
                {
                    "query": "skip correlation and train the model",
                    "continuation_signal": "skip correlation",
                    "switching_signal": "train the model",
                    "expected_winner": "SWITCHING",
                    "reasoning": "Model training = different agent (model building)"
                },
                {
                    "query": "continue with SHAP analysis and model building",
                    "continuation_signal": "continue with SHAP",
                    "switching_signal": "model building", 
                    "expected_winner": "CONTINUATION",
                    "reasoning": "SHAP is feature selection, model building is future intent"
                },
                {
                    "query": "run SHAP then clean the data",
                    "continuation_signal": "run SHAP",
                    "switching_signal": "clean the data",
                    "expected_winner": "CONTINUATION",
                    "reasoning": "SHAP is current agent, data cleaning is backward step"
                }
            ]
        }
    ]
    
    # Test results tracking
    results = {
        "total_conflicts": 0,
        "semantic_correct_resolution": 0,
        "keyword_correct_resolution": 0,
        "position_based_correct": 0,
        "context_based_correct": 0
    }
    
    for scenario in conflict_scenarios:
        session_context = scenario["session_context"]
        agent_type = session_context["agent_type"]
        
        print(f"\n🔍 TESTING CONFLICTS IN {agent_type.upper()} SESSION")
        print(f"   Current phase: {session_context['phase']}")
        print("-" * 60)
        
        for case in scenario["conflict_cases"]:
            results["total_conflicts"] += 1
            
            query = case["query"]
            expected_winner = case["expected_winner"]
            reasoning = case["reasoning"]
            
            print(f"\n📝 Conflict Query: '{query}'")
            print(f"   Continuation Signal: {case['continuation_signal']}")
            print(f"   Switching Signal: {case['switching_signal']}")
            print(f"   Expected Winner: {expected_winner}")
            print(f"   Reasoning: {reasoning}")
            
            # Test different resolution strategies
            semantic_resolution = test_semantic_conflict_resolution(query, session_context)
            keyword_resolution = test_keyword_conflict_resolution(query, session_context)
            position_resolution = test_position_based_resolution(query, session_context)
            context_resolution = test_context_based_resolution(query, session_context)
            
            # Evaluate results
            semantic_correct = (semantic_resolution == expected_winner)
            keyword_correct = (keyword_resolution == expected_winner)
            position_correct = (position_resolution == expected_winner)
            context_correct = (context_resolution == expected_winner)
            
            if semantic_correct:
                results["semantic_correct_resolution"] += 1
            if keyword_correct:
                results["keyword_correct_resolution"] += 1
            if position_correct:
                results["position_based_correct"] += 1
            if context_correct:
                results["context_based_correct"] += 1
            
            # Display results
            print(f"   🧠 Semantic: {semantic_resolution} {'✅' if semantic_correct else '❌'}")
            print(f"   ⚡ Keyword:  {keyword_resolution} {'✅' if keyword_correct else '❌'}")
            print(f"   📍 Position: {position_resolution} {'✅' if position_correct else '❌'}")
            print(f"   🎯 Context:  {context_resolution} {'✅' if context_correct else '❌'}")
            
            # Highlight best approach
            methods = [
                ("Semantic", semantic_correct),
                ("Keyword", keyword_correct), 
                ("Position", position_correct),
                ("Context", context_correct)
            ]
            
            correct_methods = [name for name, correct in methods if correct]
            if len(correct_methods) == 1:
                print(f"   🏆 ONLY {correct_methods[0].upper()} got it right!")
            elif len(correct_methods) > 1:
                print(f"   🤝 Multiple methods correct: {', '.join(correct_methods)}")
            else:
                print(f"   ❌ ALL METHODS FAILED - need better conflict resolution")
    
    # Final analysis
    analyze_conflict_resolution_results(results)
    
    return results

def test_semantic_conflict_resolution(query: str, session_context: dict) -> str:
    """Test semantic approach to conflict resolution"""
    query_lower = query.lower()
    
    # Semantic switching indicators (stronger signals)
    strong_switching = ['train', 'build', 'create', 'model', 'select features', 'feature selection']
    
    # Semantic continuation indicators
    continuation = ['skip', 'proceed', 'continue', 'next', 'target']
    
    # Check for strong switching signals
    has_strong_switch = any(signal in query_lower for signal in strong_switching)
    has_continuation = any(signal in query_lower for signal in continuation)
    
    if has_strong_switch and has_continuation:
        # Conflict detected - semantic priority to stronger signal
        return "SWITCHING" if has_strong_switch else "CONTINUATION"
    elif has_strong_switch:
        return "SWITCHING"
    elif has_continuation:
        return "CONTINUATION"
    else:
        return "UNCLEAR"

def test_keyword_conflict_resolution(query: str, session_context: dict) -> str:
    """Test keyword-based conflict resolution"""
    query_lower = query.lower()
    
    # Keyword patterns from original implementation
    switching_patterns = ['select features', 'train a', 'build a', 'create a']
    continuation_patterns = ['skip', 'proceed', 'continue', 'target']
    
    has_switching = any(pattern in query_lower for pattern in switching_patterns)
    has_continuation = any(pattern in query_lower for pattern in continuation_patterns)
    
    if has_switching and has_continuation:
        # Original logic: check position (first 20 characters)
        query_start = query_lower[:20]
        if any(pattern in query_start for pattern in switching_patterns):
            return "SWITCHING"
        else:
            return "CONTINUATION"
    elif has_switching:
        return "SWITCHING"
    elif has_continuation:
        return "CONTINUATION"
    else:
        return "UNCLEAR"

def test_position_based_resolution(query: str, session_context: dict) -> str:
    """Test position-based conflict resolution (what comes first wins)"""
    query_lower = query.lower()
    
    switching_terms = ['train', 'build', 'create', 'select features', 'model']
    continuation_terms = ['skip', 'proceed', 'continue', 'next']
    
    # Find positions of terms
    switch_positions = []
    continue_positions = []
    
    for term in switching_terms:
        pos = query_lower.find(term)
        if pos != -1:
            switch_positions.append(pos)
    
    for term in continuation_terms:
        pos = query_lower.find(term)
        if pos != -1:
            continue_positions.append(pos)
    
    if switch_positions and continue_positions:
        earliest_switch = min(switch_positions)
        earliest_continue = min(continue_positions)
        return "SWITCHING" if earliest_switch < earliest_continue else "CONTINUATION"
    elif switch_positions:
        return "SWITCHING"
    elif continue_positions:
        return "CONTINUATION"
    else:
        return "UNCLEAR"

def test_context_based_resolution(query: str, session_context: dict) -> str:
    """Test context-aware conflict resolution"""
    query_lower = query.lower()
    agent_type = session_context.get('agent_type')
    
    # Context-specific logic
    if agent_type == 'preprocessing':
        # In preprocessing, feature selection terms are strong switching signals
        if any(term in query_lower for term in ['features', 'feature selection', 'variables']):
            return "SWITCHING"
        elif any(term in query_lower for term in ['outliers', 'missing', 'target', 'clean']):
            return "CONTINUATION"
    
    elif agent_type == 'feature_selection':
        # In feature selection, model terms are strong switching signals
        if any(term in query_lower for term in ['train', 'model', 'build', 'predict']):
            return "SWITCHING"
        elif any(term in query_lower for term in ['shap', 'correlation', 'importance', 'select']):
            return "CONTINUATION"
    
    return "UNCLEAR"

def analyze_conflict_resolution_results(results: dict):
    """Analyze conflict resolution performance"""
    
    print("\n" + "=" * 80)
    print("⚖️  CONFLICT RESOLUTION ANALYSIS")
    print("=" * 80)
    
    total = results["total_conflicts"]
    if total == 0:
        print("No conflicts tested!")
        return
    
    semantic_acc = (results["semantic_correct_resolution"] / total) * 100
    keyword_acc = (results["keyword_correct_resolution"] / total) * 100
    position_acc = (results["position_based_correct"] / total) * 100
    context_acc = (results["context_based_correct"] / total) * 100
    
    print(f"\n🎯 CONFLICT RESOLUTION ACCURACY:")
    print(f"   🧠 Semantic Approach: {results['semantic_correct_resolution']}/{total} = {semantic_acc:.1f}%")
    print(f"   ⚡ Keyword Approach:  {results['keyword_correct_resolution']}/{total} = {keyword_acc:.1f}%")
    print(f"   📍 Position-Based:    {results['position_based_correct']}/{total} = {position_acc:.1f}%")
    print(f"   🎯 Context-Aware:     {results['context_based_correct']}/{total} = {context_acc:.1f}%")
    
    # Find best approach
    approaches = [
        ("Semantic", semantic_acc),
        ("Keyword", keyword_acc),
        ("Position-Based", position_acc),
        ("Context-Aware", context_acc)
    ]
    
    best_approach = max(approaches, key=lambda x: x[1])
    
    print(f"\n🏆 BEST APPROACH: {best_approach[0]} ({best_approach[1]:.1f}%)")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if best_approach[1] >= 80:
        print(f"   ✅ Use {best_approach[0]} as primary conflict resolution method")
    elif best_approach[1] >= 60:
        print(f"   🟡 Use {best_approach[0]} with fallback to other methods")
    else:
        print(f"   ⚠️  All methods need improvement - consider hybrid approach")
    
    print(f"\n🔍 INSIGHTS:")
    print(f"   📊 Conflict resolution is critical for user experience")
    print(f"   🎯 Context-aware methods likely perform better on ambiguous cases")
    print(f"   ⚡ Position-based rules provide consistent, predictable behavior")
    print(f"   🧠 Semantic understanding helps with natural language conflicts")
    
    print(f"\n✅ Conflict resolution analysis completed!")

if __name__ == "__main__":
    print("🚀 SESSION CONFLICT RESOLUTION TEST SUITE")
    print("=" * 80)
    print("⚖️  Testing conflict resolution strategies for session management")
    print("🎯 Focus: Queries with both continuation AND switching signals")
    print("=" * 80)
    
    results = test_session_conflict_resolution()
    
    print(f"\n🎉 Conflict resolution testing completed!")
    print(f"💡 Use results to implement robust session switching logic")
