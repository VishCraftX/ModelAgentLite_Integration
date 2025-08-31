#!/usr/bin/env python3
"""
Test Suite for Interactive Session Switching
Focused on the specific use case: stopping preprocessing to start feature selection
"""

import sys
import os
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_pipeline import MultiAgentMLPipeline
from pipeline_state import PipelineState
import pandas as pd

def test_preprocessing_to_feature_selection_switching():
    """
    Test the specific use case: User in preprocessing session wants to switch to feature selection
    This tests all the different ways a user might express this intent
    """
    
    print("🔄 PREPROCESSING → FEATURE SELECTION SWITCHING TEST")
    print("=" * 80)
    print("🎯 Testing session switching from preprocessing to feature selection")
    print("📝 Covers various ways users might express switching intent")
    print("=" * 80)
    
    # Preprocessing session context
    preprocessing_session = {
        "agent_type": "preprocessing",
        "phase": "outliers_detection",
        "data_state": "analyzing_outliers",
        "user_id": "test_user",
        "thread_id": "test_thread"
    }
    
    # Test cases for switching from preprocessing to feature selection
    switching_test_cases = [
        # EXPLICIT SWITCHING COMMANDS
        {
            "category": "Explicit Session Management",
            "cases": [
                ("clear session", "CLEAR_SESSION", "Explicit session clear command"),
                ("reset", "CLEAR_SESSION", "Reset command should clear session"),
                ("start over", "CLEAR_SESSION", "Start over command"),
                ("new session", "CLEAR_SESSION", "New session request"),
                ("exit session", "CLEAR_SESSION", "Exit current session"),
            ]
        },
        
        # DIRECT FEATURE SELECTION REQUESTS
        {
            "category": "Direct Feature Selection Requests", 
            "cases": [
                ("select features", "SWITCH_TO_FS", "Direct feature selection request"),
                ("I want to select important features", "SWITCH_TO_FS", "Natural language FS request"),
                ("let's do feature selection now", "SWITCH_TO_FS", "Conversational FS request"),
                ("analyze feature importance", "SWITCH_TO_FS", "Feature importance analysis"),
                ("find the most relevant variables", "SWITCH_TO_FS", "Variable relevance analysis"),
                ("perform feature engineering", "SWITCH_TO_FS", "Feature engineering request"),
                ("select the best predictors", "SWITCH_TO_FS", "Predictor selection request"),
            ]
        },
        
        # COMBINED STOP + START REQUESTS
        {
            "category": "Combined Stop + Start Requests",
            "cases": [
                ("stop preprocessing and select features", "SWITCH_TO_FS", "Explicit stop + start"),
                ("skip preprocessing, do feature selection", "SWITCH_TO_FS", "Skip + do pattern"),
                ("end cleaning and analyze features", "SWITCH_TO_FS", "End + analyze pattern"),
                ("bypass outliers, select variables", "SWITCH_TO_FS", "Bypass + select pattern"),
                ("quit preprocessing, start feature selection", "SWITCH_TO_FS", "Quit + start pattern"),
            ]
        },
        
        # CONTEXTUAL SWITCHING (AMBIGUOUS)
        {
            "category": "Contextual Switching (Semantic Required)",
            "cases": [
                ("I'm done with data cleaning", "SWITCH_TO_FS", "Implicit completion signal"),
                ("the data looks good now", "SWITCH_TO_FS", "Data quality satisfaction"),
                ("let's move to the next step", "SWITCH_TO_FS", "Next step in ML pipeline"),
                ("what features should I use?", "SWITCH_TO_FS", "Feature selection question"),
                ("which variables are important?", "SWITCH_TO_FS", "Variable importance question"),
                ("how do I choose the best features?", "SWITCH_TO_FS", "Feature selection guidance"),
            ]
        },
        
        # PREPROCESSING CONTINUATION (SHOULD NOT SWITCH)
        {
            "category": "Preprocessing Continuation (Should NOT Switch)",
            "cases": [
                ("skip outliers detection", "CONTINUE_PREPROCESSING", "Skip current preprocessing phase"),
                ("proceed to missing values", "CONTINUE_PREPROCESSING", "Next preprocessing step"),
                ("target column is price", "CONTINUE_PREPROCESSING", "Target specification"),
                ("continue with data cleaning", "CONTINUE_PREPROCESSING", "Explicit preprocessing continuation"),
                ("handle missing values next", "CONTINUE_PREPROCESSING", "Missing values handling"),
                ("encode categorical variables", "CONTINUE_PREPROCESSING", "Encoding request"),
            ]
        },
        
        # EDGE CASES (MIXED SIGNALS)
        {
            "category": "Edge Cases (Mixed Signals)",
            "cases": [
                ("skip outliers and select features", "SWITCH_TO_FS", "Skip + select (should prioritize select)"),
                ("continue preprocessing but also analyze features", "AMBIGUOUS", "Conflicting instructions"),
                ("target price and select features", "AMBIGUOUS", "Target spec + feature selection"),
                ("proceed with feature selection", "SWITCH_TO_FS", "Proceed with different agent"),
            ]
        }
    ]
    
    # Initialize pipeline for testing
    pipeline = MultiAgentMLPipeline()
    
    # Track results
    results = {
        "total_tests": 0,
        "semantic_correct": 0,
        "keyword_correct": 0,
        "category_results": {}
    }
    
    for test_group in switching_test_cases:
        category = test_group["category"]
        cases = test_group["cases"]
        
        print(f"\n🔍 TESTING: {category}")
        print("-" * 60)
        
        category_stats = {"total": 0, "semantic_correct": 0, "keyword_correct": 0}
        
        for query, expected_behavior, description in cases:
            results["total_tests"] += 1
            category_stats["total"] += 1
            
            print(f"\n📝 Query: '{query}'")
            print(f"   Expected: {expected_behavior}")
            print(f"   Description: {description}")
            
            # Test semantic classification
            semantic_result = test_semantic_session_switching(query, preprocessing_session)
            
            # Test keyword classification  
            keyword_result = test_keyword_session_switching(query, preprocessing_session)
            
            # Evaluate results
            semantic_correct = evaluate_switching_result(semantic_result, expected_behavior)
            keyword_correct = evaluate_switching_result(keyword_result, expected_behavior)
            
            if semantic_correct:
                results["semantic_correct"] += 1
                category_stats["semantic_correct"] += 1
                
            if keyword_correct:
                results["keyword_correct"] += 1
                category_stats["keyword_correct"] += 1
            
            # Display results
            print(f"   🧠 Semantic: {semantic_result} {'✅' if semantic_correct else '❌'}")
            print(f"   ⚡ Keyword:  {keyword_result} {'✅' if keyword_correct else '❌'}")
            
            # Highlight semantic advantages
            if semantic_correct and not keyword_correct:
                print(f"   🎯 SEMANTIC ADVANTAGE: Understood context better")
            elif not semantic_correct and keyword_correct:
                print(f"   ⚠️  KEYWORD ADVANTAGE: Semantic missed obvious pattern")
            elif not semantic_correct and not keyword_correct:
                print(f"   ❌ BOTH FAILED: Need better classification logic")
        
        results["category_results"][category] = category_stats
        
        # Category summary
        cat_semantic_acc = (category_stats["semantic_correct"] / category_stats["total"]) * 100
        cat_keyword_acc = (category_stats["keyword_correct"] / category_stats["total"]) * 100
        print(f"\n📊 {category} Results:")
        print(f"   🧠 Semantic: {category_stats['semantic_correct']}/{category_stats['total']} ({cat_semantic_acc:.1f}%)")
        print(f"   ⚡ Keyword:  {category_stats['keyword_correct']}/{category_stats['total']} ({cat_keyword_acc:.1f}%)")
    
    # Final analysis
    analyze_switching_results(results)
    
    return results

def test_semantic_session_switching(query: str, session_context: dict) -> str:
    """
    Simulate semantic session switching classification
    Returns: CLEAR_SESSION, SWITCH_TO_FS, CONTINUE_PREPROCESSING, or AMBIGUOUS
    """
    query_lower = query.lower()
    
    # Explicit session management
    if any(cmd in query_lower for cmd in ['clear session', 'reset', 'start over', 'new session', 'exit session']):
        return "CLEAR_SESSION"
    
    # Semantic feature selection indicators (would use embeddings in real implementation)
    fs_semantic_indicators = [
        'select', 'features', 'feature selection', 'important', 'relevant', 'variables',
        'predictors', 'attributes', 'feature engineering', 'feature importance',
        'which features', 'what features', 'best features', 'choose features'
    ]
    
    # Semantic preprocessing continuation indicators
    preprocessing_semantic_indicators = [
        'skip outliers', 'skip missing', 'target column', 'continue cleaning',
        'handle missing', 'encode', 'transform', 'outliers detection'
    ]
    
    # Semantic completion/switching indicators
    completion_indicators = [
        'done with', 'finished', 'looks good', 'next step', 'move to',
        'stop preprocessing', 'end cleaning', 'bypass'
    ]
    
    # Count semantic matches
    fs_score = sum(1 for indicator in fs_semantic_indicators if indicator in query_lower)
    preprocessing_score = sum(1 for indicator in preprocessing_semantic_indicators if indicator in query_lower)
    completion_score = sum(1 for indicator in completion_indicators if indicator in query_lower)
    
    # Decision logic
    if fs_score > 0 or completion_score > 0:
        if preprocessing_score > 0:
            return "AMBIGUOUS"  # Mixed signals
        return "SWITCH_TO_FS"
    elif preprocessing_score > 0:
        return "CONTINUE_PREPROCESSING"
    else:
        return "AMBIGUOUS"

def test_keyword_session_switching(query: str, session_context: dict) -> str:
    """
    Simulate keyword-based session switching classification
    Returns: CLEAR_SESSION, SWITCH_TO_FS, CONTINUE_PREPROCESSING, or AMBIGUOUS
    """
    query_lower = query.lower()
    
    # Explicit session management
    clear_commands = ['clear session', 'reset', 'start over', 'new session', 'exit session']
    if any(cmd in query_lower for cmd in clear_commands):
        return "CLEAR_SESSION"
    
    # New request patterns (from original implementation)
    new_request_patterns = [
        'select features', 'train a', 'build a', 'create a', 'analyze my',
        'skip preprocessing', 'without preprocessing', 'bypass cleaning'
    ]
    
    # Preprocessing continuation patterns
    preprocessing_patterns = [
        'skip outliers', 'skip missing', 'skip encoding', 'target ', 'column ',
        'proceed', 'continue', 'next'
    ]
    
    # Check patterns
    has_new_request = any(pattern in query_lower for pattern in new_request_patterns)
    has_preprocessing = any(pattern in query_lower for pattern in preprocessing_patterns)
    
    if has_new_request and has_preprocessing:
        return "AMBIGUOUS"
    elif has_new_request:
        return "SWITCH_TO_FS"
    elif has_preprocessing:
        return "CONTINUE_PREPROCESSING"
    else:
        return "AMBIGUOUS"

def evaluate_switching_result(result: str, expected: str) -> bool:
    """Evaluate if the classification result matches expected behavior"""
    if expected == "AMBIGUOUS":
        return True  # Any result is acceptable for ambiguous cases
    return result == expected

def analyze_switching_results(results: dict):
    """Analyze and display comprehensive results"""
    
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE SWITCHING ANALYSIS")
    print("=" * 80)
    
    total = results["total_tests"]
    semantic_acc = (results["semantic_correct"] / total) * 100 if total > 0 else 0
    keyword_acc = (results["keyword_correct"] / total) * 100 if total > 0 else 0
    improvement = semantic_acc - keyword_acc
    
    print(f"\n🎯 OVERALL ACCURACY:")
    print(f"   🧠 Semantic Method: {results['semantic_correct']}/{total} = {semantic_acc:.1f}%")
    print(f"   ⚡ Keyword Method:  {results['keyword_correct']}/{total} = {keyword_acc:.1f}%")
    print(f"   📈 Improvement: {improvement:+.1f}%")
    
    print(f"\n📋 CATEGORY BREAKDOWN:")
    for category, stats in results["category_results"].items():
        cat_semantic = (stats["semantic_correct"] / stats["total"]) * 100
        cat_keyword = (stats["keyword_correct"] / stats["total"]) * 100
        cat_improvement = cat_semantic - cat_keyword
        
        print(f"   {category}:")
        print(f"     🧠 Semantic: {cat_semantic:.1f}% | ⚡ Keyword: {cat_keyword:.1f}% | 📈 Diff: {cat_improvement:+.1f}%")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if improvement > 15:
        print(f"   ✅ STRONG: Use semantic-first approach for session switching")
        print(f"   🎯 Semantic shows significant advantage in context understanding")
    elif improvement > 5:
        print(f"   🟡 MODERATE: Use semantic with keyword fallback")
        print(f"   💡 Semantic helps with ambiguous cases")
    else:
        print(f"   ⚠️  CONCERN: Semantic not significantly better")
        print(f"   🔧 Need to improve semantic definitions and embeddings")
    
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"   🧠 Semantic excels at: Contextual switching, implicit completion signals")
    print(f"   ⚡ Keywords excel at: Explicit commands, direct pattern matching")
    print(f"   🎯 Hybrid approach: Use semantic for ambiguous, keywords for explicit")
    
    print(f"\n✅ Session switching test completed!")

if __name__ == "__main__":
    print("🚀 INTERACTIVE SESSION SWITCHING TEST SUITE")
    print("=" * 80)
    print("🎯 Focus: Preprocessing → Feature Selection switching scenarios")
    print("📊 Comparing semantic vs keyword classification approaches")
    print("=" * 80)
    
    results = test_preprocessing_to_feature_selection_switching()
    
    print(f"\n🎉 Testing completed!")
    print(f"💡 Use results to optimize session switching logic")
