#!/usr/bin/env python3
"""
Comprehensive Orchestrator Performance Test
Tests the real orchestrator with all three methods and provides detailed analysis
"""

import sys
import os
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator
from pipeline_state import PipelineState
import time

def test_comprehensive_orchestrator_performance():
    """
    Test the real orchestrator performance with detailed method tracking
    Shows true Semantic → LLM → Keyword performance
    """
    
    print("🎯 COMPREHENSIVE ORCHESTRATOR PERFORMANCE TEST")
    print("=" * 80)
    print("🧠 Testing REAL orchestrator with Semantic → LLM → Keyword flow")
    print("📊 Detailed method tracking and performance analysis")
    print("=" * 80)
    
    # Initialize real orchestrator
    orchestrator = Orchestrator()
    
    # Comprehensive test cases across all intents
    test_cases = [
        # PREPROCESSING CASES
        ("clean my dataset", "preprocessing", "Clear preprocessing request"),
        ("remove outliers from data", "preprocessing", "Outlier removal"),
        ("handle missing values", "preprocessing", "Missing values"),
        ("normalize the data", "preprocessing", "Data normalization"),
        ("prepare data for analysis", "preprocessing", "Data preparation"),
        
        # FEATURE SELECTION CASES  
        ("select important features", "feature_selection", "Feature selection"),
        ("analyze feature importance", "feature_selection", "Feature importance"),
        ("find relevant variables", "feature_selection", "Variable selection"),
        ("perform feature engineering", "feature_selection", "Feature engineering"),
        ("reduce dimensionality", "feature_selection", "Dimensionality reduction"),
        
        # MODEL BUILDING CASES
        ("train a machine learning model", "model_building", "Model training"),
        ("build a classifier", "model_building", "Classifier building"),
        ("create prediction model", "model_building", "Prediction model"),
        ("develop neural network", "model_building", "Neural network"),
        ("optimize hyperparameters", "model_building", "Hyperparameter tuning"),
        
        # CODE EXECUTION CASES
        ("generate scatter plot", "code_execution", "Visualization"),
        ("calculate correlation matrix", "code_execution", "Statistical calculation"),
        ("create histogram", "code_execution", "Data visualization"),
        ("compute descriptive statistics", "code_execution", "Statistical analysis"),
        ("run custom analysis", "code_execution", "Custom code execution"),
        
        # GENERAL QUERY CASES
        ("what can you help me with", "general_query", "Capability inquiry"),
        ("how does this system work", "general_query", "System explanation"),
        ("hello there", "general_query", "Greeting"),
        ("explain machine learning", "general_query", "Educational query"),
        ("what are your features", "general_query", "Feature inquiry"),
        
        # AMBIGUOUS/CHALLENGING CASES
        ("analyze my data thoroughly", "code_execution", "Ambiguous analysis"),
        ("improve my model performance", "model_building", "Performance improvement"),
        ("work with my dataset", "preprocessing", "Generic data work"),
        ("help me with features", "feature_selection", "Feature help"),
        ("make my data better", "preprocessing", "Data improvement"),
    ]
    
    # Track detailed results
    results = {
        "total_tests": 0,
        "correct_classifications": 0,
        "method_usage": {"semantic": 0, "llm": 0, "keyword": 0},
        "method_accuracy": {"semantic": {"correct": 0, "total": 0}, 
                           "llm": {"correct": 0, "total": 0},
                           "keyword": {"correct": 0, "total": 0}},
        "intent_performance": {},
        "response_times": {"semantic": [], "llm": [], "keyword": []},
        "detailed_results": []
    }
    
    print(f"\n🧪 Testing {len(test_cases)} cases with real orchestrator...\n")
    
    for i, (query, expected_intent, description) in enumerate(test_cases, 1):
        results["total_tests"] += 1
        
        print(f"📝 Test {i}: '{query}'")
        print(f"   Expected: {expected_intent}")
        print(f"   Description: {description}")
        
        # Create state and test
        state = PipelineState(user_query=query)
        
        try:
            # Time the classification
            start_time = time.time()
            
            # Get the actual classification with detailed tracking
            actual_intent, method_used, confidence_info = classify_with_method_tracking(orchestrator, state)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Evaluate correctness
            is_correct = (actual_intent == expected_intent)
            
            if is_correct:
                results["correct_classifications"] += 1
                print(f"   ✅ CORRECT: {actual_intent}")
            else:
                print(f"   ❌ WRONG: Got {actual_intent}, expected {expected_intent}")
            
            # Track method usage
            results["method_usage"][method_used] += 1
            results["method_accuracy"][method_used]["total"] += 1
            if is_correct:
                results["method_accuracy"][method_used]["correct"] += 1
            
            # Track response time
            results["response_times"][method_used].append(response_time)
            
            # Track by intent
            if expected_intent not in results["intent_performance"]:
                results["intent_performance"][expected_intent] = {"correct": 0, "total": 0}
            results["intent_performance"][expected_intent]["total"] += 1
            if is_correct:
                results["intent_performance"][expected_intent]["correct"] += 1
            
            # Store detailed result
            results["detailed_results"].append({
                "query": query,
                "expected": expected_intent,
                "actual": actual_intent,
                "method": method_used,
                "correct": is_correct,
                "time": response_time,
                "confidence": confidence_info
            })
            
            print(f"   📊 Method: {get_method_emoji(method_used)} {method_used.title()}")
            print(f"   ⏱️ Time: {response_time:.3f}s")
            if confidence_info:
                print(f"   🎯 Confidence: {confidence_info}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        
        print()
    
    # Comprehensive analysis
    analyze_comprehensive_results(results)
    
    return results

def classify_with_method_tracking(orchestrator, state):
    """
    Classify with detailed method tracking to see which method was actually used
    """
    
    query = state.user_query
    
    # Try to trace the actual orchestrator logic
    try:
        # Check semantic first
        if hasattr(orchestrator, '_intent_embeddings') and orchestrator._intent_embeddings:
            semantic_intent, semantic_confidence = orchestrator._classify_with_semantic_similarity(query)
            
            if semantic_confidence.get("threshold_met", False):
                # Semantic was used
                return semantic_intent, "semantic", semantic_confidence
        
        # If semantic failed, orchestrator tries LLM
        try:
            context = {
                "has_raw_data": False,
                "has_cleaned_data": False,
                "has_selected_features": False,
                "has_trained_model": False
            }
            llm_intent = orchestrator.classify_intent_with_llm(query, context)
            
            if llm_intent and llm_intent != "error":
                # LLM was used
                return llm_intent, "llm", {"method": "llm_classification"}
        except:
            pass
        
        # If both failed, use keyword fallback
        keyword_intent, keyword_confidence = orchestrator._classify_with_keyword_scoring(query)
        return keyword_intent, "keyword", keyword_confidence
        
    except Exception as e:
        # Fallback to just getting the result
        result = orchestrator.route(state)
        return result, "unknown", {"error": str(e)}

def get_method_emoji(method):
    """Get emoji for method"""
    emojis = {"semantic": "🧠", "llm": "🤖", "keyword": "⚡"}
    return emojis.get(method, "❓")

def analyze_comprehensive_results(results):
    """Comprehensive analysis of orchestrator performance"""
    
    print("=" * 80)
    print("🎯 COMPREHENSIVE ORCHESTRATOR ANALYSIS")
    print("=" * 80)
    
    total = results["total_tests"]
    correct = results["correct_classifications"]
    overall_accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Total Tests: {total}")
    print(f"   Correct Classifications: {correct}")
    print(f"   Overall Accuracy: {overall_accuracy:.1f}%")
    
    print(f"\n🎯 METHOD USAGE DISTRIBUTION:")
    for method, count in results["method_usage"].items():
        percentage = (count / total) * 100 if total > 0 else 0
        emoji = get_method_emoji(method)
        print(f"   {emoji} {method.title()}: {count}/{total} = {percentage:.1f}%")
    
    print(f"\n📈 METHOD ACCURACY:")
    for method in ["semantic", "llm", "keyword"]:
        method_stats = results["method_accuracy"][method]
        if method_stats["total"] > 0:
            accuracy = (method_stats["correct"] / method_stats["total"]) * 100
            emoji = get_method_emoji(method)
            print(f"   {emoji} {method.title()}: {method_stats['correct']}/{method_stats['total']} = {accuracy:.1f}%")
    
    print(f"\n⏱️ METHOD SPEED:")
    for method in ["semantic", "llm", "keyword"]:
        times = results["response_times"][method]
        if times:
            avg_time = sum(times) / len(times)
            emoji = get_method_emoji(method)
            print(f"   {emoji} {method.title()}: {avg_time:.3f}s average")
    
    print(f"\n🎯 INTENT PERFORMANCE:")
    for intent, stats in results["intent_performance"].items():
        accuracy = (stats["correct"] / stats["total"]) * 100
        print(f"   {intent}: {stats['correct']}/{stats['total']} = {accuracy:.1f}%")
    
    print(f"\n💡 KEY INSIGHTS:")
    
    # Method distribution analysis
    semantic_usage = results["method_usage"]["semantic"]
    llm_usage = results["method_usage"]["llm"] 
    keyword_usage = results["method_usage"]["keyword"]
    
    semantic_pct = (semantic_usage / total) * 100
    llm_pct = (llm_usage / total) * 100
    keyword_pct = (keyword_usage / total) * 100
    
    if semantic_pct >= 60:
        print(f"   ✅ EXCELLENT semantic usage ({semantic_pct:.1f}%)")
    elif semantic_pct >= 40:
        print(f"   🟡 GOOD semantic usage ({semantic_pct:.1f}%)")
    else:
        print(f"   ⚠️ LOW semantic usage ({semantic_pct:.1f}%)")
    
    if llm_pct <= 30:
        print(f"   ✅ EFFICIENT LLM usage ({llm_pct:.1f}%) - not overused")
    else:
        print(f"   ⚠️ HIGH LLM usage ({llm_pct:.1f}%) - may be expensive")
    
    if keyword_pct <= 25:
        print(f"   ✅ MINIMAL keyword fallback ({keyword_pct:.1f}%)")
    else:
        print(f"   ⚠️ HIGH keyword fallback ({keyword_pct:.1f}%)")
    
    print(f"\n🏆 PERFORMANCE RATING:")
    if overall_accuracy >= 85 and semantic_pct >= 60:
        print(f"   🥇 EXCELLENT: High accuracy with good semantic usage")
    elif overall_accuracy >= 75 and semantic_pct >= 40:
        print(f"   🥈 GOOD: Decent performance with room for improvement")
    else:
        print(f"   🥉 NEEDS WORK: Significant improvements needed")
    
    print(f"\n✅ Comprehensive orchestrator analysis completed!")

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE ORCHESTRATOR PERFORMANCE TEST")
    print("=" * 80)
    print("🎯 Testing real Semantic → LLM → Keyword orchestrator flow")
    print("📊 Detailed method tracking and performance analysis")
    print("=" * 80)
    
    results = test_comprehensive_orchestrator_performance()
    
    print(f"\n🎉 Comprehensive testing completed!")
    print(f"💡 This shows the true performance of the optimized orchestrator")
