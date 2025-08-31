#!/usr/bin/env python3
"""
Independent Method Comparison Test Suite
Tests semantic, keyword, and LLM classification methods independently to determine optimal ranking
"""

import sys
import os
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator
from pipeline_state import PipelineState
import time

def test_all_methods_independently():
    """Test all three classification methods independently and compare their performance"""
    
    print("🔬 INDEPENDENT METHOD COMPARISON TEST SUITE")
    print("=" * 80)
    print("🎯 Testing semantic, keyword, and LLM methods independently")
    print("📊 Measuring accuracy, speed, and reliability for optimal ranking")
    print("=" * 80)
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Comprehensive test cases with expected results
    test_cases = [
        # Clear preprocessing cases
        ("Preprocessing - Clean", "clean my dataset and remove duplicates", "preprocessing"),
        ("Preprocessing - Sanitize", "sanitize data by handling missing values", "preprocessing"),
        ("Preprocessing - Prepare", "prepare data for machine learning analysis", "preprocessing"),
        ("Preprocessing - Handle", "handle outliers and inconsistent data", "preprocessing"),
        ("Preprocessing - Process", "process raw data files for analysis", "preprocessing"),
        
        # Clear feature selection cases  
        ("Feature Selection - Select", "select the most important features", "feature_selection"),
        ("Feature Selection - Engineer", "engineer new variables from existing data", "feature_selection"),
        ("Feature Selection - Analyze", "analyze feature importance and correlation", "feature_selection"),
        ("Feature Selection - Choose", "choose relevant attributes for modeling", "feature_selection"),
        ("Feature Selection - Identify", "identify key predictors in dataset", "feature_selection"),
        
        # Clear model building cases
        ("Model Building - Train", "train machine learning algorithms", "model_building"),
        ("Model Building - Build", "build predictive models for classification", "model_building"),
        ("Model Building - Create", "create neural networks for prediction", "model_building"),
        ("Model Building - Develop", "develop ensemble methods for forecasting", "model_building"),
        ("Model Building - Construct", "construct decision trees for analysis", "model_building"),
        
        # Clear code execution cases
        ("Code Execution - Calculate", "calculate statistical summaries", "code_execution"),
        ("Code Execution - Visualize", "visualize data distributions", "code_execution"),
        ("Code Execution - Analyze", "analyze dataset characteristics", "code_execution"),
        ("Code Execution - Plot", "plot correlation matrices", "code_execution"),
        ("Code Execution - Compute", "compute descriptive statistics", "code_execution"),
        
        # Clear general query cases
        ("General Query - Help", "what can you help me with", "general_query"),
        ("General Query - Capabilities", "what are your capabilities", "general_query"),
        ("General Query - Greeting", "hello, how are you", "general_query"),
        ("General Query - Status", "show me current status", "general_query"),
        ("General Query - About", "tell me about this system", "general_query"),
        
        # Challenging/ambiguous cases
        ("Ambiguous - Data Work", "work with my data effectively", "preprocessing"),  # Could be multiple
        ("Ambiguous - Improve Model", "improve my model performance", "model_building"),  # Could be feature selection
        ("Ambiguous - Analyze Features", "analyze my features thoroughly", "feature_selection"),  # Could be code execution
        ("Ambiguous - Data Analysis", "perform comprehensive data analysis", "code_execution"),  # Could be preprocessing
        ("Ambiguous - ML Pipeline", "help me with ML pipeline", "general_query"),  # Could be multiple
        
        # Edge cases with synonyms
        ("Synonym - Purify", "purify my dataset from errors", "preprocessing"),
        ("Synonym - Craft", "craft meaningful variables", "feature_selection"), 
        ("Synonym - Architect", "architect prediction systems", "model_building"),
        ("Synonym - Investigate", "investigate data patterns", "code_execution"),
        ("Synonym - Assist", "assist me with my project", "general_query"),
    ]
    
    # Initialize results tracking
    results = {
        "semantic": {"correct": 0, "total": 0, "times": [], "errors": 0, "confidence_scores": []},
        "keyword": {"correct": 0, "total": 0, "times": [], "errors": 0, "confidence_scores": []},
        "llm": {"correct": 0, "total": 0, "times": [], "errors": 0, "confidence_scores": []}
    }
    
    print(f"\n🧪 Testing {len(test_cases)} cases with each method independently...\n")
    
    for i, (test_name, query, expected_intent) in enumerate(test_cases, 1):
        print(f"📝 Test {i}/{len(test_cases)}: {test_name}")
        print(f"   Query: '{query}'")
        print(f"   Expected: {expected_intent}")
        
        # Test 1: Semantic Classification
        semantic_result, semantic_time, semantic_confidence = test_semantic_method(orchestrator, query)
        results["semantic"]["total"] += 1
        results["semantic"]["times"].append(semantic_time)
        
        if semantic_result == expected_intent:
            results["semantic"]["correct"] += 1
            print(f"   🧠 Semantic: ✅ {semantic_result} ({semantic_time:.3f}s, conf:{semantic_confidence:.3f})")
        else:
            print(f"   🧠 Semantic: ❌ {semantic_result} (expected {expected_intent}) ({semantic_time:.3f}s)")
        
        if semantic_confidence is not None:
            results["semantic"]["confidence_scores"].append(semantic_confidence)
        
        # Test 2: Keyword Classification  
        keyword_result, keyword_time, keyword_confidence = test_keyword_method(orchestrator, query)
        results["keyword"]["total"] += 1
        results["keyword"]["times"].append(keyword_time)
        
        if keyword_result == expected_intent:
            results["keyword"]["correct"] += 1
            print(f"   ⚡ Keyword: ✅ {keyword_result} ({keyword_time:.3f}s, conf:{keyword_confidence:.3f})")
        else:
            print(f"   ⚡ Keyword: ❌ {keyword_result} (expected {expected_intent}) ({keyword_time:.3f}s)")
            
        if keyword_confidence is not None:
            results["keyword"]["confidence_scores"].append(keyword_confidence)
        
        # Test 3: LLM Classification
        llm_result, llm_time, llm_confidence = test_llm_method(orchestrator, query)
        results["llm"]["total"] += 1
        results["llm"]["times"].append(llm_time)
        
        if llm_result == expected_intent:
            results["llm"]["correct"] += 1
            print(f"   🤖 LLM: ✅ {llm_result} ({llm_time:.3f}s)")
        else:
            print(f"   🤖 LLM: ❌ {llm_result} (expected {expected_intent}) ({llm_time:.3f}s)")
        
        print()
    
    # Analyze and compare results
    analyze_method_comparison(results)
    
    return results

def test_semantic_method(orchestrator, query):
    """Test semantic classification method independently"""
    start_time = time.time()
    
    try:
        if hasattr(orchestrator, '_intent_embeddings') and orchestrator._intent_embeddings:
            intent, confidence_info = orchestrator._classify_with_semantic_similarity(query)
            confidence = confidence_info.get("max_score", 0.0)
        else:
            intent = "general_query"  # Fallback if no embeddings
            confidence = 0.0
            
        end_time = time.time()
        return intent, end_time - start_time, confidence
        
    except Exception as e:
        end_time = time.time()
        return "error", end_time - start_time, 0.0

def test_keyword_method(orchestrator, query):
    """Test keyword classification method independently"""
    start_time = time.time()
    
    try:
        intent, confidence_info = orchestrator._classify_with_keyword_scoring(query)
        confidence = confidence_info.get("max_score", 0.0)
        end_time = time.time()
        return intent, end_time - start_time, confidence
        
    except Exception as e:
        end_time = time.time()
        return "error", end_time - start_time, 0.0

def test_llm_method(orchestrator, query):
    """Test LLM classification method independently"""
    start_time = time.time()
    
    try:
        # Create mock state for LLM context
        mock_state = PipelineState(user_query=query)
        context = {
            "has_raw_data": False,
            "has_cleaned_data": False, 
            "has_selected_features": False,
            "has_trained_model": False
        }
        
        intent = orchestrator.classify_intent_with_llm(query, context)
        end_time = time.time()
        return intent, end_time - start_time, 1.0  # LLM doesn't provide confidence scores
        
    except Exception as e:
        end_time = time.time()
        return "error", end_time - start_time, 0.0

def analyze_method_comparison(results):
    """Analyze and compare the performance of all three methods"""
    
    print("=" * 80)
    print("📊 COMPREHENSIVE METHOD COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Calculate metrics for each method
    methods = ["semantic", "keyword", "llm"]
    method_names = {"semantic": "🧠 Semantic (BGE-Large)", "keyword": "⚡ Keyword Matching", "llm": "🤖 LLM (Qwen)"}
    
    print(f"\n🎯 ACCURACY COMPARISON:")
    accuracies = {}
    for method in methods:
        if results[method]["total"] > 0:
            accuracy = results[method]["correct"] / results[method]["total"] * 100
            accuracies[method] = accuracy
            print(f"   {method_names[method]}: {results[method]['correct']}/{results[method]['total']} = {accuracy:.1f}%")
        else:
            accuracies[method] = 0.0
    
    print(f"\n⚡ SPEED COMPARISON:")
    speeds = {}
    for method in methods:
        if results[method]["times"]:
            avg_time = sum(results[method]["times"]) / len(results[method]["times"])
            speeds[method] = avg_time
            print(f"   {method_names[method]}: {avg_time:.3f}s average")
        else:
            speeds[method] = float('inf')
    
    print(f"\n🎯 CONFIDENCE COMPARISON:")
    confidences = {}
    for method in methods:
        if results[method]["confidence_scores"]:
            avg_confidence = sum(results[method]["confidence_scores"]) / len(results[method]["confidence_scores"])
            confidences[method] = avg_confidence
            print(f"   {method_names[method]}: {avg_confidence:.3f} average confidence")
        else:
            confidences[method] = 0.0
    
    # Determine optimal ranking
    print(f"\n🏆 OPTIMAL METHOD RANKING:")
    
    # Rank by accuracy (primary), then speed (secondary), then confidence (tertiary)
    method_scores = []
    for method in methods:
        # Composite score: accuracy (70%) + speed_score (20%) + confidence (10%)
        accuracy_score = accuracies[method]
        speed_score = (1 / speeds[method]) * 1000 if speeds[method] > 0 else 0  # Invert speed (faster = higher score)
        confidence_score = confidences[method] * 100
        
        composite_score = (accuracy_score * 0.7) + (speed_score * 0.2) + (confidence_score * 0.1)
        method_scores.append((method, composite_score, accuracy_score, speeds[method], confidences[method]))
    
    # Sort by composite score (descending)
    method_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (method, composite_score, accuracy, speed, confidence) in enumerate(method_scores, 1):
        rank_emoji = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
        print(f"   {rank_emoji} {method_names[method]}")
        print(f"       Accuracy: {accuracy:.1f}% | Speed: {speed:.3f}s | Confidence: {confidence:.3f}")
        print(f"       Composite Score: {composite_score:.2f}")
        print()
    
    # Recommendations
    print(f"💡 RECOMMENDATIONS:")
    best_method = method_scores[0][0]
    best_accuracy = method_scores[0][2]
    
    if best_accuracy >= 90:
        print(f"   ✅ Use {method_names[best_method]} as PRIMARY method")
        if len(method_scores) > 1:
            second_best = method_scores[1][0]
            print(f"   ✅ Use {method_names[second_best]} as FALLBACK method")
    elif best_accuracy >= 75:
        print(f"   🟡 Use {method_names[best_method]} as PRIMARY with caution")
        print(f"   💡 Consider hybrid approach with multiple methods")
    else:
        print(f"   ⚠️  All methods show suboptimal performance")
        print(f"   💡 Consider improving intent definitions or training data")
    
    # Specific insights
    print(f"\n🔍 METHOD-SPECIFIC INSIGHTS:")
    
    semantic_accuracy = accuracies.get("semantic", 0)
    keyword_accuracy = accuracies.get("keyword", 0) 
    llm_accuracy = accuracies.get("llm", 0)
    
    if semantic_accuracy > keyword_accuracy + 10:
        print("   🧠 Semantic embeddings significantly outperform keywords")
        print("   💡 Prioritize semantic classification over keyword matching")
    
    if llm_accuracy > max(semantic_accuracy, keyword_accuracy) + 5:
        print("   🤖 LLM shows superior understanding of complex queries")
        print("   💡 Consider LLM-first approach for ambiguous cases")
    
    if speeds["keyword"] < speeds["semantic"] / 2:
        print("   ⚡ Keywords provide significant speed advantage")
        print("   💡 Use keywords for real-time/high-throughput scenarios")
    
    return method_scores

if __name__ == "__main__":
    print("🚀 INDEPENDENT METHOD COMPARISON TEST SUITE")
    print("=" * 80)
    print("🎯 Objective: Determine optimal classification method ranking")
    print("📊 Metrics: Accuracy, Speed, Confidence, Composite Score")
    print("🔬 Approach: Independent testing (not hierarchical fallback)")
    print("=" * 80)
    
    results = test_all_methods_independently()
    
    print("\n✅ Method comparison completed!")
    print("💡 Use results to optimize orchestrator method priority")
