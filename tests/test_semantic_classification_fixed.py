#!/usr/bin/env python3
"""
FIXED Semantic Classification Test
Tests the ACTUAL orchestrator behavior with CORRECT method tracking
"""

import sys
import os
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import Orchestrator
from pipeline_state import PipelineState
import io
import contextlib

def test_semantic_classification_fixed():
    """Test semantic classification with CORRECT method tracking"""
    
    print("🧠 FIXED SEMANTIC CLASSIFICATION TEST")
    print("=" * 80)
    print("🎯 Tests ACTUAL orchestrator with CORRECT method tracking")
    print("📊 Captures real Semantic → LLM → Keyword flow")
    print("=" * 80)
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Track statistics
    stats = {
        "total_tests": 0,
        "semantic_used": 0,
        "llm_used": 0,
        "keyword_used": 0,
        "correct_classifications": 0
    }
    
    # Test cases
    test_cases = [
        # Clear cases (should use semantic)
        ("clean my dataset", "preprocessing", "Clear preprocessing"),
        ("select important features", "feature_selection", "Clear feature selection"),
        ("train a model", "model_building", "Clear model building"),
        ("create a scatter plot", "code_execution", "Clear code execution"),
        ("what can you help with", "general_query", "Clear general query"),
        
        # Ambiguous cases (might use LLM)
        ("skip preprocessing and go straight to modeling", "model_building", "Skip preprocessing"),
        ("avoid feature engineering, use raw variables", "model_building", "Avoid feature engineering"),
        ("analyze my data thoroughly", "code_execution", "Ambiguous analysis"),
        
        # Edge cases (might use keyword fallback)
        ("gonna clean my dataset real quick", "preprocessing", "Informal preprocessing"),
        ("help me with ML stuff", "general_query", "Vague help request"),
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} cases with REAL method tracking...\n")
    
    for i, (query, expected_intent, description) in enumerate(test_cases, 1):
        stats["total_tests"] += 1
        
        print(f"📝 Test {i}: '{query}'")
        print(f"   Expected: {expected_intent}")
        print(f"   Description: {description}")
        
        # Create state
        state = PipelineState(user_query=query)
        
        # Capture orchestrator output to analyze method used
        output_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(output_buffer):
                actual_intent = orchestrator.route(state)
            
            # Get the captured output
            orchestrator_output = output_buffer.getvalue()
            
            # Analyze which method was ACTUALLY used based on orchestrator logs
            method_used = analyze_method_from_logs(orchestrator_output)
            
            # Track method usage
            if method_used == "semantic":
                stats["semantic_used"] += 1
            elif method_used == "llm":
                stats["llm_used"] += 1
            elif method_used == "keyword":
                stats["keyword_used"] += 1
            
            # Check correctness
            is_correct = (actual_intent == expected_intent)
            if is_correct:
                stats["correct_classifications"] += 1
                print(f"   ✅ CORRECT: {actual_intent}")
            else:
                print(f"   ❌ WRONG: Got {actual_intent}, expected {expected_intent}")
            
            print(f"   📊 Method: {get_method_emoji(method_used)} {method_used.title()}")
            
            # Show orchestrator logs for verification
            if orchestrator_output.strip():
                print(f"   🔍 Orchestrator logs:")
                for line in orchestrator_output.strip().split('\n'):
                    if line.strip():
                        print(f"      {line}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        
        print()
    
    # Analysis
    analyze_fixed_results(stats)
    
    return stats

def analyze_method_from_logs(output):
    """Analyze which method was used based on orchestrator log output"""
    
    if "🧠 Semantic classification accepted" in output:
        return "semantic"
    elif "🤖 Using LLM for ambiguous/complex query" in output:
        return "llm"
    elif "⚡ Using keyword fallback as last resort" in output:
        return "keyword"
    else:
        # If no clear indication, try to infer from other patterns
        if "Semantic classification:" in output and "accepted" in output:
            return "semantic"
        elif "LLM classification:" in output:
            return "llm"
        elif "Keyword classification:" in output:
            return "keyword"
        else:
            return "unknown"

def get_method_emoji(method):
    """Get emoji for method"""
    emojis = {"semantic": "🧠", "llm": "🤖", "keyword": "⚡", "unknown": "❓"}
    return emojis.get(method, "❓")

def analyze_fixed_results(stats):
    """Analyze the corrected test results"""
    
    print("=" * 80)
    print("📊 FIXED SEMANTIC CLASSIFICATION ANALYSIS")
    print("=" * 80)
    
    total = stats["total_tests"]
    semantic_pct = (stats["semantic_used"] / total) * 100 if total > 0 else 0
    llm_pct = (stats["llm_used"] / total) * 100 if total > 0 else 0
    keyword_pct = (stats["keyword_used"] / total) * 100 if total > 0 else 0
    accuracy = (stats["correct_classifications"] / total) * 100 if total > 0 else 0
    
    print(f"\n🎯 CORRECTED STATISTICS:")
    print(f"   Total Tests: {total}")
    print(f"   🧠 Semantic Used: {stats['semantic_used']}/{total} = {semantic_pct:.1f}%")
    print(f"   🤖 LLM Used: {stats['llm_used']}/{total} = {llm_pct:.1f}%")
    print(f"   ⚡ Keyword Used: {stats['keyword_used']}/{total} = {keyword_pct:.1f}%")
    print(f"   🎯 Overall Accuracy: {stats['correct_classifications']}/{total} = {accuracy:.1f}%")
    
    print(f"\n💡 CORRECTED INSIGHTS:")
    
    if semantic_pct >= 60:
        print(f"   ✅ EXCELLENT semantic usage ({semantic_pct:.1f}%)")
    elif semantic_pct >= 40:
        print(f"   🟡 GOOD semantic usage ({semantic_pct:.1f}%)")
    else:
        print(f"   ⚠️ LOW semantic usage ({semantic_pct:.1f}%)")
    
    if llm_pct <= 30:
        print(f"   ✅ REASONABLE LLM usage ({llm_pct:.1f}%) - not overused")
    else:
        print(f"   ⚠️ HIGH LLM usage ({llm_pct:.1f}%) - may be expensive")
    
    if keyword_pct <= 20:
        print(f"   ✅ MINIMAL keyword fallback ({keyword_pct:.1f}%)")
    else:
        print(f"   ⚠️ HIGH keyword fallback ({keyword_pct:.1f}%)")
    
    print(f"\n🔍 METHOD FLOW VERIFICATION:")
    print(f"   📊 This test captures the REAL orchestrator flow")
    print(f"   🎯 Method tracking based on ACTUAL orchestrator logs")
    print(f"   ✅ Shows true Semantic → LLM → Keyword distribution")
    
    print(f"\n🚀 COMPARISON WITH PREVIOUS TEST:")
    print(f"   ❌ Previous test claimed: 100% semantic, 0% LLM/keyword (impossible)")
    print(f"   ✅ This test shows: Real distribution based on actual orchestrator behavior")
    print(f"   💡 Previous test had broken method tracking logic")
    
    print(f"\n✅ Fixed semantic classification test completed!")

if __name__ == "__main__":
    print("🚀 FIXED SEMANTIC CLASSIFICATION TEST SUITE")
    print("=" * 80)
    print("🎯 Tests ACTUAL orchestrator behavior with CORRECT method tracking")
    print("📊 Captures real Semantic → LLM → Keyword flow")
    print("=" * 80)
    
    results = test_semantic_classification_fixed()
    
    print(f"\n🎉 Fixed testing completed!")
    print(f"💡 This shows the TRUE performance of the orchestrator")
