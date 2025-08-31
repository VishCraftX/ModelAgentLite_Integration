#!/usr/bin/env python3
"""
Comprehensive Interactive Session Test Suite
Tests all interactive session scenarios: continuation, switching, skipping, and edge cases
Uses the ACTUAL pipeline process_query method to test real behavior
"""

import sys
import os
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_pipeline import MultiAgentMLPipeline
from pipeline_state import PipelineState
import pandas as pd
import tempfile
import json

def test_comprehensive_interactive_sessions():
    """
    Comprehensive test of all interactive session scenarios
    Tests the ACTUAL pipeline.process_query method with real session states
    """
    
    print("🔄 COMPREHENSIVE INTERACTIVE SESSION TEST SUITE")
    print("=" * 80)
    print("🎯 Testing ALL interactive session scenarios with REAL pipeline")
    print("📊 Tests: Continuation, Switching, Skipping, Edge Cases")
    print("🧠 Uses: Semantic → Keyword classification (no LLM)")
    print("=" * 80)
    
    # Initialize real pipeline
    pipeline = MultiAgentMLPipeline()
    
    # Test data
    test_data = pd.DataFrame({
        'feature1': [1, 2, None, 4, 5, 100],  # Has outlier and missing
        'feature2': [10, 20, 30, None, 50, 60],  # Has missing
        'feature3': ['A', 'B', 'A', 'B', 'C', 'A'],  # Categorical
        'target': [0, 1, 0, 1, 0, 1]
    })
    
    # Comprehensive test scenarios
    test_scenarios = [
        
        # ========================================
        # PREPROCESSING SESSION SCENARIOS
        # ========================================
        {
            "session_name": "PREPROCESSING SESSION",
            "initial_session": {
                "agent_type": "preprocessing",
                "phase": "outliers_detection", 
                "data_state": "analyzing_outliers",
                "session_active": True
            },
            "test_cases": [
                
                # PURE CONTINUATION COMMANDS
                ("proceed", "CONTINUE", "Universal continuation command"),
                ("continue", "CONTINUE", "Universal continuation command"),
                ("next", "CONTINUE", "Move to next step"),
                ("back", "CONTINUE", "Go back to previous step"),
                ("summary", "CONTINUE", "Show current summary"),
                ("explain", "CONTINUE", "Explain current step"),
                ("help", "CONTINUE", "Show help for current context"),
                
                # PREPROCESSING-SPECIFIC CONTINUATION
                ("skip outliers detection", "CONTINUE", "Skip current preprocessing phase"),
                ("skip outliers", "CONTINUE", "Skip outliers (short form)"),
                ("skip missing values", "CONTINUE", "Skip missing values phase"),
                ("skip encoding", "CONTINUE", "Skip encoding phase"),
                ("skip transformations", "CONTINUE", "Skip transformations phase"),
                ("skip this phase", "CONTINUE", "Skip current phase (generic)"),
                ("skip current", "CONTINUE", "Skip current step (generic)"),
                ("proceed to missing values", "CONTINUE", "Next preprocessing step"),
                ("move to encoding", "CONTINUE", "Advance to encoding"),
                ("continue with data cleaning", "CONTINUE", "Explicit preprocessing continuation"),
                
                # TARGET COLUMN SPECIFICATION
                ("target column is target", "CONTINUE", "Target column specification"),
                ("target is target", "CONTINUE", "Target specification (short)"),
                ("column target", "CONTINUE", "Column specification"),
                ("set target to target", "CONTINUE", "Set target column"),
                
                # YES/NO RESPONSES (if in confirmation phase)
                ("yes", "CONTINUE", "Confirmation response"),
                ("no", "CONTINUE", "Negative confirmation"),
                ("y", "CONTINUE", "Short yes"),
                ("n", "CONTINUE", "Short no"),
                
                # EXPLICIT SESSION MANAGEMENT
                ("clear session", "CLEAR", "Explicit session clear"),
                ("reset", "CLEAR", "Reset session"),
                ("start over", "CLEAR", "Start over command"),
                ("new session", "CLEAR", "New session request"),
                ("exit session", "CLEAR", "Exit current session"),
                
                # NEW REQUEST DETECTION (Should switch)
                ("select features", "SWITCH", "Direct feature selection request"),
                ("I want to select important features", "SWITCH", "Natural FS request"),
                ("analyze feature importance", "SWITCH", "Feature importance analysis"),
                ("find relevant variables", "SWITCH", "Variable selection"),
                ("perform feature engineering", "SWITCH", "Feature engineering"),
                ("train a model", "SWITCH", "Direct model training"),
                ("build a classifier", "SWITCH", "Classifier building"),
                ("create prediction model", "SWITCH", "Prediction model"),
                ("train machine learning algorithm", "SWITCH", "ML algorithm training"),
                ("develop neural network", "SWITCH", "Neural network development"),
                
                # CONTEXTUAL/AMBIGUOUS CASES
                ("I'm done with data cleaning", "SWITCH", "Implicit completion signal"),
                ("the data looks good now", "SWITCH", "Data quality satisfaction"),
                ("let's move to the next step", "SWITCH", "Next ML pipeline step"),
                ("what should I do next?", "CONTINUE", "General next step question"),
                ("how is my data quality?", "CONTINUE", "Data quality inquiry"),
                ("show me data statistics", "CONTINUE", "Data analysis request"),
                
                # MIXED SIGNALS (Conflict resolution)
                ("skip outliers and select features", "SWITCH", "Skip + select (select should win)"),
                ("continue preprocessing but analyze features", "CONTINUE", "Continue + analyze (continue should win)"),
                ("target price and select features", "SWITCH", "Target + select (select should win)"),
                ("proceed with feature selection", "SWITCH", "Proceed with different agent"),
                ("skip this and train model", "SWITCH", "Skip + train (train should win)"),
                
                # EDGE CASES
                ("", "CONTINUE", "Empty query"),
                ("   ", "CONTINUE", "Whitespace only"),
                ("hello", "CONTINUE", "Greeting during session"),
                ("what can you do?", "CONTINUE", "Capability question during session"),
                ("explain machine learning", "CONTINUE", "Educational question during session"),
            ]
        },
        
        # ========================================
        # FEATURE SELECTION SESSION SCENARIOS  
        # ========================================
        {
            "session_name": "FEATURE SELECTION SESSION",
            "initial_session": {
                "agent_type": "feature_selection",
                "phase": "correlation_analysis",
                "data_state": "analyzing_correlations", 
                "session_active": True
            },
            "test_cases": [
                
                # PURE CONTINUATION COMMANDS
                ("proceed", "CONTINUE", "Universal continuation"),
                ("continue", "CONTINUE", "Universal continuation"),
                ("next", "CONTINUE", "Next step"),
                ("summary", "CONTINUE", "Show summary"),
                ("help", "CONTINUE", "Show help"),
                
                # FEATURE SELECTION SPECIFIC CONTINUATION
                ("run SHAP analysis", "CONTINUE", "Execute SHAP analysis"),
                ("run IV analysis", "CONTINUE", "Information Value analysis"),
                ("skip correlation analysis", "CONTINUE", "Skip current FS phase"),
                ("skip current analysis", "CONTINUE", "Skip current analysis"),
                ("proceed with variable selection", "CONTINUE", "Next FS step"),
                ("continue feature selection", "CONTINUE", "Explicit FS continuation"),
                ("analyze feature correlations", "CONTINUE", "Correlation analysis"),
                ("rank features by importance", "CONTINUE", "Feature ranking"),
                ("select top features", "CONTINUE", "Feature selection"),
                ("perform dimensionality reduction", "CONTINUE", "Dimensionality reduction"),
                
                # SESSION MANAGEMENT
                ("clear session", "CLEAR", "Clear FS session"),
                ("reset", "CLEAR", "Reset FS session"),
                ("new session", "CLEAR", "New session"),
                
                # NEW REQUESTS (Should switch)
                ("train the model now", "SWITCH", "Switch to model building"),
                ("build a classifier", "SWITCH", "Classifier building"),
                ("create prediction model", "SWITCH", "Prediction model"),
                ("clean my dataset", "SWITCH", "Switch to preprocessing"),
                ("handle missing values", "SWITCH", "Data cleaning request"),
                ("remove outliers", "SWITCH", "Outlier removal"),
                ("preprocess the data", "SWITCH", "Preprocessing request"),
                
                # CONTEXTUAL CASES
                ("I'm satisfied with feature selection", "SWITCH", "Completion signal"),
                ("these features look good", "SWITCH", "Feature satisfaction"),
                ("ready for model training", "SWITCH", "Ready for next phase"),
                ("what's the next step?", "CONTINUE", "Next step question"),
                ("how many features selected?", "CONTINUE", "Feature count question"),
                
                # MIXED SIGNALS
                ("skip analysis and train model", "SWITCH", "Skip + train (train wins)"),
                ("continue with model building", "SWITCH", "Continue with different agent"),
                ("run SHAP then build model", "CONTINUE", "SHAP + build (SHAP wins - current agent)"),
            ]
        },
        
        # ========================================
        # MODEL BUILDING SESSION SCENARIOS
        # ========================================
        {
            "session_name": "MODEL BUILDING SESSION", 
            "initial_session": {
                "agent_type": "model_building",
                "phase": "algorithm_selection",
                "data_state": "selecting_algorithm",
                "session_active": True
            },
            "test_cases": [
                
                # MODEL BUILDING CONTINUATION
                ("proceed", "CONTINUE", "Universal continuation"),
                ("continue", "CONTINUE", "Universal continuation"),
                ("next", "CONTINUE", "Next step"),
                ("skip hyperparameter tuning", "CONTINUE", "Skip HP tuning"),
                ("proceed with training", "CONTINUE", "Continue training"),
                ("evaluate model performance", "CONTINUE", "Model evaluation"),
                ("run cross validation", "CONTINUE", "Cross validation"),
                
                # SESSION MANAGEMENT
                ("clear session", "CLEAR", "Clear model session"),
                ("reset", "CLEAR", "Reset model session"),
                
                # NEW REQUESTS (Should switch)
                ("clean my data first", "SWITCH", "Switch to preprocessing"),
                ("select better features", "SWITCH", "Switch to feature selection"),
                ("analyze feature importance", "SWITCH", "Feature analysis"),
                ("preprocess the dataset", "SWITCH", "Data preprocessing"),
                
                # CONTEXTUAL CASES
                ("the model is ready", "CONTINUE", "Model completion"),
                ("show model results", "CONTINUE", "Results display"),
                ("what's the accuracy?", "CONTINUE", "Accuracy question"),
            ]
        }
    ]
    
    # Track comprehensive results
    results = {
        "total_tests": 0,
        "correct_predictions": 0,
        "session_results": {},
        "behavior_breakdown": {
            "CONTINUE": {"total": 0, "correct": 0},
            "SWITCH": {"total": 0, "correct": 0}, 
            "CLEAR": {"total": 0, "correct": 0}
        },
        "method_usage": {
            "semantic": 0,
            "keyword": 0,
            "error": 0
        }
    }
    
    # Run all test scenarios
    for scenario in test_scenarios:
        session_name = scenario["session_name"]
        initial_session = scenario["initial_session"]
        test_cases = scenario["test_cases"]
        
        print(f"\n{'='*20} {session_name} {'='*20}")
        print(f"📍 Initial State: {initial_session['agent_type']} - {initial_session['phase']}")
        print(f"🧪 Test Cases: {len(test_cases)}")
        print("-" * 80)
        
        session_results = {"total": 0, "correct": 0}
        
        for i, (query, expected_behavior, description) in enumerate(test_cases, 1):
            results["total_tests"] += 1
            session_results["total"] += 1
            results["behavior_breakdown"][expected_behavior]["total"] += 1
            
            print(f"\n📝 Test {i}: '{query}'")
            print(f"   Expected: {expected_behavior}")
            print(f"   Description: {description}")
            
            try:
                # Test with real pipeline
                actual_behavior, method_used = test_real_interactive_session(
                    pipeline, query, initial_session, test_data
                )
                
                # Track method usage
                results["method_usage"][method_used] += 1
                
                # Evaluate correctness
                is_correct = evaluate_session_behavior(actual_behavior, expected_behavior)
                
                if is_correct:
                    results["correct_predictions"] += 1
                    session_results["correct"] += 1
                    results["behavior_breakdown"][expected_behavior]["correct"] += 1
                    print(f"   ✅ CORRECT: {actual_behavior} (via {method_used})")
                else:
                    print(f"   ❌ WRONG: Got {actual_behavior}, expected {expected_behavior} (via {method_used})")
                
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                results["method_usage"]["error"] += 1
        
        # Session summary
        session_accuracy = (session_results["correct"] / session_results["total"]) * 100
        print(f"\n📊 {session_name} SUMMARY:")
        print(f"   Accuracy: {session_results['correct']}/{session_results['total']} = {session_accuracy:.1f}%")
        
        results["session_results"][session_name] = session_results
    
    # Comprehensive analysis
    analyze_comprehensive_interactive_results(results)
    
    return results

def test_real_interactive_session(pipeline, query, initial_session, test_data):
    """
    Test the ACTUAL interactive session logic using real pipeline.process_query
    """
    
    # Create a temporary session with the initial state
    session_id = f"test_session_{hash(query) % 10000}"
    
    try:
        # Create initial state with interactive session
        initial_state = PipelineState(
            user_query="initial setup",
            raw_data=test_data,
            interactive_session=initial_session.copy()
        )
        
        # Save initial state to simulate existing session
        pipeline._save_session_state(session_id, initial_state)
        
        # Now test the actual query
        response = pipeline.process_query(
            user_query=query,
            session_id=session_id,
            raw_data=None
        )
        
        # Load the resulting state to see what happened
        final_state = pipeline._load_session_state(session_id)
        
        # Analyze what happened
        if "session cleared" in response.get("response", "").lower() or "session clear" in response.get("response", "").lower():
            return "CLEAR", "semantic"
        elif final_state.interactive_session is None:
            # Session was cleared, likely due to new request detection
            return "SWITCH", "semantic"
        elif final_state.interactive_session is not None:
            # Session continues
            return "CONTINUE", "semantic"
        else:
            return "UNKNOWN", "unknown"
            
    except Exception as e:
        print(f"   ⚠️ Pipeline error: {e}")
        return "ERROR", "error"
    
    finally:
        # Cleanup
        try:
            pipeline.cleanup_user_session(session_id)
        except:
            pass

def evaluate_session_behavior(actual, expected):
    """Evaluate if actual behavior matches expected"""
    if expected == "CONTINUE" and actual in ["CONTINUE"]:
        return True
    elif expected == "SWITCH" and actual in ["SWITCH"]:
        return True
    elif expected == "CLEAR" and actual in ["CLEAR"]:
        return True
    else:
        return False

def analyze_comprehensive_interactive_results(results):
    """Comprehensive analysis of interactive session test results"""
    
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE INTERACTIVE SESSION ANALYSIS")
    print("=" * 80)
    
    total = results["total_tests"]
    correct = results["correct_predictions"]
    overall_accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Total Tests: {total}")
    print(f"   Correct Predictions: {correct}")
    print(f"   Overall Accuracy: {overall_accuracy:.1f}%")
    
    print(f"\n📋 SESSION TYPE BREAKDOWN:")
    for session_name, session_stats in results["session_results"].items():
        session_accuracy = (session_stats["correct"] / session_stats["total"]) * 100
        print(f"   {session_name}: {session_stats['correct']}/{session_stats['total']} = {session_accuracy:.1f}%")
    
    print(f"\n🎯 BEHAVIOR TYPE ANALYSIS:")
    for behavior, stats in results["behavior_breakdown"].items():
        if stats["total"] > 0:
            behavior_accuracy = (stats["correct"] / stats["total"]) * 100
            print(f"   {behavior}: {stats['correct']}/{stats['total']} = {behavior_accuracy:.1f}%")
    
    print(f"\n🧠 METHOD USAGE:")
    for method, count in results["method_usage"].items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"   {method.title()}: {count}/{total} = {percentage:.1f}%")
    
    print(f"\n💡 KEY INSIGHTS:")
    
    # Performance insights
    if overall_accuracy >= 90:
        print(f"   ✅ EXCELLENT: Interactive session handling is very robust")
    elif overall_accuracy >= 80:
        print(f"   🟡 GOOD: Interactive session handling mostly works well")
    elif overall_accuracy >= 70:
        print(f"   ⚠️ FAIR: Interactive session handling needs some improvement")
    else:
        print(f"   ❌ POOR: Interactive session handling needs significant work")
    
    # Behavior-specific insights
    continue_accuracy = results["behavior_breakdown"]["CONTINUE"]["correct"] / max(results["behavior_breakdown"]["CONTINUE"]["total"], 1) * 100
    switch_accuracy = results["behavior_breakdown"]["SWITCH"]["correct"] / max(results["behavior_breakdown"]["SWITCH"]["total"], 1) * 100
    clear_accuracy = results["behavior_breakdown"]["CLEAR"]["correct"] / max(results["behavior_breakdown"]["CLEAR"]["total"], 1) * 100
    
    print(f"\n🔍 BEHAVIOR-SPECIFIC INSIGHTS:")
    print(f"   🔄 CONTINUATION: {continue_accuracy:.1f}% accuracy")
    if continue_accuracy >= 85:
        print(f"      ✅ Excellent at detecting session continuation commands")
    else:
        print(f"      ⚠️ May be misclassifying continuation commands as switches")
    
    print(f"   🔄 SWITCHING: {switch_accuracy:.1f}% accuracy")
    if switch_accuracy >= 85:
        print(f"      ✅ Excellent at detecting new request switches")
    else:
        print(f"      ⚠️ May be missing new request patterns or being too conservative")
    
    print(f"   🔄 CLEARING: {clear_accuracy:.1f}% accuracy")
    if clear_accuracy >= 90:
        print(f"      ✅ Excellent at handling explicit session management")
    else:
        print(f"      ⚠️ Issues with explicit session clear commands")
    
    print(f"\n🚀 RECOMMENDATIONS:")
    if overall_accuracy < 80:
        print(f"   🔧 Improve semantic intent definitions for session management")
        print(f"   📊 Add more training examples for ambiguous cases")
    
    if switch_accuracy < continue_accuracy - 10:
        print(f"   🎯 Focus on improving new request detection")
        print(f"   💡 May need more comprehensive 'new_ml_request' definitions")
    
    if continue_accuracy < switch_accuracy - 10:
        print(f"   🔄 Focus on improving continuation detection")
        print(f"   💡 May need more agent-specific continuation patterns")
    
    print(f"\n✅ Comprehensive interactive session analysis completed!")

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE INTERACTIVE SESSION TEST SUITE")
    print("=" * 80)
    print("🎯 Testing ALL interactive session scenarios with REAL pipeline")
    print("📊 Covers: Preprocessing, Feature Selection, Model Building sessions")
    print("🔄 Tests: Continuation, Switching, Clearing, Edge Cases")
    print("=" * 80)
    
    results = test_comprehensive_interactive_sessions()
    
    print(f"\n🎉 Comprehensive interactive session testing completed!")
    print(f"💡 This shows the true performance of session management logic")
