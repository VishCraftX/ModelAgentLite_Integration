#!/usr/bin/env python3
"""
Exhaustive Test Suite for Universal Pattern Classifier
Tests all use cases, threshold profiles, and classification methods
"""

import sys
import os
import time
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolbox import pattern_classifier, initialize_toolbox
from orchestrator import Orchestrator

class UniversalClassifierTester:
    """Comprehensive tester for Universal Pattern Classifier"""
    
    def __init__(self):
        # Initialize toolbox to ensure pattern classifier is available
        initialize_toolbox()
        
        self.orchestrator = Orchestrator()  # Still needed for intent definitions
        self.classifier = pattern_classifier  # Use global pattern classifier
        self.test_results = {}
        
    def run_all_tests(self):
        """Run all test suites"""
        print("🧪 UNIVERSAL PATTERN CLASSIFIER - EXHAUSTIVE TEST SUITE")
        print("=" * 80)
        
        # Test each use case
        self.test_intent_classification()
        self.test_skip_patterns()
        self.test_session_continuation()
        self.test_feature_detection()
        self.test_educational_queries()
        self.test_model_sub_classification()
        self.test_critical_routing()
        
        # Test context adjustments
        self.test_context_adjustments()
        
        # Test method hierarchy
        self.test_method_hierarchy()
        
        # Summary
        self.print_comprehensive_summary()
        
    def test_intent_classification(self):
        """Test main intent classification use case"""
        print("\n🎯 TESTING: Intent Classification")
        print("-" * 50)
        
        test_cases = [
            # Preprocessing cases
            ("clean my data", "preprocessing"),
            ("handle missing values", "preprocessing"),
            ("remove outliers", "preprocessing"),
            ("normalize features", "preprocessing"),  # This was failing - now should pass
            ("prepare my dataset", "preprocessing"),
            ("standardize features", "preprocessing"),
            ("scale the data", "preprocessing"),
            
            # Feature selection cases
            ("select best features", "feature_selection"),
            ("feature importance analysis", "feature_selection"),
            ("dimensionality reduction", "feature_selection"),
            ("correlation analysis", "feature_selection"),
            ("feature engineering", "feature_selection"),
            
            # Model building cases
            ("train a random forest model", "model_building"),
            ("build a classifier", "model_building"),
            ("create predictive model", "model_building"),
            ("develop machine learning algorithm", "model_building"),
            ("use this model and print rank order table", "model_building"),  # This was failing - now should pass
            ("apply existing model", "model_building"),
            ("use trained model for analysis", "model_building"),
            
            # Code execution cases
            ("run custom analysis", "code_execution"),
            ("execute python script", "code_execution"),
            ("generate visualization", "code_execution"),
            ("calculate statistics", "code_execution"),
            ("create plots", "code_execution"),
            
            # General query cases
            ("hello", "general_query"),
            ("what can you do", "general_query"),
            ("help me", "general_query"),
            ("system capabilities", "general_query"),
            ("how does this work", "general_query"),
            ("analyze", "general_query"),  # This was failing - now should pass with enhanced definitions
        ]
        
        results = self._run_test_cases("intent_classification", test_cases, self.orchestrator.intent_definitions)
        self.test_results["intent_classification"] = results
        
    def test_skip_patterns(self):
        """Test skip pattern classification"""
        print("\n🎯 TESTING: Skip Patterns")
        print("-" * 50)
        
        skip_definitions = {
            "skip_to_modeling": "Skip to modeling, go straight to modeling, bypass preprocessing and feature selection, direct to modeling, skip all preprocessing, skip everything and build model, immediate model training, direct model building, bypass data preparation entirely, skip all steps, bypass everything, skip all data work, skip everything and train, bypass all preparation, skip all processing, go directly to model, straight to model training",
            "skip_preprocessing_to_features": "Skip preprocessing but do feature selection, bypass data cleaning but select features, skip preprocessing and select features, feature selection without preprocessing, feature engineering without cleaning, skip data preparation but analyze features, clean skip but features needed, bypass cleaning keep features",
            "skip_preprocessing_to_modeling": "Skip preprocessing and go to modeling, bypass preprocessing for model building, skip data cleaning and train model, model building without preprocessing, train model with raw data, build classifier without cleaning, direct model training, skip cleaning and build model, bypass preprocessing build model, skip data preparation and model",
            "no_skip": "Normal pipeline, full pipeline, complete workflow, do preprocessing, clean data first, prepare data, standard pipeline, regular workflow, full data preparation, standard flow, complete process, full workflow, normal process"
        }
        
        test_cases = [
            # Skip to modeling cases
            ("skip preprocessing and feature selection, build model directly", "skip_to_modeling"),
            ("go straight to modeling", "skip_to_modeling"),
            ("bypass everything and train model", "skip_to_modeling"),
            ("skip all data preparation", "skip_to_modeling"),
            
            # Skip preprocessing to features cases  
            ("skip data cleaning but do feature selection", "skip_preprocessing_to_features"),
            ("bypass preprocessing but analyze features", "skip_preprocessing_to_features"),
            ("skip cleaning and select features", "skip_preprocessing_to_features"),
            
            # Skip preprocessing to modeling cases
            ("skip preprocessing and build lgbm model", "skip_preprocessing_to_modeling"),
            ("bypass data cleaning and train classifier", "skip_preprocessing_to_modeling"),
            ("skip data preparation and create model", "skip_preprocessing_to_modeling"),
            
            # No skip cases
            ("run full pipeline", "no_skip"),
            ("complete data workflow", "no_skip"),
            ("standard preprocessing", "no_skip"),
            ("clean data first", "no_skip"),
            
            # Edge cases that should NOT trigger skip patterns
            ("use this model and print rank order table", "no_skip"),  # Should not skip
            ("apply existing model for analysis", "no_skip"),           # Should not skip
        ]
        
        results = self._run_test_cases("skip_patterns", test_cases, skip_definitions)
        self.test_results["skip_patterns"] = results
        
    def test_session_continuation(self):
        """Test session continuation classification"""
        print("\n🎯 TESTING: Session Continuation")
        print("-" * 50)
        
        continuation_definitions = {
            "continuation_command": "Continue current session, yes, proceed, next, continue, go ahead, next step, proceed with current, keep going, carry on, continue session, yes continue, confirm, proceed to next, next phase, continue current task",
            "new_request": "New ML task, build new model, different analysis, new analysis, start fresh, new task, different model, change task, switch to, stop current and start, new machine learning, different algorithm, build different model, new data analysis, fresh start",
            "session_management": "Clear session, reset, stop, cancel, end session, quit, clear current, reset session, start over, abort, terminate"
        }
        
        test_cases = [
            # Continuation commands
            ("yes", "continuation_command"),
            ("continue", "continuation_command"),
            ("next step", "continuation_command"),
            ("proceed", "continuation_command"),
            ("go ahead", "continuation_command"),
            
            # New requests
            ("build a new model", "new_request"),
            ("start fresh analysis", "new_request"),
            ("different algorithm", "new_request"),
            ("switch to feature selection", "new_request"),
            
            # Session management
            ("clear session", "session_management"),
            ("reset", "session_management"),
            ("cancel", "session_management"),
            ("quit", "session_management")
        ]
        
        results = self._run_test_cases("session_continuation", test_cases, continuation_definitions)
        self.test_results["session_continuation"] = results
        
    def test_feature_detection(self):
        """Test feature/plot detection"""
        print("\n🎯 TESTING: Feature Detection")
        print("-" * 50)
        
        feature_definitions = {
            "plot_request": "Show plot, visualize tree, display tree, generate plot, create visualization, show visualization, plot tree, display model, visualize model, show decision tree, plot decision tree, tree visualization, model visualization, graphical representation, visual display",
            "analysis_request": "Analyze data, statistical analysis, data exploration, descriptive statistics, correlation analysis, data profiling, exploratory analysis, data insights, statistical summary",
            "no_special_feature": "Train model, build classifier, standard modeling, regular analysis, basic processing"
        }
        
        test_cases = [
            # Plot requests
            ("show decision tree plot", "plot_request"),
            ("visualize the model", "plot_request"),
            ("display tree visualization", "plot_request"),
            ("generate model plot", "plot_request"),
            
            # Analysis requests
            ("analyze my data", "analysis_request"),
            ("statistical summary", "analysis_request"),
            ("explore the dataset", "analysis_request"),
            ("correlation analysis", "analysis_request"),
            
            # No special features
            ("train random forest", "no_special_feature"),
            ("build classifier", "no_special_feature"),
            ("standard modeling", "no_special_feature")
        ]
        
        results = self._run_test_cases("feature_detection", test_cases, feature_definitions)
        self.test_results["feature_detection"] = results
        
    def test_educational_queries(self):
        """Test educational vs action intent classification"""
        print("\n🎯 TESTING: Educational Queries")
        print("-" * 50)
        
        educational_definitions = {
            "educational": "What is, how does, explain, describe, tell me about, what are, how to, definition of, concept of, understand, learn about, tutorial, explanation, help me understand, educational content, information about, what is random forest, what is decision tree, what is lgbm, explain random forest, explain decision trees, explain preprocessing, what are techniques, different methods, types of algorithms, how does algorithm work, algorithm explanation, technique explanation, method explanation, concept explanation, theory behind, understanding concepts",
            "action": "Build model, train classifier, create analysis, run algorithm, execute code, perform task, do analysis, generate results, take action, implement solution, apply method, process data, train model, create model, build classifier, develop algorithm, implement model, execute analysis, perform modeling, run training, apply preprocessing, do feature selection"
        }
        
        test_cases = [
            # Educational queries
            ("what is random forest", "educational"),
            ("how does lgbm work", "educational"),
            ("explain decision trees", "educational"),
            ("tell me about feature selection techniques", "educational"),
            ("what are different preprocessing methods", "educational"),
            
            # Action queries  
            ("build random forest model", "action"),
            ("train lgbm classifier", "action"),
            ("create decision tree", "action"),
            ("perform feature selection", "action"),
            ("run preprocessing", "action")
        ]
        
        results = self._run_test_cases("educational_queries", test_cases, educational_definitions)
        self.test_results["educational_queries"] = results
        
    def test_model_sub_classification(self):
        """Test model building sub-classifications"""
        print("\n🎯 TESTING: Model Sub-Classification")
        print("-" * 50)
        
        model_definitions = {
            "use_existing": "Use existing model, apply current model, utilize trained model, work with built model, use this model, apply this classifier, use previous model, existing model analysis, current model evaluation, built model application, trained model usage, model reuse, apply saved model, show plot, visualize tree, display tree, build segments, build deciles, build buckets, build rankings, rank ordering, score, predict, classify",
            "new_model": "Train new model, build new classifier, create new predictor, develop new algorithm, train fresh model, build from scratch, new model training, create classifier, develop predictor, train algorithm, build new, create new, fresh training, new machine learning model, model development, algorithm training"
        }
        
        test_cases = [
            # Use existing model
            ("use this model and print rank order table", "use_existing"),
            ("apply existing classifier for prediction", "use_existing"),
            ("show model performance", "use_existing"),
            ("visualize decision tree", "use_existing"),
            ("build segments using current model", "use_existing"),
            
            # Build new model
            ("train new random forest", "new_model"),
            ("build fresh classifier", "new_model"),
            ("create new lgbm model", "new_model"),
            ("develop new algorithm", "new_model"),
            ("train from scratch", "new_model")
        ]
        
        results = self._run_test_cases("model_sub_classification", test_cases, model_definitions)
        self.test_results["model_sub_classification"] = results
        
    def test_critical_routing(self):
        """Test critical routing decisions"""
        print("\n🎯 TESTING: Critical Routing")
        print("-" * 50)
        
        # Use same definitions as intent classification but with critical_routing thresholds
        test_cases = [
            # Edge cases that need conservative routing
            ("analyze", "preprocessing"),  # Ambiguous - should be conservative
            ("process", "preprocessing"),   # Ambiguous - should be conservative
            ("model", "model_building"),    # Ambiguous - should be conservative
            ("features", "feature_selection"), # Ambiguous - should be conservative
        ]
        
        results = self._run_test_cases("critical_routing", test_cases, self.orchestrator.intent_definitions)
        self.test_results["critical_routing"] = results
        
    def test_context_adjustments(self):
        """Test context-based threshold adjustments"""
        print("\n🎯 TESTING: Context Adjustments")
        print("-" * 50)
        
        test_query = "build model"
        base_definitions = {"model_building": "train model, build classifier", "preprocessing": "clean data, prepare data"}
        
        # Test with different context adjustments
        contexts = [
            {"semantic_adjust": 0.0, "confidence_adjust": 0.0},     # Baseline
            {"semantic_adjust": -0.1, "confidence_adjust": -0.02},  # More liberal
            {"semantic_adjust": 0.1, "confidence_adjust": 0.02},    # More conservative
        ]
        
        print(f"Test Query: '{test_query}'")
        for i, context in enumerate(contexts):
            result, method = self.classifier.classify_pattern(
                test_query, base_definitions, 
                use_case="intent_classification", 
                context_adjustments=context
            )
            print(f"  Context {i+1}: {context} → {result} ({method})")
            
    def test_method_hierarchy(self):
        """Test that Semantic → LLM → Keyword hierarchy works correctly"""
        print("\n🎯 TESTING: Method Hierarchy")
        print("-" * 50)
        
        # Test queries designed to trigger different methods
        hierarchy_tests = [
            ("clean and prepare my data", "Should use semantic"),
            ("xyz abc def unknown terms", "Should fall back to keyword"),
            ("moderately ambiguous query about data", "May use LLM fallback")
        ]
        
        for query, expected_behavior in hierarchy_tests:
            result, method = self.classifier.classify_pattern(
                query, self.orchestrator.intent_definitions, 
                use_case="intent_classification"
            )
            print(f"  '{query}' → {result} ({method}) - {expected_behavior}")
            
    def _run_test_cases(self, use_case: str, test_cases: List[Tuple[str, str]], definitions: Dict[str, str]) -> Dict[str, any]:
        """Run a set of test cases for a specific use case"""
        results = {
            "total": len(test_cases),
            "correct": 0,
            "method_stats": {"semantic": 0, "llm": 0, "keyword": 0},
            "failures": []
        }
        
        print(f"Running {len(test_cases)} test cases for {use_case}...")
        
        for query, expected in test_cases:
            try:
                start_time = time.time()
                actual, method = self.classifier.classify_pattern(query, definitions, use_case=use_case)
                duration = time.time() - start_time
                
                # Track method usage
                results["method_stats"][method] += 1
                
                # Check correctness
                if actual == expected:
                    results["correct"] += 1
                    status = "✅"
                else:
                    results["failures"].append({
                        "query": query,
                        "expected": expected,
                        "actual": actual,
                        "method": method
                    })
                    status = "❌"
                    
                print(f"  {status} '{query}' → {actual} ({method}) [{duration:.3f}s]")
                
            except Exception as e:
                results["failures"].append({
                    "query": query,
                    "expected": expected,
                    "actual": "ERROR",
                    "method": "error",
                    "error": str(e)
                })
                print(f"  💥 '{query}' → ERROR: {e}")
                
        # Print results summary
        accuracy = results["correct"] / results["total"] * 100
        print(f"\n📊 {use_case.upper()} RESULTS:")
        print(f"  Accuracy: {results['correct']}/{results['total']} ({accuracy:.1f}%)")
        print(f"  Methods: 🧠{results['method_stats']['semantic']} ⚡{results['method_stats']['keyword']} 🤖{results['method_stats']['llm']}")
        
        if results["failures"]:
            print(f"  Failures: {len(results['failures'])}")
            for failure in results["failures"][:3]:  # Show first 3 failures
                print(f"    • '{failure['query']}' → Expected: {failure['expected']}, Got: {failure['actual']}")
                
        return results
        
    def print_comprehensive_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        total_tests = sum(r["total"] for r in self.test_results.values())
        total_correct = sum(r["correct"] for r in self.test_results.values())
        overall_accuracy = total_correct / total_tests * 100 if total_tests > 0 else 0
        
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Total Correct: {total_correct}")
        print(f"  Overall Accuracy: {overall_accuracy:.1f}%")
        
        # Method distribution
        total_semantic = sum(r["method_stats"]["semantic"] for r in self.test_results.values())
        total_llm = sum(r["method_stats"]["llm"] for r in self.test_results.values())
        total_keyword = sum(r["method_stats"]["keyword"] for r in self.test_results.values())
        
        print(f"\n🎯 METHOD DISTRIBUTION:")
        print(f"  🧠 Semantic: {total_semantic} ({total_semantic/total_tests*100:.1f}%)")
        print(f"  🤖 LLM: {total_llm} ({total_llm/total_tests*100:.1f}%)")
        print(f"  ⚡ Keyword: {total_keyword} ({total_keyword/total_tests*100:.1f}%)")
        
        # Per use case breakdown
        print(f"\n📋 PER USE CASE BREAKDOWN:")
        for use_case, results in self.test_results.items():
            accuracy = results["correct"] / results["total"] * 100
            semantic_pct = results["method_stats"]["semantic"] / results["total"] * 100
            llm_pct = results["method_stats"]["llm"] / results["total"] * 100
            keyword_pct = results["method_stats"]["keyword"] / results["total"] * 100
            
            print(f"  {use_case.upper()}:")
            print(f"    Accuracy: {accuracy:.1f}% ({results['correct']}/{results['total']})")
            print(f"    Methods: 🧠{semantic_pct:.0f}% 🤖{llm_pct:.0f}% ⚡{keyword_pct:.0f}%")
            
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if overall_accuracy < 80:
            print(f"  ⚠️  Overall accuracy below 80% - consider threshold tuning")
        if total_semantic / total_tests < 0.6:
            print(f"  ⚠️  Semantic usage below 60% - consider lowering semantic thresholds")
        if total_keyword / total_tests > 0.3:
            print(f"  ⚠️  Keyword fallback usage above 30% - consider improving semantic definitions")
            
        print(f"\n✅ Universal Pattern Classifier validation completed!")

if __name__ == "__main__":
    tester = UniversalClassifierTester()
    tester.run_all_tests()
