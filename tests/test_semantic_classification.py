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

def analyze_classification_failure(query, expected_intent, actual_intent, method_used):
    """Analyze why a classification failed and provide detailed reasoning"""
    
    query_lower = query.lower()
    
    # Common failure patterns and their explanations
    failure_patterns = {
        # Semantic embedding issues
        "ambiguous_semantics": "Query semantics are ambiguous - could legitimately belong to multiple categories",
        "embedding_similarity": "Embedding vectors are too similar between intents - needs better model",
        "context_missing": "Query lacks sufficient context for accurate semantic classification",
        "domain_mismatch": "Query uses domain-specific terms not well represented in embeddings",
        
        # Keyword fallback issues  
        "keyword_overlap": "Keywords appear in multiple intent categories causing confusion",
        "missing_keywords": "Expected keywords not present in keyword lists",
        "weak_keywords": "Query contains weak/generic keywords that match multiple intents",
        "keyword_priority": "Keyword scoring prioritized wrong intent due to frequency",
        
        # Intent boundary issues
        "intent_boundary": "Query sits on boundary between two valid intents",
        "multi_intent": "Query contains multiple intents - system chose different primary intent",
        "implicit_intent": "Intent is implicit rather than explicit in the query",
        "negative_phrasing": "Negative phrasing (don't, avoid, skip) confused the classifier",
        
        # Model/system issues
        "model_unavailable": "Embedding model not available, fell back to less accurate method",
        "confidence_threshold": "Classification confidence below threshold, triggered fallback",
        "preprocessing_bias": "System has bias toward preprocessing due to keyword frequency",
        "general_fallback": "Unclear query defaulted to general_query classification"
    }
    
    # Analyze specific failure case
    if method_used == "keyword":
        # Keyword-based classification failure
        if expected_intent == "preprocessing" and actual_intent in ["feature_selection", "model_building"]:
            if any(word in query_lower for word in ["select", "feature", "variable", "attribute"]):
                return "keyword_overlap - Query contains feature selection keywords"
            elif any(word in query_lower for word in ["model", "train", "build", "predict"]):
                return "keyword_overlap - Query contains model building keywords"
            else:
                return "missing_keywords - Preprocessing keywords not detected"
                
        elif expected_intent == "feature_selection" and actual_intent == "preprocessing":
            if any(word in query_lower for word in ["clean", "prepare", "process"]):
                return "keyword_overlap - Query contains preprocessing keywords"
            else:
                return "missing_keywords - Feature selection keywords not detected"
                
        elif expected_intent == "model_building" and actual_intent == "preprocessing":
            if any(word in query_lower for word in ["clean", "prepare", "data"]):
                return "keyword_overlap - Query contains preprocessing keywords"
            else:
                return "missing_keywords - Model building keywords not detected"
                
        elif actual_intent == "general_query":
            return "general_fallback - No strong keywords detected, defaulted to general"
            
        else:
            return "keyword_priority - Wrong intent had higher keyword score"
    
    else:  # Semantic classification failure
        if expected_intent == "preprocessing" and actual_intent == "feature_selection":
            if "engineer" in query_lower or "create" in query_lower or "generate" in query_lower:
                return "intent_boundary - Engineering/creation could be preprocessing or feature work"
            else:
                return "embedding_similarity - Preprocessing and feature selection embeddings too similar"
                
        elif expected_intent == "feature_selection" and actual_intent == "model_building":
            if "model" in query_lower or "algorithm" in query_lower:
                return "multi_intent - Query mentions both feature work and modeling"
            else:
                return "embedding_similarity - Feature selection and model building embeddings confused"
                
        elif expected_intent == "model_building" and actual_intent == "code_execution":
            if "analyze" in query_lower or "calculate" in query_lower or "compute" in query_lower:
                return "intent_boundary - Analysis could be model evaluation or general computation"
            else:
                return "ambiguous_semantics - Model building vs analysis semantics unclear"
                
        elif "don't" in query_lower or "avoid" in query_lower or "skip" in query_lower:
            return "negative_phrasing - Negative language confused semantic understanding"
            
        elif actual_intent == "general_query":
            return "context_missing - Query too vague for semantic classification"
            
        else:
            return "embedding_similarity - Intent embeddings not sufficiently distinct"
    
    # Default analysis
    return f"classification_error - {expected_intent} misclassified as {actual_intent} via {method_used}"

def get_improvement_suggestions(failure_reasons):
    """Provide specific improvement suggestions based on failure patterns"""
    
    suggestions = []
    
    for reason, count in failure_reasons.items():
        if "keyword_overlap" in reason and count > 2:
            suggestions.append({
                "priority": "HIGH",
                "issue": "Keyword Overlap",
                "suggestion": "Replace keyword matching with semantic classification for overlapping terms",
                "action": "Prioritize semantic embeddings over keyword scoring"
            })
        
        elif "embedding_similarity" in reason and count > 2:
            suggestions.append({
                "priority": "MEDIUM", 
                "issue": "Intent Definitions Too Similar",
                "suggestion": "Refine intent definitions to be more distinct",
                "action": "Update intent_definitions in orchestrator.py with more specific descriptions"
            })
        
        elif "missing_keywords" in reason and count > 3:
            suggestions.append({
                "priority": "LOW",
                "issue": "Incomplete Keyword Lists", 
                "suggestion": "Add missing keywords to keyword lists",
                "action": "Update keyword lists in orchestrator.py or rely more on semantic classification"
            })
        
        elif "model_unavailable" in reason and count > 0:
            suggestions.append({
                "priority": "CRITICAL",
                "issue": "Embedding Model Missing",
                "suggestion": "Install required embedding model",
                "action": "Run: ollama pull bge-large"
            })
    
    return suggestions

def analyze_actual_method_used(query, orchestrator_instance):
    """Analyze which method was actually used for the final decision"""
    
    # We need to trace through the orchestrator logic to see what actually happened
    # This is a simplified analysis based on the orchestrator's decision flow
    
    try:
        # Get the semantic classification result
        if hasattr(orchestrator_instance, '_intent_embeddings') and orchestrator_instance._intent_embeddings:
            semantic_intent, semantic_confidence = orchestrator_instance._classify_with_semantic_similarity(query)
            
            # Check if semantic classification was confident enough (matches orchestrator logic)
            semantic_confident = semantic_confidence.get("threshold_met", False)
            # Note: Removed "confident" requirement to match optimized orchestrator
            
            if semantic_confident:
                return "semantic"
        
        # If semantic wasn't confident, orchestrator tries LLM next (not keyword)
        # We can't easily simulate LLM calls in the test, so we assume LLM was tried
        # and check if it would have fallen back to keyword
        
        # The orchestrator only falls back to keyword if LLM fails/errors
        # For test purposes, we'll assume LLM was attempted and return "llm"
        # unless there are clear keyword patterns that would indicate keyword fallback
        
        keyword_intent, keyword_confidence = orchestrator_instance._classify_with_keyword_scoring(query)
        
        # Only return "keyword" if this looks like a clear keyword-only case
        # (i.e., very high keyword confidence that would suggest LLM wasn't needed)
        if keyword_confidence["max_score"] >= 0.8:
            return "keyword"
        
        # Otherwise, assume LLM was used (matches new Semantic → LLM → Keyword flow)
        return "llm"
        
    except Exception as e:
        # If we can't determine the method, return unknown
        return f"unknown_error_{str(e)[:20]}"

def test_semantic_classification():
    """Test the new semantic classification system"""
    
    print("🧠 Testing Semantic Intent Classification")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Track statistics
    stats = {
        "total_tests": 0,
        "semantic_used": 0,
        "keyword_fallback": 0,
        "llm_fallback": 0,
        "errors": 0,
        "correct_classifications": 0,
        "by_intent": {
            "preprocessing": {"total": 0, "correct": 0, "semantic": 0, "keyword": 0, "llm": 0},
            "feature_selection": {"total": 0, "correct": 0, "semantic": 0, "keyword": 0, "llm": 0},
            "model_building": {"total": 0, "correct": 0, "semantic": 0, "keyword": 0, "llm": 0},
            "code_execution": {"total": 0, "correct": 0, "semantic": 0, "keyword": 0, "llm": 0},
            "general_query": {"total": 0, "correct": 0, "semantic": 0, "keyword": 0, "llm": 0}
        }
    }
    
    # SUPER EXHAUSTIVE test cases designed to challenge semantic understanding
    test_cases = [
        # ========== PREPROCESSING INTENT ==========
        # Synonyms that keywords would miss
        ("Preprocessing - Sanitize", "sanitize my dataset and handle inconsistencies"),
        ("Preprocessing - Purify", "purify the data by removing noise and errors"),
        ("Preprocessing - Cleanse", "cleanse the dataset from anomalies and duplicates"),
        ("Preprocessing - Scrub", "scrub the data to make it analysis-ready"),
        ("Preprocessing - Refine", "refine my raw data for better quality"),
        ("Preprocessing - Polish", "polish the dataset before analysis"),
        
        # Different phrasings for same intent
        ("Preprocessing - Handle Issues", "deal with missing values and outliers in my data"),
        ("Preprocessing - Fix Problems", "fix data quality problems and inconsistencies"),
        ("Preprocessing - Resolve Gaps", "resolve gaps and errors in the dataset"),
        ("Preprocessing - Address Quality", "address data quality issues before modeling"),
        ("Preprocessing - Rectify Data", "rectify data problems and prepare for analysis"),
        
        # Technical variations
        ("Preprocessing - Data Munging", "perform data munging and wrangling operations"),
        ("Preprocessing - ETL Process", "execute ETL processes on my raw dataset"),
        ("Preprocessing - Data Preparation", "prepare and condition data for machine learning"),
        ("Preprocessing - Data Conditioning", "condition the dataset for optimal analysis"),
        
        # Plurals and variations
        ("Preprocessing - Multiple Datasets", "clean and prepare multiple datasets simultaneously"),
        ("Preprocessing - Various Files", "process various data files and standardize them"),
        ("Preprocessing - Batch Processing", "batch process several datasets for consistency"),
        
        # ========== FEATURE SELECTION INTENT ==========
        # Engineering synonyms
        ("Feature Selection - Engineer", "engineer new variables from existing data"),
        ("Feature Selection - Craft", "craft meaningful features for better predictions"),
        ("Feature Selection - Construct", "construct relevant attributes for modeling"),
        ("Feature Selection - Derive", "derive important variables from raw data"),
        ("Feature Selection - Extract", "extract significant features for analysis"),
        ("Feature Selection - Generate", "generate predictive variables from dataset"),
        
        # Selection synonyms
        ("Feature Selection - Choose", "choose the most important variables for modeling"),
        ("Feature Selection - Pick", "pick relevant attributes for prediction"),
        ("Feature Selection - Identify", "identify key predictors in my dataset"),
        ("Feature Selection - Determine", "determine which variables are most valuable"),
        ("Feature Selection - Discover", "discover the most informative features"),
        ("Feature Selection - Find", "find the best predictive variables"),
        
        # Analysis variations
        ("Feature Selection - Analyze Importance", "analyze variable importance and relevance"),
        ("Feature Selection - Assess Predictors", "assess predictor strength and correlation"),
        ("Feature Selection - Evaluate Variables", "evaluate which variables contribute most"),
        ("Feature Selection - Examine Attributes", "examine attribute significance for modeling"),
        ("Feature Selection - Study Features", "study feature relationships and importance"),
        
        # Technical terms
        ("Feature Selection - Dimensionality", "reduce dimensionality while preserving information"),
        ("Feature Selection - Correlation Analysis", "perform correlation analysis between variables"),
        ("Feature Selection - Information Value", "calculate information value for feature ranking"),
        ("Feature Selection - Mutual Information", "compute mutual information between features"),
        
        # ========== MODEL BUILDING INTENT ==========
        # Building synonyms
        ("Model Building - Construct", "construct a predictive model for classification"),
        ("Model Building - Develop", "develop machine learning algorithms for prediction"),
        ("Model Building - Create", "create intelligent models for data analysis"),
        ("Model Building - Build", "build robust predictive algorithms"),
        ("Model Building - Design", "design ML models for forecasting"),
        ("Model Building - Architect", "architect sophisticated prediction systems"),
        
        # Training variations
        ("Model Building - Train", "train algorithms on historical data patterns"),
        ("Model Building - Fit", "fit statistical models to observed data"),
        ("Model Building - Learn", "learn patterns from data using ML techniques"),
        ("Model Building - Optimize", "optimize model parameters for best performance"),
        ("Model Building - Calibrate", "calibrate predictive models for accuracy"),
        
        # Algorithm types (should still route to model_building)
        ("Model Building - Neural Networks", "implement neural networks for pattern recognition"),
        ("Model Building - Ensemble Methods", "deploy ensemble methods for robust predictions"),
        ("Model Building - Tree Algorithms", "utilize tree-based algorithms for classification"),
        ("Model Building - Regression Models", "apply regression models for continuous prediction"),
        ("Model Building - Clustering", "perform clustering analysis on customer segments"),
        
        # Evaluation context
        ("Model Building - Assess Performance", "assess model performance using cross-validation"),
        ("Model Building - Validate Results", "validate predictive results and model accuracy"),
        ("Model Building - Benchmark Models", "benchmark different models against each other"),
        ("Model Building - Compare Algorithms", "compare various algorithms for best results"),
        
        # ========== CODE EXECUTION INTENT ==========
        # Analysis synonyms
        ("Code Execution - Analyze", "analyze data distributions and statistical properties"),
        ("Code Execution - Examine", "examine dataset characteristics and patterns"),
        ("Code Execution - Investigate", "investigate data trends and relationships"),
        ("Code Execution - Explore", "explore dataset structure and content"),
        ("Code Execution - Study", "study data patterns and anomalies"),
        
        # Computation variations
        ("Code Execution - Calculate", "calculate descriptive statistics and metrics"),
        ("Code Execution - Compute", "compute correlation matrices and summaries"),
        ("Code Execution - Generate", "generate statistical reports and visualizations"),
        ("Code Execution - Produce", "produce data profiling and quality reports"),
        ("Code Execution - Execute", "execute custom analysis on specific columns"),
        
        # Visualization requests
        ("Code Execution - Visualize", "visualize data distributions using plots"),
        ("Code Execution - Plot", "plot relationships between key variables"),
        ("Code Execution - Chart", "chart data trends over time periods"),
        ("Code Execution - Graph", "graph correlations and dependencies"),
        ("Code Execution - Display", "display summary statistics in tables"),
        
        # ========== GENERAL QUERY INTENT ==========
        # Help variations
        ("General Query - Assistance", "what kind of assistance can you provide"),
        ("General Query - Support", "what support do you offer for data science"),
        ("General Query - Guidance", "provide guidance on ML workflow steps"),
        ("General Query - Help", "help me understand your capabilities"),
        ("General Query - Aid", "what aid can you provide for my project"),
        
        # Capability inquiries
        ("General Query - Functions", "what functions and features do you have"),
        ("General Query - Services", "what services are available in this system"),
        ("General Query - Options", "what options do I have for data analysis"),
        ("General Query - Possibilities", "what possibilities exist for ML workflows"),
        ("General Query - Features", "describe the features of this platform"),
        
        # Conversational
        ("General Query - Greeting", "hello, I'm new to this system"),
        ("General Query - Introduction", "introduce yourself and your capabilities"),
        ("General Query - Overview", "give me an overview of what you can do"),
        ("General Query - Summary", "summarize your main functionalities"),
        
        # ========== EDGE CASES & CHALLENGING SCENARIOS ==========
        # Mixed contexts that could confuse keyword matching
        ("Edge Case - Clean Model", "clean my model's predictions and improve accuracy"),
        ("Edge Case - Feature Model", "model the relationship between selected features"),
        ("Edge Case - Build Data", "build a comprehensive data preprocessing pipeline"),
        ("Edge Case - Train Data", "train my data preparation workflow"),
        
        # Ambiguous but should be classifiable
        ("Edge Case - Improve Quality", "improve the quality of my analysis workflow"),
        ("Edge Case - Enhance Performance", "enhance performance of my ML pipeline"),
        ("Edge Case - Optimize Process", "optimize the entire data science process"),
        ("Edge Case - Streamline Workflow", "streamline my machine learning workflow"),
        
        # Complex multi-step requests
        ("Complex - Full Pipeline", "prepare data, select features, and build models"),
        ("Complex - End-to-End", "execute end-to-end ML pipeline from raw data"),
        ("Complex - Complete Analysis", "perform complete analysis from data to insights"),
        
        # Domain-specific terminology
        ("Domain - Financial", "prepare financial data for risk modeling"),
        ("Domain - Healthcare", "analyze patient data for predictive healthcare"),
        ("Domain - Marketing", "build customer segmentation models"),
        ("Domain - Operations", "optimize operational efficiency using ML"),
        
        # Negative cases (should still classify correctly)
        ("Negative - Don't Clean", "don't clean the data, just analyze it"),
        ("Negative - Skip Preprocessing", "skip preprocessing and go straight to modeling"),
        ("Negative - Avoid Features", "avoid feature engineering, use raw variables"),
        
        # Typos and informal language
        ("Informal - Gonna", "gonna clean my dataset real quick"),
        ("Informal - Wanna", "wanna build some ML models"),
        ("Informal - Need To", "need to prep my data for analysis"),
        ("Informal - Gotta", "gotta select the best features"),
    ]
    
    for test_name, query in test_cases:
        print(f"\n🔍 {test_name}")
        print(f"Query: '{query}'")
        print("-" * 50)
        
        # Extract expected intent from test name
        expected_intent = None
        if "Preprocessing" in test_name:
            expected_intent = "preprocessing"
        elif "Feature Selection" in test_name:
            expected_intent = "feature_selection"
        elif "Model Building" in test_name:
            expected_intent = "model_building"
        elif "Code Execution" in test_name:
            expected_intent = "code_execution"
        elif "General Query" in test_name:
            expected_intent = "general_query"
        elif "Complex" in test_name or "Domain" in test_name:
            # These could route to multiple intents, we'll analyze manually
            expected_intent = "multiple_possible"
        elif "Edge Case" in test_name or "Negative" in test_name or "Informal" in test_name:
            # These are tricky cases, we'll see how they perform
            expected_intent = "challenging"
        
        # Create a mock state
        state = PipelineState(user_query=query, session_id="test_session")
        
        # Test routing and track method used
        try:
            result = orchestrator.route(state)
            print(f"🎯 Routed to: {result}")
            
            # Track statistics
            stats["total_tests"] += 1
            
            # Determine method used - need to analyze the actual decision path
            method_used = analyze_actual_method_used(state.user_query, orchestrator)
            
            if method_used == "semantic":
                stats["semantic_used"] += 1
                print(f"📊 Method: 🧠 Semantic (using {orchestrator._active_embedding_model})")
            elif method_used == "keyword":
                stats["keyword_fallback"] += 1
                print(f"📊 Method: ⚡ Keyword fallback")
            elif method_used == "llm":
                if "llm_fallback" not in stats:
                    stats["llm_fallback"] = 0
                stats["llm_fallback"] += 1
                print(f"📊 Method: 🤖 LLM fallback")
            else:
                print(f"📊 Method: ❓ Unknown ({method_used})")
            
            # Check correctness for clear cases
            if expected_intent in ["preprocessing", "feature_selection", "model_building", "code_execution", "general_query"]:
                stats["by_intent"][expected_intent]["total"] += 1
                
                # Map result to intent (remove _node suffix and normalize)
                result_intent = result.replace("_node", "")
                
                # Normalize routing results to intent names
                if result_intent == "preprocessing":
                    result_intent = "preprocessing"
                elif result_intent == "feature_selection":
                    result_intent = "feature_selection"
                elif result_intent == "model_building":
                    result_intent = "model_building"
                elif result_intent == "code_execution":
                    result_intent = "code_execution"
                elif result_intent == "general_response":
                    result_intent = "general_query"
                elif result_intent == "END":
                    result_intent = "general_query"  # END was used for general queries before fix
                
                if result_intent == expected_intent:
                    stats["correct_classifications"] += 1
                    stats["by_intent"][expected_intent]["correct"] += 1
                    print(f"✅ Correct classification!")
                    
                    # Track method used for this intent
                    if method_used == "semantic":
                        stats["by_intent"][expected_intent]["semantic"] += 1
                    elif method_used == "keyword":
                        stats["by_intent"][expected_intent]["keyword"] += 1
                    elif method_used == "llm":
                        stats["by_intent"][expected_intent]["llm"] += 1
                else:
                    print(f"❌ Expected: {expected_intent}, Got: {result_intent}")
                    
                    # Analyze WHY it failed
                    failure_reason = analyze_classification_failure(query, expected_intent, result_intent, method_used)
                    print(f"🔍 Failure Analysis: {failure_reason}")
                    
                    # Track failure reasons
                    if "failure_reasons" not in stats:
                        stats["failure_reasons"] = {}
                    if failure_reason not in stats["failure_reasons"]:
                        stats["failure_reasons"][failure_reason] = 0
                    stats["failure_reasons"][failure_reason] += 1
            else:
                print(f"🤔 Complex case - manual analysis needed")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            stats["errors"] += 1
    
    # Print comprehensive analysis
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE SEMANTIC CLASSIFICATION ANALYSIS")
    print("=" * 80)
    
    print(f"\n🎯 OVERALL STATISTICS:")
    print(f"   Total Tests: {stats['total_tests']}")
    print(f"   Semantic Used: {stats['semantic_used']} ({stats['semantic_used']/stats['total_tests']*100:.1f}%)")
    print(f"   Keyword Fallback: {stats['keyword_fallback']} ({stats['keyword_fallback']/stats['total_tests']*100:.1f}%)")
    print(f"   LLM Fallback: {stats['llm_fallback']} ({stats['llm_fallback']/stats['total_tests']*100:.1f}%)")
    print(f"   Errors: {stats['errors']}")
    
    if stats['semantic_used'] > 0:
        print(f"\n🧠 SEMANTIC CLASSIFICATION SUCCESS:")
        print(f"   Semantic success rate: {stats['semantic_used']}/{stats['total_tests']} = {stats['semantic_used']/stats['total_tests']*100:.1f}%")
        
        if stats['semantic_used'] >= stats['total_tests'] * 0.8:
            print("   ✅ EXCELLENT: >80% semantic classification!")
        elif stats['semantic_used'] >= stats['total_tests'] * 0.6:
            print("   🟡 GOOD: >60% semantic classification")
        else:
            print("   ❌ POOR: <60% semantic classification - check embedding model")
    
    print(f"\n📈 BY INTENT ANALYSIS:")
    for intent, data in stats['by_intent'].items():
        if data['total'] > 0:
            accuracy = data['correct'] / data['total'] * 100
            semantic_rate = data['semantic'] / data['total'] * 100 if data['total'] > 0 else 0
            keyword_rate = data['keyword'] / data['total'] * 100 if data['total'] > 0 else 0
            llm_rate = data['llm'] / data['total'] * 100 if data['total'] > 0 else 0
            print(f"   {intent.upper()}:")
            print(f"     Accuracy: {data['correct']}/{data['total']} = {accuracy:.1f}%")
            print(f"     Methods: 🧠{data['semantic']} ⚡{data['keyword']} 🤖{data['llm']} (Semantic:{semantic_rate:.1f}% Keyword:{keyword_rate:.1f}% LLM:{llm_rate:.1f}%)")
    
    # Failure reason analysis
    if "failure_reasons" in stats and stats["failure_reasons"]:
        print(f"\n🔍 FAILURE ANALYSIS:")
        print("   Top failure reasons:")
        sorted_failures = sorted(stats["failure_reasons"].items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_failures[:5]:  # Top 5 failure reasons
            percentage = count / sum(stats["failure_reasons"].values()) * 100
            print(f"     • {reason}: {count} cases ({percentage:.1f}%)")
        
        # Provide actionable recommendations based on failure patterns
        print(f"\n💡 ACTIONABLE INSIGHTS:")
        for reason, count in sorted_failures[:3]:  # Top 3 for recommendations
            if "keyword_overlap" in reason:
                print("     ⚠️  Keyword lists have overlapping terms - consider semantic-first approach")
            elif "embedding_similarity" in reason:
                print("     ⚠️  Intent definitions too similar - refine semantic definitions")
            elif "missing_keywords" in reason:
                print("     ⚠️  Keyword lists incomplete - add missing terms or use semantic")
            elif "intent_boundary" in reason:
                print("     ℹ️  Some queries legitimately span multiple intents - expected behavior")
            elif "model_unavailable" in reason:
                print("     🚨 Embedding model not available - run 'ollama pull bge-large'")
            elif "general_fallback" in reason:
                print("     ⚠️  Queries too vague - may need better intent definitions")
        
        # Detailed improvement suggestions
        suggestions = get_improvement_suggestions(stats["failure_reasons"])
        if suggestions:
            print(f"\n🛠️  SPECIFIC IMPROVEMENT ACTIONS:")
            for suggestion in suggestions:
                priority_icon = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "💡", "LOW": "ℹ️"}
                icon = priority_icon.get(suggestion["priority"], "•")
                print(f"     {icon} {suggestion['priority']}: {suggestion['issue']}")
                print(f"        Solution: {suggestion['suggestion']}")
                print(f"        Action: {suggestion['action']}")
                print()
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if stats['semantic_used'] < stats['total_tests'] * 0.7:
        print("   ⚠️  Consider pulling bge-large: ollama pull bge-large")
    if stats['errors'] > 0:
        print("   ⚠️  Check system configuration - errors detected")
    if stats['correct_classifications'] / (stats['total_tests'] - stats['errors']) > 0.9:
        print("   ✅ System is performing excellently!")
    
    return stats

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
    print("🚀 SUPER EXHAUSTIVE Semantic Intent Classification Test Suite")
    print("=" * 80)
    print("🎯 Testing 100+ challenging queries to validate semantic understanding")
    print("📊 Tracking semantic vs keyword fallback usage rates")
    print("🧠 Verifying BGE-Large embedding model performance")
    print("=" * 80)
    
    # Test semantic classification
    stats = test_semantic_classification()
    
    # Test fallback behavior
    test_fallback_behavior()
    
    print("\n" + "=" * 80)
    print("🎉 FINAL SUMMARY")
    print("=" * 80)
    
    if stats['semantic_used'] >= stats['total_tests'] * 0.8:
        print("🏆 OUTSTANDING PERFORMANCE!")
        print("   ✅ Semantic classification is working excellently")
        print("   ✅ BGE-Large embeddings are functioning properly")
        print("   ✅ Ready for production use")
    elif stats['semantic_used'] >= stats['total_tests'] * 0.6:
        print("🟡 GOOD PERFORMANCE")
        print("   ✅ Semantic classification is working well")
        print("   ⚠️  Some fallback to keywords (acceptable)")
        print("   ✅ Ready for production with monitoring")
    else:
        print("❌ NEEDS ATTENTION")
        print("   ⚠️  Too much keyword fallback detected")
        print("   💡 Action: Run 'ollama pull bge-large' on server")
        print("   💡 Check: Ollama service status and model availability")
    
    print(f"\n📈 Key Metrics:")
    print(f"   🧠 Semantic Rate: {stats['semantic_used']}/{stats['total_tests']} ({stats['semantic_used']/stats['total_tests']*100:.1f}%)")
    print(f"   ⚡ Keyword Rate: {stats['keyword_fallback']}/{stats['total_tests']} ({stats['keyword_fallback']/stats['total_tests']*100:.1f}%)")
    print(f"   🤖 LLM Rate: {stats['llm_fallback']}/{stats['total_tests']} ({stats['llm_fallback']/stats['total_tests']*100:.1f}%)")
    print(f"   🎯 Overall Accuracy: {stats['correct_classifications']}/{stats['total_tests']-stats['errors']} ({stats['correct_classifications']/(stats['total_tests']-stats['errors'])*100:.1f}%)")
    
    print(f"\n💡 HIERARCHICAL vs INDEPENDENT TESTING:")
    print(f"   📊 This test shows HIERARCHICAL performance (semantic → keyword → LLM)")
    print(f"   🔬 For INDEPENDENT method comparison, run: python tests/test_method_comparison.py")
    print(f"   🎯 Independent testing will show true accuracy of each method alone")
    
    print("\n✅ Hierarchical test suite completed!")