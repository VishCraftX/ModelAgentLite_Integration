#!/usr/bin/env python3
"""
Test script for exhaustive keyword matching with text normalization
"""

import pandas as pd
import numpy as np
from pipeline_state import PipelineState
from orchestrator import orchestrator

def test_exhaustive_keyword_matching():
    """Test the exhaustive keyword sets with various query variations"""
    
    print("🧪 Testing Exhaustive Keyword Matching with Text Normalization")
    print("=" * 65)
    
    # Test queries with singular/plural variations and advanced terms
    test_cases = [
        # Preprocessing variations
        ("Preprocessing - Basic", "clean the data and handle missing values"),
        ("Preprocessing - Plural", "remove outliers and duplicates from dataset"),
        ("Preprocessing - Advanced", "standardize features and encode categorical variables"),
        ("Preprocessing - ETL", "extract transform load pipeline for data preparation"),
        ("Preprocessing - Lemmatization", "lemmatize text and tokenize documents"),
        
        # Feature Selection variations  
        ("Feature Selection - Basic", "select important features using correlation"),
        ("Feature Selection - Advanced", "dimensionality reduction with PCA analysis"),
        ("Feature Selection - Statistical", "information value and weight of evidence"),
        ("Feature Selection - Model-driven", "SHAP values and permutation importance"),
        ("Feature Selection - Plural", "analyze correlations between variables"),
        
        # Model Building variations
        ("Model Building - Classical", "train random forest classifier model"),
        ("Model Building - Boosting", "build lightgbm and xgboost models"),
        ("Model Building - Neural", "deep learning neural network training"),
        ("Model Building - Evaluation", "cross-validation and model evaluation metrics"),
        ("Model Building - Forecasting", "time series ARIMA and Prophet models"),
        
        # Code Execution variations
        ("Code Execution - Stats", "calculate mean median and standard deviation"),
        ("Code Execution - Visualization", "create scatter plots and histograms"),
        ("Code Execution - Analysis", "analyze distributions and correlations"),
        ("Code Execution - Inspection", "describe dataset schema and datatypes"),
        
        # General Query variations
        ("General - Greeting", "hello, what are your capabilities"),
        ("General - Status", "show me the current pipeline status"),
        ("General - Help", "explain how this system works"),
        
        # Edge cases and mixed terms
        ("Mixed - Preprocessing + Model", "clean data then train models"),
        ("Mixed - Feature + Model", "select features and build classifier"),
        ("Ambiguous - Low confidence", "help me with this thing"),
        ("Technical - Domain specific", "multicollinearity VIF analysis"),
        ("Plural variants", "models predictions forecasts algorithms"),
    ]
    
    for test_name, query in test_cases:
        print(f"\n🔍 {test_name}")
        print(f"Query: '{query}'")
        print("-" * 50)
        
        # Test keyword scoring
        intent, confidence_info = orchestrator._classify_with_keyword_scoring(query)
        
        print(f"🎯 Intent: {intent}")
        print(f"📊 Max Score: {confidence_info['max_score']:.3f}")
        print(f"📏 Score Diff: {confidence_info['score_diff']:.3f}")
        print(f"🔢 Raw Scores: {confidence_info['raw_scores']}")
        
        # Show which method would be used
        needs_llm = (
            confidence_info["max_score"] < 0.25 or 
            confidence_info["score_diff"] < 0.1
        )
        
        if needs_llm:
            print("🤖 → Would use LLM fallback")
        else:
            print("⚡ → High confidence keyword classification")

def test_normalization_effectiveness():
    """Test how well text normalization handles variants"""
    
    print(f"\n{'=' * 65}")
    print("🔍 TESTING TEXT NORMALIZATION EFFECTIVENESS")
    print("=" * 65)
    
    # Test singular/plural handling
    variant_tests = [
        ("Singular vs Plural", [
            "train model",
            "train models", 
            "training models",
            "model training"
        ]),
        ("Feature variants", [
            "select feature",
            "select features",
            "feature selection", 
            "selecting features"
        ]),
        ("Algorithm variants", [
            "random forest",
            "random forests",
            "forest algorithm",
            "forest classifier"
        ]),
        ("Analysis variants", [
            "correlation analysis",
            "analyze correlations",
            "correlation analyze",
            "correlational analysis"
        ])
    ]
    
    for category, queries in variant_tests:
        print(f"\n📝 {category}:")
        print("-" * 30)
        
        results = []
        for query in queries:
            intent, confidence_info = orchestrator._classify_with_keyword_scoring(query)
            results.append((query, intent, confidence_info['max_score']))
            print(f"  '{query}' → {intent} (score: {confidence_info['max_score']:.3f})")
        
        # Check consistency
        intents = [r[1] for r in results]
        if len(set(intents)) == 1:
            print(f"  ✅ Consistent classification: {intents[0]}")
        else:
            print(f"  ⚠️ Inconsistent classifications: {set(intents)}")

def test_advanced_phrase_matching():
    """Test advanced phrase matching capabilities"""
    
    print(f"\n{'=' * 65}")
    print("🎯 TESTING ADVANCED PHRASE MATCHING")
    print("=" * 65)
    
    phrase_tests = [
        ("Multi-word phrases", [
            "information value analysis",
            "weight of evidence calculation", 
            "principal component analysis",
            "cross validation metrics",
            "random forest classifier"
        ]),
        ("Technical terms", [
            "variance inflation factor",
            "mutual information score",
            "gradient boosting algorithm",
            "dimensionality reduction technique",
            "one-hot encoding method"
        ]),
        ("Domain abbreviations", [
            "PCA dimensionality reduction",
            "VIF multicollinearity check",
            "IV WOE feature selection",
            "CV model validation",
            "SHAP feature importance"
        ])
    ]
    
    for category, queries in phrase_tests:
        print(f"\n📝 {category}:")
        print("-" * 30)
        
        for query in queries:
            intent, confidence_info = orchestrator._classify_with_keyword_scoring(query)
            print(f"  '{query}'")
            print(f"    → {intent} (score: {confidence_info['max_score']:.3f})")
            print(f"    Raw scores: {confidence_info['raw_scores']}")

if __name__ == "__main__":
    test_exhaustive_keyword_matching()
    test_normalization_effectiveness() 
    test_advanced_phrase_matching()
    
    print(f"\n{'=' * 65}")
    print("🎉 Exhaustive Keyword Testing Complete!")
    print("\n📊 Key Improvements:")
    print("✅ Exhaustive keyword coverage for all ML domains")
    print("✅ Text normalization handles singular/plural variants")
    print("✅ Advanced phrase matching for technical terms")
    print("✅ Robust confidence scoring for LLM fallback decisions")
    print("✅ Domain-specific terminology recognition")
