#!/usr/bin/env python3
"""
Example Usage of Multi-Agent ML Pipeline
Demonstrates various ways to use the integrated ML system
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

from langgraph_pipeline import initialize_pipeline
from config import print_config, validate_config


def create_sample_data():
    """Create sample dataset for testing"""
    np.random.seed(42)
    
    # Create a loan default prediction dataset
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_amount': np.random.normal(25000, 10000, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'debt_to_income': np.random.uniform(0.1, 0.8, n_samples),
        'previous_defaults': np.random.poisson(0.3, n_samples),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% default rate
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values to make it realistic
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_indices, 'employment_years'] = np.nan
    
    return df


def example_basic_usage():
    """Example 1: Basic pipeline usage"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Pipeline Usage")
    print("="*60)
    
    # Initialize pipeline
    pipeline = initialize_pipeline(enable_persistence=True)
    
    # Create sample data
    data = create_sample_data()
    print(f"📊 Created sample dataset: {data.shape}")
    
    # Load data
    session_id = f"basic_example_{int(datetime.now().timestamp())}"
    pipeline.load_data(data, session_id)
    
    # Test basic queries
    queries = [
        "Show me a summary of this dataset",
        "Clean and preprocess the data",
        "Select the most important features",
        "Train a machine learning model"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Step {i}: {query} ---")
        result = pipeline.process_query(query, session_id)
        print(f"Response: {result['response']}")
        
        if not result['success']:
            print(f"❌ Error: {result.get('error')}")
            break
    
    return session_id


def example_targeted_operations():
    """Example 2: Targeted operations on specific agents"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Targeted Operations")
    print("="*60)
    
    pipeline = initialize_pipeline(enable_persistence=True)
    data = create_sample_data()
    
    session_id = f"targeted_example_{int(datetime.now().timestamp())}"
    pipeline.load_data(data, session_id)
    
    # Preprocessing-specific queries
    preprocessing_queries = [
        "Handle missing values in the dataset",
        "Remove outliers from numeric columns",
        "Encode categorical variables"
    ]
    
    print("\n🧹 PREPROCESSING OPERATIONS:")
    for query in preprocessing_queries:
        print(f"\nQuery: {query}")
        result = pipeline.process_query(query, session_id)
        print(f"Response: {result['response']}")
    
    # Feature selection queries
    feature_queries = [
        "Calculate information value for all features",
        "Remove highly correlated features",
        "Select top 5 features for modeling"
    ]
    
    print("\n🎯 FEATURE SELECTION OPERATIONS:")
    for query in feature_queries:
        print(f"\nQuery: {query}")
        result = pipeline.process_query(query, session_id)
        print(f"Response: {result['response']}")
    
    # Model building queries
    model_queries = [
        "Train a LightGBM classifier",
        "Evaluate model performance",
        "Create feature importance plot"
    ]
    
    print("\n🤖 MODEL BUILDING OPERATIONS:")
    for query in model_queries:
        print(f"\nQuery: {query}")
        result = pipeline.process_query(query, session_id)
        print(f"Response: {result['response']}")
    
    return session_id


def example_session_management():
    """Example 3: Session management and persistence"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Session Management")
    print("="*60)
    
    pipeline = initialize_pipeline(enable_persistence=True)
    
    # Create first session
    session_1 = f"session_1_{int(datetime.now().timestamp())}"
    data_1 = create_sample_data()
    pipeline.load_data(data_1, session_1)
    
    result = pipeline.process_query("Clean and preprocess this data", session_1)
    print(f"Session 1 - Preprocessing: {result['response']}")
    
    # Create second session
    session_2 = f"session_2_{int(datetime.now().timestamp())}"
    data_2 = create_sample_data()
    data_2['new_feature'] = np.random.normal(0, 1, len(data_2))  # Add a new feature
    pipeline.load_data(data_2, session_2)
    
    result = pipeline.process_query("Select features and train a model", session_2)
    print(f"Session 2 - Full pipeline: {result['response']}")
    
    # List all sessions
    sessions = pipeline.list_sessions()
    print(f"\n📋 Available sessions: {len(sessions)}")
    for session in sessions[-5:]:  # Show last 5
        status = pipeline.get_session_status(session)
        if status['exists']:
            print(f"  • {session}: {status['data_summary']}")
    
    # Resume first session
    result = pipeline.process_query("Now select features for this cleaned data", session_1)
    print(f"\nResumed Session 1 - Feature selection: {result['response']}")
    
    return session_1, session_2


def example_error_handling():
    """Example 4: Error handling and recovery"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Error Handling")
    print("="*60)
    
    pipeline = initialize_pipeline(enable_persistence=True)
    
    # Test with problematic queries
    session_id = f"error_example_{int(datetime.now().timestamp())}"
    
    # Query without data
    result = pipeline.process_query("Train a model", session_id)
    print(f"No data query: {result['response']}")
    print(f"Success: {result['success']}")
    
    # Load data and try complex query
    data = create_sample_data()
    pipeline.load_data(data, session_id)
    
    # Query that might cause issues
    result = pipeline.process_query("Build a neural network with 100 layers", session_id)
    print(f"\nComplex query: {result['response']}")
    print(f"Success: {result['success']}")
    
    # Recovery query
    result = pipeline.process_query("Train a simple random forest model", session_id)
    print(f"\nRecovery query: {result['response']}")
    print(f"Success: {result['success']}")
    
    return session_id


def example_progress_tracking():
    """Example 5: Progress tracking with callbacks"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Progress Tracking")
    print("="*60)
    
    pipeline = initialize_pipeline(enable_persistence=True)
    data = create_sample_data()
    
    session_id = f"progress_example_{int(datetime.now().timestamp())}"
    pipeline.load_data(data, session_id)
    
    # Progress tracking callback
    progress_log = []
    
    def progress_callback(message: str, stage: str = ""):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {stage}: {message}" if stage else f"[{timestamp}] {message}"
        progress_log.append(log_entry)
        print(f"📡 {log_entry}")
    
    # Run query with progress tracking
    result = pipeline.process_query(
        "Build a complete ML pipeline with preprocessing, feature selection, and model training",
        session_id,
        progress_callback=progress_callback
    )
    
    print(f"\n✅ Final result: {result['response']}")
    print(f"\n📊 Progress log ({len(progress_log)} entries):")
    for entry in progress_log:
        print(f"  {entry}")
    
    return session_id


def main():
    """Run all examples"""
    print("🚀 Multi-Agent ML Pipeline - Example Usage")
    print("="*60)
    
    # Print configuration
    print_config()
    
    # Validate configuration
    errors = validate_config()
    if errors:
        print(f"\n⚠️ Configuration issues found:")
        for error in errors:
            print(f"  • {error}")
        print("\nSome examples may not work properly without proper configuration.")
    
    try:
        # Run examples
        session_1 = example_basic_usage()
        session_2 = example_targeted_operations()
        session_3a, session_3b = example_session_management()
        session_4 = example_error_handling()
        session_5 = example_progress_tracking()
        
        print("\n" + "="*60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Summary
        pipeline = initialize_pipeline()
        all_sessions = pipeline.list_sessions()
        print(f"\n📊 Summary:")
        print(f"  • Total sessions created: {len([s for s in all_sessions if 'example' in s])}")
        print(f"  • All sessions: {len(all_sessions)}")
        
        # Cleanup old sessions (optional)
        print(f"\n🧹 Cleaning up old sessions...")
        pipeline.cleanup_old_sessions(max_age_hours=0)  # Clean up immediately for demo
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
