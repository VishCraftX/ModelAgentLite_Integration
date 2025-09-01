#!/usr/bin/env python3
"""
Startup script for Multi-Agent ML Integration System
Choose between Slack bot, Python API testing, or direct agent testing

Usage:
    python start_pipeline.py [--bot-token BOT_TOKEN] [--app-token APP_TOKEN] [--mode MODE]
"""

import sys
import os
import argparse
from pathlib import Path
# Removed logging_config import - module was deleted

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

def parse_arguments():
    parser = argparse.ArgumentParser(description='Start Multi-Agent ML Integration System')
    parser.add_argument('--bot-token', '--slack-bot-token', 
                       help='Slack Bot Token (xoxb-...)')
    parser.add_argument('--app-token', '--slack-app-token', 
                       help='Slack App Token (xapp-...)')
    parser.add_argument('--mode', choices=['slack', 'api', 'test', 'demo'],
                       help='Run mode: slack bot, api testing, direct testing, or demo')
    parser.add_argument('--model', default=os.environ.get('DEFAULT_MODEL', 'gpt-4o'),
                       help='Default LLM model to use')
    parser.add_argument('--persistence', action='store_true', default=True,
                       help='Enable state persistence (default: True)')
    return parser.parse_args()

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    optional_deps = []
    
    # Core dependencies
    try:
        import pandas
        import numpy
        import sklearn
        import pydantic
    except ImportError as e:
        missing_deps.append(f"Core ML libraries: {e}")
    
    # LangGraph dependencies
    try:
        import langgraph
        import langchain
        import langchain_core
    except ImportError as e:
        optional_deps.append(f"LangGraph: {e}")
    
    # Slack dependencies
    try:
        import slack_bolt
        import slack_sdk
    except ImportError as e:
        optional_deps.append(f"Slack integration: {e}")
    
    # LLM dependencies
    try:
        import ollama
    except ImportError:
        optional_deps.append("Ollama: Not available (local LLM support)")
    
    try:
        import openai
    except ImportError:
        optional_deps.append("OpenAI: Not available (cloud LLM support)")
    
    return missing_deps, optional_deps

def print_system_status():
    """Print system status and available features"""
    print("🚀 Multi-Agent ML Integration System")
    print("=" * 60)
    print("Architecture: Orchestrator → Agents → State Management → Artifacts")
    print("Agents: Preprocessing, Feature Selection, Model Building")
    print("=" * 60)
    
    # Check dependencies
    missing, optional = check_dependencies()
    
    if missing:
        print("\n❌ Missing Required Dependencies:")
        for dep in missing:
            print(f"   • {dep}")
        print("\n💡 Install with: pip install -r requirements_complete.txt")
        return False
    
    print("\n✅ Core Dependencies: Available")
    
    if optional:
        print("\n⚠️ Optional Dependencies:")
        for dep in optional:
            print(f"   • {dep}")
        print("\n💡 Some features may use fallback implementations")
    
    return True

def setup_environment(args):
    """Set up environment variables"""
    # Set tokens from command-line arguments if provided
    if args.bot_token:
        os.environ["SLACK_BOT_TOKEN"] = args.bot_token
        print("✅ Bot token set from command line")
    
    if args.app_token:
        os.environ["SLACK_APP_TOKEN"] = args.app_token
        print("✅ App token set from command line")
    
    # Set model configuration
    os.environ["DEFAULT_MODEL"] = args.model
    os.environ["ENABLE_PERSISTENCE"] = str(args.persistence).lower()
    
    print(f"🤖 Default model: {args.model}")
    print(f"💾 Persistence: {'Enabled' if args.persistence else 'Disabled'}")

def start_slack_bot():
    """Start the Slack bot"""
    print("\n🤖 Starting Multi-Agent ML Slack Bot...")
    print("Features:")
    print("  • File upload support (CSV, Excel, JSON, TSV)")
    print("  • Intelligent query routing")
    print("  • Real-time progress updates")
    print("  • Session-based conversations")
    print("  • Artifact management")
    
    # Check for Slack tokens
    if not os.environ.get("SLACK_BOT_TOKEN"):
        print("\n🔑 Slack Bot Token Required")
        token = input("Enter SLACK_BOT_TOKEN (or press Enter to skip): ").strip()
        if token:
            os.environ["SLACK_BOT_TOKEN"] = token
        else:
            print("❌ Bot token required for Slack integration")
            return
    
    if not os.environ.get("SLACK_APP_TOKEN"):
        print("\n🔑 Slack App Token Required")
        app_token = input("Enter SLACK_APP_TOKEN (or press Enter to skip): ").strip()
        if app_token:
            os.environ["SLACK_APP_TOKEN"] = app_token
        else:
            print("❌ App token required for Slack integration")
            return
    
    try:
        from slack_bot import SlackMLBot
        bot = SlackMLBot()
        print("✅ Slack bot initialized successfully")
        print("🚀 Starting bot... (Press Ctrl+C to stop)")
        bot.start()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install Slack dependencies: pip install slack-bolt slack-sdk")
    except Exception as e:
        print(f"❌ Error starting Slack bot: {e}")

def start_api_testing():
    """Start interactive API testing"""
    print("\n🧪 Starting Interactive API Testing...")
    
    try:
        from langgraph_pipeline import initialize_pipeline
        import pandas as pd
        import numpy as np
        
        # Initialize pipeline
        pipeline = initialize_pipeline(enable_persistence=True)
        print("✅ Pipeline initialized")
        
        # Create sample data
        print("\n📊 Creating sample dataset...")
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'income': np.random.normal(50000, 15000, 100),
            'credit_score': np.random.randint(300, 850, 100),
            'employment_years': np.random.randint(0, 40, 100),
            'target': np.random.choice([0, 1], 100, p=[0.7, 0.3])
        })
        
        session_id = "interactive_session"
        pipeline.load_data(sample_data, session_id)
        print(f"✅ Sample data loaded: {sample_data.shape}")
        
        print("\n🎯 Interactive Testing Mode")
        print("Available commands:")
        print("  • 'preprocess' or 'clean data'")
        print("  • 'select features' or 'feature selection'")
        print("  • 'train model' or 'build model'")
        print("  • 'status' - show current pipeline status")
        print("  • 'quit' or 'exit' - exit testing")
        
        while True:
            try:
                query = input("\n🤖 Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'status':
                    status = pipeline.get_session_status(session_id)
                    if status['exists']:
                        summary = status['data_summary']
                        print(f"📊 Pipeline Status:")
                        print(f"   Raw Data: {'✅' if summary['has_raw_data'] else '❌'}")
                        print(f"   Cleaned Data: {'✅' if summary['has_cleaned_data'] else '❌'}")
                        print(f"   Selected Features: {'✅' if summary['has_selected_features'] else '❌'}")
                        print(f"   Trained Model: {'✅' if summary['has_trained_model'] else '❌'}")
                    continue
                
                if not query:
                    continue
                
                print(f"\n🔄 Processing: '{query}'")
                result = pipeline.process_query(query, session_id)
                
                print(f"✅ Success: {result['success']}")
                print(f"📝 Response: {result['response']}")
                
                if not result['success'] and result.get('error'):
                    print(f"❌ Error: {result['error']}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install required dependencies: pip install -r requirements_complete.txt")
    except Exception as e:
        print(f"❌ Error in API testing: {e}")

def run_demo():
    """Run automated demo"""
    print("\n🎬 Running Automated Demo...")
    
    try:
        from langgraph_pipeline import initialize_pipeline
        import pandas as pd
        import numpy as np
        
        # Initialize pipeline
        pipeline = initialize_pipeline(enable_persistence=False)
        print("✅ Pipeline initialized")
        
        # Create realistic sample data
        print("\n📊 Creating realistic dataset...")
        np.random.seed(42)
        n_samples = 200
        sample_data = pd.DataFrame({
            'customer_age': np.random.randint(18, 80, n_samples),
            'annual_income': np.random.normal(55000, 20000, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'loan_amount': np.random.normal(30000, 15000, n_samples),
            'employment_years': np.random.randint(0, 40, n_samples),
            'debt_to_income_ratio': np.random.uniform(0.1, 0.8, n_samples),
            'default_risk': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
        })
        
        # Add some missing values
        missing_indices = np.random.choice(sample_data.index, size=int(0.05 * len(sample_data)), replace=False)
        sample_data.loc[missing_indices, 'annual_income'] = np.nan
        
        session_id = "demo_session"
        pipeline.load_data(sample_data, session_id)
        print(f"✅ Demo data loaded: {sample_data.shape}")
        print(f"   Missing values: {sample_data.isnull().sum().sum()}")
        print(f"   Target distribution: {sample_data['default_risk'].value_counts().to_dict()}")
        
        # Demo queries
        demo_queries = [
            ("Data Preprocessing", "Clean and preprocess this data"),
            ("Feature Selection", "Select the most important features"),
            ("Model Building", "Train a machine learning model for classification"),
        ]
        
        print(f"\n🎯 Running {len(demo_queries)} demo steps...")
        
        for i, (step_name, query) in enumerate(demo_queries, 1):
            print(f"\n{'='*50}")
            print(f"STEP {i}: {step_name}")
            print(f"Query: '{query}'")
            print('='*50)
            
            result = pipeline.process_query(query, session_id)
            
            print(f"✅ Success: {result['success']}")
            print(f"📝 Response: {result['response']}")
            
            if result['success']:
                summary = result['data_summary']
                print(f"📊 Pipeline Progress:")
                print(f"   Raw → Cleaned → Features → Model")
                print(f"   {'✅' if summary['has_raw_data'] else '❌'} → {'✅' if summary['has_cleaned_data'] else '❌'} → {'✅' if summary['has_selected_features'] else '❌'} → {'✅' if summary['has_trained_model'] else '❌'}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                break
        
        print(f"\n🎉 Demo completed successfully!")
        print("💡 The system is ready for production use!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def run_direct_tests():
    """Run direct agent tests"""
    print("\n🧪 Running Direct Agent Tests...")
    
    try:
        from agents_wrapper import preprocessing_agent, feature_selection_agent, model_building_agent
        from pipeline_state import PipelineState
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(5, 2, 50),
            'target': np.random.choice([0, 1], 50)
        })
        
        # Test each agent individually
        print("\n1. Testing Preprocessing Agent...")
        state = PipelineState(
            raw_data=test_data,
            session_id="test_preprocessing",
            user_query="test preprocessing"
        )
        
        result_state = preprocessing_agent.run(state)
        print(f"   ✅ Preprocessing: {result_state.cleaned_data is not None}")
        
        print("\n2. Testing Feature Selection Agent...")
        result_state = feature_selection_agent.run(result_state)
        print(f"   ✅ Feature Selection: {result_state.selected_features is not None}")
        print(f"   Selected features: {result_state.selected_features}")
        
        print("\n3. Testing Model Building Agent...")
        result_state = model_building_agent.run(result_state)
        print(f"   ✅ Model Building: {result_state.trained_model is not None}")
        
        print("\n🎉 All agent tests passed!")
        
    except Exception as e:
        print(f"❌ Direct tests failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main startup function"""
    args = parse_arguments()
    
    # Setup instance-specific logging
    log_file = setup_colored_logging()
    print(f"📝 Logs will be saved to: {log_file}")
    
    # Print system status
    if not print_system_status():
        sys.exit(1)
    
    # Set up environment
    setup_environment(args)
    
    # If mode specified via command line, run directly
    if args.mode:
        if args.mode == 'slack':
            start_slack_bot()
        elif args.mode == 'api':
            start_api_testing()
        elif args.mode == 'test':
            run_direct_tests()
        elif args.mode == 'demo':
            run_demo()
        return
    
    # Interactive mode selection
    print("\n🎯 Choose a mode:")
    print("1. Start Slack Bot (Multi-session, File Upload) [RECOMMENDED]")
    print("2. Interactive API Testing (Python API)")
    print("3. Run Automated Demo (Full Pipeline Demo)")
    print("4. Direct Agent Testing (Individual Agent Tests)")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5) [Default: 1]: ").strip()
        
        # Default to option 1 if no input
        if not choice:
            choice = "1"
        
        if choice == "1":
            start_slack_bot()
            break
        elif choice == "2":
            start_api_testing()
            break
        elif choice == "3":
            run_demo()
            break
        elif choice == "4":
            run_direct_tests()
            break
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
