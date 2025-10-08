#!/usr/bin/env python3
from print_to_log import print_to_log
# Import master log handler to capture logger.info calls
try:
    import master_log_handler
except ImportError:
    pass

"""
Minimal Wrappers for Working Agents
Uses the actual working implementations AS-IS without modification
"""

import os
import tempfile
import time
import pandas as pd
import numpy as np
from typing import Optional, Any, Dict
from datetime import datetime

from pipeline_state import PipelineState

# Import thread logging system
from thread_logger import get_thread_logger

# Import your working agents AS-IS
try:
    from preprocessing_agent_impl import (
        run_sequential_agent as run_preprocessing_agent,
        SequentialState,
        PreprocessingPhase,
        detect_and_handle_extreme_outliers
    )
    # Also import the new Slack-compatible version
    from preprocessing_agent_slack import (
        create_slack_preprocessing_bot,
        SlackPreprocessingBot
    )
    PREPROCESSING_AVAILABLE = True
    print_to_log("✅ Preprocessing agent imported successfully")
except ImportError as e:
    print_to_log(f"❌ Preprocessing agent not available: {e}")
    PREPROCESSING_AVAILABLE = False

try:
    from feature_selection_agent_impl import (
        AgenticFeatureSelectionBot,
        UserSession,
        DataProcessor,
        LLMManager,
        MenuGenerator
    )
    FEATURE_SELECTION_AVAILABLE = True
    print_to_log("✅ Feature selection agent imported successfully")
except ImportError as e:
    print_to_log(f"❌ Feature selection agent not available: {e}")
    FEATURE_SELECTION_AVAILABLE = False

try:
    from model_building_agent_impl import LangGraphModelAgent
    MODEL_BUILDING_AVAILABLE = True
    print_to_log("✅ Model building agent imported successfully")
except ImportError as e:
    print_to_log(f"❌ Model building agent not available: {e}")
    MODEL_BUILDING_AVAILABLE = False


class PreprocessingAgentWrapper:
    """Minimal wrapper for the working preprocessing agent"""
    
    def __init__(self):
        self.available = PREPROCESSING_AVAILABLE
        self.slack_bot = None
        if self.available:
            try:
                # Initialize the Slack-compatible preprocessing bot
                self.slack_bot = create_slack_preprocessing_bot()
                print_to_log("✅ Slack preprocessing bot initialized")
            except Exception as e:
                print_to_log(f"❌ Failed to initialize Slack preprocessing bot: {e}")
                self.available = False
        

    def _apply_top10_onehot_encoding(self, df, column):
        """
        Apply one-hot encoding to top 10 most frequent categories.
        Other categories are grouped into 'Other' category.
        Max 11 columns created (top 10 + Other).
        """
        import pandas as pd
        
        # Get value counts and select top 10
        value_counts = df[column].value_counts()
        top_10_categories = value_counts.head(10).index.tolist()
        
        # Create a copy of the column
        encoded_col = df[column].copy()
        
        # Replace non-top-10 categories with 'Other'
        encoded_col = encoded_col.apply(
            lambda x: x if x in top_10_categories else 'Other'
        )
        
        # Apply one-hot encoding
        dummies = pd.get_dummies(encoded_col, prefix=column)
        
        # Concatenate with original dataframe and drop original column
        df_result = pd.concat([df, dummies], axis=1)
        df_result = df_result.drop(columns=[column])
        
        return df_result

    def run(self, state: PipelineState) -> PipelineState:
        """Route to interactive preprocessing via main Slack bot"""
        if not self.available:
            print_to_log("❌ Preprocessing agent not available - falling back to basic preprocessing")
            return self._run_basic_preprocessing_fallback(state)
            
        try:
            # Check if we have raw data
            if state.raw_data is None:
                print_to_log("❌ No raw data available for preprocessing")
                return state
                
            print_to_log(f"🚀 Launching interactive preprocessing workflow")
            print_to_log(f"🔧 DEBUG: Target column: {state.target_column}")
            print_to_log(f"🔧 DEBUG: Target column type: {type(state.target_column)}")
            print_to_log(f"🔧 DEBUG: Target column is None: {state.target_column is None}")
            print_to_log(f"🔧 DEBUG: Target column is empty string: {state.target_column == ''}")
            print_to_log(f"📊 Data shape: {state.raw_data.shape}")
            print_to_log(f"🔧 DEBUG: Available columns: {list(state.raw_data.columns)}")
            
            # Send interactive preprocessing menu via main Slack bot
            # Use the pipeline's slack_manager instead of the global one
            slack_manager = getattr(state, '_slack_manager', None)
            if not slack_manager:
                from toolbox import slack_manager as global_slack_manager
                slack_manager = global_slack_manager
            
            if slack_manager and state.chat_session:
                print_to_log(f"🔍 Debug: Sending Slack message to session: {state.chat_session}")
                
                if not state.target_column:
                    # Need target column first
                    initial_msg = f"""🧹 Sequential Preprocessing Agent

📁 Dataset loaded: {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns

🎯 Please specify your target column:
• Available columns: {', '.join(list(state.raw_data.columns)[:10])}{'...' if len(state.raw_data.columns) > 10 else ''}

📝 How to specify:
• Type: `target column_name` (e.g., `target default`)
• Or just: `column_name` (e.g., `default`)"""
                    
                    phase = "need_target"
                else:
                    # Show preprocessing menu
                    initial_msg = f"""🧹 Sequential Preprocessing Agent

📊 Current Dataset: {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns
🎯 Target Column: {state.target_column}

🔄 Preprocessing Phases:
• `Overview` - Dataset analysis and summary
• `Outliers` - Detect and handle outliers  
• `Missing Values` - Handle missing data
• `Encoding` - Categorical variable encoding
• `Transformations` - Feature transformations

💬 Your Options:
• `proceed` - Start preprocessing workflow
• `skip overview` - Skip to outlier detection
• `explain outliers` - Learn about outlier handling
• `summary` - Show current status

💬 What would you like to do?"""
                    
                    phase = "waiting_input"
                
                # Try to send message
                try:
                    print_to_log(f"🔍 DEBUG: About to call slack_manager.send_message")
                    print_to_log(f"🔍 DEBUG: slack_manager type: {type(slack_manager)}")
                    print_to_log(f"🔍 DEBUG: state.chat_session: {state.chat_session}")
                    print_to_log(f"🔍 DEBUG: message length: {len(initial_msg)}")
                    
                    result = slack_manager.send_message(state.chat_session, initial_msg)
                    print_to_log(f"🔍 DEBUG: send_message returned: {result}")
                    print_to_log("✅ Sent interactive preprocessing menu to Slack")
                except Exception as e:
                    print_to_log(f"❌ Failed to send Slack message: {e}")
                    print_to_log(f"🔍 Session channels: {getattr(slack_manager, 'session_channels', {})}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to basic preprocessing if Slack fails
                    return self._run_basic_preprocessing_fallback(state)
            else:
                print_to_log(f"❌ Cannot send Slack message - slack_manager: {slack_manager}, chat_session: {state.chat_session}")
                return self._run_basic_preprocessing_fallback(state)
            
            # Check if mode has already been selected (don't overwrite existing session)
            if hasattr(state, 'interactive_session') and state.interactive_session and state.interactive_session.get('mode_selected'):
                print_to_log(f"🎛️ Mode already selected: {state.interactive_session.get('mode_selected')} - continuing with preprocessing")
                
                # Continue with the selected mode instead of showing menu again
                if state.interactive_session.get('mode_selected') == 'slow':
                    print_to_log("🐌 Starting slow mode preprocessing workflow...")
                    return self.handle_interactive_command(state, "proceed")
                elif state.interactive_session.get('mode_selected') == 'fast':
                    print_to_log("⚡ Starting fast mode preprocessing workflow...")
                    return self._run_basic_preprocessing_fallback(state)
            
            # Set up interactive session state for continuation (only if not already set)
            if not hasattr(state, 'interactive_session') or not state.interactive_session:
                state.interactive_session = {
                    "agent_type": "preprocessing",
                    "session_active": True,
                    "session_id": state.chat_session,
                    "phase": phase,
                    "target_column": state.target_column,
                    "current_phase": "overview",
                    "needs_target": (phase == "need_target"),
                    "needs_mode_selection": False
                }
            
            # Set preprocessing state as active
            state.preprocessing_state = {
                "completed": False,
                "timestamp": datetime.now().isoformat(),
                "method": "interactive_slack",
                "session_active": True,
                "phase": phase
            }
            
            # Set appropriate response for the pipeline
            if phase == "need_target":
                state.last_response = "🎯 Please specify your target column to begin preprocessing."
            else:
                state.last_response = "🧹 Interactive preprocessing session started. Please follow the menu options sent to Slack."
            
            print_to_log("✅ Interactive preprocessing session started - user will interact via Slack")
            return state
            
        except Exception as e:
            print_to_log(f"❌ Interactive preprocessing setup failed: {e}")
            import traceback
            traceback.print_exc()
            print_to_log("🔄 Falling back to basic preprocessing")
            return self._run_basic_preprocessing_fallback(state)
    
    def handle_interactive_command(self, state: PipelineState, command: str) -> PipelineState:
        """Handle interactive commands for preprocessing"""
        # Set session context for proper logging
        try:
            from session_context import set_session_context, extract_session_from_session_id
            if state.session_id:
                user_id, thread_id = extract_session_from_session_id(state.session_id)
                set_session_context(user_id, thread_id)
        except ImportError:
            pass
            
        if not self.available:
            print_to_log("❌ Preprocessing agent not available")
            return state
        
        # Get thread logger
        if hasattr(state, 'chat_session') and state.chat_session:
            session_id = state.chat_session
            if '_' in session_id:
                parts = session_id.split('_')
                user_id = parts[0] if len(parts) >= 1 else session_id
                thread_id = '_'.join(parts[1:]) if len(parts) > 1 else session_id
            else:
                user_id = session_id
                thread_id = session_id
            thread_logger = get_thread_logger(user_id, thread_id)
            thread_logger.log_query(command, agent="preprocessing")
        else:
            thread_logger = None
            
        try:
            # Check current phase and handle accordingly
            current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
            print_to_log(f"🔧 DEBUG: Current phase: {current_phase}, Command: {command}")
            
            # 🎯 CRITICAL DEBUG: Check target column status
            print_to_log(f"🔧 DEBUG TARGET CHECK: state.target_column = '{state.target_column}'")
            print_to_log(f"🔧 DEBUG TARGET CHECK: hasattr(state, 'target_column') = {hasattr(state, 'target_column')}")
            if hasattr(state, 'interactive_session') and state.interactive_session:
                print_to_log(f"🔧 DEBUG TARGET CHECK: interactive_session['target_column'] = '{state.interactive_session.get('target_column', None)}'")
            
            # 🎯 TARGET SELECTION PHASE - Handle target column selection at the start
            # CRITICAL FIX: Check both state.target_column AND interactive_session['target_column']
            target_from_state = getattr(state, 'target_column', None)
            target_from_session = None
            if hasattr(state, 'interactive_session') and state.interactive_session:
                target_from_session = state.interactive_session.get('target_column', None)
            
            # Use target from interactive_session if state.target_column is empty
            if not target_from_state and target_from_session:
                print_to_log(f"🔧 FIX: Restoring target_column '{target_from_session}' from interactive_session to state")
                state.target_column = target_from_session
                target_from_state = target_from_session
            
            # Only handle target selection if NO target is set anywhere
            if not target_from_state and state.raw_data is not None and hasattr(state.raw_data, 'columns'):
                return self._handle_target_selection(state, command)
            
            # Log phase transition
            if thread_logger:
                thread_logger.debug(f"Preprocessing phase: {current_phase}", {"command": command})
            
            # ✅ COMPLETION PHASE HANDLER - Move to Feature Selection
            if current_phase == 'completion' and state.preprocessing_state.get('completed', False):
                print_to_log("🎯 Handling completion phase command...")
                
                # Check for feature selection intent using BGE (session continuation)
                if command.startswith('PROCEED: ') or command.lower() in ['yes', 'proceed', 'continue', 'feature_selection', 'next']:
                    print_to_log("🚀 Moving to feature selection phase...")
                    
                    # ✅ ENHANCED: Directly initialize feature selection agent with cleaned data
                    try:
                        # Get the data to use (prefer cleaned_data, fallback to raw_data)
                        data_to_use = state.cleaned_data if state.cleaned_data is not None else state.raw_data
                        if data_to_use is None:
                            raise ValueError("No data available for feature selection")
                        
                        print_to_log(f"🔧 DEBUG: Using data shape: {data_to_use.shape}")
                        print_to_log(f"🔧 DEBUG: Target column: {state.target_column}")
                        
                        # Save data to temporary CSV for feature selection agent
                        import tempfile
                        import os
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                            data_to_use.to_csv(tmp_file.name, index=False)
                            temp_csv_path = tmp_file.name
                        
                        print_to_log(f"🔧 DEBUG: Saved data to temp file: {temp_csv_path}")
                        
                        # Create UserSession for feature selection agent
                        from feature_selection_agent_impl import UserSession, DataProcessor
                        
                        session = UserSession(
                            user_id=state.chat_session,
                            file_path=temp_csv_path,
                            file_name=os.path.basename(temp_csv_path),
                            target_column=state.target_column,
                            current_features=list(data_to_use.columns),
                            analysis_chain=[],
                            snapshots={},
                            model_name="qwen2.5-coder:32b-instruct-q4_K_M"
                        )
                        
                        # Apply intelligent data cleaning (this is what FS agent does on startup)
                        success = DataProcessor.load_and_clean_data(session)
                        if not success:
                            raise ValueError("Failed to load and clean data")
                        
                        # Get the cleaned DataFrame from the session
                        clean_df = session.current_df
                        print_to_log(f"✅ Applied intelligent cleaning: {data_to_use.shape} → {clean_df.shape}")
                        
                        # Create the "after_cleaning" snapshot for revert functionality
                        from datetime import datetime as dt
                        session.snapshots["after_cleaning"] = {
                            "df": clean_df.copy(),
                            "features": list(clean_df.columns),
                            "timestamp": dt.now().isoformat()
                        }
                        print_to_log(f"✅ Created 'after_cleaning' snapshot with {clean_df.shape[1]} clean features for revert functionality")
                        
                        # Store session in feature selection agent
                        from agents_wrapper import feature_selection_agent
                        feature_selection_agent.bot.users[state.chat_session] = session
                        print_to_log(f"✅ Stored session in feature selection agent")
                        
                        # Generate and send feature selection menu
                        from feature_selection_agent_impl import MenuGenerator
                        menu_text = MenuGenerator.generate_main_menu(session)
                        
                        # Get slack manager
                        slack_manager = getattr(state, '_slack_manager', None)
                        if not slack_manager:
                            from toolbox import slack_manager as global_slack_manager
                            slack_manager = global_slack_manager
                        
                        if slack_manager and state.chat_session:
                            slack_manager.send_message(state.chat_session, menu_text)
                            print_to_log("✅ Sent feature selection menu to Slack")
                        
                        # Update state to feature selection
                        state.interactive_session = {
                            "agent_type": "feature_selection",
                            "session_active": True,
                            "phase": "waiting_input",
                            "current_phase": "menu"
                        }
                        print_to_log(f"🔧 DEBUG: Updated state.interactive_session: {state.interactive_session}")
                        print_to_log(f"✅ SUCCESS: Preprocessing to Feature Selection transition completed!")
                        print_to_log(f"✅ Feature selection agent initialized with {clean_df.shape[1]} features")
                        
                        # Set proper response message
                        state.last_response = f"🚀 Feature Selection Started! Initialized with {clean_df.shape[1]} clean features. Menu sent to Slack - ready for analysis!"
                        
                        # Clean up temp file
                        try:
                            os.unlink(temp_csv_path)
                        except:
                            pass
                        
                        return state
                        
                    except Exception as e:
                        print_to_log(f"❌ Error initializing feature selection: {e}")
                        import traceback
                        traceback.print_exc()
                        return state
                
                elif command.lower() in ['no', 'skip', 'stay', 'summary']:
                    print_to_log("📊 Staying in preprocessing completion...")
                    # Handle summary or other completion commands
                    return state
            
            # Handle the command using the preprocessing agent's interactive system
            if command.lower() == 'proceed':
                print_to_log("🚀 Starting preprocessing workflow with outliers phase")
                
                # Create a temporary file path for the DataFrame
                import tempfile
                import os
                
                # Create a temporary CSV file for the DataFrame
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                    state.raw_data.to_csv(tmp_file.name, index=False)
                    df_path = tmp_file.name
                
                try:
                    # Start the actual preprocessing workflow
                    print_to_log("🔧 Running preprocessing agent for outlier analysis...")
                    
                    # Import the preprocessing functions
                    from preprocessing_agent_impl import (
                        initialize_dataset_analysis,
                        analyze_outliers_with_confidence,
                        get_llm_from_state,
                        SequentialState
                    )
                    
                    # Create a proper SequentialState for the preprocessing functions
                    sequential_state = SequentialState(
                        df=state.raw_data,
                        df_path=df_path,
                        target_column=state.target_column,
                        model_name=os.environ.get("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
                    )
                    
                    # Initialize dataset analysis
                    print_to_log("📊 Initializing dataset analysis...")
                    sequential_state = initialize_dataset_analysis(sequential_state)
                    
                    # Run outlier detection with confidence-based approach
                    print_to_log("🔍 Running confidence-based outlier detection...")
                    outlier_results = analyze_outliers_with_confidence(sequential_state)
                    
                    # Debug: Check what we got back
                    print_to_log(f"🔍 DEBUG: outlier_results type: {type(outlier_results)}")
                    print_to_log(f"🔍 DEBUG: outlier_results content: {outlier_results}")
                    
                    # Generate a summary message
                    outlier_columns = []
                    total_outliers = 0
                    
                    # Handle different possible return types from outlier analysis
                    if isinstance(outlier_results, dict):
                        # Check for new confidence-based structure first
                        if 'outliers_columns' in outlier_results:
                            # New confidence-based format
                            outlier_columns = outlier_results['outliers_columns']
                            total_outliers = len(outlier_columns)  # Count of columns with outliers
                        elif 'outlier_columns' in outlier_results:
                            # Old format
                            outlier_columns = outlier_results['outlier_columns']
                            total_outliers = 0
                            if 'analysis_details' in outlier_results:
                                for col, details in outlier_results['analysis_details'].items():
                                    if col in outlier_columns:
                                        total_outliers += details.get('outliers_iqr_count', 0)
                        else:
                            # Fallback to old structure
                            outlier_columns = [col for col, result in outlier_results.items() if result.get('outlier_count', 0) > 0]
                            total_outliers = sum(result.get('outlier_count', 0) for result in outlier_results.values())
                    elif isinstance(outlier_results, list):
                        # If it's a list, assume it contains column names with outliers
                        outlier_columns = outlier_results
                        total_outliers = len(outlier_results)
                    else:
                        print_to_log(f"⚠️ Unexpected outlier_results type: {type(outlier_results)}")
                        outlier_columns = []
                        total_outliers = 0
                    
                    # Send results to Slack
                    slack_manager = getattr(state, '_slack_manager', None)
                    if not slack_manager:
                        from toolbox import slack_manager as global_slack_manager
                        slack_manager = global_slack_manager
                    
                    if slack_manager and state.chat_session:
                        # Build concise outlier summary for new confidence-based format
                        if isinstance(outlier_results, dict) and 'llm_recommendations' in outlier_results:
                            # New confidence-based format - show strategy summary
                            recommendations = outlier_results['llm_recommendations']
                            strategy_counts = {}
                            
                            for col, rec in recommendations.items():
                                strategy = rec.get('treatment', 'unknown')
                                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                            
                            # Create concise strategy summary
                            strategy_summary = []
                            for strategy, count in strategy_counts.items():
                                if strategy == 'keep':
                                    strategy_summary.append(f"Keep as-is: {count} columns")
                                elif strategy == 'winsorize':
                                    strategy_summary.append(f"Winsorize: {count} columns")
                                elif strategy == 'remove':
                                    strategy_summary.append(f"Remove: {count} columns")
                                else:
                                    strategy_summary.append(f"{strategy.title()}: {count} columns")
                            
                            outlier_details = chr(10).join(strategy_summary) if strategy_summary else "• No outlier treatment needed"
                            
                            # Confidence stats removed from display per user request
                            
                        elif isinstance(outlier_results, dict) and 'analysis_details' in outlier_results:
                            # Old format with analysis_details
                            outlier_details = []
                            for col in outlier_columns[:5]:
                                if col in outlier_results['analysis_details']:
                                    details = outlier_results['analysis_details'][col]
                                    outlier_count = details.get('outliers_iqr_count', 0)
                                    outlier_percentage = details.get('outliers_iqr_percentage', 0)
                                    outlier_details.append(f'• {col}: {outlier_count:,} outliers ({outlier_percentage:.1f}%)')
                                else:
                                    outlier_details.append(f'• {col}: outliers detected')
                            outlier_details = chr(10).join(outlier_details)
                        else:
                            # Fallback
                            outlier_details = "• Analysis completed"
                        
                        # Create appropriate message based on format
                        if isinstance(outlier_results, dict) and 'llm_recommendations' in outlier_results:
                            # New confidence-based format
                            analyzed_columns = len(outlier_results.get('llm_recommendations', {}))
                            columns_needing_treatment = len([col for col, rec in outlier_results.get('llm_recommendations', {}).items() 
                                                           if rec.get('treatment', 'keep') != 'keep'])
                            
                            message = f"""🔍 Outlier Analysis Complete!

📊 Dataset Overview:
• Total rows: {state.raw_data.shape[0]:,}
• Total columns: {state.raw_data.shape[1]}
• Target column: {state.target_column}

🎯 Analysis Results:
• Columns analyzed: {analyzed_columns}
• Columns needing treatment: {columns_needing_treatment}

🔧 Recommended Treatments:
{outlier_details}

💬 Next Steps:
• `continue` - Apply recommendations and move to missing values
• `skip outliers` - Move to missing values analysis
• `summary` - Show current preprocessing status"""
                        else:
                            # Old format
                            message = f"""🔍 Outlier Analysis Complete!

📊 Dataset Overview:
• Total rows: {state.raw_data.shape[0]:,}
• Total columns: {state.raw_data.shape[1]}
• Target column: {state.target_column}

🎯 Outlier Detection Results:
• Columns with outliers: {len(outlier_columns)}
• Total outliers found: {total_outliers:,}

📋 Columns with Outliers:
{outlier_details}{'...' if len(outlier_columns) > 5 else ''}

💬 Next Steps:
• `continue` - Apply recommendations and move to missing values
• `skip outliers` - Move to missing values analysis
• `summary` - Show current preprocessing status"""
                        
                        slack_manager.send_message(state.chat_session, message)
                    
                    # Update state
                    from datetime import datetime
                    state.preprocessing_state = {
                        "completed": False,
                        "timestamp": datetime.now().isoformat(),
                        "method": "interactive_sequential",
                        "current_phase": "outliers",
                        "status": "analysis_complete",
                        "outlier_results": outlier_results
                    }
                    
                    # Convert numpy types to native Python types for JSON serialization
                    if state.preprocessing_state.get('outlier_results'):
                        import json
                        # Convert the outlier_results to JSON-serializable format
                        outlier_results_serializable = json.loads(
                            json.dumps(state.preprocessing_state['outlier_results'], 
                                     default=lambda x: float(x) if hasattr(x, 'item') else x)
                        )
                        state.preprocessing_state['outlier_results'] = outlier_results_serializable
                    
                    # Update interactive session
                    if state.interactive_session:
                        state.interactive_session["current_phase"] = "outliers"
                        state.interactive_session["phase"] = "analysis_complete"
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(df_path)
                    except:
                        pass
                
                return state
            elif command.lower() == 'continue':
                print_to_log("🔄 Applying LLM recommendations and moving to next phase")
                # Get current phase from either preprocessing_state or interactive_session
                if state.preprocessing_state and 'current_phase' in state.preprocessing_state:
                    current_phase = state.preprocessing_state.get('current_phase')
                elif state.interactive_session and 'current_phase' in state.interactive_session:
                    current_phase = state.interactive_session.get('current_phase')
                else:
                    current_phase = 'overview'  # default
                print_to_log(f"🔧 DEBUG: Current phase for continue: {current_phase}")

                if current_phase == 'overview':
                    # Start preprocessing workflow - begin with outliers phase
                    print_to_log("🚀 Starting preprocessing workflow with outliers phase")
                    print_to_log("🔧 Running preprocessing agent for outlier analysis...")
                    print_to_log(f"🔧 DEBUG INPUT TO OUTLIERS: Command='{command}', State data shape: {state.raw_data.shape if state.raw_data is not None else 'None'}")
                    print_to_log(f"🔧 DEBUG INPUT TO OUTLIERS: Target column: {state.target_column}")
                    print_to_log(f"🔧 DEBUG INPUT TO OUTLIERS: Interactive session: {state.interactive_session}")
                    
                    # Initialize dataset analysis
                    try:
                        from preprocessing_agent_impl import initialize_dataset_analysis
                        
                        # Save raw data to temp file for analysis
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                            df_path = f.name
                            state.raw_data.to_csv(df_path, index=False)
                        
                        # Initialize analysis with target column
                        target_column = state.target_column
                        print_to_log(f"🔧 DEBUG: Target column: {target_column}")
                        print_to_log(f"🔧 DEBUG: Target column type: {type(target_column)}")
                        print_to_log(f"🔧 DEBUG: Target column is None: {target_column is None}")
                        print_to_log(f"🔧 DEBUG: Target column is empty string: {target_column == ''}")
                        
                        # Run outlier analysis
                        print_to_log("🔍 Running outlier detection...")
                        
                        from preprocessing_agent_impl import (
                            initialize_dataset_analysis,
                            analyze_outliers_with_confidence,
                            get_llm_from_state,
                            SequentialState
                        )
                        
                        # Create a proper SequentialState for the preprocessing functions
                        sequential_state = SequentialState(
                            df=state.raw_data,
                            df_path=df_path,
                            target_column=state.target_column,
                            model_name=os.environ.get("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
                        )
                        
                        # Initialize dataset analysis
                        print_to_log("📊 Initializing dataset analysis...")
                        sequential_state = initialize_dataset_analysis(sequential_state)
                        
                        # Run outlier analysis with confidence-based approach
                        print_to_log("🔍 Running confidence-based outlier detection...")
                        outlier_results = analyze_outliers_with_confidence(sequential_state)
                        
                        print_to_log(f"🔍 DEBUG: outlier_results type: {type(outlier_results)}")
                        print_to_log(f"🔍 DEBUG: outlier_results content: {outlier_results}")
                        
                        # Send results to Slack
                        if hasattr(state, '_slack_manager') and state._slack_manager and state.chat_session:
                            slack_manager = state._slack_manager
                            
                            if isinstance(outlier_results, dict) and 'llm_recommendations' in outlier_results:
                                # Group recommendations by treatment type
                                treatment_groups = {}
                                for column, details in outlier_results['llm_recommendations'].items():
                                    treatment = details.get('treatment', 'unknown')
                                    if treatment not in treatment_groups:
                                        treatment_groups[treatment] = []
                                    treatment_groups[treatment].append(column)
                                
                                # Create concise message
                                treatment_text = []
                                for treatment, columns in treatment_groups.items():
                                    if len(columns) <= 3:
                                        col_text = ", ".join(columns)
                                    else:
                                        col_text = f"{', '.join(columns[:3])} (+{len(columns)-3} more)"
                                    
                                    treatment_display = {
                                        'winsorize': '📊 Winsorize',
                                        'keep': '✅ Keep as-is', 
                                        'clip': '✂️ Clip',
                                        'remove': '🗑️ Remove'
                                    }.get(treatment, f'🔧 {treatment.title()}')
                                    
                                    treatment_text.append(f"{treatment_display}: {col_text}")
                                
                                message = f"""🚨 Outliers Analysis Complete!

📊 Outlier Columns Found: {len(outlier_results.get('outliers_columns', []))} columns

🔧 Recommended Treatments:
{chr(10).join(treatment_text)}

🔄 Ready for Next Step:
• `continue` - Apply treatments and move to missing values
• `skip outliers` - Skip to missing values phase  
• `summary` - Show current preprocessing status"""
                            else:
                                message = f"""🚨 Outliers Analysis Complete!

📊 Analysis Results:
{outlier_results}

🔄 Ready for Next Step:
• `continue` - Apply treatments and move to missing values
• `skip outliers` - Skip to missing values phase
• `summary` - Show current preprocessing status"""
                            
                            slack_manager.send_message(state.chat_session, message)
                        
                        # Update state with outlier results
                        state.preprocessing_state.update({
                            "current_phase": "outliers",
                            "outlier_results": outlier_results,
                            "status": "analysis_complete"
                        })
                        
                        # Clean up temp file
                        try:
                            os.unlink(df_path)
                        except:
                            pass
                        
                        return state
                        
                    except Exception as e:
                        print_to_log(f"❌ Outlier analysis failed: {e}")
                        import traceback
                        traceback.print_exc()
                        return state

                elif current_phase == 'outliers':
                    # Apply outlier treatments and move to missing_values
                    print_to_log("🔧 Applying outlier treatments...")
                    
                    # Get outlier results from state
                    outlier_results = state.preprocessing_state.get('outlier_results', {})
                    if not outlier_results:
                        print_to_log("❌ No outlier results found in state")
                        return state
                    
                    # Apply treatments based on LLM recommendations
                    df = state.raw_data.copy()
                    
                    # 🚨 STEP 1: Always handle extreme outliers first (data quality)
                    print_to_log("🚨 Handling extreme outliers during outlier treatment...")
                    import preprocessing_agent_impl
                    df, extreme_report = preprocessing_agent_impl.detect_and_handle_extreme_outliers(df)
                    
                    if extreme_report['total_extreme_outliers'] > 0:
                        print_to_log(f"🚨 Handled {extreme_report['total_extreme_outliers']} extreme outliers during treatment:")
                        for col, details in extreme_report['extreme_outliers_found'].items():
                            print_to_log(f"   • {col}: {details['count']} extreme values ({details['percentage']:.2f}%) → NaN")
                    else:
                        print_to_log("✅ No extreme outliers found - data quality is good")
                    
                    # 🎯 STEP 2: Apply LLM/rule recommendations on cleaned data
                    print_to_log("🔧 Applying outlier treatment recommendations...")
                    treatment_counts = {}
                    
                    if isinstance(outlier_results, dict) and 'llm_recommendations' in outlier_results:
                        for col, recommendation in outlier_results['llm_recommendations'].items():
                            raw_treatment = recommendation.get('treatment', 'winsorize')
                            treatment = str(raw_treatment).lower().replace('-', '_')
                            if treatment == 'winsorize':
                                treatment_counts['Winsorized'] = treatment_counts.get('Winsorized', 0) + 1
                                # Apply winsorization
                                lower_percentile = 1
                                upper_percentile = 99
                                lower_val = df[col].quantile(lower_percentile / 100)
                                upper_val = df[col].quantile(upper_percentile / 100)
                                df[col] = df[col].clip(lower=lower_val, upper=upper_val)
                            elif treatment == 'remove':
                                treatment_counts['Outliers removed'] = treatment_counts.get('Outliers removed', 0) + 1
                                # Remove outliers using IQR method
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                            elif treatment == 'mark_missing':
                                treatment_counts['Marked as missing'] = treatment_counts.get('Marked as missing', 0) + 1
                                # Mark detected outliers as NaN for later imputation
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                                df.loc[outlier_mask, col] = np.nan
                            elif treatment == 'keep':
                                treatment_counts['Kept as-is'] = treatment_counts.get('Kept as-is', 0) + 1
                    
                    # Create concise treatment summary
                    applied_treatments = []
                    for treatment, count in treatment_counts.items():
                        applied_treatments.append(f"{treatment}: {count} columns")
                    
                    # Update state with processed data
                    state.cleaned_data = df
                    print_to_log(f"🔧 DEBUG: Set cleaned_data shape: {df.shape}")
                    
                    # Send confirmation message
                    slack_manager = getattr(state, '_slack_manager', None)
                    if not slack_manager:
                        from toolbox import slack_manager as global_slack_manager
                        slack_manager = global_slack_manager
                    
                    if slack_manager and state.chat_session:
                        treatments_text = "\n".join(applied_treatments) if applied_treatments else "• No treatments applied"
                        
                        message = f"""✅ Outlier Treatments Applied!

🔧 Applied Treatments:
{treatments_text}

📊 Data Summary:
• Original: {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns
• Processed: {df.shape[0]:,} rows × {df.shape[1]} columns
• Rows removed: {state.raw_data.shape[0] - df.shape[0]:,}

🔄 Moving to Next Phase: Missing Values Analysis

💬 Next Steps:
• `continue` - Start missing values analysis
• `skip missing` - Move to encoding phase
• `summary` - Show current status"""
                        
                        slack_manager.send_message(state.chat_session, message)
                    
                    # 🎯 SAVE OUTLIER STRATEGIES TO SESSION STATE
                    state.save_preprocessing_strategy(
                        phase="outliers",
                        phase_results=outlier_results,
                        target_column=state.target_column,
                        original_columns=list(state.raw_data.columns) if state.raw_data is not None else []
                    )
                    
                    # Update state for next phase
                    state.preprocessing_state.update({
                        "current_phase": "missing_values",
                        "status": "ready_for_next_phase",
                        "treatments_applied": applied_treatments
                    })
                    
                    # Convert numpy types to native Python types for JSON serialization
                    import json
                    outlier_results_serializable = json.loads(
                        json.dumps(outlier_results,
                                 default=lambda x: float(x) if hasattr(x, 'item') else x)
                    )
                    state.preprocessing_state["outlier_results"] = outlier_results_serializable
                    
                    if state.interactive_session:
                        state.interactive_session["current_phase"] = "missing_values"
                        state.interactive_session["phase"] = "ready"
                    
                    return state

                elif current_phase == 'missing_values':
                    # Check if we already have missing values results
                    missing_results = state.preprocessing_state.get('missing_results')
                    if missing_results:
                        # Apply missing values treatments and move to encoding
                        print_to_log("🔧 Applying missing values treatments...")
                        
                        df = state.cleaned_data.copy() if state.cleaned_data is not None else state.raw_data.copy()
                        strategy_counts = {}
                        
                        if isinstance(missing_results, dict) and 'llm_recommendations' in missing_results:
                            for col, recommendation in missing_results['llm_recommendations'].items():
                                raw_strategy = recommendation.get('strategy', 'median')
                                strategy = str(raw_strategy).lower().replace('-', '_')
                                if strategy == 'median':
                                    df[col] = df[col].fillna(df[col].median())
                                    strategy_counts['Median'] = strategy_counts.get('Median', 0) + 1
                                elif strategy == 'mean':
                                    df[col] = df[col].fillna(df[col].mean())
                                    strategy_counts['Mean'] = strategy_counts.get('Mean', 0) + 1
                                elif strategy == 'mode':
                                    mode_val = df[col].mode()
                                    fill_val = mode_val.iloc[0] if not mode_val.empty else df[col].dropna().iloc[0] if df[col].dropna().shape[0] else 0
                                    df[col] = df[col].fillna(fill_val)
                                    strategy_counts['Mode'] = strategy_counts.get('Mode', 0) + 1
                                elif strategy == 'constant':
                                    # Smart constant defaults based on data type
                                    if 'constant_value' in recommendation:
                                        constant_value = recommendation['constant_value']
                                    elif pd.api.types.is_numeric_dtype(df[col]):
                                        constant_value = -222  # Standard constant for numeric columns
                                    else:
                                        constant_value = 'Unknown'  # 'Unknown' for categorical columns
                                    
                                    df[col] = df[col].fillna(constant_value)
                                    strategy_counts['Constant'] = strategy_counts.get('Constant', 0) + 1
                                elif strategy == 'model_based':
                                    # Simple model-based imputation (fallback to median/mode)
                                    if pd.api.types.is_numeric_dtype(df[col]):
                                        df[col] = df[col].fillna(df[col].median())
                                        strategy_counts['Model-based'] = strategy_counts.get('Model-based', 0) + 1
                                    else:
                                        mode_val = df[col].mode()
                                        fill_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                                        df[col] = df[col].fillna(fill_val)
                                        strategy_counts['Model-based'] = strategy_counts.get('Model-based', 0) + 1
                                else:
                                    # Simple fallback for any unknown strategy
                                    if pd.api.types.is_numeric_dtype(df[col]):
                                        df[col] = df[col].fillna(df[col].median())
                                        strategy_counts['Median (default)'] = strategy_counts.get('Median (default)', 0) + 1
                                    else:
                                        mode_val = df[col].mode()
                                        fill_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                                        df[col] = df[col].fillna(fill_val)
                                        strategy_counts['Mode (default)'] = strategy_counts.get('Mode (default)', 0) + 1
                        
                        # Create concise treatment summary
                        applied_treatments = []
                        for strategy, count in strategy_counts.items():
                            applied_treatments.append(f"{strategy}: {count} columns")

                        # Update state with processed data
                        state.cleaned_data = df
                        print_to_log(f"🔧 DEBUG: Set cleaned_data shape after missing values: {df.shape}")
                        
                        # Send confirmation message
                        slack_manager = getattr(state, '_slack_manager', None)
                        if not slack_manager:
                            from toolbox import slack_manager as global_slack_manager
                            slack_manager = global_slack_manager
                        
                        if slack_manager and state.chat_session:
                            treatments_text = "\n".join(applied_treatments) if applied_treatments else "• No treatments applied"
                            
                            message = f"""✅ Missing Values Treatments Applied!

🔧 Applied Treatments:
{treatments_text}

📊 Data Summary:
• Processed: {df.shape[0]:,} rows × {df.shape[1]} columns
• Missing values filled: {len(applied_treatments)} columns

🔄 Moving to Next Phase: Encoding Analysis

💬 Next Steps:
• `continue` - Start encoding analysis
• `skip encoding` - Move to transformations phase
• `summary` - Show current status"""
                            
                            slack_manager.send_message(state.chat_session, message)
                        
                        # 🎯 SAVE MISSING VALUES STRATEGIES TO SESSION STATE
                        state.save_preprocessing_strategy(
                            phase="missing_values",
                            phase_results=missing_results,
                            target_column=state.target_column
                        )                        
                        # 🎯 SAVE MISSING VALUES STRATEGIES TO SESSION STATE
                        state.save_preprocessing_strategy(
                            phase="missing_values",
                            phase_results=missing_results,
                            target_column=state.target_column
                        )
                        
                        # Update state for next phase
                        state.preprocessing_state.update({
                            "current_phase": "encoding",
                            "status": "ready_for_next_phase",
                            "missing_treatments_applied": applied_treatments
                        })
                        
                        # Convert numpy types to native Python types for JSON serialization
                        import json
                        missing_results_serializable = json.loads(
                            json.dumps(missing_results,
                                     default=lambda x: float(x) if hasattr(x, 'item') else x)
                        )
                        state.preprocessing_state["missing_results"] = missing_results_serializable
                        
                        if state.interactive_session:
                            state.interactive_session["current_phase"] = "encoding"
                            state.interactive_session["phase"] = "ready"
                        return state
                    else:
                        # Start missing values analysis
                        print_to_log("🔍 Starting missing values analysis...")
                        print_to_log(f"🔧 DEBUG INPUT TO MISSING_VALUES: Command='{command}', State data shape: {state.raw_data.shape if state.raw_data is not None else 'None'}")
                        print_to_log(f"🔧 DEBUG INPUT TO MISSING_VALUES: Target column: {state.target_column}")
                        print_to_log(f"🔧 DEBUG INPUT TO MISSING_VALUES: Interactive session: {state.interactive_session}")
                        
                        # Import missing values functions
                        from preprocessing_agent_impl import (
                            analyze_missing_values_with_confidence,
                            get_llm_from_state,
                            SequentialState
                        )
                        
                        # Create a temporary file path for the DataFrame
                        import tempfile
                        import os
                        
                        # Use cleaned_data if available, otherwise raw_data
                        data_to_analyze = state.cleaned_data if state.cleaned_data is not None else state.raw_data
                        
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                            data_to_analyze.to_csv(tmp_file.name, index=False)
                            df_path = tmp_file.name
                        
                        try:
                            # Create SequentialState for missing values analysis
                            sequential_state = SequentialState(
                                df=data_to_analyze,
                                df_path=df_path,
                                target_column=state.target_column,
                                model_name=os.environ.get("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
                            )
                            
                            # Run missing values analysis
                            print_to_log("🔍 Running confidence-based missing values analysis...")
                            missing_results = analyze_missing_values_with_confidence(sequential_state)
                            
                            # Send results to Slack
                            slack_manager = getattr(state, '_slack_manager', None)
                            if not slack_manager:
                                from toolbox import slack_manager as global_slack_manager
                                slack_manager = global_slack_manager
                            
                            if slack_manager and state.chat_session:
                                # Build missing values details and LLM recommendations
                                if isinstance(missing_results, dict) and 'missing_values_columns' in missing_results:
                                    # New confidence-based format
                                    missing_columns = missing_results['missing_values_columns']
                                    llm_recommendations = missing_results.get('llm_recommendations', {})
                                    
                                    if missing_columns and llm_recommendations:
                                        # Group columns by strategy for concise display
                                        strategy_counts = {}
                                        for col, rec in llm_recommendations.items():
                                            strategy = rec.get('strategy', 'unknown')
                                            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                                        
                                        # Build concise strategy summary
                                        strategy_summary = []
                                        for strategy, count in strategy_counts.items():
                                            if strategy == 'drop_column':
                                                strategy_summary.append(f"Drop columns: {count} columns")
                                            elif strategy == 'drop_missing':
                                                strategy_summary.append(f"Drop missing rows: {count} columns")
                                            else:
                                                strategy_summary.append(f"{strategy.title()} imputation: {count} columns")
                                        
                                        strategy_text = "\n".join(strategy_summary)
                                        
                                        # Confidence stats removed from display per user request
                                        
                                        message = f"""🔍 Missing Values Analysis Complete!

📊 Missing Values Found: {len(missing_columns)} columns

🔧 Recommended Strategies:
{strategy_text}

🔄 Ready for Next Step:
• `continue` - Apply missing values treatments and move to encoding
• `skip missing` - Move directly to encoding phase
• `summary` - Show current preprocessing status"""
                                    else:
                                        message = f"""🔍 Missing Values Analysis Complete!

📊 No missing values found - Dataset is complete!

🔄 Ready for Next Step:
• `continue` - Move to encoding phase
• `summary` - Show current preprocessing status"""
                                
                                elif isinstance(missing_results, dict) and 'missing_columns' in missing_results:
                                    # Old format
                                    missing_columns = missing_results['missing_columns']
                                    llm_recommendations = missing_results.get('llm_recommendations', {})
                                    
                                    # Group columns by strategy for concise display
                                    strategy_groups = {}
                                    for col in missing_columns:
                                        if col in llm_recommendations:
                                            strategy = llm_recommendations[col].get('strategy', 'unknown')
                                            if strategy not in strategy_groups:
                                                strategy_groups[strategy] = []
                                            strategy_groups[strategy].append(col)
                                    
                                    # Build concise strategy summary
                                    strategy_summary = []
                                    for strategy, cols in strategy_groups.items():
                                        cols_str = ', '.join(cols[:5])  # Show first 5 columns
                                        if len(cols) > 5:
                                            cols_str += f" (+{len(cols)-5} more)"
                                        strategy_summary.append(f"{strategy.title()} imputation: {cols_str}")
                                    
                                    strategy_text = "\n".join(strategy_summary)
                                    
                                    message = f"""🔍 Missing Values Analysis Complete!

📊 Missing Values Found: {len(missing_columns)} columns

🔧 Recommended Strategies:
{strategy_text}

🔄 Ready for Next Step:
• `continue` - Apply missing values treatments and move to encoding
• `skip missing` - Move directly to encoding phase
• `summary` - Show current preprocessing status"""
                                else:
                                    # Fallback - avoid showing raw JSON
                                    message = f"""🔍 Missing Values Analysis Complete!

📊 Analysis completed successfully

🔄 Ready for Next Step:
• `continue` - Apply missing values treatments and move to encoding
• `skip missing` - Move directly to encoding phase
• `summary` - Show current preprocessing status"""
                                
                                slack_manager.send_message(state.chat_session, message)
                            
                            # Update state with missing values results
                            # Convert numpy types to native Python types for JSON serialization
                            import json
                            missing_results_serializable = json.loads(
                                json.dumps(missing_results,
                                         default=lambda x: float(x) if hasattr(x, 'item') else x)
                            )
                            state.preprocessing_state.update({
                                "missing_results": missing_results_serializable,
                                "status": "missing_analysis_complete"
                            })
                            
                            try:
                                os.unlink(df_path)
                            except:
                                pass
                            
                            return state
                        except Exception as e:
                            print_to_log(f"❌ Missing values analysis failed: {e}")
                            import traceback
                            traceback.print_exc()
                            return state

                elif current_phase == 'encoding':
                    # Check if we already have encoding results
                    encoding_results = state.preprocessing_state.get('encoding_results')
                    if encoding_results:
                        # Apply encoding treatments and move to transformations
                        print_to_log("🔧 Applying encoding treatments...")
                        
                        df = state.cleaned_data.copy() if state.cleaned_data is not None else state.raw_data.copy()
                        strategy_counts = {}
                        
                        if isinstance(encoding_results, dict) and 'llm_recommendations' in encoding_results:
                            for col, recommendation in encoding_results['llm_recommendations'].items():
                                # Normalize key names and values from LLM/analysis
                                raw_type = recommendation.get('encoding_type') or recommendation.get('strategy') or recommendation.get('encoding') or 'label_encoding'
                                enc_norm = str(raw_type).lower().replace('-', '_')
                                if enc_norm in ['label_encoding', 'label']:
                                    enc_choice = 'label'
                                elif enc_norm in ['onehot_encoding', 'one_hot', 'onehot']:
                                    enc_choice = 'onehot'
                                elif enc_norm in ['ordinal_encoding', 'ordinal']:
                                    enc_choice = 'ordinal'
                                elif enc_norm in ['target_encoding', 'target']:
                                    enc_choice = 'target'
                                elif enc_norm in ['binary_encoding', 'binary']:
                                    enc_choice = 'binary'
                                elif enc_norm in ['drop_column', 'drop']:
                                    enc_choice = 'drop_column'
                                elif enc_norm in ['skip']:
                                    enc_choice = 'skip'
                                else:
                                    enc_choice = 'label'
                                
                                if enc_choice == 'label':
                                    from sklearn.preprocessing import LabelEncoder
                                    le = LabelEncoder()
                                    df[col] = le.fit_transform(df[col].astype(str))
                                    strategy_counts['Label encoding'] = strategy_counts.get('Label encoding', 0) + 1
                                elif enc_choice == 'ordinal':
                                    # Apply ordinal encoding
                                    unique_values = df[col].astype(str).unique()
                                    value_map = {val: idx for idx, val in enumerate(unique_values)}
                                    df[col] = df[col].astype(str).map(value_map)
                                    strategy_counts['Ordinal encoding'] = strategy_counts.get('Ordinal encoding', 0) + 1
                                elif enc_choice == 'target':
                                    # Simple target mean encoding
                                    try:
                                        target_col = state.target_column
                                        if target_col and target_col in df.columns:
                                            means = df.groupby(col)[target_col].mean()
                                            df[col] = df[col].map(means)
                                            strategy_counts['Target encoding'] = strategy_counts.get('Target encoding', 0) + 1
                                        else:
                                            # Fallback to label if target not available
                                            from sklearn.preprocessing import LabelEncoder
                                            le = LabelEncoder()
                                            df[col] = le.fit_transform(df[col].astype(str))
                                            strategy_counts['Label encoding (fallback)'] = strategy_counts.get('Label encoding (fallback)', 0) + 1
                                    except Exception:
                                        from sklearn.preprocessing import LabelEncoder
                                        le = LabelEncoder()
                                        df[col] = le.fit_transform(df[col].astype(str))
                                        strategy_counts['Label encoding (fallback)'] = strategy_counts.get('Label encoding (fallback)', 0) + 1
                                elif enc_choice == 'onehot' or enc_choice == 'onehot_encoding':
                                    # Smart one-hot encoding: Top-10 + Other for medium cardinality
                                    unique_count = df[col].nunique()
                                    if unique_count > 10:  # Use top-10 for medium/high cardinality
                                        try:
                                            df = self._apply_top10_onehot_encoding(df, col)
                                            strategy_counts['One-hot (top-10)'] = strategy_counts.get('One-hot (top-10)', 0) + 1
                                        except Exception:
                                            # Fallback to regular one-hot
                                            df = pd.get_dummies(df, columns=[col], prefix=col)
                                            strategy_counts['One-hot encoding'] = strategy_counts.get('One-hot encoding', 0) + 1
                                    else:
                                        # Regular one-hot for low cardinality
                                        df = pd.get_dummies(df, columns=[col], prefix=col)
                                        strategy_counts['One-hot encoding'] = strategy_counts.get('One-hot encoding', 0) + 1
                                elif enc_choice == 'skip':
                                    # Skip encoding - leave column as-is
                                    strategy_counts['Skipped'] = strategy_counts.get('Skipped', 0) + 1
                                elif enc_choice == 'drop_column':
                                    if col in df.columns:
                                        df = df.drop(columns=[col])
                                        strategy_counts['Dropped'] = strategy_counts.get('Dropped', 0) + 1
                        
                        # Create concise treatment summary
                        applied_treatments = []
                        for strategy, count in strategy_counts.items():
                            applied_treatments.append(f"{strategy}: {count} columns")

                        # Update state with processed data
                        state.cleaned_data = df
                        print_to_log(f"🔧 DEBUG: Set cleaned_data shape after encoding: {df.shape}")
                        
                        # 🎯 SAVE ENCODING STRATEGIES TO SESSION STATE
                        state.save_preprocessing_strategy(
                            phase="encoding",
                            phase_results=encoding_results,
                            target_column=state.target_column
                        )
                        
                        # Store success message for later sending (after CSV save)
                        treatments_text = "\n".join(applied_treatments) if applied_treatments else "• No treatments applied"
                        state.pending_slack_message = f"""✅ Encoding Treatments Applied!

🔧 Applied Treatments:
{treatments_text}

📊 Data Summary:
• Processed: {df.shape[0]:,} rows × {df.shape[1]} columns
• Encodings applied: {len(applied_treatments)} columns

🔄 Moving to Next Phase: Transformations Analysis

💬 Next Steps:
• `continue` - Start transformations analysis
• `skip transformations` - Complete preprocessing
• `summary` - Show current status"""
                        
                        # Update state for next phase
                        state.preprocessing_state.update({
                            "current_phase": "transformations",
                            "status": "ready_for_next_phase",
                            "encoding_treatments_applied": applied_treatments
                        })
                        
                        if state.interactive_session:
                            state.interactive_session["current_phase"] = "transformations"
                            state.interactive_session["phase"] = "ready"
                        return state
                    else:
                        # Start encoding analysis
                        print_to_log("🔍 Starting encoding analysis...")
                        print_to_log(f"🔧 DEBUG INPUT TO ENCODING: Command='{command}', State data shape: {state.raw_data.shape if state.raw_data is not None else 'None'}")
                        print_to_log(f"🔧 DEBUG INPUT TO ENCODING: Target column: {state.target_column}")
                        print_to_log(f"🔧 DEBUG INPUT TO ENCODING: Interactive session: {state.interactive_session}")
                        
                        # Import encoding functions
                        from preprocessing_agent_impl import (
                            analyze_encoding_with_confidence,
                            get_llm_from_state,
                            SequentialState
                        )
                        
                        # Create a temporary file path for the DataFrame
                        import tempfile
                        import os
                        
                        # Use cleaned_data if available, otherwise raw_data
                        data_to_analyze = state.cleaned_data if state.cleaned_data is not None else state.raw_data
                        
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                            data_to_analyze.to_csv(tmp_file.name, index=False)
                            df_path = tmp_file.name
                        
                        try:
                            # Create SequentialState for encoding analysis
                            sequential_state = SequentialState(
                                df=data_to_analyze,
                                df_path=df_path,
                                target_column=state.target_column,
                                model_name=os.environ.get("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
                            )
                            
                            # Run encoding analysis with confidence-based approach
                            print_to_log("🔍 Running confidence-based encoding analysis...")
                            encoding_results = analyze_encoding_with_confidence(sequential_state)
                            
                            print_to_log(f"🔍 DEBUG: encoding_results type: {type(encoding_results)}")
                            print_to_log(f"🔍 DEBUG: encoding_results content: {encoding_results}")
                            
                            # Send results to Slack
                            slack_manager = getattr(state, '_slack_manager', None)
                            if not slack_manager:
                                from toolbox import slack_manager as global_slack_manager
                                slack_manager = global_slack_manager
                            
                            if slack_manager and state.chat_session:
                                # Handle new confidence-based format
                                if isinstance(encoding_results, dict) and 'encoding_columns' in encoding_results and 'llm_recommendations' in encoding_results:
                                    # New confidence-based format
                                    encoding_columns = encoding_results['encoding_columns']
                                    llm_recommendations = encoding_results['llm_recommendations']
                                    
                                    if encoding_columns and llm_recommendations:
                                        # Group columns by encoding strategy for concise display
                                        strategy_counts = {}
                                        for col, rec in llm_recommendations.items():
                                            strategy = rec.get('strategy', 'unknown')
                                            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                                        
                                        # Build concise strategy summary
                                        strategy_summary = []
                                        for strategy, count in strategy_counts.items():
                                            if strategy == 'label_encoding':
                                                strategy_summary.append(f"Label encoding: {count} columns")
                                            elif strategy == 'onehot_encoding':
                                                strategy_summary.append(f"One-hot encoding: {count} columns")
                                            elif strategy == 'target_encoding':
                                                strategy_summary.append(f"Target encoding: {count} columns")
                                            elif strategy == 'binary_encoding':
                                                strategy_summary.append(f"Binary encoding: {count} columns")
                                            else:
                                                strategy_summary.append(f"{strategy.replace('_', ' ').title()}: {count} columns")
                                        
                                        strategy_text = "\n".join(strategy_summary)
                                        
                                        # Confidence stats removed from display per user request
                                        
                                        message = f"""🔍 Encoding Analysis Complete!

📊 Categorical Columns Found: {len(encoding_columns)} columns

🔧 Recommended Strategies:
{strategy_text}

🔄 Ready for Next Step:
• `continue` - Apply encoding treatments and move to transformations
• `skip encoding` - Move directly to transformations phase
• `summary` - Show current preprocessing status"""
                                    else:
                                        message = f"""🔍 Encoding Analysis Complete!

📊 No categorical columns found - All columns are numeric!

🔄 Ready for Next Step:
• `continue` - Move to transformations phase
• `summary` - Show current preprocessing status"""
                                
                                elif isinstance(encoding_results, dict) and 'categorical_columns' in encoding_results:
                                    # Old format fallback
                                    categorical_columns = encoding_results['categorical_columns']
                                    llm_recommendations = encoding_results.get('llm_recommendations', {})
                                    
                                    # Group columns by encoding type for concise display
                                    encoding_groups = {}
                                    for col in categorical_columns:
                                        if col in llm_recommendations:
                                            raw_type = llm_recommendations[col].get('encoding_type') or llm_recommendations[col].get('strategy') or llm_recommendations[col].get('encoding') or 'unknown'
                                            enc_norm = str(raw_type).lower().replace('-', '_')
                                            if enc_norm in ['label_encoding', 'label']:
                                                key = 'Label'
                                            elif enc_norm in ['onehot_encoding', 'one_hot', 'onehot']:
                                                key = 'One-hot'
                                            elif enc_norm in ['ordinal_encoding', 'ordinal']:
                                                key = 'Ordinal'
                                            elif enc_norm in ['target_encoding', 'target']:
                                                key = 'Target'
                                            elif enc_norm in ['binary_encoding', 'binary']:
                                                key = 'Binary'
                                            elif enc_norm in ['drop_column', 'drop']:
                                                key = 'Drop column'
                                            else:
                                                key = enc_norm.title()
                                            if key not in encoding_groups:
                                                encoding_groups[key] = []
                                            encoding_groups[key].append(col)
                                    
                                    # Build concise encoding summary
                                    encoding_summary = []
                                    for encoding_type, cols in encoding_groups.items():
                                        encoding_summary.append(f"{encoding_type} encoding: {len(cols)} columns")
                                    
                                    encoding_text = "\n".join(encoding_summary)
                                    
                                    message = f"""🔍 Encoding Analysis Complete!

📊 Categorical Columns Found: {len(categorical_columns)} columns

🔧 Recommended Encoding:
{encoding_text}

🔄 Ready for Next Step:
• `continue` - Apply encoding treatments and move to transformations
• `skip encoding` - Move directly to transformations phase
• `summary` - Show current preprocessing status"""
                                else:
                                    # Fallback - avoid showing raw JSON
                                    message = f"""🔍 Encoding Analysis Complete!

📊 Analysis completed successfully

🔄 Ready for Next Step:
• `continue` - Apply encoding treatments and move to transformations
• `skip encoding` - Move directly to transformations phase
• `summary` - Show current preprocessing status"""
                                
                                slack_manager.send_message(state.chat_session, message)
                            
                            # Update state with encoding results
                            state.preprocessing_state.update({
                                "encoding_results": encoding_results,
                                "status": "encoding_analysis_complete"
                            })
                            
                            try:
                                os.unlink(df_path)
                            except:
                                pass
                            return state
                        except Exception as e:
                            print_to_log(f"❌ Encoding analysis failed: {e}")
                            import traceback
                            traceback.print_exc()
                            return state

                elif current_phase == 'transformations':
                    # Check if we already have transformation results
                    transformation_results = state.preprocessing_state.get('transformation_results')
                    if transformation_results:
                        # Apply transformation treatments and complete preprocessing
                        print_to_log("🔧 Applying transformation treatments...")
                        
                        df = state.cleaned_data.copy() if state.cleaned_data is not None else state.raw_data.copy()
                        strategy_counts = {}
                        
                        if isinstance(transformation_results, dict) and 'llm_recommendations' in transformation_results:
                            for col, recommendation in transformation_results['llm_recommendations'].items():
                                # DEBUG: Log the recommendation structure
                                print_to_log(f"🔍 DEBUG TRANSFORMATION: {col} -> {recommendation}")
                                
                                # Fix key mismatch: LLM uses 'strategy' key, not 'transformation'
                                raw_t = recommendation.get('strategy') or recommendation.get('transformation') or recommendation.get('transformation_type') or 'none'
                                t = str(raw_t).lower().replace('-', '_')
                                
                                print_to_log(f"🔍 DEBUG: {col} -> strategy='{raw_t}' -> applying...")
                                
                                # Use EXACT strategy names from LLM (no confusing alternatives)
                                if raw_t == 'log1p':
                                    df[col] = np.log1p(df[col])
                                    strategy_counts['Log1p'] = strategy_counts.get('Log1p', 0) + 1
                                    print_to_log(f"✅ Applied log1p to {col}")
                                    
                                elif raw_t == 'yeo_johnson':
                                    from sklearn.preprocessing import PowerTransformer
                                    pt = PowerTransformer(method='yeo-johnson')
                                    df[col] = pt.fit_transform(df[[col]])
                                    strategy_counts['Yeo-Johnson'] = strategy_counts.get('Yeo-Johnson', 0) + 1
                                    print_to_log(f"✅ Applied yeo_johnson to {col}")
                                    
                                elif raw_t == 'standardize':
                                    from sklearn.preprocessing import StandardScaler
                                    scaler = StandardScaler()
                                    df[col] = scaler.fit_transform(df[[col]])
                                    strategy_counts['Standardize'] = strategy_counts.get('Standardize', 0) + 1
                                    print_to_log(f"✅ Applied standardize to {col}")
                                    
                                elif raw_t == 'log':
                                    if df[col].min() > 0:
                                        df[col] = np.log(df[col])
                                    else:
                                        df[col] = np.log1p(df[col] - df[col].min() + 1)
                                    strategy_counts['Log'] = strategy_counts.get('Log', 0) + 1
                                    print_to_log(f"✅ Applied log to {col}")
                                    
                                elif raw_t == 'box_cox':
                                    from scipy.stats import boxcox
                                    if df[col].min() > 0:
                                        df[col], _ = boxcox(df[col])
                                    else:
                                        df[col] = np.log1p(df[col] - df[col].min() + 1)
                                    strategy_counts['Box-Cox'] = strategy_counts.get('Box-Cox', 0) + 1
                                    print_to_log(f"✅ Applied box_cox to {col}")
                                    
                                elif raw_t == 'sqrt':
                                    # Square root transformation
                                    if df[col].min() >= 0:
                                        df[col] = np.sqrt(df[col])
                                    else:
                                        df[col] = np.sqrt(df[col] - df[col].min())
                                    strategy_counts['Square root'] = strategy_counts.get('Square root', 0) + 1
                                    print_to_log(f"✅ Applied sqrt to {col}")
                                    
                                elif raw_t == 'robust_scale':
                                    from sklearn.preprocessing import RobustScaler
                                    scaler = RobustScaler()
                                    df[col] = scaler.fit_transform(df[[col]])
                                    strategy_counts['Robust scale'] = strategy_counts.get('Robust scale', 0) + 1
                                    print_to_log(f"✅ Applied robust_scale to {col}")
                                    
                                elif raw_t == 'quantile':
                                    from sklearn.preprocessing import QuantileTransformer
                                    qt = QuantileTransformer(output_distribution='uniform', random_state=42)
                                    df[col] = qt.fit_transform(df[[col]])
                                    strategy_counts['Quantile'] = strategy_counts.get('Quantile', 0) + 1
                                    print_to_log(f"✅ Applied quantile to {col}")
                                    
                                elif raw_t == 'square':
                                    # Square transformation (for negative skew)
                                    df[col] = df[col] ** 2
                                    strategy_counts['Square'] = strategy_counts.get('Square', 0) + 1
                                    print_to_log(f"✅ Applied square to {col}")
                                    
                                elif raw_t == 'none':
                                    strategy_counts['Kept as-is'] = strategy_counts.get('Kept as-is', 0) + 1
                                    print_to_log(f"✅ Kept {col} as-is (strategy: none)")
                                    
                                else:
                                    strategy_counts['Kept as-is'] = strategy_counts.get('Kept as-is', 0) + 1
                                    print_to_log(f"⚠️ Unknown strategy '{raw_t}' for {col}, kept as-is")
                        
                        # Create concise treatment summary
                        applied_treatments = []
                        for strategy, count in strategy_counts.items():
                            applied_treatments.append(f"{strategy}: {count} columns")

                        # Update state with processed data
                        state.cleaned_data = df
                        print_to_log(f"🔧 DEBUG: Set cleaned_data shape after transformations: {df.shape}")
                        print_to_log(f"🔧 DEBUG: Applied treatments count: {len(applied_treatments)}")
                        print_to_log(f"🔧 DEBUG: Applied treatments: {applied_treatments}")
                        
                        # Send confirmation message
                        slack_manager = getattr(state, '_slack_manager', None)
                        if not slack_manager:
                            from toolbox import slack_manager as global_slack_manager
                            slack_manager = global_slack_manager
                        
                        if slack_manager and state.chat_session:
                            treatments_text = "\n".join(applied_treatments) if applied_treatments else "• No treatments applied"
                            
                            message = f"""✅ Transformations Applied!

🔧 Applied Treatments:
{treatments_text}

📊 Data Summary:
• Final processed: {df.shape[0]:,} rows × {df.shape[1]} columns
• Transformations applied: {len(applied_treatments)} columns

🎉 Preprocessing Complete!

💬 Next Steps:
• `summary` - Show complete preprocessing summary
• `feature_selection` - Move to feature selection phase
• `model_building` - Move to model building phase"""
                            
                            # Store message for sending after CSV save
                            state.pending_slack_message = message
                        
                        # 🎯 SAVE TRANSFORMATION STRATEGIES TO SESSION STATE
                        state.save_preprocessing_strategy(
                            phase="transformations",
                            phase_results=transformation_results,
                            target_column=state.target_column
                        )
                        
                        # ✅ COMPLETE PREPROCESSING - Mark as completed and prompt for feature selection
                        from datetime import datetime
                        state.preprocessing_state.update({
                            "completed": True,  # ✅ Mark preprocessing as completed
                            "current_phase": "completion",
                            "status": "preprocessing_complete",
                            "timestamp": datetime.now().isoformat(),
                            "transformation_treatments_applied": applied_treatments
                        })
                        
                        if state.interactive_session:
                            state.interactive_session["current_phase"] = "completion"
                            state.interactive_session["phase"] = "complete"
                        
                        # Note: Completion message is already stored in pending_slack_message above (line 1624)
                        # and will be sent after CSV files are saved. No need to send duplicate message here.
                        
                        return state
                    else:
                        # Start transformations analysis
                        print_to_log("🔍 Starting transformations analysis...")
                        print_to_log(f"🔧 DEBUG INPUT TO TRANSFORMATIONS: Command='{command}', State data shape: {state.raw_data.shape if state.raw_data is not None else 'None'}")
                        print_to_log(f"🔧 DEBUG INPUT TO TRANSFORMATIONS: Target column: {state.target_column}")
                        print_to_log(f"🔧 DEBUG INPUT TO TRANSFORMATIONS: Interactive session: {state.interactive_session}")
                        
                        # Import transformations functions
                        from preprocessing_agent_impl import (
                            analyze_transformations_with_confidence,
                            get_llm_from_state,
                            SequentialState
                        )
                        
                        # Create a temporary file path for the DataFrame
                        import tempfile
                        import os
                        
                        # Use cleaned_data if available, otherwise raw_data
                        data_to_analyze = state.cleaned_data if state.cleaned_data is not None else state.raw_data
                        
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                            data_to_analyze.to_csv(tmp_file.name, index=False)
                            df_path = tmp_file.name
                        
                        try:
                            # Create SequentialState for transformations analysis
                            sequential_state = SequentialState(
                                df=data_to_analyze,
                                df_path=df_path,
                                target_column=state.target_column,
                                model_name=os.environ.get("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
                            )
                            
                            # Run transformations analysis with confidence-based approach
                            print_to_log("🔍 Running confidence-based transformations analysis...")
                            transformation_results = analyze_transformations_with_confidence(sequential_state)
                            
                            print_to_log(f"🔍 DEBUG: transformation_results type: {type(transformation_results)}")
                            print_to_log(f"🔍 DEBUG: transformation_results content: {transformation_results}")
                            
                            # Send results to Slack
                            slack_manager = getattr(state, '_slack_manager', None)
                            if not slack_manager:
                                from toolbox import slack_manager as global_slack_manager
                                slack_manager = global_slack_manager
                            
                            if slack_manager and state.chat_session:
                                # Handle new confidence-based format
                                if isinstance(transformation_results, dict) and 'transformations_columns' in transformation_results and 'llm_recommendations' in transformation_results:
                                    # New confidence-based format
                                    transformations_columns = transformation_results['transformations_columns']
                                    llm_recommendations = transformation_results['llm_recommendations']
                                    
                                    if transformations_columns and llm_recommendations:
                                        # Group columns by transformation strategy for concise display
                                        strategy_counts = {}
                                        for col, rec in llm_recommendations.items():
                                            strategy = rec.get('strategy', 'none')
                                            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                                        
                                        # Build concise strategy summary
                                        strategy_summary = []
                                        for strategy, count in strategy_counts.items():
                                            if strategy == 'none':
                                                strategy_summary.append(f"No transformation needed: {count} columns")
                                            elif strategy == 'log':
                                                strategy_summary.append(f"Log transformation: {count} columns")
                                            elif strategy == 'log1p':
                                                strategy_summary.append(f"Log1p transformation: {count} columns")
                                            elif strategy == 'sqrt':
                                                strategy_summary.append(f"Square root transformation: {count} columns")
                                            elif strategy == 'box_cox':
                                                strategy_summary.append(f"Box-Cox transformation: {count} columns")
                                            elif strategy == 'yeo_johnson':
                                                strategy_summary.append(f"Yeo-Johnson transformation: {count} columns")
                                            elif strategy == 'quantile':
                                                strategy_summary.append(f"Quantile transformation: {count} columns")
                                            else:
                                                strategy_summary.append(f"{strategy.replace('_', ' ').title()} transformation: {count} columns")
                                        
                                        strategy_text = "\n".join(strategy_summary)
                                        
                                        # Confidence stats removed from display per user request
                                        
                                        message = f"""🔍 Transformations Analysis Complete!

📊 Numerical Columns Analyzed: {len(transformations_columns)} columns

🔧 Recommended Strategies:
{strategy_text}

🔄 Ready for Next Step:
• `continue` - Apply transformations and complete preprocessing
• `skip transformations` - Complete preprocessing without transformations
• `summary` - Show current preprocessing status"""
                                    else:
                                        message = f"""🔍 Transformations Analysis Complete!

📊 No transformations needed - All columns are well-distributed!

🔄 Ready for Next Step:
• `continue` - Complete preprocessing
• `summary` - Show current preprocessing status"""
                                
                                elif isinstance(transformation_results, dict) and 'transformation_columns' in transformation_results:
                                    # Old format fallback
                                    numerical_columns = transformation_results['transformation_columns']
                                    llm_recommendations = transformation_results.get('llm_recommendations', {})
                                    
                                    # Group columns by transformation type for concise display
                                    transformation_groups = {}
                                    for col in numerical_columns:
                                        if col in llm_recommendations:
                                            # Fix key mismatch: LLM uses 'strategy' key
                                            raw_t = llm_recommendations[col].get('strategy') or llm_recommendations[col].get('transformation') or llm_recommendations[col].get('transformation_type') or 'none'
                                            print_to_log(f'🔍 DEBUG GROUPING: {col} -> raw_t={raw_t}')
                                            t_norm = str(raw_t).lower().replace('-', '_')
                                            # Friendly label mapping
                                            if t_norm in ['log']:
                                                key = 'Log'
                                            elif t_norm in ['log1p']:
                                                key = 'Log1p'
                                            elif t_norm == 'sqrt':
                                                key = 'Square root'
                                            elif t_norm == 'square':
                                                key = 'Square'
                                            elif t_norm in ['box_cox', 'boxcox']:
                                                key = 'Box-Cox'
                                            elif t_norm in ['yeo_johnson', 'yeojohnson']:
                                                key = 'Yeo-Johnson'
                                            elif t_norm in ['standardize', 'standard_scaler', 'zscore']:
                                                key = 'Standardize'
                                            elif t_norm in ['robust_scale', 'robust_scaler']:
                                                key = 'Robust scale'
                                            elif t_norm in ['quantile', 'quantile_transform']:
                                                key = 'Quantile transform'
                                            elif t_norm in ['normalize', 'minmax', 'minmax_scaler']:
                                                key = 'MinMax normalize'
                                            else:
                                                key = 'No transformation needed'
                                            if key not in transformation_groups:
                                                transformation_groups[key] = []
                                            transformation_groups[key].append(col)
                                    
                                    # Build concise transformation summary
                                    transformation_summary = []
                                    for transformation_type, cols in transformation_groups.items():
                                        transformation_summary.append(f"{transformation_type}: {len(cols)} columns")
                                    
                                    transformation_text = "\n".join(transformation_summary)
                                    
                                    message = f"""🔍 Transformations Analysis Complete!

📊 Numerical Columns Analyzed: {len(numerical_columns)} columns

🔧 Recommended Transformations:
{transformation_text}

🔄 Ready for Next Step:
• `continue` - Apply transformations and complete preprocessing
• `skip transformations` - Complete preprocessing without transformations
• `summary` - Show current preprocessing status"""
                                else:
                                    # Fallback - avoid showing raw JSON
                                    message = f"""🔍 Transformations Analysis Complete!

📊 Analysis completed successfully

🔄 Ready for Next Step:
• `continue` - Apply transformations and complete preprocessing
• `skip transformations` - Complete preprocessing without transformations
• `summary` - Show current preprocessing status"""
                                
                                slack_manager.send_message(state.chat_session, message)
                            
                            # Update state with transformation results
                            state.preprocessing_state.update({
                                "transformation_results": transformation_results,
                                "status": "transformation_analysis_complete"
                            })
                            
                            try:
                                os.unlink(df_path)
                            except:
                                pass
                            return state
                        except Exception as e:
                            print_to_log(f"❌ Transformations analysis failed: {e}")
                            import traceback
                            traceback.print_exc()
                            return state

                else:
                    print_to_log(f"❌ Unknown phase for continue command: {current_phase}")
                    return state
            
            # Handle BGE-classified queries with clear intent signal (Level 4 BGE result)
            elif command.startswith('QUERY: '):
                # Extract the actual query from the intent signal
                actual_query = command[7:]  # Remove 'QUERY: ' prefix
                print_to_log("🔍 Processing BGE-classified query with enhanced LLM...")
                print_to_log(f"🔍 DEBUG: BGE classified query: '{actual_query}'")
                
                try:
                    # Initialize LLM using the same pattern as preprocessing strategies
                    from preprocessing_agent_impl import get_llm_from_state, SequentialState
                    import tempfile
                    import os
                    
                    print_to_log("🔍 DEBUG: Importing required modules...")
                    
                    # Use the data from state for analysis context
                    data_to_analyze = state.cleaned_data if hasattr(state, 'cleaned_data') and state.cleaned_data is not None else state.raw_data
                    print_to_log(f"🔍 DEBUG: Using data for context - shape: {data_to_analyze.shape}")
                    
                    # Create temporary file for LLM processing
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                        data_to_analyze.to_csv(tmp_file.name, index=False)
                        df_path = tmp_file.name
                        print_to_log(f"🔍 DEBUG: Created temp file: {df_path}")
                    
                    try:
                        # Create SequentialState for LLM processing
                        print_to_log("🔍 DEBUG: Creating SequentialState for LLM...")
                        sequential_state = SequentialState(
                            df=data_to_analyze,
                            df_path=df_path,
                            target_column=state.target_column,
                            model_name=os.environ.get("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
                        )
                        
                        # Initialize LLM
                        print_to_log("🔍 DEBUG: Initializing LLM...")
                        llm = get_llm_from_state(sequential_state)
                        print_to_log(f"🔍 DEBUG: LLM initialized successfully")
                        
                        # Analyze query context and generate response
                        print_to_log("🔍 DEBUG: Analyzing query context...")
                        query_analysis = self._analyze_query_context(actual_query, state)
                        query_type = query_analysis['type']
                        context_data = query_analysis['context']
                        
                        print_to_log(f"🔍 DEBUG: Query analysis - Type: {query_type}, Context length: {len(str(context_data))}")
                        
                        # Generate appropriate prompt based on query type
                        if query_type == 'general':
                            print_to_log("🔍 DEBUG: Creating general query prompt...")
                            prompt = f"""You are a data preprocessing expert. Answer the user's general question about data preprocessing concepts and methods.

QUESTION: "{actual_query}"

Provide a clear, informative explanation about the preprocessing concept or method being asked about.
"""
                        elif query_type == 'column_specific':
                            column = query_analysis.get('column', 'unknown')
                            print_to_log(f"🔍 DEBUG: Creating column-specific query prompt for column: {column}")
                            prompt = f"""You are a data preprocessing expert. Answer the user's question about a specific column and its preprocessing strategy.

QUESTION: "{actual_query}"
TARGET COLUMN: {state.target_column}
COLUMN OF INTEREST: {column}

COLUMN ANALYSIS AND RECOMMENDATIONS:
{context_data}

Explain the preprocessing strategy for this column based on the analysis data and reasoning provided.
"""
                        elif query_type == 'comparative':
                            print_to_log("🔍 DEBUG: Creating comparative query prompt...")
                            prompt = f"""You are a data preprocessing expert. Answer the user's comparative question about multiple columns or strategies.

QUESTION: "{actual_query}"
TARGET COLUMN: {state.target_column}

FULL DATASET ANALYSIS:
{context_data}

Compare and explain the different strategies, columns, or preprocessing approaches based on the analysis data provided.
"""
                        else:  # phase_specific
                            current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                            print_to_log(f"🔍 DEBUG: Creating phase-specific query prompt for phase: {current_phase}")
                            prompt = f"""You are a data preprocessing expert. Answer the user's question about the current preprocessing phase.

QUESTION: "{actual_query}"
CURRENT PHASE: {current_phase}
TARGET COLUMN: {state.target_column}

PHASE ANALYSIS:
{context_data}

Explain the current preprocessing phase, strategies, and recommendations based on the analysis data provided.
"""
                        
                        print_to_log(f"🔍 DEBUG: Prompt created - length: {len(prompt)} characters")
                        
                        # Get LLM response
                        print_to_log("🤖 DEBUG: Sending prompt to LLM...")
                        from langchain_core.messages import HumanMessage
                        response = llm.invoke([HumanMessage(content=prompt)]).content
                        
                        print_to_log(f"🤖 DEBUG: LLM response received - length: {len(response)} characters")
                        print_to_log(f"🤖 DEBUG: Response preview: {response[:100]}...")
                        
                        # Clean up temp file
                        try:
                            os.unlink(df_path)
                            print_to_log(f"🔍 DEBUG: Cleaned up temp file: {df_path}")
                        except Exception as cleanup_error:
                            print_to_log(f"⚠️ DEBUG: Failed to clean up temp file: {cleanup_error}")
                        
                        # Send response to Slack
                        print_to_log("📤 DEBUG: Preparing Slack response...")
                        slack_manager = getattr(state, '_slack_manager', None)
                        if not slack_manager:
                            print_to_log("📤 DEBUG: No slack_manager in state, using global")
                            from toolbox import slack_manager as global_slack_manager
                            slack_manager = global_slack_manager
                        
                        if slack_manager and state.chat_session:
                            print_to_log(f"📤 DEBUG: Sending message to Slack session: {state.chat_session}")
                            message = f"""🤖 Query Response:

{response}

💬 Continue with preprocessing:
• `continue` - Continue with current phase
• `summary` - Show current status  
• `help` - Get more assistance"""
                            
                            slack_manager.send_message(state.chat_session, message)
                            print_to_log("📤 DEBUG: Slack message sent successfully")
                        else:
                            print_to_log("⚠️ DEBUG: No Slack session available - message not sent")
                        
                        print_to_log("✅ DEBUG: BGE-classified query processing completed successfully")
                        return state
                        
                    except Exception as e:
                        print_to_log(f"❌ DEBUG: BGE query processing failed with error: {e}")
                        # Clean up temp file on error
                        try:
                            os.unlink(df_path)
                        except:
                            pass
                        
                        # Fallback to basic response
                        slack_manager = getattr(state, '_slack_manager', None)
                        if not slack_manager:
                            from toolbox import slack_manager as global_slack_manager
                            slack_manager = global_slack_manager
                        
                        if slack_manager and state.chat_session:
                            fallback_message = f"""🤖 Query Response:

I understand you're asking: "{actual_query}"

I'm having trouble accessing detailed analysis data right now, but I can help with general preprocessing questions.

💬 Continue with preprocessing:
• `continue` - Continue with current phase
• `summary` - Show current status"""
                            
                            slack_manager.send_message(state.chat_session, fallback_message)
                        
                        return state
                        
                except Exception as e:
                    print_to_log(f"❌ DEBUG: Complete BGE query processing failed: {e}")
                    return state
            
            # Handle other BGE-classified intents with clear intent signals (Level 4 BGE results)
            elif command.startswith('PROCEED: '):
                # Extract the actual command from the intent signal
                actual_command = command[9:]  # Remove 'PROCEED: ' prefix
                print_to_log("🚀 Processing BGE-classified PROCEED command...")
                print_to_log(f"🚀 DEBUG: BGE classified proceed: '{actual_command}'")
                # Route to continue handler (proceed maps to continue)
                return self.handle_interactive_command(state, actual_command)
            
            elif command.startswith('SKIP: '):
                # Extract the actual command from the intent signal  
                actual_command = command[6:]  # Remove 'SKIP: ' prefix
                print_to_log("⏭️ Processing BGE-classified SKIP command...")
                print_to_log(f"⏭️ DEBUG: BGE classified skip: '{actual_command}'")
                # Route to skip handler
                return self.handle_interactive_command(state, actual_command)
            
            elif command.startswith('OVERRIDE: '):
                # Extract the actual query from the intent signal
                actual_query = command[10:]  # Remove 'OVERRIDE: ' prefix
                print_to_log("🔧 Processing BGE-classified OVERRIDE command...")
                print_to_log(f"🔧 DEBUG: BGE classified override: '{actual_query}'")
                # Route to override handler with 'override ' prefix to match existing logic
                return self.handle_interactive_command(state, f"override {actual_query}")
            
            elif command.startswith('SUMMARY: '):
                # Extract the actual command from the intent signal
                actual_command = command[9:]  # Remove 'SUMMARY: ' prefix
                print_to_log("📊 Processing BGE-classified SUMMARY command...")
                print_to_log(f"📊 DEBUG: BGE classified summary: '{actual_command}'")
                # Route to summary handler
                return self.handle_interactive_command(state, actual_command)
            
            elif command.lower() in ['query', 'question', 'help', 'what', 'how', 'why', 'explain']:
                # Enhanced query handling with intelligent context passing
                print_to_log("🔍 Processing user query with enhanced LLM...")
                print_to_log(f"🔍 DEBUG: Raw query command: '{command}'")
                
                try:
                    # Initialize LLM using the same pattern as preprocessing strategies
                    from preprocessing_agent_impl import get_llm_from_state, SequentialState
                    import tempfile
                    import os
                    import re
                    
                    print_to_log("🔍 DEBUG: Importing required modules for query processing")
                    
                    # Create SequentialState for LLM initialization
                    data_to_analyze = state.cleaned_data if state.cleaned_data is not None else state.raw_data
                    print_to_log(f"🔍 DEBUG: Using {'cleaned' if state.cleaned_data is not None else 'raw'} data for analysis")
                    print_to_log(f"🔍 DEBUG: Data shape: {data_to_analyze.shape}")
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                        data_to_analyze.to_csv(tmp_file.name, index=False)
                        df_path = tmp_file.name
                    
                    print_to_log(f"🔍 DEBUG: Created temp file: {df_path}")
                    
                    sequential_state = SequentialState(
                        df=data_to_analyze,
                        df_path=df_path,
                        target_column=state.target_column,
                        model_name=os.environ.get("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M"),
                        current_phase=state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                    )
                    
                    print_to_log(f"🔍 DEBUG: Created SequentialState - target: {state.target_column}, model: {sequential_state.model_name}")
                    
                    # Initialize LLM using the same pattern as strategy generation
                    print_to_log("🔍 DEBUG: Initializing LLM...")
                    llm = get_llm_from_state(sequential_state)
                    print_to_log(f"🔍 DEBUG: LLM initialized successfully: {type(llm).__name__}")
                    
                    # Classify query type and determine context
                    print_to_log("🔍 DEBUG: Starting query context analysis...")
                    query_analysis = self._analyze_query_context(command, state)
                    query_type = query_analysis['type']
                    context_data = query_analysis['context']
                    
                    print_to_log(f"🔍 DEBUG: Query analysis complete:")
                    print_to_log(f"   - Query type: {query_type}")
                    print_to_log(f"   - Context level: {query_analysis['context_level']}")
                    print_to_log(f"   - Mentioned column: {query_analysis.get('column', 'None')}")
                    print_to_log(f"   - Context data length: {len(str(context_data)) if context_data else 0} characters")
                    
                    # Generate LLM response based on query type and context
                    print_to_log(f"🔍 DEBUG: Generating prompt for query type: {query_type}")
                    
                    if query_type == 'general':
                        # General methodology questions - no data context needed
                        prompt = f"""You are a data preprocessing expert. Answer this question clearly and concisely:

QUESTION: "{command}"

Provide a clear, educational explanation about preprocessing concepts, methods, and best practices. Focus on practical understanding.
"""
                        print_to_log("🔍 DEBUG: Using general methodology prompt (no context)")
                    
                    elif query_type == 'column_specific':
                        # Questions about specific columns and their strategies
                        column_name = query_analysis.get('column')
                        print_to_log(f"🔍 DEBUG: Column-specific query for column: {column_name}")
                        prompt = f"""You are a data preprocessing expert analyzing a specific column. Answer the user's question using the provided analysis data.

QUESTION: "{command}"
TARGET COLUMN: {state.target_column}
COLUMN OF INTEREST: {column_name}

COLUMN ANALYSIS:
{context_data}

Provide a specific explanation about this column's preprocessing strategy, including why it was recommended based on the data characteristics shown above.
"""
                        print_to_log(f"🔍 DEBUG: Column-specific prompt created with context for '{column_name}'")
                    
                    elif query_type == 'comparative':
                        # Questions about multiple columns or comparisons
                        print_to_log("🔍 DEBUG: Comparative query - using full analysis context")
                        prompt = f"""You are a data preprocessing expert. Answer the user's question using the complete analysis data provided.

QUESTION: "{command}"
TARGET COLUMN: {state.target_column}

COMPLETE ANALYSIS:
{context_data}

Analyze the data and provide a comprehensive answer comparing columns, strategies, or identifying patterns as requested.
"""
                        print_to_log("🔍 DEBUG: Comparative prompt created with full context")
                    
                    else:  # phase_specific
                        # Questions about current phase strategies
                        current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                        print_to_log(f"🔍 DEBUG: Phase-specific query for phase: {current_phase}")
                        prompt = f"""You are a data preprocessing expert. Answer the user's question about the current preprocessing phase.

QUESTION: "{command}"
CURRENT PHASE: {current_phase}
TARGET COLUMN: {state.target_column}

PHASE ANALYSIS:
{context_data}

Explain the current preprocessing phase, strategies, and recommendations based on the analysis data provided.
"""
                        print_to_log(f"🔍 DEBUG: Phase-specific prompt created for '{current_phase}'")
                    
                    # Get LLM response
                    print_to_log("🤖 DEBUG: Sending prompt to LLM...")
                    print_to_log(f"🤖 DEBUG: Prompt length: {len(prompt)} characters")
                    
                    from langchain_core.messages import HumanMessage
                    response = llm.invoke([HumanMessage(content=prompt)]).content
                    
                    print_to_log(f"🤖 DEBUG: LLM response received - length: {len(response)} characters")
                    print_to_log(f"🤖 DEBUG: Response preview: {response[:100]}...")
                    
                    # Clean up temp file
                    try:
                        os.unlink(df_path)
                        print_to_log(f"🔍 DEBUG: Cleaned up temp file: {df_path}")
                    except Exception as cleanup_error:
                        print_to_log(f"⚠️ DEBUG: Failed to clean up temp file: {cleanup_error}")
                    
                    # Send response to Slack
                    print_to_log("📤 DEBUG: Preparing Slack response...")
                    slack_manager = getattr(state, '_slack_manager', None)
                    if not slack_manager:
                        print_to_log("📤 DEBUG: No slack_manager in state, using global")
                        from toolbox import slack_manager as global_slack_manager
                        slack_manager = global_slack_manager
                    
                    if slack_manager and state.chat_session:
                        print_to_log(f"📤 DEBUG: Sending message to Slack session: {state.chat_session}")
                        message = f"""🤖 Query Response:

{response}

💬 Continue with preprocessing:
• `continue` - Continue with current phase
• `summary` - Show current status  
• `help` - Get more assistance"""
                        
                        slack_manager.send_message(state.chat_session, message)
                        print_to_log("📤 DEBUG: Slack message sent successfully")
                    else:
                        print_to_log("⚠️ DEBUG: No Slack session available - message not sent")
                    
                    print_to_log("✅ DEBUG: Enhanced query processing completed successfully")
                    return state
                    
                except Exception as e:
                    print_to_log(f"❌ DEBUG: Enhanced query processing failed with error: {e}")
                    print_to_log(f"❌ DEBUG: Error type: {type(e).__name__}")
                    import traceback
                    print_to_log(f"❌ DEBUG: Full traceback:")
                    traceback.print_exc()
                    
                    # Fallback to basic response
                    print_to_log("🔄 DEBUG: Attempting fallback response...")
                    try:
                        slack_manager = getattr(state, '_slack_manager', None)
                        if not slack_manager:
                            print_to_log("🔄 DEBUG: Using global slack manager for fallback")
                            from toolbox import slack_manager as global_slack_manager
                            slack_manager = global_slack_manager
                        
                        if slack_manager and state.chat_session:
                            print_to_log("🔄 DEBUG: Sending fallback message to Slack")
                            fallback_message = f"""🤖 Query Response:

I understand you're asking: "{command}"

I'm having trouble accessing detailed analysis data right now, but I can help with general preprocessing questions. Here are some common topics:

🔧 Preprocessing Methods:
• Outliers: Winsorize (clip extreme values) vs Keep (leave as-is)
• Missing Values: Mean/Median imputation vs Model-based vs Drop
• Encoding: One-hot vs Label vs Target encoding for categories  
• Transformations: Log/Square root for skewed data, Scaling for normalization

💬 Try asking:
• `"explain median imputation"`
• `"what is winsorization"`  
• `"why use one-hot encoding"`
• `summary` - Show current preprocessing status

💬 Continue with preprocessing:
• `continue` - Continue with current phase
• `summary` - Show current status"""
                            
                            slack_manager.send_message(state.chat_session, fallback_message)
                            print_to_log("🔄 DEBUG: Fallback message sent successfully")
                        else:
                            print_to_log("⚠️ DEBUG: No Slack session for fallback message")
                    
                    except Exception as fallback_error:
                        print_to_log(f"❌ DEBUG: Fallback response also failed: {fallback_error}")
                        print_to_log(f"❌ DEBUG: Fallback error type: {type(fallback_error).__name__}")
                    
                    return state

            elif command.lower() in ['override', 'change', 'modify', 'custom'] or any(override_word in command.lower() for override_word in ['modify', 'change', 'use', 'apply', 'do', 'keep', 'dont', 'dont transform', 'leave unchanged']):
                # Handle user overrides using the existing process_user_input_with_llm function
                print_to_log("🔧 Processing user override request...")
                
                # Lightweight parser to capture overrides like:
                # - "use median for age, income"
                # - "city: onehot; subscription_type: ordinal"
                # - "apply winsorize to account_balance"
                # - "dont transform income" / "keep age as is"
                import re
                def parse_override_list(text: str, phase: str, available_cols: list):
                    text_l = text.lower()
                    overrides = {}
                    # Pattern A: key:value pairs separated by , or ;
                    pairs = re.findall(r"([a-zA-Z0-9_\- ]+)\s*:\s*([a-zA-Z0-9_\-]+)", text_l)
                    for col_raw, strat_raw in pairs:
                        col = col_raw.strip().replace(' ', '_')
                        # Find real column by case-insensitive match
                        match = next((c for c in available_cols if c.lower()==col or c.lower().replace(' ','_')==col), None)
                        if not match:
                            continue
                        strat = strat_raw.strip().replace('-', '_')
                        overrides.setdefault(match, strat)
                    
                    # Pattern B: "use <strategy> for/on <col[, col2]>"
                    m = re.search(r"\b(use|apply|do)\s+([a-zA-Z0-9_\-]+)(?:\s+(?:imputation|encoding|transformation))?\s+(?:for|on|to)\s+(.+)", text_l)
                    if m:
                        strat = m.group(2).strip().replace('-', '_')
                        cols_part = m.group(3)
                        cols = re.split(r"[,;]|\band\b", cols_part)
                        for c in cols:
                            col_key = c.strip().strip('.')
                            if not col_key:
                                continue
                            match = next((c2 for c2 in available_cols if c2.lower()==col_key or c2.lower().replace(' ','_')==col_key), None)
                            if match:
                                overrides.setdefault(match, strat)
                    
                    # Pattern C: "dont transform <col>" / "keep <col> as is"
                    m2 = re.findall(r"(?:dont\s+transform|no\s+transform|keep\s+)([a-zA-Z0-9_\- ]+)", text_l)
                    for col_raw in m2:
                        match = next((c for c in available_cols if c.lower()==col_raw.strip() or c.lower().replace(' ','_')==col_raw.strip()), None)
                        if match:
                            if phase == 'transformations':
                                overrides.setdefault(match, 'none')
                            elif phase == 'outliers':
                                overrides.setdefault(match, 'keep')
                    return overrides
                
                # Persist overrides into state.user_overrides by phase
                current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                data_cols = list((state.cleaned_data if state.cleaned_data is not None else state.raw_data).columns)
                parsed = parse_override_list(command, current_phase, data_cols)
                if parsed:
                    print_to_log(f"🔧 Parsed overrides for phase {current_phase}: {parsed}")
                    if not hasattr(state, 'user_overrides') or state.user_overrides is None:
                        state.user_overrides = {}
                    if current_phase not in state.user_overrides or state.user_overrides[current_phase] is None:
                        state.user_overrides[current_phase] = {}
                    # Map phase-specific keys
                    for col, strat in parsed.items():
                        if current_phase == 'encoding':
                            state.user_overrides[current_phase][col] = { 'encoding_type': strat }
                        elif current_phase == 'missing_values':
                            # support constant=value
                            if strat.startswith('constant='):
                                try:
                                    val = strat.split('=',1)[1]
                                    # try numeric
                                    const_val = float(val) if re.match(r"^\d+(\.\d+)?$", val) else val
                                    state.user_overrides[current_phase][col] = { 'strategy': 'constant', 'constant_value': const_val }
                                except Exception:
                                    state.user_overrides[current_phase][col] = { 'strategy': 'constant', 'constant_value': -222 }
                            else:
                                state.user_overrides[current_phase][col] = { 'strategy': strat }
                        elif current_phase == 'outliers':
                            state.user_overrides[current_phase][col] = { 'treatment': strat }
                        elif current_phase == 'transformations':
                            state.user_overrides[current_phase][col] = { 'transformation_type': strat }
                
                # Import LLM helper for natural-language confirmations (existing behavior)
                from preprocessing_agent_impl import process_user_input_with_llm, SequentialState
                
                # Create a temporary file path for the DataFrame
                import tempfile
                import os
                
                # Use cleaned_data if available, otherwise raw_data
                data_to_analyze = state.cleaned_data if state.cleaned_data is not None else state.raw_data
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                    data_to_analyze.to_csv(tmp_file.name, index=False)
                    df_path = tmp_file.name
                
                try:
                    # Create SequentialState for override processing
                    sequential_state = SequentialState(
                        df=data_to_analyze,
                        df_path=df_path,
                        target_column=state.target_column,
                        model_name=os.environ.get("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M"),
                        current_phase=current_phase
                    )
                    
                    # Process the user input for natural response
                    updated_sequential_state = process_user_input_with_llm(sequential_state, command)
                    
                    # Build confirmation message
                    override_response = updated_sequential_state.query_response or "Overrides captured."
                    if parsed:
                        lines = []
                        for col, v in parsed.items():
                            lines.append(f"• {col}: {v}")
                        captured = "\n".join(lines)
                        override_response += f"\n\nCaptured Overrides:\n{captured}"
                    
                    # Send response to Slack
                    slack_manager = getattr(state, '_slack_manager', None)
                    if not slack_manager:
                        from toolbox import slack_manager as global_slack_manager
                        slack_manager = global_slack_manager
                    
                    if slack_manager and state.chat_session:
                        message = f"""🔧 Override Request:

{override_response}

💬 Continue with preprocessing:
• `continue` - Apply current phase with overrides
• `summary` - Show current status  
• `help` - Get more assistance"""
                        
                        slack_manager.send_message(state.chat_session, message)
                    
                    try:
                        os.unlink(df_path)
                    except:
                        pass
                    
                    return state
                    
                except Exception as e:
                    print_to_log(f"❌ Override processing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return state

            elif command.lower() in ['skip', 'next', 'advance'] or any(skip_cmd in command.lower() for skip_cmd in ['skip outliers', 'skip missing', 'skip encoding', 'skip transformations']):
                # Handle skip commands to move to next phase
                print_to_log("⏭️ Skipping current phase...")
                
                current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                
                # Define phase progression
                phase_progression = {
                    'overview': 'outliers',
                    'outliers': 'missing_values',
                    'missing_values': 'encoding',
                    'encoding': 'transformations',
                    'transformations': 'completion'
                }
                
                # Handle specific skip commands
                if 'skip outliers' in command.lower():
                    next_phase = 'missing_values'
                elif 'skip missing' in command.lower():
                    next_phase = 'encoding'
                elif 'skip encoding' in command.lower():
                    next_phase = 'transformations'
                elif 'skip transformations' in command.lower():
                    next_phase = 'completion'
                else:
                    # Default progression
                    next_phase = phase_progression.get(current_phase, 'completion')
                
                print_to_log(f"🔧 DEBUG: Skipping from {current_phase} to {next_phase}")
                
                # ✅ SPECIAL HANDLING FOR SKIP TO COMPLETION
                if next_phase == 'completion':
                    from datetime import datetime
                    # Mark preprocessing as completed when skipping to completion
                    state.preprocessing_state.update({
                        "completed": True,  # ✅ Mark preprocessing as completed
                        "current_phase": next_phase,
                        "status": "preprocessing_complete",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    if state.interactive_session:
                        state.interactive_session["current_phase"] = next_phase
                        state.interactive_session["phase"] = "complete"
                else:
                    # Regular phase skip
                    state.preprocessing_state.update({
                        "current_phase": next_phase,
                        "status": "phase_skipped"
                    })
                    
                    if state.interactive_session:
                        state.interactive_session["current_phase"] = next_phase
                        state.interactive_session["phase"] = "ready"
                
                # Send confirmation message
                slack_manager = getattr(state, '_slack_manager', None)
                if not slack_manager:
                    from toolbox import slack_manager as global_slack_manager
                    slack_manager = global_slack_manager
                
                if slack_manager and state.chat_session:
                    if next_phase == 'completion':
                        # ✅ COMPLETION MESSAGE WITH FEATURE SELECTION PROMPT
                        if state.cleaned_data is not None:
                            final_data_shape = state.cleaned_data.shape
                        elif state.raw_data is not None:
                            final_data_shape = state.raw_data.shape
                        else:
                            final_data_shape = (0, 0)
                        message = f"""🎉 Preprocessing Complete!

✅ Skipped transformations - preprocessing finished!
• Final dataset: {final_data_shape[0]:,} rows × {final_data_shape[1]} columns
• Data is ready for machine learning

🚀 Would you like to move to feature selection?
• `yes` - Start feature selection with cleaned data
• `no` - Stay in preprocessing for summary/export
• `summary` - Show complete preprocessing summary"""
                    else:
                        # Regular skip message
                        message = f"""⏭️ Phase Skipped!

🔄 Moved from {current_phase} to {next_phase}

💬 Next Steps:
• `continue` - Start {next_phase} analysis
• `summary` - Show current status
• `help` - Get assistance"""
                    
                    slack_manager.send_message(state.chat_session, message)
                
                return state

            elif command.lower() in ['summary', 'status', 'progress']:
                # Show current preprocessing status
                print_to_log("📊 Generating preprocessing summary...")
                
                current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                status = state.preprocessing_state.get('status', 'unknown')
                
                # Build summary message
                summary_parts = []
                summary_parts.append(f"📊 Preprocessing Status:")
                summary_parts.append(f"• Current Phase: {current_phase}")
                summary_parts.append(f"• Status: {status}")
                
                if state.cleaned_data is not None:
                    summary_parts.append(f"• Data Shape: {state.cleaned_data.shape[0]:,} rows × {state.cleaned_data.shape[1]} columns")
                elif state.raw_data is not None:
                    summary_parts.append(f"• Data Shape: {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns")
                else:
                    summary_parts.append(f"• Data Shape: No data available")
                
                # Add phase-specific information
                if current_phase == 'outliers' and state.preprocessing_state.get('outlier_results'):
                    outlier_results = state.preprocessing_state['outlier_results']
                    if isinstance(outlier_results, dict) and 'outlier_columns' in outlier_results:
                        outlier_count = len(outlier_results['outlier_columns'])
                        summary_parts.append(f"• Outliers Found: {outlier_count} columns")
                
                elif current_phase == 'missing_values' and state.preprocessing_state.get('missing_results'):
                    missing_results = state.preprocessing_state['missing_results']
                    if isinstance(missing_results, dict) and 'missing_columns' in missing_results:
                        missing_count = len(missing_results['missing_columns'])
                        summary_parts.append(f"• Missing Values: {missing_count} columns")
                
                elif current_phase == 'encoding' and state.preprocessing_state.get('encoding_results'):
                    encoding_results = state.preprocessing_state['encoding_results']
                    if isinstance(encoding_results, dict) and 'categorical_columns' in encoding_results:
                        categorical_count = len(encoding_results['categorical_columns'])
                        summary_parts.append(f"• Categorical Columns: {categorical_count} columns")
                
                elif current_phase == 'transformations' and state.preprocessing_state.get('transformation_results'):
                    transformation_results = state.preprocessing_state['transformation_results']
                    if isinstance(transformation_results, dict) and 'numerical_columns' in transformation_results:
                        numerical_count = len(transformation_results['numerical_columns'])
                        summary_parts.append(f"• Numerical Columns: {numerical_count} columns")
                
                summary_parts.append(f"\n💬 Available Commands:")
                summary_parts.append(f"• `continue` - Continue with current phase")
                summary_parts.append(f"• `skip` - Skip to next phase")
                summary_parts.append(f"• `help` - Get assistance")
                summary_parts.append(f"• `query` - Ask questions about preprocessing")
                
                summary_message = "\n".join(summary_parts)
                
                # Send summary to Slack
                slack_manager = getattr(state, '_slack_manager', None)
                if not slack_manager:
                    from toolbox import slack_manager as global_slack_manager
                    slack_manager = global_slack_manager
                
                if slack_manager and state.chat_session:
                    slack_manager.send_message(state.chat_session, summary_message)
                
                return state

            else:
                print_to_log(f"❌ Unknown interactive command: {command}")
                return state
        
        except Exception as e:
            print_to_log(f"❌ Interactive command handling failed: {e}")
            import traceback
            traceback.print_exc()
            return state

    def _analyze_query_context(self, query: str, state: PipelineState) -> dict:
        """Analyze query to determine what context to provide to LLM"""
        print_to_log(f"🔍 DEBUG: [_analyze_query_context] Starting analysis for query: '{query}'")
        
        query_lower = query.lower()
        print_to_log(f"🔍 DEBUG: [_analyze_query_context] Normalized query: '{query_lower}'")
        
        # Extract column names mentioned in query
        data_cols = list((state.cleaned_data if state.cleaned_data is not None else state.raw_data).columns)
        mentioned_columns = [col for col in data_cols if col.lower() in query_lower]
        print_to_log(f"🔍 DEBUG: [_analyze_query_context] Available columns: {data_cols}")
        print_to_log(f"🔍 DEBUG: [_analyze_query_context] Mentioned columns: {mentioned_columns}")
        
        # Determine query type
        general_keywords = ['what is', 'explain', 'how does', 'what are', 'define', 'meaning of', 'concept']
        column_keywords = ['this column', 'for this', 'why median', 'why mean', 'why winsorize', 'strategy for']
        comparative_keywords = ['which column', 'what columns', 'how many', 'compare', 'maximum', 'minimum', 'most', 'least']
        
        print_to_log(f"🔍 DEBUG: [_analyze_query_context] Checking keyword matches:")
        general_matches = [kw for kw in general_keywords if kw in query_lower]
        column_matches = [kw for kw in column_keywords if kw in query_lower]
        comparative_matches = [kw for kw in comparative_keywords if kw in query_lower]
        print_to_log(f"   - General matches: {general_matches}")
        print_to_log(f"   - Column matches: {column_matches}")
        print_to_log(f"   - Comparative matches: {comparative_matches}")
        
        if any(keyword in query_lower for keyword in general_keywords) and not mentioned_columns:
            query_type = 'general'
            context_level = 'none'
            context_data = None
            print_to_log(f"🔍 DEBUG: [_analyze_query_context] Classified as GENERAL (no context needed)")
            
        elif mentioned_columns or any(keyword in query_lower for keyword in column_keywords):
            query_type = 'column_specific'
            context_level = 'column'
            # Get context for specific column
            if mentioned_columns:
                column_name = mentioned_columns[0]  # Use first mentioned column
                print_to_log(f"🔍 DEBUG: [_analyze_query_context] Using mentioned column: {column_name}")
            else:
                # Try to infer from current phase context
                current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                column_name = None
                print_to_log(f"🔍 DEBUG: [_analyze_query_context] No specific column mentioned, current phase: {current_phase}")
            
            print_to_log(f"🔍 DEBUG: [_analyze_query_context] Classified as COLUMN_SPECIFIC for column: {column_name}")
            context_data = self._get_column_context(column_name, state)
            
        elif any(keyword in query_lower for keyword in comparative_keywords):
            query_type = 'comparative' 
            context_level = 'full'
            print_to_log(f"🔍 DEBUG: [_analyze_query_context] Classified as COMPARATIVE (full context)")
            context_data = self._get_full_analysis_context(state)
            
        else:
            query_type = 'phase_specific'
            context_level = 'phase'
            print_to_log(f"🔍 DEBUG: [_analyze_query_context] Classified as PHASE_SPECIFIC (current phase context)")
            context_data = self._get_phase_context(state)
        
        result = {
            'type': query_type,
            'context_level': context_level,
            'context': context_data,
            'column': mentioned_columns[0] if mentioned_columns else None
        }
        
        print_to_log(f"🔍 DEBUG: [_analyze_query_context] Final analysis result:")
        print_to_log(f"   - Type: {result['type']}")
        print_to_log(f"   - Context level: {result['context_level']}")
        print_to_log(f"   - Column: {result['column']}")
        print_to_log(f"   - Context available: {'Yes' if result['context'] else 'No'}")
        
        return result
    
    def _get_column_context(self, column_name: str, state: PipelineState) -> str:
        """Get analysis context for a specific column"""
        print_to_log(f"🔍 DEBUG: [_get_column_context] Getting context for column: '{column_name}'")
        
        try:
            context_parts = []
            
            # Basic column info
            data_to_analyze = state.cleaned_data if state.cleaned_data is not None else state.raw_data
            print_to_log(f"🔍 DEBUG: [_get_column_context] Using {'cleaned' if state.cleaned_data is not None else 'raw'} data")
            
            if column_name and column_name in data_to_analyze.columns:
                print_to_log(f"🔍 DEBUG: [_get_column_context] Column '{column_name}' found in data")
                col_data = data_to_analyze[column_name]
                context_parts.append(f"""COLUMN: {column_name}
- Data Type: {col_data.dtype}
- Missing Values: {col_data.isnull().sum()} ({col_data.isnull().mean()*100:.1f}%)
- Unique Values: {col_data.nunique()}
- Sample Values: {col_data.dropna().head(3).tolist()}""")
                
                if col_data.dtype in ['int64', 'float64']:
                    print_to_log(f"🔍 DEBUG: [_get_column_context] Adding numeric statistics for '{column_name}'")
                    context_parts.append(f"""- Mean: {col_data.mean():.2f}
- Median: {col_data.median():.2f}
- Std Dev: {col_data.std():.2f}
- Min: {col_data.min():.2f}, Max: {col_data.max():.2f}""")
            else:
                print_to_log(f"⚠️ DEBUG: [_get_column_context] Column '{column_name}' not found in data")
            
            # Get recommendations from analysis results
            if hasattr(state, 'preprocessing_state') and state.preprocessing_state:
                print_to_log(f"🔍 DEBUG: [_get_column_context] Checking preprocessing state for analysis results")
                
                # Check outlier results
                if 'outlier_results' in state.preprocessing_state:
                    print_to_log(f"🔍 DEBUG: [_get_column_context] Found outlier results")
                    outlier_results = state.preprocessing_state['outlier_results']
                    if isinstance(outlier_results, dict) and 'llm_recommendations' in outlier_results:
                        if column_name in outlier_results['llm_recommendations']:
                            rec = outlier_results['llm_recommendations'][column_name]
                            print_to_log(f"🔍 DEBUG: [_get_column_context] Adding outlier analysis for '{column_name}'")
                            context_parts.append(f"""OUTLIER ANALYSIS:
- Recommended Treatment: {rec.get('treatment', 'N/A')}
- Reasoning: {rec.get('reasoning', 'N/A')}
- Severity: {rec.get('severity', 'N/A')}""")
                        else:
                            print_to_log(f"🔍 DEBUG: [_get_column_context] No outlier analysis found for '{column_name}'")
                
                # Check missing values results  
                if 'missing_results' in state.preprocessing_state:
                    print_to_log(f"🔍 DEBUG: [_get_column_context] Found missing values results")
                    missing_results = state.preprocessing_state['missing_results']
                    if isinstance(missing_results, dict) and 'llm_recommendations' in missing_results:
                        if column_name in missing_results['llm_recommendations']:
                            rec = missing_results['llm_recommendations'][column_name]
                            print_to_log(f"🔍 DEBUG: [_get_column_context] Adding missing values analysis for '{column_name}'")
                            context_parts.append(f"""MISSING VALUES ANALYSIS:
- Recommended Strategy: {rec.get('strategy', 'N/A')}
- Reasoning: {rec.get('reasoning', 'N/A')}
- Priority: {rec.get('priority', 'N/A')}""")
                        else:
                            print_to_log(f"🔍 DEBUG: [_get_column_context] No missing values analysis found for '{column_name}'")
                
                # Check encoding results
                if 'encoding_results' in state.preprocessing_state:
                    print_to_log(f"🔍 DEBUG: [_get_column_context] Found encoding results")
                    encoding_results = state.preprocessing_state['encoding_results']
                    if isinstance(encoding_results, dict) and 'llm_recommendations' in encoding_results:
                        if column_name in encoding_results['llm_recommendations']:
                            rec = encoding_results['llm_recommendations'][column_name]
                            print_to_log(f"🔍 DEBUG: [_get_column_context] Adding encoding analysis for '{column_name}'")
                            context_parts.append(f"""ENCODING ANALYSIS:
- Recommended Strategy: {rec.get('strategy', 'N/A')}
- Reasoning: {rec.get('reasoning', 'N/A')}
- Cardinality Level: {rec.get('cardinality_level', 'N/A')}""")
                        else:
                            print_to_log(f"🔍 DEBUG: [_get_column_context] No encoding analysis found for '{column_name}'")
                
                # Check transformation results
                if 'transformation_results' in state.preprocessing_state:
                    print_to_log(f"🔍 DEBUG: [_get_column_context] Found transformation results")
                    transformation_results = state.preprocessing_state['transformation_results']
                    if isinstance(transformation_results, dict) and 'llm_recommendations' in transformation_results:
                        if column_name in transformation_results['llm_recommendations']:
                            rec = transformation_results['llm_recommendations'][column_name]
                            print_to_log(f"🔍 DEBUG: [_get_column_context] Adding transformation analysis for '{column_name}'")
                            context_parts.append(f"""TRANSFORMATION ANALYSIS:
- Recommended Transformation: {rec.get('transformation', 'N/A')}
- Reasoning: {rec.get('reasoning', 'N/A')}
- Priority: {rec.get('priority', 'N/A')}""")
                        else:
                            print_to_log(f"🔍 DEBUG: [_get_column_context] No transformation analysis found for '{column_name}'")
            else:
                print_to_log(f"⚠️ DEBUG: [_get_column_context] No preprocessing state available")
            
            final_context = '\n\n'.join(context_parts) if context_parts else f"Limited context available for column: {column_name}"
            print_to_log(f"🔍 DEBUG: [_get_column_context] Generated context with {len(context_parts)} sections")
            return final_context
            
        except Exception as e:
            print_to_log(f"⚠️ DEBUG: [_get_column_context] Error getting column context: {e}")
            return f"Unable to retrieve detailed context for column: {column_name}"
    
    def _get_phase_context(self, state: PipelineState) -> str:
        """Get analysis context for current phase"""
        print_to_log(f"🔍 DEBUG: [_get_phase_context] Getting phase context")
        
        try:
            current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
            print_to_log(f"🔍 DEBUG: [_get_phase_context] Current phase: {current_phase}")
            
            context_parts = [f"CURRENT PHASE: {current_phase}"]
            
            if hasattr(state, 'preprocessing_state') and state.preprocessing_state:
                phase_key = f"{current_phase}_results"
                print_to_log(f"🔍 DEBUG: [_get_phase_context] Looking for phase key: {phase_key}")
                
                if phase_key in state.preprocessing_state:
                    print_to_log(f"🔍 DEBUG: [_get_phase_context] Found results for phase: {current_phase}")
                    results = state.preprocessing_state[phase_key]
                    if isinstance(results, dict) and 'llm_recommendations' in results:
                        context_parts.append("PHASE RECOMMENDATIONS:")
                        rec_count = 0
                        for col, rec in results['llm_recommendations'].items():
                            strategy = rec.get('treatment') or rec.get('strategy') or rec.get('transformation', 'N/A')
                            reasoning = rec.get('reasoning', 'N/A')
                            context_parts.append(f"- {col}: {strategy} ({reasoning})")
                            rec_count += 1
                        print_to_log(f"🔍 DEBUG: [_get_phase_context] Added {rec_count} recommendations")
                    else:
                        print_to_log(f"🔍 DEBUG: [_get_phase_context] No LLM recommendations in phase results")
                else:
                    print_to_log(f"🔍 DEBUG: [_get_phase_context] No results found for phase key: {phase_key}")
            else:
                print_to_log(f"⚠️ DEBUG: [_get_phase_context] No preprocessing state available")
            
            final_context = '\n'.join(context_parts)
            print_to_log(f"🔍 DEBUG: [_get_phase_context] Generated phase context with {len(context_parts)} parts")
            return final_context
            
        except Exception as e:
            print_to_log(f"⚠️ DEBUG: [_get_phase_context] Error getting phase context: {e}")
            return "Unable to retrieve phase context"
    
    def _get_full_analysis_context(self, state: PipelineState) -> str:
        """Get complete analysis context for comparative queries"""
        print_to_log(f"🔍 DEBUG: [_get_full_analysis_context] Getting full analysis context")
        
        try:
            context_parts = []
            
            # Dataset overview
            data_to_analyze = state.cleaned_data if state.cleaned_data is not None else state.raw_data
            print_to_log(f"🔍 DEBUG: [_get_full_analysis_context] Dataset shape: {data_to_analyze.shape}")
            
            context_parts.append(f"""DATASET OVERVIEW:
- Shape: {data_to_analyze.shape[0]} rows × {data_to_analyze.shape[1]} columns
- Target Column: {state.target_column}
- Columns: {', '.join(data_to_analyze.columns.tolist())}""")
            
            # All analysis results
            if hasattr(state, 'preprocessing_state') and state.preprocessing_state:
                print_to_log(f"🔍 DEBUG: [_get_full_analysis_context] Checking all phase results")
                
                phases_found = 0
                for phase in ['outlier', 'missing', 'encoding', 'transformation']:
                    phase_key = f"{phase}_results"
                    if phase_key in state.preprocessing_state:
                        print_to_log(f"🔍 DEBUG: [_get_full_analysis_context] Found results for phase: {phase}")
                        results = state.preprocessing_state[phase_key]
                        if isinstance(results, dict) and 'llm_recommendations' in results:
                            context_parts.append(f"\n{phase.upper()} RECOMMENDATIONS:")
                            rec_count = 0
                            for col, rec in results['llm_recommendations'].items():
                                strategy = rec.get('treatment') or rec.get('strategy') or rec.get('transformation', 'N/A')
                                reasoning = rec.get('reasoning', 'N/A')
                                context_parts.append(f"- {col}: {strategy} ({reasoning})")
                                rec_count += 1
                            print_to_log(f"🔍 DEBUG: [_get_full_analysis_context] Added {rec_count} recommendations for {phase}")
                            phases_found += 1
                
                print_to_log(f"🔍 DEBUG: [_get_full_analysis_context] Total phases with results: {phases_found}")
            else:
                print_to_log(f"⚠️ DEBUG: [_get_full_analysis_context] No preprocessing state available")
            
            final_context = '\n'.join(context_parts)
            print_to_log(f"🔍 DEBUG: [_get_full_analysis_context] Generated full context with {len(context_parts)} sections")
            return final_context
            
        except Exception as e:
            print_to_log(f"⚠️ DEBUG: [_get_full_analysis_context] Error getting full context: {e}")
            return "Unable to retrieve complete analysis context"

    def _handle_target_selection(self, state: PipelineState, command: str) -> PipelineState:
        """Handle interactive target column selection at the start of preprocessing"""
        print_to_log("🎯 Starting interactive target column selection...")
        
        # Get available columns
        available_columns = list(state.raw_data.columns)
        print_to_log(f"📊 Available columns: {available_columns}")
        
        # Check if user is specifying a target column
        command_lower = command.lower().strip()
        
        # Handle various target selection formats
        target_column = None
        if command_lower.startswith('target '):
            # Format: "target column_name"
            target_column = command[7:].strip()
        elif command_lower.startswith('set target '):
            # Format: "set target column_name"
            target_column = command[11:].strip()
        elif command_lower.startswith('use ') and ' as target' in command_lower:
            # Format: "use column_name as target"
            target_column = command_lower.replace('use ', '').replace(' as target', '').strip()
        elif command.strip() in available_columns:
            # Direct column name
            target_column = command.strip()
        elif len(available_columns) == 1:
            # Only one column available, ask for confirmation
            target_column = available_columns[0]
            
        # Validate target column
        if target_column and target_column in available_columns:
            # Set target column and continue preprocessing
            state.target_column = target_column
            
            # IMMEDIATE SAVE: Persist target column to session state for fallback
            try:
                from pipeline_state import state_manager
                state_manager.save_state(state)
                print_to_log(f"💾 Immediately saved target column '{target_column}' to session state")
            except Exception as e:
                print_to_log(f"⚠️ Could not immediately save target column: {e}")
            print_to_log(f"✅ Target column set: {target_column}")
            
            # Send confirmation message
            slack_manager = getattr(state, '_slack_manager', None)
            if not slack_manager:
                from toolbox import slack_manager as global_slack_manager
                slack_manager = global_slack_manager
            
            if slack_manager and state.chat_session:
                confirmation_msg = f"""✅ Target column set: `{target_column}`

🧹 Sequential Preprocessing Agent

📊 Current Dataset: {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns
🎯 Target Column: {target_column}

🔄 Preprocessing Phases:
• `Overview` - Dataset analysis and summary
• `Outliers` - Detect and handle outliers  
• `Missing Values` - Handle missing data
• `Encoding` - Categorical variable encoding
• `Transformations` - Feature transformations

💬 Your Options:
• `proceed` - Start preprocessing workflow
• `skip overview` - Skip to outlier detection
• `explain outliers` - Learn about outlier handling
• `summary` - Show current status

💬 What would you like to do?"""
                
                slack_manager.send_message(state.chat_session, confirmation_msg)
            
            # Initialize preprocessing state
            if not state.preprocessing_state:
                state.preprocessing_state = {}
            state.preprocessing_state['current_phase'] = 'overview'
            
            return state
            
        else:
            # Show target selection prompt
            slack_manager = getattr(state, '_slack_manager', None)
            if not slack_manager:
                from toolbox import slack_manager as global_slack_manager
                slack_manager = global_slack_manager
            
            if slack_manager and state.chat_session:
                # Create column list with numbering for easy selection
                column_list = []
                for i, col in enumerate(available_columns, 1):
                    column_list.append(f"{i}. `{col}`")
                
                columns_text = "\n".join(column_list)
                
                selection_msg = f"""🎯 Target Column Selection Required

📊 Dataset: {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns

📋 Available Columns:
{columns_text}

💬 How to select:
• Type: `target column_name` (e.g., `target price`)
• Type: `use column_name as target` (e.g., `use price as target`)
• Or just type the column name directly

❓ Which column should be used as the target variable for prediction?"""
                
                if target_column and target_column not in available_columns:
                    error_msg = f"\n❌ Column '{target_column}' not found. Please choose from the list above."
                    selection_msg += error_msg
                
                slack_manager.send_message(state.chat_session, selection_msg)
            
            return state


class FeatureSelectionAgentWrapper:
    """Minimal wrapper for the working feature selection agent"""
    
    def __init__(self):
        self.available = FEATURE_SELECTION_AVAILABLE
        self.bot = None
        if self.available:
            try:
                # Initialize the working bot AS-IS
                self.bot = AgenticFeatureSelectionBot()
                print_to_log("✅ Feature selection bot initialized")
            except Exception as e:
                print_to_log(f"❌ Failed to initialize feature selection bot: {e}")
                self.available = False

    def handle_interactive_command(self, state: PipelineState, command: str) -> PipelineState:
        """Handle interactive commands for feature selection with 4-level BGE classification"""
# Set session context for proper logging
        try:
            from session_context import set_session_context, extract_session_from_session_id
            if hasattr(state, 'session_id') and state.session_id:
                user_id, thread_id = extract_session_from_session_id(state.session_id)
                set_session_context(user_id, thread_id)
            elif hasattr(state, 'chat_session') and state.chat_session:
                user_id, thread_id = extract_session_from_session_id(state.chat_session)
                set_session_context(user_id, thread_id)
        except ImportError:
            pass
        print_to_log(f"🔧 DEBUG FS HANDLER: Called with command='{command}'")
        print_to_log(f"🔧 DEBUG FS HANDLER: Available={self.available}, Bot={self.bot is not None}")
        print_to_log(f"🔧 DEBUG FS HANDLER: State chat_session={state.chat_session}")
        print_to_log(f"🔧 DEBUG FS HANDLER: State interactive_session={state.interactive_session}")
        
        if not self.available or not self.bot:
            print_to_log("❌ Feature selection agent not available")
            return state
        
        print_to_log(f"🎯 Feature Selection Interactive Command: '{command}'")
        
        # Get thread logger
        if hasattr(state, 'chat_session') and state.chat_session:
            session_id = state.chat_session
            if '_' in session_id:
                parts = session_id.split('_')
                user_id = parts[0] if len(parts) >= 1 else session_id
                thread_id = '_'.join(parts[1:]) if len(parts) > 1 else session_id
            else:
                user_id = session_id
                thread_id = session_id
            thread_logger = get_thread_logger(user_id, thread_id)
            thread_logger.log_query(command, agent="feature_selection")
        else:
            thread_logger = None
        
        # Get slack_manager from state or fallback
        slack_manager = getattr(state, '_slack_manager', None)
        if not slack_manager:
            from toolbox import slack_manager as global_slack_manager
            slack_manager = global_slack_manager
        print_to_log(f"🔧 DEBUG FS HANDLER: Using slack_manager id: {id(slack_manager)}")
        print_to_log(f"🔧 DEBUG FS HANDLER: Slack manager available: {slack_manager is not None}")
        
        # Get or create session for this user
        session_id = state.chat_session
        if not session_id:
            print_to_log("❌ No chat session ID available")
            return state
        
        try:
            # Check if session exists in bot
            print_to_log(f"🔧 DEBUG FS HANDLER: Checking if session {session_id} exists in bot")
            print_to_log(f"🔧 DEBUG FS HANDLER: Bot users keys: {list(self.bot.users.keys())}")
            
            if session_id not in self.bot.users:
                print_to_log(f"🔧 Session {session_id} not found in bot users")
                print_to_log(f"🔧 DEBUG FS HANDLER: Checking if this is a continuation with existing interactive session")
                
                # If we have an active interactive session, we should continue it, not create new
                if (state.interactive_session and 
                    state.interactive_session.get('agent_type') == 'feature_selection' and
                    state.interactive_session.get('session_active', False)):
                    
                    print_to_log(f"🔧 Found active FS interactive session, recreating bot session from state")
                    
                    # Recreate the session from the pipeline state instead of calling run()
                    if state.cleaned_data is None:
                        print_to_log("❌ No cleaned data available to recreate session")
                        return state
                    
                    # Create session for the working agent from existing state
                    import tempfile
                    import os
                    temp_file = os.path.join(tempfile.gettempdir(), f"cleaned_data_{state.session_id}.csv")
                    state.cleaned_data.to_csv(temp_file, index=False)
                    
                    # Use stored state if available, otherwise defaults
                    stored_phase = "waiting_input"
                    stored_features = list(state.cleaned_data.columns)
                    
                    if state.feature_selection_state:
                        stored_phase = state.feature_selection_state.get('phase', stored_phase)
                        stored_features = state.feature_selection_state.get('current_features', stored_features)
                        print_to_log(f"🔧 Using stored FS state: phase={stored_phase}, features={len(stored_features)}")
                    
                    from feature_selection_agent_impl import UserSession, AnalysisStep
                    from datetime import datetime
                    import pandas as pd
                    
                    # ✅ INTELLIGENT CLEANING: Apply the same smart cleaning logic as load_and_clean_data
                    print_to_log(f"🧠 Applying intelligent cleaning to pipeline data...")
                    df = state.cleaned_data.copy()
                    print_to_log(f"📊 Pipeline data: {df.shape[0]} rows, {df.shape[1]} columns")
                    
                    # Step 1: Remove single value columns
                    single_value_cols = []
                    for col in df.columns:
                        if df[col].nunique() <= 1:
                            single_value_cols.append(col)
                    
                    # Step 2: Smart object column handling - try to convert to numeric first
                    object_cols = [col for col in df.columns if col not in single_value_cols and df[col].dtype == 'object']
                    converted_cols = []
                    failed_conversion_cols = []
                    
                    if object_cols:
                        print_to_log(f"🔍 Found {len(object_cols)} object columns, attempting numeric conversion...")
                        
                        for col in object_cols:
                            try:
                                original_series = df[col].copy()
                                
                                # Try direct conversion
                                converted = pd.to_numeric(original_series, errors='coerce')
                                
                                # If that fails, try cleaning string formats
                                if converted.isna().sum() > len(original_series) * 0.5:
                                    cleaned_series = original_series.astype(str).str.replace(',', '').str.replace(' ', '').str.strip()
                                    converted = pd.to_numeric(cleaned_series, errors='coerce')
                                
                                # Check conversion success rate
                                non_null_before = original_series.notna().sum()
                                non_null_after = converted.notna().sum()
                                
                                if non_null_after >= non_null_before * 0.8:  # 80% success rate
                                    df[col] = converted
                                    converted_cols.append(col)
                                    print_to_log(f"   ✅ Converted '{col}' to numeric ({non_null_after}/{non_null_before} values)")
                                else:
                                    failed_conversion_cols.append(col)
                                    print_to_log(f"   ❌ Failed to convert '{col}' (only {non_null_after}/{non_null_before} values convertible)")
                                    
                            except Exception as e:
                                failed_conversion_cols.append(col)
                                print_to_log(f"   ❌ Error converting '{col}': {str(e)[:50]}")
                    
                    # Step 3: Remove remaining non-numeric columns
                    cols_to_remove = single_value_cols + failed_conversion_cols
                    if cols_to_remove:
                        clean_df = df.drop(columns=cols_to_remove)
                        print_to_log(f"📈 Removed {len(cols_to_remove)} columns: {len(single_value_cols)} single-value + {len(failed_conversion_cols)} non-convertible")
                    else:
                        clean_df = df.copy()
                        print_to_log(f"📈 No columns needed removal - all data is numeric-ready")
                    
                    print_to_log(f"✅ Final clean dataset: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns")
                    
                    # Update the temp file with cleaned data
                    clean_df.to_csv(temp_file, index=False)
                    
                    session = UserSession(
                        file_path=temp_file,
                        file_name=f"cleaned_data_{state.session_id}.csv",
                        user_id=session_id,
                        target_column=state.target_column,
                        original_df=state.cleaned_data.copy(),  # Keep original for reference
                        current_df=clean_df.copy(),             # Use cleaned data
                        current_features=list(clean_df.columns),  # Use cleaned features
                        phase=stored_phase
                    )
                    
                    # Add intelligent cleaning step to analysis chain
                    if cols_to_remove or converted_cols:
                        cleaning_step = AnalysisStep(
                            type="intelligent_data_cleaning",
                            parameters={"removed_cols": cols_to_remove},
                            features_before=df.shape[1],
                            features_after=clean_df.shape[1],
                            timestamp=datetime.now().isoformat(),
                            metadata={
                                "single_value_cols": single_value_cols,
                                "converted_to_numeric": converted_cols,
                                "failed_conversion_cols": failed_conversion_cols,
                                "conversion_strategy": "smart_numeric_conversion",
                                "source": "pipeline_integration"
                            }
                        )
                        session.analysis_chain.append(cleaning_step)
                    
                    # ✅ CRITICAL FIX: Create the "after_cleaning" snapshot with the cleaned data
                    session.snapshots["after_cleaning"] = {
                        "df": clean_df.copy(),
                        "features": list(clean_df.columns),
                        "timestamp": datetime.now().isoformat()
                    }
                    print_to_log(f"✅ Created 'after_cleaning' snapshot with {clean_df.shape[1]} numeric columns for revert functionality")
                    
                    # Store session in the working bot
                    self.bot.users[session_id] = session
                    print_to_log(f"🔧 Recreated bot session: phase={session.phase}, features={len(session.current_features)}")
                else:
                    print_to_log(f"🔧 No active FS interactive session, creating new session")
                    # Initialize session if it doesn't exist
                    return self.run(state)
            
            # Get existing session
            session = self.bot.users[session_id]
            print_to_log(f"📊 Existing session found: phase={session.phase}, features={len(session.current_features)}")
            print_to_log(f"🔧 DEBUG FS HANDLER: Session target={session.target_column}, file={session.file_name}")
            
            # Create a mock Slack say function that sends to our pipeline (MOVED UP)
            def mock_say(message, thread_ts=None):
                print_to_log(f"🔧 DEBUG MOCK_SAY: Attempting to send message via slack_manager id: {id(slack_manager)}")
                print_to_log(f"🔧 DEBUG FS HANDLER: Slack manager available: {slack_manager is not None}")
                
                if slack_manager and state.chat_session:
                    slack_manager.send_message(state.chat_session, message)
                else:
                    print_to_log(f"[Mock Slack Response]: {message}")
            
            print_to_log(f"✅ Session loaded with {len(session.current_features)} clean features (cleaned at load time)")
            
            # ✅ 4-LEVEL BGE CLASSIFICATION FOR FEATURE SELECTION
            print_to_log(f"🧠 [FS] Starting 4-level BGE intent classification...")
            print_to_log(f"🔧 DEBUG FS BGE: Command to classify='{command}'")
            
            # Import pipeline to access the classification method
            from langgraph_pipeline import MultiAgentMLPipeline
            pipeline_instance = MultiAgentMLPipeline()
            action_intent = pipeline_instance._classify_feature_selection_action(command)
            print_to_log(f"🎯 [FS] 4-level BGE classified intent: '{action_intent}'")
            print_to_log(f"🔧 DEBUG FS BGE: Classification result='{action_intent}'")
            
            # Handle BGE-classified intents with clear intent signals
            if action_intent == 'proceed':
                mapped_command = f'PROCEED: {command}'
                print_to_log(f"🔄 [FS] Mapping 'proceed' → '{mapped_command}' command (BGE intent)")
            elif action_intent == 'analysis':
                mapped_command = f'ANALYSIS: {command}'
                print_to_log(f"🔄 [FS] Mapping 'analysis' → '{mapped_command}' command (BGE intent)")
            elif action_intent == 'query':
                mapped_command = f'QUERY: {command}'
                print_to_log(f"🔄 [FS] Mapping 'query' → '{mapped_command}' command (BGE intent)")
            elif action_intent == 'summary':
                mapped_command = f'SUMMARY: {command}'
                print_to_log(f"🔄 [FS] Mapping 'summary' → '{mapped_command}' command (BGE intent)")
            elif action_intent == 'revert':
                mapped_command = f'REVERT: {command}'
                print_to_log(f"🔄 [FS] Mapping 'revert' → '{mapped_command}' command (BGE intent)")
            elif action_intent == 'datetime':
                mapped_command = f'DATETIME: {command}'
                print_to_log(f"🔄 [FS] Mapping 'datetime' → '{mapped_command}' command (BGE intent)")
            elif action_intent == 'suggestion':
                mapped_command = f'SUGGESTION: {command}'
                print_to_log(f"🔄 [FS] Mapping 'suggestion' → '{mapped_command}' command (BGE intent)")
            else:
                mapped_command = command  # Fallback
                print_to_log(f"🔄 [FS] Mapping '{action_intent}' → '{command}' command (fallback)")
            
            # Handle BGE-classified intents
            print_to_log(f"🔧 DEBUG FS INTENT: Mapped command='{mapped_command}'")
            
            if mapped_command.startswith('PROCEED: '):
                actual_command = mapped_command[9:]  # Remove 'PROCEED: ' prefix
                print_to_log("✅ Processing BGE-classified PROCEED command for feature selection...")
                print_to_log(f"✅ DEBUG: BGE classified proceed: '{actual_command}'")
                print_to_log(f"🔧 DEBUG FS PROCEED: Current session phase={session.phase}")
                
                # Proceed is a completion command - generate final summary
                print_to_log(f"🔧 DEBUG FS PROCEED: Generating final summary (completion command)")
                session.phase = "completed"
                
                # 💾 Save selected features with target before generating summary
                print_to_log(f"�� FS PROCEED: Saving selected features before summary generation")
                print_to_log(f"📋 FS PROCEED: Current features = {session.current_features}")
                print_to_log(f"🎯 FS PROCEED: Target column = {state.target_column}")
                
                self.bot.generate_final_summary(session, mock_say)
                
            elif mapped_command.startswith('ANALYSIS: '):
                actual_command = mapped_command[10:]  # Remove 'ANALYSIS: ' prefix
                print_to_log("🔬 Processing BGE-classified ANALYSIS command for feature selection...")
                print_to_log(f"🔬 DEBUG: BGE classified analysis: '{actual_command}'")
                # Route to analysis handler
                self.bot.handle_analysis_request(session, actual_command, mock_say)
                
            elif mapped_command.startswith('QUERY: '):
                actual_command = mapped_command[7:]  # Remove 'QUERY: ' prefix
                print_to_log("❓ Processing BGE-classified QUERY command for feature selection...")
                print_to_log(f"❓ DEBUG: BGE classified query: '{actual_command}'")
                # Route to query handler
                self.bot.handle_analysis_request(session, actual_command, mock_say)
                
            elif mapped_command.startswith('SUMMARY: '):
                actual_command = mapped_command[9:]  # Remove 'SUMMARY: ' prefix
                print_to_log("📊 Processing BGE-classified SUMMARY command for feature selection...")
                print_to_log(f"📊 DEBUG: BGE classified summary: '{actual_command}'")
                # Route to summary handler
                from feature_selection_agent_impl import MenuGenerator
                MenuGenerator.show_crisp_summary(session, mock_say)
                
            elif mapped_command.startswith('REVERT: '):
                actual_command = mapped_command[8:]  # Remove 'REVERT: ' prefix
                print_to_log("↩️ Processing BGE-classified REVERT command for feature selection...")
                print_to_log(f"↩️ DEBUG: BGE classified revert: '{actual_command}'")
                # Route to revert handler
                self.bot.handle_revert(session, mock_say)
                
            elif mapped_command.startswith('DATETIME: '):
                actual_command = mapped_command[10:]  # Remove 'DATETIME: ' prefix
                print_to_log("📅 Processing BGE-classified DATETIME command for feature selection...")
                print_to_log(f"📅 DEBUG: BGE classified datetime: '{actual_command}'")
                # Route to datetime handler - use the correct method name
                self.bot.handle_datetime_setup(session, actual_command, mock_say)
                
            elif mapped_command.startswith('SUGGESTION: '):
                actual_command = mapped_command[12:]  # Remove 'SUGGESTION: ' prefix
                print_to_log("💡 Processing BGE-classified SUGGESTION command for feature selection...")
                print_to_log(f"💡 DEBUG: BGE classified suggestion: '{actual_command}'")
                
                # Custom suggestion handler with LLM
                try:
                    from feature_selection_agent_impl import LLMManager
                    llm = LLMManager.get_llm(session.model_name)
                    
                    # Get current session context
                    current_features = len(session.current_features)
                    completed_analyses = [step.type for step in session.analysis_chain]
                    total_original = len(session.original_df.columns) if hasattr(session, 'original_df') else current_features
                    
                    # Create LLM prompt for suggestions
                    prompt = f"""You are a data science advisor. Based on the current feature selection progress, provide 2-3 concise bullet point suggestions for next steps.

CURRENT STATE:
• Features remaining: {current_features}
• Original features: {total_original}
• Completed analyses: {', '.join(completed_analyses) if completed_analyses else 'None yet'}
• Target column: {session.target_column}

USER REQUEST: "{actual_command}"

Provide exactly 2-3 bullet points with:
• Specific analysis recommendations (IV, Correlation, VIF, SHAP, etc.)
• Practical thresholds/parameters
• Brief rationale for each suggestion

Format as:
• Analysis Name - Brief description with suggested parameters
• Analysis Name - Brief description with suggested parameters  
• Analysis Name - Brief description with suggested parameters

Keep it concise and actionable."""

                    from langchain_core.messages import HumanMessage
                    response = llm.invoke([HumanMessage(content=prompt)])
                    
                    # Format and send suggestion
                    suggestion_message = f"💡 Data Science Suggestions:\n\n{response.content}\n\n💬 Just tell me which analysis you'd like to run!"
                    mock_say(suggestion_message)
                    
                except Exception as e:
                    print_to_log(f"❌ Suggestion generation failed: {e}")
                    mock_say("💡 I'd recommend starting with IV analysis to evaluate feature importance, then correlation analysis to remove redundant features!")
                
            else:
                # Fallback to regular processing
                print_to_log(f"🔄 Processing regular command for feature selection: '{command}'")
                if session.phase == "need_target":
                    # Handle target selection
                    self.bot.handle_target_selection(session, command, mock_say)
                else:
                    # Handle analysis requests
                    self.bot.handle_analysis_request(session, command, mock_say)
            
            # Sync session state back to pipeline state
            print_to_log(f"🔧 DEBUG FS HANDLER: Syncing session state back to pipeline state")
            self._sync_session_to_state(session, state)
            print_to_log(f"🔧 DEBUG FS HANDLER: State sync completed")
            
        except Exception as e:
            print_to_log(f"❌ Error handling interactive command: {e}")
            import traceback
            traceback.print_exc()
        
        print_to_log(f"🔧 DEBUG FS HANDLER: Returning state, interactive_session={state.interactive_session}")
        return state
    
    def _sync_session_to_state(self, session, state: PipelineState):
        """Sync UserSession back to PipelineState"""
        try:
            # Update feature selection state
            state.feature_selection_state.update({
                "current_features": session.current_features,
                "dropped_features": getattr(session, 'dropped_features', []),
                "analysis_chain": [{"type": step.type, "parameters": step.parameters} for step in getattr(session, 'analysis_chain', [])],
                "phase": session.phase,
                "session_active": True,
                "current_feature_count": len(session.current_features)  # ✅ Add current feature count
            })
            
            # Update selected features if analysis is complete
            if session.phase == "completed":
                # Ensure target column is included with selected features for modeling
                selected_features = session.current_features.copy()
                
                # Add target column if it's not already in selected features
                if state.target_column and state.target_column not in selected_features:
                    selected_features.append(state.target_column)
                    print_to_log(f"🎯 Added target column '{state.target_column}' to selected features for modeling")
                
                state.selected_features = selected_features
                print_to_log(f"✅ Final selected features for modeling: {len(state.selected_features)} (including target)")
            
            print_to_log(f"🔄 Synced session state: {len(session.current_features)} features, phase={session.phase}")
            
        except Exception as e:
            print_to_log(f"⚠️ Error syncing session state: {e}")
        
    def run(self, state: PipelineState) -> PipelineState:
        """Route to the actual working feature selection agent"""
        if not self.available or not self.bot:
            print_to_log("❌ Feature selection agent not available")
            return state
            
        try:
            # ✅ RAW DATA FALLBACK: Use raw_data if no cleaned_data available
            if state.cleaned_data is None:
                if state.raw_data is not None:
                    print_to_log("⚠️  No cleaned data found - using raw data for feature selection")
                    state.cleaned_data = state.raw_data.copy()
                    print_to_log(f"📊 Using raw data: {state.cleaned_data.shape}")
                else:
                    print_to_log("❌ No data available for feature selection (no raw_data or cleaned_data)")
                    
                    # Send helpful message to user
                    slack_manager = getattr(state, '_slack_manager', None)
                    if not slack_manager:
                        from toolbox import slack_manager as global_slack_manager
                        slack_manager = global_slack_manager
                    
                    # Set proper error response and clear any cached responses
                    error_message = """❌ No Data Available for Feature Selection

Please provide data first:
• Upload a CSV file to Slack
• Or run preprocessing first with your data

Example:
• Upload `data.csv` file
• Or say: "preprocess my data" first"""
                    
                    # Clear any previous responses and set the error message
                    state.last_response = error_message
                    state.last_error = "No data available for feature selection"
                    
                    if slack_manager and state.chat_session:
                        slack_manager.send_message(state.chat_session, error_message)
                    
                    return state
                
            # ✅ INTELLIGENT CLEANING: Apply smart cleaning at data load time
            print_to_log(f"🧠 Applying intelligent cleaning at data load...")
            df = state.cleaned_data.copy()
            original_shape = df.shape
            print_to_log(f"📊 Original pipeline data: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Step 1: Remove single value columns (excluding target)
            single_value_cols = []
            for col in df.columns:
                if col != state.target_column and df[col].nunique() <= 1:
                    single_value_cols.append(col)
            
            # Step 2: Smart object column handling - try to convert to numeric first
            object_cols = [col for col in df.columns if col != state.target_column and df[col].dtype == 'object']
            converted_cols = []
            failed_conversion_cols = []
            
            if object_cols:
                print_to_log(f"🔍 Found {len(object_cols)} object columns, attempting numeric conversion...")
                
                for col in object_cols:
                    try:
                        original_series = df[col].copy()
                        
                        # Try direct conversion
                        converted = pd.to_numeric(original_series, errors='coerce')
                        
                        # If that fails, try cleaning string formats
                        if converted.isna().sum() > len(original_series) * 0.5:
                            cleaned_series = original_series.astype(str).str.replace(',', '').str.replace(' ', '').str.strip()
                            converted = pd.to_numeric(cleaned_series, errors='coerce')
                        
                        # Check conversion success rate
                        non_null_before = original_series.notna().sum()
                        non_null_after = converted.notna().sum()
                        
                        if non_null_after >= non_null_before * 0.8:  # 80% success rate
                            df[col] = converted
                            converted_cols.append(col)
                            print_to_log(f"   ✅ Converted '{col}' to numeric ({non_null_after}/{non_null_before} values)")
                        else:
                            failed_conversion_cols.append(col)
                            print_to_log(f"   ❌ Failed to convert '{col}' (only {non_null_after}/{non_null_before} values convertible)")
                            
                    except Exception as e:
                        failed_conversion_cols.append(col)
                        print_to_log(f"   ❌ Error converting '{col}': {str(e)[:50]}")
            
            # Step 3: Remove remaining non-numeric columns (excluding target)
            cols_to_remove = single_value_cols + failed_conversion_cols
            if cols_to_remove:
                clean_df = df.drop(columns=cols_to_remove)
                print_to_log(f"📈 Removed {len(cols_to_remove)} columns: {len(single_value_cols)} single-value + {len(failed_conversion_cols)} non-convertible")
            else:
                clean_df = df.copy()
                print_to_log(f"📈 No columns needed removal - all data is numeric-ready")
            
            print_to_log(f"✅ Final clean dataset: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns")
            
            # Save cleaned data to temp file
            temp_file = os.path.join(tempfile.gettempdir(), f"cleaned_data_{state.session_id}.csv")
            clean_df.to_csv(temp_file, index=False)
            
            # Create session for the working agent with clean data
            session = UserSession(
                file_path=temp_file,
                file_name=f"cleaned_data_{state.session_id}.csv",
                user_id=state.chat_session,
                target_column=state.target_column,
                original_df=state.cleaned_data.copy(),  # Keep original for reference
                current_df=clean_df.copy(),             # Use cleaned data
                current_features=list(clean_df.columns),  # Use cleaned features
                # ✅ PHASE FIX: Set correct phase based on target column availability
                phase="waiting_input" if state.target_column else "need_target"
            )
            
            # Add intelligent cleaning step to analysis chain
            if cols_to_remove or converted_cols:
                from feature_selection_agent_impl import AnalysisStep
                from datetime import datetime
                cleaning_step = AnalysisStep(
                    type="intelligent_data_cleaning_at_load",
                    parameters={"removed_cols": cols_to_remove},
                    features_before=original_shape[1],
                    features_after=clean_df.shape[1],
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        "single_value_cols": single_value_cols,
                        "converted_to_numeric": converted_cols,
                        "failed_conversion_cols": failed_conversion_cols,
                        "conversion_strategy": "smart_numeric_conversion_at_load",
                        "source": "data_load_time"
                    }
                )
                session.analysis_chain.append(cleaning_step)
            
            # ✅ Create the "after_cleaning" snapshot for revert functionality
            from datetime import datetime  # Ensure datetime is available in this scope
            session.snapshots["after_cleaning"] = {
                "df": clean_df.copy(),
                "features": list(clean_df.columns),
                "timestamp": datetime.now().isoformat()
            }
            print_to_log(f"✅ Created 'after_cleaning' snapshot with {clean_df.shape[1]} clean features for revert functionality")
            
            print_to_log(f"🚀 Launching actual feature selection agent")
            print_to_log(f"📊 Data shape: {state.cleaned_data.shape}")
            print_to_log(f"🎯 Target column: {state.target_column}")
            
            # Store session in the working bot
            self.bot.users[state.chat_session] = session
            
            # ✅ DISPLAY INITIAL MENU: Show menu to guide user
            try:
                
                if state.target_column:
                    print_to_log(f"🔧 DEBUG TARGET: Target column found: {state.target_column}")
                    
                    # Target column known, show main menu
                    menu = MenuGenerator.generate_main_menu(session)
                    session.last_menu = menu
                    print_to_log(f"🔧 DEBUG MENU: Generated menu with {len(menu)} characters")
                    
                    # Send menu via slack - FIX SESSION CHANNEL ISSUE
                    slack_manager = getattr(state, '_slack_manager', None)
                    if not slack_manager:
                        from toolbox import slack_manager as global_slack_manager
                        slack_manager = global_slack_manager
                    
                    # ✅ CRITICAL FIX: Store slack_manager in state for consistency
                    state._slack_manager = slack_manager
                    
                    # ✅ STORE SESSION INFO: Backup current session info in state
                    state.slack_session_info = {
                        'channels': dict(slack_manager.session_channels),
                        'threads': dict(slack_manager.session_threads)
                    }
                    print_to_log(f"💾 BACKUP: Stored session info in state: {state.slack_session_info}")
                    
                    # ✅ EMERGENCY FIX: If slack_manager is empty, try to recover from state
                    if not slack_manager.session_channels and hasattr(state, 'slack_session_info'):
                        print_to_log(f"🚨 EMERGENCY RECOVERY: Restoring session info from state")
                        slack_manager.session_channels.update(state.slack_session_info.get('channels', {}))
                        slack_manager.session_threads.update(state.slack_session_info.get('threads', {}))
                        print_to_log(f"🚨 RECOVERED: {slack_manager.session_channels}")
                    
                    # ✅ FINAL FALLBACK: If still empty, create the session entry directly
                    if state.chat_session not in slack_manager.session_channels:
                        print_to_log(f"🚨 FINAL FALLBACK: Creating session entry directly")
                        # Try to infer channel from any available session with same user
                        user_id = state.chat_session.split('_')[0] if '_' in state.chat_session else state.chat_session
                        fallback_channel = None
                        
                        # Check stored session info first
                        if hasattr(state, 'slack_session_info'):
                            for session_id, channel in state.slack_session_info.get('channels', {}).items():
                                if user_id in session_id:
                                    fallback_channel = channel
                                    break
                        
                        # If still no channel, check current slack_manager
                        if not fallback_channel:
                            for session_id, channel in slack_manager.session_channels.items():
                                if user_id in session_id:
                                    fallback_channel = channel
                                    break
                        
                        # If we found a channel, use it
                        if fallback_channel:
                            print_to_log(f"🚨 FALLBACK: Using channel {fallback_channel} for session {state.chat_session}")
                            slack_manager.session_channels[state.chat_session] = fallback_channel
                        else:
                            print_to_log(f"❌ FALLBACK FAILED: No channel found for user {user_id}")
                    
                    print_to_log(f"🔧 DEBUG SLACK: slack_manager available: {slack_manager is not None}")
                    print_to_log(f"🔧 DEBUG SLACK: state.chat_session: {state.chat_session}")
                    
                    if slack_manager and state.chat_session:
                        print_to_log(f"🔧 DEBUG SLACK: Current session channels: {list(slack_manager.session_channels.keys())}")
                        print_to_log(f"🔧 DEBUG SLACK: Current session threads: {list(slack_manager.session_threads.keys())}")
                        
                        # Check if session is already registered
                        if state.chat_session in slack_manager.session_channels:
                            channel = slack_manager.session_channels[state.chat_session]
                            thread = slack_manager.session_threads.get(state.chat_session)
                            print_to_log(f"🔧 DEBUG SLACK: Session already registered - Channel: {channel}, Thread: {thread}")
                        else:
                            print_to_log(f"⚠️ DEBUG SLACK: Session {state.chat_session} not in channels")
                            print_to_log(f"🔧 DEBUG SLACK: Available channels: {slack_manager.session_channels}")
                            
                            # Try to find the channel from other active sessions
                            found_channel = False
                            for session_id, channel in slack_manager.session_channels.items():
                                user_part = state.chat_session.split('_')[0] if '_' in state.chat_session else state.chat_session
                                if session_id.startswith(user_part):
                                    print_to_log(f"🔧 DEBUG SLACK: Found matching channel {channel} from session {session_id}")
                                    slack_manager.session_channels[state.chat_session] = channel
                                    if session_id in slack_manager.session_threads:
                                        slack_manager.session_threads[state.chat_session] = slack_manager.session_threads[session_id]
                                        print_to_log(f"🔧 DEBUG SLACK: Copied thread {slack_manager.session_threads[session_id]}")
                                    found_channel = True
                                    break
                            
                            if not found_channel:
                                print_to_log(f"❌ DEBUG SLACK: No matching channel found for {state.chat_session}")
                                
                                # ✅ LAST RESORT: Try to find ANY active channel for this user
                                user_id = state.chat_session.split('_')[0] if '_' in state.chat_session else state.chat_session
                                for session_id, channel in list(slack_manager.session_channels.items()):
                                    if user_id in session_id:
                                        print_to_log(f"🔧 DEBUG SLACK: LAST RESORT - Found channel {channel} from any session with user {user_id}")
                                        slack_manager.session_channels[state.chat_session] = channel
                                        # Don't copy thread - let it create new thread
                                        found_channel = True
                                        break
                                
                                if not found_channel:
                                    print_to_log(f"❌ DEBUG SLACK: No channel found at all - menu will not be sent to Slack!")
                                    print_to_log(f"❌ DEBUG SLACK: Available channels: {list(slack_manager.session_channels.items())}")
                                    print_to_log(f"❌ DEBUG SLACK: User ID extracted: {user_id}")
                        
                        print_to_log(f"🔧 DEBUG MENU SEND: About to send menu...")
                        slack_manager.send_message(state.chat_session, menu)
                        print_to_log(f"✅ Feature selection menu sent to Slack")
                        
                        # Add a concise action prompt (no duplicate analysis options)
                        action_prompt = """🎯 Ready to start feature selection!

Or ask questions:
• `how many features do we have?`
• `what analysis should I run first?`
• `explain IV analysis`

When finished with all analyses:
• `proceed` - Complete feature selection and show final results

💬 What would you like to do first?"""
                        
                        print_to_log(f"🔧 DEBUG PROMPT SEND: About to send action prompt...")
                        slack_manager.send_message(state.chat_session, action_prompt)
                        print_to_log(f"✅ Action prompt sent to guide user interaction")
                    else:
                        print_to_log(f"❌ DEBUG SLACK: Cannot send menu - slack_manager: {slack_manager is not None}, chat_session: {state.chat_session}")
                else:
                    print_to_log(f"🔧 DEBUG TARGET: No target column found, will show target selection prompt")
                    # Need target column, send target selection prompt
                    slack_manager = getattr(state, '_slack_manager', None)
                    if not slack_manager:
                        from toolbox import slack_manager as global_slack_manager
                        slack_manager = global_slack_manager
                        
                    if slack_manager and state.chat_session:
                        print_to_log(f"🔧 DEBUG: Sending target prompt to session {state.chat_session}")
                        
                        # Ensure session is registered properly for target prompt too
                        if state.chat_session not in slack_manager.session_channels:
                            print_to_log(f"⚠️ Session {state.chat_session} not in channels, attempting to find channel...")
                            for session_id, channel in slack_manager.session_channels.items():
                                if session_id.startswith(state.chat_session.split('_')[0]):
                                    print_to_log(f"🔧 Found channel {channel} from similar session {session_id}")
                                    slack_manager.session_channels[state.chat_session] = channel
                                    if session_id in slack_manager.session_threads:
                                        slack_manager.session_threads[state.chat_session] = slack_manager.session_threads[session_id]
                                    break
                        available_cols = ', '.join(session.current_features[:5])
                        target_prompt = f"""🎯 Target Column Selection

Please specify your target column for feature selection analysis.

📋 Available columns: {available_cols}{'...' if len(session.current_features) > 5 else ''}

💬 How to specify: 
• Type: `target column_name`
• Or just: `column_name`

Example: `target is_fraud` or `is_fraud`"""
                        slack_manager.send_message(state.chat_session, target_prompt)
                        
            except Exception as e:
                print_to_log(f"⚠️ Could not display initial menu: {e}")
            
            # The working agent will handle all Slack interactions from here
            # It will show menus, process user input, run analyses, etc.
            
            # For now, just set up the session and let the bot handle the rest
            state.feature_selection_state = {
                "completed": False,
                "timestamp": datetime.now().isoformat(),
                "method": "agentic_interactive",
                "session_active": True,
                "bot_session_exists": True,
                "session_id": state.chat_session,
                "target_column": state.target_column,
                "current_features": list(state.cleaned_data.columns),
                "phase": session.phase
            }
            
            # ✅ SET INTERACTIVE SESSION - This was missing!
            state.interactive_session = {
                "agent_type": "feature_selection",
                "session_active": True,
                "phase": session.phase,
                "current_phase": "menu",
                "original_query": state.user_query  # CRITICAL: Store original user query
            }
            
            print_to_log("✅ Feature selection session started - bot will handle Slack interactions")
            print_to_log(f"💾 Set interactive_session for session persistence: {state.interactive_session}")
            
            # ✅ TEMP FILE FIX: Don't delete immediately, let the bot use it
            # The temp file will be cleaned up when the session ends or by OS temp cleanup
            print_to_log(f"📁 Temp file preserved for feature selection: {temp_file}")
                
            return state
            
        except Exception as e:
            print_to_log(f"❌ Feature selection agent failed: {e}")
            import traceback
            traceback.print_exc()
            return state


class ModelBuildingAgentWrapper:
    """Minimal wrapper for the working model building agent"""
    
    def __init__(self):
        self.available = MODEL_BUILDING_AVAILABLE
        self.agent = None
        if self.available:
            try:
                # Initialize the working agent AS-IS
                self.agent = LangGraphModelAgent()
                print_to_log("✅ Model building agent initialized")
            except Exception as e:
                print_to_log(f"❌ Failed to initialize model building agent: {e}")
                self.available = False
        
    def run(self, state: PipelineState) -> PipelineState:
        """Route to the actual working model building agent"""
        if not self.available or not self.agent:
            print_to_log("❌ Model building agent not available")
            return state
        
        # Get thread logger
        if hasattr(state, 'chat_session') and state.chat_session:
            session_id = state.chat_session
            if '_' in session_id:
                parts = session_id.split('_')
                user_id = parts[0] if len(parts) >= 1 else session_id
                thread_id = '_'.join(parts[1:]) if len(parts) > 1 else session_id
            else:
                user_id = session_id
                thread_id = session_id
            thread_logger = get_thread_logger(user_id, thread_id)
            thread_logger.log_query(state.user_query, agent="model_building")
        else:
            thread_logger = None
            
        try:
            # Determine which data to use (cleaned > raw > None)
            data_to_use = None
            if state.cleaned_data is not None:
                data_to_use = state.cleaned_data
                print_to_log(f"🚀 Using cleaned data for model building")
            elif state.raw_data is not None:
                data_to_use = state.raw_data
                print_to_log(f"🚀 Using raw data for model building (preprocessing skipped)")
            else:
                print_to_log("❌ No data available - letting model building agent handle this")
                # Let the actual agent handle "no data" case with proper messaging
                
            # Determine features to use (selected > all columns)
            features_to_use = None
            if state.selected_features is not None and (hasattr(state.selected_features, "__len__") and len(state.selected_features) > 0):
                features_to_use = state.selected_features
                print_to_log(f"🎯 Using selected features: {len(state.selected_features)}")
            elif data_to_use is not None:
                # Use all columns except target as features
                all_cols = list(data_to_use.columns)
                if state.target_column and state.target_column in all_cols:
                    features_to_use = [col for col in all_cols if col != state.target_column]
                else:
                    features_to_use = all_cols
                print_to_log(f"🎯 Using all available features: {len(features_to_use)} (feature selection skipped)")
            
            if data_to_use is not None:
                print_to_log(f"📊 Data shape: {data_to_use.shape}")
            
            print_to_log(f"🚀 Launching actual model building agent")
            
            # Load data into the agent if available
            if data_to_use is not None:
                print_to_log(f"📊 Loading data into model building agent")
                
                # CRITICAL: Filter data to use only selected features + target column
                if features_to_use is not None and state.target_column:
                    # Create filtered dataset with selected features + target
                    columns_to_keep = features_to_use.copy()
                    if state.target_column not in columns_to_keep:
                        columns_to_keep.append(state.target_column)
                    
                    # Filter the data to only include selected columns
                    filtered_data = data_to_use[columns_to_keep]
                    print_to_log(f"🎯 FEATURE FILTERING: {data_to_use.shape} → {filtered_data.shape}")
                    print_to_log(f"   📊 Selected features: {len(features_to_use)}")
                    print_to_log(f"   🎯 Target column: {state.target_column}")
                    print_to_log(f"   📋 Total columns for modeling: {len(columns_to_keep)}")
                    
                    data_to_use = filtered_data
                else:
                    print_to_log("⚠️ No feature filtering applied - using all columns")
                
                self.agent.load_data(data_to_use, state.chat_session)
                
                # Set target column if available
                if state.target_column:
                    if state.chat_session not in self.agent.user_states:
                        self.agent.user_states[state.chat_session] = {}
                    self.agent.user_states[state.chat_session]["target_column"] = state.target_column
                    print_to_log(f"🎯 Set target column: {state.target_column}")
            
            # The working agent will handle all the model building process
            # including LLM interactions, Slack updates, etc.
            
            # Create progress callback function
            def progress_callback(message: str, stage: str = "Processing"):
                try:
                    from toolbox import progress_tracker
                    if progress_tracker:
                        progress_tracker.update(state, f"{stage}: {message}")
                except Exception as e:
                    print_to_log(f"⚠️ Progress callback error: {e}")
            
            # Call the working agent's process_query method
            result = self.agent.process_query(
                query=state.user_query,
                user_id=state.chat_session,
                progress_callback=progress_callback
            )
            
            # Extract results
            if result and isinstance(result, dict):
                # Extract response message (for no data cases, error messages, etc.)
                if 'response' in result:
                    state.last_response = result['response']
                    print_to_log(f"📤 Model building response: {result['response'][:100]}...")
                
                # Extract model if built
                if 'model' in result:
                    state.trained_model = result['model']
                
                # Extract metrics if available
                if 'metrics' in result:
                    state.model_building_state = {
                        "completed": True,
                        "timestamp": datetime.now().isoformat(),
                        "method": "langgraph_interactive",
                        "metrics": result['metrics']
                    }
                
                # CRITICAL: Extract and persist target column from model building agent's internal state
                # This ensures target column is available for subsequent queries
                if state.chat_session and state.chat_session in self.agent.user_states:
                    agent_target = self.agent.user_states[state.chat_session].get("target_column")
                    if agent_target and not state.target_column:
                        state.target_column = agent_target
                        print_to_log(f"🎯 Extracted target column from model agent: {agent_target}")
                        
                        # IMMEDIATE SAVE: Persist target column to session state for future queries
                        try:
                            from pipeline_state import state_manager
                            state_manager.save_state(state)
                            print_to_log(f"💾 Persisted target column '{agent_target}' to session state")
                        except Exception as e:
                            print_to_log(f"⚠️ Could not persist target column: {e}")
                
                # CRITICAL: Clear interactive session after successful model building
                # This prevents future queries from getting stuck in preprocessing mode
                if hasattr(state, 'interactive_session') and state.interactive_session is not None:
                    print_to_log(f"🔄 Clearing interactive session after successful model building")
                    state.interactive_session = None
                    
                    # Save the cleared session state
                    try:
                        from pipeline_state import state_manager
                        state_manager.save_state(state)
                        print_to_log(f"💾 Saved cleared interactive session to prevent future conflicts")
                    except Exception as e:
                        print_to_log(f"⚠️ Could not save cleared session: {e}")
                
                # Store execution result for later file uploads (after response processing)
                execution_result = result.get('execution_result') if isinstance(result, dict) else None
            
            # Add files to pending uploads for batch processing later
            if execution_result and isinstance(execution_result, dict):
                print_to_log(f"🔍 UPLOAD DEBUG: Adding files to pending uploads...")
                print_to_log(f"🔍 UPLOAD DEBUG: Execution result keys: {list(execution_result.keys())}")
                
                # Check for artifacts structure first
                if 'artifacts' in execution_result and 'files' in execution_result['artifacts']:
                    print_to_log(f"🔍 UPLOAD DEBUG: Found artifacts files: {execution_result['artifacts']['files']}")
                    for file_info in execution_result['artifacts']['files']:
                        if isinstance(file_info, dict):
                            state.add_pending_file_upload(file_info)
                        else:
                            # Convert string path to file info dict
                            state.add_pending_file_upload({
                                "path": file_info,
                                "title": self._get_title_from_path(file_info),
                                "comment": "Generated file"
                            })
                
                # Check for direct plot_path (decision tree plots)
                elif 'plot_path' in execution_result and execution_result['plot_path']:
                    plot_path = execution_result['plot_path']
                    print_to_log(f"🔍 UPLOAD DEBUG: Found plot_path: {plot_path}")
                    state.add_pending_plot_upload(
                        plot_path, 
                        title="Decision Tree Visualization", 
                        comment="Generated decision tree plot"
                    )
                
                # Check for predictions dataset
                if 'full_predictions' in execution_result:
                    full_predictions = execution_result['full_predictions']
                    full_probabilities = execution_result.get('full_probabilities', None)
                    print_to_log(f"🔍 UPLOAD DEBUG: Found full_predictions: {len(full_predictions)} predictions")
                    if full_probabilities is not None:
                        print_to_log(f"🔍 UPLOAD DEBUG: Found full_probabilities: {len(full_probabilities)} probability arrays")
                    
                    # Add predictions and probabilities to pipeline state
                    success = state.add_predictions_to_dataset(full_predictions, "predictions", full_probabilities)
                    if success:
                        print_to_log(f"✅ Added predictions and probabilities to dataset")
                        
                        # Save predictions dataset to artifacts
                        if 'model_path' in execution_result:
                            artifacts_dir = os.path.dirname(execution_result['model_path'])
                            
                            # Extract model name from execution result
                            model_name = "unknown_model"
                            if 'model' in execution_result:
                                model = execution_result['model']
                                if hasattr(model, '__class__'):
                                    model_name = model.__class__.__name__.lower()
                                elif hasattr(model, 'name'):
                                    model_name = model.name.lower()
                            
                            # Create filename with model name
                            timestamp = int(time.time())
                            predictions_file = state.save_predictions_dataset(
                                os.path.join(artifacts_dir, f"predictions_dataset_{model_name}_{timestamp}.csv")
                            )
                            if predictions_file:
                                print_to_log(f"✅ Predictions dataset saved to: {predictions_file}")
                                
                                # Add predictions dataset to pending uploads
                                state.add_pending_file_upload({
                                    "path": predictions_file,
                                    "title": f"Dataset with {model_name.title()} Predictions & Scores",
                                    "comment": f"Complete dataset with {model_name} model predictions and probability scores"
                                })
                    else:
                        print_to_log(f"⚠️ Failed to add predictions to dataset")
                
                # Check for any other file paths in execution result
                else:
                    print_to_log(f"🔍 UPLOAD DEBUG: Searching for file paths in execution result...")
                    for key, value in execution_result.items():
                        if isinstance(value, str) and any(ext in value.lower() for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.csv', '.xlsx']):
                            if os.path.exists(value):
                                print_to_log(f"🔍 UPLOAD DEBUG: Found file via key '{key}': {value}")
                                state.add_pending_file_upload({
                                    "path": value,
                                    "title": self._get_title_from_path(value),
                                    "comment": f"Generated {key}"
                                })
                
                # Log pending uploads summary
                if hasattr(state, 'get_pending_upload_summary'):
                    print_to_log(f"📎 {state.get_pending_upload_summary()}")
            else:
                print_to_log(f"🔍 UPLOAD DEBUG: No execution result or not a dict")
            
            print_to_log("✅ Model building completed")
            return state
            
        except Exception as e:
            print_to_log(f"❌ Model building agent failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Set proper error response instead of returning unchanged state
            state.last_response = f"❌ Model building failed: {str(e)}"
            state.last_error = str(e)
            
            # Clear any existing model results to prevent confusion
            if hasattr(state, 'model_building_state'):
                state.model_building_state = None
            
            return state
    
    def _upload_files_to_slack(self, files_list, session_id):
        """Upload a list of files to Slack"""
        try:
            from toolbox import slack_manager
            for file_info in files_list:
                print_to_log(f"🔍 UPLOAD DEBUG: Processing file_info: {file_info}")
                if isinstance(file_info, dict) and 'path' in file_info:
                    file_path = file_info['path']
                    title = file_info.get('title', 'Generated File')
                    print_to_log(f"🔍 UPLOAD DEBUG: Attempting upload - Path: {file_path}, Title: {title}")
                    if os.path.exists(file_path):
                        print_to_log(f"📤 Uploading {title}: {file_path}")
                        slack_manager.upload_file(
                            session_id=session_id,
                            file_path=file_path,
                            title=title,
                            comment=f"Generated {title.lower()}"
                        )
                    else:
                        print_to_log(f"⚠️ File not found for upload: {file_path}")
                else:
                    print_to_log(f"🔍 UPLOAD DEBUG: Invalid file_info format: {file_info}")
        except Exception as e:
            print_to_log(f"❌ Failed to upload files: {e}")
            import traceback
            print_to_log(f"🔍 UPLOAD DEBUG: Full traceback: {traceback.format_exc()}")
    
    def _get_title_from_path(self, file_path):
        """Generate appropriate title from file path"""
        if not file_path:
            return "Generated File"
        
        filename = os.path.basename(file_path).lower()
        
        if 'decision_tree' in filename or 'tree' in filename:
            return "Decision Tree Plot"
        elif 'rank_order' in filename or 'rank' in filename:
            return "Rank Order Table"  
        elif 'confusion_matrix' in filename:
            return "Confusion Matrix"
        elif 'roc' in filename:
            return "ROC Curve"
        elif filename.endswith('.csv'):
            return "Data Table (CSV)"
        elif filename.endswith('.xlsx'):
            return "Data Table (Excel)"
        elif filename.endswith(('.png', '.jpg', '.jpeg')):
            return "Generated Plot"
        elif filename.endswith('.pdf'):
            return "Generated Report"
        else:
            return "Generated File"


# Global instances - these are the agents the orchestrator will use
preprocessing_agent = PreprocessingAgentWrapper()
feature_selection_agent = FeatureSelectionAgentWrapper()
model_building_agent = ModelBuildingAgentWrapper()

print_to_log("🎯 Agent wrappers initialized - using actual working implementations AS-IS")
