#!/usr/bin/env python3
"""
Minimal Wrappers for Working Agents
Uses the actual working implementations AS-IS without modification
"""

import os
import tempfile
import pandas as pd
import numpy as np
from typing import Optional, Any, Dict
from datetime import datetime

from pipeline_state import PipelineState

# Import your working agents AS-IS
try:
    from preprocessing_agent_impl import (
        run_sequential_agent as run_preprocessing_agent,
        SequentialState,
        PreprocessingPhase
    )
    # Also import the new Slack-compatible version
    from preprocessing_agent_slack import (
        create_slack_preprocessing_bot,
        SlackPreprocessingBot
    )
    PREPROCESSING_AVAILABLE = True
    print("✅ Preprocessing agent imported successfully")
except ImportError as e:
    print(f"❌ Preprocessing agent not available: {e}")
    PREPROCESSING_AVAILABLE = False

try:
    from feature_selection_agent_impl import (
        AgenticFeatureSelectionBot,
        UserSession,
        DataProcessor,
        LLMManager
    )
    FEATURE_SELECTION_AVAILABLE = True
    print("✅ Feature selection agent imported successfully")
except ImportError as e:
    print(f"❌ Feature selection agent not available: {e}")
    FEATURE_SELECTION_AVAILABLE = False

try:
    from model_building_agent_impl import LangGraphModelAgent
    MODEL_BUILDING_AVAILABLE = True
    print("✅ Model building agent imported successfully")
except ImportError as e:
    print(f"❌ Model building agent not available: {e}")
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
                print("✅ Slack preprocessing bot initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Slack preprocessing bot: {e}")
                self.available = False
        
    def run(self, state: PipelineState) -> PipelineState:
        """Route to interactive preprocessing via main Slack bot"""
        if not self.available:
            print("❌ Preprocessing agent not available - falling back to basic preprocessing")
            return self._run_basic_preprocessing_fallback(state)
            
        try:
            # Check if we have raw data
            if state.raw_data is None:
                print("❌ No raw data available for preprocessing")
                return state
                
            print(f"🚀 Launching interactive preprocessing workflow")
            print(f"🎯 Target column: {state.target_column}")
            print(f"📊 Data shape: {state.raw_data.shape}")
            
            # Send interactive preprocessing menu via main Slack bot
            from toolbox import slack_manager
            if slack_manager and state.chat_session:
                print(f"🔍 Debug: Sending Slack message to session: {state.chat_session}")
                
                if not state.target_column:
                    # Need target column first
                    initial_msg = f"""🧹 **Sequential Preprocessing Agent**

📁 **Dataset loaded:** {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns

🎯 **Please specify your target column:**
• Available columns: {', '.join(list(state.raw_data.columns)[:10])}{'...' if len(state.raw_data.columns) > 10 else ''}

📝 **How to specify:**
• Type: `target column_name` (e.g., `target default`)
• Or just: `column_name` (e.g., `default`)"""
                    
                    phase = "need_target"
                else:
                    # Show preprocessing menu
                    initial_msg = f"""🧹 **Sequential Preprocessing Agent**

📊 **Current Dataset:** {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns
🎯 **Target Column:** {state.target_column}

**🔄 Preprocessing Phases:**
• `Overview` - Dataset analysis and summary
• `Outliers` - Detect and handle outliers  
• `Missing Values` - Handle missing data
• `Encoding` - Categorical variable encoding
• `Transformations` - Feature transformations

**💬 Your Options:**
• `proceed` - Start preprocessing workflow
• `skip overview` - Skip to outlier detection
• `explain outliers` - Learn about outlier handling
• `summary` - Show current status

💬 **What would you like to do?**"""
                    
                    phase = "waiting_input"
                
                # Try to send message
                try:
                    print(f"🔍 DEBUG: About to call slack_manager.send_message")
                    print(f"🔍 DEBUG: slack_manager type: {type(slack_manager)}")
                    print(f"🔍 DEBUG: state.chat_session: {state.chat_session}")
                    print(f"🔍 DEBUG: message length: {len(initial_msg)}")
                    
                    result = slack_manager.send_message(state.chat_session, initial_msg)
                    print(f"🔍 DEBUG: send_message returned: {result}")
                    print("✅ Sent interactive preprocessing menu to Slack")
                except Exception as e:
                    print(f"❌ Failed to send Slack message: {e}")
                    print(f"🔍 Session channels: {getattr(slack_manager, 'session_channels', {})}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to basic preprocessing if Slack fails
                    return self._run_basic_preprocessing_fallback(state)
            else:
                print(f"❌ Cannot send Slack message - slack_manager: {slack_manager}, chat_session: {state.chat_session}")
                return self._run_basic_preprocessing_fallback(state)
            
            # Set up interactive session state for continuation
            state.interactive_session = {
                "agent_type": "preprocessing",
                "session_active": True,
                "session_id": state.chat_session,
                "phase": phase,
                "target_column": state.target_column,
                "current_phase": "overview"
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
            
            print("✅ Interactive preprocessing session started - user will interact via Slack")
            return state
            
        except Exception as e:
            print(f"❌ Interactive preprocessing setup failed: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Falling back to basic preprocessing")
            return self._run_basic_preprocessing_fallback(state)
    
    def _run_basic_preprocessing_fallback(self, state: PipelineState) -> PipelineState:
        """Basic preprocessing fallback that works"""
        try:
            if state.raw_data is None:
                print("❌ No raw data for basic preprocessing")
                return state
            
            df = state.raw_data.copy()
            original_shape = df.shape
            
            print(f"[PreprocessingAgent] Basic preprocessing started: {original_shape}")
            
            # Remove duplicates
            duplicates_removed = len(df) - len(df.drop_duplicates())
            df = df.drop_duplicates()
            if duplicates_removed > 0:
                print(f"  - Removed {duplicates_removed} duplicate rows")
            
            # Fill missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            
            # Fill numeric columns with median
            numeric_filled = 0
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
                    numeric_filled += 1
            
            # Fill categorical columns with mode
            categorical_filled = 0
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
                    categorical_filled += 1
            
            print(f"  - Filled missing values in {numeric_filled} numeric columns")
            print(f"  - Filled missing values in {categorical_filled} categorical columns")
            
            # Update state
            state.cleaned_data = df
            state.processed_data = df  # Also set processed_data
            state.preprocessing_state = {
                "completed": True,
                "timestamp": datetime.now().isoformat(),
                "original_shape": original_shape,
                "cleaned_shape": df.shape,
                "method": "basic_fallback"
            }
            
            print(f"[PreprocessingAgent] Basic preprocessing completed: {original_shape} → {df.shape}")
            return state
            
        except Exception as e:
            print(f"❌ Basic preprocessing failed: {e}")
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
                print("✅ Feature selection bot initialized")
            except Exception as e:
                print(f"❌ Failed to initialize feature selection bot: {e}")
                self.available = False
        
    def run(self, state: PipelineState) -> PipelineState:
        """Route to the actual working feature selection agent"""
        if not self.available or not self.bot:
            print("❌ Feature selection agent not available")
            return state
            
        try:
            if state.cleaned_data is None:
                print("❌ No cleaned data available for feature selection")
                return state
                
            # Save cleaned data to temp file
            temp_file = os.path.join(tempfile.gettempdir(), f"cleaned_data_{state.session_id}.csv")
            state.cleaned_data.to_csv(temp_file, index=False)
            
            # Create session for the working agent
            session = UserSession(
                file_path=temp_file,
                file_name=f"cleaned_data_{state.session_id}.csv",
                user_id=state.chat_session,
                target_column=state.target_column,
                original_df=state.cleaned_data.copy(),
                current_df=state.cleaned_data.copy(),
                current_features=list(state.cleaned_data.columns),
                phase="waiting_input"  # Ready for analysis
            )
            
            print(f"🚀 Launching actual feature selection agent")
            print(f"📊 Data shape: {state.cleaned_data.shape}")
            print(f"🎯 Target column: {state.target_column}")
            
            # Store session in the working bot
            self.bot.users[state.chat_session] = session
            
            # The working agent will handle all Slack interactions from here
            # It will show menus, process user input, run analyses, etc.
            
            # For now, just set up the session and let the bot handle the rest
            state.feature_selection_state = {
                "completed": False,
                "timestamp": datetime.now().isoformat(),
                "method": "agentic_interactive",
                "session_active": True
            }
            
            print("✅ Feature selection session started - bot will handle Slack interactions")
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
                
            return state
            
        except Exception as e:
            print(f"❌ Feature selection agent failed: {e}")
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
                print("✅ Model building agent initialized")
            except Exception as e:
                print(f"❌ Failed to initialize model building agent: {e}")
                self.available = False
        
    def run(self, state: PipelineState) -> PipelineState:
        """Route to the actual working model building agent"""
        if not self.available or not self.agent:
            print("❌ Model building agent not available")
            return state
            
        try:
            # Determine which data to use (cleaned > raw > None)
            data_to_use = None
            if state.cleaned_data is not None:
                data_to_use = state.cleaned_data
                print(f"🚀 Using cleaned data for model building")
            elif state.raw_data is not None:
                data_to_use = state.raw_data
                print(f"🚀 Using raw data for model building (preprocessing skipped)")
            else:
                print("❌ No data available - letting model building agent handle this")
                # Let the actual agent handle "no data" case with proper messaging
                
            # Determine features to use (selected > all columns)
            features_to_use = None
            if state.selected_features:
                features_to_use = state.selected_features
                print(f"🎯 Using selected features: {len(state.selected_features)}")
            elif data_to_use is not None:
                # Use all columns except target as features
                all_cols = list(data_to_use.columns)
                if state.target_column and state.target_column in all_cols:
                    features_to_use = [col for col in all_cols if col != state.target_column]
                else:
                    features_to_use = all_cols
                print(f"🎯 Using all available features: {len(features_to_use)} (feature selection skipped)")
            
            if data_to_use is not None:
                print(f"📊 Data shape: {data_to_use.shape}")
            
            print(f"🚀 Launching actual model building agent")
            
            # Load data into the agent if available
            if data_to_use is not None:
                print(f"📊 Loading data into model building agent")
                self.agent.load_data(data_to_use, state.chat_session)
                
                # Set target column if available
                if state.target_column:
                    if state.chat_session not in self.agent.user_states:
                        self.agent.user_states[state.chat_session] = {}
                    self.agent.user_states[state.chat_session]["target_column"] = state.target_column
                    print(f"🎯 Set target column: {state.target_column}")
            
            # The working agent will handle all the model building process
            # including LLM interactions, Slack updates, etc.
            
            # Create progress callback function
            def progress_callback(message: str, stage: str = "Processing"):
                try:
                    from toolbox import progress_tracker
                    if progress_tracker:
                        progress_tracker.update(state, f"{stage}: {message}")
                except Exception as e:
                    print(f"⚠️ Progress callback error: {e}")
            
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
                    print(f"📤 Model building response: {result['response'][:100]}...")
                
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
                
                # Handle file uploads (plots, etc.) - Check execution_result
                print(f"🔍 UPLOAD DEBUG: Checking for artifacts in result...")
                print(f"🔍 UPLOAD DEBUG: Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                
                # The actual execution result is in result['execution_result']
                execution_result = result.get('execution_result') if isinstance(result, dict) else None
                print(f"🔍 UPLOAD DEBUG: Execution result type: {type(execution_result)}")
                
                if execution_result and isinstance(execution_result, dict):
                    print(f"🔍 UPLOAD DEBUG: Execution result keys: {list(execution_result.keys())}")
                    
                    # Check for artifacts structure first
                    if 'artifacts' in execution_result:
                        print(f"🔍 UPLOAD DEBUG: Found artifacts: {execution_result['artifacts']}")
                        if 'files' in execution_result['artifacts']:
                            print(f"🔍 UPLOAD DEBUG: Found files: {execution_result['artifacts']['files']}")
                            self._upload_files_to_slack(execution_result['artifacts']['files'], state.chat_session)
                        else:
                            print(f"🔍 UPLOAD DEBUG: No 'files' key in artifacts")
                    
                    # Check for direct plot_path (from logs we see this exists)
                    elif 'plot_path' in execution_result and execution_result['plot_path']:
                        plot_path = execution_result['plot_path']
                        print(f"🔍 UPLOAD DEBUG: Found plot_path: {plot_path}")
                        if os.path.exists(plot_path):
                            try:
                                from toolbox import slack_manager
                                print(f"📤 Uploading decision tree plot: {plot_path}")
                                slack_manager.upload_file(
                                    session_id=state.chat_session,
                                    file_path=plot_path,
                                    title="Decision Tree Visualization",
                                    comment="Generated decision tree plot"
                                )
                                print(f"✅ Successfully uploaded plot: {plot_path}")
                            except Exception as e:
                                print(f"❌ Failed to upload plot: {e}")
                                import traceback
                                print(f"🔍 UPLOAD DEBUG: Full traceback: {traceback.format_exc()}")
                        else:
                            print(f"⚠️ Plot file not found: {plot_path}")
                    
                    # Check for any other file paths in execution result
                    else:
                        print(f"🔍 UPLOAD DEBUG: Searching for file paths in execution result...")
                        file_found = False
                        for key, value in execution_result.items():
                            if isinstance(value, str) and any(ext in value.lower() for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.csv', '.xlsx']):
                                if os.path.exists(value):
                                    print(f"🔍 UPLOAD DEBUG: Found file via key '{key}': {value}")
                                    try:
                                        from toolbox import slack_manager
                                        title = self._get_title_from_path(value)
                                        print(f"📤 Uploading {title}: {value}")
                                        slack_manager.upload_file(
                                            session_id=state.chat_session,
                                            file_path=value,
                                            title=title,
                                            comment=f"Generated {title.lower()}"
                                        )
                                        file_found = True
                                        break
                                    except Exception as e:
                                        print(f"❌ Failed to upload file: {e}")
                        
                        if not file_found:
                            print(f"🔍 UPLOAD DEBUG: No file artifacts found in execution result")
                else:
                    print(f"🔍 UPLOAD DEBUG: No execution result or not a dict")
            
            print("✅ Model building completed")
            return state
            
        except Exception as e:
            print(f"❌ Model building agent failed: {e}")
            import traceback
            traceback.print_exc()
            return state
    
    def _upload_files_to_slack(self, files_list, session_id):
        """Upload a list of files to Slack"""
        try:
            from toolbox import slack_manager
            for file_info in files_list:
                print(f"🔍 UPLOAD DEBUG: Processing file_info: {file_info}")
                if isinstance(file_info, dict) and 'path' in file_info:
                    file_path = file_info['path']
                    title = file_info.get('title', 'Generated File')
                    print(f"🔍 UPLOAD DEBUG: Attempting upload - Path: {file_path}, Title: {title}")
                    if os.path.exists(file_path):
                        print(f"📤 Uploading {title}: {file_path}")
                        slack_manager.upload_file(
                            session_id=session_id,
                            file_path=file_path,
                            title=title,
                            comment=f"Generated {title.lower()}"
                        )
                    else:
                        print(f"⚠️ File not found for upload: {file_path}")
                else:
                    print(f"🔍 UPLOAD DEBUG: Invalid file_info format: {file_info}")
        except Exception as e:
            print(f"❌ Failed to upload files: {e}")
            import traceback
            print(f"🔍 UPLOAD DEBUG: Full traceback: {traceback.format_exc()}")
    
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

print("🎯 Agent wrappers initialized - using actual working implementations AS-IS")
