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
            print(f"🔧 DEBUG: Target column: {state.target_column}")
            print(f"🔧 DEBUG: Target column type: {type(state.target_column)}")
            print(f"🔧 DEBUG: Target column is None: {state.target_column is None}")
            print(f"🔧 DEBUG: Target column is empty string: {state.target_column == ''}")
            print(f"📊 Data shape: {state.raw_data.shape}")
            print(f"🔧 DEBUG: Available columns: {list(state.raw_data.columns)}")
            
            # Send interactive preprocessing menu via main Slack bot
            # Use the pipeline's slack_manager instead of the global one
            slack_manager = getattr(state, '_slack_manager', None)
            if not slack_manager:
                from toolbox import slack_manager as global_slack_manager
                slack_manager = global_slack_manager
            
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
    
    def handle_interactive_command(self, state: PipelineState, command: str) -> PipelineState:
        """Handle interactive commands for preprocessing"""
        if not self.available:
            print("❌ Preprocessing agent not available")
            return state
            
        try:
            # Check current phase and handle accordingly
            current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
            print(f"🔧 DEBUG: Current phase: {current_phase}, Command: {command}")
            
            # Handle the command using the preprocessing agent's interactive system
            if command.lower() == 'proceed':
                print("🚀 Starting preprocessing workflow with outliers phase")
                
                # Create a temporary file path for the DataFrame
                import tempfile
                import os
                
                # Create a temporary CSV file for the DataFrame
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                    state.raw_data.to_csv(tmp_file.name, index=False)
                    df_path = tmp_file.name
                
                try:
                    # Start the actual preprocessing workflow
                    print("🔧 Running preprocessing agent for outlier analysis...")
                    
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
                        model_name=os.environ.get("DEFAULT_MODEL", "gpt-4o")
                    )
                    
                    # Initialize dataset analysis
                    print("📊 Initializing dataset analysis...")
                    sequential_state = initialize_dataset_analysis(sequential_state)
                    
                    # Run outlier detection with confidence-based approach
                    print("🔍 Running confidence-based outlier detection...")
                    outlier_results = analyze_outliers_with_confidence(sequential_state)
                    
                    # Debug: Check what we got back
                    print(f"🔍 DEBUG: outlier_results type: {type(outlier_results)}")
                    print(f"🔍 DEBUG: outlier_results content: {outlier_results}")
                    
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
                        print(f"⚠️ Unexpected outlier_results type: {type(outlier_results)}")
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
                                    strategy_summary.append(f"**Keep as-is**: {count} columns")
                                elif strategy == 'winsorize':
                                    strategy_summary.append(f"**Winsorize**: {count} columns")
                                elif strategy == 'remove':
                                    strategy_summary.append(f"**Remove**: {count} columns")
                                else:
                                    strategy_summary.append(f"**{strategy.title()}**: {count} columns")
                            
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
                            
                            message = f"""🔍 **Outlier Analysis Complete!**

📊 **Dataset Overview:**
• Total rows: {state.raw_data.shape[0]:,}
• Total columns: {state.raw_data.shape[1]}
• Target column: {state.target_column}

🎯 **Analysis Results:**
• Columns analyzed: {analyzed_columns}
• Columns needing treatment: {columns_needing_treatment}

**🔧 Recommended Treatments:**
{outlier_details}

**💬 Next Steps:**
• `continue` - Apply recommendations and move to missing values
• `skip outliers` - Move to missing values analysis
• `summary` - Show current preprocessing status"""
                        else:
                            # Old format
                            message = f"""🔍 **Outlier Analysis Complete!**

📊 **Dataset Overview:**
• Total rows: {state.raw_data.shape[0]:,}
• Total columns: {state.raw_data.shape[1]}
• Target column: {state.target_column}

🎯 **Outlier Detection Results:**
• Columns with outliers: {len(outlier_columns)}
• Total outliers found: {total_outliers:,}

**📋 Columns with Outliers:**
{outlier_details}{'...' if len(outlier_columns) > 5 else ''}

**💬 Next Steps:**
• `continue` - Apply recommendations and move to missing values
• `skip outliers` - Move to missing values analysis
• `summary` - Show current preprocessing status"""
                        
                        slack_manager.send_message(state.chat_session, message)
                    
                    # Update state
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
                print("🔄 Applying LLM recommendations and moving to next phase")
                # Get current phase from either preprocessing_state or interactive_session
                if state.preprocessing_state and 'current_phase' in state.preprocessing_state:
                    current_phase = state.preprocessing_state.get('current_phase')
                elif state.interactive_session and 'current_phase' in state.interactive_session:
                    current_phase = state.interactive_session.get('current_phase')
                else:
                    current_phase = 'overview'  # default
                print(f"🔧 DEBUG: Current phase for continue: {current_phase}")

                if current_phase == 'overview':
                    # Start preprocessing workflow - begin with outliers phase
                    print("🚀 Starting preprocessing workflow with outliers phase")
                    print("🔧 Running preprocessing agent for outlier analysis...")
                    
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
                        print(f"🔧 DEBUG: Target column: {target_column}")
                        print(f"🔧 DEBUG: Target column type: {type(target_column)}")
                        print(f"🔧 DEBUG: Target column is None: {target_column is None}")
                        print(f"🔧 DEBUG: Target column is empty string: {target_column == ''}")
                        
                        # Run outlier analysis
                        print("🔍 Running outlier detection...")
                        
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
                            model_name=os.environ.get("DEFAULT_MODEL", "gpt-4o")
                        )
                        
                        # Initialize dataset analysis
                        print("📊 Initializing dataset analysis...")
                        sequential_state = initialize_dataset_analysis(sequential_state)
                        
                        # Run outlier analysis with confidence-based approach
                        print("🔍 Running confidence-based outlier detection...")
                        outlier_results = analyze_outliers_with_confidence(sequential_state)
                        
                        print(f"🔍 DEBUG: outlier_results type: {type(outlier_results)}")
                        print(f"🔍 DEBUG: outlier_results content: {outlier_results}")
                        
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
                                    
                                    treatment_text.append(f"**{treatment_display}:** {col_text}")
                                
                                message = f"""🚨 **Outliers Analysis Complete!**

**📊 Outlier Columns Found:** {len(outlier_results.get('outliers_columns', []))} columns

**🔧 Recommended Treatments:**
{chr(10).join(treatment_text)}

**🔄 Ready for Next Step:**
• `continue` - Apply treatments and move to missing values
• `skip outliers` - Skip to missing values phase  
• `summary` - Show current preprocessing status"""
                            else:
                                message = f"""🚨 **Outliers Analysis Complete!**

**📊 Analysis Results:**
{outlier_results}

**🔄 Ready for Next Step:**
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
                        print(f"❌ Outlier analysis failed: {e}")
                        import traceback
                        traceback.print_exc()
                        return state

                elif current_phase == 'outliers':
                    # Apply outlier treatments and move to missing_values
                    print("🔧 Applying outlier treatments...")
                    
                    # Get outlier results from state
                    outlier_results = state.preprocessing_state.get('outlier_results', {})
                    if not outlier_results:
                        print("❌ No outlier results found in state")
                        return state
                    
                    # Apply treatments based on LLM recommendations
                    df = state.raw_data.copy()
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
                        applied_treatments.append(f"**{treatment}**: {count} columns")
                    
                    # Update state with processed data
                    state.cleaned_data = df
                    print(f"🔧 DEBUG: Set cleaned_data shape: {df.shape}")
                    
                    # Send confirmation message
                    slack_manager = getattr(state, '_slack_manager', None)
                    if not slack_manager:
                        from toolbox import slack_manager as global_slack_manager
                        slack_manager = global_slack_manager
                    
                    if slack_manager and state.chat_session:
                        treatments_text = "\n".join(applied_treatments) if applied_treatments else "• No treatments applied"
                        
                        message = f"""✅ **Outlier Treatments Applied!**

**🔧 Applied Treatments:**
{treatments_text}

**📊 Data Summary:**
• Original: {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns
• Processed: {df.shape[0]:,} rows × {df.shape[1]} columns
• Rows removed: {state.raw_data.shape[0] - df.shape[0]:,}

**🔄 Moving to Next Phase: Missing Values Analysis**

**💬 Next Steps:**
• `continue` - Start missing values analysis
• `skip missing` - Move to encoding phase
• `summary` - Show current status"""
                        
                        slack_manager.send_message(state.chat_session, message)
                    
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
                        print("🔧 Applying missing values treatments...")
                        
                        df = state.cleaned_data.copy() if state.cleaned_data is not None else state.raw_data.copy()
                        applied_treatments = []
                        
                        if isinstance(missing_results, dict) and 'llm_recommendations' in missing_results:
                            for col, recommendation in missing_results['llm_recommendations'].items():
                                raw_strategy = recommendation.get('strategy', 'median')
                                strategy = str(raw_strategy).lower().replace('-', '_')
                                if strategy == 'median':
                                    df[col] = df[col].fillna(df[col].median())
                                    applied_treatments.append(f"• {col}: Filled with median")
                                elif strategy == 'mean':
                                    df[col] = df[col].fillna(df[col].mean())
                                    applied_treatments.append(f"• {col}: Filled with mean")
                                elif strategy == 'mode':
                                    mode_val = df[col].mode()
                                    fill_val = mode_val.iloc[0] if not mode_val.empty else df[col].dropna().iloc[0] if df[col].dropna().shape[0] else 0
                                    df[col] = df[col].fillna(fill_val)
                                    applied_treatments.append(f"• {col}: Filled with mode")
                                elif strategy == 'constant':
                                    constant_value = recommendation.get('constant_value', 0)
                                    df[col] = df[col].fillna(constant_value)
                                    applied_treatments.append(f"• {col}: Filled with constant ({constant_value})")
                                elif strategy == 'drop_column':
                                    if col in df.columns:
                                        df = df.drop(columns=[col])
                                        applied_treatments.append(f"• {col}: Dropped due to high missing%")
                                elif strategy == 'keep_missing':
                                    # Leave NaNs; optionally add indicator
                                    indicator_col = f"{col}_was_missing"
                                    df[indicator_col] = df[col].isna().astype(int)
                                    applied_treatments.append(f"• {col}: Kept missing (added indicator)")
                                elif strategy == 'model_based':
                                    # Placeholder: fall back to median/most_frequent depending on dtype
                                    if pd.api.types.is_numeric_dtype(df[col]):
                                        df[col] = df[col].fillna(df[col].median())
                                        applied_treatments.append(f"• {col}: Model-based (fallback median)")
                                    else:
                                        mode_val = df[col].mode()
                                        fill_val = mode_val.iloc[0] if not mode_val.empty else df[col].dropna().iloc[0] if df[col].dropna().shape[0] else ''
                                        df[col] = df[col].fillna(fill_val)
                                        applied_treatments.append(f"• {col}: Model-based (fallback mode)")

                        # Update state with processed data
                        state.cleaned_data = df
                        print(f"🔧 DEBUG: Set cleaned_data shape after missing values: {df.shape}")
                        
                        # Send confirmation message
                        slack_manager = getattr(state, '_slack_manager', None)
                        if not slack_manager:
                            from toolbox import slack_manager as global_slack_manager
                            slack_manager = global_slack_manager
                        
                        if slack_manager and state.chat_session:
                            treatments_text = "\n".join(applied_treatments) if applied_treatments else "• No treatments applied"
                            
                            message = f"""✅ **Missing Values Treatments Applied!**

**🔧 Applied Treatments:**
{treatments_text}

**📊 Data Summary:**
• Processed: {df.shape[0]:,} rows × {df.shape[1]} columns
• Missing values filled: {len(applied_treatments)} columns

**🔄 Moving to Next Phase: Encoding Analysis**

**💬 Next Steps:**
• `continue` - Start encoding analysis
• `skip encoding` - Move to transformations phase
• `summary` - Show current status"""
                            
                            slack_manager.send_message(state.chat_session, message)
                        
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
                        print("🔍 Starting missing values analysis...")
                        
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
                                model_name=os.environ.get("DEFAULT_MODEL", "gpt-4o")
                            )
                            
                            # Run missing values analysis
                            print("🔍 Running confidence-based missing values analysis...")
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
                                                strategy_summary.append(f"**Drop columns**: {count} columns")
                                            elif strategy == 'drop_missing':
                                                strategy_summary.append(f"**Drop missing rows**: {count} columns")
                                            else:
                                                strategy_summary.append(f"**{strategy.title()} imputation**: {count} columns")
                                        
                                        strategy_text = "\n".join(strategy_summary)
                                        
                                        # Confidence stats removed from display per user request
                                        
                                        message = f"""🔍 **Missing Values Analysis Complete!**

**📊 Missing Values Found:** {len(missing_columns)} columns

**🔧 Recommended Strategies:**
{strategy_text}

**🔄 Ready for Next Step:**
• `continue` - Apply missing values treatments and move to encoding
• `skip missing` - Move directly to encoding phase
• `summary` - Show current preprocessing status"""
                                    else:
                                        message = f"""🔍 **Missing Values Analysis Complete!**

**📊 No missing values found** - Dataset is complete!

**🔄 Ready for Next Step:**
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
                                        strategy_summary.append(f"**{strategy.title()} imputation:** {cols_str}")
                                    
                                    strategy_text = "\n".join(strategy_summary)
                                    
                                    message = f"""🔍 **Missing Values Analysis Complete!**

**📊 Missing Values Found:** {len(missing_columns)} columns

**🔧 Recommended Strategies:**
{strategy_text}

**🔄 Ready for Next Step:**
• `continue` - Apply missing values treatments and move to encoding
• `skip missing` - Move directly to encoding phase
• `summary` - Show current preprocessing status"""
                                else:
                                    # Fallback - avoid showing raw JSON
                                    message = f"""🔍 **Missing Values Analysis Complete!**

**📊 Analysis completed successfully**

**🔄 Ready for Next Step:**
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
                            print(f"❌ Missing values analysis failed: {e}")
                            import traceback
                            traceback.print_exc()
                            return state

                elif current_phase == 'encoding':
                    # Check if we already have encoding results
                    encoding_results = state.preprocessing_state.get('encoding_results')
                    if encoding_results:
                        # Apply encoding treatments and move to transformations
                        print("🔧 Applying encoding treatments...")
                        
                        df = state.cleaned_data.copy() if state.cleaned_data is not None else state.raw_data.copy()
                        applied_treatments = []
                        
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
                                else:
                                    enc_choice = 'label'
                                
                                if enc_choice == 'label':
                                    from sklearn.preprocessing import LabelEncoder
                                    le = LabelEncoder()
                                    df[col] = le.fit_transform(df[col].astype(str))
                                    applied_treatments.append(f"• {col}: Label encoded")
                                elif enc_choice == 'onehot':
                                    # Apply one-hot encoding
                                    df = pd.get_dummies(df, columns=[col], prefix=col)
                                    applied_treatments.append(f"• {col}: One-hot encoded")
                                elif enc_choice == 'ordinal':
                                    # Apply ordinal encoding
                                    unique_values = df[col].astype(str).unique()
                                    value_map = {val: idx for idx, val in enumerate(unique_values)}
                                    df[col] = df[col].astype(str).map(value_map)
                                    applied_treatments.append(f"• {col}: Ordinal encoded")
                                elif enc_choice == 'target':
                                    # Simple target mean encoding
                                    try:
                                        target_col = state.target_column
                                        if target_col and target_col in df.columns:
                                            means = df.groupby(col)[target_col].mean()
                                            df[col] = df[col].map(means)
                                            applied_treatments.append(f"• {col}: Target mean encoded")
                                        else:
                                            # Fallback to label if target not available
                                            from sklearn.preprocessing import LabelEncoder
                                            le = LabelEncoder()
                                            df[col] = le.fit_transform(df[col].astype(str))
                                            applied_treatments.append(f"• {col}: Label encoded (no target)")
                                    except Exception:
                                        from sklearn.preprocessing import LabelEncoder
                                        le = LabelEncoder()
                                        df[col] = le.fit_transform(df[col].astype(str))
                                        applied_treatments.append(f"• {col}: Label encoded (fallback)")
                                elif enc_choice == 'binary':
                                    # Fallback: treat as one-hot
                                    df = pd.get_dummies(df, columns=[col], prefix=col)
                                    applied_treatments.append(f"• {col}: One-hot encoded (binary fallback)")
                                elif enc_choice == 'drop_column':
                                    if col in df.columns:
                                        df = df.drop(columns=[col])
                                        applied_treatments.append(f"• {col}: Dropped due to high missing%")

                        # Update state with processed data
                        state.cleaned_data = df
                        print(f"🔧 DEBUG: Set cleaned_data shape after encoding: {df.shape}")
                        
                        # Send confirmation message
                        slack_manager = getattr(state, '_slack_manager', None)
                        if not slack_manager:
                            from toolbox import slack_manager as global_slack_manager
                            slack_manager = global_slack_manager
                        
                        if slack_manager and state.chat_session:
                            treatments_text = "\n".join(applied_treatments) if applied_treatments else "• No treatments applied"
                            
                            message = f"""✅ **Encoding Treatments Applied!**

**🔧 Applied Treatments:**
{treatments_text}

**📊 Data Summary:**
• Processed: {df.shape[0]:,} rows × {df.shape[1]} columns
• Encodings applied: {len(applied_treatments)} columns

**🔄 Moving to Next Phase: Transformations Analysis**

**💬 Next Steps:**
• `continue` - Start transformations analysis
• `skip transformations` - Complete preprocessing
• `summary` - Show current status"""
                            
                            slack_manager.send_message(state.chat_session, message)
                        
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
                        print("🔍 Starting encoding analysis...")
                        
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
                                model_name=os.environ.get("DEFAULT_MODEL", "gpt-4o")
                            )
                            
                            # Run encoding analysis with confidence-based approach
                            print("🔍 Running confidence-based encoding analysis...")
                            encoding_results = analyze_encoding_with_confidence(sequential_state)
                            
                            print(f"🔍 DEBUG: encoding_results type: {type(encoding_results)}")
                            print(f"🔍 DEBUG: encoding_results content: {encoding_results}")
                            
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
                                                strategy_summary.append(f"**Label encoding**: {count} columns")
                                            elif strategy == 'onehot_encoding':
                                                strategy_summary.append(f"**One-hot encoding**: {count} columns")
                                            elif strategy == 'target_encoding':
                                                strategy_summary.append(f"**Target encoding**: {count} columns")
                                            elif strategy == 'binary_encoding':
                                                strategy_summary.append(f"**Binary encoding**: {count} columns")
                                            else:
                                                strategy_summary.append(f"**{strategy.replace('_', ' ').title()}**: {count} columns")
                                        
                                        strategy_text = "\n".join(strategy_summary)
                                        
                                        # Confidence stats removed from display per user request
                                        
                                        message = f"""🔍 **Encoding Analysis Complete!**

**📊 Categorical Columns Found:** {len(encoding_columns)} columns

**🔧 Recommended Strategies:**
{strategy_text}

**🔄 Ready for Next Step:**
• `continue` - Apply encoding treatments and move to transformations
• `skip encoding` - Move directly to transformations phase
• `summary` - Show current preprocessing status"""
                                    else:
                                        message = f"""🔍 **Encoding Analysis Complete!**

**📊 No categorical columns found** - All columns are numeric!

**🔄 Ready for Next Step:**
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
                                        encoding_summary.append(f"**{encoding_type} encoding**: {len(cols)} columns")
                                    
                                    encoding_text = "\n".join(encoding_summary)
                                    
                                    message = f"""🔍 **Encoding Analysis Complete!**

**📊 Categorical Columns Found:** {len(categorical_columns)} columns

**🔧 Recommended Encoding:**
{encoding_text}

**🔄 Ready for Next Step:**
• `continue` - Apply encoding treatments and move to transformations
• `skip encoding` - Move directly to transformations phase
• `summary` - Show current preprocessing status"""
                                else:
                                    # Fallback - avoid showing raw JSON
                                    message = f"""🔍 **Encoding Analysis Complete!**

**📊 Analysis completed successfully**

**🔄 Ready for Next Step:**
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
                            print(f"❌ Encoding analysis failed: {e}")
                            import traceback
                            traceback.print_exc()
                            return state

                elif current_phase == 'transformations':
                    # Check if we already have transformation results
                    transformation_results = state.preprocessing_state.get('transformation_results')
                    if transformation_results:
                        # Apply transformation treatments and complete preprocessing
                        print("🔧 Applying transformation treatments...")
                        
                        df = state.cleaned_data.copy() if state.cleaned_data is not None else state.raw_data.copy()
                        applied_treatments = []
                        
                        if isinstance(transformation_results, dict) and 'llm_recommendations' in transformation_results:
                            for col, recommendation in transformation_results['llm_recommendations'].items():
                                raw_t = recommendation.get('transformation_type') or recommendation.get('transformation') or 'none'
                                t = str(raw_t).lower().replace('-', '_')
                                if t in ['log', 'log1p']:
                                    # Apply log1p for numerical stability
                                    df[col] = np.log1p(df[col])
                                    applied_treatments.append(f"• {col}: Log1p transformation applied")
                                elif t == 'sqrt':
                                    df[col] = np.sqrt(df[col].clip(lower=0))
                                    applied_treatments.append(f"• {col}: Square root transformation applied")
                                elif t in ['box_cox', 'boxcox']:
                                    from scipy.stats import boxcox
                                    # Shift if needed to ensure positivity
                                    shift = 0
                                    if (df[col] <= 0).any():
                                        shift = abs(df[col].min()) + 1
                                    df[col], _ = boxcox(df[col] + shift)
                                    applied_treatments.append(f"• {col}: Box-Cox transformation applied")
                                elif t in ['yeo_johnson', 'yeojohnson']:
                                    from sklearn.preprocessing import PowerTransformer
                                    pt = PowerTransformer(method='yeo-johnson')
                                    df[col] = pt.fit_transform(df[[col]])
                                    applied_treatments.append(f"• {col}: Yeo-Johnson transformation applied")
                                elif t in ['standardize', 'standard_scaler', 'zscore']:
                                    from sklearn.preprocessing import StandardScaler
                                    scaler = StandardScaler()
                                    df[col] = scaler.fit_transform(df[[col]])
                                    applied_treatments.append(f"• {col}: Standardized")
                                elif t in ['robust_scale', 'robust_scaler']:
                                    from sklearn.preprocessing import RobustScaler
                                    scaler = RobustScaler()
                                    df[col] = scaler.fit_transform(df[[col]])
                                    applied_treatments.append(f"• {col}: Robust scaled")
                                elif t in ['quantile', 'quantile_transform']:
                                    from sklearn.preprocessing import QuantileTransformer
                                    qt = QuantileTransformer(output_distribution='normal', random_state=0)
                                    df[col] = qt.fit_transform(df[[col]])
                                    applied_treatments.append(f"• {col}: Quantile transformed")
                                elif t in ['normalize', 'minmax', 'minmax_scaler']:
                                    from sklearn.preprocessing import MinMaxScaler
                                    scaler = MinMaxScaler()
                                    df[col] = scaler.fit_transform(df[[col]])
                                    applied_treatments.append(f"• {col}: MinMax normalized")
                                elif t in ['none', 'keep', 'no_transform']:
                                    # Explicit no-op
                                    applied_treatments.append(f"• {col}: Kept as-is")

                        # Update state with processed data
                        state.cleaned_data = df
                        print(f"🔧 DEBUG: Set cleaned_data shape after transformations: {df.shape}")
                        
                        # Send confirmation message
                        slack_manager = getattr(state, '_slack_manager', None)
                        if not slack_manager:
                            from toolbox import slack_manager as global_slack_manager
                            slack_manager = global_slack_manager
                        
                        if slack_manager and state.chat_session:
                            treatments_text = "\n".join(applied_treatments) if applied_treatments else "• No treatments applied"
                            
                            message = f"""✅ **Transformations Applied!**

**🔧 Applied Treatments:**
{treatments_text}

**📊 Data Summary:**
• Final processed: {df.shape[0]:,} rows × {df.shape[1]} columns
• Transformations applied: {len(applied_treatments)} columns

**🎉 Preprocessing Complete!**

**💬 Next Steps:**
• `summary` - Show complete preprocessing summary
• `feature_selection` - Move to feature selection phase
• `model_building` - Move to model building phase"""
                            
                            slack_manager.send_message(state.chat_session, message)
                        
                        # Update state for completion
                        state.preprocessing_state.update({
                            "current_phase": "completion",
                            "status": "preprocessing_complete",
                            "transformation_treatments_applied": applied_treatments
                        })
                        
                        if state.interactive_session:
                            state.interactive_session["current_phase"] = "completion"
                            state.interactive_session["phase"] = "complete"
                        return state
                    else:
                        # Start transformations analysis
                        print("🔍 Starting transformations analysis...")
                        
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
                                model_name=os.environ.get("DEFAULT_MODEL", "gpt-4o")
                            )
                            
                            # Run transformations analysis with confidence-based approach
                            print("🔍 Running confidence-based transformations analysis...")
                            transformation_results = analyze_transformations_with_confidence(sequential_state)
                            
                            print(f"🔍 DEBUG: transformation_results type: {type(transformation_results)}")
                            print(f"🔍 DEBUG: transformation_results content: {transformation_results}")
                            
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
                                                strategy_summary.append(f"**No transformation needed**: {count} columns")
                                            elif strategy == 'log':
                                                strategy_summary.append(f"**Log transformation**: {count} columns")
                                            elif strategy == 'log1p':
                                                strategy_summary.append(f"**Log1p transformation**: {count} columns")
                                            elif strategy == 'sqrt':
                                                strategy_summary.append(f"**Square root transformation**: {count} columns")
                                            elif strategy == 'box_cox':
                                                strategy_summary.append(f"**Box-Cox transformation**: {count} columns")
                                            elif strategy == 'yeo_johnson':
                                                strategy_summary.append(f"**Yeo-Johnson transformation**: {count} columns")
                                            elif strategy == 'quantile':
                                                strategy_summary.append(f"**Quantile transformation**: {count} columns")
                                            else:
                                                strategy_summary.append(f"**{strategy.replace('_', ' ').title()} transformation**: {count} columns")
                                        
                                        strategy_text = "\n".join(strategy_summary)
                                        
                                        # Confidence stats removed from display per user request
                                        
                                        message = f"""🔍 **Transformations Analysis Complete!**

**📊 Numerical Columns Analyzed:** {len(transformations_columns)} columns

**🔧 Recommended Strategies:**
{strategy_text}

**🔄 Ready for Next Step:**
• `continue` - Apply transformations and complete preprocessing
• `skip transformations` - Complete preprocessing without transformations
• `summary` - Show current preprocessing status"""
                                    else:
                                        message = f"""🔍 **Transformations Analysis Complete!**

**📊 No transformations needed** - All columns are well-distributed!

**🔄 Ready for Next Step:**
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
                                            raw_t = llm_recommendations[col].get('transformation_type') or llm_recommendations[col].get('transformation') or 'none'
                                            t_norm = str(raw_t).lower().replace('-', '_')
                                            # Friendly label mapping
                                            if t_norm in ['log', 'log1p']:
                                                key = 'Log1p'
                                            elif t_norm == 'sqrt':
                                                key = 'Square root'
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
                                        transformation_summary.append(f"**{transformation_type}**: {len(cols)} columns")
                                    
                                    transformation_text = "\n".join(transformation_summary)
                                    
                                    message = f"""🔍 **Transformations Analysis Complete!**

**📊 Numerical Columns Analyzed:** {len(numerical_columns)} columns

**🔧 Recommended Transformations:**
{transformation_text}

**🔄 Ready for Next Step:**
• `continue` - Apply transformations and complete preprocessing
• `skip transformations` - Complete preprocessing without transformations
• `summary` - Show current preprocessing status"""
                                else:
                                    # Fallback - avoid showing raw JSON
                                    message = f"""🔍 **Transformations Analysis Complete!**

**📊 Analysis completed successfully**

**🔄 Ready for Next Step:**
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
                            print(f"❌ Transformations analysis failed: {e}")
                            import traceback
                            traceback.print_exc()
                            return state

                else:
                    print(f"❌ Unknown phase for continue command: {current_phase}")
                    return state
            
            # Handle BGE-classified queries with clear intent signal (Level 4 BGE result)
            elif command.startswith('QUERY: '):
                # Extract the actual query from the intent signal
                actual_query = command[7:]  # Remove 'QUERY: ' prefix
                print("🔍 Processing BGE-classified query with enhanced LLM...")
                print(f"🔍 DEBUG: BGE classified query: '{actual_query}'")
                
                try:
                    # Initialize LLM using the same pattern as preprocessing strategies
                    from preprocessing_agent_impl import get_llm_from_state, SequentialState
                    import tempfile
                    import os
                    
                    print("🔍 DEBUG: Importing required modules...")
                    
                    # Use the data from state for analysis context
                    data_to_analyze = state.cleaned_data if hasattr(state, 'cleaned_data') and state.cleaned_data is not None else state.raw_data
                    print(f"🔍 DEBUG: Using data for context - shape: {data_to_analyze.shape}")
                    
                    # Create temporary file for LLM processing
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                        data_to_analyze.to_csv(tmp_file.name, index=False)
                        df_path = tmp_file.name
                        print(f"🔍 DEBUG: Created temp file: {df_path}")
                    
                    try:
                        # Create SequentialState for LLM processing
                        print("🔍 DEBUG: Creating SequentialState for LLM...")
                        sequential_state = SequentialState(
                            df=data_to_analyze,
                            df_path=df_path,
                            target_column=state.target_column,
                            model_name=os.environ.get("DEFAULT_MODEL", "gpt-4o")
                        )
                        
                        # Initialize LLM
                        print("🔍 DEBUG: Initializing LLM...")
                        llm = get_llm_from_state(sequential_state)
                        print(f"🔍 DEBUG: LLM initialized successfully")
                        
                        # Analyze query context and generate response
                        print("🔍 DEBUG: Analyzing query context...")
                        query_analysis = self._analyze_query_context(actual_query, state)
                        query_type = query_analysis['type']
                        context_data = query_analysis['context']
                        
                        print(f"🔍 DEBUG: Query analysis - Type: {query_type}, Context length: {len(str(context_data))}")
                        
                        # Generate appropriate prompt based on query type
                        if query_type == 'general':
                            print("🔍 DEBUG: Creating general query prompt...")
                            prompt = f"""You are a data preprocessing expert. Answer the user's general question about data preprocessing concepts and methods.

QUESTION: "{actual_query}"

Provide a clear, informative explanation about the preprocessing concept or method being asked about.
"""
                        elif query_type == 'column_specific':
                            column = query_analysis.get('column', 'unknown')
                            print(f"🔍 DEBUG: Creating column-specific query prompt for column: {column}")
                            prompt = f"""You are a data preprocessing expert. Answer the user's question about a specific column and its preprocessing strategy.

QUESTION: "{actual_query}"
TARGET COLUMN: {state.target_column}
COLUMN OF INTEREST: {column}

COLUMN ANALYSIS AND RECOMMENDATIONS:
{context_data}

Explain the preprocessing strategy for this column based on the analysis data and reasoning provided.
"""
                        elif query_type == 'comparative':
                            print("🔍 DEBUG: Creating comparative query prompt...")
                            prompt = f"""You are a data preprocessing expert. Answer the user's comparative question about multiple columns or strategies.

QUESTION: "{actual_query}"
TARGET COLUMN: {state.target_column}

FULL DATASET ANALYSIS:
{context_data}

Compare and explain the different strategies, columns, or preprocessing approaches based on the analysis data provided.
"""
                        else:  # phase_specific
                            current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                            print(f"🔍 DEBUG: Creating phase-specific query prompt for phase: {current_phase}")
                            prompt = f"""You are a data preprocessing expert. Answer the user's question about the current preprocessing phase.

QUESTION: "{actual_query}"
CURRENT PHASE: {current_phase}
TARGET COLUMN: {state.target_column}

PHASE ANALYSIS:
{context_data}

Explain the current preprocessing phase, strategies, and recommendations based on the analysis data provided.
"""
                        
                        print(f"🔍 DEBUG: Prompt created - length: {len(prompt)} characters")
                        
                        # Get LLM response
                        print("🤖 DEBUG: Sending prompt to LLM...")
                        from langchain_core.messages import HumanMessage
                        response = llm.invoke([HumanMessage(content=prompt)]).content
                        
                        print(f"🤖 DEBUG: LLM response received - length: {len(response)} characters")
                        print(f"🤖 DEBUG: Response preview: {response[:100]}...")
                        
                        # Clean up temp file
                        try:
                            os.unlink(df_path)
                            print(f"🔍 DEBUG: Cleaned up temp file: {df_path}")
                        except Exception as cleanup_error:
                            print(f"⚠️ DEBUG: Failed to clean up temp file: {cleanup_error}")
                        
                        # Send response to Slack
                        print("📤 DEBUG: Preparing Slack response...")
                        slack_manager = getattr(state, '_slack_manager', None)
                        if not slack_manager:
                            print("📤 DEBUG: No slack_manager in state, using global")
                            from toolbox import slack_manager as global_slack_manager
                            slack_manager = global_slack_manager
                        
                        if slack_manager and state.chat_session:
                            print(f"📤 DEBUG: Sending message to Slack session: {state.chat_session}")
                            message = f"""🤖 **Query Response:**

{response}

**💬 Continue with preprocessing:**
• `continue` - Continue with current phase
• `summary` - Show current status  
• `help` - Get more assistance"""
                            
                            slack_manager.send_message(state.chat_session, message)
                            print("📤 DEBUG: Slack message sent successfully")
                        else:
                            print("⚠️ DEBUG: No Slack session available - message not sent")
                        
                        print("✅ DEBUG: BGE-classified query processing completed successfully")
                        return state
                        
                    except Exception as e:
                        print(f"❌ DEBUG: BGE query processing failed with error: {e}")
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
                            fallback_message = f"""🤖 **Query Response:**

I understand you're asking: "{actual_query}"

I'm having trouble accessing detailed analysis data right now, but I can help with general preprocessing questions.

**💬 Continue with preprocessing:**
• `continue` - Continue with current phase
• `summary` - Show current status"""
                            
                            slack_manager.send_message(state.chat_session, fallback_message)
                        
                        return state
                        
                except Exception as e:
                    print(f"❌ DEBUG: Complete BGE query processing failed: {e}")
                    return state
            
            # Handle other BGE-classified intents with clear intent signals (Level 4 BGE results)
            elif command.startswith('PROCEED: '):
                # Extract the actual command from the intent signal
                actual_command = command[9:]  # Remove 'PROCEED: ' prefix
                print("🚀 Processing BGE-classified PROCEED command...")
                print(f"🚀 DEBUG: BGE classified proceed: '{actual_command}'")
                # Route to continue handler (proceed maps to continue)
                return self.handle_interactive_command(state, actual_command)
            
            elif command.startswith('SKIP: '):
                # Extract the actual command from the intent signal  
                actual_command = command[6:]  # Remove 'SKIP: ' prefix
                print("⏭️ Processing BGE-classified SKIP command...")
                print(f"⏭️ DEBUG: BGE classified skip: '{actual_command}'")
                # Route to skip handler
                return self.handle_interactive_command(state, actual_command)
            
            elif command.startswith('OVERRIDE: '):
                # Extract the actual query from the intent signal
                actual_query = command[10:]  # Remove 'OVERRIDE: ' prefix
                print("🔧 Processing BGE-classified OVERRIDE command...")
                print(f"🔧 DEBUG: BGE classified override: '{actual_query}'")
                # Route to override handler with 'override ' prefix to match existing logic
                return self.handle_interactive_command(state, f"override {actual_query}")
            
            elif command.startswith('SUMMARY: '):
                # Extract the actual command from the intent signal
                actual_command = command[9:]  # Remove 'SUMMARY: ' prefix
                print("📊 Processing BGE-classified SUMMARY command...")
                print(f"📊 DEBUG: BGE classified summary: '{actual_command}'")
                # Route to summary handler
                return self.handle_interactive_command(state, actual_command)
            
            elif command.lower() in ['query', 'question', 'help', 'what', 'how', 'why', 'explain']:
                # Enhanced query handling with intelligent context passing
                print("🔍 Processing user query with enhanced LLM...")
                print(f"🔍 DEBUG: Raw query command: '{command}'")
                
                try:
                    # Initialize LLM using the same pattern as preprocessing strategies
                    from preprocessing_agent_impl import get_llm_from_state, SequentialState
                    import tempfile
                    import os
                    import re
                    
                    print("🔍 DEBUG: Importing required modules for query processing")
                    
                    # Create SequentialState for LLM initialization
                    data_to_analyze = state.cleaned_data if state.cleaned_data is not None else state.raw_data
                    print(f"🔍 DEBUG: Using {'cleaned' if state.cleaned_data is not None else 'raw'} data for analysis")
                    print(f"🔍 DEBUG: Data shape: {data_to_analyze.shape}")
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                        data_to_analyze.to_csv(tmp_file.name, index=False)
                        df_path = tmp_file.name
                    
                    print(f"🔍 DEBUG: Created temp file: {df_path}")
                    
                    sequential_state = SequentialState(
                        df=data_to_analyze,
                        df_path=df_path,
                        target_column=state.target_column,
                        model_name=os.environ.get("DEFAULT_MODEL", "gpt-4o"),
                        current_phase=state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                    )
                    
                    print(f"🔍 DEBUG: Created SequentialState - target: {state.target_column}, model: {sequential_state.model_name}")
                    
                    # Initialize LLM using the same pattern as strategy generation
                    print("🔍 DEBUG: Initializing LLM...")
                    llm = get_llm_from_state(sequential_state)
                    print(f"🔍 DEBUG: LLM initialized successfully: {type(llm).__name__}")
                    
                    # Classify query type and determine context
                    print("🔍 DEBUG: Starting query context analysis...")
                    query_analysis = self._analyze_query_context(command, state)
                    query_type = query_analysis['type']
                    context_data = query_analysis['context']
                    
                    print(f"🔍 DEBUG: Query analysis complete:")
                    print(f"   - Query type: {query_type}")
                    print(f"   - Context level: {query_analysis['context_level']}")
                    print(f"   - Mentioned column: {query_analysis.get('column', 'None')}")
                    print(f"   - Context data length: {len(str(context_data)) if context_data else 0} characters")
                    
                    # Generate LLM response based on query type and context
                    print(f"🔍 DEBUG: Generating prompt for query type: {query_type}")
                    
                    if query_type == 'general':
                        # General methodology questions - no data context needed
                        prompt = f"""You are a data preprocessing expert. Answer this question clearly and concisely:

QUESTION: "{command}"

Provide a clear, educational explanation about preprocessing concepts, methods, and best practices. Focus on practical understanding.
"""
                        print("🔍 DEBUG: Using general methodology prompt (no context)")
                    
                    elif query_type == 'column_specific':
                        # Questions about specific columns and their strategies
                        column_name = query_analysis.get('column')
                        print(f"🔍 DEBUG: Column-specific query for column: {column_name}")
                        prompt = f"""You are a data preprocessing expert analyzing a specific column. Answer the user's question using the provided analysis data.

QUESTION: "{command}"
TARGET COLUMN: {state.target_column}
COLUMN OF INTEREST: {column_name}

COLUMN ANALYSIS:
{context_data}

Provide a specific explanation about this column's preprocessing strategy, including why it was recommended based on the data characteristics shown above.
"""
                        print(f"🔍 DEBUG: Column-specific prompt created with context for '{column_name}'")
                    
                    elif query_type == 'comparative':
                        # Questions about multiple columns or comparisons
                        print("🔍 DEBUG: Comparative query - using full analysis context")
                        prompt = f"""You are a data preprocessing expert. Answer the user's question using the complete analysis data provided.

QUESTION: "{command}"
TARGET COLUMN: {state.target_column}

COMPLETE ANALYSIS:
{context_data}

Analyze the data and provide a comprehensive answer comparing columns, strategies, or identifying patterns as requested.
"""
                        print("🔍 DEBUG: Comparative prompt created with full context")
                    
                    else:  # phase_specific
                        # Questions about current phase strategies
                        current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                        print(f"🔍 DEBUG: Phase-specific query for phase: {current_phase}")
                        prompt = f"""You are a data preprocessing expert. Answer the user's question about the current preprocessing phase.

QUESTION: "{command}"
CURRENT PHASE: {current_phase}
TARGET COLUMN: {state.target_column}

PHASE ANALYSIS:
{context_data}

Explain the current preprocessing phase, strategies, and recommendations based on the analysis data provided.
"""
                        print(f"🔍 DEBUG: Phase-specific prompt created for '{current_phase}'")
                    
                    # Get LLM response
                    print("🤖 DEBUG: Sending prompt to LLM...")
                    print(f"🤖 DEBUG: Prompt length: {len(prompt)} characters")
                    
                    from langchain_core.messages import HumanMessage
                    response = llm.invoke([HumanMessage(content=prompt)]).content
                    
                    print(f"🤖 DEBUG: LLM response received - length: {len(response)} characters")
                    print(f"🤖 DEBUG: Response preview: {response[:100]}...")
                    
                    # Clean up temp file
                    try:
                        os.unlink(df_path)
                        print(f"🔍 DEBUG: Cleaned up temp file: {df_path}")
                    except Exception as cleanup_error:
                        print(f"⚠️ DEBUG: Failed to clean up temp file: {cleanup_error}")
                    
                    # Send response to Slack
                    print("📤 DEBUG: Preparing Slack response...")
                    slack_manager = getattr(state, '_slack_manager', None)
                    if not slack_manager:
                        print("📤 DEBUG: No slack_manager in state, using global")
                        from toolbox import slack_manager as global_slack_manager
                        slack_manager = global_slack_manager
                    
                    if slack_manager and state.chat_session:
                        print(f"📤 DEBUG: Sending message to Slack session: {state.chat_session}")
                        message = f"""🤖 **Query Response:**

{response}

**💬 Continue with preprocessing:**
• `continue` - Continue with current phase
• `summary` - Show current status  
• `help` - Get more assistance"""
                        
                        slack_manager.send_message(state.chat_session, message)
                        print("📤 DEBUG: Slack message sent successfully")
                    else:
                        print("⚠️ DEBUG: No Slack session available - message not sent")
                    
                    print("✅ DEBUG: Enhanced query processing completed successfully")
                    return state
                    
                except Exception as e:
                    print(f"❌ DEBUG: Enhanced query processing failed with error: {e}")
                    print(f"❌ DEBUG: Error type: {type(e).__name__}")
                    import traceback
                    print(f"❌ DEBUG: Full traceback:")
                    traceback.print_exc()
                    
                    # Fallback to basic response
                    print("🔄 DEBUG: Attempting fallback response...")
                    try:
                        slack_manager = getattr(state, '_slack_manager', None)
                        if not slack_manager:
                            print("🔄 DEBUG: Using global slack manager for fallback")
                            from toolbox import slack_manager as global_slack_manager
                            slack_manager = global_slack_manager
                        
                        if slack_manager and state.chat_session:
                            print("🔄 DEBUG: Sending fallback message to Slack")
                            fallback_message = f"""🤖 **Query Response:**

I understand you're asking: "{command}"

I'm having trouble accessing detailed analysis data right now, but I can help with general preprocessing questions. Here are some common topics:

**🔧 Preprocessing Methods:**
• **Outliers**: Winsorize (clip extreme values) vs Keep (leave as-is)
• **Missing Values**: Mean/Median imputation vs Model-based vs Drop
• **Encoding**: One-hot vs Label vs Target encoding for categories  
• **Transformations**: Log/Square root for skewed data, Scaling for normalization

**💬 Try asking:**
• `"explain median imputation"`
• `"what is winsorization"`  
• `"why use one-hot encoding"`
• `summary` - Show current preprocessing status

**💬 Continue with preprocessing:**
• `continue` - Continue with current phase
• `summary` - Show current status"""
                            
                            slack_manager.send_message(state.chat_session, fallback_message)
                            print("🔄 DEBUG: Fallback message sent successfully")
                        else:
                            print("⚠️ DEBUG: No Slack session for fallback message")
                    
                    except Exception as fallback_error:
                        print(f"❌ DEBUG: Fallback response also failed: {fallback_error}")
                        print(f"❌ DEBUG: Fallback error type: {type(fallback_error).__name__}")
                    
                    return state

            elif command.lower() in ['override', 'change', 'modify', 'custom'] or any(override_word in command.lower() for override_word in ['modify', 'change', 'use', 'apply', 'do', 'keep', 'dont', 'dont transform', 'leave unchanged']):
                # Handle user overrides using the existing process_user_input_with_llm function
                print("🔧 Processing user override request...")
                
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
                    print(f"🔧 Parsed overrides for phase {current_phase}: {parsed}")
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
                                    state.user_overrides[current_phase][col] = { 'strategy': 'constant', 'constant_value': 0 }
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
                        model_name=os.environ.get("DEFAULT_MODEL", "gpt-4o"),
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
                        override_response += f"\n\n**Captured Overrides:**\n{captured}"
                    
                    # Send response to Slack
                    slack_manager = getattr(state, '_slack_manager', None)
                    if not slack_manager:
                        from toolbox import slack_manager as global_slack_manager
                        slack_manager = global_slack_manager
                    
                    if slack_manager and state.chat_session:
                        message = f"""🔧 **Override Request:**

{override_response}

**💬 Continue with preprocessing:**
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
                    print(f"❌ Override processing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return state

            elif command.lower() in ['skip', 'next', 'advance'] or any(skip_cmd in command.lower() for skip_cmd in ['skip outliers', 'skip missing', 'skip encoding', 'skip transformations']):
                # Handle skip commands to move to next phase
                print("⏭️ Skipping current phase...")
                
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
                
                print(f"🔧 DEBUG: Skipping from {current_phase} to {next_phase}")
                
                # Update state for next phase
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
                    message = f"""⏭️ **Phase Skipped!**

**🔄 Moved from {current_phase} to {next_phase}**

**💬 Next Steps:**
• `continue` - Start {next_phase} analysis
• `summary` - Show current status
• `help` - Get assistance"""
                    
                    slack_manager.send_message(state.chat_session, message)
                
                return state

            elif command.lower() in ['summary', 'status', 'progress']:
                # Show current preprocessing status
                print("📊 Generating preprocessing summary...")
                
                current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                status = state.preprocessing_state.get('status', 'unknown')
                
                # Build summary message
                summary_parts = []
                summary_parts.append(f"**📊 Preprocessing Status:**")
                summary_parts.append(f"• **Current Phase:** {current_phase}")
                summary_parts.append(f"• **Status:** {status}")
                
                if state.cleaned_data is not None:
                    summary_parts.append(f"• **Data Shape:** {state.cleaned_data.shape[0]:,} rows × {state.cleaned_data.shape[1]} columns")
                else:
                    summary_parts.append(f"• **Data Shape:** {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns")
                
                # Add phase-specific information
                if current_phase == 'outliers' and state.preprocessing_state.get('outlier_results'):
                    outlier_results = state.preprocessing_state['outlier_results']
                    if isinstance(outlier_results, dict) and 'outlier_columns' in outlier_results:
                        outlier_count = len(outlier_results['outlier_columns'])
                        summary_parts.append(f"• **Outliers Found:** {outlier_count} columns")
                
                elif current_phase == 'missing_values' and state.preprocessing_state.get('missing_results'):
                    missing_results = state.preprocessing_state['missing_results']
                    if isinstance(missing_results, dict) and 'missing_columns' in missing_results:
                        missing_count = len(missing_results['missing_columns'])
                        summary_parts.append(f"• **Missing Values:** {missing_count} columns")
                
                elif current_phase == 'encoding' and state.preprocessing_state.get('encoding_results'):
                    encoding_results = state.preprocessing_state['encoding_results']
                    if isinstance(encoding_results, dict) and 'categorical_columns' in encoding_results:
                        categorical_count = len(encoding_results['categorical_columns'])
                        summary_parts.append(f"• **Categorical Columns:** {categorical_count} columns")
                
                elif current_phase == 'transformations' and state.preprocessing_state.get('transformation_results'):
                    transformation_results = state.preprocessing_state['transformation_results']
                    if isinstance(transformation_results, dict) and 'numerical_columns' in transformation_results:
                        numerical_count = len(transformation_results['numerical_columns'])
                        summary_parts.append(f"• **Numerical Columns:** {numerical_count} columns")
                
                summary_parts.append(f"\n**💬 Available Commands:**")
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
                print(f"❌ Unknown interactive command: {command}")
                return state
        
        except Exception as e:
            print(f"❌ Interactive command handling failed: {e}")
            import traceback
            traceback.print_exc()
            return state

    def _analyze_query_context(self, query: str, state: PipelineState) -> dict:
        """Analyze query to determine what context to provide to LLM"""
        print(f"🔍 DEBUG: [_analyze_query_context] Starting analysis for query: '{query}'")
        
        query_lower = query.lower()
        print(f"🔍 DEBUG: [_analyze_query_context] Normalized query: '{query_lower}'")
        
        # Extract column names mentioned in query
        data_cols = list((state.cleaned_data if state.cleaned_data is not None else state.raw_data).columns)
        mentioned_columns = [col for col in data_cols if col.lower() in query_lower]
        print(f"🔍 DEBUG: [_analyze_query_context] Available columns: {data_cols}")
        print(f"🔍 DEBUG: [_analyze_query_context] Mentioned columns: {mentioned_columns}")
        
        # Determine query type
        general_keywords = ['what is', 'explain', 'how does', 'what are', 'define', 'meaning of', 'concept']
        column_keywords = ['this column', 'for this', 'why median', 'why mean', 'why winsorize', 'strategy for']
        comparative_keywords = ['which column', 'what columns', 'how many', 'compare', 'maximum', 'minimum', 'most', 'least']
        
        print(f"🔍 DEBUG: [_analyze_query_context] Checking keyword matches:")
        general_matches = [kw for kw in general_keywords if kw in query_lower]
        column_matches = [kw for kw in column_keywords if kw in query_lower]
        comparative_matches = [kw for kw in comparative_keywords if kw in query_lower]
        print(f"   - General matches: {general_matches}")
        print(f"   - Column matches: {column_matches}")
        print(f"   - Comparative matches: {comparative_matches}")
        
        if any(keyword in query_lower for keyword in general_keywords) and not mentioned_columns:
            query_type = 'general'
            context_level = 'none'
            context_data = None
            print(f"🔍 DEBUG: [_analyze_query_context] Classified as GENERAL (no context needed)")
            
        elif mentioned_columns or any(keyword in query_lower for keyword in column_keywords):
            query_type = 'column_specific'
            context_level = 'column'
            # Get context for specific column
            if mentioned_columns:
                column_name = mentioned_columns[0]  # Use first mentioned column
                print(f"🔍 DEBUG: [_analyze_query_context] Using mentioned column: {column_name}")
            else:
                # Try to infer from current phase context
                current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                column_name = None
                print(f"🔍 DEBUG: [_analyze_query_context] No specific column mentioned, current phase: {current_phase}")
            
            print(f"🔍 DEBUG: [_analyze_query_context] Classified as COLUMN_SPECIFIC for column: {column_name}")
            context_data = self._get_column_context(column_name, state)
            
        elif any(keyword in query_lower for keyword in comparative_keywords):
            query_type = 'comparative' 
            context_level = 'full'
            print(f"🔍 DEBUG: [_analyze_query_context] Classified as COMPARATIVE (full context)")
            context_data = self._get_full_analysis_context(state)
            
        else:
            query_type = 'phase_specific'
            context_level = 'phase'
            print(f"🔍 DEBUG: [_analyze_query_context] Classified as PHASE_SPECIFIC (current phase context)")
            context_data = self._get_phase_context(state)
        
        result = {
            'type': query_type,
            'context_level': context_level,
            'context': context_data,
            'column': mentioned_columns[0] if mentioned_columns else None
        }
        
        print(f"🔍 DEBUG: [_analyze_query_context] Final analysis result:")
        print(f"   - Type: {result['type']}")
        print(f"   - Context level: {result['context_level']}")
        print(f"   - Column: {result['column']}")
        print(f"   - Context available: {'Yes' if result['context'] else 'No'}")
        
        return result
    
    def _get_column_context(self, column_name: str, state: PipelineState) -> str:
        """Get analysis context for a specific column"""
        print(f"🔍 DEBUG: [_get_column_context] Getting context for column: '{column_name}'")
        
        try:
            context_parts = []
            
            # Basic column info
            data_to_analyze = state.cleaned_data if state.cleaned_data is not None else state.raw_data
            print(f"🔍 DEBUG: [_get_column_context] Using {'cleaned' if state.cleaned_data is not None else 'raw'} data")
            
            if column_name and column_name in data_to_analyze.columns:
                print(f"🔍 DEBUG: [_get_column_context] Column '{column_name}' found in data")
                col_data = data_to_analyze[column_name]
                context_parts.append(f"""COLUMN: {column_name}
- Data Type: {col_data.dtype}
- Missing Values: {col_data.isnull().sum()} ({col_data.isnull().mean()*100:.1f}%)
- Unique Values: {col_data.nunique()}
- Sample Values: {col_data.dropna().head(3).tolist()}""")
                
                if col_data.dtype in ['int64', 'float64']:
                    print(f"🔍 DEBUG: [_get_column_context] Adding numeric statistics for '{column_name}'")
                    context_parts.append(f"""- Mean: {col_data.mean():.2f}
- Median: {col_data.median():.2f}
- Std Dev: {col_data.std():.2f}
- Min: {col_data.min():.2f}, Max: {col_data.max():.2f}""")
            else:
                print(f"⚠️ DEBUG: [_get_column_context] Column '{column_name}' not found in data")
            
            # Get recommendations from analysis results
            if hasattr(state, 'preprocessing_state') and state.preprocessing_state:
                print(f"🔍 DEBUG: [_get_column_context] Checking preprocessing state for analysis results")
                
                # Check outlier results
                if 'outlier_results' in state.preprocessing_state:
                    print(f"🔍 DEBUG: [_get_column_context] Found outlier results")
                    outlier_results = state.preprocessing_state['outlier_results']
                    if isinstance(outlier_results, dict) and 'llm_recommendations' in outlier_results:
                        if column_name in outlier_results['llm_recommendations']:
                            rec = outlier_results['llm_recommendations'][column_name]
                            print(f"🔍 DEBUG: [_get_column_context] Adding outlier analysis for '{column_name}'")
                            context_parts.append(f"""OUTLIER ANALYSIS:
- Recommended Treatment: {rec.get('treatment', 'N/A')}
- Reasoning: {rec.get('reasoning', 'N/A')}
- Severity: {rec.get('severity', 'N/A')}""")
                        else:
                            print(f"🔍 DEBUG: [_get_column_context] No outlier analysis found for '{column_name}'")
                
                # Check missing values results  
                if 'missing_results' in state.preprocessing_state:
                    print(f"🔍 DEBUG: [_get_column_context] Found missing values results")
                    missing_results = state.preprocessing_state['missing_results']
                    if isinstance(missing_results, dict) and 'llm_recommendations' in missing_results:
                        if column_name in missing_results['llm_recommendations']:
                            rec = missing_results['llm_recommendations'][column_name]
                            print(f"🔍 DEBUG: [_get_column_context] Adding missing values analysis for '{column_name}'")
                            context_parts.append(f"""MISSING VALUES ANALYSIS:
- Recommended Strategy: {rec.get('strategy', 'N/A')}
- Reasoning: {rec.get('reasoning', 'N/A')}
- Priority: {rec.get('priority', 'N/A')}""")
                        else:
                            print(f"🔍 DEBUG: [_get_column_context] No missing values analysis found for '{column_name}'")
                
                # Check encoding results
                if 'encoding_results' in state.preprocessing_state:
                    print(f"🔍 DEBUG: [_get_column_context] Found encoding results")
                    encoding_results = state.preprocessing_state['encoding_results']
                    if isinstance(encoding_results, dict) and 'llm_recommendations' in encoding_results:
                        if column_name in encoding_results['llm_recommendations']:
                            rec = encoding_results['llm_recommendations'][column_name]
                            print(f"🔍 DEBUG: [_get_column_context] Adding encoding analysis for '{column_name}'")
                            context_parts.append(f"""ENCODING ANALYSIS:
- Recommended Strategy: {rec.get('strategy', 'N/A')}
- Reasoning: {rec.get('reasoning', 'N/A')}
- Cardinality Level: {rec.get('cardinality_level', 'N/A')}""")
                        else:
                            print(f"🔍 DEBUG: [_get_column_context] No encoding analysis found for '{column_name}'")
                
                # Check transformation results
                if 'transformation_results' in state.preprocessing_state:
                    print(f"🔍 DEBUG: [_get_column_context] Found transformation results")
                    transformation_results = state.preprocessing_state['transformation_results']
                    if isinstance(transformation_results, dict) and 'llm_recommendations' in transformation_results:
                        if column_name in transformation_results['llm_recommendations']:
                            rec = transformation_results['llm_recommendations'][column_name]
                            print(f"🔍 DEBUG: [_get_column_context] Adding transformation analysis for '{column_name}'")
                            context_parts.append(f"""TRANSFORMATION ANALYSIS:
- Recommended Transformation: {rec.get('transformation', 'N/A')}
- Reasoning: {rec.get('reasoning', 'N/A')}
- Priority: {rec.get('priority', 'N/A')}""")
                        else:
                            print(f"🔍 DEBUG: [_get_column_context] No transformation analysis found for '{column_name}'")
            else:
                print(f"⚠️ DEBUG: [_get_column_context] No preprocessing state available")
            
            final_context = '\n\n'.join(context_parts) if context_parts else f"Limited context available for column: {column_name}"
            print(f"🔍 DEBUG: [_get_column_context] Generated context with {len(context_parts)} sections")
            return final_context
            
        except Exception as e:
            print(f"⚠️ DEBUG: [_get_column_context] Error getting column context: {e}")
            return f"Unable to retrieve detailed context for column: {column_name}"
    
    def _get_phase_context(self, state: PipelineState) -> str:
        """Get analysis context for current phase"""
        print(f"🔍 DEBUG: [_get_phase_context] Getting phase context")
        
        try:
            current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
            print(f"🔍 DEBUG: [_get_phase_context] Current phase: {current_phase}")
            
            context_parts = [f"CURRENT PHASE: {current_phase}"]
            
            if hasattr(state, 'preprocessing_state') and state.preprocessing_state:
                phase_key = f"{current_phase}_results"
                print(f"🔍 DEBUG: [_get_phase_context] Looking for phase key: {phase_key}")
                
                if phase_key in state.preprocessing_state:
                    print(f"🔍 DEBUG: [_get_phase_context] Found results for phase: {current_phase}")
                    results = state.preprocessing_state[phase_key]
                    if isinstance(results, dict) and 'llm_recommendations' in results:
                        context_parts.append("PHASE RECOMMENDATIONS:")
                        rec_count = 0
                        for col, rec in results['llm_recommendations'].items():
                            strategy = rec.get('treatment') or rec.get('strategy') or rec.get('transformation', 'N/A')
                            reasoning = rec.get('reasoning', 'N/A')
                            context_parts.append(f"- {col}: {strategy} ({reasoning})")
                            rec_count += 1
                        print(f"🔍 DEBUG: [_get_phase_context] Added {rec_count} recommendations")
                    else:
                        print(f"🔍 DEBUG: [_get_phase_context] No LLM recommendations in phase results")
                else:
                    print(f"🔍 DEBUG: [_get_phase_context] No results found for phase key: {phase_key}")
            else:
                print(f"⚠️ DEBUG: [_get_phase_context] No preprocessing state available")
            
            final_context = '\n'.join(context_parts)
            print(f"🔍 DEBUG: [_get_phase_context] Generated phase context with {len(context_parts)} parts")
            return final_context
            
        except Exception as e:
            print(f"⚠️ DEBUG: [_get_phase_context] Error getting phase context: {e}")
            return "Unable to retrieve phase context"
    
    def _get_full_analysis_context(self, state: PipelineState) -> str:
        """Get complete analysis context for comparative queries"""
        print(f"🔍 DEBUG: [_get_full_analysis_context] Getting full analysis context")
        
        try:
            context_parts = []
            
            # Dataset overview
            data_to_analyze = state.cleaned_data if state.cleaned_data is not None else state.raw_data
            print(f"🔍 DEBUG: [_get_full_analysis_context] Dataset shape: {data_to_analyze.shape}")
            
            context_parts.append(f"""DATASET OVERVIEW:
- Shape: {data_to_analyze.shape[0]} rows × {data_to_analyze.shape[1]} columns
- Target Column: {state.target_column}
- Columns: {', '.join(data_to_analyze.columns.tolist())}""")
            
            # All analysis results
            if hasattr(state, 'preprocessing_state') and state.preprocessing_state:
                print(f"🔍 DEBUG: [_get_full_analysis_context] Checking all phase results")
                
                phases_found = 0
                for phase in ['outlier', 'missing', 'encoding', 'transformation']:
                    phase_key = f"{phase}_results"
                    if phase_key in state.preprocessing_state:
                        print(f"🔍 DEBUG: [_get_full_analysis_context] Found results for phase: {phase}")
                        results = state.preprocessing_state[phase_key]
                        if isinstance(results, dict) and 'llm_recommendations' in results:
                            context_parts.append(f"\n{phase.upper()} RECOMMENDATIONS:")
                            rec_count = 0
                            for col, rec in results['llm_recommendations'].items():
                                strategy = rec.get('treatment') or rec.get('strategy') or rec.get('transformation', 'N/A')
                                reasoning = rec.get('reasoning', 'N/A')
                                context_parts.append(f"- {col}: {strategy} ({reasoning})")
                                rec_count += 1
                            print(f"🔍 DEBUG: [_get_full_analysis_context] Added {rec_count} recommendations for {phase}")
                            phases_found += 1
                
                print(f"🔍 DEBUG: [_get_full_analysis_context] Total phases with results: {phases_found}")
            else:
                print(f"⚠️ DEBUG: [_get_full_analysis_context] No preprocessing state available")
            
            final_context = '\n'.join(context_parts)
            print(f"🔍 DEBUG: [_get_full_analysis_context] Generated full context with {len(context_parts)} sections")
            return final_context
            
        except Exception as e:
            print(f"⚠️ DEBUG: [_get_full_analysis_context] Error getting full context: {e}")
            return "Unable to retrieve complete analysis context"


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
                
                # Handle file uploads (plots, etc.)
                print(f"🔍 UPLOAD DEBUG: Checking for artifacts in result...")
                print(f"🔍 UPLOAD DEBUG: Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                if isinstance(result, dict) and 'artifacts' in result:
                    print(f"🔍 UPLOAD DEBUG: Found artifacts: {result['artifacts']}")
                    if 'files' in result['artifacts']:
                        print(f"🔍 UPLOAD DEBUG: Found files: {result['artifacts']['files']}")
                        try:
                            from toolbox import slack_manager
                            for file_info in result['artifacts']['files']:
                                print(f"🔍 UPLOAD DEBUG: Processing file_info: {file_info}")
                                if isinstance(file_info, dict) and 'path' in file_info:
                                    file_path = file_info['path']
                                    title = file_info.get('title', 'Generated Plot')
                                    print(f"🔍 UPLOAD DEBUG: Attempting upload - Path: {file_path}, Title: {title}")
                                    if os.path.exists(file_path):
                                        print(f"📤 Uploading {title}: {file_path}")
                                        slack_manager.upload_file(
                                            session_id=state.chat_session,
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
                    else:
                        print(f"🔍 UPLOAD DEBUG: No 'files' key in artifacts")
                else:
                    print(f"🔍 UPLOAD DEBUG: No artifacts in result")
            
            print("✅ Model building completed")
            return state
            
        except Exception as e:
            print(f"❌ Model building agent failed: {e}")
            import traceback
            traceback.print_exc()
            return state


# Global instances - these are the agents the orchestrator will use
preprocessing_agent = PreprocessingAgentWrapper()
feature_selection_agent = FeatureSelectionAgentWrapper()
model_building_agent = ModelBuildingAgentWrapper()

print("🎯 Agent wrappers initialized - using actual working implementations AS-IS")
