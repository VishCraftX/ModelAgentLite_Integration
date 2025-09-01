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
                        analyze_outliers_single_batch,
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
                    
                    # Run outlier detection
                    print("🔍 Running outlier detection...")
                    outlier_results = analyze_outliers_single_batch(sequential_state)
                    
                    # Debug: Check what we got back
                    print(f"🔍 DEBUG: outlier_results type: {type(outlier_results)}")
                    print(f"🔍 DEBUG: outlier_results content: {outlier_results}")
                    
                    # Generate a summary message
                    outlier_columns = []
                    total_outliers = 0
                    
                    # Handle different possible return types from outlier analysis
                    if isinstance(outlier_results, dict):
                        # Check if it has the expected structure with 'outlier_columns'
                        if 'outlier_columns' in outlier_results:
                            outlier_columns = outlier_results['outlier_columns']
                            # Calculate total outliers from analysis_details
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
                        # Build the outlier details based on the type
                        if isinstance(outlier_results, dict):
                            if 'analysis_details' in outlier_results:
                                # New structure with analysis_details
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
                                # Old structure
                                outlier_details = chr(10).join([f'• {col}: {outlier_results[col].get("outlier_count", 0):,} outliers ({outlier_results[col].get("outlier_percentage", 0):.1f}%)' for col in outlier_columns[:5]])
                        elif isinstance(outlier_results, list):
                            outlier_details = chr(10).join([f'• {col}: outliers detected' for col in outlier_columns[:5]])
                        else:
                            outlier_details = "• Analysis completed"
                        
                        # Build LLM recommendations summary
                        llm_recommendations = ""
                        if isinstance(outlier_results, dict) and 'llm_recommendations' in outlier_results:
                            recommendations = outlier_results['llm_recommendations']
                            llm_recommendations = "\n**🤖 LLM Recommendations:**\n"
                            for col, rec in recommendations.items():
                                if col in outlier_columns:
                                    llm_recommendations += f"• **{col}**: {rec.get('treatment', 'keep')} ({rec.get('severity', 'unknown')} severity)\n"
                        
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
{llm_recommendations}
**💬 Next Steps:**
• `continue` - Apply LLM recommendations and move to missing values
• `skip outliers` - Move to missing values analysis
• `summary` - Show current status
• `explain [column]` - Get detailed analysis for a specific column

**🔧 Available Actions:**
• `remove outliers` - Remove all outliers
• `cap outliers` - Cap outliers to 95th percentile
• `keep outliers` - Keep outliers as-is"""
                        
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
                current_phase = state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                print(f"🔧 DEBUG: Current phase for continue: {current_phase}")

                if current_phase == 'outliers':
                    # Apply outlier treatments and move to missing_values
                    print("🔧 Applying outlier treatments...")
                    
                    # Get outlier results from state
                    outlier_results = state.preprocessing_state.get('outlier_results', {})
                    if not outlier_results:
                        print("❌ No outlier results found in state")
                        return state
                    
                    # Apply treatments based on LLM recommendations
                    df = state.raw_data.copy()
                    applied_treatments = []
                    
                    if isinstance(outlier_results, dict) and 'llm_recommendations' in outlier_results:
                        for col, recommendation in outlier_results['llm_recommendations'].items():
                            raw_treatment = recommendation.get('treatment', 'winsorize')
                            treatment = str(raw_treatment).lower().replace('-', '_')
                            if treatment == 'winsorize':
                                # Apply winsorization
                                lower_percentile = 1
                                upper_percentile = 99
                                lower_val = df[col].quantile(lower_percentile / 100)
                                upper_val = df[col].quantile(upper_percentile / 100)
                                df[col] = df[col].clip(lower=lower_val, upper=upper_val)
                                applied_treatments.append(f"• {col}: Winsorized ({lower_percentile}st-{upper_percentile}th percentile)")
                            elif treatment == 'remove':
                                # Remove outliers using IQR method
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                                applied_treatments.append(f"• {col}: Outliers removed (IQR method)")
                            elif treatment == 'mark_missing':
                                # Mark detected outliers as NaN for later imputation
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                                df.loc[outlier_mask, col] = np.nan
                                applied_treatments.append(f"• {col}: Outliers marked as missing")
                            elif treatment == 'keep':
                                applied_treatments.append(f"• {col}: Kept outliers as-is")
                    
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
                            analyze_missing_values_single_batch,
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
                            print("🔍 Running missing values analysis...")
                            missing_results = analyze_missing_values_single_batch(sequential_state, data_to_analyze)
                            
                            # Send results to Slack
                            slack_manager = getattr(state, '_slack_manager', None)
                            if not slack_manager:
                                from toolbox import slack_manager as global_slack_manager
                                slack_manager = global_slack_manager
                            
                            if slack_manager and state.chat_session:
                                # Build missing values details and LLM recommendations
                                if isinstance(missing_results, dict) and 'missing_columns' in missing_results:
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
                                    message = f"""🔍 **Missing Values Analysis Complete!**

**📊 Analysis Results:**
{missing_results}

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
                            analyze_encoding_single_batch,
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
                            
                            # Run encoding analysis
                            print("🔍 Running encoding analysis...")
                            encoding_results = analyze_encoding_single_batch(sequential_state, data_to_analyze)
                            
                            print(f"🔍 DEBUG: encoding_results type: {type(encoding_results)}")
                            print(f"🔍 DEBUG: encoding_results content: {encoding_results}")
                            
                            # Send results to Slack
                            slack_manager = getattr(state, '_slack_manager', None)
                            if not slack_manager:
                                from toolbox import slack_manager as global_slack_manager
                                slack_manager = global_slack_manager
                            
                            if slack_manager and state.chat_session:
                                # Build encoding details and LLM recommendations
                                if isinstance(encoding_results, dict) and 'categorical_columns' in encoding_results:
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
                                        cols_str = ', '.join(cols[:5])  # Show first 5 columns
                                        if len(cols) > 5:
                                            cols_str += f" (+{len(cols)-5} more)"
                                        encoding_summary.append(f"**{encoding_type} encoding:** {cols_str}")
                                    
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
                                    message = f"""🔍 **Encoding Analysis Complete!**

**📊 Analysis Results:**
{encoding_results}

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
                            analyze_transformations_single_batch,
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
                            
                            # Run transformations analysis
                            print("🔍 Running transformations analysis...")
                            transformation_results = analyze_transformations_single_batch(sequential_state, data_to_analyze)
                            
                            print(f"🔍 DEBUG: transformation_results type: {type(transformation_results)}")
                            print(f"🔍 DEBUG: transformation_results content: {transformation_results}")
                            
                            # Send results to Slack
                            slack_manager = getattr(state, '_slack_manager', None)
                            if not slack_manager:
                                from toolbox import slack_manager as global_slack_manager
                                slack_manager = global_slack_manager
                            
                            if slack_manager and state.chat_session:
                                # Build transformations details and LLM recommendations
                                if isinstance(transformation_results, dict) and 'transformation_columns' in transformation_results:
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
                                        cols_str = ', '.join(cols[:5])  # Show first 5 columns
                                        if len(cols) > 5:
                                            cols_str += f" (+{len(cols)-5} more)"
                                        # For the friendly label, avoid appending the word 'transformation' for 'No transformation needed'
                                        if transformation_type == 'No transformation needed':
                                            transformation_summary.append(f"**{transformation_type}:** {cols_str}")
                                        else:
                                            transformation_summary.append(f"**{transformation_type}:** {cols_str}")
                                    
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
                                    message = f"""🔍 **Transformations Analysis Complete!**

**📊 Analysis Results:**
{transformation_results}

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
            
            elif command.lower() in ['query', 'question', 'help', 'what', 'how', 'why', 'explain']:
                # Handle user queries using the existing process_user_input_with_llm function
                print("🔍 Processing user query with LLM...")
                
                # Import the function from preprocessing_agent_impl
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
                    # Create SequentialState for query processing
                    sequential_state = SequentialState(
                        df=data_to_analyze,
                        df_path=df_path,
                        target_column=state.target_column,
                        model_name=os.environ.get("DEFAULT_MODEL", "gpt-4o"),
                        current_phase=state.preprocessing_state.get('current_phase', 'overview') if state.preprocessing_state else 'overview'
                    )
                    
                    # Process the user input
                    updated_sequential_state = process_user_input_with_llm(sequential_state, command)
                    
                    # Get the response
                    query_response = updated_sequential_state.query_response or "I can help you with preprocessing questions. What would you like to know?"
                    
                    # Send response to Slack
                    slack_manager = getattr(state, '_slack_manager', None)
                    if not slack_manager:
                        from toolbox import slack_manager as global_slack_manager
                        slack_manager = global_slack_manager
                    
                    if slack_manager and state.chat_session:
                        message = f"""🤖 **Query Response:**

{query_response}

**💬 Continue with preprocessing:**
• `continue` - Continue with current phase
• `summary` - Show current status
• `help` - Get more assistance"""
                        
                        slack_manager.send_message(state.chat_session, message)
                    
                    try:
                        os.unlink(df_path)
                    except:
                        pass
                    
                    return state
                    
                except Exception as e:
                    print(f"❌ Query processing failed: {e}")
                    import traceback
                    traceback.print_exc()
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
            if state.cleaned_data is None:
                print("❌ No cleaned data available for model building")
                return state
                
            if not state.selected_features:
                print("❌ No features selected for model building")
                return state
                
            print(f"🚀 Launching actual model building agent")
            print(f"📊 Data shape: {state.cleaned_data.shape}")
            print(f"🎯 Selected features: {len(state.selected_features)}")
            
            # The working agent will handle all the model building process
            # including LLM interactions, Slack updates, etc.
            
            # Call the working agent's run method
            # This will handle the entire model building workflow
            result = self.agent.run_agent(
                user_query=state.user_query,
                user_id=state.chat_session,
                data=state.cleaned_data,
                target_column=state.target_column,
                selected_features=state.selected_features
            )
            
            # Extract results
            if result and isinstance(result, dict):
                if 'model' in result:
                    state.trained_model = result['model']
                if 'metrics' in result:
                    state.model_building_state = {
                        "completed": True,
                        "timestamp": datetime.now().isoformat(),
                        "method": "langgraph_interactive",
                        "metrics": result['metrics']
                    }
            
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
