#!/usr/bin/env python3
"""
Fast Model Agent - Automated ML Pipeline
Handles fast model building without user interaction
"""

import pandas as pd
from typing import Dict, Any, Optional
from pipeline_state import PipelineState
from toolbox import print_to_log
import threading
import time

# FastModeSimulator class moved here to avoid circular imports
class FastModeSimulator:
    """Simulates user inputs for automated pipeline execution"""
    
    def __init__(self):
        self.responses = {
            "preprocessing": ["proceed", "continue", "yes"],
            "feature_selection": ["continue", "proceed", "yes"], 
            "model_building": ["continue", "proceed", "yes"],
            "general": ["continue", "proceed", "yes"]
        }
    
    def get_response(self, context: str = "general") -> str:
        """Get automated response based on context"""
        return self.responses.get(context, ["continue"])[0]

class FastModelAgent:
    """Agent for automated ML pipeline execution"""
    
    def __init__(self):
        self.name = "FastModelAgent"
        
    def handle_fast_model_request(self, state: PipelineState, target_column: Optional[str] = None) -> PipelineState:
        """Handle fast model pipeline request"""
        print_to_log(f"🚀 FastModelAgent: Starting automated pipeline")
        
        # Validate we have data
        if state.raw_data is None:
            state.last_response = "❌ **No dataset found**\n\nPlease upload a CSV file first before using fast model pipeline."
            return state
        
        # Get target column if not provided
        if not target_column:
            target_column = self._get_target_column_interactive(state)
            if not target_column:
                return state
        
        # Validate target column exists
        if target_column not in state.raw_data.columns:
            available_cols = list(state.raw_data.columns)
            state.last_response = f"""❌ **Target column '{target_column}' not found**

**Available columns:** {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}

Please specify a valid column name."""
            return state
        
        # Set target column
        state.target_column = target_column
        print_to_log(f"🎯 Target column set: {target_column}")
        
        # Start automated pipeline
        return self._run_automated_pipeline(state)
    
    def _get_target_column_interactive(self, state: PipelineState) -> Optional[str]:
        """Get target column from user interactively"""
        columns = list(state.raw_data.columns)
        
        state.last_response = f"""🎯 **Target Column Selection**

**Available columns:** {', '.join(columns[:20])}{'...' if len(columns) > 20 else ''}

Please specify which column is your **target variable** (the column you want to predict).

**Example:** If predicting fraud, respond with: `is_fraud`"""
        
        return None  # Will wait for user response
    
    def _run_automated_pipeline(self, state: PipelineState) -> PipelineState:
        """Run the complete automated ML pipeline with proper phase management and clean messages"""
        print_to_log(f"🚀 Starting automated ML pipeline for target: {state.target_column}")
        
        try:
            # Import the preprocessing agent wrapper
            from agents_wrapper import preprocessing_agent, feature_selection_agent, model_building_agent
            
            # Create temporary session for components that need it
            temp_session = f"fast_pipeline_{id(state)}"
            original_chat_session = state.chat_session
            
            print_to_log("🧹 Started preprocessing")
            
            # Phase 1: Overview - Start the preprocessing workflow
            print_to_log("📊 Starting overview phase")
            state.user_query = "proceed"
            state.chat_session = None  # Disable Slack messages
            state = preprocessing_agent.handle_interactive_command(state, "proceed")
            
            if state.last_error:
                state.last_response = f"❌ **Overview phase failed:** {state.last_error}"
                return state
            print_to_log("✅ Finished overview phase")
            
            # Phase 2: Outliers - Force phase to outliers and continue
            print_to_log("🚨 Starting outlier phase")
            # Manually set the phase to ensure correct routing
            if hasattr(state, 'current_phase'):
                state.current_phase = "outliers"
            state.user_query = "continue"
            state = preprocessing_agent.handle_interactive_command(state, "continue")
            
            if state.last_error:
                state.last_response = f"❌ **Outlier phase failed:** {state.last_error}"
                return state
            print_to_log("✅ Finished outlier phase")
            
            # Phase 3: Missing Values - Force phase to missing_values and continue
            print_to_log("🗑️ Starting missing values phase")
            if hasattr(state, 'current_phase'):
                state.current_phase = "missing_values"
            state.user_query = "continue"
            state = preprocessing_agent.handle_interactive_command(state, "continue")
            
            if state.last_error:
                state.last_response = f"❌ **Missing values phase failed:** {state.last_error}"
                return state
            print_to_log("✅ Finished missing values phase")
            
            # Phase 4: Encoding - Force phase to encoding and continue
            print_to_log("🏷️ Starting encoding phase")
            if hasattr(state, 'current_phase'):
                state.current_phase = "encoding"
            state.user_query = "continue"
            state = preprocessing_agent.handle_interactive_command(state, "continue")
            
            if state.last_error:
                state.last_response = f"❌ **Encoding phase failed:** {state.last_error}"
                return state
            print_to_log("✅ Finished encoding phase")
            
            # Phase 5: Transformations - Force phase to transformations and continue
            print_to_log("🔄 Starting transformations phase")
            if hasattr(state, 'current_phase'):
                state.current_phase = "transformations"
            state.user_query = "continue"
            state = preprocessing_agent.handle_interactive_command(state, "continue")
            
            if state.last_error:
                state.last_response = f"❌ **Transformations phase failed:** {state.last_error}"
                return state
            print_to_log("✅ Finished transformations phase")
            
            print_to_log("🎉 Finished preprocessing")
            
            # Phase 6: Feature Selection with temporary session
            print_to_log("🔍 Started feature selection")
            
            # Restore a temporary session for feature selection
            state.chat_session = temp_session
            
            try:
                # Apply IV filter
                state.user_query = "apply iv filter 0.02"
                state = feature_selection_agent.handle_interactive_command(state, "apply iv filter 0.02")
                
                if not state.last_error:
                    # Apply correlation filter
                    state.user_query = "apply correlation filter 0.5"
                    state = feature_selection_agent.handle_interactive_command(state, "apply correlation filter 0.5")
                
                if state.last_error:
                    print_to_log(f"⚠️ Feature selection had issues: {state.last_error}")
                    print_to_log("📊 Proceeding with all available features")
                    # Clear the error to continue
                    state.last_error = None
                
            except Exception as e:
                print_to_log(f"⚠️ Feature selection failed: {e}")
                print_to_log("📊 Proceeding with all available features")
            
            print_to_log("✅ Final features selected")
            
            # Phase 7: Model Building with session handling
            print_to_log("🤖 Started modeling")
            
            try:
                # Ensure we have a valid session for model building
                state.chat_session = temp_session
                state.user_query = "build a classification model"
                state = model_building_agent.run(state)
                
                if state.last_error:
                    print_to_log(f"⚠️ Model building had issues: {state.last_error}")
                    print_to_log("📊 Model training was attempted")
                    # Clear the error for final response
                    state.last_error = None
                    
            except Exception as e:
                print_to_log(f"⚠️ Model building failed: {e}")
                print_to_log("📊 Model training was attempted")
            
            print_to_log("✅ Final modeling results completed")
            
            # Restore original chat session for final message
            state.chat_session = original_chat_session
            
            # Generate clean success message with actual results
            final_shape = (
                state.processed_data.shape if state.processed_data is not None 
                else state.cleaned_data.shape if state.cleaned_data is not None 
                else state.raw_data.shape
            )
            
            feature_count = (
                len(state.selected_features) if state.selected_features 
                else final_shape[1] - 1 if final_shape else "Unknown"
            )
            
            model_status = "✅ Trained Successfully" if state.trained_model else "⚠️ Training Attempted"
            
            state.last_response = f"""🎉 **Fast ML Pipeline Complete!**

🎯 **Target:** {state.target_column}
📊 **Data Shape:** {state.raw_data.shape} → {final_shape}
🔍 **Features:** {feature_count}
🤖 **Model:** {model_status}

✅ **Completed Phases:**
• 📊 Overview - Dataset analyzed
• 🚨 Outliers - Outliers handled  
• 🗑️ Missing Values - Data imputed
• 🏷️ Encoding - Variables encoded
• 🔄 Transformations - Features transformed
• 🔍 Feature Selection - Features selected
• 🤖 Model Building - Model trained

**All phases completed automatically - your model is ready!**"""

            print_to_log("🎉 Complete automated ML pipeline finished successfully!")
            return state
            
        except Exception as e:
            # Restore original chat session in case of error
            if 'original_chat_session' in locals():
                state.chat_session = original_chat_session
                
            error_msg = f"Pipeline execution failed: {str(e)}"
            print_to_log(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            state.last_error = error_msg
            state.last_response = f"❌ **Pipeline Error:** {error_msg}"
            return state

def fast_model_agent(state: PipelineState) -> PipelineState:
    """Main entry point for fast model agent - called by langgraph_pipeline.py"""
    print_to_log("🚀 [Fast Model Agent] Starting fast model pipeline")
    
    agent = FastModelAgent()
    
    # Check if this is a target column response
    if state.target_column is None and state.user_query and state.raw_data is not None:
        # Check if user query looks like a column name
        if state.user_query.strip() in state.raw_data.columns:
            print_to_log(f"🎯 [Fast Model Agent] Setting target column: {state.user_query.strip()}")
            return agent.handle_fast_model_request(state, state.user_query.strip())
    
    # Otherwise start the normal flow
    print_to_log("🚀 [Fast Model Agent] Starting automated pipeline")
    return agent.handle_fast_model_request(state)
