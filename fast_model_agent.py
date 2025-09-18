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
        """Run the complete automated ML pipeline"""
        print_to_log(f"🚀 Starting automated ML pipeline for target: {state.target_column}")
        
        try:
            # Import pipeline components
            from agents_wrapper import preprocessing_agent, feature_selection_agent, model_building_agent
            
            # Create simulator for automated responses
            simulator = FastModeSimulator()
            
            # Phase 1: Preprocessing
            print_to_log("🧹 Phase 1: Automated Preprocessing")
            state = preprocessing_agent.handle_interactive_command(state, simulator.get_response("preprocessing"))
            
            if state.last_error:
                state.last_response = f"❌ **Preprocessing failed:** {state.last_error}"
                return state
            
            # Phase 2: Feature Selection  
            print_to_log("🔍 Phase 2: Automated Feature Selection")
            state = feature_selection_agent.handle_interactive_command(state, simulator.get_response("feature_selection"))
            
            if state.last_error:
                state.last_response = f"❌ **Feature selection failed:** {state.last_error}"
                return state
            
            # Phase 3: Model Building
            print_to_log("🤖 Phase 3: Automated Model Building")
            state = model_building_agent.handle_interactive_command(state, simulator.get_response("model_building"))
            
            if state.last_error:
                state.last_response = f"❌ **Model building failed:** {state.last_error}"
                return state
            
            # Success message
            state.last_response = f"""🎉 **Fast Model Pipeline Complete!**

✅ **Preprocessing:** Complete
✅ **Feature Selection:** Complete  
✅ **Model Training:** Complete

🎯 **Target:** {state.target_column}
📊 **Final Dataset:** {state.processed_data.shape if state.processed_data is not None else 'N/A'}
🤖 **Model:** Ready for predictions

Your automated ML pipeline has finished successfully! The model is ready to use."""

            print_to_log("🎉 Fast model pipeline completed successfully!")
            return state
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            print_to_log(f"❌ {error_msg}")
            state.last_error = error_msg
            state.last_response = f"❌ **Pipeline Error:** {error_msg}"
            return state

def fast_model_agent(state: PipelineState) -> PipelineState:
    """Main entry point for fast model agent"""
    agent = FastModelAgent()
    
    # Check if this is a target column response
    if state.target_column is None and state.user_query and state.raw_data is not None:
        # Check if user query looks like a column name
        if state.user_query.strip() in state.raw_data.columns:
            return agent.handle_fast_model_request(state, state.user_query.strip())
    
    # Otherwise start the normal flow
    return agent.handle_fast_model_request(state)
