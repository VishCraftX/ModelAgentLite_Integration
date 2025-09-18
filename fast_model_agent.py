#!/usr/bin/env python3
"""
Fast Model Agent - Automated ML Pipeline
Handles fast model building without user interaction
"""

import pandas as pd
from typing import Dict, Any, Optional
from pipeline_state import PipelineState
from toolbox import print_to_log
from langgraph_pipeline import FastModeSimulator
import threading
import time

class FastModelAgent:
    """Agent for automated ML pipeline execution"""
    
    def __init__(self):
        self.name = "FastModelAgent"
        
    def handle_fast_model_request(self, state: PipelineState, target_column: Optional[str] = None) -> PipelineState:
        """Handle fast model pipeline request"""
        print_to_log(f"🚀 FastModelAgent: Starting automated pipeline")
        
        # Validate we have data
        if state.raw_data is None:
            state.messages.append({
                "role": "assistant", 
                "content": "❌ **No dataset found**\n\nPlease upload a CSV file first before using fast model pipeline."
            })
            return state
        
        # Get target column if not provided
        if not target_column:
            target_column = self._get_target_column_interactive(state)
            if not target_column:
                return state
        
        # Validate target column exists
        if target_column not in state.raw_data.columns:
            available_cols = list(state.raw_data.columns)
            state.messages.append({
                "role": "assistant",
                "content": f"""❌ **Target column '{target_column}' not found**

**Available columns:** {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}

Please specify a valid column name."""
            })
            return state
        
        # Set up fast mode
        state.target_column = target_column
        state._fast_mode_active = True
        state._fast_mode_simulator = FastModeSimulator()
        
        # Send initial confirmation
        state.messages.append({
            "role": "assistant",
            "content": f"""🚀 **Fast ML Pipeline Started!**

📊 **Dataset:** {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns
🎯 **Target Column:** `{target_column}` ✅
⚡ **Mode:** Fully Automated

🔄 **Processing automatically through all phases...**
⏱️ **Estimated Time:** 5-7 minutes

🤖 **You'll receive updates as each phase completes...**"""
        })
        
        # Start the automated pipeline in background
        self._execute_fast_pipeline_async(state)
        
        return state
    
    def _get_target_column_interactive(self, state: PipelineState) -> Optional[str]:
        """Get target column from user interactively"""
        available_columns = list(state.raw_data.columns)
        
        # Create column list for user selection
        column_list = []
        for i, col in enumerate(available_columns, 1):
            column_list.append(f"**{i}.** `{col}`")
        
        columns_text = "\n".join(column_list)
        
        state.messages.append({
            "role": "assistant",
            "content": f"""🎯 **Fast ML Pipeline - Target Selection**

📊 **Dataset:** {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns

**📋 Available Columns:**
{columns_text}

❓ **Which column should be used as the target variable for prediction?**

💬 **You can respond with:**
• The column name directly (e.g., `is_fraud`)
• Or: `target column_name` (e.g., `target is_fraud`)"""
        })
        
        # Set a flag to indicate we're waiting for target selection
        state._awaiting_target_selection = True
        return None
    
    def handle_target_selection(self, state: PipelineState, user_input: str) -> PipelineState:
        """Handle target column selection from user"""
        if not getattr(state, '_awaiting_target_selection', False):
            return state
        
        # Parse target column from user input
        target_column = self._parse_target_column(user_input, state.raw_data.columns)
        
        if target_column:
            # Clear the waiting flag
            state._awaiting_target_selection = False
            
            # Continue with fast pipeline
            return self.handle_fast_model_request(state, target_column)
        else:
            # Invalid target column
            available_cols = list(state.raw_data.columns)
            state.messages.append({
                "role": "assistant",
                "content": f"""❌ **Invalid target column**

Please choose from: {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}

💬 **Try again with a valid column name.**"""
            })
            return state
    
    def _parse_target_column(self, user_input: str, available_columns) -> Optional[str]:
        """Parse target column from user input"""
        user_input = user_input.strip().lower()
        
        # Check direct column name match (case insensitive)
        for col in available_columns:
            if col.lower() == user_input:
                return col
        
        # Check "target column_name" format
        if user_input.startswith('target '):
            target_name = user_input[7:].strip()
            for col in available_columns:
                if col.lower() == target_name:
                    return col
        
        # Check if it's just the column name without exact case match
        for col in available_columns:
            if user_input in col.lower() or col.lower() in user_input:
                return col
        
        return None
    
    def _execute_fast_pipeline_async(self, state: PipelineState):
        """Execute fast pipeline in background thread"""
        
        def run_pipeline():
            try:
                print_to_log(f"🚀 Fast pipeline thread started for session {state.session_id}")
                
                # Import the main pipeline
                from langgraph_pipeline import get_pipeline
                pipeline = get_pipeline()
                
                # Send progress updates for each phase
                phases = [
                    ("🔍 Overview", "Dataset analysis and summary"),
                    ("⚠️ Outliers", "Detecting and handling outliers"),
                    ("❓ Missing Values", "Handling missing data"),
                    ("🏷️ Encoding", "Converting categorical variables"),
                    ("🎯 Feature Selection", "Selecting best features"),
                    ("🤖 Model Building", "Training ML model")
                ]
                
                for i, (phase_name, description) in enumerate(phases, 1):
                    print_to_log(f"📈 Phase {i}/6: {phase_name} - {description}")
                    
                    # Send progress update
                    state.messages.append({
                        "role": "assistant",
                        "content": f"""🔄 **Phase {i}/6: {phase_name}**

{description}...

⏱️ **Progress:** {i*16:.0f}% complete"""
                    })
                    
                    # Brief delay between phases for demo
                    if i < len(phases):
                        time.sleep(3)
                
                # Execute the actual pipeline with 'proceed' command
                print_to_log(f"🚀 Executing actual pipeline for session {state.session_id}")
                result = pipeline.process_query('proceed', state.session_id)
                
                # Send completion message
                state.messages.append({
                    "role": "assistant",
                    "content": f"""🎉 **Fast ML Pipeline Complete!**

✅ **Target:** `{state.target_column}`
📊 **Dataset:** {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns
⏱️ **Processing Time:** ~5-7 minutes

🚀 **Your model is ready for predictions!**

{result}"""
                })
                
                print_to_log(f"✅ Fast pipeline completed successfully for session {state.session_id}")
                
            except Exception as e:
                print_to_log(f"❌ Pipeline thread error: {e}")
                import traceback
                traceback.print_exc()
                
                state.messages.append({
                    "role": "assistant",
                    "content": f"""❌ **Pipeline Error**

An error occurred during processing: {str(e)}

Please try again or use interactive mode for debugging."""
                })
        
        # Start pipeline in background thread
        pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
        pipeline_thread.start()
        print_to_log(f"🚀 Fast pipeline thread launched for session {state.session_id}")

def fast_model_agent(state: PipelineState) -> PipelineState:
    """Main entry point for fast model agent"""
    agent = FastModelAgent()
    
    # Check if we're waiting for target selection
    if getattr(state, '_awaiting_target_selection', False):
        return agent.handle_target_selection(state, state.user_query)
    
    # Otherwise start the fast model pipeline
    return agent.handle_fast_model_request(state) 