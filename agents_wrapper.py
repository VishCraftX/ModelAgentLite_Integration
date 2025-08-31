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
        
    def run(self, state: PipelineState) -> PipelineState:
        """Route to the actual working preprocessing agent"""
        if not self.available:
            print("❌ Preprocessing agent not available - falling back to basic preprocessing")
            return self._run_basic_preprocessing_fallback(state)
            
        try:
            # Save data to temp file for the working agent
            if state.raw_data is None:
                print("❌ No raw data available for preprocessing")
                return state
                
            temp_file = os.path.join(tempfile.gettempdir(), f"temp_data_{state.session_id}.csv")
            state.raw_data.to_csv(temp_file, index=False)
            
            # Determine target column
            target_column = state.target_column or "target"
            
            print(f"🚀 Launching your actual interactive preprocessing agent")
            print(f"🎯 Target column: {target_column}")
            print(f"📁 Temp file: {temp_file}")
            
            # Call your working preprocessing agent AS-IS
            # This should launch the full interactive workflow with Slack menus
            final_state = run_preprocessing_agent(
                df_path=temp_file,
                target_column=target_column,
                model_name=os.environ.get("DEFAULT_MODEL", "gpt-4o")
            )
            
            # Extract results from the working agent
            if hasattr(final_state, 'df') and final_state.df is not None:
                state.cleaned_data = final_state.df
                state.processed_data = final_state.df
                state.preprocessing_state = {
                    "completed": True,
                    "timestamp": datetime.now().isoformat(),
                    "method": "sequential_interactive",
                    "phases_completed": final_state.completed_phases if hasattr(final_state, 'completed_phases') else []
                }
                print(f"✅ Interactive preprocessing completed. Final shape: {final_state.df.shape}")
            else:
                print("⚠️ No final dataframe returned from interactive agent")
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
                
            return state
            
        except Exception as e:
            print(f"❌ Interactive preprocessing agent failed: {e}")
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
