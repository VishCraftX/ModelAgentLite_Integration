#!/usr/bin/env python3
"""
Integrated Agents for Multi-Agent ML System
Uses actual implementations from existing agent directories
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import tempfile
import json

from pipeline_state import PipelineState
from toolbox import progress_tracker, artifact_manager, user_directory_manager

# Import actual agent implementations
try:
    from preprocessing_agent_impl import (
        SequentialState as PreprocessingState,
        PreprocessingPhase,
        get_llm_from_state,
        analyze_column_comprehensive,
        analyze_missing_values_with_llm,
        apply_missing_values_treatment,
        analyze_outliers_with_llm,
        apply_outliers_treatment,
        analyze_encoding_with_llm,
        apply_encoding_treatment,
        analyze_transformations_with_llm,
        apply_transformations_treatment,
        detect_and_handle_extreme_outliers,
        get_current_data_state,
        initialize_dataset_analysis,
        run_sequential_agent,
        create_sequential_preprocessing_agent,
        export_cleaned_dataset,
        process_user_input_with_llm,
        classify_user_intent_with_llm,
        generate_overview_summary,
        get_available_actions,
        get_next_phase,
        apply_preprocessing_pipeline
    )
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Preprocessing agent implementation not available: {e}")
    PREPROCESSING_AVAILABLE = False

try:
    from feature_selection_agent_impl import (
        UserSession as FeatureSelectionSession,
        AnalysisStep,
        LLMManager,
        DataProcessor,
        AnalysisEngine,
        MenuGenerator,
        AgenticFeatureSelectionBot as FeatureSelectionBot
    )
    FEATURE_SELECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Feature selection agent implementation not available: {e}")
    FEATURE_SELECTION_AVAILABLE = False

try:
    from model_building_agent_impl import (
        LangGraphModelAgent,
        preload_ollama_models
    )
    MODEL_BUILDING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Model building agent implementation not available: {e}")
    MODEL_BUILDING_AVAILABLE = False


class BaseAgent:
    """Base class for all integrated agents"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.progress_tracker = progress_tracker
        self.artifact_manager = artifact_manager
    
    def run(self, state: PipelineState) -> PipelineState:
        """Main execution method - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement run method")
    
    def _update_progress(self, state: PipelineState, message: str, stage: str = None):
        """Helper to update progress"""
        state.current_agent = self.agent_name
        self.progress_tracker.update(state, message, stage)


class IntegratedPreprocessingAgent(BaseAgent):
    """
    Integrated Preprocessing Agent using actual SequentialPreprocessingAgent implementation
    """
    
    def __init__(self):
        super().__init__("PreprocessingAgent")
        self.preprocessing_available = PREPROCESSING_AVAILABLE
        self.langgraph_workflow = None
        if PREPROCESSING_AVAILABLE:
            try:
                self.langgraph_workflow = create_sequential_preprocessing_agent()
                if self.langgraph_workflow:
                    print(f"✅ {self.agent_name} LangGraph workflow initialized")
                else:
                    print(f"⚠️ {self.agent_name} LangGraph workflow not available")
            except Exception as e:
                print(f"⚠️ {self.agent_name} LangGraph workflow failed to initialize: {e}")
                self.langgraph_workflow = None
    
    def run(self, state: PipelineState) -> PipelineState:
        """Execute preprocessing using actual implementation"""
        self._update_progress(state, "Starting data preprocessing", "Preprocessing")
        
        try:
            # For model building requests, use basic preprocessing to avoid LLM delays
            user_query = (state.user_query or "").lower()
            use_basic = any(keyword in user_query for keyword in [
                "model", "train", "build", "lgbm", "classifier", "regressor", "predict"
            ])
            
            if not self.preprocessing_available or use_basic:
                self._update_progress(state, "Using fast basic preprocessing for model building", "Basic Preprocessing")
                cleaned_data = self._run_basic_preprocessing(state)
                if cleaned_data is not None:
                    state.cleaned_data = cleaned_data
                    state.preprocessing_state = {
                        "completed": True,
                        "timestamp": datetime.now().isoformat(),
                        "original_shape": state.raw_data.shape if state.raw_data is not None else None,
                        "cleaned_shape": cleaned_data.shape,
                        "method": "basic_fast"
                    }
                    self._update_progress(state, f"✅ Basic preprocessing completed. Shape: {cleaned_data.shape}")
                else:
                    state.preprocessing_state = {
                        "completed": False,
                        "timestamp": datetime.now().isoformat(),
                        "error": "No raw data available for preprocessing",
                        "method": "basic"
                    }
                    self._update_progress(state, "❌ No raw data available for preprocessing")
                return state
            
            # Check if we have raw data
            if state.raw_data is None:
                self._update_progress(state, "No raw data available for preprocessing")
                return state
            
            # Launch full interactive preprocessing workflow (exactly like original agent)
            if self.preprocessing_available and self.langgraph_workflow:
                self._update_progress(state, "Launching interactive preprocessing workflow", "Interactive Mode")
                return self.run_interactive_workflow(state)
            else:
                # Fallback to comprehensive preprocessing without interactivity
                self._update_progress(state, "Running comprehensive preprocessing", "Comprehensive Mode")
                preprocessing_state = self._create_preprocessing_state(state)
                cleaned_data = self._run_comprehensive_preprocessing(preprocessing_state, state)
                
                # Update pipeline state
                state.cleaned_data = cleaned_data
                state.preprocessing_state = {
                    "completed": True,
                    "timestamp": datetime.now().isoformat(),
                    "original_shape": state.raw_data.shape if state.raw_data is not None else None,
                    "cleaned_shape": cleaned_data.shape if cleaned_data is not None else None,
                    "phases_completed": preprocessing_state.completed_phases if hasattr(preprocessing_state, 'completed_phases') else []
                }
                
                # Save artifact
                if cleaned_data is not None and state.chat_session:
                    try:
                        artifact_path = self.artifact_manager.save_artifact(
                            state.chat_session, 
                            "cleaned_data.csv", 
                            cleaned_data, 
                            "dataframe"
                        )
                        if artifact_path:
                            state.artifacts = state.artifacts or {}
                            state.artifacts["cleaned_data_path"] = artifact_path
                    except Exception as e:
                        print(f"[{self.agent_name}] Failed to save artifact: {e}")
                
                self._update_progress(state, f"Preprocessing completed. Shape: {cleaned_data.shape if cleaned_data is not None else 'None'}")
            
        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            self._update_progress(state, error_msg)
            state.last_error = error_msg
            print(f"[{self.agent_name}] Error: {e}")
        
        return state
    
    def _create_preprocessing_state(self, state: PipelineState) -> Any:
        """Create preprocessing state from pipeline state"""
        if not PREPROCESSING_AVAILABLE:
            return None
        
        # Create temporary file path
        temp_path = os.path.join(tempfile.gettempdir(), f"temp_data_{state.session_id}.csv")
        
        # Determine target column
        target_column = "target"  # Default
        if state.raw_data is not None:
            # Look for common target column names
            target_candidates = ['target', 'label', 'y', 'class', 'outcome', 'default']
            for col in target_candidates:
                if col in state.raw_data.columns:
                    target_column = col
                    break
        
        preprocessing_state = PreprocessingState(
            df=state.raw_data,
            df_path=temp_path,
            target_column=target_column,
            model_name=os.getenv("DEFAULT_MODEL", "gpt-4o")
        )
        
        return preprocessing_state
    
    def _run_comprehensive_preprocessing(self, preprocessing_state: Any, pipeline_state: PipelineState) -> pd.DataFrame:
        """Run comprehensive preprocessing using actual implementation"""
        if not PREPROCESSING_AVAILABLE or preprocessing_state is None:
            return self._run_basic_preprocessing(pipeline_state)
        
        try:
            df = preprocessing_state.df.copy()
            
            # Phase 1: Overview and Analysis
            self._update_progress(pipeline_state, "Analyzing dataset structure", "Analysis")
            preprocessing_state.current_phase = PreprocessingPhase.OVERVIEW
            
            # Initialize dataset analysis
            preprocessing_state = initialize_dataset_analysis(preprocessing_state)
            
            # Analyze all columns
            for col in df.columns:
                if col != preprocessing_state.target_column:
                    analysis = analyze_column_comprehensive(df[col], df[preprocessing_state.target_column], col)
                    preprocessing_state.column_analysis[col] = analysis
            
            preprocessing_state.completed_phases.append(PreprocessingPhase.OVERVIEW)
            
            # Phase 2: Handle Missing Values
            self._update_progress(pipeline_state, "Handling missing values", "Missing Values")
            preprocessing_state.current_phase = PreprocessingPhase.MISSING_VALUES
            
            missing_results = analyze_missing_values_with_llm(preprocessing_state)
            if missing_results and 'llm_recommendations' in missing_results:
                df = apply_missing_values_treatment(df, missing_results['llm_recommendations'])
                preprocessing_state.df = df
                preprocessing_state.phase_results['missing_values'] = missing_results
            
            preprocessing_state.completed_phases.append(PreprocessingPhase.MISSING_VALUES)
            
            # Phase 3: Handle Outliers
            self._update_progress(pipeline_state, "Detecting and handling outliers", "Outliers")
            preprocessing_state.current_phase = PreprocessingPhase.OUTLIERS
            
            outlier_results = analyze_outliers_with_llm(preprocessing_state)
            if outlier_results and 'llm_recommendations' in outlier_results:
                df = apply_outliers_treatment(df, outlier_results['llm_recommendations'])
                preprocessing_state.df = df
                preprocessing_state.phase_results['outliers'] = outlier_results
            
            preprocessing_state.completed_phases.append(PreprocessingPhase.OUTLIERS)
            
            # Phase 4: Encode Categorical Variables
            self._update_progress(pipeline_state, "Encoding categorical variables", "Encoding")
            preprocessing_state.current_phase = PreprocessingPhase.ENCODING
            
            encoding_results = analyze_encoding_with_llm(preprocessing_state)
            if encoding_results and 'llm_recommendations' in encoding_results:
                df = apply_encoding_treatment(df, encoding_results['llm_recommendations'])
                preprocessing_state.df = df
                preprocessing_state.phase_results['encoding'] = encoding_results
            
            preprocessing_state.completed_phases.append(PreprocessingPhase.ENCODING)
            
            # Phase 5: Apply Transformations
            self._update_progress(pipeline_state, "Applying transformations", "Transformations")
            preprocessing_state.current_phase = PreprocessingPhase.TRANSFORMATIONS
            
            transformation_results = analyze_transformations_with_llm(preprocessing_state)
            if transformation_results and 'llm_recommendations' in transformation_results:
                df = apply_transformations_treatment(df, transformation_results['llm_recommendations'])
                preprocessing_state.df = df
                preprocessing_state.phase_results['transformations'] = transformation_results
            
            preprocessing_state.completed_phases.append(PreprocessingPhase.TRANSFORMATIONS)
            
            # Final phase completion
            preprocessing_state.current_phase = PreprocessingPhase.COMPLETION
            
            return df
            
        except Exception as e:
            print(f"[{self.agent_name}] Comprehensive preprocessing failed: {e}")
            return self._run_basic_preprocessing(pipeline_state)
    
    def run_interactive_workflow(self, state: PipelineState, user_input: str = None) -> PipelineState:
        """Run the full interactive preprocessing workflow exactly like the original agent"""
        if not PREPROCESSING_AVAILABLE or not self.langgraph_workflow:
            return self._run_basic_preprocessing_fallback(state)
        
        try:
            # Initialize or continue interactive session
            if not hasattr(state, 'interactive_session') or state.interactive_session is None:
                # Start new interactive session
                self._update_progress(state, "🚀 Starting Interactive Preprocessing Session", "Interactive Mode")
                
                # Create initial preprocessing state
                preprocessing_state = self._create_preprocessing_state(state)
                
                # Run initial overview
                result = self.langgraph_workflow.invoke(preprocessing_state)
                
                print(f"[{self.agent_name}] Interactive workflow result:")
                print(f"  - Type: {type(result)}")
                print(f"  - Has query_response: {hasattr(result, 'query_response')}")
                print(f"  - Has current_step: {hasattr(result, 'current_step')}")
                if hasattr(result, 'current_step'):
                    print(f"  - Current step: {result.current_step}")
                if hasattr(result, 'query_response'):
                    print(f"  - Query response: {result.query_response[:100] if result.query_response else 'None'}...")
                
                # Store interactive session state
                state.interactive_session = {
                    "agent_type": "preprocessing",
                    "current_state": result,
                    "active": True,
                    "session_id": state.chat_session
                }
                
                # Send initial response to user
                if hasattr(result, 'query_response') and result.query_response:
                    self._send_interactive_message(state, result.query_response)
                    state.last_response = result.query_response
                    print(f"[{self.agent_name}] Sent initial response to Slack")
                
                # If waiting for input, prompt user
                if hasattr(result, 'current_step') and result.current_step == "awaiting_user_input":
                    prompt = self._generate_input_prompt(result)
                    self._send_interactive_message(state, prompt)
                    state.last_response += f"\n\n{prompt}"
                    print(f"[{self.agent_name}] Sent interactive prompt to Slack")
                else:
                    # Force interactive prompt even if not explicitly awaiting input
                    print(f"[{self.agent_name}] Not awaiting input, but sending interactive prompt anyway")
                    prompt = self._generate_input_prompt(result)
                    self._send_interactive_message(state, prompt)
                    if state.last_response:
                        state.last_response += f"\n\n{prompt}"
                    else:
                        state.last_response = prompt
                
                return state
            
            else:
                # Continue existing interactive session
                current_preprocessing_state = state.interactive_session["current_state"]
                
                # Process user input if provided
                if user_input:
                    # Use the original agent's input processing
                    from preprocessing_agent_impl import process_user_input_with_llm
                    updated_state = process_user_input_with_llm(current_preprocessing_state, user_input)
                    
                    # Continue workflow
                    result = self.langgraph_workflow.invoke(updated_state)
                    
                    # Update session state
                    state.interactive_session["current_state"] = result
                    
                    # Send response to user
                    if hasattr(result, 'query_response') and result.query_response:
                        self._send_interactive_message(state, result.query_response)
                        state.last_response = result.query_response
                    
                    # Check if session is complete
                    if (hasattr(result, 'current_phase') and 
                        str(result.current_phase) == "PreprocessingPhase.COMPLETION"):
                        
                        # Session completed - extract final results
                        if hasattr(result, 'df') and result.df is not None:
                            state.cleaned_data = result.df
                            state.preprocessing_state = {
                                "completed": True,
                                "timestamp": datetime.now().isoformat(),
                                "original_shape": state.raw_data.shape if state.raw_data is not None else None,
                                "cleaned_shape": result.df.shape,
                                "method": "interactive_langgraph",
                                "phases_completed": getattr(result, 'completed_phases', [])
                            }
                            
                            completion_msg = f"🎉 Interactive preprocessing completed!\n\n" \
                                           f"📊 **Final Results:**\n" \
                                           f"• Original shape: {state.raw_data.shape if state.raw_data is not None else 'Unknown'}\n" \
                                           f"• Final shape: {result.df.shape}\n" \
                                           f"• Phases completed: {len(getattr(result, 'completed_phases', []))}"
                            
                            self._send_interactive_message(state, completion_msg)
                            state.last_response = completion_msg
                        
                        # Clear interactive session
                        state.interactive_session = None
                    
                    elif (hasattr(result, 'current_step') and 
                          result.current_step == "awaiting_user_input"):
                        # Still waiting for more input
                        prompt = self._generate_input_prompt(result)
                        self._send_interactive_message(state, prompt)
                        if state.last_response:
                            state.last_response += f"\n\n{prompt}"
                        else:
                            state.last_response = prompt
                
                return state
            
        except Exception as e:
            print(f"[{self.agent_name}] Interactive workflow failed: {e}")
            # Clear interactive session on error
            state.interactive_session = None
            return self._run_basic_preprocessing_fallback(state)
    
    def _send_interactive_message(self, state: PipelineState, message: str):
        """Send interactive message via Slack"""
        try:
            from toolbox import slack_manager
            if slack_manager and state.chat_session:
                # Correct parameter order: session_id, text
                slack_manager.send_message(state.chat_session, message)
                print(f"[{self.agent_name}] ✅ Sent interactive message to Slack")
            else:
                print(f"[{self.agent_name}] ❌ No slack_manager or chat_session available")
                print(f"  - slack_manager: {slack_manager}")
                print(f"  - chat_session: {state.chat_session}")
        except Exception as e:
            print(f"[{self.agent_name}] ❌ Failed to send interactive message: {e}")
            import traceback
            traceback.print_exc()
        
        # Always print to console as backup
        print(f"📤 Interactive Message: {message}")
    
    def _generate_input_prompt(self, preprocessing_state) -> str:
        """Generate appropriate input prompt based on current phase"""
        try:
            phase_name = str(preprocessing_state.current_phase).split('.')[-1].title()
        except:
            phase_name = "Processing"
        
        prompt = f"🔄 **{phase_name} Phase**\n\n"
        prompt += "💬 **Your options:**\n" \
                 "• `proceed` or `yes` - Continue with recommended approach\n" \
                 "• `skip` - Skip this phase\n" \
                 "• `modify [details]` - Change the approach\n" \
                 "• `explain` or `what` - Get more information\n" \
                 "• `summary` - Show current strategies\n\n" \
                 "**What would you like to do?**"
        
        return prompt
    
    def _run_basic_preprocessing_fallback(self, state: PipelineState) -> PipelineState:
        """Fallback to basic preprocessing"""
        processed_data = self._run_basic_preprocessing(state)
        state.processed_data = processed_data
        return state
    
    def export_dataset(self, state: PipelineState, output_path: str = None) -> str:
        """Export cleaned dataset using preprocessing agent functionality"""
        if not PREPROCESSING_AVAILABLE:
            # Fallback export
            if state.processed_data is not None:
                if output_path is None:
                    output_path = f"cleaned_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                state.processed_data.to_csv(output_path, index=False)
                return output_path
            return None
        
        try:
            # Create preprocessing state
            preprocessing_state = self._create_preprocessing_state(state)
            if hasattr(preprocessing_state, 'df') and preprocessing_state.df is not None:
                return export_cleaned_dataset(preprocessing_state, output_path)
            return None
        except Exception as e:
            print(f"[{self.agent_name}] Export failed: {e}")
            return None
    
    def _run_basic_preprocessing(self, state: PipelineState) -> pd.DataFrame:
        """Fallback basic preprocessing"""
        if state.raw_data is None:
            return None
        
        df = state.raw_data.copy()
        
        # Basic cleaning operations
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col].fillna(mode_value[0], inplace=True)
                else:
                    df[col].fillna("Unknown", inplace=True)
        
        print(f"[{self.agent_name}] Basic preprocessing completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Filled missing values in {len(numeric_columns)} numeric columns")
        print(f"  - Filled missing values in {len(categorical_columns)} categorical columns")
        
        return df


class IntegratedFeatureSelectionAgent(BaseAgent):
    """
    Integrated Feature Selection Agent using actual FeatureSelectionBot implementation
    """
    
    def __init__(self):
        super().__init__("FeatureSelectionAgent")
        self.feature_selection_available = FEATURE_SELECTION_AVAILABLE
        self.bot = None
        self.llm_manager = None
        self.data_processor = None
        self.analysis_engine = None
        self.menu_generator = None
        
        if FEATURE_SELECTION_AVAILABLE:
            try:
                # Initialize core components (adapted for unified pipeline)
                self.llm_manager = LLMManager()
                self.data_processor = DataProcessor()
                self.analysis_engine = AnalysisEngine()
                self.menu_generator = MenuGenerator()
                # Note: Not initializing full bot as per your instruction
            except Exception as e:
                print(f"[{self.agent_name}] Failed to initialize components: {e}")
                self.feature_selection_available = False
    
    def run(self, state: PipelineState) -> PipelineState:
        """Execute feature selection using actual implementation with full interactivity"""
        self._update_progress(state, "Starting feature selection", "Feature Selection")
        
        try:
            if state.cleaned_data is None:
                self._update_progress(state, "No cleaned data available for feature selection")
                return state
            
            # Check if this is actually a general query that got routed here
            if self._is_general_query(state.user_query):
                return self._handle_general_query(state)
            
            # Launch full interactive feature selection workflow (exactly like original agent)
            if self.feature_selection_available:
                self._update_progress(state, "Launching interactive feature selection workflow", "Interactive Mode")
                return self.run_interactive_workflow(state)
            else:
                # Fallback to basic feature selection
                self._update_progress(state, "Running basic feature selection", "Basic Mode")
                selected_features = self._run_basic_feature_selection(state)
                
                # Update pipeline state
                state.selected_features = selected_features
                state.feature_selection_state = {
                    "completed": True,
                    "timestamp": datetime.now().isoformat(),
                    "total_features": len(state.cleaned_data.columns),
                    "selected_features": len(selected_features) if selected_features else 0,
                    "selection_method": "basic"
                }
                
                # Save artifact
                if selected_features and state.chat_session:
                    try:
                        artifact_path = self.artifact_manager.save_artifact(
                            state.chat_session, 
                            "selected_features.json", 
                            {"selected_features": selected_features}, 
                            "json"
                        )
                        if artifact_path:
                            state.artifacts = state.artifacts or {}
                            state.artifacts["selected_features_path"] = artifact_path
                    except Exception as e:
                        print(f"[{self.agent_name}] Failed to save artifact: {e}")
                
                self._update_progress(state, f"Feature selection completed. Selected {len(selected_features) if selected_features else 0} features")
            
        except Exception as e:
            error_msg = f"Feature selection failed: {str(e)}"
            self._update_progress(state, error_msg)
            state.last_error = error_msg
            print(f"[{self.agent_name}] Error: {e}")
        
        return state
    
    def _is_general_query(self, query: str) -> bool:
        """Check if this is a general query about feature selection concepts"""
        if not query:
            return False
        
        query_lower = query.lower()
        general_patterns = [
            "what is", "what are", "explain", "tell me about", "how does", "how do",
            "describe", "definition of", "meaning of", "concept of", "different types",
            "techniques you can", "methods available", "capabilities"
        ]
        
        return any(pattern in query_lower for pattern in general_patterns)
    
    def _handle_general_query(self, state: PipelineState) -> PipelineState:
        """Handle general queries about feature selection concepts"""
        try:
            # Use LLM to generate educational response about feature selection
            import requests
            import json
            
            prompt = f"""You are an expert in feature selection and machine learning. The user asked: "{state.user_query}"

Please provide a comprehensive, educational response about feature selection concepts, techniques, and best practices. 

Include information about:
- Different feature selection methods (filter, wrapper, embedded)
- Statistical measures (IV, correlation, VIF, CSI)
- Advanced techniques (SHAP, LASSO, PCA)
- When to use each approach
- Practical considerations

Keep the response informative but accessible, with examples where helpful."""

            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "qwen2.5-coder:32b-instruct-q4_K_M",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    llm_response = result.get("response", "")
                    
                    if llm_response.strip():
                        state.last_response = llm_response
                        return state
            
            except Exception as e:
                print(f"[{self.agent_name}] LLM call failed: {e}")
            
            # Fallback response
            fallback_response = f"""🔍 **Feature Selection Overview**

Feature selection is the process of selecting the most relevant features for your machine learning model. Here are the main approaches I can help you with:

**📊 Statistical Methods:**
• **IV Analysis** - Measures predictive power of features
• **Correlation Analysis** - Identifies redundant features  
• **VIF Analysis** - Detects multicollinearity
• **CSI Analysis** - Checks feature stability over time

**🤖 Advanced Techniques:**
• **SHAP Analysis** - Model-agnostic feature importance
• **LASSO Selection** - Regularized feature selection
• **PCA Analysis** - Dimensionality reduction

**💡 Interactive Capabilities:**
• Upload your dataset and I'll guide you through the analysis
• Ask questions like "run IV analysis" or "show correlation with target"
• Get recommendations on which techniques to use
• Understand your data through exploratory analysis

Would you like to upload a dataset to start interactive feature selection, or do you have specific questions about any of these techniques?"""
            
            state.last_response = fallback_response
            return state
            
        except Exception as e:
            print(f"[{self.agent_name}] Error handling general query: {e}")
            state.last_response = "I can help you with feature selection analysis. Please upload a dataset to get started!"
            return state
    
    def run_interactive_workflow(self, state: PipelineState, user_input: str = None) -> PipelineState:
        """Run the full interactive feature selection workflow exactly like the original agent"""
        if not FEATURE_SELECTION_AVAILABLE:
            return self._run_basic_feature_selection_fallback(state)
        
        try:
            # Initialize or continue interactive session
            if not hasattr(state, 'interactive_session') or state.interactive_session is None:
                # Start new interactive feature selection session
                self._update_progress(state, "🚀 Starting Interactive Feature Selection Session", "Interactive Mode")
                
                # Create feature selection session (like original agent)
                temp_file = os.path.join(tempfile.gettempdir(), f"temp_data_{state.session_id}.csv")
                state.cleaned_data.to_csv(temp_file, index=False)
                
                from feature_selection_agent_impl import UserSession
                fs_session = UserSession(
                    file_path=temp_file,
                    file_name=f"data_{state.session_id}.csv",
                    user_id=state.chat_session,
                    original_df=state.cleaned_data.copy(),
                    current_df=state.cleaned_data.copy(),
                    current_features=list(state.cleaned_data.columns),
                    phase="analyzing"  # Start in analyzing phase
                )
                
                # Set target column if available
                if state.target_column:
                    fs_session.target_column = state.target_column
                    fs_session.phase = "waiting_input"
                else:
                    fs_session.phase = "need_target"
                
                # Store interactive session state
                state.interactive_session = {
                    "agent_type": "feature_selection",
                    "fs_session": fs_session,
                    "active": True,
                    "session_id": state.chat_session
                }
                
                # Generate and send initial menu
                from feature_selection_agent_impl import MenuGenerator
                if fs_session.phase == "need_target":
                    initial_msg = "🎯 Please specify your target column to start feature selection analysis."
                else:
                    initial_msg = MenuGenerator.generate_main_menu(fs_session)
                
                self._send_interactive_message(state, initial_msg)
                state.last_response = initial_msg
                
                return state
            
            else:
                # Continue existing interactive session
                fs_session = state.interactive_session["fs_session"]
                
                # Process user input if provided
                if user_input:
                    # Use the original agent's bot logic
                    from feature_selection_agent_impl import AgenticFeatureSelectionBot
                    
                    # Create a temporary bot instance for processing
                    bot = AgenticFeatureSelectionBot()
                    
                    # Mock Slack 'say' function to capture responses
                    responses = []
                    def mock_say(message, thread_ts=None):
                        responses.append(message)
                    
                    # Handle the user input
                    if fs_session.phase == "need_target":
                        bot.handle_target_selection(fs_session, user_input, mock_say)
                    else:
                        bot.handle_analysis_request(fs_session, user_input, mock_say)
                    
                    # Send all responses to user
                    for response in responses:
                        self._send_interactive_message(state, response)
                    
                    # Update state with latest response
                    if responses:
                        state.last_response = responses[-1]
                    
                    # Check if analysis is complete (user said "proceed" or similar)
                    if (fs_session.phase == "completed" or 
                        (hasattr(fs_session, 'analysis_chain') and 
                         len(fs_session.analysis_chain) > 0 and
                         "proceed" in user_input.lower())):
                        
                        # Extract final selected features
                        selected_features = fs_session.current_features
                        
                        state.selected_features = selected_features
                        state.feature_selection_state = {
                            "completed": True,
                            "timestamp": datetime.now().isoformat(),
                            "original_features": len(fs_session.original_df.columns) if fs_session.original_df is not None else 0,
                            "selected_features": len(selected_features),
                            "method": "interactive_agentic",
                            "analysis_chain": [step.type for step in fs_session.analysis_chain] if hasattr(fs_session, 'analysis_chain') else []
                        }
                        
                        completion_msg = f"🎉 Interactive feature selection completed!\n\n" \
                                       f"📊 **Final Results:**\n" \
                                       f"• Original features: {len(fs_session.original_df.columns) if fs_session.original_df is not None else 0}\n" \
                                       f"• Selected features: {len(selected_features)}\n" \
                                       f"• Analysis steps: {len(fs_session.analysis_chain) if hasattr(fs_session, 'analysis_chain') else 0}"
                        
                        self._send_interactive_message(state, completion_msg)
                        state.last_response = completion_msg
                        
                        # Clear interactive session
                        state.interactive_session = None
                
                return state
            
        except Exception as e:
            print(f"[{self.agent_name}] Interactive workflow failed: {e}")
            # Clear interactive session on error
            state.interactive_session = None
            return self._run_basic_feature_selection_fallback(state)
    
    def _send_interactive_message(self, state: PipelineState, message: str):
        """Send interactive message via Slack"""
        try:
            from toolbox import slack_manager
            if slack_manager and state.chat_session:
                # Correct parameter order: session_id, text
                slack_manager.send_message(state.chat_session, message)
                print(f"[{self.agent_name}] ✅ Sent interactive message to Slack")
            else:
                print(f"[{self.agent_name}] ❌ No slack_manager or chat_session available")
                print(f"  - slack_manager: {slack_manager}")
                print(f"  - chat_session: {state.chat_session}")
        except Exception as e:
            print(f"[{self.agent_name}] ❌ Failed to send interactive message: {e}")
            import traceback
            traceback.print_exc()
        
        # Always print to console as backup
        print(f"📤 Interactive Message: {message}")
    
    def _run_basic_feature_selection_fallback(self, state: PipelineState) -> PipelineState:
        """Fallback to basic feature selection"""
        selected_features = self._run_basic_feature_selection(state)
        state.selected_features = selected_features
        return state
    
    def _run_comprehensive_feature_selection(self, state: PipelineState) -> List[str]:
        """Run comprehensive feature selection using actual implementation with all components"""
        try:
            # Create feature selection session
            temp_file = os.path.join(tempfile.gettempdir(), f"temp_data_{state.session_id}.csv")
            state.cleaned_data.to_csv(temp_file, index=False)
            
            session = FeatureSelectionSession(
                file_path=temp_file,
                file_name=f"data_{state.session_id}.csv",
                user_id=state.chat_session,
                original_df=state.cleaned_data.copy(),
                current_df=state.cleaned_data.copy(),
                current_features=list(state.cleaned_data.columns)
            )
            
            # Use DataProcessor to enhance data processing
            if self.data_processor:
                self._update_progress(state, "Processing data with DataProcessor", "Data Processing")
                processed_df = self.data_processor.process_dataframe(session.current_df)
                session.current_df = processed_df
            
            # Determine target column
            target_candidates = ['target', 'label', 'y', 'class', 'outcome', 'default']
            target_column = None
            for col in target_candidates:
                if col in state.cleaned_data.columns:
                    target_column = col
                    break
            
            if target_column:
                session.target_column = target_column
                session.phase = "analyzing"
            
            # Run comprehensive analysis if target is available
            if target_column:
                # Use AnalysisEngine for sophisticated feature analysis
                if self.analysis_engine:
                    self._update_progress(state, "Running AnalysisEngine feature analysis", "Advanced Analysis")
                    try:
                        analysis_results = self.analysis_engine.analyze_features(session)
                        session.analysis_results = analysis_results
                        
                        # Use LLMManager for intelligent recommendations
                        if self.llm_manager:
                            self._update_progress(state, "Generating LLM-based recommendations", "AI Recommendations")
                            recommendations = self.llm_manager.get_feature_recommendations(session)
                            session.recommendations = recommendations
                            
                            # Apply recommendations to select features
                            selected_features = []
                            for feature, recommendation in recommendations.items():
                                if recommendation.get('include', True) and feature != target_column:
                                    selected_features.append(feature)
                            
                            if selected_features:
                                return selected_features
                    except Exception as e:
                        print(f"[{self.agent_name}] AnalysisEngine failed: {e}")
                
                # Fallback: Run IV analysis using bot if available
                if hasattr(self, 'bot') and self.bot:
                    self._update_progress(state, "Running IV analysis", "IV Analysis")
                    try:
                        iv_result = self.bot.run_iv_analysis(session)
                        
                        if iv_result and "iv_scores" in iv_result:
                            # Select features with IV > 0.02 (common threshold)
                            iv_threshold = 0.02
                            selected_features = []
                            for feature, iv_score in iv_result["iv_scores"].items():
                                if iv_score > iv_threshold and feature != target_column:
                                    selected_features.append(feature)
                            
                            if selected_features:
                                return selected_features
                    except Exception as e:
                        print(f"[{self.agent_name}] IV analysis failed: {e}")
            
            # Fallback to correlation analysis
            self._update_progress(state, "Running correlation analysis", "Correlation Analysis")
            numeric_features = list(state.cleaned_data.select_dtypes(include=[np.number]).columns)
            
            if target_column and target_column in numeric_features:
                # Select features with high correlation to target
                correlations = state.cleaned_data[numeric_features].corr()[target_column].abs()
                selected_features = correlations[correlations > 0.1].index.tolist()
                if target_column in selected_features:
                    selected_features.remove(target_column)
                return selected_features
            
            # Final fallback
            return self._run_basic_feature_selection(state)
            
        except Exception as e:
            print(f"[{self.agent_name}] Comprehensive feature selection failed: {e}")
            return self._run_basic_feature_selection(state)
    
    def _run_basic_feature_selection(self, state: PipelineState) -> List[str]:
        """Fallback basic feature selection"""
        df = state.cleaned_data
        
        # Select numeric columns with some variance
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = []
        
        for col in numeric_columns:
            if df[col].nunique() > 1:  # Has variance
                missing_pct = df[col].isnull().sum() / len(df)
                if missing_pct < 0.5:  # Less than 50% missing
                    selected_features.append(col)
        
        print(f"[{self.agent_name}] Basic feature selection completed:")
        print(f"  - Selected {len(selected_features)} numeric features")
        print(f"  - Features: {selected_features[:5]}{'...' if len(selected_features) > 5 else ''}")
        
        return selected_features


class IntegratedModelBuildingAgent(BaseAgent):
    """
    Integrated Model Building Agent using actual LangGraphModelAgent implementation
    """
    
    def __init__(self):
        super().__init__("ModelBuildingAgent")
        self.model_building_available = MODEL_BUILDING_AVAILABLE
        self.agent = None
        if MODEL_BUILDING_AVAILABLE:
            try:
                self.agent = LangGraphModelAgent()
            except Exception as e:
                print(f"[{self.agent_name}] Failed to initialize LangGraphModelAgent: {e}")
                self.model_building_available = False
    
    def run(self, state: PipelineState) -> PipelineState:
        """Execute model building using actual implementation"""
        self._update_progress(state, "Starting model building", "Model Building")
        
        try:
            # Check if this is actually a general query that got routed here
            if self._is_general_query(state.user_query):
                return self._handle_general_query(state)
            
            if state.cleaned_data is None:
                self._update_progress(state, "No cleaned data available for model building")
                return state
            
            if not state.selected_features:
                self._update_progress(state, "No features selected for model building")
                return state
            
            # Build model and get structured result
            if not self.model_building_available or self.agent is None:
                model_result = self._run_basic_model_building(state)
            else:
                model_result = self._run_comprehensive_model_building(state)
            
            # Store result in models section with unique ID
            if model_result:
                model_id = self._store_model_result(state, model_result)
                
                # Update backward compatibility field
                if isinstance(model_result, dict) and 'model' in model_result:
                    state.trained_model = model_result['model']
                else:
                    state.trained_model = model_result
                
                # Generate and send Slack summary
                if model_id and state.models and model_id in state.models:
                    summary = self._format_model_summary(state.models[model_id])
                    self._update_progress(state, summary)
            
            state.model_building_state = {
                "completed": True,
                "timestamp": datetime.now().isoformat(),
                "model_type": "comprehensive" if self.model_building_available else "basic",
                "features_used": len(state.selected_features)
            }
            
            self._update_progress(state, "Model building completed successfully")
            
        except Exception as e:
            error_msg = f"Model building failed: {str(e)}"
            self._update_progress(state, error_msg)
            state.last_error = error_msg
            print(f"[{self.agent_name}] Error: {e}")
        
        return state
    
    def _run_comprehensive_model_building(self, state: PipelineState):
        """Run comprehensive model building using actual implementation"""
        try:
            # Load data into the agent
            session_id = state.chat_session or "default_session"
            self.agent.load_data(state.cleaned_data, session_id)
            
            # Create model building query
            query = state.user_query or "build lgbm model"
            
            # Check if this is a "use existing" request and we have model data in pipeline state
            if self._is_use_existing_request(query) and state.models:
                return self._handle_use_existing_model(state, query)
            
            # Process query through the agent for new model building
            self._update_progress(state, "Processing model building request", "Model Training")
            result = self.agent.process_query(query, session_id)
            
            # Extract results
            if result and "execution_result" in result:
                execution_result = result["execution_result"]
                
                # Check if model was trained successfully
                user_state = self.agent.user_states.get(session_id, {})
                if user_state.get("has_existing_model"):
                    model_path = user_state.get("model_path")
                    if model_path and os.path.exists(model_path):
                        # Load the trained model
                        import joblib
                        trained_model = joblib.load(model_path)
                        return trained_model
                
                # Return execution result as model representation
                return execution_result
            
            # Fallback to basic model building
            return self._run_basic_model_building(state)
            
        except Exception as e:
            print(f"[{self.agent_name}] Comprehensive model building failed: {e}")
            return self._run_basic_model_building(state)
    
    def _run_basic_model_building(self, state: PipelineState):
        """Fallback basic model building"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.metrics import accuracy_score, r2_score
            
            df = state.cleaned_data
            features = state.selected_features
            
            # Prepare data
            X = df[features]
            
            # Find target column
            target_col = None
            target_candidates = ['target', 'label', 'y', 'class', 'outcome', 'default']
            for col in target_candidates:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col is None:
                non_feature_cols = [col for col in df.columns if col not in features]
                if non_feature_cols:
                    target_col = non_feature_cols[0]
                else:
                    return "No target column available for model training"
            
            y = df[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Choose model type
            if y.nunique() <= 10:  # Classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                metric_name = "accuracy"
            else:  # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                metric_name = "r2_score"
            
            # Train model
            self._update_progress(state, f"Training {type(model).__name__}", "Training")
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            print(f"[{self.agent_name}] Basic model building completed:")
            print(f"  - Model type: {type(model).__name__}")
            print(f"  - Features used: {len(features)}")
            print(f"  - Train {metric_name}: {train_score:.4f}")
            print(f"  - Test {metric_name}: {test_score:.4f}")
            
            # Return structured result
            result = {
                'model': model,
                'model_type': type(model).__name__,
                'params': model.get_params() if hasattr(model, 'get_params') else {},
                'train_score': float(train_score),
                'test_score': float(test_score),
                'metric_name': metric_name,
                'features_used': len(features),
                'target_column': target_col,
                'data_split': {
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                }
            }
            return result
            
        except Exception as e:
            print(f"[{self.agent_name}] Basic model building failed: {e}")
            return f"Model building failed: {str(e)}"
    
    def _store_model_result(self, state: PipelineState, model_result) -> str:
        """Store model result in state.models with unique ID"""
        import time
        
        # Generate unique model ID
        timestamp = int(time.time())
        model_count = len(state.models) + 1
        model_id = f"model_{model_count:03d}_{timestamp}"
        
        # Initialize models dict if needed
        if state.models is None:
            state.models = {}
        
        # Structure the result according to the design
        if isinstance(model_result, dict) and 'model' in model_result:
            # Comprehensive model building result
            structured_result = {
                "model_id": model_id,
                "type": model_result.get('model_type', 'Unknown'),
                "params": model_result.get('params', {}),
                "metrics": {
                    k: v for k, v in model_result.items() 
                    if k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'log_loss', 'train_score', 'test_score']
                },
                "model_path": self._save_model_artifact(state, model_result['model'], model_id),
                "timestamp": datetime.now().isoformat(),
                "features_used": len(state.selected_features) if state.selected_features else 0,
                "data_info": {
                    "train_size": model_result.get('data_split', {}).get('train_size'),
                    "test_size": model_result.get('data_split', {}).get('test_size')
                }
            }
        else:
            # Basic model building or single model object
            model_obj = model_result if hasattr(model_result, 'predict') else None
            structured_result = {
                "model_id": model_id,
                "type": type(model_obj).__name__ if model_obj else "Unknown",
                "params": {},
                "metrics": {},
                "model_path": self._save_model_artifact(state, model_obj, model_id) if model_obj else None,
                "timestamp": datetime.now().isoformat(),
                "features_used": len(state.selected_features) if state.selected_features else 0,
                "data_info": {}
            }
        
        # Store in state
        state.models[model_id] = structured_result
        
        # Save rank-order results as CSV if available
        if isinstance(model_result, dict) and 'rank_ordering_table' in model_result:
            self._save_rank_order_csv(state, model_id, model_result['rank_ordering_table'])
        
        # Set as best model if it's the first one or has better metrics
        if not state.best_model or self._is_better_model(structured_result, state.models.get(state.best_model, {})):
            state.best_model = model_id
            print(f"[{self.agent_name}] Set {model_id} as best model")
        
        return model_id
    
    def _save_model_artifact(self, state: PipelineState, model_obj, model_id: str) -> str:
        """Save model as artifact in organized directory structure"""
        if not model_obj or not state.chat_session:
            return None
        
        try:
            import joblib
            
            # Create organized directory structure: /models/model_001/
            models_dir = user_directory_manager.get_models_dir(state.chat_session)
            model_dir = os.path.join(models_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Use consistent .joblib format
            model_filename = "model.joblib"
            
            # Save model to organized path
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(model_obj, model_path)
            
            # Update artifacts tracking
            state.artifacts = state.artifacts or {}
            state.artifacts[f"{model_id}_path"] = model_path
            
            print(f"[{self.agent_name}] Saved model to: {model_path}")
            return model_path
                
        except Exception as e:
            print(f"[{self.agent_name}] Failed to save model artifact for {model_id}: {e}")
        
        return None
    
    def _save_rank_order_csv(self, state: PipelineState, model_id: str, rank_ordering_table: List[Dict]) -> str:
        """Save rank-order table as CSV in model directory"""
        try:
            # Get model directory
            models_dir = user_directory_manager.get_models_dir(state.chat_session)
            model_dir = os.path.join(models_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Convert to DataFrame and save as CSV
            import pandas as pd
            rank_df = pd.DataFrame(rank_ordering_table)
            csv_path = os.path.join(model_dir, "rank_order.csv")
            rank_df.to_csv(csv_path, index=False)
            
            # Update model info in state
            if model_id in state.models:
                state.models[model_id]['rank_order'] = {
                    "artifact": csv_path,
                    "computed_at": datetime.now().isoformat()
                }
            
            print(f"[{self.agent_name}] Saved rank-order CSV to: {csv_path}")
            return csv_path
            
        except Exception as e:
            print(f"[{self.agent_name}] Failed to save rank-order CSV for {model_id}: {e}")
            return None
    
    def _is_general_query(self, query: str) -> bool:
        """Check if query is a general conversational query"""
        if not query:
            return False
        
        query_lower = query.lower()
        general_patterns = [
            "hello", "hi", "hey", "greetings", "morning", "afternoon", "evening",
            "how are you", "what's up", "what can you do", "help", "capabilities",
            "tell me about", "explain", "what is", "how does", "how do",
            "thanks", "thank you", "bye", "goodbye"
        ]
        
        # Check if it's a general question about ML concepts
        ml_concept_patterns = [
            "what is lgbm", "how does lgbm work", "explain lgbm", "tell me about lgbm",
            "what is lightgbm", "how does lightgbm work", "explain lightgbm",
            "what is random forest", "how does random forest work",
            "what is machine learning", "explain machine learning",
            "what is classification", "what is regression"
        ]
        
        return (any(pattern in query_lower for pattern in general_patterns) or 
                any(pattern in query_lower for pattern in ml_concept_patterns))
    
    def _handle_general_query(self, state: PipelineState) -> PipelineState:
        """Handle general conversational queries using LLM"""
        try:
            # Import LLM functionality
            try:
                import ollama
                LLM_AVAILABLE = True
            except ImportError:
                LLM_AVAILABLE = False
            
            if not LLM_AVAILABLE:
                # Fallback response when LLM not available
                state.last_response = "Hello! I'm your ML assistant. I can help you with data preprocessing, feature selection, and model building. How can I assist you today?"
                return state
            
            query = state.user_query or ""
            
            # Generate context-aware response
            if state.raw_data is not None:
                context_prompt = f"The user said: '{query}'. I have their dataset with {state.raw_data.shape[0]:,} rows and {state.raw_data.shape[1]} columns. Respond naturally and conversationally about machine learning concepts or general assistance. If they ask about specific ML algorithms like LGBM, explain them clearly."
            else:
                context_prompt = f"The user said: '{query}'. Respond naturally and conversationally as an AI assistant specialized in machine learning and data science. If they ask about ML algorithms or concepts, explain them clearly."
            
            print(f"[{self.agent_name}] Generating conversational response for: '{query}'")
            
            # Use LLM for conversational response
            response = ollama.chat(
                model="qwen2.5-coder:32b-instruct-q4_K_M",  # Same model as ModelBuildingAgent
                messages=[
                    {"role": "system", "content": "You are a specialized AI assistant for data science and machine learning. You help users understand ML concepts, build models, analyze data, and work with datasets. When explaining algorithms like LightGBM, Random Forest, etc., be clear and educational. Keep responses conversational but informative."},
                    {"role": "user", "content": context_prompt}
                ]
            )
            
            generated_response = response["message"]["content"].strip()
            state.last_response = generated_response
            
            print(f"[{self.agent_name}] Generated response: {generated_response[:100]}...")
            
        except Exception as e:
            print(f"[{self.agent_name}] Error generating conversational response: {e}")
            # Fallback response
            if "lgbm" in (state.user_query or "").lower() or "lightgbm" in (state.user_query or "").lower():
                state.last_response = "LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework that uses tree-based learning algorithms. It's designed to be distributed and efficient with faster training speed and higher efficiency, lower memory usage, and better accuracy compared to other boosting frameworks."
            else:
                state.last_response = "Hello! I'm your ML assistant. I can help you with data preprocessing, feature selection, model building, and explaining machine learning concepts. How can I assist you today?"
        
        return state
    
    def _is_use_existing_request(self, query: str) -> bool:
        """Check if query is requesting to use existing model"""
        if not query:
            return False
        
        query_lower = query.lower()
        use_existing_patterns = [
            "use this model", "use the model", "with this model", "for this model",
            "use existing", "existing model", "current model", "built model", "trained model",
            "use previous", "previous model", "last model", "latest model",
            "show plot", "visualize", "display", "rank ordering", "rank order",
            "build segments", "build deciles", "build buckets", "build rankings",
            "populate", "score", "predict", "classify"
        ]
        
        return any(pattern in query_lower for pattern in use_existing_patterns)
    
    def _handle_use_existing_model(self, state: PipelineState, query: str):
        """Handle use existing model requests by pulling data from pipeline state"""
        try:
            # Find the target model to use
            target_model = self._get_target_model_from_state(state, query)
            if not target_model:
                print(f"[{self.agent_name}] No suitable model found in pipeline state")
                # Fallback to normal model building agent processing
                result = self.agent.process_query(query, state.chat_session)
                return result.get("execution_result") if result else None
            
            model_id = target_model['model_id']
            print(f"[{self.agent_name}] Using existing model from pipeline state: {model_id}")
            
            # Check what data is requested and what's available
            query_lower = query.lower()
            
            # If requesting rank-order and it exists, return it
            if any(keyword in query_lower for keyword in ["rank", "ranking", "rank order", "decile", "segment"]):
                if 'rank_order' in target_model:
                    rank_order_info = target_model['rank_order']
                    # Load rank-order data from CSV if available
                    csv_path = rank_order_info.get('artifact')
                    if csv_path and os.path.exists(csv_path):
                        import pandas as pd
                        rank_df = pd.read_csv(csv_path)
                        return {
                            'model': None,  # Don't return model object for rank-order queries
                            'rank_ordering_table': rank_df.to_dict('records'),
                            'model_id': model_id,
                            'model_type': target_model.get('type', 'Unknown'),
                            'from_pipeline_state': True
                        }
            
            # If requesting validation metrics, return existing metrics
            if any(keyword in query_lower for keyword in ["metrics", "validation", "performance", "accuracy"]):
                metrics = target_model.get('metrics', {})
                return {
                    'model': None,  # Don't return model object for metrics queries
                    'model_id': model_id,
                    'model_type': target_model.get('type', 'Unknown'),
                    'from_pipeline_state': True,
                    **metrics  # Include all existing metrics
                }
            
            # For other requests, might need to load the actual model and generate missing data
            model_path = target_model.get('model_path')
            if model_path and os.path.exists(model_path):
                import joblib
                model_obj = joblib.load(model_path)
                
                # Return the model with existing metadata
                return {
                    'model': model_obj,
                    'model_id': model_id,
                    'model_type': target_model.get('type', 'Unknown'),
                    'model_path': model_path,
                    'from_pipeline_state': True,
                    **target_model.get('metrics', {})
                }
            
            # If we can't find the model file, fallback to normal processing
            print(f"[{self.agent_name}] Model file not found: {model_path}, falling back to normal processing")
            result = self.agent.process_query(query, state.chat_session)
            return result.get("execution_result") if result else None
            
        except Exception as e:
            print(f"[{self.agent_name}] Error handling existing model: {e}")
            # Fallback to normal processing
            result = self.agent.process_query(query, state.chat_session)
            return result.get("execution_result") if result else None
    
    def _get_target_model_from_state(self, state: PipelineState, query: str) -> Optional[Dict]:
        """Get target model from pipeline state based on query"""
        if not state.models or len(state.models) == 0:
            return None
        
        query_lower = query.lower()
        
        # Check for specific model ID in query
        for model_id, model_info in state.models.items():
            if model_id.lower() in query_lower:
                return model_info
        
        # Check for "best" keyword
        if "best" in query_lower and state.best_model and state.best_model in state.models:
            return state.models[state.best_model]
        
        # Check for "previous", "existing", "last", "latest" keywords
        if any(keyword in query_lower for keyword in ["previous", "existing", "last", "latest", "current"]):
            # Get the most recent model (highest timestamp)
            latest_model = max(state.models.values(), key=lambda x: x.get('timestamp', ''))
            return latest_model
        
        # Default: use best model if available, otherwise latest
        if state.best_model and state.best_model in state.models:
            return state.models[state.best_model]
        else:
            return list(state.models.values())[-1]  # Most recently added
    
    def _is_better_model(self, new_model: dict, current_best: dict) -> bool:
        """Compare models to determine if new model is better"""
        if not current_best:
            return True
        
        new_metrics = new_model.get('metrics', {})
        current_metrics = current_best.get('metrics', {})
        
        # Priority order for comparison
        comparison_metrics = ['roc_auc', 'f1_score', 'accuracy', 'test_score']
        
        for metric in comparison_metrics:
            if metric in new_metrics and metric in current_metrics:
                return new_metrics[metric] > current_metrics[metric]
        
        # If no comparable metrics, newer is better
        return True
    
    def _format_model_summary(self, model_info: dict) -> str:
        """Format model info for user-friendly Slack summary"""
        model_id = model_info.get('model_id', 'Unknown')
        model_type = model_info.get('type', 'Unknown')
        metrics = model_info.get('metrics', {})
        
        summary_parts = [f"✅ Model Built: {model_type} ({model_id})"]
        
        # Add key metrics
        if 'accuracy' in metrics:
            summary_parts.append(f"- Accuracy: {metrics['accuracy']:.3f}")
        if 'precision' in metrics:
            summary_parts.append(f"- Precision: {metrics['precision']:.3f}")
        if 'recall' in metrics:
            summary_parts.append(f"- Recall: {metrics['recall']:.3f}")
        if 'f1_score' in metrics:
            summary_parts.append(f"- F1 Score: {metrics['f1_score']:.3f}")
        if 'roc_auc' in metrics:
            summary_parts.append(f"- ROC-AUC: {metrics['roc_auc']:.3f}")
        
        # Add regression metrics if available
        if 'test_score' in metrics:
            summary_parts.append(f"- Test Score: {metrics['test_score']:.3f}")
        
        # Add model path
        if model_info.get('model_path'):
            summary_parts.append(f"📂 Path: {model_info['model_path']}")
        
        return "\n".join(summary_parts)



# Global agent instances
preprocessing_agent = IntegratedPreprocessingAgent()
feature_selection_agent = IntegratedFeatureSelectionAgent()
model_building_agent = IntegratedModelBuildingAgent()
