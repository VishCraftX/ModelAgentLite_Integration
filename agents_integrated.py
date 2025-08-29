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
            self.langgraph_workflow = create_sequential_preprocessing_agent()
    
    def run(self, state: PipelineState) -> PipelineState:
        """Execute preprocessing using actual implementation"""
        self._update_progress(state, "Starting data preprocessing", "Preprocessing")
        
        try:
            if not self.preprocessing_available:
                cleaned_data = self._run_basic_preprocessing(state)
                if cleaned_data is not None:
                    state.cleaned_data = cleaned_data
                    state.preprocessing_state = {
                        "completed": True,
                        "timestamp": datetime.now().isoformat(),
                        "original_shape": state.raw_data.shape if state.raw_data is not None else None,
                        "cleaned_shape": cleaned_data.shape,
                        "method": "basic"
                    }
                    self._update_progress(state, f"Basic preprocessing completed. Shape: {cleaned_data.shape}")
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
            
            # Create preprocessing state from pipeline state
            preprocessing_state = self._create_preprocessing_state(state)
            
            # Run comprehensive preprocessing
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
        """Run the interactive LangGraph preprocessing workflow"""
        if not PREPROCESSING_AVAILABLE or not self.langgraph_workflow:
            return self._run_basic_preprocessing_fallback(state)
        
        try:
            # Create preprocessing state from pipeline state
            preprocessing_state = self._create_preprocessing_state(state)
            
            # If user input provided, process it
            if user_input:
                preprocessing_state = process_user_input_with_llm(preprocessing_state, user_input)
            
            # Run the LangGraph workflow
            result = self.langgraph_workflow.invoke(preprocessing_state)
            
            # Update pipeline state with results
            if hasattr(result, 'df') and result.df is not None:
                state.processed_data = result.df
                state.preprocessing_state = {
                    "completed": True,
                    "timestamp": datetime.now().isoformat(),
                    "original_shape": state.raw_data.shape if state.raw_data is not None else None,
                    "processed_shape": result.df.shape,
                    "method": "interactive_langgraph",
                    "phases_completed": result.completed_phases,
                    "phase_results": result.phase_results
                }
            
            # Store any response for user
            if hasattr(result, 'query_response') and result.query_response:
                state.last_response = result.query_response
            
            return state
            
        except Exception as e:
            print(f"[{self.agent_name}] Interactive workflow failed: {e}")
            return self._run_basic_preprocessing_fallback(state)
    
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
        """Execute feature selection using actual implementation"""
        self._update_progress(state, "Starting feature selection", "Feature Selection")
        
        try:
            if state.cleaned_data is None:
                self._update_progress(state, "No cleaned data available for feature selection")
                return state
            
            if not self.feature_selection_available or self.bot is None:
                selected_features = self._run_basic_feature_selection(state)
            else:
                selected_features = self._run_comprehensive_feature_selection(state)
            
            # Update pipeline state
            state.selected_features = selected_features
            state.feature_selection_state = {
                "completed": True,
                "timestamp": datetime.now().isoformat(),
                "total_features": len(state.cleaned_data.columns),
                "selected_features": len(selected_features) if selected_features else 0,
                "selection_method": "comprehensive" if self.feature_selection_available else "basic"
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
            if state.cleaned_data is None:
                self._update_progress(state, "No cleaned data available for model building")
                return state
            
            if not state.selected_features:
                self._update_progress(state, "No features selected for model building")
                return state
            
            if not self.model_building_available or self.agent is None:
                trained_model = self._run_basic_model_building(state)
            else:
                trained_model = self._run_comprehensive_model_building(state)
            
            # Update pipeline state
            state.trained_model = trained_model
            state.model_building_state = {
                "completed": True,
                "timestamp": datetime.now().isoformat(),
                "model_type": "comprehensive" if self.model_building_available else "basic",
                "features_used": len(state.selected_features)
            }
            
            # Save artifact
            if trained_model and state.chat_session:
                try:
                    import joblib
                    model_filename = f"trained_model_{int(datetime.now().timestamp())}.joblib"
                    temp_path = os.path.join(tempfile.gettempdir(), model_filename)
                    joblib.dump(trained_model, temp_path)
                    
                    with open(temp_path, 'rb') as f:
                        artifact_path = self.artifact_manager.save_artifact(
                            state.chat_session, 
                            model_filename, 
                            f.read(), 
                            "binary"
                        )
                    if artifact_path:
                        state.artifacts = state.artifacts or {}
                        state.artifacts["trained_model_path"] = artifact_path
                except Exception as e:
                    print(f"[{self.agent_name}] Failed to save model artifact: {e}")
            
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
            
            # Process query through the agent
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
            
            return model
            
        except Exception as e:
            print(f"[{self.agent_name}] Basic model building failed: {e}")
            return f"Model building failed: {str(e)}"


# Global agent instances
preprocessing_agent = IntegratedPreprocessingAgent()
feature_selection_agent = IntegratedFeatureSelectionAgent()
model_building_agent = IntegratedModelBuildingAgent()
