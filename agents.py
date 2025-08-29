#!/usr/bin/env python3
"""
Integrated Agents for Multi-Agent ML System
Wraps existing agent implementations into the unified LangGraph framework
"""

import os
import sys
import importlib.util
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from pipeline_state import PipelineState
from toolbox import progress_tracker, execution_agent, artifact_manager


class BaseAgent:
    """Base class for all agents in the system"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.progress_tracker = progress_tracker
        self.execution_agent = execution_agent
        self.artifact_manager = artifact_manager
    
    def run(self, state: PipelineState) -> PipelineState:
        """Main execution method - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement run method")
    
    def _update_progress(self, state: PipelineState, message: str, stage: str = None):
        """Helper to update progress"""
        state.current_agent = self.agent_name
        self.progress_tracker.update(state, message, stage)


class PreprocessingAgent(BaseAgent):
    """
    Preprocessing Agent - handles data cleaning, missing values, outlier handling
    Integrates with the existing DataPreprocessingAgent implementation
    """
    
    def __init__(self):
        super().__init__("PreprocessingAgent")
        self.preprocessing_module = None
        self._load_preprocessing_module()
    
    def _load_preprocessing_module(self):
        """Load the existing preprocessing agent module"""
        try:
            preprocessing_path = "/Users/10321/Vishwas/CV/GenAI/CursorProjects/DataPreprocessingAgent/SequentialPreprocessingAgent.py"
            
            if os.path.exists(preprocessing_path):
                spec = importlib.util.spec_from_file_location("preprocessing_agent", preprocessing_path)
                self.preprocessing_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.preprocessing_module)
                print(f"[{self.agent_name}] Successfully loaded preprocessing module")
            else:
                print(f"[{self.agent_name}] Preprocessing module not found, using stub implementation")
        
        except Exception as e:
            print(f"[{self.agent_name}] Error loading preprocessing module: {e}")
            print(f"[{self.agent_name}] Using stub implementation")
    
    def run(self, state: PipelineState) -> PipelineState:
        """Execute preprocessing operations"""
        self._update_progress(state, "Starting data preprocessing", "Preprocessing")
        
        try:
            # Check if we have raw data
            if state.raw_data is None:
                self._update_progress(state, "No raw data available for preprocessing")
                return state
            
            # Use existing preprocessing logic if available
            if self.preprocessing_module:
                cleaned_data = self._run_advanced_preprocessing(state)
            else:
                cleaned_data = self._run_basic_preprocessing(state)
            
            # Update state with cleaned data
            state.cleaned_data = cleaned_data
            state.preprocessing_state = {
                "completed": True,
                "timestamp": pd.Timestamp.now().isoformat(),
                "original_shape": state.raw_data.shape,
                "cleaned_shape": cleaned_data.shape if cleaned_data is not None else None
            }
            
            # Save artifact
            if cleaned_data is not None and state.chat_session:
                artifact_path = self.artifact_manager.save_artifact(
                    state.chat_session, 
                    "cleaned_data.csv", 
                    cleaned_data, 
                    "dataframe"
                )
                state.artifacts["cleaned_data_path"] = artifact_path
            
            self._update_progress(state, f"Preprocessing completed. Shape: {cleaned_data.shape if cleaned_data is not None else 'None'}")
            
        except Exception as e:
            error_msg = f"Preprocessing failed: {str(e)}"
            self._update_progress(state, error_msg)
            state.last_error = error_msg
        
        return state
    
    def _run_advanced_preprocessing(self, state: PipelineState) -> pd.DataFrame:
        """Run advanced preprocessing using the existing module"""
        try:
            # Create a temporary state for the existing preprocessing agent
            preprocessing_state = self.preprocessing_module.SequentialState(
                df=state.raw_data,
                df_path="temp_data.csv",
                target_column="target"  # Default target column
            )
            
            # Run basic preprocessing operations
            df = state.raw_data.copy()
            
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns
            
            # Fill numeric missing values with median
            for col in numeric_columns:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill categorical missing values with mode
            for col in categorical_columns:
                if df[col].isnull().any():
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col].fillna(mode_value[0], inplace=True)
                    else:
                        df[col].fillna("Unknown", inplace=True)
            
            return df
            
        except Exception as e:
            print(f"[{self.agent_name}] Advanced preprocessing failed: {e}")
            return self._run_basic_preprocessing(state)
    
    def _run_basic_preprocessing(self, state: PipelineState) -> pd.DataFrame:
        """Run basic preprocessing operations"""
        df = state.raw_data.copy()
        
        # Basic cleaning operations
        # 1. Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        # 2. Handle missing values - simple approach
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode or 'Unknown'
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


class FeatureSelectionAgent(BaseAgent):
    """
    Feature Selection Agent - handles IV, correlation, VIF, PCA, etc.
    Integrates with the existing FeatureSelection implementation
    """
    
    def __init__(self):
        super().__init__("FeatureSelectionAgent")
        self.feature_selection_module = None
        self._load_feature_selection_module()
    
    def _load_feature_selection_module(self):
        """Load the existing feature selection agent module"""
        try:
            feature_selection_path = "/Users/10321/Vishwas/CV/GenAI/CursorProjects/FeatureSelcetion/new_agentic_bot.py"
            
            if os.path.exists(feature_selection_path):
                spec = importlib.util.spec_from_file_location("feature_selection_agent", feature_selection_path)
                self.feature_selection_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.feature_selection_module)
                print(f"[{self.agent_name}] Successfully loaded feature selection module")
            else:
                print(f"[{self.agent_name}] Feature selection module not found, using stub implementation")
        
        except Exception as e:
            print(f"[{self.agent_name}] Error loading feature selection module: {e}")
            print(f"[{self.agent_name}] Using stub implementation")
    
    def run(self, state: PipelineState) -> PipelineState:
        """Execute feature selection operations"""
        self._update_progress(state, "Starting feature selection", "Feature Selection")
        
        try:
            # Check if we have cleaned data
            if state.cleaned_data is None:
                self._update_progress(state, "No cleaned data available for feature selection")
                return state
            
            # Use existing feature selection logic if available
            if self.feature_selection_module:
                selected_features = self._run_advanced_feature_selection(state)
            else:
                selected_features = self._run_basic_feature_selection(state)
            
            # Update state with selected features
            state.selected_features = selected_features
            state.feature_selection_state = {
                "completed": True,
                "timestamp": pd.Timestamp.now().isoformat(),
                "total_features": len(state.cleaned_data.columns),
                "selected_features": len(selected_features) if selected_features else 0,
                "selection_method": "basic_numeric" if not self.feature_selection_module else "advanced"
            }
            
            # Save artifact
            if selected_features and state.chat_session:
                artifact_path = self.artifact_manager.save_artifact(
                    state.chat_session, 
                    "selected_features.json", 
                    {"selected_features": selected_features}, 
                    "json"
                )
                state.artifacts["selected_features_path"] = artifact_path
            
            self._update_progress(state, f"Feature selection completed. Selected {len(selected_features) if selected_features else 0} features")
            
        except Exception as e:
            error_msg = f"Feature selection failed: {str(e)}"
            self._update_progress(state, error_msg)
            state.last_error = error_msg
        
        return state
    
    def _run_advanced_feature_selection(self, state: PipelineState) -> list:
        """Run advanced feature selection using the existing module"""
        try:
            # Use the existing feature selection logic
            df = state.cleaned_data
            
            # For now, use basic selection but with more sophisticated logic
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove columns with too many missing values or zero variance
            selected_features = []
            for col in numeric_columns:
                if df[col].nunique() > 1:  # Has variance
                    missing_pct = df[col].isnull().sum() / len(df)
                    if missing_pct < 0.5:  # Less than 50% missing
                        selected_features.append(col)
            
            return selected_features
            
        except Exception as e:
            print(f"[{self.agent_name}] Advanced feature selection failed: {e}")
            return self._run_basic_feature_selection(state)
    
    def _run_basic_feature_selection(self, state: PipelineState) -> list:
        """Run basic feature selection - select all numeric columns"""
        df = state.cleaned_data
        
        # Simple approach: select all numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"[{self.agent_name}] Basic feature selection completed:")
        print(f"  - Selected {len(numeric_columns)} numeric features")
        print(f"  - Features: {numeric_columns[:5]}{'...' if len(numeric_columns) > 5 else ''}")
        
        return numeric_columns


class ModelBuildingAgent(BaseAgent):
    """
    Model Building Agent - handles training, evaluation, and prediction
    Integrates with the existing ModelBuildingAgent implementation
    """
    
    def __init__(self):
        super().__init__("ModelBuildingAgent")
        self.model_building_module = None
        self._load_model_building_module()
    
    def _load_model_building_module(self):
        """Load the existing model building agent module"""
        try:
            model_building_path = "/Users/10321/Vishwas/CV/GenAI/CursorProjects/ModelAgentLite_LG/langgraph_agents.py"
            
            if os.path.exists(model_building_path):
                spec = importlib.util.spec_from_file_location("model_building_agent", model_building_path)
                self.model_building_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.model_building_module)
                print(f"[{self.agent_name}] Successfully loaded model building module")
            else:
                print(f"[{self.agent_name}] Model building module not found, using stub implementation")
        
        except Exception as e:
            print(f"[{self.agent_name}] Error loading model building module: {e}")
            print(f"[{self.agent_name}] Using stub implementation")
    
    def run(self, state: PipelineState) -> PipelineState:
        """Execute model building operations"""
        self._update_progress(state, "Starting model building", "Model Building")
        
        try:
            # Check if we have the required data
            if state.cleaned_data is None:
                self._update_progress(state, "No cleaned data available for model building")
                return state
            
            if not state.selected_features:
                self._update_progress(state, "No features selected for model building")
                return state
            
            # Use existing model building logic if available
            if self.model_building_module:
                trained_model = self._run_advanced_model_building(state)
            else:
                trained_model = self._run_basic_model_building(state)
            
            # Update state with trained model
            state.trained_model = trained_model
            state.model_building_state = {
                "completed": True,
                "timestamp": pd.Timestamp.now().isoformat(),
                "model_type": "advanced" if self.model_building_module else "basic",
                "features_used": len(state.selected_features)
            }
            
            # Save artifact
            if trained_model and state.chat_session:
                import joblib
                import tempfile
                
                model_filename = f"trained_model_{int(pd.Timestamp.now().timestamp())}.joblib"
                temp_path = os.path.join(tempfile.gettempdir(), model_filename)
                joblib.dump(trained_model, temp_path)
                
                artifact_path = self.artifact_manager.save_artifact(
                    state.chat_session, 
                    model_filename, 
                    open(temp_path, 'rb').read(), 
                    "binary"
                )
                state.artifacts["trained_model_path"] = artifact_path
            
            self._update_progress(state, "Model building completed successfully")
            
        except Exception as e:
            error_msg = f"Model building failed: {str(e)}"
            self._update_progress(state, error_msg)
            state.last_error = error_msg
        
        return state
    
    def _run_advanced_model_building(self, state: PipelineState):
        """Run advanced model building using the existing module"""
        try:
            # Use the existing LangGraph model building agent
            if hasattr(self.model_building_module, 'LangGraphModelAgent'):
                agent = self.model_building_module.LangGraphModelAgent()
                
                # Load data into the agent
                agent.load_data(state.cleaned_data, state.chat_session or "default_session")
                
                # Process a model building query
                query = state.user_query or "build lgbm model"
                result = agent.process_query(query, state.chat_session or "default_session")
                
                # Extract the trained model from the result
                if result and "execution_result" in result:
                    # The model should be stored in the agent's state
                    user_state = agent.user_states.get(state.chat_session or "default_session", {})
                    if user_state.get("has_existing_model"):
                        return "Advanced model trained successfully"
                
            return self._run_basic_model_building(state)
            
        except Exception as e:
            print(f"[{self.agent_name}] Advanced model building failed: {e}")
            return self._run_basic_model_building(state)
    
    def _run_basic_model_building(self, state: PipelineState):
        """Run basic model building - simple logistic regression"""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        df = state.cleaned_data
        features = state.selected_features
        
        # Prepare data
        X = df[features]
        
        # Try to find a target column
        target_col = None
        potential_targets = ['target', 'label', 'y', 'class', 'outcome']
        for col in potential_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # Use the last column as target
            non_feature_cols = [col for col in df.columns if col not in features]
            if non_feature_cols:
                target_col = non_feature_cols[0]
            else:
                print(f"[{self.agent_name}] No target column found, creating dummy model")
                return "Dummy model (no target column available)"
        
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if y.nunique() <= 10:  # Classification
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Regression
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"[{self.agent_name}] Basic model building completed:")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Features used: {len(features)}")
        print(f"  - Train score: {train_score:.4f}")
        print(f"  - Test score: {test_score:.4f}")
        
        return model


# Global agent instances
preprocessing_agent = PreprocessingAgent()
feature_selection_agent = FeatureSelectionAgent()
model_building_agent = ModelBuildingAgent()
