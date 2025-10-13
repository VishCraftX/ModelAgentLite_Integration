#!/usr/bin/env python3
from print_to_log import print_to_log
"""
Global Pipeline State for Multi-Agent ML System
Defines the shared state that flows through all agents in the LangGraph pipeline
"""

from typing import Optional, List, Any, Dict
import pandas as pd
from pydantic import BaseModel, Field
from datetime import datetime
import json
import os
import pickle
import tempfile
import time


class PipelineState(BaseModel):
    """
    Global state schema for the multi-agent ML pipeline.
    This state is shared across all agents and persists throughout the session.
    """
    # Core data fields
    raw_data: Optional[pd.DataFrame] = None
    cleaned_data: Optional[pd.DataFrame] = None
    processed_data: Optional[pd.DataFrame] = None  # Add missing field
    selected_features: Optional[List[str]] = None
    target_column: Optional[str] = None  # Add missing target column field
    trained_model: Optional[Any] = None  # Keep for backward compatibility
    
    # Multi-model storage
    models: Optional[Dict[str, Dict]] = Field(default_factory=dict)  # Store multiple models with metrics
    best_model: Optional[str] = None  # Pointer to the best model ID
    
    # Session management
    artifacts: Optional[Dict] = Field(default_factory=dict)
    chat_session: Optional[str] = None
    progress: Optional[str] = None
    
    # Query tracking
    user_query: Optional[str] = None
    last_code: Optional[str] = None
    last_error: Optional[str] = None
    last_response: Optional[str] = None
    
    # Agent-specific state extensions
    preprocessing_state: Optional[Dict] = Field(default_factory=dict)
    preprocessing_strategies: Optional[Dict] = Field(default_factory=dict)  # Reusable preprocessing strategies
    feature_selection_state: Optional[Dict] = Field(default_factory=dict)
    model_building_state: Optional[Dict] = Field(default_factory=dict)
    
    # Interactive session management
    interactive_session: Optional[Dict] = None
    
    # Slack session management
    slack_session_info: Optional[Dict] = Field(default_factory=dict)
    pending_slack_message: Optional[str] = None  # Message to send after CSV save
    
    # Execution context
    current_agent: Optional[str] = None
    execution_history: List[Dict] = Field(default_factory=list)
    
    # Session metadata
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # File upload management
    pending_file_uploads: Optional[Dict] = None
    
    # Predictions dataset
    predictions_dataset: Optional[pd.DataFrame] = None  # Dataset with predictions column
    
    # Non-data science query context
    non_data_science_context: Optional[Dict] = None  # Context for intelligent non-DS responses
    
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            pd.DataFrame: lambda v: v.to_dict() if v is not None else None,
            datetime: lambda v: v.isoformat() if v is not None else None
        }
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def update_progress(self, message: str, agent: str = None):
        """Update progress with timestamp and agent info"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if agent:
            self.progress = f"[{timestamp}] {agent}: {message}"
        else:
            self.progress = f"[{timestamp}] {message}"
        self.updated_at = datetime.now()
    
    def add_execution_record(self, agent: str, action: str, result: Any = None, error: str = None):
        """Add execution record to history"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "result": str(result) if result is not None else None,
            "error": error,
            "user_query": self.user_query
        }
        self.execution_history.append(record)
        self.updated_at = datetime.now()
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of current data state"""
        summary = {
            "has_raw_data": self.raw_data is not None,
            "has_cleaned_data": self.cleaned_data is not None,
            "has_selected_features": self.selected_features is not None and len(self.selected_features) > 0,
            "has_trained_model": self.trained_model is not None,
            "session_id": self.session_id,
            "chat_session": self.chat_session
        }
        
        # Add current agent information for context-aware summaries
        if hasattr(self, 'interactive_session') and self.interactive_session:
            summary["current_agent"] = self.interactive_session.get("agent_type")
            if self.interactive_session.get("agent_type") == "feature_selection":
                summary["feature_selection_active"] = True
        
        # Add feature selection state info
        if hasattr(self, 'feature_selection_state') and self.feature_selection_state:
            summary["feature_selection_active"] = self.feature_selection_state.get("session_active", False)
        
        if self.raw_data is not None:
            summary["raw_data_shape"] = self.raw_data.shape
            summary["raw_data_columns"] = list(self.raw_data.columns)
        
        if self.cleaned_data is not None:
            # ✅ Use actual feature selection count if available, otherwise use cleaned_data shape
            if (hasattr(self, 'feature_selection_state') and 
                self.feature_selection_state and 
                self.feature_selection_state.get("current_feature_count")):
                # Use the cleaned feature count from feature selection
                actual_feature_count = self.feature_selection_state["current_feature_count"]
                summary["cleaned_data_shape"] = (self.cleaned_data.shape[0], actual_feature_count)
                summary["actual_feature_count"] = actual_feature_count
                summary["original_feature_count"] = self.cleaned_data.shape[1]
            else:
                # Use original cleaned_data shape
                summary["cleaned_data_shape"] = self.cleaned_data.shape
            
            summary["cleaned_data_columns"] = list(self.cleaned_data.columns)
        
        if self.selected_features:
            summary["selected_features_count"] = len(self.selected_features)
            summary["selected_features"] = self.selected_features
        
        return summary
    
    def save_preprocessing_strategy(self, phase: str, phase_results: Dict, target_column: str = None, original_columns: List[str] = None):
        """Save preprocessing strategy for a completed phase"""
        from datetime import datetime
        
        if not self.preprocessing_strategies:
            # Initialize strategy structure with metadata
            self.preprocessing_strategies = {
                "strategy_metadata": {
                    "created_date": datetime.now().isoformat(),
                    "target_column": target_column or self.target_column,
                    "original_columns": original_columns or [],
                    "processing_order": [],
                    "user_overrides": {}
                },
                "outlier_strategies": {},
                "missing_value_strategies": {},
                "encoding_strategies": {},
                "transformation_strategies": {}
            }
        
        # Add phase to processing order if not already there
        if phase not in self.preprocessing_strategies["strategy_metadata"]["processing_order"]:
            self.preprocessing_strategies["strategy_metadata"]["processing_order"].append(phase)
        
        # Save phase-specific strategies
        if phase == "outliers":
            self._save_outlier_strategies(phase_results)
        elif phase == "missing_values":
            self._save_missing_value_strategies(phase_results)
        elif phase == "encoding":
            self._save_encoding_strategies(phase_results)
        elif phase == "transformations":
            self._save_transformation_strategies(phase_results)
        
        print_to_log(f"✅ Saved {phase} strategies to session state")
    
    def _save_outlier_strategies(self, phase_results: Dict):
        """Save outlier treatment strategies"""
        recommendations = phase_results.get('llm_recommendations', {})
        
        for col, rec in recommendations.items():
            if isinstance(rec, dict):
                self.preprocessing_strategies["outlier_strategies"][col] = {
                    "treatment": rec.get('treatment', 'keep'),
                    "method": "iqr",  # Default method used in analysis
                    "parameters": {
                        "iqr_multiplier": 1.5,
                        "lower_percentile": 1,
                        "upper_percentile": 99
                    },
                    "reasoning": rec.get('reasoning', 'No reasoning provided')
                }
    
    def _save_missing_value_strategies(self, phase_results: Dict):
        """Save missing value treatment strategies"""
        recommendations = phase_results.get('llm_recommendations', {})
        
        for col, rec in recommendations.items():
            if isinstance(rec, dict):
                strategy = rec.get('strategy', 'median')
                strategy_data = {
                    "strategy": strategy,
                    "parameters": {},
                    "computed_values": {},
                    "reasoning": rec.get('reasoning', 'No reasoning provided')
                }
                
                # Store constant value if specified
                if strategy == 'constant' and 'constant_value' in rec:
                    strategy_data["parameters"]["constant_value"] = rec['constant_value']
                
                # Store computed values for later use (will be computed during application)
                if hasattr(self, 'cleaned_data') and self.cleaned_data is not None and col in self.cleaned_data.columns:
                    col_data = self.cleaned_data[col]
                    if strategy == 'mean':
                        strategy_data["computed_values"]["mean_value"] = col_data.mean()
                    elif strategy == 'median':
                        strategy_data["computed_values"]["median_value"] = col_data.median()
                    elif strategy == 'mode' and not col_data.mode().empty:
                        strategy_data["computed_values"]["mode_value"] = col_data.mode().iloc[0]
                
                self.preprocessing_strategies["missing_value_strategies"][col] = strategy_data
    
    def _save_encoding_strategies(self, phase_results: Dict):
        """Save encoding strategies"""
        recommendations = phase_results.get('llm_recommendations', {})
        
        for col, rec in recommendations.items():
            if isinstance(rec, dict):
                strategy = rec.get('strategy', 'label_encoding')
                strategy_data = {
                    "strategy": strategy,
                    "encoders": {},
                    "parameters": {
                        "handle_unknown": "ignore"
                    },
                    "reasoning": rec.get('reasoning', 'No reasoning provided')
                }
                
                # For label encoding, we'll compute the mapping during application
                # For now, just store the strategy choice
                if strategy in ['onehot_encoding', 'onehot']:
                    strategy_data["parameters"]["drop_first"] = False
                elif strategy in ['target_encoding', 'target']:
                    strategy_data["parameters"]["handle_unknown"] = "global_mean"
                
                self.preprocessing_strategies["encoding_strategies"][col] = strategy_data
    
    def _save_transformation_strategies(self, phase_results: Dict):
        """Save transformation strategies"""
        recommendations = phase_results.get('llm_recommendations', {})
        
        # Initialize transformation_strategies if not present
        if "transformation_strategies" not in self.preprocessing_strategies:
            self.preprocessing_strategies["transformation_strategies"] = {}
        
        for col, rec in recommendations.items():
            if isinstance(rec, dict):
                # Fix key mismatch: LLM uses 'strategy' key, not 'transformation'
                transformation = rec.get('strategy') or rec.get('transformation') or rec.get('transformation_type') or 'standardize'
                strategy_data = {
                    "transformation": transformation,
                    "parameters": {},
                    "reasoning": rec.get('reasoning', 'No reasoning provided')
                }
                
                # Store computed parameters for later use
                if hasattr(self, 'cleaned_data') and self.cleaned_data is not None and col in self.cleaned_data.columns:
                    col_data = self.cleaned_data[col]
                    
                    if transformation == 'standardize':
                        strategy_data["parameters"]["mean"] = col_data.mean()
                        strategy_data["parameters"]["std"] = col_data.std()
                    elif transformation == 'normalize':
                        strategy_data["parameters"]["min"] = col_data.min()
                        strategy_data["parameters"]["max"] = col_data.max()
                    elif transformation == 'robust_scale':
                        strategy_data["parameters"]["median"] = col_data.median()
                        strategy_data["parameters"]["iqr"] = col_data.quantile(0.75) - col_data.quantile(0.25)
                    elif transformation == 'log':
                        strategy_data["parameters"]["min_value"] = float(col_data.min())
                        strategy_data["parameters"]["requires_positive"] = True
                        strategy_data["parameters"]["transformation_formula"] = 'log(x)' if col_data.min() > 0 else 'log1p(x - min + 1)'
                    elif transformation == 'log1p':
                        strategy_data["parameters"]["min_value"] = float(col_data.min())
                        strategy_data["parameters"]["transformation_formula"] = 'log1p(x)'
                    elif transformation == 'sqrt':
                        strategy_data["parameters"]["min_value"] = float(col_data.min())
                        strategy_data["parameters"]["transformation_formula"] = 'sqrt(x - min)' if col_data.min() < 0 else 'sqrt(x)'
                    elif transformation == 'square':
                        strategy_data["parameters"]["transformation_formula"] = 'x^2'
                        strategy_data["parameters"]["original_range"] = [float(col_data.min()), float(col_data.max())]
                    elif transformation == 'box_cox':
                        strategy_data["parameters"]["min_value"] = float(col_data.min())
                        strategy_data["parameters"]["requires_positive"] = True
                        strategy_data["parameters"]["transformation_formula"] = 'box_cox(x)' if col_data.min() > 0 else 'box_cox(x + shift)'
                    elif transformation == 'yeo_johnson':
                        strategy_data["parameters"]["transformation_formula"] = 'yeo_johnson(x)'
                        strategy_data["parameters"]["handles_negative"] = True
                    elif transformation == 'quantile':
                        # Compute quantile transformation parameters
                        quantiles = [i/100.0 for i in range(0, 101, 5)]  # Every 5th percentile
                        quantile_values = [float(col_data.quantile(q)) for q in quantiles]
                        strategy_data["parameters"]["quantiles"] = quantiles
                        strategy_data["parameters"]["quantile_values"] = quantile_values
                        strategy_data["parameters"]["n_quantiles"] = len(quantiles)
                    elif transformation == 'none':
                        strategy_data["parameters"]["no_transformation"] = True
                
                self.preprocessing_strategies["transformation_strategies"][col] = strategy_data
    
    def has_preprocessing_strategies(self) -> bool:
        """Check if preprocessing strategies are saved"""
        return bool(self.preprocessing_strategies and 
                   any(self.preprocessing_strategies.get(key, {}) for key in 
                       ["outlier_strategies", "missing_value_strategies", "encoding_strategies", "transformation_strategies"]))
    
    def get_strategy_summary(self) -> str:
        """Get a human-readable summary of saved strategies"""
        if not self.has_preprocessing_strategies():
            return "No preprocessing strategies saved."
        
        summary = ["📋 Saved Preprocessing Strategies:\n"]
        
        metadata = self.preprocessing_strategies.get("strategy_metadata", {})
        if metadata.get("created_date"):
            summary.append(f"Created: {metadata['created_date']}")
        if metadata.get("target_column"):
            summary.append(f"Target: {metadata['target_column']}")
        
        # Count strategies by phase
        outlier_count = len(self.preprocessing_strategies.get("outlier_strategies", {}))
        missing_count = len(self.preprocessing_strategies.get("missing_value_strategies", {}))
        encoding_count = len(self.preprocessing_strategies.get("encoding_strategies", {}))
        transform_count = len(self.preprocessing_strategies.get("transformation_strategies", {}))
        
        summary.append(f"\nStrategy Counts:")
        summary.append(f"• Outlier treatments: {outlier_count} columns")
        summary.append(f"• Missing value strategies: {missing_count} columns") 
        summary.append(f"• Encoding strategies: {encoding_count} columns")
        summary.append(f"• Transformations: {transform_count} columns")
        
        return "\n".join(summary)
    
    def add_pending_file_upload(self, file_info: Dict[str, Any]):
        """Add a file to the pending uploads list"""
        if self.pending_file_uploads is None:
            self.pending_file_uploads = {"files": []}
        
        if "files" not in self.pending_file_uploads:
            self.pending_file_uploads["files"] = []
        
        self.pending_file_uploads["files"].append(file_info)
        print_to_log(f"📎 Added file to pending uploads: {file_info.get('path', 'unknown')}")
    
    def add_pending_plot_upload(self, plot_path: str, title: str = "Generated Plot", comment: str = "Generated plot"):
        """Add a plot file to pending uploads"""
        if os.path.exists(plot_path):
            file_info = {
                "path": plot_path,
                "title": title,
                "comment": comment,
                "type": "plot",
                "timestamp": datetime.now().isoformat()
            }
            self.add_pending_file_upload(file_info)
        else:
            print_to_log(f"⚠️ Plot file not found: {plot_path}")
    
    def process_pending_file_uploads(self) -> bool:
        """Process all pending file uploads and return True if any uploads were processed"""
        if not self.pending_file_uploads or not self.pending_file_uploads.get("files"):
            return False
        
        try:
            from toolbox import slack_manager
            files_to_upload = self.pending_file_uploads["files"]
            
            print_to_log(f"🔍 UPLOAD DEBUG: Processing {len(files_to_upload)} pending file uploads...")
            
            uploaded_count = 0
            for file_info in files_to_upload:
                try:
                    file_path = file_info.get("path")
                    title = file_info.get("title", "Generated File")
                    comment = file_info.get("comment", "Generated file")
                    
                    if file_path and os.path.exists(file_path):
                        print_to_log(f"📤 Uploading {title}: {file_path}")
                        slack_manager.upload_file(
                            session_id=self.chat_session,
                            file_path=file_path,
                            title=title,
                            comment=comment
                        )
                        print_to_log(f"✅ Successfully uploaded {title}: {file_path}")
                        uploaded_count += 1
                    else:
                        print_to_log(f"⚠️ File not found: {file_path}")
                except Exception as e:
                    print_to_log(f"❌ Failed to upload file {file_info.get('path', 'unknown')}: {e}")
            
            # Clear pending uploads after processing
            self.pending_file_uploads = None
            print_to_log(f"✅ Processed {uploaded_count} file uploads")
            return uploaded_count > 0
            
        except Exception as e:
            print_to_log(f"❌ Failed to process pending file uploads: {e}")
            import traceback
            print_to_log(f"🔍 UPLOAD DEBUG: Full traceback: {traceback.format_exc()}")
            return False
    
    def get_pending_upload_summary(self) -> str:
        """Get a summary of pending file uploads"""
        if not self.pending_file_uploads or not self.pending_file_uploads.get("files"):
            return "No pending file uploads"
        
        files = self.pending_file_uploads["files"]
        file_list = []
        for file_info in files:
            file_path = file_info.get("path", "unknown")
            title = file_info.get("title", "Generated File")
            file_list.append(f"• {title} ({os.path.basename(file_path)})")
        
        return f"Pending uploads ({len(files)} files):\n" + "\n".join(file_list)
    
    def add_predictions_to_dataset(self, predictions: List, prediction_column_name: str = "predictions", probabilities: List = None):
        """
        Add predictions and probabilities as new columns to the dataset
        
        Args:
            predictions: List or array of predictions (classes)
            prediction_column_name: Name for the predictions column
            probabilities: Optional list or array of prediction probabilities
        """
        if self.cleaned_data is not None:
            # Use cleaned data if available
            self.predictions_dataset = self.cleaned_data.copy()
        elif self.raw_data is not None:
            # Fall back to raw data if no cleaned data
            self.predictions_dataset = self.raw_data.copy()
        else:
            print_to_log("⚠️ No data available to add predictions to")
            return False
        
        # Add predictions column
        self.predictions_dataset[prediction_column_name] = predictions
        print_to_log(f"✅ Added predictions column '{prediction_column_name}' to dataset")
        
        # Add probabilities columns if provided
        if probabilities is not None:
            import numpy as np
            probabilities_array = np.array(probabilities)
            
            if probabilities_array.ndim == 2:  # Multi-class probabilities
                n_classes = probabilities_array.shape[1]
                
                if n_classes == 2:  # Binary classification
                    # For binary classification, use probability of class 1 (positive class)
                    prob_column_name = f"{prediction_column_name}_probability"
                    self.predictions_dataset[prob_column_name] = probabilities_array[:, 1]
                    print_to_log(f"✅ Added positive class (class 1) probability column '{prob_column_name}' to dataset")
                else:  # Multi-class classification
                    # For multi-class, use the probability of the predicted class
                    predicted_probs = []
                    for i, pred in enumerate(predictions):
                        pred_idx = int(pred) if pred < n_classes else 0
                        predicted_probs.append(probabilities_array[i, pred_idx])
                    
                    prob_column_name = f"{prediction_column_name}_probability"
                    self.predictions_dataset[prob_column_name] = predicted_probs
                    print_to_log(f"✅ Added predicted class probability column '{prob_column_name}' to dataset")
            else:  # Single probability array
                prob_column_name = f"{prediction_column_name}_probability"
                self.predictions_dataset[prob_column_name] = probabilities_array
                print_to_log(f"✅ Added probability column '{prob_column_name}' to dataset")
        
        return True
    
    def save_predictions_dataset(self, file_path: str = None, model_name: str = None) -> str:
        """
        Save the predictions dataset to a file
        
        Args:
            file_path: Optional custom file path
            model_name: Optional model name to include in filename
            
        Returns:
            Path to the saved file
        """
        if self.predictions_dataset is None:
            print_to_log("⚠️ No predictions dataset available to save")
            return None
        
        if file_path is None:
            # Generate default filename with model name if available
            timestamp = int(time.time())
            if model_name:
                file_path = f"predictions_dataset_{model_name}_{timestamp}.csv"
            else:
                file_path = f"predictions_dataset_{timestamp}.csv"
        
        try:
            self.predictions_dataset.to_csv(file_path, index=False)
            print_to_log(f"✅ Predictions dataset saved to: {file_path}")
            return file_path
        except Exception as e:
            print_to_log(f"❌ Failed to save predictions dataset: {e}")
            return None
    
    def get_predictions_summary(self) -> str:
        """Get a summary of the predictions dataset"""
        if self.predictions_dataset is None:
            return "No predictions dataset available"
        
        shape = self.predictions_dataset.shape
        columns = list(self.predictions_dataset.columns)
        
        return f"Predictions dataset: {shape[0]} rows × {shape[1]} columns\nColumns: {', '.join(columns)}"

    def __getitem__(self, key):
        """Enable dictionary-style access for PipelineState."""
        if key == 'response':
            return self.last_response
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Enable dictionary-style access for PipelineState."""
        if key == 'response':
            self.last_response = value
        else:
            setattr(self, key, value)


class StateManager:
    """
    Manages persistence and retrieval of pipeline states
    """
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # Check for environment variable first
            base_dir = os.environ.get("MAL_STATES_DIR")
            if base_dir is None:
                # Use a more reliable directory location
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="mal_integration_states_")
                base_dir = temp_dir
        
        self.base_dir = base_dir
        try:
            os.makedirs(self.base_dir, exist_ok=True)
        except PermissionError:
            # Fallback to user's home directory if /tmp has permission issues
            fallback_dir = os.path.expanduser("~/mal_integration_states")
            print_to_log(f"⚠️ Permission denied for {self.base_dir}, using fallback: {fallback_dir}")
            self.base_dir = fallback_dir
            os.makedirs(self.base_dir, exist_ok=True)
    
    def save_state(self, state: PipelineState) -> str:
        """Save pipeline state to disk"""
        if not state.session_id:
            state.session_id = f"session_{int(datetime.now().timestamp())}"
        
        session_dir = os.path.join(self.base_dir, state.session_id)
        try:
            os.makedirs(session_dir, exist_ok=True)
        except PermissionError as e:
            print_to_log(f"❌ Permission error creating session directory: {e}")
            # Try creating in a more accessible location
            fallback_session_dir = os.path.expanduser(f"~/mal_integration_sessions/{state.session_id}")
            print_to_log(f"🔄 Using fallback directory: {fallback_session_dir}")
            os.makedirs(fallback_session_dir, exist_ok=True)
            session_dir = fallback_session_dir
        
        # Save state metadata (without DataFrames and non-serializable objects)
        state_dict = state.dict()
        
        # Handle DataFrames separately
        if state.raw_data is not None:
            raw_data_path = os.path.join(session_dir, "raw_data.pkl")
            state.raw_data.to_pickle(raw_data_path)
            state_dict["raw_data"] = raw_data_path
        
        if state.cleaned_data is not None:
            cleaned_data_path = os.path.join(session_dir, "cleaned_data.pkl")
            state.cleaned_data.to_pickle(cleaned_data_path)
            state_dict["cleaned_data"] = cleaned_data_path
        
        if state.predictions_dataset is not None:
            predictions_data_path = os.path.join(session_dir, "predictions_dataset.pkl")
            state.predictions_dataset.to_pickle(predictions_data_path)
            state_dict["predictions_dataset"] = predictions_data_path
        
        if state.processed_data is not None:
            processed_data_path = os.path.join(session_dir, "processed_data.pkl")
            state.processed_data.to_pickle(processed_data_path)
            state_dict["processed_data"] = processed_data_path
        
        # Handle trained model
        if state.trained_model is not None:
            model_path = os.path.join(session_dir, "trained_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(state.trained_model, f)
            state_dict["trained_model"] = model_path
        
        # Handle pending_file_uploads (ensure it's JSON serializable)
        if state.pending_file_uploads is not None:
            # pending_file_uploads should only contain file paths and metadata, no model objects
            state_dict["pending_file_uploads"] = state.pending_file_uploads
        
        # Save state metadata
        state_file = os.path.join(session_dir, "state.json")
        with open(state_file, 'w') as f:
            json.dump(state_dict, f, indent=2, default=str)
        
        return session_dir
    
    def load_state(self, session_id: str) -> Optional[PipelineState]:
        """Load pipeline state from disk"""
        session_dir = os.path.join(self.base_dir, session_id)
        state_file = os.path.join(session_dir, "state.json")
        
        if not os.path.exists(state_file):
            return None
        
        try:
            with open(state_file, 'r') as f:
                state_dict = json.load(f)
            
            # Load DataFrames
            if isinstance(state_dict.get("raw_data"), str) and os.path.exists(state_dict["raw_data"]):
                state_dict["raw_data"] = pd.read_pickle(state_dict["raw_data"])
            else:
                state_dict["raw_data"] = None
            
            if isinstance(state_dict.get("cleaned_data"), str) and os.path.exists(state_dict["cleaned_data"]):
                state_dict["cleaned_data"] = pd.read_pickle(state_dict["cleaned_data"])
            else:
                state_dict["cleaned_data"] = None
            
            if isinstance(state_dict.get("predictions_dataset"), str) and os.path.exists(state_dict["predictions_dataset"]):
                state_dict["predictions_dataset"] = pd.read_pickle(state_dict["predictions_dataset"])
            else:
                state_dict["predictions_dataset"] = None
            
            if isinstance(state_dict.get("processed_data"), str) and os.path.exists(state_dict["processed_data"]):
                state_dict["processed_data"] = pd.read_pickle(state_dict["processed_data"])
            else:
                state_dict["processed_data"] = None
            
            # Load trained model
            if isinstance(state_dict.get("trained_model"), str) and os.path.exists(state_dict["trained_model"]):
                with open(state_dict["trained_model"], 'rb') as f:
                    state_dict["trained_model"] = pickle.load(f)
            else:
                state_dict["trained_model"] = None
            
            # Convert datetime strings back to datetime objects
            if state_dict.get("created_at"):
                state_dict["created_at"] = datetime.fromisoformat(state_dict["created_at"])
            if state_dict.get("updated_at"):
                state_dict["updated_at"] = datetime.fromisoformat(state_dict["updated_at"])
            
            return PipelineState(**state_dict)
        
        except Exception as e:
            print_to_log(f"Error loading state for session {session_id}: {e}")
            return None
    
    def list_sessions(self) -> List[str]:
        """List all available session IDs"""
        if not os.path.exists(self.base_dir):
            return []
        
        sessions = []
        for item in os.listdir(self.base_dir):
            session_dir = os.path.join(self.base_dir, item)
            if os.path.isdir(session_dir) and os.path.exists(os.path.join(session_dir, "state.json")):
                sessions.append(item)
        
        return sorted(sessions)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data"""
        import shutil
        
        session_dir = os.path.join(self.base_dir, session_id)
        if os.path.exists(session_dir):
            try:
                shutil.rmtree(session_dir)
                return True
            except Exception as e:
                print_to_log(f"Error deleting session {session_id}: {e}")
                return False
        return False
    
    def get_session_directory_path(self, session_id: str) -> str:
        """Get the directory path for a session"""
        return os.path.join(self.base_dir, session_id)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up sessions older than specified hours"""
        
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        for session_id in self.list_sessions():
            session_dir = os.path.join(self.base_dir, session_id)
            state_file = os.path.join(session_dir, "state.json")
            
            if os.path.exists(state_file):
                file_mtime = os.path.getmtime(state_file)
                if file_mtime < cutoff_time:
                    print_to_log(f"Cleaning up old session: {session_id}")
                    self.delete_session(session_id)


# Global state manager instance
state_manager = StateManager()
