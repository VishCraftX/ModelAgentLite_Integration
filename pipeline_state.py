#!/usr/bin/env python3
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
    
    # Execution context
    current_agent: Optional[str] = None
    execution_history: List[Dict] = Field(default_factory=list)
    
    # Session metadata
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
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
        
        print(f"✅ Saved {phase} strategies to session state")
    
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
        
        for col, rec in recommendations.items():
            if isinstance(rec, dict):
                transformation = rec.get('transformation', 'standardize')
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
                    elif transformation in ['log1p', 'sqrt']:
                        strategy_data["parameters"]["offset"] = 1
                
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
        
        summary = ["📋 **Saved Preprocessing Strategies:**\n"]
        
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
        
        summary.append(f"\n**Strategy Counts:**")
        summary.append(f"• Outlier treatments: {outlier_count} columns")
        summary.append(f"• Missing value strategies: {missing_count} columns") 
        summary.append(f"• Encoding strategies: {encoding_count} columns")
        summary.append(f"• Transformations: {transform_count} columns")
        
        return "\n".join(summary)
    
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
        
        print(f"✅ Saved {phase} strategies to session state")
    
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
        
        for col, rec in recommendations.items():
            if isinstance(rec, dict):
                transformation = rec.get('transformation', 'standardize')
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
                    elif transformation in ['log1p', 'sqrt']:
                        strategy_data["parameters"]["offset"] = 1
                
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
        
        summary = ["📋 **Saved Preprocessing Strategies:**\n"]
        
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
        
        summary.append(f"\n**Strategy Counts:**")
        summary.append(f"• Outlier treatments: {outlier_count} columns")
        summary.append(f"• Missing value strategies: {missing_count} columns") 
        summary.append(f"• Encoding strategies: {encoding_count} columns")
        summary.append(f"• Transformations: {transform_count} columns")
        
        return "\n".join(summary)


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
            print(f"⚠️ Permission denied for {self.base_dir}, using fallback: {fallback_dir}")
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
            print(f"❌ Permission error creating session directory: {e}")
            # Try creating in a more accessible location
            fallback_session_dir = os.path.expanduser(f"~/mal_integration_sessions/{state.session_id}")
            print(f"🔄 Using fallback directory: {fallback_session_dir}")
            os.makedirs(fallback_session_dir, exist_ok=True)
            session_dir = fallback_session_dir
        
        # Save state metadata (without DataFrames)
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
        
        # Handle trained model
        if state.trained_model is not None:
            model_path = os.path.join(session_dir, "trained_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(state.trained_model, f)
            state_dict["trained_model"] = model_path
        
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
            print(f"Error loading state for session {session_id}: {e}")
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
                print(f"Error deleting session {session_id}: {e}")
                return False
        return False
    
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
                    print(f"Cleaning up old session: {session_id}")
                    self.delete_session(session_id)


# Global state manager instance
state_manager = StateManager()
