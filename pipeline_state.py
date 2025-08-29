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


class PipelineState(BaseModel):
    """
    Global state schema for the multi-agent ML pipeline.
    This state is shared across all agents and persists throughout the session.
    """
    # Core data fields
    raw_data: Optional[pd.DataFrame] = None
    cleaned_data: Optional[pd.DataFrame] = None
    selected_features: Optional[List[str]] = None
    trained_model: Optional[Any] = None
    
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
    feature_selection_state: Optional[Dict] = Field(default_factory=dict)
    model_building_state: Optional[Dict] = Field(default_factory=dict)
    
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
        
        if self.raw_data is not None:
            summary["raw_data_shape"] = self.raw_data.shape
            summary["raw_data_columns"] = list(self.raw_data.columns)
        
        if self.cleaned_data is not None:
            summary["cleaned_data_shape"] = self.cleaned_data.shape
            summary["cleaned_data_columns"] = list(self.cleaned_data.columns)
        
        if self.selected_features:
            summary["selected_features_count"] = len(self.selected_features)
            summary["selected_features"] = self.selected_features
        
        return summary


class StateManager:
    """
    Manages persistence and retrieval of pipeline states
    """
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(tempfile.gettempdir(), "mal_integration_states")
        
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    def save_state(self, state: PipelineState) -> str:
        """Save pipeline state to disk"""
        if not state.session_id:
            state.session_id = f"session_{int(datetime.now().timestamp())}"
        
        session_dir = os.path.join(self.base_dir, state.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
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
        import time
        
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
