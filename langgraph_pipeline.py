#!/usr/bin/env python3
"""
LangGraph Multi-Agent ML Pipeline
Main orchestration system that coordinates preprocessing, feature selection, and model building agents
"""

from typing import Dict, Any, Optional, Callable, List
import tempfile
import os
from datetime import datetime

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
    
    # Try to import SqliteSaver separately (optional for persistence)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        SQLITE_AVAILABLE = True
    except ImportError:
        SqliteSaver = None
        sqlite3 = None
        SQLITE_AVAILABLE = False
        print("⚠️ SQLite checkpointer not available, persistence will be limited")
        
except ImportError as e:
    print(f"⚠️ LangGraph import failed: {e}")
    StateGraph = None
    END = "END"
    SqliteSaver = None
    sqlite3 = None
    LANGGRAPH_AVAILABLE = False
    SQLITE_AVAILABLE = False

from pipeline_state import PipelineState, state_manager
from orchestrator import orchestrator, AgentType
from agents_integrated import preprocessing_agent, feature_selection_agent, model_building_agent
from toolbox import initialize_toolbox, progress_tracker, slack_manager, user_directory_manager


class MultiAgentMLPipeline:
    """
    Main pipeline class that orchestrates the multi-agent ML system using LangGraph
    """
    
    def __init__(self, 
                 slack_token: str = None, 
                 artifacts_dir: str = None,
                 user_data_dir: str = None,
                 enable_persistence: bool = True):
        
        # Always initialize toolbox (needed for both LangGraph and simplified pipeline)
        initialize_toolbox(slack_token, artifacts_dir, user_data_dir)
        
        if not LANGGRAPH_AVAILABLE:
            print("⚠️ LangGraph not available, using simplified pipeline")
            self.app = None
            self.graph = None
            self.checkpointer = None
            self.enable_persistence = False
        else:
            
            # Set up persistence
            self.enable_persistence = enable_persistence and SQLITE_AVAILABLE
            self.checkpointer = None
            if self.enable_persistence:
                self._setup_persistence()
            elif enable_persistence and not SQLITE_AVAILABLE:
                print("⚠️ Persistence requested but SQLite not available, using memory-only mode")
            
            # Build the graph
            self.graph = self._build_graph()
            self.app = self.graph.compile(checkpointer=self.checkpointer)
        
        # User session management
        self.user_sessions = {}  # Store per-user-thread sessions
        
        print("🚀 Multi-Agent ML Pipeline initialized")
        print(f"   Persistence: {'✅ Enabled' if enable_persistence else '❌ Disabled'}")
        print(f"   Agents: Preprocessing, Feature Selection, Model Building")
        print(f"   Orchestrator: ✅ Ready")
    
    def _setup_persistence(self):
        """Set up SQLite-based persistence for LangGraph"""
        if not SQLITE_AVAILABLE:
            print("⚠️ SQLite not available, cannot set up persistence")
            self.enable_persistence = False
            self.checkpointer = None
            return
            
        try:
            # Create a persistent database file
            db_path = os.path.join(tempfile.gettempdir(), "mal_integration_checkpoints.db")
            
            # Initialize the database
            conn = sqlite3.connect(db_path)
            conn.close()
            
            # Create checkpointer
            self.checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
            print(f"📁 LangGraph persistence enabled: {db_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to set up LangGraph persistence: {e}")
            print("   Continuing without LangGraph persistence...")
            self.enable_persistence = False
            self.checkpointer = None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        graph = StateGraph(PipelineState)
        
        # Add nodes
        graph.add_node("orchestrator", self._orchestrator_node)
        graph.add_node("preprocessing", self._preprocessing_node)
        graph.add_node("feature_selection", self._feature_selection_node)
        graph.add_node("model_building", self._model_building_node)
        graph.add_node("code_execution", self._code_execution_node)
        
        # Add edges from orchestrator to agents
        graph.add_conditional_edges(
            "orchestrator",
            self._route_to_agent,
            {
                AgentType.PREPROCESSING.value: "preprocessing",
                AgentType.FEATURE_SELECTION.value: "feature_selection",
                AgentType.MODEL_BUILDING.value: "model_building",
                "code_execution": "code_execution",
                AgentType.END.value: END
            }
        )
        
        # Add edges between agents for pipeline flow
        graph.add_conditional_edges(
            "preprocessing",
            self._determine_next_step,
            {
                "feature_selection": "feature_selection",
                "model_building": "model_building",
                "orchestrator": "orchestrator",
                END: END
            }
        )
        
        graph.add_conditional_edges(
            "feature_selection",
            self._determine_next_step,
            {
                "model_building": "model_building",
                "preprocessing": "preprocessing",
                "orchestrator": "orchestrator",
                END: END
            }
        )
        
        graph.add_conditional_edges(
            "model_building",
            self._determine_next_step,
            {
                "preprocessing": "preprocessing",
                "feature_selection": "feature_selection",
                "orchestrator": "orchestrator",
                END: END
            }
        )
        
        # Code execution always ends after completion
        graph.add_edge("code_execution", END)
        
        # Set entry point
        graph.set_entry_point("orchestrator")
        
        return graph
    
    def _orchestrator_node(self, state: PipelineState) -> PipelineState:
        """Orchestrator node - routes queries to appropriate agents"""
        print(f"\n🎯 [Orchestrator] Processing query: '{state.user_query}'")
        
        # Route the query
        selected_agent = orchestrator.route(state)
        
        # Get routing explanation
        explanation = orchestrator.get_routing_explanation(state, selected_agent)
        
        # Update state
        state.current_agent = "Orchestrator"
        state.add_execution_record("Orchestrator", f"routed_to_{selected_agent}", explanation)
        
        # Store routing decision for conditional edges
        state.artifacts = state.artifacts or {}
        state.artifacts["routing_decision"] = selected_agent
        
        progress_tracker.update(state, f"Routed to {selected_agent}: {explanation}")
        
        return state
    
    def _preprocessing_node(self, state: PipelineState) -> PipelineState:
        """Preprocessing node"""
        print(f"\n🧹 [Preprocessing] Starting data preprocessing")
        return preprocessing_agent.run(state)
    
    def _feature_selection_node(self, state: PipelineState) -> PipelineState:
        """Feature selection node"""
        print(f"\n🎯 [Feature Selection] Starting feature selection")
        return feature_selection_agent.run(state)
    
    def _model_building_node(self, state: PipelineState) -> PipelineState:
        """Model building node"""
        print(f"\n🤖 [Model Building] Starting model building")
        return model_building_agent.run(state)
    
    def _code_execution_node(self, state: PipelineState) -> PipelineState:
        """Code execution node - handles general code execution requests"""
        print(f"\n💻 [Code Execution] Processing code execution request")
        
        # Use the ExecutionAgent from toolbox for code execution
        try:
            # Generate code using LLM (similar to ModelBuildingAgent approach)
            from toolbox import execution_agent
            
            # For now, we'll use a simple approach - the user query is treated as code
            # In a more sophisticated implementation, we'd use LLM to generate code
            user_code = state.user_query
            
            # If the query doesn't look like code, generate a response
            if not any(keyword in user_code.lower() for keyword in ['print', 'plot', 'calculate', 'import', '=', 'def']):
                # This looks like a request for code generation, not direct code
                state.last_response = f"I can help you execute code! Please provide Python code or be more specific about what you'd like me to calculate or analyze. For example:\n- 'print(sample_data.describe())'\n- 'plot correlation matrix'\n- 'calculate mean of numeric columns'"
                return state
            
            # Execute the code
            context = {
                "raw_data": state.raw_data,
                "cleaned_data": state.cleaned_data,
                "selected_features": state.selected_features,
                "trained_model": state.trained_model
            }
            
            result_state = execution_agent.run_code(state, user_code, context)
            
            # Update response
            if result_state.last_error:
                state.last_response = f"❌ Code execution failed: {result_state.last_error}"
            else:
                state.last_response = "✅ Code executed successfully! Check the results above."
            
            return result
    
    def _get_user_session_dir(self, session_id: str) -> str:
        """Get user session directory for conversation history"""
        return user_directory_manager.ensure_user_directory(session_id)
    
    def _save_conversation_history(self, session_id: str, user_query: str, response: str):
        """Save conversation history to user directory"""
        try:
            user_dir = self._get_user_session_dir(session_id)
            history_file = os.path.join(user_dir, "conversation_history.json")
            
            # Load existing history
            history = []
            if os.path.exists(history_file):
                import json
                with open(history_file, 'r') as f:
                    history = json.load(f)
            
            # Add new conversation
            conversation = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "response": response,
                "session_id": session_id
            }
            history.append(conversation)
            
            # Save updated history
            import json
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save conversation history: {e}")
    
    def _load_conversation_history(self, session_id: str) -> List[Dict]:
        """Load conversation history from user directory"""
        try:
            user_dir = self._get_user_session_dir(session_id)
            history_file = os.path.join(user_dir, "conversation_history.json")
            
            if os.path.exists(history_file):
                import json
                with open(history_file, 'r') as f:
                    return json.load(f)
            return []
            
        except Exception as e:
            print(f"⚠️ Failed to load conversation history: {e}")
            return []
    
    def _save_session_state(self, session_id: str, state: PipelineState):
        """Save session state to user directory"""
        try:
            user_dir = self._get_user_session_dir(session_id)
            state_file = os.path.join(user_dir, "session_state.json")
            
            # Convert state to dict for JSON serialization
            state_dict = state.dict()
            # Remove non-serializable data
            if 'raw_data' in state_dict:
                state_dict['raw_data'] = f"DataFrame({state.raw_data.shape})" if state.raw_data is not None else None
            if 'processed_data' in state_dict:
                state_dict['processed_data'] = f"DataFrame({state.processed_data.shape})" if state.processed_data is not None else None
            if 'cleaned_data' in state_dict:
                state_dict['cleaned_data'] = f"DataFrame({state.cleaned_data.shape})" if state.cleaned_data is not None else None
            if 'selected_features' in state_dict:
                state_dict['selected_features'] = f"DataFrame({state.selected_features.shape})" if state.selected_features is not None else None
            
            import json
            with open(state_file, 'w') as f:
                json.dump(state_dict, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save session state: {e}")
    
    def _load_session_state(self, session_id: str) -> Optional[Dict]:
        """Load session state from user directory"""
        try:
            user_dir = self._get_user_session_dir(session_id)
            state_file = os.path.join(user_dir, "session_state.json")
            
            if os.path.exists(state_file):
                import json
                with open(state_file, 'r') as f:
                    return json.load(f)
            return None
            
        except Exception as e:
            print(f"⚠️ Failed to load session state: {e}")
            return None
    
    def _route_to_agent(self, state: PipelineState) -> str:
        """Conditional edge function for routing from orchestrator"""
        routing_decision = state.artifacts.get("routing_decision", AgentType.END.value)
        print(f"🔀 Routing to: {routing_decision}")
        return routing_decision
    
    def _determine_next_step(self, state: PipelineState) -> str:
        """Determine next step after agent execution"""
        # Check if there was an error
        if state.last_error:
            print(f"❌ Error detected: {state.last_error}")
            return END
        
        # Check if this was a single-step request
        query = (state.user_query or "").lower()
        
        # Single-step requests should end after completion
        single_step_patterns = [
            "clean data", "preprocess", "select features", "train model",
            "build model", "analyze features", "show", "display"
        ]
        
        if any(pattern in query for pattern in single_step_patterns):
            print("✅ Single-step request completed")
            return END
        
        # For pipeline requests, continue to next logical step
        current_agent = state.current_agent
        
        if current_agent == "PreprocessingAgent":
            if state.cleaned_data is not None:
                # Check if user wants full pipeline
                if any(word in query for word in ["train", "model", "pipeline", "complete"]):
                    return "feature_selection"
                else:
                    return END
        
        elif current_agent == "FeatureSelectionAgent":
            if state.selected_features:
                # Check if user wants to build model
                if any(word in query for word in ["train", "model", "build", "pipeline"]):
                    return "model_building"
                else:
                    return END
        
        elif current_agent == "ModelBuildingAgent":
            # Model building is typically the end
            return END
        
        return END
    
    def _run_simplified_pipeline(self, state: PipelineState) -> PipelineState:
        """Run simplified pipeline without LangGraph"""
        print("🔄 Running simplified pipeline (LangGraph not available)")
        
        # Route the query
        selected_agent = orchestrator.route(state)
        
        # Execute the appropriate agent
        if selected_agent == AgentType.PREPROCESSING.value:
            state = preprocessing_agent.run(state)
        elif selected_agent == AgentType.FEATURE_SELECTION.value:
            state = feature_selection_agent.run(state)
        elif selected_agent == AgentType.MODEL_BUILDING.value:
            state = model_building_agent.run(state)
        
        return state
    
    def process_query(self, 
                     query: str, 
                     session_id: str = None,
                     raw_data: Optional[Any] = None,
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Main entry point for processing user queries
        """
        print(f"\n🚀 Processing query: '{query}'")
        
        # Generate session ID if not provided
        if not session_id:
            import time
            session_id = f"session_{int(time.time())}"
        
        # Ensure user directory exists
        user_dir = self._get_user_session_dir(session_id)
        print(f"📁 User session directory: {user_dir}")
        
        # Load conversation history
        conversation_history = self._load_conversation_history(session_id)
        if conversation_history:
            print(f"📚 Loaded {len(conversation_history)} previous conversations")
        
        # Load or create state
        state = state_manager.load_state(session_id)
        if state is None:
            state = PipelineState(
                session_id=session_id,
                chat_session=session_id,
                user_query=query
            )
            # Load previous session state if available
            previous_state = self._load_session_state(session_id)
            if previous_state:
                print(f"📂 Loaded previous session state for {session_id}")
                # Restore relevant state information (but not DataFrames)
                if 'preprocessing_state' in previous_state:
                    state.preprocessing_state = previous_state['preprocessing_state']
                if 'feature_selection_state' in previous_state:
                    state.feature_selection_state = previous_state['feature_selection_state']
                if 'model_building_state' in previous_state:
                    state.model_building_state = previous_state['model_building_state']
        else:
            state.user_query = query
        
        # Add raw data if provided
        if raw_data is not None:
            state.raw_data = raw_data
        
        # Set up progress callback
        if progress_callback:
            original_update = progress_tracker.update
            def enhanced_update(state, message, stage=None, send_to_slack=True):
                original_update(state, message, stage, send_to_slack)
                progress_callback(message, stage or "")
            progress_tracker.update = enhanced_update
        
        try:
            if LANGGRAPH_AVAILABLE and self.app:
                # Configure for persistence
                config = {"configurable": {"thread_id": session_id}} if self.enable_persistence else None
                
                # Run the pipeline
                result_state = self.app.invoke(state, config=config)
            else:
                # Simplified pipeline without LangGraph
                result_state = self._run_simplified_pipeline(state)
            
            # Save state
            state_manager.save_state(result_state)
            
            # Prepare response
            response = self._prepare_response(result_state)
            
            print(f"✅ Query processing completed for session {session_id}")
            return response
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Update state with error
            state.last_error = error_msg
            state_manager.save_state(state)
            
            return {
                "success": False,
                "error": error_msg,
                "session_id": session_id,
                "response": f"❌ Sorry, I encountered an error: {error_msg}"
            }
    
    def _prepare_response(self, state: PipelineState) -> Dict[str, Any]:
        """Prepare response from final state"""
        response = {
            "success": state.last_error is None,
            "session_id": state.session_id,
            "chat_session": state.chat_session,
            "response": self._generate_response_text(state),
            "data_summary": state.get_data_summary(),
            "artifacts": state.artifacts or {},
            "execution_history": state.execution_history[-5:] if state.execution_history else []  # Last 5 records
        }
        
        if state.last_error:
            response["error"] = state.last_error
        
        # Save conversation history and session state
        self._save_conversation_history(state.session_id, state.user_query, response["response"])
        self._save_session_state(state.session_id, state)
        
        return response
    
    def _generate_response_text(self, state: PipelineState) -> str:
        """Generate human-readable response text"""
        if state.last_error:
            return f"❌ Operation failed: {state.last_error}"
        
        # Generate response based on what was accomplished
        accomplishments = []
        
        if state.cleaned_data is not None and state.preprocessing_state.get("completed"):
            shape = state.cleaned_data.shape
            accomplishments.append(f"✅ Data preprocessing completed ({shape[0]:,} rows × {shape[1]} columns)")
        
        if state.selected_features and state.feature_selection_state.get("completed"):
            count = len(state.selected_features)
            accomplishments.append(f"✅ Feature selection completed ({count} features selected)")
        
        if state.trained_model is not None and state.model_building_state.get("completed"):
            accomplishments.append("✅ Model training completed")
        
        if accomplishments:
            return "\n".join(accomplishments)
        else:
            return "✅ Operation completed successfully"
    
    def load_data(self, data: Any, session_id: str):
        """Load data into a session"""
        state = state_manager.load_state(session_id)
        if state is None:
            state = PipelineState(session_id=session_id, chat_session=session_id)
        
        state.raw_data = data
        state_manager.save_state(state)
        
        print(f"📊 Data loaded for session {session_id}: {data.shape if hasattr(data, 'shape') else 'Unknown shape'}")
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a session"""
        state = state_manager.load_state(session_id)
        if state is None:
            return {"exists": False}
        
        return {
            "exists": True,
            "session_id": session_id,
            "created_at": state.created_at.isoformat() if state.created_at else None,
            "updated_at": state.updated_at.isoformat() if state.updated_at else None,
            "data_summary": state.get_data_summary(),
            "last_query": state.user_query,
            "last_error": state.last_error,
            "progress": state.progress
        }
    
    def list_sessions(self) -> List[str]:
        """List all available sessions"""
        return state_manager.list_sessions()
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions"""
        state_manager.cleanup_old_sessions(max_age_hours)


# Global pipeline instance
pipeline = None


def get_pipeline(slack_token: str = None, artifacts_dir: str = None, user_data_dir: str = None, enable_persistence: bool = True) -> MultiAgentMLPipeline:
    """Get or create global pipeline instance"""
    global pipeline
    
    if pipeline is None:
        pipeline = MultiAgentMLPipeline(
            slack_token=slack_token,
            artifacts_dir=artifacts_dir,
            user_data_dir=user_data_dir,
            enable_persistence=enable_persistence
        )
    
    return pipeline


def initialize_pipeline(slack_token: str = None, artifacts_dir: str = None, user_data_dir: str = None, enable_persistence: bool = True):
    """Initialize the global pipeline"""
    global pipeline
    pipeline = MultiAgentMLPipeline(
        slack_token=slack_token,
        artifacts_dir=artifacts_dir,
        user_data_dir=user_data_dir,
        enable_persistence=enable_persistence
    )
    return pipeline


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create sample data
    sample_data = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'credit_score': np.random.randint(300, 850, 1000),
        'loan_amount': np.random.normal(25000, 10000, 1000),
        'employment_years': np.random.randint(0, 40, 1000),
        'target': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })
    
    # Initialize pipeline
    ml_pipeline = initialize_pipeline(enable_persistence=True)
    
    # Test queries
    test_queries = [
        "Clean and preprocess this dataset",
        "Select the best features for modeling",
        "Train a machine learning model",
        "Build a complete ML pipeline from this data"
    ]
    
    session_id = "test_session"
    ml_pipeline.load_data(sample_data, session_id)
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: {query}")
        print('='*60)
        
        result = ml_pipeline.process_query(query, session_id)
        print(f"\nResult: {result['response']}")
        
        if not result['success']:
            print(f"Error: {result.get('error')}")
            break
