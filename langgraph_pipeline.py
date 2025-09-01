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
    print("✅ LangGraph core components imported successfully")
    
except ImportError as e:
    print(f"❌ LangGraph core import failed: {e}")
    StateGraph = None
    END = "END"
    LANGGRAPH_AVAILABLE = False

# SQLite checkpointer not needed - we use our own persistence system

from pipeline_state import PipelineState, state_manager
from orchestrator import orchestrator, AgentType
from agents_wrapper import preprocessing_agent, feature_selection_agent, model_building_agent
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
            print("❌ LangGraph not available, using simplified pipeline")
            self.app = None
            self.graph = None
            self.checkpointer = None
            self.enable_persistence = False
        else:
            print("✅ LangGraph available, building full pipeline")
            
            # LangGraph checkpointing is optional - we use our own persistence
            self.enable_persistence = False  # Disable LangGraph checkpointing
            self.checkpointer = None
            print("📁 Using user directory persistence (LangGraph checkpointing disabled)")
            
            # Build the graph
            self.graph = self._build_graph()
            self.app = self.graph.compile(checkpointer=self.checkpointer)
        
        # User session management
        self.user_sessions = {}  # Store per-user-thread sessions
        
        # Store references to global toolbox components
        self.slack_manager = slack_manager
        self.progress_tracker = progress_tracker
        self.user_directory_manager = user_directory_manager
        
        print("🚀 Multi-Agent ML Pipeline initialized")
        print(f"   LangGraph: {'✅ Full Pipeline' if LANGGRAPH_AVAILABLE else '⚠️ Simplified Pipeline'}")
        print(f"   Persistence: ✅ User Directory + Session State")
        print(f"   Agents: Preprocessing, Feature Selection, Model Building")
        print(f"   Orchestrator: ✅ Ready")
    

    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        graph = StateGraph(PipelineState)
        
        # Add nodes
        graph.add_node("orchestrator", self._orchestrator_node)
        graph.add_node("preprocessing", self._preprocessing_node)
        graph.add_node("feature_selection", self._feature_selection_node)
        graph.add_node("model_building", self._model_building_node)
        graph.add_node("general_response", self._general_response_node)
        graph.add_node("code_execution", self._code_execution_node)
        
        # Add edges from orchestrator to agents
        graph.add_conditional_edges(
            "orchestrator",
            self._route_to_agent,
            {
                                AgentType.PREPROCESSING.value: "preprocessing",
                AgentType.FEATURE_SELECTION.value: "feature_selection", 
                AgentType.MODEL_BUILDING.value: "model_building",
                "general_response": "general_response",
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
        
        # Code execution and general response always end after completion
        graph.add_edge("code_execution", END)
        graph.add_edge("general_response", END)
        
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
        
        # Send a friendly next-steps hint via Slack
        try:
            slack_manager = self.slack_manager
            if slack_manager and state.chat_session:
                next_hint = {
                    'preprocessing': "Type `proceed` to begin preprocessing, or `help` for options.",
                    'feature_selection': "Type `proceed` to begin feature selection, or `help` for options.",
                    'model_building': "Type `proceed` to begin model building, or `help` for options.",
                    'general_response': "You can ask a question or say `help` for options.",
                    'code_execution': "Type your code question or `help` for options.",
                }.get(selected_agent, "Type `help` for options.")
                slack_manager.send_message(state.chat_session, f"{explanation}\n\n{next_hint}")
        except Exception as e:
            print(f"⚠️ Failed to send orchestrator next-steps hint: {e}")
        
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
    
    def _general_response_node(self, state: PipelineState) -> PipelineState:
        """General response node - handles conversational queries using LLM"""
        print(f"\n💬 [General Response] Generating conversational response")
        
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
            
            # Generate context-aware response
            query = state.user_query or ""
            
            if state.raw_data is not None:
                context_prompt = f"The user said: '{query}'. I have their dataset with {state.raw_data.shape[0]:,} rows and {state.raw_data.shape[1]} columns. Respond naturally and conversationally. Only mention specific capabilities if they ask 'what can you do' or similar questions."
            else:
                context_prompt = f"The user said: '{query}'. Respond naturally and conversationally as an AI assistant. Don't list capabilities unless they specifically ask what you can do."
            
            print(f"🔍 Generating conversational response for: '{query}'")
            
            # Use LLM for conversational response
            response = ollama.chat(
                model=os.getenv("DEFAULT_MODEL", "gpt-4o"),  # Use environment variable
                messages=[
                    {"role": "system", "content": "You are a specialized AI assistant for data science and machine learning. You help users build models, analyze data, and work with datasets. When greeting users, be friendly and natural. When asked about capabilities, mention your ML/data science skills like building models, data analysis, visualization, etc. Keep responses conversational and concise."},
                    {"role": "user", "content": context_prompt}
                ]
            )
            
            generated_response = response["message"]["content"].strip()
            state.last_response = generated_response
            
            print(f"✅ Generated response: {generated_response[:100]}...")
            
        except Exception as e:
            print(f"❌ Error generating conversational response: {e}")
            # Fallback response
            state.last_response = "Hello! I'm your ML assistant. I can help you with data preprocessing, feature selection, and model building. How can I assist you today?"
        
        return state
    
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
            
            return state
            
        except Exception as e:
            print(f"❌ Code execution error: {e}")
            state.last_error = str(e)
            state.last_response = f"❌ Code execution failed: {str(e)}"
            return state
    
    def _get_user_session_dir(self, session_id: str) -> str:
        """Get user session directory for conversation history"""
        return user_directory_manager.ensure_user_directory(session_id)
    
    def _save_conversation_history(self, session_id: str, user_query: str, response: str):
        """Save conversation history to user directory"""
        try:
            user_dir = self._get_user_session_dir(session_id)
            history_file = os.path.join(user_dir, "conversation_history.json")
            
            print(f"💾 Saving conversation history to: {history_file}")
            
            # Load existing history with robust error handling
            history = []
            if os.path.exists(history_file):
                try:
                    import json
                    with open(history_file, 'r') as f:
                        loaded_data = json.load(f)
                        if isinstance(loaded_data, list):
                            history = loaded_data
                        else:
                            print(f"⚠️ Invalid conversation history format (not a list), starting fresh")
                            history = []
                    print(f"📚 Loaded {len(history)} existing conversations")
                except (json.JSONDecodeError, Exception) as e:
                    print(f"⚠️ Corrupted conversation history file, starting fresh: {e}")
                    history = []
            
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
            
            print(f"✅ Conversation history saved ({len(history)} total conversations)")
                
        except Exception as e:
            print(f"⚠️ Failed to save conversation history: {e}")
            import traceback
            traceback.print_exc()
    
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
        """Save session state to user directory including DataFrames"""
        try:
            user_dir = self._get_user_session_dir(session_id)
            state_file = os.path.join(user_dir, "session_state.json")
            
            # Convert state to dict for JSON serialization
            state_dict = state.dict()

            # Recursively convert numpy types to native Python
            import numpy as np
            def to_native(obj):
                if isinstance(obj, dict):
                    return {k: to_native(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [to_native(v) for v in obj]
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                return obj
            state_dict = to_native(state_dict)
            
            # Save DataFrames as CSV files and store references
            if 'raw_data' in state_dict and state.raw_data is not None:
                raw_data_file = os.path.join(user_dir, "raw_data.csv")
                state.raw_data.to_csv(raw_data_file, index=False)
                state_dict['raw_data'] = {"type": "dataframe", "file": "raw_data.csv", "shape": list(state.raw_data.shape)}
            else:
                state_dict['raw_data'] = None
                
            if 'processed_data' in state_dict and state.processed_data is not None:
                processed_data_file = os.path.join(user_dir, "processed_data.csv")
                state.processed_data.to_csv(processed_data_file, index=False)
                state_dict['processed_data'] = {"type": "dataframe", "file": "processed_data.csv", "shape": list(state.processed_data.shape)}
            else:
                state_dict['processed_data'] = None
                
            if 'cleaned_data' in state_dict and state.cleaned_data is not None:
                cleaned_data_file = os.path.join(user_dir, "cleaned_data.csv")
                state.cleaned_data.to_csv(cleaned_data_file, index=False)
                state_dict['cleaned_data'] = {"type": "dataframe", "file": "cleaned_data.csv", "shape": list(state.cleaned_data.shape)}
                print(f"💾 Saved cleaned_data to session: {state.cleaned_data.shape}")
                print(f"📁 Data saved to: {cleaned_data_file}")
                print(f"🔧 DEBUG: Data columns: {list(state.cleaned_data.columns)}")
            else:
                state_dict['cleaned_data'] = None
            
            # Handle interactive_session which may contain DataFrames
            if 'interactive_session' in state_dict and state_dict['interactive_session'] is not None:
                # Save interactive session state separately if it contains DataFrames
                interactive_session = state_dict['interactive_session'].copy()
                if 'current_state' in interactive_session:
                    # Remove or serialize any DataFrames in the interactive session
                    current_state = interactive_session['current_state']
                    if hasattr(current_state, 'df') and current_state.df is not None:
                        interactive_df_file = os.path.join(user_dir, "interactive_df.csv")
                        current_state.df.to_csv(interactive_df_file, index=False)
                        interactive_session['current_state'] = "serialized_to_interactive_df.csv"
                    elif hasattr(current_state, '__dict__'):
                        # Convert to dict and remove DataFrames
                        state_as_dict = {}
                        for key, value in current_state.__dict__.items():
                            if hasattr(value, 'to_csv'):  # It's a DataFrame
                                state_as_dict[key] = f"DataFrame({value.shape})"
                            else:
                                state_as_dict[key] = value
                        interactive_session['current_state'] = state_as_dict
                state_dict['interactive_session'] = interactive_session
            
            import json
            from datetime import datetime
            
            # Custom JSON encoder for datetime objects
            class DateTimeEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    return super().default(obj)
            
            with open(state_file, 'w') as f:
                json.dump(state_dict, f, indent=2, cls=DateTimeEncoder)
                
        except Exception as e:
            print(f"⚠️ Failed to save session state: {e}")
    
    def _load_session_state(self, session_id: str) -> Optional[Dict]:
        """Load session state from user directory including DataFrames"""
        try:
            user_dir = self._get_user_session_dir(session_id)
            state_file = os.path.join(user_dir, "session_state.json")
            
            if os.path.exists(state_file):
                import json
                import pandas as pd
                
                with open(state_file, 'r') as f:
                    state_dict = json.load(f)
                
                # Restore DataFrames from CSV files
                if state_dict.get('raw_data') and isinstance(state_dict['raw_data'], dict):
                    raw_data_file = os.path.join(user_dir, state_dict['raw_data']['file'])
                    if os.path.exists(raw_data_file):
                        state_dict['raw_data'] = pd.read_csv(raw_data_file)
                        print(f"📂 Restored raw_data: {state_dict['raw_data'].shape}")
                
                if state_dict.get('processed_data') and isinstance(state_dict['processed_data'], dict):
                    processed_data_file = os.path.join(user_dir, state_dict['processed_data']['file'])
                    if os.path.exists(processed_data_file):
                        state_dict['processed_data'] = pd.read_csv(processed_data_file)
                        print(f"📂 Restored processed_data: {state_dict['processed_data'].shape}")
                
                if state_dict.get('cleaned_data') and isinstance(state_dict['cleaned_data'], dict):
                    cleaned_data_file = os.path.join(user_dir, state_dict['cleaned_data']['file'])
                    if os.path.exists(cleaned_data_file):
                        state_dict['cleaned_data'] = pd.read_csv(cleaned_data_file)
                        print(f"📂 Restored cleaned_data: {state_dict['cleaned_data'].shape}")
                
                return state_dict
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
        # NOTE: "build model" is NOT a single-step - it requires preprocessing + feature selection + model building
        single_step_patterns = [
            "clean data", "preprocess data", "select features only", "analyze features",
            "show", "display", "export", "save"
        ]
        
        # Check for truly single-step patterns (not model building which is multi-step)
        is_single_step = False
        for pattern in single_step_patterns:
            if pattern in query and not any(multi in query for multi in ["model", "train", "pipeline", "complete"]):
                is_single_step = True
                break
        
        if is_single_step:
            print("✅ Single-step request completed")
            return END
        
        # For pipeline requests, continue to next logical step
        current_agent = state.current_agent
        
        if current_agent == "PreprocessingAgent":
            if state.cleaned_data is not None:
                # For model building requests, always continue to feature selection
                if any(word in query for word in ["train", "model", "build", "pipeline", "complete", "lgbm", "classifier", "regressor"]):
                    print("🔄 Continuing to feature selection for model building request")
                    return "feature_selection"
                else:
                    return END
        
        elif current_agent == "FeatureSelectionAgent":
            if state.selected_features is not None:
                # For model building requests, always continue to model building
                if any(word in query for word in ["train", "model", "build", "pipeline", "complete", "lgbm", "classifier", "regressor"]):
                    print("🔄 Continuing to model building")
                    return "model_building"
                else:
                    return END
        
        elif current_agent == "ModelBuildingAgent":
            # Model building is typically the end
            print("✅ Model building completed - pipeline finished")
            return END
        
        return END
    
    def _run_simplified_pipeline(self, state: PipelineState) -> PipelineState:
        """Run simplified pipeline without LangGraph"""
        print("🔄 Running simplified pipeline (LangGraph not available)")
        
        # Store the original user intent
        original_query = state.user_query
        original_intent = orchestrator._classify_with_keyword_scoring(original_query)[0]
        print(f"[SimplifiedPipeline] Original intent: {original_intent}")
        
        # Execute pipeline steps sequentially based on original intent
        if original_intent in ["model_building", "full_pipeline"]:
            # For model building, run full pipeline if needed
            state = self._run_sequential_pipeline_steps(state, target_intent="model_building")
        elif original_intent == "feature_selection":
            # For feature selection, run preprocessing + feature selection if needed
            state = self._run_sequential_pipeline_steps(state, target_intent="feature_selection")
        else:
            # For single-step intents, route normally
            selected_agent = orchestrator.route(state)
            state = self._execute_single_agent(state, selected_agent)
        
        return state
    
    def _run_sequential_pipeline_steps(self, state: PipelineState, target_intent: str) -> PipelineState:
        """Run pipeline steps sequentially until target is reached"""
        max_steps = 5  # Prevent infinite loops
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            
            # Route based on current state
            selected_agent = orchestrator.route(state)
            print(f"[SimplifiedPipeline] Step {step_count}: Routing to {selected_agent}")
            
            # If we've reached the target or end, stop
            if selected_agent == target_intent or selected_agent == AgentType.END.value:
                if selected_agent != AgentType.END.value:
                    state = self._execute_single_agent(state, selected_agent)
                break
            
            # Execute the current step
            state = self._execute_single_agent(state, selected_agent)
            
            # Check if we should continue to next step
            if selected_agent == AgentType.PREPROCESSING.value and target_intent in ["feature_selection", "model_building"]:
                continue  # Continue to next step
            elif selected_agent == AgentType.FEATURE_SELECTION.value and target_intent == "model_building":
                continue  # Continue to model building
            else:
                break  # Stop here
        
        return state
    
    def _execute_single_agent(self, state: PipelineState, agent_type: str) -> PipelineState:
        """Execute a single agent"""
        if agent_type == AgentType.PREPROCESSING.value:
            return preprocessing_agent.run(state)
        elif agent_type == AgentType.FEATURE_SELECTION.value:
            return feature_selection_agent.run(state)
        elif agent_type == AgentType.MODEL_BUILDING.value:
            return model_building_agent.run(state)
        elif agent_type == "general_response":
            # Handle general response in simplified pipeline
            return self._general_response_node(state)
        else:
            print(f"[SimplifiedPipeline] Unknown agent type: {agent_type}")
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
        
        # Always load session state if available (for interactive sessions)
        previous_state = self._load_session_state(session_id)
        if previous_state:
            print(f"📂 Loaded previous session state for {session_id}")
            print(f"🔧 DEBUG: Previous state keys: {list(previous_state.keys())}")
            # Restore relevant state information INCLUDING DataFrames
            if 'preprocessing_state' in previous_state:
                state.preprocessing_state = previous_state['preprocessing_state']
                print(f"🔧 DEBUG: Restored preprocessing_state: {state.preprocessing_state}")
                if state.preprocessing_state:
                    print(f"🔧 DEBUG: Current phase after restore: {state.preprocessing_state.get('current_phase')}")
                    print(f"🔧 DEBUG: Missing results after restore: {state.preprocessing_state.get('missing_results') is not None}")
            if 'feature_selection_state' in previous_state:
                state.feature_selection_state = previous_state['feature_selection_state']
            if 'model_building_state' in previous_state:
                state.model_building_state = previous_state['model_building_state']
            # Restore interactive session if available
            if 'interactive_session' in previous_state:
                state.interactive_session = previous_state['interactive_session']
                print(f"🔧 DEBUG: Restored interactive_session: {state.interactive_session}")
            # Restore DataFrames
            if 'raw_data' in previous_state and previous_state['raw_data'] is not None:
                state.raw_data = previous_state['raw_data']
            if 'processed_data' in previous_state and previous_state['processed_data'] is not None:
                state.processed_data = previous_state['processed_data']
            if 'cleaned_data' in previous_state and previous_state['cleaned_data'] is not None:
                state.cleaned_data = previous_state['cleaned_data']
            # Restore other important fields
            if 'selected_features' in previous_state and previous_state['selected_features'] is not None:
                state.selected_features = previous_state['selected_features']
            if 'models' in previous_state and previous_state['models'] is not None:
                state.models = previous_state['models']
            if 'best_model' in previous_state and previous_state['best_model'] is not None:
                state.best_model = previous_state['best_model']
            if 'target_column' in previous_state and previous_state['target_column'] is not None:
                state.target_column = previous_state['target_column']
        else:
            print(f"🔧 DEBUG: No previous state found for {session_id}")
        
        # Update user query
        state.user_query = query
        
        # Check if we have an active interactive session that needs to continue
        # BUT only if the query is a continuation command, not a new request
        print(f"🔍 DEBUG: Checking interactive session:")
        print(f"  Has interactive_session attr: {hasattr(state, 'interactive_session')}")
        print(f"  Interactive session is None: {state.interactive_session is None}")
        print(f"  Session active: {state.interactive_session.get('session_active', False) if state.interactive_session else 'N/A'}")
        print(f"  Interactive session details: {state.interactive_session}")
        
        if (hasattr(state, 'interactive_session') and 
            state.interactive_session is not None and 
            state.interactive_session.get('session_active', False)):
            
            print(f"🔄 Interactive session active: {state.interactive_session['agent_type']}")
            
            # Check if this is a continuation command vs a new request
            query_lower = query.lower().strip()
            print(f"🔍 DEBUG: Query to check: '{query_lower}'")
            
            # Pure continuation commands (context-independent)
            pure_continuation_commands = ['proceed', 'continue', 'next', 'back', 'summary', 'explain', 'help']
            
            # Check for explicit session management commands
            clear_session_commands = ['clear session', 'reset', 'start over', 'new session', 'exit session']
            is_clear_command = any(cmd in query_lower for cmd in clear_session_commands)
            
            if is_clear_command:
                print(f"🔄 Explicit session clear requested")
                state.interactive_session = None
                slack_manager = self.slack_manager
                slack_manager.send_message(state.chat_session, "✅ Session cleared. You can now start a new workflow.")
                return self._prepare_response(state, "Session cleared successfully.")
            
            # Also check for target column specification patterns
            target_column_patterns = ['target ', 'column ']
            is_target_specification = any(pattern in query_lower for pattern in target_column_patterns)
            
            # Semantic new request detection
            def is_semantic_new_request(query_lower: str, current_session: Dict) -> bool:
                """
                Use semantic similarity to detect if query is a new ML request
                that should bypass current interactive session
                """
                if not current_session:
                    return False
                
                try:
                    # Define what constitutes a "new request" vs "session continuation"
                    new_request_definitions = {
                        "new_ml_request": "Start new machine learning task, begin different ML workflow, switch to new agent, train new model, build new classifier, create new predictor, analyze different dataset, perform new analysis, start fresh preprocessing, begin new feature selection, initiate model building, commence new ML pipeline",
                        "session_continuation": "Continue current workflow, proceed with current task, advance current session, move to next step in current process, skip current phase, bypass current step, proceed in current agent, continue current analysis"
                    }
                    
                    from orchestrator import Orchestrator
                    temp_orchestrator = Orchestrator()
                    temp_orchestrator.intent_definitions = new_request_definitions
                    temp_orchestrator._initialize_intent_embeddings()
                    
                    if temp_orchestrator._intent_embeddings:
                        intent, confidence_info = temp_orchestrator._classify_with_semantic_similarity(query_lower)
                        print(f"[Session] New request semantic analysis: {intent} (confidence: {confidence_info['max_score']:.3f})")
                        
                        # Use moderate threshold for new request detection
                        if confidence_info.get("max_score", 0) > 0.25:
                            is_new = (intent == "new_ml_request")
                            print(f"[Session] Semantic new request decision: {'NEW REQUEST' if is_new else 'CONTINUATION'}")
                            return is_new
                    
                    print(f"[Session] New request semantic analysis failed, using keyword fallback")
                    return False
                    
                except Exception as e:
                    print(f"[Session] Semantic new request error: {e}, using keyword fallback")
                    return False
            
            # Check for new ML requests that should bypass interactive session (keyword fallback)
            new_request_patterns = [
                'clean my data', 'select features', 'train a', 'build a', 'create a', 'analyze my',
                'skip preprocessing', 'skip feature selection', 'without preprocessing', 'bypass cleaning'
            ]
            
            # Semantic continuation classification system
            def is_context_continuation(query_lower: str, session: Dict) -> bool:
                """
                Semantic-aware continuation detection with LLM fallback
                Uses the same sophisticated classification as main orchestrator
                """
                if not session:
                    return False
                
                agent_type = session.get('agent_type')
                current_phase = session.get('phase')
                
                # Phase 1: Pure continuation commands (always valid)
                pure_continuation_commands = ['proceed', 'continue', 'next', 'back', 'summary', 'explain', 'help']
                if any(cmd in query_lower for cmd in pure_continuation_commands):
                    return True
                
                # Phase 2: Yes/No for confirmation contexts (explicit)
                if current_phase in ['confirmation', 'approval_needed']:
                    return query_lower.strip() in ['yes', 'no', 'y', 'n']
                
                # Phase 3: Semantic continuation classification
                try:
                    continuation_definitions = {
                        'preprocessing': {
                            'continue_preprocessing': "Skip current step, proceed to next phase, continue preprocessing workflow, move forward in data cleaning, advance to next preprocessing stage, bypass current analysis, skip outliers detection, skip missing values handling, skip encoding step, skip transformations, target column specification, column selection, data cleaning continuation",
                            'new_request': "Start feature selection, begin model building, train new model, create model, build classifier, analyze features, select variables, stop preprocessing, end cleaning, switch to modeling, move to next agent"
                        },
                        'feature_selection': {
                            'continue_feature_selection': "Skip current analysis, run information value, execute SHAP analysis, continue feature selection, proceed with variable selection, advance feature engineering, move to next selection step, bypass current feature analysis, select important features, rank features, analyze correlations",
                            'new_request': "Start model building, train model, build classifier, create predictor, stop feature selection, end variable selection, switch to modeling, begin training, start preprocessing"
                        }
                    }
                    
                    if agent_type in continuation_definitions:
                        # Use semantic similarity to classify continuation vs new request
                        from orchestrator import Orchestrator
                        temp_orchestrator = Orchestrator()
                        
                        # Create temporary intent definitions for this context
                        context_intents = continuation_definitions[agent_type]
                        temp_orchestrator.intent_definitions = context_intents
                        temp_orchestrator._initialize_intent_embeddings()
                        
                        if temp_orchestrator._intent_embeddings:
                            intent, confidence_info = temp_orchestrator._classify_with_semantic_similarity(query_lower)
                            print(f"[Session] Semantic continuation analysis: {intent} (confidence: {confidence_info['max_score']:.3f})")
                            
                            # Use lower threshold for continuation detection (more permissive)
                            if confidence_info.get("max_score", 0) > 0.2:
                                is_continuation = (intent.startswith('continue_') or intent.endswith('_continuation'))
                                print(f"[Session] Semantic decision: {'CONTINUATION' if is_continuation else 'NEW REQUEST'}")
                                return is_continuation
                        
                        print(f"[Session] Semantic analysis failed, falling back to keyword matching")
                
                except Exception as e:
                    print(f"[Session] Semantic continuation error: {e}, using keyword fallback")
                
                # Phase 4: Keyword fallback (original logic)
                if agent_type == 'preprocessing':
                    preprocessing_continuations = [
                        'skip outliers', 'skip missing', 'skip encoding', 'skip transformations',
                        'skip this phase', 'skip current', 'target ', 'column ',
                        'go ahead', 'go ahead with', 'proceed with', 'continue with',
                        'start outliers', 'start missing', 'start encoding', 'start transformations',
                        'begin outliers', 'begin missing', 'begin encoding', 'begin transformations',
                        'run outliers', 'run missing', 'run encoding', 'run transformations',
                        'do outliers', 'do missing', 'do encoding', 'do transformations',
                        'handle outliers', 'handle missing', 'handle encoding', 'handle transformations'
                    ]
                    return any(cmd in query_lower for cmd in preprocessing_continuations)
                
                elif agent_type == 'feature_selection':
                    fs_continuations = [
                        'skip analysis', 'run iv', 'run shap', 'select features'
                    ]
                    return any(cmd in query_lower for cmd in fs_continuations)
                
                return False
            
            # Phase 1: Semantic analysis for session management
            semantic_new_request = is_semantic_new_request(query_lower, state.interactive_session)
            semantic_continuation = is_context_continuation(query_lower, state.interactive_session)
            
            # Phase 2: Keyword fallback analysis
            keyword_new_request = any(pattern in query_lower for pattern in new_request_patterns)
            
            # Phase 3: Combine semantic and keyword results (semantic takes priority)
            is_new_request = semantic_new_request or (not semantic_continuation and keyword_new_request)
            is_continuation = semantic_continuation
            
            print(f"[Session] Analysis results:")
            print(f"  Semantic new request: {semantic_new_request}")
            print(f"  Semantic continuation: {semantic_continuation}")
            print(f"  Keyword new request: {keyword_new_request}")
            print(f"  Final decision - New request: {is_new_request}, Continuation: {is_continuation}")
            
            # Handle conflicts: if both detected, prioritize based on context
            if is_new_request and is_continuation:
                # If query starts with new request pattern, treat as new request
                query_start = query_lower[:20]  # First 20 characters
                if any(pattern in query_start for pattern in new_request_patterns):
                    print(f"🆕 New ML request detected (despite continuation words) - clearing session")
                    state.interactive_session = None
                else:
                    print(f"🔄 Treating as continuation command despite new request words")
                    is_new_request = False
            elif is_new_request:
                print(f"🆕 New ML request detected - clearing interactive session and routing through orchestrator")
                state.interactive_session = None
            elif is_continuation or is_target_specification:
                print(f"🔄 Continuing interactive session: {state.interactive_session['agent_type']}")
                
                # Route to the appropriate agent to continue the interactive session
                agent_type = state.interactive_session['agent_type']
                if agent_type == "preprocessing":
                    # Handle preprocessing commands directly
                    return self._handle_preprocessing_interaction(state, query)
                elif agent_type == "feature_selection":
                    from agents_wrapper import feature_selection_agent
                    return self._prepare_response(feature_selection_agent.run(state))
                # Add other interactive agents as needed
        
        # Add raw data if provided
        if raw_data is not None:
            state.raw_data = raw_data
        
        # Progress callback is handled by ProgressTracker internally via SlackManager
        # No need for enhanced_update wrapper that causes duplicates
        
        try:
            if LANGGRAPH_AVAILABLE and self.app:
                # Run the full LangGraph pipeline
                result_state = self.app.invoke(state)
                
                # LangGraph returns dict when ending at END node - convert back to PipelineState
                if isinstance(result_state, dict):
                    # Update original state with dict values
                    for key, value in result_state.items():
                        if hasattr(state, key):
                            setattr(state, key, value)
                    result_state = state
                elif not isinstance(result_state, PipelineState):
                    # Fallback to original state for any other type
                    result_state = state
            else:
                # Simplified pipeline without LangGraph
                result_state = self._run_simplified_pipeline(state)
            
            # Save state (now guaranteed to be PipelineState)
            state_manager.save_state(result_state)
            
            # Prepare response
            response = self._prepare_response(result_state)
            
            # Log the response for debugging/monitoring
            print(f"📤 Response: {response['response']}")
            print(f"✅ Query processing completed for session {session_id}")
            return response
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Update state with error
            state.last_error = error_msg
            state_manager.save_state(state)
            
            # Prepare error response and save conversation history
            error_response = {
                "success": False,
                "error": error_msg,
                "session_id": session_id,
                "response": f"❌ Sorry, I encountered an error: {error_msg}"
            }
            
            # Log the error response for debugging/monitoring
            print(f"📤 Error Response: {error_response['response']}")
            
            # Save session state and conversation history for errors (single point)
            self._save_session_state(session_id, state)
            self._save_conversation_history(session_id, state.user_query, error_response["response"])
            
            return error_response
    
    def _prepare_response(self, state: PipelineState, custom_message: str = None) -> Dict[str, Any]:
        """Prepare response from final state"""
        response = {
            "success": state.last_error is None,
            "session_id": state.session_id,
            "chat_session": state.chat_session,
            "response": custom_message if custom_message else self._generate_response_text(state),
            "data_summary": state.get_data_summary(),
            "artifacts": state.artifacts or {},
            "execution_history": state.execution_history[-5:] if state.execution_history else []  # Last 5 records
        }
        
        if state.last_error:
            response["error"] = state.last_error
        
        # Save session state and conversation history (single point of truth)
        self._save_session_state(state.session_id, state)
        self._save_conversation_history(state.session_id, state.user_query, response["response"])
        
        return response
    
    def _handle_preprocessing_interaction(self, state: PipelineState, query: str):
        """Handle interactive preprocessing commands"""
        try:
            # Load previous session state to ensure we have the latest state
            previous_state = self._load_session_state(state.chat_session)
            if previous_state:
                print(f"📂 Loaded previous session state in preprocessing handler")
                print(f"🔧 DEBUG: Previous state keys: {list(previous_state.keys())}")
                # Restore preprocessing state if available
                if 'preprocessing_state' in previous_state:
                    state.preprocessing_state = previous_state['preprocessing_state']
                    print(f"🔧 DEBUG: Restored preprocessing_state in handler: {state.preprocessing_state}")
                    print(f"🔧 DEBUG: Current phase after restore: {state.preprocessing_state.get('current_phase')}")
                    print(f"🔧 DEBUG: Missing results after restore: {state.preprocessing_state.get('missing_results') is not None}")
                # Restore interactive session if available
                if 'interactive_session' in previous_state:
                    state.interactive_session = previous_state['interactive_session']
                    print(f"🔧 DEBUG: Restored interactive_session in handler: {state.interactive_session}")
            else:
                print(f"🔧 DEBUG: No previous state found in preprocessing handler")
            
            # Use the pipeline's slack_manager instead of the global one
            slack_manager = self.slack_manager
            query_lower = query.lower().strip()
            
            # Handle target column specification
            if state.interactive_session and state.interactive_session.get('phase') == 'need_target':
                if query_lower.startswith('target '):
                    target_col = query[7:].strip()
                else:
                    target_col = query.strip()
                
                # Validate target column
                if target_col in state.raw_data.columns:
                    state.target_column = target_col
                    state.interactive_session['target_column'] = target_col
                    state.interactive_session['phase'] = 'waiting_input'
                    
                    response_msg = f"""✅ **Target column set:** `{target_col}`

🧹 **Sequential Preprocessing Agent**

📊 **Current Dataset:** {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns
🎯 **Target Column:** {target_col}

**🔄 Preprocessing Phases:**
• `Overview` - Dataset analysis and summary
• `Outliers` - Detect and handle outliers  
• `Missing Values` - Handle missing data
• `Encoding` - Categorical variable encoding
• `Transformations` - Feature transformations

**💬 Your Options:**
• `proceed` - Start preprocessing workflow
• `skip overview` - Skip to outlier detection
• `explain outliers` - Learn about outlier handling
• `summary` - Show current status

💬 **What would you like to do?**"""
                    
                    slack_manager.send_message(state.chat_session, response_msg)
                    return self._prepare_response(state, f"Target column set to '{target_col}'. Ready for preprocessing!")
                else:
                    available_cols = list(state.raw_data.columns)
                    error_msg = f"""❌ **Column '{target_col}' not found.**

**Available columns:** {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}

Please specify a valid column name."""
                    
                    slack_manager.send_message(state.chat_session, error_msg)
                    return self._prepare_response(state, f"Column '{target_col}' not found. Please try again.")
            
            # Handle preprocessing commands
            elif 'proceed' in query_lower:
                # Start the actual preprocessing phases
                from agents_wrapper import preprocessing_agent
                
                # Check if we need to start the interactive workflow
                active_phase = (state.preprocessing_state or {}).get('current_phase')
                phase_active = active_phase in ['outliers', 'missing_values', 'encoding', 'transformations']
                if not phase_active:
                    # Start the interactive preprocessing workflow
                    print("🚀 Starting interactive preprocessing workflow")
                    # Pass the pipeline's slack_manager to the state
                    state._slack_manager = self.slack_manager
                    processed_state = preprocessing_agent.handle_interactive_command(state, 'proceed')
                    
                    # Debug: Check if preprocessing state was set
                    print(f"🔧 DEBUG: After proceed - preprocessing_state: {processed_state.preprocessing_state}")
                    print(f"🔧 DEBUG: After proceed - has outlier_results: {processed_state.preprocessing_state.get('outlier_results') is not None if processed_state.preprocessing_state else False}")
                    
                    # Save the updated state to session state file
                    self._save_session_state(processed_state.session_id, processed_state)
                    
                    # The preprocessing agent should handle the interactive flow
                    # and return the state with the interactive session set up
                    return self._prepare_response(processed_state, "Interactive preprocessing started.")
                else:
                    # Phase-aware: treat 'proceed' as 'continue' when already inside a phase
                    print("🔄 Proceed received in-phase → treating as 'continue'")
                    state._slack_manager = self.slack_manager
                    processed_state = preprocessing_agent.handle_interactive_command(state, 'continue')
                    self._save_session_state(processed_state.session_id, processed_state)
                    return self._prepare_response(processed_state, "Proceed mapped to continue in current phase.")
            
            elif 'continue' in query_lower:
                # Handle continue command for applying recommendations and moving to next phase
                from agents_wrapper import preprocessing_agent
                
                print("🔄 Handling continue command for preprocessing")
                # Debug: Check current state before continue
                print(f"🔧 DEBUG: Before continue - preprocessing_state: {state.preprocessing_state}")
                
                # Pass the pipeline's slack_manager to the state
                state._slack_manager = self.slack_manager
                processed_state = preprocessing_agent.handle_interactive_command(state, 'continue')
                
                # Save the updated state to session state file
                self._save_session_state(processed_state.session_id, processed_state)
                
                # Let the preprocessing agent handle the response, don't override it
                return self._prepare_response(processed_state, "Continue command processed.")
            
            # Check if we're in an active preprocessing phase and route to preprocessing agent
            elif state.preprocessing_state and state.preprocessing_state.get('current_phase') in ['outliers', 'missing_values', 'encoding', 'transformations']:
                # Route to preprocessing agent for phase-specific handling
                from agents_wrapper import preprocessing_agent
                
                print(f"🔄 Routing to preprocessing agent for phase: {state.preprocessing_state.get('current_phase')}")
                # Classify in-phase intent first (embeddings→keywords→heuristics)
                intent = self._classify_in_phase_intent(query)
                # Map intent to underlying commands handled by wrapper
                mapped = query
                if intent == 'proceed':
                    mapped = 'continue'
                elif intent == 'summary':
                    mapped = 'summary'
                elif intent == 'override':
                    mapped = 'override ' + query
                elif intent == 'skip':
                    mapped = query  # allows 'skip encoding' etc.
                else:
                    mapped = query  # query/help/navigate pass through
                # Pass the pipeline's slack_manager to the state
                state._slack_manager = self.slack_manager
                processed_state = preprocessing_agent.handle_interactive_command(state, mapped)
                
                # Save the updated state to session state file
                self._save_session_state(processed_state.session_id, processed_state)
                
                return self._prepare_response(processed_state, f"Processed in {state.preprocessing_state.get('current_phase')} phase.")
            
            elif 'summary' in query_lower:
                summary_msg = f"""📋 **Preprocessing Status**

📊 **Dataset:** {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns
🎯 **Target:** {state.target_column or 'Not set'}
🔄 **Phase:** {state.interactive_session.get('current_phase', 'Overview')}

**Available Commands:**
• `proceed` - Start preprocessing
• `explain [phase]` - Learn about phases
• `summary` - This status"""
                
                slack_manager.send_message(state.chat_session, summary_msg)
                return self._prepare_response(state, "Status summary sent.")
            
            else:
                # Default help message
                help_msg = """💬 **Available Commands:**
• `proceed` - Start preprocessing workflow
• `summary` - Show current status
• `explain outliers` - Learn about outlier handling
• `target column_name` - Set target column (if not set)

What would you like to do?"""
                
                slack_manager.send_message(state.chat_session, help_msg)
                return self._prepare_response(state, "Help message sent.")
                
        except Exception as e:
            print(f"❌ Error in preprocessing interaction: {e}")
            return self._prepare_response(state, f"Error processing command: {str(e)}")
    
    def _generate_response_text(self, state: PipelineState) -> str:
        """Generate human-readable response text"""
        if state.last_error:
            return f"❌ Operation failed: {state.last_error}"
        
        # Check if orchestrator generated a specific response (e.g., for general queries)
        if state.last_response:
            return state.last_response
        
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
        
        # Auto-detect target column if not set
        target_was_auto_detected = False
        if state.target_column is None and hasattr(data, 'columns'):
            print(f"🔧 DEBUG: Auto-detecting target column from columns: {list(data.columns)}")
            
            # Common target column names
            common_target_names = ['target', 'label', 'class', 'y', 'outcome', 'result', 'prediction', 'is_fraud', 'default_risk', 'churn', 'conversion']
            
            for col in data.columns:
                if col.lower() in common_target_names:
                    state.target_column = col
                    print(f"🎯 Auto-detected target column: {col}")
                    target_was_auto_detected = True
                    break
            
            # If no common name found, use the last column as target
            if state.target_column is None:
                state.target_column = data.columns[-1]
                print(f"🎯 Using last column as target: {state.target_column}")
                target_was_auto_detected = True
        
        state_manager.save_state(state)
        
        print(f"📊 Data loaded for session {session_id}: {data.shape if hasattr(data, 'shape') else 'Unknown shape'}")
        if state.target_column:
            print(f"🎯 Target column: {state.target_column}")
        
        # If target was auto-detected, automatically show preprocessing menu
        if target_was_auto_detected and state.target_column:
            # Only auto-show when the current intent is preprocessing/full_pipeline
            current_intent = getattr(state, 'current_intent', None) or getattr(state, 'user_intent', None)
            if not current_intent:
                # Try to infer from last routing decision if stored
                current_intent = getattr(state, 'last_route', None)
            if current_intent not in ['preprocessing', 'full_pipeline']:
                print("🎯 Target auto-detected but intent is not preprocessing/full_pipeline; skipping auto menu")
                return
            print("🎯 Target auto-detected - automatically showing preprocessing menu")
            # Set up interactive session
            state.interactive_session = {
                "agent_type": "preprocessing",
                "session_active": True,
                "session_id": state.chat_session,
                "phase": "waiting_input",
                "target_column": state.target_column,
                "current_phase": "overview"
            }
            
            # Send preprocessing menu via Slack
            # Use the pipeline's slack_manager instead of the global one
            slack_manager = self.slack_manager
            if slack_manager and state.chat_session:
                menu_msg = f"""🧹 **Sequential Preprocessing Agent**

📊 **Current Dataset:** {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns
🎯 **Target Column:** {state.target_column} (auto-detected)

**🔄 Preprocessing Phases:**
• `Overview` - Dataset analysis and summary
• `Outliers` - Detect and handle outliers  
• `Missing Values` - Handle missing data
• `Encoding` - Categorical variable encoding
• `Transformations` - Feature transformations

**💬 Your Options:**
• `proceed` - Start preprocessing workflow
• `skip overview` - Skip to outlier detection
• `explain outliers` - Learn about outlier handling
• `summary` - Show current status

💬 **What would you like to do?**"""
                
                slack_manager.send_message(state.chat_session, menu_msg)
                print("✅ Auto-sent preprocessing menu to Slack")
            
            state_manager.save_state(state)
    
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
