#!/usr/bin/env python3
from print_to_log import print_to_log
# Import master log handler to capture logger.info calls
try:
    import master_log_handler
except ImportError:
    pass

"""
LangGraph Multi-Agent ML Pipeline
Main orchestration system that coordinates preprocessing, feature selection, and model building agents
"""

from typing import Dict, Any, Optional, Callable, List
import tempfile
import os
import time
from datetime import datetime

import pandas as pd
# Import thread logging system
from thread_logger import get_thread_logger, close_thread_logger

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
    print_to_log("✅ LangGraph core components imported successfully")
    
except ImportError as e:
    print_to_log(f"❌ LangGraph core import failed: {e}")
    StateGraph = None
    END = "END"
    LANGGRAPH_AVAILABLE = False

# SQLite checkpointer not needed - we use our own persistence system

from pipeline_state import PipelineState, state_manager
from toolbox import initialize_toolbox


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
        
        # Get references to global toolbox components after initialization
        from toolbox import slack_manager, progress_tracker, user_directory_manager, execution_agent
        self.slack_manager = slack_manager
        self.progress_tracker = progress_tracker
        self.user_directory_manager = user_directory_manager
        self.execution_agent = execution_agent
        
        # Add state manager reference
        self.state_manager = state_manager
        
        if not LANGGRAPH_AVAILABLE:
            print_to_log("❌ LangGraph not available, using simplified pipeline")
            self.app = None
            self.graph = None
            self.checkpointer = None
            self.enable_persistence = False
        else:
            print_to_log("✅ LangGraph available, building full pipeline")
            
            # LangGraph checkpointing is optional - we use our own persistence
            self.enable_persistence = False  # Disable LangGraph checkpointing
            self.checkpointer = None
            print_to_log("📁 Using user directory persistence (LangGraph checkpointing disabled)")
            
            # Build the graph
            self.graph = self._build_graph()
            self.app = self.graph.compile(checkpointer=self.checkpointer)
        
        # User session management
        self.user_sessions = {}  # Store per-user-thread sessions
        
        print_to_log("🚀 Multi-Agent ML Pipeline initialized")
        print_to_log(f"   LangGraph: {'✅ Full Pipeline' if LANGGRAPH_AVAILABLE else '⚠️ Simplified Pipeline'}")
        print_to_log(f"   Persistence: ✅ User Directory + Session State")
        print_to_log(f"   Agents: Preprocessing, Feature Selection, Model Building")
        print_to_log(f"   Orchestrator: ✅ Ready")
    

    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        # Lazy import to avoid circular dependencies
        from orchestrator import orchestrator, AgentType
        from agents_wrapper import preprocessing_agent, feature_selection_agent, model_building_agent
        
        graph = StateGraph(PipelineState)
        
        # Add nodes
        graph.add_node("orchestrator", self._orchestrator_node)
        graph.add_node("preprocessing", self._preprocessing_node)
        graph.add_node("feature_selection", self._feature_selection_node)
        graph.add_node("model_building", self._model_building_node)
        graph.add_node("general_response", self._general_response_node)
        graph.add_node("code_execution", self._code_execution_node)
        graph.add_node("fast_pipeline", self._fast_pipeline_node)
        
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
                "fast_pipeline": "fast_pipeline",
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
        
        # Code execution, general response, and fast pipeline always end after completion
        graph.add_edge("code_execution", END)
        graph.add_edge("general_response", END)
        graph.add_edge("fast_pipeline", END)
        
        # Set entry point
        graph.set_entry_point("orchestrator")
        
        return graph
    
    def _orchestrator_node(self, state: PipelineState) -> PipelineState:
        """Orchestrator node - routes queries to appropriate agents"""
        # Lazy import to avoid circular dependencies
        from orchestrator import orchestrator
        
        print_to_log(f"\n🎯 [Orchestrator] Processing query: '{state.user_query}'")
        
        # Get thread logger
        if hasattr(state, 'session_id') and state.session_id:
            if '_' in state.session_id:
                parts = state.session_id.split('_')
                user_id = parts[0] if len(parts) >= 1 else state.session_id
                thread_id = '_'.join(parts[1:]) if len(parts) > 1 else state.session_id
            else:
                user_id = state.session_id
                thread_id = state.session_id
            thread_logger = get_thread_logger(user_id, thread_id)
        else:
            thread_logger = None
        
        # Route the query
        selected_agent = orchestrator.route(state)
        
        # Get routing explanation
        explanation = orchestrator.get_routing_explanation(state, selected_agent)
        
        # Log routing decision
        if thread_logger:
            thread_logger.log_routing(state.user_query, selected_agent)
            thread_logger.log_agent_switch(state.current_agent, selected_agent, explanation)
        
        # Update state
        state.current_agent = "Orchestrator"
        state.add_execution_record("Orchestrator", f"routed_to_{selected_agent}", explanation)
        
        # Store routing decision for conditional edges
        state.artifacts = state.artifacts or {}
        state.artifacts["routing_decision"] = selected_agent
        
        # Log routing decision (console only, not Slack)
        print_to_log(f"🔀 [Orchestrator] Routed to {selected_agent}: {explanation}")
        
        # Don't send routing details to Slack - user doesn't need to see internal routing
        
        # Send agent-specific interactive menus ONLY when user needs guidance
        try:
            slack_manager = self.slack_manager
            if slack_manager and state.chat_session:
                if selected_agent == 'preprocessing':
                    # Don't send preprocessing intro here - it will be shown after mode selection
                    print_to_log("⏭️ Skipping preprocessing intro - will show after mode selection")
                
                # For all other agents (model_building, feature_selection, code_execution, etc.):
                # Don't send ANY additional messages - the agents will handle their own responses
                
        except Exception as e:
            print_to_log(f"⚠️ Failed to send orchestrator message: {e}")
        
        return state
    
    def _preprocessing_node(self, state: PipelineState) -> PipelineState:
        """Preprocessing node - Ask for fast/slow mode selection before starting preprocessing"""
        print_to_log(f"\n🧹 [Preprocessing] Starting data preprocessing")
        
        # CRITICAL: Check if we have raw data before proceeding
        if state.raw_data is None:
            print_to_log(f"❌ [Preprocessing] No raw data available - cannot start preprocessing")
            state.last_response = "❌ No data available for preprocessing. Please upload a dataset first."
            return state
        
        # CRITICAL: Skip if we already have an interactive session for mode selection
        # This prevents overwriting state when user is in the middle of target/mode selection
        if (hasattr(state, 'interactive_session') and 
            state.interactive_session is not None and 
            state.interactive_session.get('needs_mode_selection', False)):
            print_to_log(f"⏭️ [Preprocessing] Skipping preprocessing node - interactive session already active for mode selection")
            return state
        
        # CRITICAL: Always ask for fast/slow mode selection when preprocessing is triggered
        # This ensures user can choose between automated (fast) or interactive (slow) pipeline
        
        # Check if target column is set, if not, we need to handle that first
        if not hasattr(state, 'target_column') or not state.target_column:
            print_to_log(f"🎯 [Preprocessing] Target column not set - will prompt user during preprocessing")
        
        # Check if orchestrator already handled mode selection
        if (hasattr(state, 'interactive_session') and 
            state.interactive_session and 
            state.interactive_session.get('mode_selected') == 'slow'):
            
            print_to_log(f"🎛️ [Preprocessing] Orchestrator already selected slow mode - proceeding with preprocessing")
            
            # Update existing session for preprocessing
            state.interactive_session.update({
                'agent_type': 'preprocessing',
                'current_phase': 'overview',
                'needs_mode_selection': False  # Mode already selected by orchestrator
            })
            
        else:
            # Create interactive session for mode selection (fallback case)
            state.interactive_session = {
                'agent_type': 'preprocessing',
                'session_active': True,
                'session_id': state.chat_session,
                'phase': 'need_target' if not (hasattr(state, 'target_column') and state.target_column) else 'mode_selection',
                'target_column': getattr(state, 'target_column', None),
                'current_phase': 'overview',
                'needs_target': not (hasattr(state, 'target_column') and state.target_column),
                'needs_mode_selection': True,
                'original_query': state.user_query  # CRITICAL: Store original query for fast pipeline
            }
        
        # Show appropriate message based on mode selection status
        if (hasattr(state, 'interactive_session') and 
            state.interactive_session and 
            state.interactive_session.get('mode_selected') == 'slow'):
            
            # Mode already selected - show preprocessing start message
            mode_choice_msg = f"""🎛️ Slow Mode Selected - Starting interactive preprocessing...

📋 Preprocessing Workflow:
Phase 1: 🚨 Outliers - Handle extreme values
Phase 2: 🗑️ Missing Values - Impute or remove nulls
Phase 3: 🏷️ Encoding - Convert categorical to numeric
Phase 4: 🔄 Transformations - Normalize and scale features

---

📊 Current Phase: Outlier Analysis
I'll detect extreme values that might affect your model and recommend handling strategies.

💬 Your Options:
• Type `proceed` or `yes` - Start outlier analysis
• Type `skip` - Skip this phase
• Type `explain` - Learn more about outliers

Ready to proceed?"""
            
        elif hasattr(state, 'target_column') and state.target_column:
            # Target set but mode not selected - show mode selection
            mode_choice_msg = f"""✅ Target column set: `{state.target_column}`

🚀 Choose Your ML Pipeline Mode



⚡ Fast Mode (Automated): 
• Complete ML pipeline without interaction
• AI handles all preprocessing decisions
• Get results in 2-3 minutes

🎛️ Slow Mode (Interactive): 
• Step-by-step guided process
• Review and approve each phase
• Full control over decisions

💬 Choose: Type `fast` or `slow`"""
        else:
            # No target column - prompt for target
            mode_choice_msg = f"""👋 Welcome to ModelAgent PRO!

📊 Dataset Uploaded: {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns

I'll help you build a machine learning model. Let's start!

💬 Step 1: Type your target column name
"""
        
        # Send appropriate message to Slack
        self.slack_manager.send_message(state.chat_session, mode_choice_msg)
        
        # CRITICAL: Clear last_response to prevent old cached responses from being returned
        # This prevents the previous bot introduction message from being displayed again
        state.last_response = None
        print_to_log(f"🧹 [Preprocessing] Cleared last_response to prevent cached response display")
        
        return state
    
    def _feature_selection_node(self, state: PipelineState) -> PipelineState:
        """Feature selection node"""
        # Lazy import to avoid circular dependencies
        from agents_wrapper import feature_selection_agent
        
        print_to_log(f"\n🎯 [Feature Selection] Starting feature selection")
        
        # ✅ USE PIPELINE'S SLACK MANAGER: Use the exact same instance as orchestrator
        slack_manager = self.slack_manager
        
        print_to_log(f"🔧 [FS Node] Using slack_manager id: {id(self.slack_manager)}")
        print_to_log(f"🔧 [FS Node] Slack manager has {len(self.slack_manager.session_channels)} channels")
        
        state.slack_session_info = {
            'channels': dict(self.slack_manager.session_channels),
            'threads': dict(self.slack_manager.session_threads)
        }
        state._slack_manager = self.slack_manager
        print_to_log(f"💾 [FS Node] Passed slack session info: {len(state.slack_session_info['channels'])} channels")
        
        return feature_selection_agent.run(state)
    
    def _model_building_node(self, state: PipelineState) -> PipelineState:
        """Model building node"""
        # Lazy import to avoid circular dependencies
        from agents_wrapper import model_building_agent
        
        print_to_log(f"\n🤖 [Model Building] Starting model building")
        print_to_log(f"🔍 [Model Building] Query: '{state.user_query}'")
        print_to_log(f"🔍 [Model Building] Raw data: {'✅' if state.raw_data is not None else '❌'}")
        print_to_log(f"🔍 [Model Building] Cleaned data: {'✅' if state.cleaned_data is not None else '❌'}")
        print_to_log(f"🔍 [Model Building] Selected features: {'✅' if state.selected_features is not None else '❌'}")
        return model_building_agent.run(state)
    
    def _general_response_node(self, state: PipelineState) -> PipelineState:
        """General response node - handles conversational queries using LLM"""
        print_to_log(f"\n💬 [General Response] Generating conversational response")
        # Note: No progress update to Slack - user doesn't need routing details
        
        # CRITICAL: Check if orchestrator already set a response (e.g., target selection prompt)
        if state.last_response:
            print_to_log(f"💬 [General Response] Using orchestrator-provided response: {state.last_response[:100]}...")
            return state
        
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
            
            # Check if this is a non-data science query with context from orchestrator
            if hasattr(state, 'non_data_science_context') and state.non_data_science_context:
                print_to_log(f"🤖 Handling non-data science query with intelligent context")
                context = state.non_data_science_context
                
                # Create intelligent prompt based on query type
                context_prompt = f"""The user said: '{query}'

This query was classified as not directly related to data science (confidence: {context['classification_confidence']:.2f}).

INSTRUCTIONS FOR RESPONSE:
1. If it's a greeting (hi, hello, hey, etc.) → Respond warmly and briefly introduce yourself as a data science assistant
2. If it's a polite question somewhat related to data/analysis → Answer helpfully but gently guide toward data science topics
3. If it's completely off-topic (weather, sports, etc.) → Politely redirect while being friendly
4. If it's a general question about AI/ML concepts → Answer educationally and encourage specific data science tasks

Be conversational, friendly, and helpful. Don't be harsh or robotic. Show personality while staying professional."""
                
                system_prompt = "You are a friendly, specialized AI assistant for data science and machine learning. You're helpful and conversational, but your expertise is in ML, data analysis, and statistics. Handle non-data science queries gracefully - be warm with greetings, helpful with related topics, and gently redirect off-topic queries while maintaining a friendly tone."
                
            elif state.raw_data is not None:
                context_prompt = f"The user said: '{query}'. I have their dataset with {state.raw_data.shape[0]:,} rows and {state.raw_data.shape[1]} columns. Respond naturally and conversationally. Only mention specific capabilities if they ask 'what can you do' or similar questions."
                system_prompt = "You are a specialized AI assistant for data science and machine learning. You help users build models, analyze data, and work with datasets. When greeting users, be friendly and natural. When asked about capabilities, mention your ML/data science skills like building models, data analysis, visualization, etc. Keep responses conversational and concise."
            else:
                context_prompt = f"The user said: '{query}'. Respond naturally and conversationally as an AI assistant. Don't list capabilities unless they specifically ask what you can do."
                system_prompt = "You are a specialized AI assistant for data science and machine learning. You help users build models, analyze data, and work with datasets. When greeting users, be friendly and natural. When asked about capabilities, mention your ML/data science skills like building models, data analysis, visualization, etc. Keep responses conversational and concise."
            
            print_to_log(f"🔍 Generating contextual response for: '{query}'")
            
            # Use LLM for intelligent contextual response
            response = ollama.chat(
                model=os.getenv("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M"),  # Use environment variable
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_prompt}
                ]
            )
            
            generated_response = response["message"]["content"].strip()
            state.last_response = generated_response
            
            # Clear non-data science context after use
            if hasattr(state, 'non_data_science_context'):
                state.non_data_science_context = None
            
            print_to_log(f"✅ Generated contextual response: {generated_response[:100]}...")
            
        except Exception as e:
            print_to_log(f"❌ Error generating conversational response: {e}")
            # Fallback response
            state.last_response = "Hello! I'm your ML assistant. I can help you with data preprocessing, feature selection, and model building. How can I assist you today?"
        
        return state
    
    def _fast_pipeline_node(self, state: PipelineState) -> PipelineState:
        """Fast pipeline node - runs automated ML pipeline"""
        print_to_log(f"\n⚡ [Fast Pipeline] Starting automated ML pipeline")
        
        try:
            # Import and run automated pipeline agent
            from automated_pipeline_agent import automated_pipeline_agent
            
            # Save state before calling automated pipeline
            self._save_session_state(state.session_id, state)
            
            # Run automated pipeline
            result_state = automated_pipeline_agent(state)
            
            # Save result state
            self._save_session_state(state.session_id, result_state)
            
            print_to_log(f"✅ [Fast Pipeline] Automated pipeline completed")
            return result_state
            
        except Exception as e:
            print_to_log(f"❌ [Fast Pipeline] Error: {e}")
            state.last_error = str(e)
            state.last_response = f"❌ Fast pipeline failed: {str(e)}"
            return state
    
    def _code_execution_node(self, state: PipelineState) -> PipelineState:
        """Code execution node - handles general code execution requests by generating Python code first"""
        print_to_log(f"\n💻 [Code Execution] Processing code execution request")
        
        try:
            # Import the LLM-based code generation from ModelBuildingAgent
            from model_building_agent_impl import generate_model_code
            
            user_query = state.user_query
            user_id = state.chat_session
            
            print_to_log(f"🔍 [Code Execution] Query: '{user_query}'")
            print_to_log(f"🔍 [Code Execution] Raw data: {'✅' if state.raw_data is not None else '❌'}")
            print_to_log(f"🔍 [Code Execution] Cleaned data: {'✅' if state.cleaned_data is not None else '❌'}")
            
            # Generate Python code using LLM (similar to ModelBuildingAgent approach)
            code_generation_prompt = f"""You are a Python expert. Generate executable Python code for the user's request.

CRITICAL INSTRUCTIONS:
- Return ONLY executable Python code (no markdown, no explanations)
- The variable `sample_data` is ALREADY LOADED and available
- Use `safe_plt_savefig()` for saving plots (already available, no import needed)
- DO NOT import datasets or load external data
- Code should be complete and ready to execute
- For calculations, return results in a `result` dictionary

AVAILABLE DATA:
- `sample_data`: Main DataFrame with shape {state.raw_data.shape if state.raw_data is not None else 'No data'}
- Standard libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns)

USER REQUEST: {user_query}

Generate Python code to fulfill this request:"""

            # Generate code using the same LLM as ModelBuildingAgent
            print_to_log(f"🤔 Generating Python code for: {user_query}")
            
            # Use the same code generation function as ModelBuildingAgent
            reply, generated_code, system_prompt = generate_model_code(code_generation_prompt, user_id, user_query)
            
            if not generated_code:
                state.last_response = f"❌ Code generation failed: {reply}"
                return state
            
            if not generated_code or generated_code.strip() == "":
                state.last_response = f"❌ No code was generated. Please provide a more specific request."
                return state
            
            print_to_log(f"✅ Generated {len(generated_code)} characters of Python code")
            print_to_log(f"📝 Code preview: {generated_code[:100]}...")
            
            # Execute the generated code using ExecutionAgent
            context = {
                "raw_data": state.raw_data,
                "cleaned_data": state.cleaned_data,
                "selected_features": state.selected_features,
                "trained_model": state.trained_model,
                "sample_data": state.raw_data or state.cleaned_data  # Ensure sample_data is available
            }
            
            print_to_log(f"⚙️ Executing generated code...")
            result_state = self.execution_agent.run_code(state, generated_code, context)
            
            # Update response based on execution result
            if result_state.last_error:
                state.last_response = f"❌ Code execution failed: {result_state.last_error}"
            else:
                state.last_response = "✅ Code executed successfully! Check the results above."
            
            return state
            
        except Exception as e:
            print_to_log(f"❌ Code execution error: {e}")
            import traceback
            print_to_log(f"🔍 Full traceback: {traceback.format_exc()}")
            state.last_error = str(e)
            state.last_response = f"❌ Code execution failed: {str(e)}"
            return state
    
    
    def _get_user_session_dir(self, session_id: str) -> str:
        """Get user session directory for conversation history"""
        return self.user_directory_manager.ensure_user_directory(session_id)
    
    def _save_conversation_history(self, session_id: str, user_query: str, response: str, state: PipelineState = None):
        """Save conversation history to user directory with comprehensive strategy details"""
        try:
            user_dir = self._get_user_session_dir(session_id)
            history_file = os.path.join(user_dir, "conversation_history.json")
            
            print_to_log(f"💾 Saving conversation history to: {history_file}")
            
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
                            print_to_log(f"⚠️ Invalid conversation history format (not a list), starting fresh")
                            history = []
                    print_to_log(f"📚 Loaded {len(history)} existing conversations")
                except (json.JSONDecodeError, Exception) as e:
                    print_to_log(f"⚠️ Corrupted conversation history file, starting fresh: {e}")
                    history = []
            
            # Prepare comprehensive conversation entry
            conversation = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "response": response,
                "session_id": session_id
            }
            
            # Add comprehensive preprocessing strategies if available
            if state and hasattr(state, 'preprocessing_strategies') and state.preprocessing_strategies:
                conversation["preprocessing_pipeline"] = self._extract_comprehensive_strategies(state)
                print_to_log(f"📊 Added comprehensive preprocessing strategies to conversation history")
            
            # Add dataset information if available
            if state:
                conversation["dataset_info"] = self._extract_dataset_info(state)
            
            history.append(conversation)
            
            # Save updated history
            import json
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            print_to_log(f"✅ Conversation history saved ({len(history)} total conversations)")
                
        except Exception as e:
            print_to_log(f"⚠️ Failed to save conversation history: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_comprehensive_strategies(self, state: PipelineState) -> Dict:
        """Extract comprehensive preprocessing strategies with all parameters for reapplication"""
        try:
            strategies = state.preprocessing_strategies
            if not strategies:
                return {}
            
            # Get current data for computing parameters
            current_data = None
            if hasattr(state, 'cleaned_data') and state.cleaned_data is not None:
                current_data = state.cleaned_data
            elif hasattr(state, 'raw_data') and state.raw_data is not None:
                current_data = state.raw_data
            
            comprehensive_strategies = {
                "execution_order": strategies.get("strategy_metadata", {}).get("processing_order", []),
                "total_phases_completed": len(strategies.get("strategy_metadata", {}).get("processing_order", [])),
                "user_overrides": strategies.get("strategy_metadata", {}).get("user_overrides", {}),
                "strategies": {
                    "outlier_strategies": self._enhance_outlier_strategies(strategies.get("outlier_strategies", {}), current_data),
                    "missing_value_strategies": self._enhance_missing_value_strategies(strategies.get("missing_value_strategies", {}), current_data),
                    "encoding_strategies": self._enhance_encoding_strategies(strategies.get("encoding_strategies", {}), current_data),
                    "transformation_strategies": self._enhance_transformation_strategies(strategies.get("transformation_strategies", {}), current_data)
                }
            }
            
            return comprehensive_strategies
            
        except Exception as e:
            print_to_log(f"⚠️ Error extracting comprehensive strategies: {e}")
            return {}
    
    def _extract_dataset_info(self, state: PipelineState) -> Dict:
        """Extract dataset information for conversation history"""
        try:
            dataset_info = {
                "target_column": state.target_column,
                "session_created": state.created_at.isoformat() if state.created_at else None,
                "last_updated": state.updated_at.isoformat() if state.updated_at else None
            }
            
            # Add data shape information
            if hasattr(state, 'raw_data') and state.raw_data is not None:
                dataset_info["original_shape"] = list(state.raw_data.shape)
                dataset_info["original_columns"] = list(state.raw_data.columns)
            
            if hasattr(state, 'cleaned_data') and state.cleaned_data is not None:
                dataset_info["processed_shape"] = list(state.cleaned_data.shape)
                dataset_info["final_columns"] = list(state.cleaned_data.columns)
            
            if hasattr(state, 'selected_features') and state.selected_features:
                dataset_info["selected_features"] = state.selected_features
                dataset_info["selected_feature_count"] = len(state.selected_features)
            
            return dataset_info
            
        except Exception as e:
            print_to_log(f"⚠️ Error extracting dataset info: {e}")
            return {}
    
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
            print_to_log(f"⚠️ Failed to load conversation history: {e}")
            return []
    
    def _enhance_outlier_strategies(self, outlier_strategies: Dict, current_data) -> Dict:
        """Enhance outlier strategies with computed values for reapplication"""
        import pandas as pd
        import pandas as pd
        
        enhanced = {}
        
        for col, strategy in outlier_strategies.items():
            enhanced_strategy = strategy.copy()
            
            # Add computed values if data is available
            if current_data is not None and col in current_data.columns:
                try:
                    col_data = current_data[col]
                    if pd.api.types.is_numeric_dtype(col_data):
                        treatment = strategy.get('treatment', 'keep')
                        
                        # Initialize computed_values if not present
                        if 'computed_values' not in enhanced_strategy:
                            enhanced_strategy['computed_values'] = {}
                        
                        # Compute percentiles for winsorization
                        if treatment in ['winsorize', 'winsorize_1st_99th']:
                            enhanced_strategy['computed_values']['q01'] = float(col_data.quantile(0.01))
                            enhanced_strategy['computed_values']['q99'] = float(col_data.quantile(0.99))
                        elif treatment == 'winsorize_5th_95th':
                            enhanced_strategy['computed_values']['q05'] = float(col_data.quantile(0.05))
                            enhanced_strategy['computed_values']['q95'] = float(col_data.quantile(0.95))
                        
                        # Compute values for imputation strategies
                        if treatment in ['mean_imputation', 'median_imputation']:
                            enhanced_strategy['computed_values']['mean_value'] = float(col_data.mean())
                            enhanced_strategy['computed_values']['median_value'] = float(col_data.median())
                            # Store outlier bounds for identification
                            enhanced_strategy['computed_values']['outlier_bounds'] = {
                                'lower': float(col_data.quantile(0.01)),
                                'upper': float(col_data.quantile(0.99))
                            }
                        
                        # Add timestamp
                        enhanced_strategy['applied_at'] = datetime.now().isoformat()
                        
                except Exception as e:
                    print_to_log(f"⚠️ Error computing outlier values for {col}: {e}")
            
            enhanced[col] = enhanced_strategy
        
        return enhanced
    
    def _enhance_missing_value_strategies(self, missing_strategies: Dict, current_data) -> Dict:
        """Enhance missing value strategies with computed values for reapplication"""
        import pandas as pd
        
        enhanced = {}
        
        import pandas as pd
        
        for col, strategy in missing_strategies.items():
            enhanced_strategy = strategy.copy()
            
            # Add computed values if data is available
            if current_data is not None and col in current_data.columns:
                try:
                    col_data = current_data[col]
                    strategy_type = strategy.get('strategy', 'median')
                    
                    # Initialize computed_values if not present
                    if 'computed_values' not in enhanced_strategy:
                        enhanced_strategy['computed_values'] = {}
                    
                    # Compute statistical values
                    if strategy_type == 'mean' and pd.api.types.is_numeric_dtype(col_data):
                        enhanced_strategy['computed_values']['mean_value'] = float(col_data.mean())
                    elif strategy_type == 'median' and pd.api.types.is_numeric_dtype(col_data):
                        enhanced_strategy['computed_values']['median_value'] = float(col_data.median())
                    elif strategy_type == 'mode':
                        mode_values = col_data.mode()
                        if not mode_values.empty:
                            enhanced_strategy['computed_values']['mode_value'] = mode_values.iloc[0]
                    
                    # For model-based imputation, store predictor information
                    if strategy_type == 'model_based':
                        # Find correlated columns (simplified approach)
                        numeric_cols = current_data.select_dtypes(include=['number']).columns
                        correlations = {}
                        for other_col in numeric_cols:
                            if other_col != col and not current_data[other_col].isna().all():
                                try:
                                    corr = current_data[col].corr(current_data[other_col])
                                    if abs(corr) > 0.3:  # Threshold for useful correlation
                                        correlations[other_col] = float(corr)
                                except:
                                    continue
                        
                        if correlations:
                            enhanced_strategy['computed_values']['predictor_columns'] = list(correlations.keys())
                            enhanced_strategy['computed_values']['predictor_means'] = {
                                pred_col: float(current_data[pred_col].mean()) 
                                for pred_col in correlations.keys()
                            }
                    
                    # Add missing statistics
                    enhanced_strategy['computed_values']['missing_count'] = int(col_data.isna().sum())
                    enhanced_strategy['computed_values']['missing_percentage'] = float(col_data.isna().sum() / len(col_data) * 100)
                    enhanced_strategy['applied_at'] = datetime.now().isoformat()
                    
                except Exception as e:
                    print_to_log(f"⚠️ Error computing missing value parameters for {col}: {e}")
            
            enhanced[col] = enhanced_strategy
        
        return enhanced
    
    def _enhance_encoding_strategies(self, encoding_strategies: Dict, current_data) -> Dict:
        """Enhance encoding strategies with computed mappings for reapplication"""
        import pandas as pd
        
        enhanced = {}
        import pandas as pd        
        for col, strategy in encoding_strategies.items():
            enhanced_strategy = strategy.copy()
            
            # Add computed mappings if data is available
            if current_data is not None and col in current_data.columns:
                try:
                    col_data = current_data[col]
                    strategy_type = strategy.get('strategy', 'label_encoding')
                    
                    # Initialize computed_mappings if not present
                    if 'computed_mappings' not in enhanced_strategy:
                        enhanced_strategy['computed_mappings'] = {}
                    
                    if strategy_type in ['label_encoding', 'label']:
                        # Create label mapping
                        unique_values = col_data.dropna().unique()
                        label_mapping = {str(val): idx for idx, val in enumerate(sorted(unique_values))}
                        reverse_mapping = {idx: str(val) for val, idx in label_mapping.items()}
                        
                        enhanced_strategy['computed_mappings']['label_mapping'] = label_mapping
                        enhanced_strategy['computed_mappings']['reverse_mapping'] = reverse_mapping
                        enhanced_strategy['computed_mappings']['categories_count'] = len(unique_values)
                    
                    elif strategy_type in ['onehot_encoding', 'onehot']:
                        # Store categories for one-hot encoding
                        unique_values = sorted(col_data.dropna().unique())
                        enhanced_strategy['computed_mappings']['categories'] = [str(val) for val in unique_values]
                        enhanced_strategy['computed_mappings']['column_names'] = [f"{col}_{val}" for val in unique_values]
                        
                        # Add category counts
                        category_counts = col_data.value_counts().to_dict()
                        enhanced_strategy['computed_mappings']['category_counts'] = {
                            str(k): int(v) for k, v in category_counts.items()
                        }
                    
                    elif strategy_type in ['target_encoding', 'target']:
                        # Compute target means (requires target column)
                        target_col = current_data.columns[-1]  # Assume last column is target for now
                        if target_col in current_data.columns and pd.api.types.is_numeric_dtype(current_data[target_col]):
                            target_means = current_data.groupby(col)[target_col].mean().to_dict()
                            global_mean = current_data[target_col].mean()
                            
                            enhanced_strategy['computed_mappings']['target_means'] = {
                                str(k): float(v) for k, v in target_means.items()
                            }
                            enhanced_strategy['computed_mappings']['global_mean'] = float(global_mean)
                            enhanced_strategy['computed_mappings']['smoothing_factor'] = 10  # Default smoothing
                    
                    elif strategy_type == 'binary_encoding':
                        # Compute binary encoding mapping
                        unique_values = sorted(col_data.dropna().unique())
                        import math
                        num_bits = max(1, math.ceil(math.log2(len(unique_values))))
                        
                        binary_mapping = {}
                        for idx, val in enumerate(unique_values):
                            binary_str = format(idx, f'0{num_bits}b')
                            binary_mapping[str(val)] = binary_str
                        
                        enhanced_strategy['computed_mappings']['binary_mapping'] = binary_mapping
                        enhanced_strategy['computed_mappings']['num_bits'] = num_bits
                    
                    enhanced_strategy['applied_at'] = datetime.now().isoformat()
                    
                except Exception as e:
                    print_to_log(f"⚠️ Error computing encoding parameters for {col}: {e}")
            
            enhanced[col] = enhanced_strategy
        
        return enhanced
    
    def _enhance_transformation_strategies(self, transform_strategies: Dict, current_data) -> Dict:
        """Enhance transformation strategies with computed parameters for reapplication"""
        import pandas as pd
        
        enhanced = {}
        
        import pandas as pd
        
        for col, strategy in transform_strategies.items():
            enhanced_strategy = strategy.copy()
            
            # Add computed parameters if data is available
            if current_data is not None and col in current_data.columns:
                try:
                    col_data = current_data[col]
                    if pd.api.types.is_numeric_dtype(col_data):
                        transformation = strategy.get('transformation', 'standardize')
                        
                        # Initialize computed_values if not present
                        if 'computed_values' not in enhanced_strategy:
                            enhanced_strategy['computed_values'] = {}
                        
                        if transformation == 'standardize':
                            enhanced_strategy['computed_values']['mean'] = float(col_data.mean())
                            enhanced_strategy['computed_values']['std'] = float(col_data.std())
                        
                        elif transformation == 'normalize':
                            enhanced_strategy['computed_values']['min'] = float(col_data.min())
                            enhanced_strategy['computed_values']['max'] = float(col_data.max())
                        
                        elif transformation == 'robust_scale':
                            enhanced_strategy['computed_values']['median'] = float(col_data.median())
                            enhanced_strategy['computed_values']['q25'] = float(col_data.quantile(0.25))
                            enhanced_strategy['computed_values']['q75'] = float(col_data.quantile(0.75))
                            enhanced_strategy['computed_values']['iqr'] = float(col_data.quantile(0.75) - col_data.quantile(0.25))
                        
                        elif transformation in ['log1p', 'sqrt']:
                            enhanced_strategy['computed_values']['min_value'] = float(col_data.min())
                            enhanced_strategy['computed_values']['offset'] = 1.0
                            if transformation == 'log1p':
                                enhanced_strategy['computed_values']['transformation_formula'] = 'log1p(x - min + offset)'
                            else:
                                enhanced_strategy['computed_values']['transformation_formula'] = 'sqrt(x - min + offset)'
                        
                        elif transformation == 'square':
                            enhanced_strategy['computed_values']['transformation_formula'] = 'x^2'
                        
                        elif transformation == 'quantile':
                            # Compute quantile transformation parameters
                            quantiles = [i/100.0 for i in range(101)]  # 0.00 to 1.00 in steps of 0.01
                            quantile_values = [float(col_data.quantile(q)) for q in quantiles]
                            
                            enhanced_strategy['computed_values']['quantiles'] = quantiles
                            enhanced_strategy['computed_values']['quantile_values'] = quantile_values
                            enhanced_strategy['computed_values']['n_quantiles'] = 100
                        
                        # Add distribution statistics
                        enhanced_strategy['computed_values']['original_skewness'] = float(col_data.skew())
                        enhanced_strategy['applied_at'] = datetime.now().isoformat()
                    
                except Exception as e:
                    print_to_log(f"⚠️ Error computing transformation parameters for {col}: {e}")
            
            enhanced[col] = enhanced_strategy
        
        return enhanced
    
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
                # Only write raw_data if file doesn't exist (raw data is constant)
                if not os.path.exists(raw_data_file):
                    print_to_log(f"💾 Writing raw_data for first time: {state.raw_data.shape}")
                    state.raw_data.to_csv(raw_data_file, index=False)
                else:
                    print_to_log(f"⏭️ Skipping raw_data rewrite (file exists): {state.raw_data.shape}")
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
                print_to_log(f"💾 Saved cleaned_data to session: {state.cleaned_data.shape}")
                print_to_log(f"📁 Data saved to: {cleaned_data_file}")
                print_to_log(f"🔧 DEBUG: Data columns: {list(state.cleaned_data.columns)}")
            else:
                state_dict['cleaned_data'] = None
                
            # Handle predictions_dataset DataFrame
            if 'predictions_dataset' in state_dict and state.predictions_dataset is not None:
                predictions_data_file = os.path.join(user_dir, "predictions_dataset.csv")
                state.predictions_dataset.to_csv(predictions_data_file, index=False)
                state_dict['predictions_dataset'] = {"type": "dataframe", "file": "predictions_dataset.csv", "shape": list(state.predictions_dataset.shape)}
                print_to_log(f"💾 Saved predictions_dataset to session: {state.predictions_dataset.shape}")
                print_to_log(f"📁 Predictions data saved to: {predictions_data_file}")
            else:
                state_dict['predictions_dataset'] = None
            
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
            print_to_log(f"⚠️ Failed to save session state: {e}")
    
    def _load_session_state(self, session_id: str) -> Optional[Dict]:
        """Load session state from user directory including DataFrames"""
        print_to_log(f"🔧 DEBUG LOAD_SESSION: Called for session {session_id}")
        try:
            user_dir = self._get_user_session_dir(session_id)
            state_file = os.path.join(user_dir, "session_state.json")
            print_to_log(f"🔧 DEBUG LOAD_SESSION: Looking for state file: {state_file}")
            print_to_log(f"🔧 DEBUG LOAD_SESSION: State file exists: {os.path.exists(state_file)}")
            
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
                        print_to_log(f"📂 Restored raw_data: {state_dict['raw_data'].shape}")
                
                if state_dict.get('processed_data'):
                    if isinstance(state_dict['processed_data'], dict):
                        processed_data_file = os.path.join(user_dir, state_dict['processed_data']['file'])
                        if os.path.exists(processed_data_file):
                            state_dict['processed_data'] = pd.read_csv(processed_data_file)
                            print_to_log(f"📂 Restored processed_data: {state_dict['processed_data'].shape}")
                        else:
                            state_dict['processed_data'] = None
                            print_to_log(f"🔧 DEBUG LOAD_SESSION: processed_data file not found, set to None")
                    elif isinstance(state_dict['processed_data'], str):
                        # Handle case where processed_data was saved as string repr instead of file path
                        state_dict['processed_data'] = None
                        print_to_log(f"🔧 DEBUG LOAD_SESSION: processed_data was string, set to None")
                else:
                    print_to_log(f"🔧 DEBUG LOAD_SESSION: No processed_data in state_dict or not dict format")
                
                if state_dict.get('cleaned_data') and isinstance(state_dict['cleaned_data'], dict):
                    cleaned_data_file = os.path.join(user_dir, state_dict['cleaned_data']['file'])
                    print_to_log(f"🔧 DEBUG LOAD_SESSION: cleaned_data file path: {cleaned_data_file}")
                    print_to_log(f"🔧 DEBUG LOAD_SESSION: cleaned_data file exists: {os.path.exists(cleaned_data_file)}")
                    if os.path.exists(cleaned_data_file):
                        state_dict['cleaned_data'] = pd.read_csv(cleaned_data_file)
                        print_to_log(f"📂 Restored cleaned_data: {state_dict['cleaned_data'].shape}")
                    else:
                        print_to_log(f"⚠️ DEBUG LOAD_SESSION: cleaned_data file not found, setting to None")
                        state_dict['cleaned_data'] = None
                else:
                    print_to_log(f"🔧 DEBUG LOAD_SESSION: No cleaned_data in state_dict or not dict format")
                
                # Restore predictions_dataset from CSV file
                if state_dict.get('predictions_dataset') and isinstance(state_dict['predictions_dataset'], dict):
                    predictions_data_file = os.path.join(user_dir, state_dict['predictions_dataset']['file'])
                    if os.path.exists(predictions_data_file):
                        state_dict['predictions_dataset'] = pd.read_csv(predictions_data_file)
                        print_to_log(f"📂 Restored predictions_dataset: {state_dict['predictions_dataset'].shape}")
                    else:
                        print_to_log(f"⚠️ DEBUG LOAD_SESSION: predictions_dataset file not found, setting to None")
                        state_dict['predictions_dataset'] = None
                else:
                    print_to_log(f"🔧 DEBUG LOAD_SESSION: No predictions_dataset in state_dict or not dict format")
                
                print_to_log(f"🔧 DEBUG LOAD_SESSION: Final state_dict keys: {list(state_dict.keys())}")
                print_to_log(f"🔧 DEBUG LOAD_SESSION: cleaned_data is None: {state_dict.get('cleaned_data') is None}")
                return state_dict
            return None
            
        except Exception as e:
            print_to_log(f"⚠️ Failed to load session state: {e}")
            return None
    
    def _route_to_agent(self, state: PipelineState) -> str:
        """Conditional edge function for routing from orchestrator"""
        # Lazy import to avoid circular dependencies
        from orchestrator import AgentType
        
        routing_decision = state.artifacts.get("routing_decision", AgentType.END.value)
        print_to_log(f"🔀 Routing to: {routing_decision}")
        return routing_decision
    
    def _determine_next_step(self, state: PipelineState) -> str:
        """Determine next step after agent execution"""
        # Check if there was an error
        if state.last_error:
            print_to_log(f"❌ Error detected: {state.last_error}")
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
            print_to_log("✅ Single-step request completed")
            return END
        
        # For pipeline requests, continue to next logical step
        current_agent = state.current_agent
        
        if current_agent == "PreprocessingAgent":
            if state.cleaned_data is not None:
                # For model building requests, always continue to feature selection
                if any(word in query for word in ["train", "model", "build", "pipeline", "complete", "lgbm", "classifier", "regressor"]):
                    print_to_log("🔄 Continuing to feature selection for model building request")
                    return "feature_selection"
                else:
                    return END
        
        elif current_agent == "FeatureSelectionAgent":
            if state.selected_features is not None:
                # For model building requests, always continue to model building
                if any(word in query for word in ["train", "model", "build", "pipeline", "complete", "lgbm", "classifier", "regressor"]):
                    print_to_log("🔄 Continuing to model building")
                    return "model_building"
                else:
                    return END
        
        elif current_agent == "ModelBuildingAgent":
            # Model building is typically the end
            print_to_log("✅ Model building completed - pipeline finished")
            return END
        
        return END
    
    def _run_simplified_pipeline(self, state: PipelineState) -> PipelineState:
        """Run simplified pipeline without LangGraph"""
        # Lazy import to avoid circular dependencies
        from orchestrator import orchestrator, AgentType
        
        print_to_log("🔄 Running simplified pipeline (LangGraph not available)")
        
        # Store the original user intent
        original_query = state.user_query
        original_intent = orchestrator._classify_with_keyword_scoring(original_query)[0]
        print_to_log(f"[SimplifiedPipeline] Original intent: {original_intent}")
        
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
            print_to_log(f"[SimplifiedPipeline] Step {step_count}: Routing to {selected_agent}")
            
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
        # Lazy import to avoid circular dependencies
        from orchestrator import AgentType
        from agents_wrapper import preprocessing_agent, feature_selection_agent, model_building_agent
        
        if agent_type == AgentType.PREPROCESSING.value:
            return preprocessing_agent.run(state)
        elif agent_type == AgentType.FEATURE_SELECTION.value:
            # ✅ USE PIPELINE'S SLACK MANAGER: Use the exact same instance as orchestrator
            slack_manager = self.slack_manager
            
            print_to_log(f"🔧 [Execute Agent] Using slack_manager id: {id(self.slack_manager)}")
            print_to_log(f"🔧 [Execute Agent] Slack manager has {len(self.slack_manager.session_channels)} channels")
            
            state.slack_session_info = {
                'channels': dict(self.slack_manager.session_channels),
                'threads': dict(self.slack_manager.session_threads)
            }
            state._slack_manager = self.slack_manager
            print_to_log(f"💾 [Execute Agent] Passed slack session info: {len(state.slack_session_info['channels'])} channels")
            return feature_selection_agent.run(state)
        elif agent_type == AgentType.MODEL_BUILDING.value:
            return model_building_agent.run(state)
        elif agent_type == "general_response":
            # Handle general response in simplified pipeline
            return self._general_response_node(state)
        else:
            print_to_log(f"[SimplifiedPipeline] Unknown agent type: {agent_type}")
            return state
    
    def _find_target_column_enhanced(self, user_specified: str, available_columns: list) -> tuple:
        """
        Find target column using ENHANCED fuzzy matching.
        
        Handles variations like:
        - f_segment → f_segment (exact)
        - fsegment → f_segment (normalized)
        - F_segment → f_segment (case insensitive)
        - f segment → f_segment (normalized)
        - segment → f_segment (partial)
        """
        if not user_specified or not available_columns:
            return None, 'none'
        
        user_specified = user_specified.strip()
        available_columns = [col.strip() for col in available_columns]
        
        # 1. Exact match
        if user_specified in available_columns:
            return user_specified, 'exact'
        
        # 2. Case-insensitive match
        user_lower = user_specified.lower()
        for col in available_columns:
            if col.lower() == user_lower:
                return col, 'case_insensitive'
        
        # 3. ENHANCED: Normalized matching (remove spaces, underscores, special chars)
        def normalize_text(text: str) -> str:
            """Remove spaces, underscores, and convert to lowercase"""
            import re
            return re.sub(r'[_\s\-\.]+', '', text.lower())
        
        user_normalized = normalize_text(user_specified)
        if user_normalized:  # Only proceed if we have normalized text
            for col in available_columns:
                col_normalized = normalize_text(col)
                if user_normalized == col_normalized and col_normalized:
                    return col, 'normalized'
        
        # 4. Partial match (user input is substring of column name)
        for col in available_columns:
            if user_lower in col.lower():
                return col, 'partial'
        
        # 5. Reverse partial match (column name is substring of user input)
        for col in available_columns:
            if col.lower() in user_lower:
                return col, 'partial'
        
        # 6. ENHANCED: Normalized partial matching
        if user_normalized:
            for col in available_columns:
                col_normalized = normalize_text(col)
                if col_normalized and (user_normalized in col_normalized or col_normalized in user_normalized):
                    return col, 'normalized_partial'
        
        return None, 'none'

    def process_query(self, 
                     query: str, 
                     session_id: str = None,
                     raw_data: Optional[Any] = None,
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Main entry point for processing user queries
        """
        print_to_log(f"\n🚀 Processing query: '{query}'")
        
        # Set global session context for logging
        try:
            from session_context import set_session_context, extract_session_from_session_id
            user_id, thread_id = extract_session_from_session_id(session_id)
            set_session_context(user_id, thread_id)
        except ImportError:
            pass
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{int(time.time())}"
        
        # Extract user_id and thread_id from session_id (format: user_id_thread_id)
        if '_' in session_id:
            parts = session_id.split('_')
            user_id = parts[0] if len(parts) >= 1 else session_id
            thread_id = '_'.join(parts[1:]) if len(parts) > 1 else session_id
        else:
            user_id = session_id
            thread_id = session_id
        
        # Initialize thread logger
        thread_logger = get_thread_logger(user_id, thread_id)
        thread_logger.log_query(query)
        
        # Start performance tracking
        start_time = time.time()
        
        # Ensure user directory exists
        user_dir = self._get_user_session_dir(session_id)
        print_to_log(f"📁 User session directory: {user_dir}")
        thread_logger.log_data_operation("user_directory_access", {"user_dir": user_dir})
        
        # Load conversation history
        conversation_history = self._load_conversation_history(session_id)
        if conversation_history:
            print_to_log(f"📚 Loaded {len(conversation_history)} previous conversations")
        
        # Load or create state
        state = state_manager.load_state(session_id)
        if state is None:
            state = PipelineState(
                session_id=session_id,
                chat_session=session_id,
                user_query=query
            )
        
        # Always load session state if available (for interactive sessions)
        print_to_log(f"🔧 DEBUG MAIN PIPELINE: About to load session state for {session_id}")
        previous_state = self._load_session_state(session_id)
        print_to_log(f"🔧 DEBUG MAIN PIPELINE: _load_session_state returned: {previous_state is not None}")
        if previous_state:
                print_to_log(f"📂 Loaded previous session state for {session_id}")
                print_to_log(f"🔧 DEBUG: Previous state keys: {list(previous_state.keys())}")
                # Restore relevant state information INCLUDING DataFrames
                if 'preprocessing_state' in previous_state:
                    state.preprocessing_state = previous_state['preprocessing_state']
                print_to_log(f"🔧 DEBUG: Restored preprocessing_state: {state.preprocessing_state}")
                if state.preprocessing_state:
                    print_to_log(f"🔧 DEBUG: Current phase after restore: {state.preprocessing_state.get('current_phase')}")
                    print_to_log(f"🔧 DEBUG: Missing results after restore: {state.preprocessing_state.get('missing_results') is not None}")
                if 'feature_selection_state' in previous_state:
                    state.feature_selection_state = previous_state['feature_selection_state']
                if 'model_building_state' in previous_state:
                    state.model_building_state = previous_state['model_building_state']
                # Restore interactive session if available
                if 'interactive_session' in previous_state:
                    state.interactive_session = previous_state['interactive_session']
                print_to_log(f"🔧 DEBUG: Restored interactive_session: {state.interactive_session}")
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
            print_to_log(f"🔧 DEBUG: No previous state found for {session_id}")
        
        # Update user query (always use the new query, not the saved one)
        state.user_query = query
        
        # ===== AUTOMATIC TARGET COLUMN DETECTION =====
        # If dataset has a 'target' column and no target is set, automatically use it
        if (state.raw_data is not None and 
            (not hasattr(state, 'target_column') or not state.target_column)):
            
            available_columns = list(state.raw_data.columns)
            
            # Check if dataset has a 'target' column
            if 'target' in available_columns:
                state.target_column = 'target'
                print_to_log(f"🎯 [Auto Detection] Found 'target' column in dataset - automatically set as target")
                
                # Save the automatically detected target column
                try:
                    state_manager.save_state(state)
                    print_to_log(f"💾 Automatically saved target column 'target' to session state")
                except Exception as e:
                    print_to_log(f"⚠️ Could not save automatically detected target column: {e}")
                
                # If there's an interactive session waiting for target, clear it
                if (hasattr(state, 'interactive_session') and 
                    state.interactive_session is not None and 
                    state.interactive_session.get('needs_target', False)):
                    
                    print_to_log(f"🔀 [Auto Detection] Clearing interactive session - target automatically detected")
                    state.interactive_session = None
        
        # ===== ALL ROUTING NOW HANDLED BY ORCHESTRATOR =====
        # Target selection, mode selection, skip patterns - all moved to orchestrator
        if False:  # DISABLED - orchestrator is single source of routing
            
            print_to_log(f"🎯 [Early Interception] Target column needed, checking query: '{query}'")
            
            # CRITICAL: Check if target column is already set - if so, skip target detection
            if hasattr(state, 'target_column') and state.target_column:
                print_to_log(f"🎯 [Early Interception] Target column already set: '{state.target_column}' - skipping detection")
                state.interactive_session['target_column'] = state.target_column
                state.interactive_session['needs_target'] = False
                
                # Check if user wants to skip using the existing sophisticated system
                from orchestrator import Orchestrator
                orchestrator = Orchestrator()
                skip_result = orchestrator._classify_skip_patterns(query)
                
                if skip_result and skip_result != "no_skip":
                    print_to_log(f"🎯 [Early Interception] Skip pattern detected: {skip_result} - bypassing mode selection")
                    print_to_log(f"🔀 [Early Interception] Clearing interactive session and routing to orchestrator")
                    state.interactive_session = None  # Clear interactive session
                else:
                    # User needs preprocessing - ask fast or slow mode
                    state.interactive_session['needs_mode_selection'] = True
                
                # CRITICAL FIX: Clear old last_response to prevent stale responses
                state.last_response = None
                
                # CRITICAL FIX: Save state after updating interactive_session
                try:
                    state_manager.save_state(state)
                    print_to_log(f"💾 Saved updated interactive_session with target '{state.target_column}'")
                except Exception as e:
                    print_to_log(f"⚠️ Could not save updated interactive_session: {e}")
                
                # CRITICAL: Don't continue to fuzzy matching - fall through to mode selection handler below
                print_to_log(f"🔀 [Early Interception] Falling through to mode selection handler")
            
            # Only try fuzzy matching if target is NOT already set
            elif not (hasattr(state, 'target_column') and state.target_column) and state.raw_data is not None:
                print_to_log(f"🎯 [Early Interception] No target column set - proceeding with detection")
                
                # Check if query looks like a column name with FUZZY MATCHING
                available_columns = list(state.raw_data.columns)
                
                # Use enhanced fuzzy matching function
                matched_column, match_type = self._find_target_column_enhanced(query.strip(), available_columns)
                
                if matched_column:
                    target_col = matched_column
                    state.target_column = target_col
                    
                    # CRITICAL FIX: Update interactive_session BEFORE saving state
                    state.interactive_session['target_column'] = target_col
                    state.interactive_session['needs_target'] = False
                    
                    # IMMEDIATE SAVE: Persist target column to session state for fallback
                    try:
                        state_manager.save_state(state)
                        print_to_log(f"💾 Immediately saved target column '{target_col}' to session state")
                    except Exception as e:
                        print_to_log(f"⚠️ Could not immediately save target column: {e}")
                    
                    # CRITICAL: Use existing skip pattern detection system instead of duplicate logic
                    from orchestrator import Orchestrator
                    orchestrator = Orchestrator()
                    
                    # Check if user wants to skip using the existing sophisticated system
                    skip_result = orchestrator._classify_skip_patterns(query)
                    
                    if skip_result and skip_result != "no_skip":
                        # User explicitly wants to skip - clear interactive session and let orchestrator handle
                        print_to_log(f"🎯 [Early Interception] Skip pattern detected: {skip_result} - bypassing mode selection")
                        print_to_log(f"🔀 [Early Interception] Clearing interactive session and routing to orchestrator")
                        state.interactive_session = None  # Clear interactive session
                        # Let the query continue to orchestrator for skip pattern routing
                    else:
                        # User needs preprocessing - ask fast or slow mode
                        state.interactive_session['needs_mode_selection'] = True
                    
                    # Log the fuzzy matching result
                    if match_type == 'exact':
                        print_to_log(f"✅ [Early Interception] Target column set: {target_col} (exact match)")
                    else:
                        print_to_log(f"✅ [Early Interception] Target column set: {target_col} (fuzzy match: {match_type})")
                        print_to_log(f"   🔍 User input: '{query.strip()}' → Matched: '{target_col}'")
                    
                    # CRITICAL: Show mode selection immediately after setting target (don't wait for next iteration)
                    print_to_log(f"🎯 [Early Interception] Sending mode selection message for target: {target_col}")
                    mode_choice_msg = f"""✅ Target column set: `{target_col}`

🚀 Choose Your ML Pipeline Mode



⚡ Fast Mode (Automated): 
• Complete ML pipeline without interaction
• AI handles all preprocessing decisions
• Get results in 2-3 minutes

🎛️ Slow Mode (Interactive): 
• Step-by-step guided process
• Review and approve each phase
• Full control over decisions

💬 Choose: Type `fast` or `slow`"""
                    
                    self.slack_manager.send_message(session_id, mode_choice_msg)
                    print_to_log(f"✅ [Early Interception] Mode selection message sent via Slack")
                    
                    # CRITICAL FIX: Clear old last_response so it doesn't override our new response
                    state.last_response = f" "
                    
                    # Save state and return
                    self._save_session_state(session_id, state)
                    print_to_log(f"🔧 [Early Interception] Returning with mode selection response")
                    return self._prepare_response(state, f" ")
                else:
                    # No match found - show available columns
                    available_cols_preview = ', '.join(available_columns[:5])
                    if len(available_columns) > 5:
                        available_cols_preview += f" ... and {len(available_columns) - 5} more"
                    
                    error_msg = f"""❌ Target column not found

🔍 Your input: `{query.strip()}`

📊 Available columns: {available_cols_preview}

💡 Try: Type the exact column name (case-sensitive)"""
                    
                    self.slack_manager.send_message(session_id, error_msg)
                    return self._prepare_response(state, f"Target column '{query.strip()}' not found. Please try again.")
        
        # ===== MODE SELECTION NOW HANDLED BY ORCHESTRATOR =====
        # All mode selection logic moved to orchestrator for single source of routing
        
        # Clear pending file uploads for new queries - they should only be relevant to current query
        if state.pending_file_uploads:
            print_to_log(f"🔍 UPLOAD DEBUG: Clearing {len(state.pending_file_uploads.get('files', []))} pending uploads from previous session")
            state.pending_file_uploads = None
        
        # Check if we have an active interactive session that needs to continue
        # BUT only if the query is a continuation command, not a new request
        print_to_log(f"🔍 DEBUG: Checking interactive session:")
        print_to_log(f"  Has interactive_session attr: {hasattr(state, 'interactive_session')}")
        print_to_log(f"  Interactive session is None: {state.interactive_session is None}")
        print_to_log(f"  Session active: {state.interactive_session.get('session_active', False) if state.interactive_session else 'N/A'}")
        print_to_log(f"  Interactive session details: {state.interactive_session}")
        
        if (hasattr(state, 'interactive_session') and 
            state.interactive_session is not None and 
            state.interactive_session.get('session_active', False)):
            
            print_to_log(f"🔄 Interactive session active: {state.interactive_session['agent_type']}")
            
            # Check if this is a continuation command vs a new request
            query_lower = query.lower().strip()
            print_to_log(f"🔍 DEBUG: Query to check: '{query_lower}'")
            
            # Pure continuation commands (context-independent)
            pure_continuation_commands = ['proceed', 'continue', 'next', 'back', 'summary', 'explain', 'help', 'yes', 'okay', 'cool', 'nice', 'go ahead', 'yeah', 'fine', 'good', 'sure', 'alright', 'agreed', 'approve', 'sounds good', 'move forward']
            
            # Check for explicit session management commands
            clear_session_commands = ['clear session', 'reset', 'start over', 'new session', 'exit session']
            is_clear_command = any(cmd in query_lower for cmd in clear_session_commands)
            
            if is_clear_command:
                print_to_log(f"🔄 Explicit session clear requested")
                state.interactive_session = None
                self.slack_manager.send_message(state.chat_session, "✅ Session cleared. You can now start a new workflow.")
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
                        "session_continuation": "Continue current workflow, proceed with current task, advance current session, move to next step in current process, skip current phase, bypass current step, proceed in current agent, continue current analysis, yes, yeah, ok, fine, good, sure, cool, alright, agreed, approve, proceed, continue, next, go ahead, yes please continue, yeah let's proceed, ok sounds good, fine go ahead, sure continue, cool proceed, alright next step, agreed move forward, approve this step, yeah continue, ok proceed, fine next, sure go ahead, cool let's continue, alright proceed, yes next phase, yeah move forward, ok advance, yep sounds good, right let's continue, correct proceed, absolutely continue, definitely proceed, of course continue, naturally proceed, certainly go ahead, obviously continue, clearly proceed, exactly continue, precisely proceed, indeed continue, absolutely go ahead, definitely next step, certainly continue now, obviously proceed ahead, clearly move forward, exactly next phase, naturally advance, certainly go forward, obviously next step, clearly continue process, exactly proceed ahead, precisely move forward, indeed advance now, naturally continue task, certainly proceed phase, obviously go ahead now, clearly next stage, exactly continue flow, precisely proceed step, indeed move ahead, naturally continue now, skip this step, skip current phase, bypass this, move to next, skip this analysis, pass on this, ignore this step, no thanks, change this strategy, modify approach, use different method, apply different strategy, override this, alter this, what does this mean, explain this step, help me understand, why this recommendation, how does this work, show me options, skip outliers detection, skip missing values, skip encoding, change imputation strategy, use median for age, modify outlier treatment, what is mean imputation, explain encoding methods, show current strategies, skip correlation analysis, skip SHAP step, skip VIF analysis, do analysis with IV values 0.02, run SHAP with 100 samples, change correlation threshold, modify selection criteria, what is information value, explain SHAP, show selected features, execute CSI analysis, perform VIF calculation, run IV analysis"
                    }
                    
                    from orchestrator import Orchestrator
                    temp_orchestrator = Orchestrator()
                    temp_orchestrator.intent_definitions = new_request_definitions
                    temp_orchestrator._initialize_intent_embeddings()
                    
                    if temp_orchestrator._intent_embeddings:
                        intent, confidence_info = temp_orchestrator._classify_with_semantic_similarity(query_lower)
                        print_to_log(f"[Session] New request semantic analysis: {intent} (confidence: {confidence_info['max_score']:.3f})")
                        
                        # Use moderate threshold for new request detection
                        if confidence_info.get("max_score", 0) > 0.25:
                            is_new = (intent == "new_ml_request")
                            print_to_log(f"[Session] Semantic new request decision: {'NEW REQUEST' if is_new else 'CONTINUATION'}")
                            return is_new
                    
                    print_to_log(f"[Session] New request semantic analysis failed, using keyword fallback")
                    return False
                    
                except Exception as e:
                    print_to_log(f"[Session] Semantic new request error: {e}, using keyword fallback")
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
                pure_continuation_commands = ['proceed', 'continue', 'next', 'back', 'summary', 'explain', 'help', 'yes', 'okay', 'cool', 'nice', 'go ahead', 'yeah', 'fine', 'good', 'sure', 'alright', 'agreed', 'approve', 'sounds good', 'move forward']
                if any(cmd in query_lower for cmd in pure_continuation_commands):
                    return True
                
                # Phase 2: Yes/No for confirmation contexts (explicit)
                if current_phase in ['confirmation', 'approval_needed']:
                    return query_lower.strip() in ['yes', 'no', 'y', 'n']
                
                # Phase 3: Semantic continuation classification
                try:
                    continuation_definitions = {
                    'preprocessing': {
                        'continue_preprocessing': "Continue preprocessing workflow, advance preprocessing stage, data cleaning continuation, preprocessing workflow advancement, outlier detection phase, missing values handling, encoding step, transformation phase, target column specification, data preparation, proceed with preprocessing, continue data cleaning, advance preprocessing, next preprocessing step, move forward in preprocessing, skip current step, proceed to next phase, move forward in data cleaning, advance to next preprocessing stage, bypass current analysis, skip outliers detection, skip missing values handling, skip encoding step, skip transformations, target column specification, column selection, data cleaning continuation, datetime column specification, datetime transaction_date, datetime order_date, datetime created_at, datetime timestamp, datetime date_column, oot month 2023M08, oot month 2024M01, oot month 2023M12, oot month specification, out of time month specification, date format specification, timestamp format specification, datetime format specification, date column format, time column format, date specification, time specification, datetime info, date info, time info, column datetime, column date, column time, date column name, time column name, datetime column name, set datetime column, specify datetime column, provide datetime column, datetime column is, date column is, time column is, target column specification, target is, target column is, column target, target column name, target variable, target variable is, dependent variable, dependent variable is, outcome variable, outcome variable is, response variable, response variable is, label column, label column is, label variable, label variable is",
                        'new_request': "Start feature selection, begin model building, train new model, create model, build classifier, analyze features, select variables, stop preprocessing, end cleaning, switch to modeling, move to next agent, go to feature selection, move to model training, exit preprocessing"
                    },
                    'feature_selection': {
                        'continue_feature_selection': "Continue feature selection workflow, advance feature analysis, feature selection continuation, variable selection advancement, correlation analysis phase, SHAP analysis step, information value calculation, VIF analysis phase, CSI analysis step, feature ranking process, proceed with feature selection, continue feature analysis, advance selection, next feature step, move forward in feature selection, do analysis with IV values 0.02, run SHAP with 100 samples, execute correlation analysis, perform VIF calculation, run CSI analysis, calculate information value 0.1, analyze with IV threshold 0.05, run SHAP analysis now, perform correlation with threshold 0.8, execute VIF analysis, do feature importance analysis, IV with 0.02, IV with 0.01, IV with 0.03, IV with 0.04, IV with 0.05, what are top 10 important features according to decision tree feature importance, top 10 important features according to decision tree, decision tree feature importance, train decision tree and show importance, decision tree classifier importance, most important features according to tree, top important features decision tree, feature importance from decision tree, tree importance ranking, decision tree based importance, what are top 10 features by rfe, what are top 10 features by lasso, top features by rfe, top features by lasso, features by rfe, features by lasso, rfe selection, lasso selection, recursive feature elimination, rfe ranking, lasso ranking, rfe importance, lasso importance, skip correlation analysis, skip SHAP step, skip current analysis, bypass feature ranking, do VIF analysis, VIF analysis with threshold, VIF analysis with 5, VIF analysis with 4, VIF analysis with 10, VIF with 5, VIF with 4, VIF with 10, VIF with 3, VIF with 2, perform VIF analysis, execute VIF analysis, run VIF analysis, calculate VIF, VIF calculation, VIF threshold, VIF filtering, VIF removal, multicollinearity analysis, multicollinearity check, variance inflation factor, variance inflation factor analysis, VIF score calculation, VIF based filtering, remove multicollinear features, filter by VIF, VIF based selection, correlation analysis 0.8, correlation analysis 0.9, correlation analysis 0.7, correlation with 0.8, correlation with 0.9, correlation with 0.7, correlation threshold 0.8, correlation filtering, correlation matrix, correlation analysis, IV analysis 0.05, IV analysis 0.02, IV analysis 0.1, IV analysis 0.03, information value analysis, information value calculation, IV calculation, IV filtering, IV threshold, IV based selection, CSI analysis 0.2, CSI analysis 0.3, CSI analysis 0.5, CSI with 0.2, CSI with 0.3, CSI with 0.5, CSI threshold, characteristic stability index, stability analysis, feature stability, SHAP analysis, SHAP importance, SHAP values, SHAP ranking, feature importance SHAP, SHAP based selection, PCA analysis, principal component analysis, dimensionality reduction, PCA transformation, PCA with components, LASSO selection, LASSO regularization, LASSO feature selection, L1 regularization, LASSO with alpha, datetime column specification for CSI, datetime transaction_date, datetime order_date, datetime created_at, datetime timestamp, datetime date_column, oot month 2023M08, oot month 2024M01, oot month 2023M12, oot month specification, out of time month specification for CSI analysis, date format specification for CSI, timestamp format for CSI, datetime format for CSI, date column format for CSI, time column format for CSI, date specification for CSI, time specification for CSI, datetime info for CSI, date info for CSI, time info for CSI, column datetime for CSI, column date for CSI, column time for CSI, date column name for CSI, time column name for CSI, datetime column name for CSI, set datetime column, specify datetime column, provide datetime column, datetime column is, date column is, time column is, revert to cleaned dataset, restore original features, reset feature selection, undo feature analysis, go back to cleaned data, revert feature changes, restore initial features, reset to clean state, undo feature filtering, back to cleaned dataset, restore feature set, reset analysis chain, undo analysis steps, revert all analysis, restore previous features, reset feature pipeline, undo feature removal, go back to start, revert selection, restore backup features, reset everything, undo modifications, revert to original, restore clean state, cool, proceed, continue, next, go ahead, yes, yeah, ok, fine, good, sure, alright, agreed, approve, sounds good, let's proceed, move forward",
                        'new_request': "Start model building, train model, build classifier, create predictor, stop feature selection, end variable selection, switch to modeling, begin training, start preprocessing, go to preprocessing, move to model training, exit feature selection"
                    }
                }
                
                    if agent_type in continuation_definitions:
                        # Use semantic similarity to classify continuation vs new request
                        from orchestrator import Orchestrator
                        
                        # Create temporary orchestrator for context-aware semantic classification
                        temp_orchestrator = Orchestrator()
                        
                        # Create temporary intent definitions for this context
                        context_intents = continuation_definitions[agent_type]
                        temp_orchestrator.intent_definitions = context_intents
                        temp_orchestrator._initialize_intent_embeddings()
                        
                        if temp_orchestrator._intent_embeddings:
                            intent, confidence_info = temp_orchestrator._classify_with_semantic_similarity(query_lower)
                            print_to_log(f"[Session] Semantic continuation analysis: {intent} (confidence: {confidence_info['max_score']:.3f})")
                            
                            # Use lower threshold for continuation detection (more permissive)
                            if confidence_info.get("max_score", 0) > 0.2:
                                is_continuation = (intent.startswith('continue_') or intent.endswith('_continuation'))
                                print_to_log(f"[Session] Semantic decision: {'CONTINUATION' if is_continuation else 'NEW REQUEST'}")
                                return is_continuation
                        
                        print_to_log(f"[Session] Semantic analysis failed, falling back to keyword matching")
                
                except Exception as e:
                    print_to_log(f"[Session] Semantic continuation error: {e}, using keyword fallback")
                
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
                        'proceed', 'continue', 'finish', 'complete', 'done', 'finalize',
                        # IV Analysis patterns
                        'run iv', 'do iv', 'iv analysis', 'implement iv', 'apply iv', 'iv filtering', 'iv value', 'iv cutoff',
                        'information value', 'iv threshold', 'iv greater than', 'iv less than', 'filter by iv',
                        'with iv', 'iv >', 'iv <', 'iv >=', 'iv <=', 'filter all values', 'filter values',
                        'iv with', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09',
                        # CSI Analysis patterns  
                        'run csi', 'do csi', 'csi analysis', 'implement csi', 'apply csi', 'csi filtering',
                        'characteristic stability', 'stability index', 'csi threshold', 'csi cutoff',
                        # Correlation Analysis patterns
                        'run correlation', 'correlation analysis', 'implement correlation', 'apply correlation', 'correlation filtering',
                        'correlation with target', 'target correlation', 'correlate with target', 'correlation cutoff',
                        'highly correlated', 'remove correlated', 'correlation threshold', 'corr analysis',
                        'show correlation', 'correlation between', 'correlation matrix', 'correlation scores',
                        'feature correlation', 'correlation values', 'correlation results',
                        # SHAP Analysis patterns
                        'run shap', 'do shap', 'shap analysis', 'implement shap', 'apply shap', 'shap filtering',
                        'shap values', 'shap value', 'shap importance', 'feature importance', 'shap cutoff', 'shap threshold',
                        'top shap features', 'top shap', 'shap ranking', 'importance ranking', 'feature ranking',
                        # VIF Analysis patterns
                        'run vif', 'do vif', 'vif analysis', 'implement vif', 'apply vif', 'vif filtering',
                        'variance inflation', 'multicollinearity', 'vif threshold', 'vif cutoff', 'vif greater than',
                        'vif analysis with', 'vif with', 'perform vif', 'execute vif', 'calculate vif',
                        'vif analysis with 5', 'vif analysis with 4', 'vif analysis with 10', 'vif analysis with 3',
                        'vif with 5', 'vif with 4', 'vif with 10', 'vif with 3', 'vif with 2',
                        # Other ML techniques
                        'run lasso', 'lasso selection', 'implement lasso', 'apply lasso', 'lasso regularization',
                        'run pca', 'pca analysis', 'principal component', 'dimensionality reduction',
                        'random forest importance', 'tree importance', 'model importance',
                        'decision tree', 'decision tree importance', 'decision tree feature importance',
                        'train decision tree', 'decision tree classifier', 'tree classifier',
                        'important features according to', 'top important features', 'feature importance',
                        'important features', 'most important features', 'top 10 important', 'top 5 important',
                        'top 20 important', 'according to decision tree', 'according to tree',
                        'rfe', 'recursive feature elimination', 'features by rfe', 'top features by rfe',
                        'what are top features by rfe', 'rfe selection', 'rfe ranking', 'rfe importance',
                        'features by lasso', 'top features by lasso', 'what are top features by lasso',
                        'lasso features', 'lasso ranking', 'lasso importance', 'lasso coefficients',
                        # General feature selection patterns
                        'filter features', 'select features', 'analyze features', 'filtering with', 'cutoff', 'threshold',
                        'remove features', 'keep features', 'feature subset', 'top features', 'best features',
                        'rank features', 'score features', 'evaluate features', 'assess features',
                        'filter all', 'filter with', 'values with', 'greater than', 'less than', 'equal to',
                        '>', '<', '>=', '<=', '==', '!=', '0.01', '0.05', '0.1', '0.2', '0.3', '0.5',
                        # Action words
                        'implement', 'apply', 'execute', 'perform', 'calculate', 'compute', 'run analysis',
                        'do analysis', 'start analysis', 'begin analysis', 'conduct analysis',
                        # Query patterns
                        'what is', 'explain', 'how does', 'show me', 'tell me about', 'describe',
                        'what are', 'what are the', 'which are', 'which are the', 'give me', 'list',
                        'top 5', 'top 10', 'top 20', 'top 30', 'top 50', 'best 5', 'best 10', 'best 20',
                        'highest', 'lowest', 'most important', 'least important', 'value features',
                        'how many features', 'current state', 'summary', 'show progress', 
                        'revert', 'reset', 'restore', 'undo', 'go back', 'back to original', 'back to cleaned',
                        'revert to cleaned', 'restore original', 'reset features', 'undo analysis', 'restore features',
                        'reset analysis', 'undo changes', 'go back to start', 'restore backup', 'reset everything',
                        'undo modifications', 'revert changes', 'restore clean state', 'reset to clean',
                        'datetime', 'date column', 'time column', 'timestamp', 'oot month', 'out of time',
                        'date format', 'time format', 'datetime format', 'datetime column', 'date_column',
                        'transaction_date', 'order_date', 'created_at', 'timestamp_column', 'date_time',
                        'datetime specification', 'date specification', 'time specification', 'column datetime',
                        'set datetime', 'specify datetime', 'provide datetime', 'datetime is', 'date is', 'time is'
                    ]
                    is_fs_continuation = any(cmd in query_lower for cmd in fs_continuations)
                    print_to_log(f"🔧 DEBUG FS CONTINUATION: Query='{query_lower}', Is continuation={is_fs_continuation}")
                    if is_fs_continuation:
                        matched_keywords = [cmd for cmd in fs_continuations if cmd in query_lower]
                        print_to_log(f"🔧 DEBUG FS CONTINUATION: Matched keywords={matched_keywords}")
                    return is_fs_continuation
                
                return False
            
            # Phase 1: Semantic analysis for session management
            semantic_new_request = is_semantic_new_request(query_lower, state.interactive_session)
            semantic_continuation = is_context_continuation(query_lower, state.interactive_session)
            
            # Phase 2: Keyword fallback analysis
            keyword_new_request = any(pattern in query_lower for pattern in new_request_patterns)
            
            # Phase 3: Combine semantic and keyword results (semantic takes priority)
            is_new_request = semantic_new_request or (not semantic_continuation and keyword_new_request)
            is_continuation = semantic_continuation
            
            print_to_log(f"[Session] Analysis results:")
            print_to_log(f"  Semantic new request: {semantic_new_request}")
            print_to_log(f"  Semantic continuation: {semantic_continuation}")
            print_to_log(f"  Keyword new request: {keyword_new_request}")
            print_to_log(f"  Final decision - New request: {is_new_request}, Continuation: {is_continuation}")
            print_to_log(f"🔧 DEBUG SESSION: Has interactive_session: {state.interactive_session is not None}")
            if state.interactive_session:
                print_to_log(f"🔧 DEBUG SESSION: Interactive session: {state.interactive_session}")
            print_to_log(f"🔧 DEBUG SESSION: Query: '{query}'")
            
            # Handle conflicts: if both detected, prioritize based on context
            if is_new_request and is_continuation:
                # Special handling for feature selection queries that should be continuations
                fs_query_patterns = [
                    'what are top', 'top features by', 'features by', 'show me features',
                    'top 10 features', 'top 20 features', 'best features', 'important features',
                    'what are the top', 'which features', 'list features', 'display features'
                ]
                
                # If this is a feature selection query asking about results, treat as continuation
                if (state.interactive_session and 
                    state.interactive_session.get('agent_type') == 'feature_selection' and
                    any(pattern in query_lower for pattern in fs_query_patterns)):
                    print_to_log(f"🔄 Feature selection query detected - treating as continuation despite new request words")
                    is_new_request = False
                # If query starts with new request pattern, treat as new request
                elif any(pattern in query_lower[:20] for pattern in new_request_patterns):
                    print_to_log(f"🆕 New ML request detected (despite continuation words) - clearing session")
                    state.interactive_session = None
                else:
                    print_to_log(f"🔄 Treating as continuation command despite new request words")
                    is_new_request = False
            
            # Now handle the routing based on the final decision
            if is_new_request:
                print_to_log(f"🆕 New ML request detected - clearing interactive session and routing through orchestrator")
                state.interactive_session = None
            elif is_continuation or is_target_specification:
                print_to_log(f"🔄 Continuing interactive session: {state.interactive_session['agent_type']}")
                print_to_log(f"�� DEBUG CONTINUATION: Query='{query}', Agent={state.interactive_session['agent_type']}")
                print_to_log(f"🔧 DEBUG CONTINUATION: Session ID={state.chat_session}")
                print_to_log(f"🔧 DEBUG CONTINUATION: Interactive session details={state.interactive_session}")
                
                # Route to the appropriate agent to continue the interactive session
                agent_type = state.interactive_session['agent_type']
                
                # CRITICAL: Check if we're in mode selection phase - route to orchestrator instead
                if (agent_type == "preprocessing" and 
                    state.interactive_session.get('phase') == 'mode_selection'):
                    print_to_log("🚀 DEBUG: Mode selection detected - routing to orchestrator instead of preprocessing")
                    # Clear the interactive session temporarily so orchestrator can handle mode selection
                    # The orchestrator will restore it or create a new one as needed
                    return self._route_to_agent(state)
                
                elif agent_type == "preprocessing":
                    print_to_log("🔧 DEBUG: Routing to preprocessing interactive handler")
                    # Handle preprocessing commands directly
                    return self._handle_preprocessing_interaction(state, query)
                elif agent_type == "feature_selection":
                    print_to_log("🔧 DEBUG: Routing to feature selection interactive handler")
                    
                    # ✅ CRITICAL FIX: Ensure slack_manager is passed to state before calling handler
                    print_to_log(f"🔧 [Interactive FS] Using slack_manager id: {id(self.slack_manager)}")
                    print_to_log(f"🔧 [Interactive FS] Slack manager has {len(self.slack_manager.session_channels)} channels")
                    
                    state._slack_manager = self.slack_manager
                    state.slack_session_info = {
                        'channels': dict(self.slack_manager.session_channels),
                        'threads': dict(self.slack_manager.session_threads)
                    }
                    print_to_log(f"💾 [Interactive FS] Passed slack session info: {len(state.slack_session_info['channels'])} channels")
                    
                    from agents_wrapper import feature_selection_agent
                    # Use interactive command handler for feature selection
                    result = feature_selection_agent.handle_interactive_command(state, query)
                    print_to_log(f"🔧 DEBUG: Feature selection handler returned: {type(result)}")
                    return self._prepare_response(result)
                else:
                    print_to_log(f"❌ DEBUG: Unknown agent type for continuation: {agent_type}")
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
            
            # Log performance and response
            processing_time = time.time() - start_time
            thread_logger.log_performance("query_processing", processing_time, {
                "session_id": session_id,
                "query_length": len(query),
                "success": response.get("success", True)
            })
            thread_logger.log_response(response['response'], success=response.get("success", True))
            
            # Log the response for debugging/monitoring
            print_to_log(f"📤 Response: {response['response']}")
            print_to_log(f"✅ Query processing completed for session {session_id}")
            return response
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            print_to_log(f"❌ {error_msg}")
            
            # Log error with thread logger
            processing_time = time.time() - start_time
            thread_logger.error(f"Pipeline execution failed", e, {
                "session_id": session_id,
                "query": query,
                "processing_time": processing_time
            })
            thread_logger.log_performance("query_processing_failed", processing_time, {
                "session_id": session_id,
                "error": error_msg
            })
            
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
            
            # Log error response
            thread_logger.log_response(error_response['response'], success=False)
            
            # Log the error response for debugging/monitoring
            print_to_log(f"📤 Error Response: {error_response['response']}")
            
            # Save session state and conversation history for errors (single point)
            self._save_session_state(session_id, state)
            self._save_conversation_history(session_id, state.user_query, error_response["response"], state)
            
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
        self._save_conversation_history(state.session_id, state.user_query, response["response"], state)
        
        # Send any pending Slack message AFTER CSV files are saved
        if state.pending_slack_message and self.slack_manager and state.chat_session:
            print_to_log("📱 Sending pending Slack message after successful CSV save")
            self.slack_manager.send_message(state.chat_session, state.pending_slack_message)
            state.pending_slack_message = None  # Clear after sending
        
        return response
    
    def _handle_preprocessing_interaction(self, state: PipelineState, query: str):
        """Handle interactive preprocessing commands"""
        try:
            # Load previous session state to ensure we have the latest state
            previous_state = self._load_session_state(state.chat_session)
            if previous_state:
                print_to_log(f"📂 Loaded previous session state in preprocessing handler")
                print_to_log(f"🔧 DEBUG: Previous state keys: {list(previous_state.keys())}")
                # Restore preprocessing state if available
                if 'preprocessing_state' in previous_state:
                    state.preprocessing_state = previous_state['preprocessing_state']
                    print_to_log(f"🔧 DEBUG: Restored preprocessing_state in handler: {state.preprocessing_state}")
                    print_to_log(f"🔧 DEBUG: Current phase after restore: {state.preprocessing_state.get('current_phase')}")
                    print_to_log(f"🔧 DEBUG: Missing results after restore: {state.preprocessing_state.get('missing_results') is not None}")
                # Restore interactive session if available
                if 'interactive_session' in previous_state:
                    state.interactive_session = previous_state['interactive_session']
                    print_to_log(f"🔧 DEBUG: Restored interactive_session in handler: {state.interactive_session}")
                
                # ✅ CRITICAL FIX: Restore DataFrame data to prevent session loading/saving problems
                if 'cleaned_data' in previous_state and previous_state['cleaned_data'] is not None:
                    state.cleaned_data = previous_state['cleaned_data']
                    print_to_log(f"🔧 DEBUG: Restored cleaned_data in handler: {state.cleaned_data.shape if state.cleaned_data is not None else 'None'}")
                if 'raw_data' in previous_state and previous_state['raw_data'] is not None:
                    state.raw_data = previous_state['raw_data']
                    print_to_log(f"🔧 DEBUG: Restored raw_data in handler: {state.raw_data.shape if state.raw_data is not None else 'None'}")
            else:
                print_to_log(f"🔧 DEBUG: No previous state found in preprocessing handler")
            
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
                    
                    # After target is set, show mode selection instead of preprocessing intro
                    response_msg = f"""✅ Target column set: `{target_col}`

🚀 Choose Your ML Pipeline Mode


⚡ Fast Mode (Automated): 
• Complete ML pipeline without interaction
• AI handles all preprocessing decisions
• Get results in 2-3 minutes

🎛️ Slow Mode (Interactive): 
• Step-by-step guided process
• Review and approve each phase
• Full control over decisions

💬 Choose: Type `fast` or `slow`"""
                    
                    self.slack_manager.send_message(state.chat_session, response_msg)
                    return self._prepare_response(state, f" ")
                else:
                    available_cols = list(state.raw_data.columns)
                    error_msg = f"""❌ Column '{target_col}' not found.

Available columns: {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}

Please specify a valid column name."""
                    
                    self.slack_manager.send_message(state.chat_session, error_msg)
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
                    print_to_log("🚀 Starting interactive preprocessing workflow")
                    # Pass the pipeline's slack_manager to the state
                    state._slack_manager = self.slack_manager
                    processed_state = preprocessing_agent.handle_interactive_command(state, 'proceed')
                    
                    # Debug: Check if preprocessing state was set
                    print_to_log(f"🔧 DEBUG: After proceed - preprocessing_state: {processed_state.preprocessing_state}")
                    print_to_log(f"🔧 DEBUG: After proceed - has outlier_results: {processed_state.preprocessing_state.get('outlier_results') is not None if processed_state.preprocessing_state else False}")
                    
                    # Save the updated state to session state file
                    self._save_session_state(processed_state.session_id, processed_state)
                    
                    # The preprocessing agent should handle the interactive flow
                    # and return the state with the interactive session set up
                    return self._prepare_response(processed_state, " ")
                else:
                    # Phase-aware: treat 'proceed' as 'continue' when already inside a phase
                    print_to_log("🔄 Proceed received in-phase → treating as 'continue'")
                    state._slack_manager = self.slack_manager
                    processed_state = preprocessing_agent.handle_interactive_command(state, 'continue')
                    self._save_session_state(processed_state.session_id, processed_state)
                    return self._prepare_response(processed_state, "Proceed mapped to continue in current phase.")
            

            # Check if we're in an active preprocessing phase and route to preprocessing agent
            current_phase = None
            if state.preprocessing_state and 'current_phase' in state.preprocessing_state:
                current_phase = state.preprocessing_state.get('current_phase')
            elif state.interactive_session and 'current_phase' in state.interactive_session:
                current_phase = state.interactive_session.get('current_phase')
            print_to_log(f"🔧 DEBUG: Checking 4-Level BGE routing - current_phase: '{current_phase}'")
            print_to_log(f"🔧 DEBUG: preprocessing_state current_phase: {state.preprocessing_state.get('current_phase') if state.preprocessing_state else 'N/A'}")
            print_to_log(f"🔧 DEBUG: interactive_session current_phase: {state.interactive_session.get('current_phase') if state.interactive_session else 'N/A'}")
            
            if current_phase in ['overview', 'outliers', 'missing_values', 'encoding', 'transformations', 'completion']:
                # Route to preprocessing agent for phase-specific handling
                from agents_wrapper import preprocessing_agent
                
                print_to_log(f"🔄 [4-Level Flow] Routing to preprocessing agent for phase: {current_phase}")
                
                # 4-Level Classification Flow:
                print_to_log(f"🎯 [4-Level Flow] Starting classification cascade for: '{query}'")
                print_to_log(f"   📍 Level 1: SKIP (already in preprocessing session)")
                print_to_log(f"   📍 Level 2: session_continuation (already determined - we're in preprocessing phase)")
                print_to_log(f"   📍 Level 3: continue_preprocessing (already determined - we're staying in preprocessing)")
                print_to_log(f"   📍 Level 4: Classifying specific preprocessing action...")
                
                action_intent = self._classify_preprocessing_action(query)
                print_to_log(f"🎯 [4-Level Flow] Final Level 4 Action Intent: '{action_intent}'")
                
                # Map Level 4 action to underlying commands handled by wrapper
                mapped = query
                if action_intent == 'proceed':
                    mapped = 'PROCEED: continue'  # Clear intent signal
                    print_to_log(f"🔄 [4-Level Flow] Mapping '{action_intent}' → 'PROCEED: continue' command (BGE intent)")
                elif action_intent == 'summary':
                    mapped = 'SUMMARY: summary'  # Clear intent signal
                    print_to_log(f"🔄 [4-Level Flow] Mapping '{action_intent}' → 'SUMMARY: summary' command (BGE intent)")
                elif action_intent == 'override':
                    mapped = f'OVERRIDE: {query}'  # Clear intent signal + original query
                    print_to_log(f"🔄 [4-Level Flow] Mapping '{action_intent}' → 'OVERRIDE: {query}' command (BGE intent)")
                elif action_intent == 'skip':
                    # Check if it's a specific phase skip (skip outliers, skip encoding, etc.)
                    specific_skips = ['skip outliers', 'skip missing', 'skip encoding', 'skip transformations']
                    if any(skip_cmd in query.lower() for skip_cmd in specific_skips):
                        mapped = f'SKIP: {query}'  # preserve specific skip commands with intent signal
                        print_to_log(f"🔄 [4-Level Flow] Mapping '{action_intent}' → 'SKIP: {query}' command (specific phase skip with BGE intent)")
                    else:
                        mapped = 'SKIP: skip'  # generic skip with intent signal
                        print_to_log(f"🔄 [4-Level Flow] Mapping '{action_intent}' → 'SKIP: skip' command (generic skip with BGE intent)")
                elif action_intent == 'query':
                    mapped = f"QUERY: {query}"  # Clear intent signal + original query
                    print_to_log(f"🔄 [4-Level Flow] Mapping '{action_intent}' → 'QUERY: {query}' command (query with intent)")
                else:
                    mapped = query  # fallback
                    print_to_log(f"🔄 [4-Level Flow] Mapping '{action_intent}' → '{query}' command (fallback)")
                
                print_to_log(f"✅ [4-Level Flow] Final mapped command: '{mapped}' → Sending to preprocessing agent")
                
                # Pass the pipeline's slack_manager to the state
                state._slack_manager = self.slack_manager
                processed_state = preprocessing_agent.handle_interactive_command(state, mapped)
                
                # Save the updated state to session state file
                self._save_session_state(processed_state.session_id, processed_state)
                
                return self._prepare_response(processed_state, f" ")
            

            else:
                # Default help message
                help_msg = """💬 Available Commands:
• `proceed` - Start preprocessing workflow
• `summary` - Show current status
• `explain outliers` - Learn about outlier handling
• `target column_name` - Set target column (if not set)

What would you like to do?"""
                
                self.slack_manager.send_message(state.chat_session, help_msg)
                return self._prepare_response(state, "Help message sent.")
                
        except Exception as e:
            print_to_log(f"❌ Error in preprocessing interaction: {e}")
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
            return " "
    
    def load_data(self, data: Any, session_id: str):
        """Load data into a session"""
        state = state_manager.load_state(session_id)
        if state is None:
            state = PipelineState(session_id=session_id, chat_session=session_id)
        
        state.raw_data = data
        
        # Target column will be set interactively during preprocessing
        target_was_auto_detected = False
        if state.target_column is None and hasattr(data, 'columns'):
            print_to_log(f"🎯 Target column not set - will prompt user during preprocessing")
            print_to_log(f"📊 Available columns: {list(data.columns)}")
            # Note: Target selection now happens interactively in agents_wrapper.py
        
        state_manager.save_state(state)
        
        print_to_log(f"📊 Data loaded for session {session_id}: {data.shape if hasattr(data, 'shape') else 'Unknown shape'}")
        if state.target_column:
            print_to_log(f"🎯 Target column: {state.target_column}")
        
        # If target was auto-detected, automatically show preprocessing menu
        # BUT only if there's no existing interactive session already loaded
        if target_was_auto_detected and state.target_column:
            # Check if we already have an active interactive session (from loaded state)
            if (hasattr(state, 'interactive_session') and 
                state.interactive_session is not None and 
                state.interactive_session.get('session_active', False)):
                print_to_log("🎯 Target auto-detected but interactive session already active - skipping auto menu")
                print_to_log(f"🎯 Existing session: {state.interactive_session}")
                return
                
            # Only auto-show when the current intent is preprocessing/full_pipeline
            current_intent = getattr(state, 'current_intent', None) or getattr(state, 'user_intent', None)
            if not current_intent:
                # Try to infer from last routing decision if stored
                current_intent = getattr(state, 'last_route', None)
            if current_intent not in ['preprocessing', 'full_pipeline']:
                print_to_log("🎯 Target auto-detected but intent is not preprocessing/full_pipeline; skipping auto menu")
                return
            print_to_log("🎯 Target auto-detected - automatically showing preprocessing menu")
            # Set up interactive session
            state.interactive_session = {
                "agent_type": "preprocessing",
                "session_active": True,
                "session_id": state.chat_session,
                "phase": "waiting_input",
                "target_column": state.target_column,
                "current_phase": "overview"
            }
            
            # Don't send preprocessing intro here - it will be shown after mode selection
            print_to_log("⏭️ Skipping preprocessing intro for auto-detected target - will show after mode selection")
            
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

    def _classify_preprocessing_action(self, query: str) -> str:
        """
        Level 4: Classify specific preprocessing actions using BGE with keyword fallbacks.
        
        Args:
            query: User's input query
            
        Returns:
            Classified action: 'proceed', 'skip', 'override', 'query', 'summary'
        """
        print_to_log(f"🔍 [Level 4] Processing query: '{query}'")
        
        try:
            # BGE-based classification using orchestrator's embeddings (same as Levels 1-3)
            from orchestrator import orchestrator
            if hasattr(orchestrator, '_intent_embeddings') and orchestrator._intent_embeddings is not None:
                print_to_log(f"🧠 [Level 4] BGE embeddings available via orchestrator, attempting semantic classification...")
                
                action_definitions = {
                    "proceed_action": "proceed with current phase, continue current step, apply current strategy, move forward with current plan, advance current phase, execute current strategy, cool, yes, ok, fine, good, sure, yeah, alright, sounds good, let's go, proceed now, continue current, apply this, do this, execute this, go ahead, go ahead with this, go ahead in preprocessing, go ahead with current, go ahead and proceed, go ahead with analysis, move ahead, carry on, keep going, continue ahead, go forward, advance ahead, proceed ahead, go on, go through, go with this, go with current, let's proceed, let's continue, let's go ahead, let's move forward, start processing, begin processing, start analysis, begin analysis",
                    "skip_action": "skip current phase, skip this step, bypass current analysis, move to next phase, skip outliers detection, skip missing values handling, skip encoding step, skip transformations, pass on this, ignore this step, skip to next, move on, bypass this, no thanks to this step",
                    "override_action": "change strategy, modify approach, use different method, apply different strategy, override current, alter this approach, use median for age, apply mean imputation, change to winsorize, modify outlier treatment, use different encoding, apply one-hot, change transformation, use standard scaling, apply robust scaling",
                    "datetime_action": "datetime column specification, datetime transaction_date, datetime order_date, datetime created_at, datetime timestamp, datetime date_column, oot month 2023M08, oot month 2024M01, oot month 2023M12, oot month specification, out of time month specification, date format specification, timestamp format specification, datetime format specification, date column format, time column format, date specification, time specification, datetime info, date info, time info, column datetime, column date, column time, date column name, time column name, datetime column name, set datetime column, specify datetime column, provide datetime column, datetime column is, date column is, time column is, target column specification, target is, target column is, column target, target column name, target variable, target variable is, dependent variable, dependent variable is, outcome variable, outcome variable is, response variable, response variable is, label column, label column is, label variable, label variable is",
                    "query_action": "what is this strategy, explain current approach, how does this work, why this recommendation, what happens to columns, how does imputation work, explain outlier detection, what is encoding, how does transformation work, what does this mean, help me understand, why median imputation, why are you applying, why applying, why use this, why this strategy, why this treatment, why this method, why winsorization, why winsorize, why are you using, why do you recommend, why is this recommended, explain why, tell me why, why this choice, why this approach, what's the reason, what's the reasoning, why specifically, why particularly, why for this column, why this column, explain the reason, explain reasoning, why did you choose, why was this chosen, how come this strategy, how come you chose, justify this choice, justify this strategy, reasoning behind this, reason for this, rationale behind, rationale for this",
                    "summary_action": "show current strategies, display current plan, what's planned, show me current approach, current strategy summary, what are we doing, show strategies for all columns, current preprocessing plan"
                }
                
                # Use orchestrator's embedding system (same as main pipeline)
                query_embedding = orchestrator._get_embedding(query)
                if query_embedding is not None:
                    print_to_log(f"✅ [Level 4] Query embedding generated successfully via orchestrator")
                    similarities = {}
                    for intent_name, definition in action_definitions.items():
                        intent_embedding = orchestrator._get_embedding(definition)
                        if intent_embedding is not None:
                            # Use cosine similarity directly (same as orchestrator does)
                            from sklearn.metrics.pairwise import cosine_similarity
                            similarity = cosine_similarity(
                                query_embedding.reshape(1, -1),
                                intent_embedding.reshape(1, -1)
                            )[0][0]
                            similarities[intent_name] = float(similarity)
                    
                    if similarities:
                        # Show all similarity scores for debugging
                        print_to_log(f"🔍 [Level 4] BGE similarity scores:")
                        for intent, score in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
                            action_name = intent.replace('_action', '')
                            print_to_log(f"   {action_name}: {score:.3f}")
                        
                        best_intent = max(similarities.items(), key=lambda x: x[1])
                        action_intent = best_intent[0].replace('_action', '')  # Remove _action suffix
                        print_to_log(f"🎯 [Level 4] BGE classified '{query}' as '{action_intent}' (confidence: {best_intent[1]:.3f})")
                        
                        if best_intent[1] > 0.3:  # Confidence threshold
                            print_to_log(f"✅ [Level 4] BGE confidence above threshold (0.3), returning: '{action_intent}'")
                            return action_intent
                        else:
                            print_to_log(f"⚠️ [Level 4] BGE confidence below threshold ({best_intent[1]:.3f} < 0.3), falling back to keywords")
                else:
                    print_to_log(f"❌ [Level 4] Failed to generate query embedding via orchestrator")
            else:
                print_to_log(f"⚠️ [Level 4] BGE embeddings not available in orchestrator, using keyword fallback")
        
        except Exception as e:
            print_to_log(f"❌ [Level 4] BGE classification error: {e}")
            print_to_log(f"🔄 [Level 4] Falling back to keyword classification")
        
        # Keyword fallback classification
        print_to_log(f"🔑 [Level 4] Starting keyword fallback classification...")
        query_lower = query.lower().strip()
        print_to_log(f"🔍 [Level 4] Normalized query: '{query_lower}'")
        
        # Query keywords - check FIRST for highest priority (questions should override other keywords)
        # Use more precise matching to avoid "how" matching in "show"
        question_keywords = ['what', 'when', 'where', 'which', 'who', 'explain', 'help', '?', 'why are you', 'why applying', 'why use', 'why this', 'reasoning', 'rationale']
        
        # Check for "how" more precisely (avoid matching in "show")
        if ' how ' in f' {query_lower} ' or query_lower.startswith('how ') or query_lower.endswith(' how'):
            print_to_log(f"✅ [Level 4] Matched precise 'how' query pattern")
            return 'query'
        
        # Check for "why" more precisely 
        if ' why ' in f' {query_lower} ' or query_lower.startswith('why ') or query_lower.endswith(' why'):
            print_to_log(f"✅ [Level 4] Matched precise 'why' query pattern")
            return 'query'
            
        matched_query = [kw for kw in question_keywords if kw in query_lower]
        if matched_query:
            print_to_log(f"✅ [Level 4] Matched query keywords: {matched_query}")
            return 'query'
        
        # Skip keywords (check second - higher priority for commands like "skip next phase")
        skip_keywords = ['skip', 'pass', 'ignore', 'no thanks', 'bypass', 'move on']
        matched_skip = [kw for kw in skip_keywords if kw in query_lower]
        if matched_skip:
            print_to_log(f"✅ [Level 4] Matched skip keywords: {matched_skip}")
            return 'skip'
        
        # Proceed keywords (enhanced with "go ahead" variations)
        proceed_keywords = ['proceed', 'continue', 'next', 'go ahead', 'go', 'yes', 'ok', 'cool', 'sure', 'good',
                           'yeah', 'yep', 'fine', 'alright', 'right', 'correct', 'agreed', 'approve', 'carry on', 'keep going', 'move ahead', 'go forward', 'go through', 'go with', 'let\'s go', 'let\'s proceed', 'start', 'begin']
        matched_proceed = [kw for kw in proceed_keywords if kw in query_lower]
        if matched_proceed:
            print_to_log(f"✅ [Level 4] Matched proceed keywords: {matched_proceed}")
            return 'proceed'
        
        # Override keywords
        override_keywords = ['use', 'set', 'change', 'override', 'apply', 'modify', 'alter']
        matched_override = [kw for kw in override_keywords if kw in query_lower]
        if matched_override:
            print_to_log(f"✅ [Level 4] Matched override keywords: {matched_override}")
            return 'override'
        
        # Summary keywords
        summary_keywords = ['summary', 'show', 'strategies', 'current', 'plan', 'display']
        matched_summary = [kw for kw in summary_keywords if kw in query_lower]
        if matched_summary:
            print_to_log(f"✅ [Level 4] Matched summary keywords: {matched_summary}")
            return 'summary'
        
        # Datetime/Target keywords
        datetime_keywords = [
            'datetime', 'date column', 'time column', 'timestamp', 'oot month', 'out of time',
            'date format', 'time format', 'datetime format', 'datetime column', 'date_column',
            'transaction_date', 'order_date', 'created_at', 'timestamp_column', 'date_time',
            'datetime specification', 'date specification', 'time specification', 'column datetime',
            'set datetime', 'specify datetime', 'provide datetime', 'datetime is', 'date is', 'time is',
            'target', 'target column', 'target is', 'column target', 'target variable', 'dependent variable',
            'outcome variable', 'response variable', 'label column', 'label variable',
            '2023m08', '2024m01', '2023m12', '2022m', '2021m', '2020m', 'month 2023', 'month 2024'
        ]
        matched_datetime = [kw for kw in datetime_keywords if kw in query_lower]
        if matched_datetime:
            print_to_log(f"✅ [Level 4] Matched datetime/target keywords: {matched_datetime}")
            return 'datetime'
        
        # Default to proceed for short unrecognized patterns (likely affirmative)
        if len(query_lower.strip()) <= 5:  # Short responses likely affirmative
            print_to_log(f"🔄 [Level 4] Short response (≤5 chars), defaulting to 'proceed'")
            return 'proceed'
        
        print_to_log(f"🔄 [Level 4] No keywords matched, defaulting to 'query'")
        return 'query'

    def _classify_feature_selection_action(self, query: str) -> str:
        """
        Level 4: Classify specific feature selection actions using BGE with keyword fallbacks.
        
        Args:
            query: User's input query
            
        Returns:
            Classified action: 'proceed', 'analysis', 'query', 'summary', 'revert'
        """
        print_to_log(f"🔍 [FS Level 4] Processing query: '{query}'")
        
        try:
            # BGE-based classification using orchestrator's embeddings (same as Levels 1-3)
            from orchestrator import orchestrator
            if hasattr(orchestrator, '_intent_embeddings') and orchestrator._intent_embeddings is not None:
                print_to_log(f"🧠 [FS Level 4] BGE embeddings available via orchestrator, attempting semantic classification...")
                
                action_definitions = {
                    "proceed_action": "proceed with final summary, finish analysis, complete the analysis, I'm done, finalize, okay looks good, looks good, looks great, that's perfect, this is fine, go ahead, all set, perfect, good to go, proceed with final, finish feature selection, complete feature selection, done with analysis, ready to finish, wrap up analysis, finalize selection, end analysis, conclude analysis, complete this, finish this, done here, all done, looks perfect, this looks good, satisfied with results, happy with selection, ready to proceed, ready to continue, ready to move on, continue",
                    "analysis_action": "run IV analysis with 0.02, do IV with 0.02, do CSI analysis with 0.2, apply correlation filter with , run SHAP analysis with , do VIF filtering, run LASSO selection, apply PCA, run feature importance, do correlation analysis, filter features, select features, filter with threshold, analyze with cutoff, run with parameters, do with settings, apply with criteria, execute with threshold, perform with cutoff, IV analysis with threshold, CSI analysis with cutoff, correlation analysis with threshold, SHAP analysis with cutoff, VIF filtering with threshold, LASSO with parameters, PCA with components, implement iv, apply iv, iv filtering, iv value, iv cutoff, csi with, csi analysis, do CSI, do iv threshold, iv greater than, iv less than, filter by iv, with iv, iv >, iv <, iv >=, iv <=, filter all values, filter values, IV with 0.01, IV with 0.02, IV with 0.03, IV with 0.04, IV with 0.05, IV 0.01, IV 0.02, IV 0.03, IV 0.04, IV 0.05, IV 0.1, IV 0.2, IV 0.3, IV 0.4, IV 0.5, CSI 0.01, CSI 0.02, CSI 0.03, CSI 0.04, CSI 0.05, CSI 0.1, CSI 0.2, CSI 0.3, CSI 0.4, CSI 0.5, correlation 0.7, correlation 0.8, correlation 0.9, SHAP 0.01, SHAP 0.02, SHAP 0.05, VIF 5, VIF 10, decision tree feature importance, train decision tree, decision tree classifier, decision tree importance, top important features according to decision tree, what are top 10 important features according to decision tree, run rfe, recursive feature elimination, rfe selection, rfe analysis, features by rfe, top features by rfe, what are top features by rfe, implement csi, apply csi, csi filtering, characteristic stability, stability index, csi threshold, csi cutoff, implement correlation, apply correlation, correlation filtering, highly correlated, remove correlated, correlation threshold, corr analysis, implement shap, apply shap, shap filtering, shap cutoff, shap threshold, variance inflation, multicollinearity, vif threshold, vif cutoff, vif greater than, implement lasso, apply lasso, lasso regularization, features by lasso, top features by lasso, what are top features by lasso, lasso coefficients, principal component, dimensionality reduction, random forest importance, tree importance, model importance, remove features, keep features, feature subset, filter all, filter with, values with, greater than, less than, equal to, >, <, >=, <=, ==, !=, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5, implement, apply, execute, perform, calculate, compute, run analysis, do analysis, start analysis, begin analysis, conduct analysis",
                    "datetime_action":  "datetime transaction_date, datetime order_date, datetime created_at, datetime timestamp, datetime date_column, oot month 2023M08, oot month 2024M01, oot month 2023M12, oot month , out of time month specification, date format specification, timestamp format specification, datetime format specification, date column format, time column format, date specification, time specification, datetime info, date info, time info, column datetime, column date, column time, date column name, time column name, datetime column name, set datetime column, specify datetime column, provide datetime column, datetime column is, date column is, time column is",
                    "query_action": "what can you do, what are your capabilities, explain IV analysis, what is , how does this work, what is , explain , what is , how does x work, how many features remain, current state, what analyses have been done, current dataset info, show me results, give me scores, display values, show analysis results, what are the scores, how many features, which features, what happened, show me features, display features, tell me about features, explain results, help me understand, what does this mean, why these features, how does this work, what's the process, explain the analysis, describe the method, tell me more, show me more, give me details, provide information, current status, analysis status, feature status, selection status, progress status, what is, explain, how does, show me, tell me about, describe, what are, what are the, which are, which are the, give me, list, top 5, top 10, top 20, top 30, top 50, best 5, best 10, best 20, highest, lowest, most important, least important, value features, shap values, shap value, top shap features, top shap, importance ranking, feature ranking, what are top 10 important features according to decision tree feature importance, top important features according to decision tree, decision tree feature importance, most important features according to tree, what are top 10 features by rfe, what are top 10 features by lasso, top features by rfe, top features by lasso, features by rfe, features by lasso, rfe ranking, lasso ranking, what are top 10 features by pca, what are top 10 features based on pca, top features by pca, features by pca, pca ranking, pca components, show pca results, display pca analysis, pca feature importance, show correlation, correlation between, correlation matrix, correlation scores, feature correlation, correlation values, correlation results, top 20 correlation pairs, show correlation pairs, highest correlation pairs, correlation matrix, feature correlation pairs, most correlated features",
                    "summary_action": "current summary, show me pipeline summary, what analyses have I done, current state, pipeline status, show current progress, display analysis chain, show me what's done, current pipeline, analysis summary, feature selection summary, what have we accomplished, show progress, display status, current analysis state, show analysis history, display pipeline, show completed analyses, what's been done, analysis progress, selection progress, current results summary, show results summary, pipeline progress, analysis chain summary, completed steps, finished analyses, done analyses",
                    "revert_action": "revert to original, go back to initial state, start from start, reset to beginning, go back to clean data, restart from cleaned dataset, undo all changes, reset, revert",
                    "suggestion_action": "what should I do next, suggest next steps, recommend analysis, what analysis should I run next, guide me to next step, recommendations for analysis, advise me on next analysis, what analysis to do, help me choose, recommend analysis, suggest analysis, what's next, next step recommendation, analysis recommendation, guidance, advice, recommend me, guide me, help me decide, what would you suggest, any suggestions, any recommendations, what do you think I should do, what's the next best step, what analysis would be good, what should I focus on, what would help most, best next analysis, optimal next step, strategic recommendation, data science advice"
                }
                
                # Use orchestrator's embedding system (same as main pipeline)
                query_embedding = orchestrator._get_embedding(query)
                if query_embedding is not None:
                    print_to_log(f"✅ [FS Level 4] Query embedding generated successfully via orchestrator")
                    similarities = {}
                    for intent_name, definition in action_definitions.items():
                        intent_embedding = orchestrator._get_embedding(definition)
                        if intent_embedding is not None:
                            # Use cosine similarity directly (same as orchestrator does)
                            from sklearn.metrics.pairwise import cosine_similarity
                            similarity = cosine_similarity(
                                query_embedding.reshape(1, -1),
                                intent_embedding.reshape(1, -1)
                            )[0][0]
                            similarities[intent_name] = float(similarity)
                    
                    if similarities:
                        # Show all similarity scores for debugging
                        print_to_log(f"🔍 [FS Level 4] BGE similarity scores:")
                        for intent, score in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
                            action_name = intent.replace('_action', '')
                            print_to_log(f"   {action_name}: {score:.3f}")
                        
                        best_intent = max(similarities.items(), key=lambda x: x[1])
                        action_intent = best_intent[0].replace('_action', '')  # Remove _action suffix
                        print_to_log(f"🎯 [FS Level 4] BGE classified '{query}' as '{action_intent}' (confidence: {best_intent[1]:.3f})")
                        
                        if best_intent[1] > 0.3:  # Confidence threshold
                            print_to_log(f"✅ [FS Level 4] BGE confidence above threshold (0.3), returning: '{action_intent}'")
                            return action_intent
                        else:
                            print_to_log(f"⚠️ [FS Level 4] BGE confidence below threshold ({best_intent[1]:.3f} < 0.3), falling back to keywords")
                else:
                    print_to_log(f"❌ [FS Level 4] Failed to generate query embedding via orchestrator")
            else:
                print_to_log(f"⚠️ [FS Level 4] BGE embeddings not available in orchestrator, using keyword fallback")
        
        except Exception as e:
            print_to_log(f"❌ [FS Level 4] BGE classification error: {e}")
            print_to_log(f"🔄 [FS Level 4] Falling back to keyword classification")
        
        # Keyword fallback classification
        print_to_log(f"🔑 [FS Level 4] Starting keyword fallback classification...")
        query_lower = query.lower().strip()
        print_to_log(f"🔍 [FS Level 4] Normalized query: '{query_lower}'")
        
        # Suggestion keywords - check EARLY for recommendation requests
        suggestion_keywords = [
            'suggest', 'recommend', 'advice', 'guidance', 'what should i do', 'what next',
            'next step', 'what analysis', 'best analysis', 'help me choose', 'guide me',
            'recommend analysis', 'suggest analysis', 'what would you suggest', 'any suggestions',
            'any recommendations', 'what do you think', 'what would be good', 'what should i focus',
            'optimal next', 'strategic recommendation', 'data science advice', 'help me decide',
            'what should i', 'what would be', 'which analysis', 'best next', 'next best'
        ]
        
        if any(keyword in query_lower for keyword in suggestion_keywords):
            print_to_log(f"🎯 [FS Level 4] Keyword match found for SUGGESTION intent")
            return "suggestion"
        
        # Summary keywords - check FIRST for specific progress/status queries
        summary_keywords = [
            'summary', 'current state', 'pipeline status', 'show progress', 'analysis chain',
            'what have i done', 'what analyses', 'current progress', 'pipeline summary',
            'show current', 'display status', 'analysis summary', 'show current progress',
            'display progress', 'current analysis state', 'show analysis history'
        ]
        
        if any(keyword in query_lower for keyword in summary_keywords):
            print_to_log(f"🎯 [FS Level 4] Keyword match found for SUMMARY intent")
            return "summary"
        
        # Query keywords - check after summary for general questions
        query_keywords = [
            'what', 'how', 'why', 'explain', 'tell me', 'show me', 'help', '?', 'describe',
            'what is', 'how does', 'what are', 'can you explain', 'help me understand',
            'what does', 'how do', 'why do', 'why does', 'what happens', 'how come',
            'tell me about', 'show me about', 'explain to me', 'describe to me'
        ]
        
        if any(keyword in query_lower for keyword in query_keywords):
            print_to_log(f"🎯 [FS Level 4] Keyword match found for QUERY intent")
            return "query"
        
        # Analysis keywords (run/do/apply analysis)
        analysis_keywords = [
            'run iv', 'do iv', 'apply iv', 'iv analysis', 'run csi', 'do csi', 'apply csi', 'csi analysis',
            'run correlation', 'do correlation', 'correlation analysis', 'run shap', 'do shap', 'shap analysis',
            'run vif', 'do vif', 'vif analysis', 'run lasso', 'do lasso', 'lasso selection',
            'run pca', 'do pca', 'pca analysis', 'filter features', 'select features', 'analyze features',
            'run analysis', 'do analysis', 'apply analysis', 'execute analysis', 'perform analysis',
            'start analysis', 'begin analysis', 'with threshold', 'with cutoff', 'with parameters',
            # Simple analysis + threshold patterns
            'iv 0.01', 'iv 0.02', 'iv 0.03', 'iv 0.04', 'iv 0.05', 'iv 0.1', 'iv 0.2', 'iv 0.3', 'iv 0.4', 'iv 0.5',
            'csi 0.01', 'csi 0.02', 'csi 0.03', 'csi 0.04', 'csi 0.05', 'csi 0.1', 'csi 0.2', 'csi 0.3', 'csi 0.4', 'csi 0.5',
            'correlation 0.7', 'correlation 0.8', 'correlation 0.9', 'shap 0.01', 'shap 0.02', 'shap 0.05', 'vif 5', 'vif 10'
        ]
        
        if any(keyword in query_lower for keyword in analysis_keywords):
            print_to_log(f"🎯 [FS Level 4] Keyword match found for ANALYSIS intent")
            return "analysis"
        
        # Proceed keywords (finish/complete)
        proceed_keywords = [
            'proceed', 'finish', 'complete', 'done', 'finalize', 'looks good', 'looks great',
            'perfect', 'good to go', 'all set', 'ready to proceed', 'wrap up', 'conclude'
        ]
        
        if any(keyword in query_lower for keyword in proceed_keywords):
            print_to_log(f"🎯 [FS Level 4] Keyword match found for PROCEED intent")
            return "proceed"
        
        # Revert keywords
        revert_keywords = [
            'revert', 'reset', 'undo', 'go back', 'start over', 'begin again', 'clear',
            'restore', 'return to original', 'back to beginning', 'start fresh'
        ]
        
        if any(keyword in query_lower for keyword in revert_keywords):
            print_to_log(f"🎯 [FS Level 4] Keyword match found for REVERT intent")
            return "revert"
        
        # Datetime keywords for CSI analysis
        datetime_keywords = [
            'datetime', 'date column', 'time column', 'timestamp', 'oot month', 'out of time',
            'date format', 'time format', 'datetime format', 'datetime column', 'date_column',
            'transaction_date', 'order_date', 'created_at', 'timestamp_column', 'date_time',
            'datetime specification', 'date specification', 'time specification', 'column datetime',
            'set datetime', 'specify datetime', 'provide datetime', 'datetime is', 'date is', 'time is',
            '2023m08', '2024m01', '2023m12', '2022m', '2021m', '2020m', 'month 2023', 'month 2024'
        ]
        
        if any(keyword in query_lower for keyword in datetime_keywords):
            print_to_log(f"🎯 [FS Level 4] Keyword match found for DATETIME intent")
            return "datetime"
        
        # Default fallback
        print_to_log(f"🤷 [FS Level 4] No specific keyword matches found, defaulting to ANALYSIS")
        return "analysis"

    def _classify_in_phase_intent(self, query: str) -> str:
        """
        Classify user intent within an active preprocessing phase using embeddings with keyword fallbacks.
        
        Args:
            query: User's input query
            
        Returns:
            Classified intent: 'proceed', 'skip', 'query', 'override', 'summary', or original query
        """
        try:
            # Try semantic classification first (if embeddings available)
            from orchestrator import orchestrator
            if hasattr(orchestrator, '_intent_embeddings') and orchestrator._intent_embeddings is not None:
                intent, scores = orchestrator._classify_with_semantic_similarity(query)
                if scores.get('max_score', 0) > 0.6:  # High confidence threshold for in-phase
                    print_to_log(f"🧠 Semantic classification: '{query}' → '{intent}' (confidence: {scores.get('max_score', 0):.2f})")
                    return intent
            
            # Fallback to keyword matching
            query_lower = query.lower().strip()
            
            # Continuation keywords - expanded to include all affirmative responses
            continue_keywords = ['proceed', 'continue', 'next', 'go', 'yes', 'ok', 'cool', 'sure', 'good', 
                               'yeah', 'yep', 'fine', 'alright', 'right', 'correct', 'agreed', 'approve']
            if any(kw in query_lower for kw in continue_keywords):
                return 'proceed'
            
            # Skip keywords  
            skip_keywords = ['skip', 'pass', 'ignore', 'no thanks']
            if any(kw in query_lower for kw in skip_keywords):
                return 'skip'
            
            # Override keywords
            override_keywords = ['use', 'set', 'change', 'override', 'apply', 'modify']
            if any(kw in query_lower for kw in override_keywords):
                return 'override'
            
            # Summary/status keywords
            summary_keywords = ['summary', 'status', 'show', 'current', 'progress']
            if any(kw in query_lower for kw in summary_keywords):
                return 'summary'
            
            # Query keywords - explicit questions only
            question_keywords = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'explain', 'help', '?']
            if any(kw in query_lower for kw in question_keywords):
                return 'query'
            
            # Default to proceed for short unrecognized patterns (likely affirmative)
            if len(query_lower.strip()) <= 5:  # Short responses likely affirmative
                return 'proceed'
            
            return 'query'
            
        except Exception as e:
            print_to_log(f"⚠️ Error in in-phase intent classification: {e}")
            return 'query'  # Safe fallback


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
        print_to_log(f"\n{'='*60}")
        print_to_log(f"Testing query: {query}")
        print_to_log('='*60)
        
        result = ml_pipeline.process_query(query, session_id)
        print_to_log(f"\nResult: {result['response']}")
        
        if not result['success']:
            print_to_log(f"Error: {result.get('error')}")
            break
