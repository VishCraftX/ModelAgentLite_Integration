#!/usr/bin/env python3
"""
Global Toolbox for Multi-Agent ML System
Contains shared utilities used by all agents: SlackManager, ArtifactManager, 
ProgressTracker, and ExecutionAgent with fallback mechanism
"""

from print_to_log import print_to_log
import os
import json
import time
import tempfile
import traceback
import subprocess
import sys
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Pattern Classification imports (optional)
try:
    import ollama
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Text normalization imports (optional)
try:
    from nltk.stem import WordNetLemmatizer
    import nltk
    lemmatizer = WordNetLemmatizer()
    LEMMATIZER_AVAILABLE = True
except ImportError:
    LEMMATIZER_AVAILABLE = False

# Slack integration (optional)
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    WebClient = None
    SlackApiError = Exception
    SLACK_AVAILABLE = False

# LLM for fallback mechanism (optional)
try:
    import ollama
    # from langchain_openai import ChatOpenAI  # Removed - using Qwen models only.e
    from langchain_community.chat_models import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage
    LLM_AVAILABLE = True
except ImportError:
    ollama = None
    # ChatOpenAI = None  # Removed - using Qwen models only
    ChatOllama = None
    HumanMessage = None
    SystemMessage = None
    LLM_AVAILABLE = False

from pipeline_state import PipelineState


class SlackManager:
    """
    Manages Slack integration with multi-session support
    Each chat_session has its own thread and context
    """
    
    def __init__(self, bot_token: str = None):
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        if SLACK_AVAILABLE and self.bot_token:
            self.client = WebClient(token=self.bot_token)
        else:
            self.client = None
        self.session_threads = {}  # Maps session_id to thread_ts
        self.session_channels = {}  # Maps session_id to channel_id
        self.user_cache = {}  # Maps user_id to username for folder naming
    
    def get_username_from_user_id(self, user_id: str) -> str:
        """
        Fetch username from Slack API and cache it for future use.
        Returns a sanitized username suitable for folder names.
        """
        # Check cache first
        if user_id in self.user_cache:
            return self.user_cache[user_id]
        
        # If no Slack client available, return sanitized user_id
        if not self.client:
            sanitized_name = self._sanitize_for_folder_name(user_id)
            self.user_cache[user_id] = sanitized_name
            return sanitized_name
        
        try:
            # Fetch user info from Slack API
            response = self.client.users_info(user=user_id)
            user_info = response.get('user', {})
            
            # Try to get display name, real name, or username in that order
            username = (
                user_info.get('profile', {}).get('display_name') or
                user_info.get('profile', {}).get('real_name') or
                user_info.get('name') or
                user_id
            )
            
            # Sanitize the username for folder naming
            sanitized_name = self._sanitize_for_folder_name(username)
            
            # Cache the result
            self.user_cache[user_id] = sanitized_name
            
            print_to_log(f"👤 Fetched username for {user_id}: {sanitized_name}")
            return sanitized_name
            
        except Exception as e:
            print_to_log(f"⚠️ Failed to fetch username for {user_id}: {e}")
            # Fallback to sanitized user_id
            sanitized_name = self._sanitize_for_folder_name(user_id)
            self.user_cache[user_id] = sanitized_name
            return sanitized_name
    
    def _sanitize_for_folder_name(self, name: str) -> str:
        """
        Sanitize a name to be safe for use as a folder name.
        Removes or replaces characters that are not allowed in folder names.
        """
        import re
        
        # Remove or replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        sanitized = re.sub(r'[^\w\-_.]', '_', sanitized)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Ensure it's not empty and not too long
        if not sanitized:
            sanitized = "unknown_user"
        elif len(sanitized) > 50:
            sanitized = sanitized[:50]
        
        return sanitized
    
    def register_session(self, session_id: str, channel: str, thread_ts: str = None):
        """Register a new session with channel and thread information"""
        print(f"🔍 DEBUG: register_session called:")
        print(f"  Session ID: {session_id}")
        print(f"  Channel: {channel}")
        print(f"  Thread TS: {thread_ts}")
        print(f"  Before registration - sessions: {self.session_channels}")
        print(f"  Before registration - threads: {self.session_threads}")
        
        print(f"🔍 DEBUG SlackManager.register_session:")
        print(f"  Before: session_channels = {self.session_channels}")
        self.session_channels[session_id] = channel
        if thread_ts:
            self.session_threads[session_id] = thread_ts
        print(f"  After: session_channels = {self.session_channels}")
        print(f"  Registered session {session_id} with channel {channel}")
            
        print(f"🔍 DEBUG: After registration:")
        print(f"  Sessions: {self.session_channels}")
        print(f"  Threads: {self.session_threads}")
    
    def send_message(self, session_id: str, text: str, channel: str = None, thread_ts: str = None):
        """Send message to specific session"""
        if not self.client:
            print(f"[Slack:{session_id}] {text}")
            return
        
        try:
            # Use provided channel/thread or get from session
            if not channel:
                channel = self.session_channels.get(session_id)
            if not thread_ts:
                thread_ts = self.session_threads.get(session_id)
            
            print(f"🔍 DEBUG SlackManager.send_message:")
            print(f"  Session ID: {session_id}")
            print(f"  Channel: {channel}")
            print(f"  Thread TS: {thread_ts}")
            print(f"  Available sessions: {list(self.session_channels.keys())}")
            print(f"  Session channels: {self.session_channels}")
            
            # If still no channel, extract from session_id (format: user_threadts)
            if not channel and "_" in session_id:
                # This is a fallback - ideally channel should be set properly
                print(f"⚠️ No channel stored for session {session_id}, skipping Slack message")
                print(f"🔍 DEBUG: Available sessions: {self.session_channels}")
                print(f"[Slack:{session_id}] {text}")
                return
            
            if not channel:
                print(f"🔍 DEBUG SlackManager.send_message - SESSION NOT FOUND:")
                print(f"  Session ID: {session_id}")
                print(f"  Available sessions: {list(self.session_channels.keys())}")
                print(f"  Session channels: {self.session_channels}")
                print(f"❌ No channel available for session {session_id} - message will be logged to console only")
                print(f"[Slack:{session_id}] {text}")
                return
            
            print(f"🚀 Attempting to send Slack message to channel {channel}")
            response = self.client.chat_postMessage(
                channel=channel,
                text=text,
                thread_ts=thread_ts
            )
            
            # Store thread info for future messages
            if response["ok"]:
                self.session_threads[session_id] = response["ts"]
                self.session_channels[session_id] = channel
                
        except Exception as e:
            if SLACK_AVAILABLE and "SlackApiError" in str(type(e)):
                print(f"Slack API error for session {session_id}: {e}")
            else:
                print(f"Error sending Slack message for session {session_id}: {e}")
    
    def send_progress_update(self, session_id: str, stage: str, message: str):
        """Send formatted progress update"""
        progress_text = f"⏳ *{stage}:* {message}"
        self.send_message(session_id, progress_text)
    
    def send_error_message(self, session_id: str, error: str):
        """Send formatted error message"""
        error_text = f"❌ *Error:* {error}"
        self.send_message(session_id, error_text)
    
    def send_success_message(self, session_id: str, message: str):
        """Send formatted success message"""
        success_text = f"✅ {message}"
        self.send_message(session_id, success_text)
    
    
    def upload_file(self, session_id: str, file_path: str, title: str = None, comment: str = None):
        """Upload file to Slack thread"""
        if not self.client or not os.path.exists(file_path):
            return
        
        try:
            channel = self.session_channels.get(session_id)
            thread_ts = self.session_threads.get(session_id)
            
            if not channel:
                print(f"No channel found for session {session_id}")
                return
            
            with open(file_path, 'rb') as file_content:
                response = self.client.files_upload_v2(
                    channel=channel,
                    thread_ts=thread_ts,
                    file=file_content,
                    title=title or os.path.basename(file_path),
                    initial_comment=comment,
                    filename=os.path.basename(file_path)
                )
                
            if response["ok"]:
                print(f"Successfully uploaded {file_path} to session {session_id}")
            else:
                print(f"Failed to upload {file_path}: {response.get('error')}")
                
        except Exception as e:
            print(f"Error uploading file to Slack: {e}")


class ArtifactManager:
    """
    Manages artifacts (files, plots, models) with session isolation
    Each session has its own folder structure using usernames
    """
    
    def __init__(self, base_dir: str = None, slack_manager: SlackManager = None):
        if base_dir is None:
            base_dir = os.path.join(tempfile.gettempdir(), "mal_integration_artifacts")
        
        self.base_dir = Path(base_dir)
        self.slack_manager = slack_manager
        self.base_dir.mkdir(exist_ok=True)
    
    def _get_username_session_id(self, session_id: str) -> str:
        """Convert session_id to use username instead of user_id"""
        if "_" in session_id:
            user_id_part, thread_ts = session_id.split("_", 1)
            if self.slack_manager:
                username = self.slack_manager.get_username_from_user_id(user_id_part)
                return f"{username}_{thread_ts}"
            else:
                # Fallback to sanitized user_id
                sanitized_user = self._sanitize_for_folder_name(user_id_part)
                return f"{sanitized_user}_{thread_ts}"
        else:
            # Single user_id case
            if self.slack_manager:
                username = self.slack_manager.get_username_from_user_id(session_id)
                return username
            else:
                return self._sanitize_for_folder_name(session_id)
    
    def _sanitize_for_folder_name(self, name: str) -> str:
        """Sanitize a name to be safe for use as a folder name"""
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        sanitized = re.sub(r'[^\w\-_.]', '_', sanitized)
        sanitized = sanitized.strip('. ')
        if not sanitized:
            sanitized = "unknown_user"
        elif len(sanitized) > 50:
            sanitized = sanitized[:50]
        return sanitized
    
    def get_session_dir(self, session_id: str) -> Path:
        """Get or create session directory using username"""
        username_session_id = self._get_username_session_id(session_id)
        session_dir = self.base_dir / username_session_id
        session_dir.mkdir(exist_ok=True)
        return session_dir
    
    def save_artifact(self, session_id: str, filename: str, content: Any, artifact_type: str = "file") -> str:
        """Save artifact to session folder"""
        try:
            session_dir = self.get_session_dir(session_id)
            file_path = session_dir / filename
            
            if artifact_type == "dataframe" and hasattr(content, 'to_csv'):
                content.to_csv(file_path, index=False)
            elif artifact_type == "json":
                with open(file_path, 'w') as f:
                    json.dump(content, f, indent=2, default=str)
            elif artifact_type == "text":
                with open(file_path, 'w') as f:
                    f.write(str(content))
            elif artifact_type == "binary":
                with open(file_path, 'wb') as f:
                    f.write(content)
            else:
                # Default: try to save as text
                with open(file_path, 'w') as f:
                    f.write(str(content))
            
            print(f"Saved artifact for {session_id}: {filename}")
            return str(file_path)
            
        except Exception as e:
            print(f"Error saving artifact {filename} for session {session_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_artifact(self, session_id: str, filename: str, artifact_type: str = "file") -> Any:
        """Load artifact from session folder"""
        session_dir = self.get_session_dir(session_id)
        file_path = session_dir / filename
        
        if not file_path.exists():
            return None
        
        try:
            if artifact_type == "dataframe":
                return pd.read_csv(file_path)
            elif artifact_type == "json":
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif artifact_type == "text":
                with open(file_path, 'r') as f:
                    return f.read()
            elif artifact_type == "binary":
                with open(file_path, 'rb') as f:
                    return f.read()
            else:
                with open(file_path, 'r') as f:
                    return f.read()
                    
        except Exception as e:
            print(f"Error loading artifact {filename} for session {session_id}: {e}")
            return None
    
    def list_artifacts(self, session_id: str) -> List[str]:
        """List all artifacts for a session"""
        session_dir = self.get_session_dir(session_id)
        if not session_dir.exists():
            return []
        
        return [f.name for f in session_dir.iterdir() if f.is_file()]
    
    def cleanup_session(self, session_id: str):
        """Clean up all artifacts for a session"""
        import shutil
        session_dir = self.get_session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
            print(f"Cleaned up artifacts for session {session_id}")


class ProgressTracker:
    """
    Tracks progress and sends updates to both state and Slack - BALANCED MODE
    Shows key milestones and processing confirmations so users know their request is being handled
    Filters out only very verbose internal debugging messages
    """
    
    def __init__(self, slack_manager: SlackManager = None):
        self.slack_manager = slack_manager
        self._last_message = None
        self._thinking_message_id = None  # Track thinking message for updates
        self._thinking_emojis = ["🤔", "💭", "🧠", "⚡", "🔍", "✨"]
        self._emoji_index = 0
    
    # REMOVED: Thinking animation methods (caused spam)
    # Users don't want constant message updates - keep it minimal
    
    def update(self, state: PipelineState, message: str, stage: str = None, send_to_slack: bool = True):
        """Update progress - Balanced approach: key milestones + processing confirmations"""
        # Create full message for deduplication
        full_message = f"{stage}: {message}" if stage else message
        
        # Skip if this is the same message as last time
        if full_message == self._last_message:
            return
        self._last_message = full_message
        
        # Key stages that users should always see
        key_stages = [
            "Preprocessing",     # Data preprocessing started
            "Feature Selection", # Feature selection started
            "Model Building",    # Model building started
            "Code Generation",   # LLM generating code
            "Code Execution",    # Code being executed
            "Analysis",          # Analysis in progress
            "Completed",         # Task completed
            "Error"             # Errors
        ]
        
        # Important messages that indicate processing/progress
        key_messages = [
            "request received",
            "starting",
            "processing",
            "analyzing", 
            "generating",
            "executing",
            "training",
            "building",
            "complete",
            "finished",
            "ready",
            "error",
            "failed"
        ]
        
        # Skip only very verbose internal messages
        skip_messages = [
            "debug",
            "trace",
            "internal",
            "cache hit",
            "loading cached"
        ]
        
        # Check if this should be sent to Slack
        should_send = (
            # Always send key stages
            stage in key_stages or
            # Send messages with key progress indicators
            any(key_msg in message.lower() for key_msg in key_messages) or
            # Send unless it's a skip message
            not any(skip_msg in message.lower() for skip_msg in skip_messages)
        )
        
        # Console logging (always show for debugging)
        timestamp = datetime.now().strftime("%H:%M:%S")
        agent_info = f" [{state.current_agent}]" if state.current_agent else ""
        print(f"[{timestamp}]{agent_info} {message}")
        
        # Send to Slack with better user experience
        if send_to_slack and should_send and self.slack_manager and state.chat_session:
            if stage:
                self.slack_manager.send_progress_update(state.chat_session, stage, message)
            else:
                self.slack_manager.send_message(state.chat_session, f"⏳ {message}")
        
        # Update state (after logging to avoid duplicate console output)
        if stage:
            state.update_progress(f"{stage}: {message}", state.current_agent)
        else:
            state.update_progress(message, state.current_agent)


class UniversalPatternClassifier:
    """Standalone Universal Pattern Classifier - Semantic → LLM → Keyword for all pattern detection"""
    
    # Predefined threshold profiles for different use cases
    THRESHOLD_PROFILES = {
        "critical_routing": {
            "semantic_threshold": 0.50,      # Conservative - avoid misrouting
            "confidence_threshold": 0.12,    # Need clear winner
            "description": "High-stakes routing decisions"
        },
        "intent_classification": {
            "semantic_threshold": 0.4,       # Balanced
            "confidence_threshold": 0.08,    # Moderate confidence
            "description": "Main intent classification"
        },
        "skip_patterns": {
            "semantic_threshold": 0.35,      # More liberal for better skip detection
            "confidence_threshold": 0.08,    # Lower confidence needed
            "description": "Workflow skip detection"
        },
        "session_continuation": {
            "semantic_threshold": 0.35,      # Liberal - better to continue wrongly than break flow
            "confidence_threshold": 0.06,    # Low confidence OK
            "description": "Interactive session flow"
        },
        "feature_detection": {
            "semantic_threshold": 0.3,       # Liberal - plot/visualization detection
            "confidence_threshold": 0.05,    # Low confidence OK
            "description": "Feature/plot/analysis detection"
        },
        "educational_queries": {
            "semantic_threshold": 0.3,       # More liberal for educational detection
            "confidence_threshold": 0.06,    # Lower confidence for better detection
            "description": "Educational vs action intent"
        },
        "model_sub_classification": {
            "semantic_threshold": 0.3,       # More liberal - within model building context
            "confidence_threshold": 0.04,    # Very low confidence OK - we're already in model context
            "description": "Model building sub-classifications"
        }
    }
    
    def __init__(self):
        self._embedding_cache = {}
        self.default_model = os.getenv("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
        
    def classify_pattern(self, query: str, pattern_definitions: dict, 
                        use_case: str = "intent_classification",
                        context_adjustments: dict = None) -> tuple:
        """
        Universal pattern classification: Semantic → LLM → Keyword
        Returns: (classification_result, method_used)
        """
        # Get base thresholds for use case
        if use_case not in self.THRESHOLD_PROFILES:
            print(f"⚠️ Unknown use case '{use_case}', using 'intent_classification'")
            use_case = "intent_classification"
            
        base_thresholds = self.THRESHOLD_PROFILES[use_case]
        semantic_threshold = base_thresholds["semantic_threshold"]
        confidence_threshold = base_thresholds["confidence_threshold"]
        
        # Apply context-based adjustments
        if context_adjustments:
            semantic_threshold += context_adjustments.get("semantic_adjust", 0)
            confidence_threshold += context_adjustments.get("confidence_adjust", 0)
            
        # Clamp to reasonable bounds
        semantic_threshold = max(0.2, min(0.7, semantic_threshold))
        confidence_threshold = max(0.03, min(0.2, confidence_threshold))
        
        print(f"🎯 Pattern classification for {use_case}: semantic={semantic_threshold:.3f}, confidence={confidence_threshold:.3f}")
        
        # Step 1: Semantic Classification
        if EMBEDDINGS_AVAILABLE:
            semantic_result = self._semantic_classify(
                query, pattern_definitions, semantic_threshold, confidence_threshold
            )
            if semantic_result:
                print(f"🧠 Semantic pattern classification used (use_case: {use_case})")
                return semantic_result, "semantic"
        
        # Step 2: LLM Classification  
        llm_result = self._llm_classify(query, pattern_definitions, use_case)
        if llm_result:
            print(f"🤖 LLM pattern classification used (use_case: {use_case})")
            return llm_result, "llm"
            
        # Step 3: Keyword Fallback
        print(f"⚡ Keyword pattern fallback used (use_case: {use_case})")
        keyword_result = self._keyword_classify(query, pattern_definitions)
        return keyword_result, "keyword"
    
    def _get_embedding(self, text: str):
        """Get embedding for text using Ollama with caching"""
        if not EMBEDDINGS_AVAILABLE:
            return None
            
        # Check cache first
        if text in self._embedding_cache:
            return self._embedding_cache[text]
            
        # Try different embedding models in order of preference
        embedding_models = ["bge-large", "mxbai-embed-large", "nomic-embed-text", "all-minilm"]
        
        for model in embedding_models:
            try:
                response = ollama.embeddings(model=model, prompt=text)
                embedding = np.array(response["embedding"])
                self._embedding_cache[text] = embedding
                return embedding
            except Exception as e:
                continue
                
        print(f"⚠️ Failed to get embedding for text: {text[:50]}...")
        return None
    
    def _semantic_classify(self, query: str, pattern_definitions: dict, 
                          semantic_threshold: float, confidence_threshold: float) -> Optional[str]:
        """Semantic classification using embeddings"""
        try:
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return None
                
            similarities = {}
            for pattern_name, pattern_description in pattern_definitions.items():
                pattern_embedding = self._get_embedding(pattern_description)
                if pattern_embedding is not None:
                    similarity = cosine_similarity([query_embedding], [pattern_embedding])[0][0]
                    similarities[pattern_name] = similarity
            
            if not similarities:
                return None
                
            # Find best and second best
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            best_pattern, best_score = sorted_similarities[0]
            second_score = sorted_similarities[1][1] if len(sorted_similarities) > 1 else 0
            
            score_diff = best_score - second_score
            
            print(f"[Semantic] Best: {best_pattern} (score: {best_score:.3f}, diff: {score_diff:.3f})")
            
            # Check both thresholds
            if best_score > semantic_threshold and score_diff > confidence_threshold:
                return best_pattern
                
            print(f"[Semantic] Below threshold - score: {best_score:.3f} < {semantic_threshold:.3f} or diff: {score_diff:.3f} < {confidence_threshold:.3f}")
            return None
            
        except Exception as e:
            print(f"[Semantic] Error in semantic classification: {e}")
            return None
    
    def _llm_classify(self, query: str, pattern_definitions: dict, use_case: str) -> Optional[str]:
        """LLM classification with context-aware prompts"""
        try:
            # Create context-aware prompt based on use case
            if use_case == "skip_patterns":
                system_prompt = """You are a workflow pattern classifier. Analyze user queries to detect EXPLICIT workflow skipping requests.

CRITICAL: Only classify as skip patterns when user EXPLICITLY asks to skip/bypass steps.

Pattern meanings:
- "skip_to_modeling": User explicitly asks to skip ALL steps and go directly to NEW model building
  Examples: "skip preprocessing and feature selection", "go straight to modeling", "bypass everything"
  
- "skip_preprocessing_to_modeling": User explicitly asks to skip preprocessing but wants NEW model building  
  Examples: "skip data cleaning and train NEW model", "bypass preprocessing for NEW model"
  
- "skip_preprocessing_to_features": User explicitly asks to skip preprocessing but wants feature selection
  Examples: "skip cleaning but do feature selection", "bypass preprocessing and select features"
  
- "no_skip": Normal workflow OR existing model usage (VERY IMPORTANT!)
  Examples: "use this model", "apply existing model", "use current model for analysis"
  
CRITICAL RULES:
1. If query mentions "use this model", "apply existing", "current model" → ALWAYS classify as "no_skip"
2. Only classify as skip patterns if user explicitly uses words like "skip", "bypass", "go straight to"
3. Model application/usage queries are NOT skip patterns, they are normal workflow ("no_skip")"""
            elif use_case == "session_continuation":
                system_prompt = """You are a session flow classifier. Determine if the user wants to continue their current task or start something new."""
            elif use_case == "educational_queries":
                system_prompt = """You are an educational vs action intent classifier. Your job is to distinguish between learning requests and action requests.

CRITICAL DISTINCTION:

"educational" - User wants to LEARN or GET INFORMATION about something:
- Questions: "what is", "how does", "explain", "tell me about", "what are"
- Learning: "what are different methods", "what are various techniques", "types of"
- Information seeking: "describe", "define", "help me understand"
- Examples: "what is random forest?", "explain decision trees", "what are different preprocessing methods?"

"action" - User wants to DO something or PERFORM a task:
- Commands: "build", "train", "create", "run", "execute", "apply", "use"
- Task execution: "implement", "perform", "generate", "develop"
- Examples: "build a model", "train a classifier", "apply preprocessing"

CRITICAL RULE: If the query asks "what are different [methods/techniques/types]" → ALWAYS classify as "educational"
The word "different" + methods/techniques = learning request, NOT action request."""
            else:
                system_prompt = """You are a pattern classifier. Analyze the user query and classify it according to the provided patterns."""
            
            pattern_list = "\n".join([f"- {name}: {desc[:100]}..." for name, desc in pattern_definitions.items()])
            
            prompt = f"""{system_prompt}

Available patterns:
{pattern_list}

User query: "{query}"

Respond with ONLY the pattern name that best matches the query. If none match well, respond with "uncertain"."""

            # Try Ollama 
            try:
                response = ollama.chat(
                    model=self.default_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    options={"temperature": 0.1}
                )
                
                if response and "message" in response and "content" in response["message"]:
                    result = response["message"]["content"].strip().lower()
                    
                    # Validate that response is one of the valid patterns
                    for pattern_name in pattern_definitions.keys():
                        if pattern_name.lower() == result:
                            return pattern_name
            except Exception as e:
                print(f"[LLM] Ollama error: {e}")
                        
            return None
            
        except Exception as e:
            print(f"[LLM] Error in LLM classification: {e}")
            return None
    
    def _keyword_classify(self, query: str, pattern_definitions: dict) -> str:
        """Enhanced keyword-based classification fallback"""
        query_lower = query.lower()
        
        # Convert pattern definitions to keyword lists with weighted scoring
        pattern_scores = {}
        
        for pattern_name, pattern_description in pattern_definitions.items():
            # Extract keywords from description 
            keywords = pattern_description.lower().split()
            
            # Weighted scoring: exact matches get higher scores
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    # Exact word match gets higher score
                    if f" {keyword} " in f" {query_lower} ":
                        score += 2
                    else:
                        score += 1
                        
            # Bonus for pattern-specific indicators
            if pattern_name == "educational" and any(edu in query_lower for edu in ["what is", "how does", "explain", "tell me"]):
                score += 5
            elif pattern_name == "skip_to_modeling" and any(skip in query_lower for skip in ["skip everything", "bypass all", "skip all"]):
                score += 5
            elif pattern_name == "preprocessing" and any(prep in query_lower for prep in ["normalize", "standardize", "scale"]):
                score += 3
            elif pattern_name == "model_building" and any(model in query_lower for model in ["use this model", "apply model", "existing model"]):
                score += 5
            elif pattern_name == "no_skip" and any(existing in query_lower for existing in ["use this model", "use existing", "apply existing", "current model", "existing model"]):
                score += 10  # High priority for existing model usage
            elif pattern_name == "action" and any(action in query_lower for action in ["what is", "how does", "explain", "tell me", "what are", "different methods", "different techniques", "types of", "kinds of"]):
                score -= 8  # Strong penalty for action when educational language detected
            elif pattern_name == "educational" and any(edu in query_lower for edu in ["different methods", "different techniques", "types of", "kinds of", "what are different", "various methods", "various techniques"]):
                score += 8  # Strong bonus for educational when asking about methods/techniques
            elif pattern_name == "use_existing" and any(operation in query_lower for operation in ["create", "build", "generate"]):
                # For model operations, assume use_existing unless explicitly "new"
                if "new" not in query_lower and any(context in query_lower for context in ["rank order", "segments", "deciles", "buckets", "table"]):
                    score += 15  # Very strong bonus for model operations without "new"
                
            pattern_scores[pattern_name] = score
            
        # Return pattern with highest score, or first pattern if tie
        if pattern_scores:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            max_score = pattern_scores[best_pattern]
            
            # If max score is 0, use fallback logic
            if max_score == 0:
                # Smart fallback based on common patterns
                if any(q in query_lower for q in ["what", "how", "explain", "tell me"]):
                    return "educational" if "educational" in pattern_definitions else list(pattern_definitions.keys())[0]
                elif any(q in query_lower for q in ["skip", "bypass"]):
                    skip_patterns = [k for k in pattern_definitions.keys() if "skip" in k]
                    return skip_patterns[0] if skip_patterns else list(pattern_definitions.keys())[0]
                    
            return best_pattern
        
        # Fallback to first pattern
        return list(pattern_definitions.keys())[0]


class UserDirectoryManager:
    """Manages user directory structure using usernames instead of user IDs"""
    
    def __init__(self, base_data_dir: str = "user_data", slack_manager: SlackManager = None):
        self.base_data_dir = base_data_dir
        self.slack_manager = slack_manager
        self._ensure_base_directory()
    
    def _ensure_base_directory(self):
        """Ensure base user data directory exists"""
        if not os.path.exists(self.base_data_dir):
            os.makedirs(self.base_data_dir)
            print(f"📁 Created base directory: {self.base_data_dir}")
    
    def _get_thread_id(self, user_id: str) -> tuple[str, str]:
        """Extract user and thread from user_id format: user_threadts"""
        if "_" in user_id:
            parts = user_id.split("_", 1)  # Split only on first underscore
            return parts[0], parts[1]  # user, thread_ts
        return user_id, "main"  # fallback to main thread
    
    def _get_username_for_user_id(self, user_id: str) -> str:
        """Get username for user_id, using Slack API if available"""
        if self.slack_manager:
            return self.slack_manager.get_username_from_user_id(user_id)
        else:
            # Fallback to sanitized user_id
            return self._sanitize_for_folder_name(user_id)
    
    def _sanitize_for_folder_name(self, name: str) -> str:
        """Sanitize a name to be safe for use as a folder name"""
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        sanitized = re.sub(r'[^\w\-_.]', '_', sanitized)
        sanitized = sanitized.strip('. ')
        if not sanitized:
            sanitized = "unknown_user"
        elif len(sanitized) > 50:
            sanitized = sanitized[:50]
        return sanitized
    
    def _get_user_thread_dir(self, user_id: str) -> str:
        """Get directory path for specific user thread using username"""
        user_id_part, thread_ts = self._get_thread_id(user_id)
        username = self._get_username_for_user_id(user_id_part)
        return os.path.join(self.base_data_dir, username, thread_ts)
    
    def ensure_user_directory(self, user_id: str) -> str:
        """Ensure user thread directory exists and return path"""
        user_dir = self._get_user_thread_dir(user_id)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            print(f"📁 Created user directory: {user_dir}")
        return user_dir
    
    def get_artifacts_dir(self, user_id: str) -> str:
        """Get artifacts directory for user"""
        user_dir = self.ensure_user_directory(user_id)
        artifacts_dir = os.path.join(user_dir, "artifacts")
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
        return artifacts_dir
    
    def get_data_dir(self, user_id: str) -> str:
        """Get data directory for user"""
        user_dir = self.ensure_user_directory(user_id)
        data_dir = os.path.join(user_dir, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir
    
    def get_models_dir(self, user_id: str) -> str:
        """Get models directory for user"""
        user_dir = self.ensure_user_directory(user_id)
        models_dir = os.path.join(user_dir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        return models_dir
    
    def cleanup_user_session(self, user_id: str, keep_artifacts: bool = True):
        """Clean up user session data"""
        user_dir = self._get_user_thread_dir(user_id)
        if os.path.exists(user_dir):
            if keep_artifacts:
                # Only clean temp files, keep artifacts and models
                temp_files = [f for f in os.listdir(user_dir) if f.startswith('temp_')]
                for temp_file in temp_files:
                    os.remove(os.path.join(user_dir, temp_file))
            else:
                # Remove entire user directory
                import shutil
                shutil.rmtree(user_dir)
                print(f"🗑️ Cleaned up user directory: {user_dir}")


class ExecutionAgent:
    """
    Centralized code execution with LLM-powered fallback mechanism
    Handles all code execution for the multi-agent system
    """
    
    def __init__(self, 
                 main_model: str = "qwen2.5-coder:32b-instruct-q4_K_M",
                 fallback_model_1: str = None,
                 fallback_model_2: str = "deepseek-coder-v2:latest"):
        
        self.main_model = main_model
        self.fallback_model_1 = fallback_model_1 or main_model
        self.fallback_model_2 = fallback_model_2
        
        # Execution context
        self.execution_globals = {
            'pd': pd,
            'np': np,
            'os': os,
            'json': json,
            'datetime': datetime
        }
    
    def run_code(self, state: PipelineState, code: str, context: Dict[str, Any] = None) -> PipelineState:
        """
        Execute code with error handling and fallback mechanism
        """
        if context:
            self.execution_globals.update(context)
        
        try:
            # Create a safe execution environment
            exec_globals = self.execution_globals.copy()
            exec_locals = {}
            
            # Execute the code
            exec(code, exec_globals, exec_locals)
            
            # Store successful execution
            state.last_code = code
            state.last_error = None
            state.add_execution_record(
                agent=state.current_agent or "ExecutionAgent",
                action="code_execution",
                result="Success"
            )
            
            # Update execution globals with any new variables
            self.execution_globals.update(exec_locals)
            
            return state
            
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            
            print(f"[ExecutionAgent] Code execution failed: {error_msg}")
            print(f"[ExecutionAgent] Traceback: {error_traceback}")
            
            # Store error
            state.last_error = error_msg
            state.add_execution_record(
                agent=state.current_agent or "ExecutionAgent",
                action="code_execution",
                error=error_msg
            )
            
            # Attempt fallback fix
            return self.fallback_fix(state, code, error_msg, attempt=1)
    
    def fallback_fix(self, state: PipelineState, code: str, error: str, attempt: int = 1) -> PipelineState:
        """
        Use LLM to fix code errors with tiered fallback approach
        """
        if not LLM_AVAILABLE:
            print(f"[ExecutionAgent] LLM not available for fallback, skipping fix")
            state.last_error = f"Code execution failed: {error} (LLM fallback not available)"
            return state
            
        if attempt > 2:
            print(f"[ExecutionAgent] Maximum fallback attempts reached")
            state.last_error = f"Code execution failed after {attempt-1} fix attempts: {error}"
            return state
        
        # Select model based on attempt
        model_to_use = self.fallback_model_1 if attempt == 1 else self.fallback_model_2
        
        print(f"[ExecutionAgent] Attempting fix #{attempt} using {model_to_use}")
        
        try:
            # Prepare fallback prompt
            fallback_prompt = self._create_fallback_prompt(code, error, state.user_query or "")
            
            # Get LLM instance
            llm = self._get_llm(model_to_use)
            
            # Get fix suggestion
            if HumanMessage:
                response = llm.invoke([HumanMessage(content=fallback_prompt)])
                fixed_code = self._extract_code_from_response(response.content)
            else:
                print(f"[ExecutionAgent] HumanMessage not available")
                return self.fallback_fix(state, code, error, attempt + 1)
            
            if not fixed_code or fixed_code.strip() == code.strip():
                print(f"[ExecutionAgent] No meaningful fix suggested by {model_to_use}")
                return self.fallback_fix(state, code, error, attempt + 1)
            
            print(f"[ExecutionAgent] Attempting to execute fixed code...")
            
            # Try executing the fixed code
            try:
                exec_globals = self.execution_globals.copy()
                exec_locals = {}
                exec(fixed_code, exec_globals, exec_locals)
                
                # Success!
                state.last_code = fixed_code
                state.last_error = None
                state.add_execution_record(
                    agent=state.current_agent or "ExecutionAgent",
                    action=f"fallback_fix_attempt_{attempt}",
                    result="Success"
                )
                
                self.execution_globals.update(exec_locals)
                print(f"[ExecutionAgent] Fix #{attempt} successful!")
                return state
                
            except Exception as fix_error:
                fix_error_msg = str(fix_error)
                print(f"[ExecutionAgent] Fix #{attempt} failed: {fix_error_msg}")
                
                # Try next fallback
                return self.fallback_fix(state, fixed_code, fix_error_msg, attempt + 1)
                
        except Exception as llm_error:
            print(f"[ExecutionAgent] LLM fallback error: {llm_error}")
            return self.fallback_fix(state, code, error, attempt + 1)
    
    def _create_fallback_prompt(self, code: str, error: str, user_query: str) -> str:
        """Create prompt for LLM fallback"""
        return f"""You are a Python code debugging expert. Fix the following code that failed with an error.

USER QUERY: {user_query}

FAILING CODE:
```python
{code}
```

ERROR MESSAGE:
{error}

INSTRUCTIONS:
1. Analyze the error and identify the root cause
2. Provide a fixed version of the code
3. Ensure the fix addresses the specific error
4. Keep the same functionality and intent
5. Only return the corrected Python code in a code block

FIXED CODE:"""
    
    def _get_llm(self, model_name: str):
        """Get appropriate LLM instance"""
        if not LLM_AVAILABLE:
            raise Exception("LLM libraries not available")
            
        # Check for unsupported OpenAI models
        if model_name.startswith("gpt-"):
            raise ValueError(f"OpenAI models not supported. Use Qwen models instead. Model requested: {model_name}")
        elif ChatOllama:
            return ChatOllama(
                model=model_name,
                temperature=0
            )
        else:
            raise Exception("No LLM implementation available")
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response"""
        import re
        
        # Look for code blocks
        code_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        match = re.search(code_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: return the response as-is if no code block found
        return response.strip()


# Global toolbox instances
slack_manager = SlackManager()
artifact_manager = ArtifactManager(slack_manager=slack_manager)
progress_tracker = ProgressTracker(slack_manager)
execution_agent = ExecutionAgent()
user_directory_manager = UserDirectoryManager(slack_manager=slack_manager)
pattern_classifier = UniversalPatternClassifier()


def initialize_toolbox(slack_token: str = None, artifacts_dir: str = None, user_data_dir: str = None):
    """Initialize global toolbox with custom configuration"""
    global slack_manager, artifact_manager, progress_tracker, execution_agent, user_directory_manager, pattern_classifier
    
    # Preserve existing session data if slack_manager already exists
    existing_sessions = {}
    existing_threads = {}
    if slack_manager and hasattr(slack_manager, 'session_channels'):
        existing_sessions = slack_manager.session_channels.copy()
        existing_threads = slack_manager.session_threads.copy()
        print(f"🔍 DEBUG: Preserving {len(existing_sessions)} existing sessions")
        print(f"🔍 DEBUG: Existing sessions: {existing_sessions}")
        print(f"🔍 DEBUG: Existing threads: {existing_threads}")
        print(f"🔍 DEBUG: slack_manager type: {type(slack_manager)}")
        print(f"🔍 DEBUG: slack_manager id: {id(slack_manager)}")
    
    if slack_token and not hasattr(slack_manager, 'client'):
        # Only create new SlackManager if one doesn't exist yet
        slack_manager = SlackManager(slack_token)
        progress_tracker = ProgressTracker(slack_manager)
        print("🔧 Created new SlackManager instance")
    elif slack_token and hasattr(slack_manager, 'client'):
        # Reuse existing SlackManager to preserve session channels
        print(f"🔄 Reusing existing SlackManager (has {len(slack_manager.session_channels)} sessions)")
    elif slack_token:
        # Fallback - create new one
        slack_manager = SlackManager(slack_token)
        # Restore existing sessions
        if existing_sessions:
            print(f"🔍 DEBUG: Before restoration - new slack_manager sessions: {slack_manager.session_channels}")
            print(f"🔍 DEBUG: Before restoration - new slack_manager threads: {slack_manager.session_threads}")
            slack_manager.session_channels.update(existing_sessions)
            slack_manager.session_threads.update(existing_threads)
            print(f"🔍 DEBUG: Restored {len(existing_sessions)} sessions to new SlackManager")
            print(f"🔍 DEBUG: Restored sessions: {slack_manager.session_channels}")
            print(f"🔍 DEBUG: Restored threads: {slack_manager.session_threads}")
            print(f"🔍 DEBUG: New slack_manager id: {id(slack_manager)}")
        progress_tracker = ProgressTracker(slack_manager)
        print("🔧 Created fallback SlackManager instance")
    
    if artifacts_dir:
        artifact_manager = ArtifactManager(artifacts_dir, slack_manager)
    else:
        # Update existing artifact_manager with slack_manager reference
        if 'artifact_manager' in globals() and artifact_manager:
            artifact_manager.slack_manager = slack_manager
    
    if user_data_dir:
        user_directory_manager = UserDirectoryManager(user_data_dir, slack_manager)
    else:
        # Update existing user_directory_manager with slack_manager reference
        if 'user_directory_manager' in globals() and user_directory_manager:
            user_directory_manager.slack_manager = slack_manager
    
    # Initialize universal pattern classifier
    pattern_classifier = UniversalPatternClassifier()
    
    print("🧰 Global toolbox initialized with Universal Pattern Classifier")
    print(f"   Slack: {'✅ Enabled' if slack_manager.client else '❌ Disabled'}")
    print(f"   Artifacts: {artifact_manager.base_dir}")
    print(f"   User Data: {user_directory_manager.base_data_dir}")
    print(f"   Execution: ✅ Ready with fallback support")
