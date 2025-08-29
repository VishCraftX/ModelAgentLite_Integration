#!/usr/bin/env python3
"""
Global Toolbox for Multi-Agent ML System
Contains shared utilities used by all agents: SlackManager, ArtifactManager, 
ProgressTracker, and ExecutionAgent with fallback mechanism
"""

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
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_models import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage
    LLM_AVAILABLE = True
except ImportError:
    ollama = None
    ChatOpenAI = None
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
    
    def register_session(self, session_id: str, channel: str, thread_ts: str = None):
        """Register a new session with channel and thread information"""
        self.session_channels[session_id] = channel
        if thread_ts:
            self.session_threads[session_id] = thread_ts
    
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
            
            # If still no channel, extract from session_id (format: user_threadts)
            if not channel and "_" in session_id:
                # This is a fallback - ideally channel should be set properly
                print(f"⚠️ No channel stored for session {session_id}, skipping Slack message")
                print(f"[Slack:{session_id}] {text}")
                return
            
            if not channel:
                print(f"❌ No channel available for session {session_id}")
                print(f"[Slack:{session_id}] {text}")
                return
            
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
    Each session has its own folder structure
    """
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(tempfile.gettempdir(), "mal_integration_artifacts")
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def get_session_dir(self, session_id: str) -> Path:
        """Get or create session directory"""
        session_dir = self.base_dir / session_id
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
    Tracks progress and sends updates to both state and Slack
    """
    
    def __init__(self, slack_manager: SlackManager = None):
        self.slack_manager = slack_manager
        self._last_message = None  # Simple debounce mechanism
    
    def update(self, state: PipelineState, message: str, stage: str = None, send_to_slack: bool = True):
        """Update progress in state and optionally send to Slack"""
        # Create full message for deduplication
        full_message = f"{stage}: {message}" if stage else message
        
        # Skip if this is the same message as last time (simple debounce)
        if full_message == self._last_message:
            return
        self._last_message = full_message
        
        # Send to Slack if enabled and manager available
        if send_to_slack and self.slack_manager and state.chat_session:
            if stage:
                self.slack_manager.send_progress_update(state.chat_session, stage, message)
            else:
                self.slack_manager.send_message(state.chat_session, f"⏳ {message}")
        
        # Console logging (single source of truth)
        timestamp = datetime.now().strftime("%H:%M:%S")
        agent_info = f" [{state.current_agent}]" if state.current_agent else ""
        print(f"[{timestamp}]{agent_info} {message}")
        
        # Update state (after logging to avoid duplicate console output)
        if stage:
            state.update_progress(f"{stage}: {message}", state.current_agent)
        else:
            state.update_progress(message, state.current_agent)


class UserDirectoryManager:
    """Manages user directory structure like in ModelBuildingAgent"""
    
    def __init__(self, base_data_dir: str = "user_data"):
        self.base_data_dir = base_data_dir
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
    
    def _get_user_thread_dir(self, user_id: str) -> str:
        """Get directory path for specific user thread"""
        user, thread_ts = self._get_thread_id(user_id)
        return os.path.join(self.base_data_dir, user, thread_ts)
    
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
            
        if model_name.startswith("gpt-") and ChatOpenAI:
            return ChatOpenAI(
                model=model_name,
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
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
artifact_manager = ArtifactManager()
progress_tracker = ProgressTracker(slack_manager)
execution_agent = ExecutionAgent()
user_directory_manager = UserDirectoryManager()


def initialize_toolbox(slack_token: str = None, artifacts_dir: str = None, user_data_dir: str = None):
    """Initialize global toolbox with custom configuration"""
    global slack_manager, artifact_manager, progress_tracker, execution_agent, user_directory_manager
    
    if slack_token:
        slack_manager = SlackManager(slack_token)
        progress_tracker = ProgressTracker(slack_manager)
    
    if artifacts_dir:
        artifact_manager = ArtifactManager(artifacts_dir)
    
    if user_data_dir:
        user_directory_manager = UserDirectoryManager(user_data_dir)
    
    print("🧰 Global toolbox initialized")
    print(f"   Slack: {'✅ Enabled' if slack_manager.client else '❌ Disabled'}")
    print(f"   Artifacts: {artifact_manager.base_dir}")
    print(f"   User Data: {user_directory_manager.base_data_dir}")
    print(f"   Execution: ✅ Ready with fallback support")
