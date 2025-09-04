#!/usr/bin/env python3
"""
Slack-Compatible Sequential Preprocessing Agent
Adapted from preprocessing_agent_impl.py to work with Slack integration
Using feature_selection_agent_impl.py as template for Slack patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import json
import os
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import tempfile
import time
import warnings
import logging
from dataclasses import dataclass, field
from datetime import datetime

# Slack imports (same as feature selection agent)
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Setup logging (same pattern as feature selection agent)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import the original preprocessing components
from preprocessing_agent_impl import (
    PreprocessingPhase,
    SequentialState,
    get_llm_from_state,
    classify_user_intent_with_llm,
    # Import the actual analysis functions from your implementation
    initialize_dataset_analysis,
    generate_overview_summary,
    analyze_outliers_with_llm,
    analyze_missing_values_with_llm,
    analyze_encoding_with_llm,
    analyze_transformations_with_llm,
    # Import processing functions
    apply_outliers_treatment,
    apply_missing_values_treatment,
    apply_encoding_treatment,
    apply_transformations_treatment,
    get_current_data_state
)

@dataclass
class PreprocessingSession:
    """Preprocessing session state - adapted from feature selection UserSession pattern"""
    # File and target info
    file_path: str
    file_name: str
    user_id: Optional[str] = None
    target_column: Optional[str] = None
    
    # Data states
    original_df: Optional[pd.DataFrame] = None
    current_df: Optional[pd.DataFrame] = None
    
    # Preprocessing workflow state
    current_phase: str = PreprocessingPhase.OVERVIEW
    completed_phases: List[str] = field(default_factory=list)
    phase_results: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis results
    column_analysis: Dict[str, Dict] = field(default_factory=dict)
    outlier_analysis: Dict[str, Dict] = field(default_factory=dict)
    missing_analysis: Dict[str, Dict] = field(default_factory=dict)
    encoding_analysis: Dict[str, Dict] = field(default_factory=dict)
    transformation_analysis: Dict[str, Dict] = field(default_factory=dict)
    
    # User interaction state
    phase: str = "need_target"  # need_target, analyzing, waiting_input, completed
    last_menu: Optional[str] = None
    pending_message: Optional[str] = None
    thread_ts: Optional[str] = None
    
    # Configuration
    model_name: str = os.environ.get("DEFAULT_MODEL", "gpt-4o")
    missing_threshold: float = 50.0
    outlier_threshold: float = 10.0
    high_cardinality_threshold: int = 50
    onehot_top_categories: int = 10
    
    # User messages and approvals
    user_messages: List[str] = field(default_factory=list)
    user_overrides: Dict[str, Dict] = field(default_factory=dict)
    phase_approved: bool = False
    is_query: bool = False
    query_response: Optional[str] = None

class PreprocessingMenuGenerator:
    """Generate interactive menus for preprocessing - adapted from feature selection MenuGenerator"""
    
    @staticmethod
    def generate_main_menu(session: PreprocessingSession) -> str:
        """Generate the main preprocessing menu"""
        current_phase_name = session.current_phase.replace('_', ' ').title()
        completed_count = len(session.completed_phases)
        
        menu = f"""🧹 **Sequential Preprocessing Agent**

📊 **Current Dataset:** {session.current_df.shape if session.current_df is not None else 'Loading...'}
📈 **Progress:** {current_phase_name} ({completed_count} phases completed)

**🔄 Preprocessing Phases:**
• `Overview` - Dataset analysis and summary
• `Outliers` - Detect and handle outliers  
• `Missing Values` - Handle missing data
• `Encoding` - Categorical variable encoding
• `Transformations` - Feature transformations
• `Completion` - Finalize preprocessing

**💬 Your Options:**
• `proceed` or `yes` - Continue with recommended approach
• `skip` - Skip current phase
• `modify [details]` - Change the approach
• `explain` or `what` - Get more information about current phase
• `summary` - Show current preprocessing strategies

**❓ Ask Me Anything:**
• Phase questions: `what are outliers?`, `explain missing value strategies`
• Current state: `show current strategies`, `what's the current plan`
• Navigation: `jump to missing values`, `go to encoding phase`

💬 **What would you like to do next?**"""
        
        return menu
    
    @staticmethod
    def generate_phase_summary(session: PreprocessingSession, phase: str) -> str:
        """Generate summary for a specific phase"""
        if phase == PreprocessingPhase.OVERVIEW:
            return f"""📊 **Dataset Overview Phase**

Current dataset: {session.current_df.shape if session.current_df is not None else 'Not loaded'}
Target column: {session.target_column or 'Not specified'}

This phase analyzes your dataset structure, identifies data types, and provides initial insights."""

        elif phase == PreprocessingPhase.OUTLIERS:
            return f"""🎯 **Outliers Detection Phase**

This phase identifies and handles outliers in your numeric columns using statistical methods.

**Options:**
• Remove outliers beyond 3 standard deviations
• Cap outliers using IQR method
• Keep outliers (no changes)"""

        elif phase == PreprocessingPhase.MISSING_VALUES:
            return f"""🔍 **Missing Values Handling Phase**

This phase handles missing data in your dataset.

**Options:**
• Drop columns with >50% missing values
• Impute numeric columns (mean/median/mode)
• Impute categorical columns (mode/forward fill)"""

        elif phase == PreprocessingPhase.ENCODING:
            return f"""🔤 **Categorical Encoding Phase**

This phase encodes categorical variables for machine learning.

**Options:**
• One-hot encoding for low cardinality
• Label encoding for ordinal variables
• Target encoding for high cardinality"""

        elif phase == PreprocessingPhase.TRANSFORMATIONS:
            return f"""⚡ **Feature Transformations Phase**

This phase applies mathematical transformations to improve model performance.

**Options:**
• Log transformation for skewed data
• Scaling/normalization
• Polynomial features"""

        return f"Phase: {phase}"

class SlackPreprocessingBot:
    """Main Slack preprocessing bot - adapted from AgenticFeatureSelectionBot pattern"""
    
    def __init__(self):
        self.app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
        self.users: Dict[str, PreprocessingSession] = {}
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup all Slack event handlers - same pattern as feature selection"""
        
        # File upload handler
        @self.app.event("file_shared")
        def handle_file_upload(event, say, client):
            self.handle_file_upload(event, say, client)
        
        # Message handler - check if bot is mentioned
        @self.app.event("message")
        def handle_message_event(body, say, logger):
            event = body["event"]
            text = event.get("text", "")
            
            # Check if this message mentions the bot
            import re
            if re.search(r'<@[^>]+>', text):
                # Extract bot ID and check if it's the correct bot
                bot_match = re.search(r'<@([^>]+)>', text)
                if bot_match:
                    bot_id = bot_match.group(1)
                    # Process the mention
                    self.handle_app_mention(body, say, logger)
    
    def handle_file_upload(self, event, say, client):
        """Handle CSV file uploads - adapted from feature selection pattern"""
        try:
            user_id = event.get("user_id") or event.get("user")
            file_id = event["file_id"]
            
            logger.info(f"📁 FILE UPLOAD: User={user_id}, File={file_id}")
            
            # Get file info
            file_info = client.files_info(file=file_id)["file"]
            
            if not file_info["name"].endswith('.csv'):
                say("❌ Please upload a CSV file for preprocessing.")
                return
            
            # Show immediate welcome message
            say(f"👋 **Welcome to Sequential Preprocessing!**\n🔄 **Processing your file:** {file_info['name']}...")
            
            # Download file
            import requests
            headers = {"Authorization": f"Bearer {os.environ.get('SLACK_BOT_TOKEN')}"}
            response = requests.get(file_info["url_private_download"], headers=headers)
            
            # Save file temporarily
            file_path = f"temp_{user_id}_{file_info['name']}"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Show data scanning message
            say("🔍 **Scanning dataset and preparing for preprocessing...**")
            
            # Load and analyze data
            df = pd.read_csv(file_path)
            
            # Create preprocessing session
            session = PreprocessingSession(
                file_path=file_path,
                file_name=file_info["name"],
                user_id=user_id,
                original_df=df.copy(),
                current_df=df.copy(),
                model_name=os.environ.get("DEFAULT_MODEL", "gpt-4o")
            )
            
            self.users[user_id] = session
            logger.info(f"✅ SESSION CREATED: User={user_id}, File={file_info['name']}, Shape={df.shape}")
            
            # Show target column selection
            columns = list(df.columns)
            say(f"""📁 **File uploaded:** {file_info['name']}
📊 **Dataset:** {df.shape[0]:,} rows × {df.shape[1]} columns

🎯 **Target Column Selection**

Available columns: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}

📝 **Please specify your target column:**
• Type the column name directly (e.g., `target`)  
• Or use: `target column_name`

**Example:** `target` or `target my_target_column`""")
                
        except Exception as e:
            say(f"❌ Error processing file: {str(e)}")
            logger.exception(f"Error in file upload: {e}")
    
    def handle_message(self, message, say, thread_ts=None):
        """Handle all user messages - adapted from feature selection pattern"""
        user_id = message["user"]
        text = message["text"].strip()
        
        logger.info(f"📨 RECEIVED MESSAGE | User: {user_id} | Text: '{text}' | Thread: {thread_ts}")
        
        # Skip bot messages
        if message.get("bot_id"):
            logger.info(f"⏭️ SKIPPING BOT MESSAGE | Bot ID: {message.get('bot_id')}")
            return
        
        # Check if user has an active session
        if user_id not in self.users:
            logger.info(f"❌ NO ACTIVE SESSION | User: {user_id} | No file uploaded yet")
            say("👋 Please upload a CSV file first to start preprocessing!")
            return
        
        session = self.users[user_id]
        logger.info(f"✅ ACTIVE SESSION FOUND | User: {user_id} | Phase: {session.phase}")
        
        # Store thread_ts in session
        session.thread_ts = thread_ts
        
        try:
            if session.phase == "need_target":
                self.handle_target_selection(session, text, say)
            elif session.phase == "waiting_input":
                self.handle_preprocessing_request(session, text, say)
            else:
                # Default to preprocessing request handling
                self.handle_preprocessing_request(session, text, say)
                
        except Exception as e:
            logger.exception(f"Error handling message: {e}")
            say(f"❌ Error processing your request: {str(e)}")
    
    def handle_target_selection(self, session: PreprocessingSession, text: str, say):
        """Handle target column selection"""
        # Extract target column from user input
        text_lower = text.lower().strip()
        
        # Handle "target column_name" format
        if text_lower.startswith("target "):
            target_column = text[7:].strip()
        else:
            # Assume the entire text is the column name
            target_column = text.strip()
        
        # Validate target column exists
        if target_column in session.current_df.columns:
            session.target_column = target_column
            session.phase = "waiting_input"
            
            say(f"""✅ **Target column set:** `{target_column}`

🚀 **Ready to start preprocessing!**

{PreprocessingMenuGenerator.generate_main_menu(session)}""")
            
            session.last_menu = PreprocessingMenuGenerator.generate_main_menu(session)
            
        else:
            available_cols = list(session.current_df.columns)
            say(f"""❌ **Column '{target_column}' not found.**

**Available columns:** {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}

Please specify a valid column name.""")
    
    def handle_preprocessing_request(self, session: PreprocessingSession, text: str, say):
        """Handle preprocessing requests using original agent logic"""
        try:
            # Convert session to SequentialState for compatibility with original functions
            sequential_state = self._convert_to_sequential_state(session)
            
            # Use original intent classification
            intent_result = classify_user_intent_with_llm(text, sequential_state.current_phase, [], sequential_state)
            intent = intent_result.get("intent", "query")
            
            logger.info(f"🎯 INTENT CLASSIFIED | User: {session.user_id} | Intent: {intent}")
            
            if intent == "proceed":
                self.handle_proceed(session, say)
            elif intent == "skip":
                self.handle_skip(session, say)
            elif intent == "override":
                self.handle_override(session, text, say)
            elif intent == "query":
                self.handle_query(session, text, say)
            elif intent == "summary":
                self.handle_summary(session, say)
            elif intent == "navigate":
                self.handle_navigate(session, text, say)
            elif intent == "exit":
                self.handle_exit(session, say)
            else:
                # Default to showing menu
                say(session.last_menu or PreprocessingMenuGenerator.generate_main_menu(session))
                
        except Exception as e:
            logger.exception(f"Error in preprocessing request: {e}")
            say(f"❌ Error processing request: {str(e)}")
    
    def handle_proceed(self, session: PreprocessingSession, say):
        """Handle proceed command - execute current phase"""
        try:
            current_phase = session.current_phase
            say(f"⚙️ **Processing {current_phase.replace('_', ' ').title()} phase...**")
            
            # Convert to sequential state
            sequential_state = self._convert_to_sequential_state(session)
            
            if current_phase == PreprocessingPhase.OVERVIEW:
                # Run overview analysis using your actual functions
                updated_state = initialize_dataset_analysis(sequential_state)
                overview_summary = generate_overview_summary(updated_state)
                
                # Update session with results
                session.column_analysis = updated_state.column_analysis
                session.phase_results[current_phase] = updated_state.phase_results.get(current_phase, {})
                
                say(f"""📊 **Dataset Overview Complete**

{overview_summary}

**Next Phase:** Outlier Detection
""")
                
                # Move to next phase
                session.current_phase = PreprocessingPhase.OUTLIERS
                session.completed_phases.append(PreprocessingPhase.OVERVIEW)
                
            elif current_phase == PreprocessingPhase.OUTLIERS:
                # Run outlier analysis using your actual functions
                analysis_result = analyze_outliers_with_llm(sequential_state)
                
                # Apply outlier treatment
                current_df = get_current_data_state(sequential_state)
                processed_df = apply_outliers_treatment(current_df, analysis_result)
                
                # Update session
                session.outlier_analysis = analysis_result
                session.current_df = processed_df
                session.phase_results[current_phase] = analysis_result
                
                say(f"""🎯 **Outlier Detection Complete**

**Outliers handled in numeric columns**
• Dataset shape: {session.current_df.shape[0]:,} rows × {session.current_df.shape[1]} columns

**Next Phase:** Missing Values Handling
""")
                
                # Move to next phase
                session.current_phase = PreprocessingPhase.MISSING_VALUES
                session.completed_phases.append(PreprocessingPhase.OUTLIERS)
                
            elif current_phase == PreprocessingPhase.MISSING_VALUES:
                # Run missing values analysis using your actual functions
                analysis_result = analyze_missing_values_with_llm(sequential_state)
                
                # Apply missing values treatment
                current_df = get_current_data_state(sequential_state)
                processed_df = apply_missing_values_treatment(current_df, analysis_result)
                
                # Update session
                session.missing_analysis = analysis_result
                session.current_df = processed_df
                session.phase_results[current_phase] = analysis_result
                
                say(f"""🔍 **Missing Values Handling Complete**

**Missing data processed**
• Dataset shape: {session.current_df.shape[0]:,} rows × {session.current_df.shape[1]} columns

**Next Phase:** Categorical Encoding
""")
                
                # Move to next phase
                session.current_phase = PreprocessingPhase.ENCODING
                session.completed_phases.append(PreprocessingPhase.MISSING_VALUES)
                
            elif current_phase == PreprocessingPhase.ENCODING:
                # Run encoding analysis using your actual functions
                analysis_result = analyze_encoding_with_llm(sequential_state)
                
                # Apply encoding treatment
                current_df = get_current_data_state(sequential_state)
                processed_df = apply_encoding_treatment(current_df, analysis_result)
                
                # Update session
                session.encoding_analysis = analysis_result
                session.current_df = processed_df
                session.phase_results[current_phase] = analysis_result
                
                say(f"""🔤 **Categorical Encoding Complete**

**Categorical variables encoded**
• Dataset shape: {session.current_df.shape[0]:,} rows × {session.current_df.shape[1]} columns

**Next Phase:** Feature Transformations
""")
                
                # Move to next phase
                session.current_phase = PreprocessingPhase.TRANSFORMATIONS
                session.completed_phases.append(PreprocessingPhase.ENCODING)
                
            elif current_phase == PreprocessingPhase.TRANSFORMATIONS:
                # Run transformations analysis using your actual functions
                analysis_result = analyze_transformations_with_llm(sequential_state)
                
                # Apply transformations treatment
                current_df = get_current_data_state(sequential_state)
                processed_df = apply_transformations_treatment(current_df, analysis_result)
                
                # Update session
                session.transformation_analysis = analysis_result
                session.current_df = processed_df
                session.phase_results[current_phase] = analysis_result
                
                say(f"""⚡ **Feature Transformations Complete**

**Transformations applied**
• Final dataset shape: {session.current_df.shape[0]:,} rows × {session.current_df.shape[1]} columns

**Preprocessing Complete!** ✅
""")
                
                # Move to completion
                session.current_phase = PreprocessingPhase.COMPLETION
                session.completed_phases.append(PreprocessingPhase.TRANSFORMATIONS)
                session.phase = "completed"
                
                # Show final summary
                self.generate_final_summary(session, say)
                return
            
            # Show next phase menu
            if session.current_phase != PreprocessingPhase.COMPLETION:
                next_menu = PreprocessingMenuGenerator.generate_main_menu(session)
                say(next_menu)
                session.last_menu = next_menu
                
        except Exception as e:
            logger.exception(f"Error in proceed: {e}")
            say(f"❌ Error processing phase: {str(e)}")
    
    def handle_skip(self, session: PreprocessingSession, say):
        """Handle skip command - skip current phase"""
        current_phase = session.current_phase
        say(f"⏭️ **Skipping {current_phase.replace('_', ' ').title()} phase**")
        
        # Move to next phase without processing
        if current_phase == PreprocessingPhase.OVERVIEW:
            session.current_phase = PreprocessingPhase.OUTLIERS
        elif current_phase == PreprocessingPhase.OUTLIERS:
            session.current_phase = PreprocessingPhase.MISSING_VALUES
        elif current_phase == PreprocessingPhase.MISSING_VALUES:
            session.current_phase = PreprocessingPhase.ENCODING
        elif current_phase == PreprocessingPhase.ENCODING:
            session.current_phase = PreprocessingPhase.TRANSFORMATIONS
        elif current_phase == PreprocessingPhase.TRANSFORMATIONS:
            session.current_phase = PreprocessingPhase.COMPLETION
            session.phase = "completed"
            self.generate_final_summary(session, say)
            return
        
        session.completed_phases.append(current_phase)
        
        # Show next phase menu
        next_menu = PreprocessingMenuGenerator.generate_main_menu(session)
        say(next_menu)
        session.last_menu = next_menu
    
    def handle_query(self, session: PreprocessingSession, text: str, say):
        """Handle query/explanation requests"""
        current_phase = session.current_phase
        phase_summary = PreprocessingMenuGenerator.generate_phase_summary(session, current_phase)
        say(phase_summary)
    
    def handle_summary(self, session: PreprocessingSession, say):
        """Handle summary requests"""
        completed_phases = len(session.completed_phases)
        total_phases = 5  # Overview, Outliers, Missing, Encoding, Transformations
        
        summary = f"""📋 **Preprocessing Progress Summary**

**Current Phase:** {session.current_phase.replace('_', ' ').title()}
**Completed Phases:** {completed_phases}/{total_phases}
**Dataset Shape:** {session.current_df.shape if session.current_df is not None else 'Not loaded'}
**Target Column:** {session.target_column or 'Not set'}

**Completed Steps:**
{chr(10).join([f"✅ {phase.replace('_', ' ').title()}" for phase in session.completed_phases])}

**Remaining Steps:**
{chr(10).join([f"⏳ {phase.replace('_', ' ').title()}" for phase in [PreprocessingPhase.OVERVIEW, PreprocessingPhase.OUTLIERS, PreprocessingPhase.MISSING_VALUES, PreprocessingPhase.ENCODING, PreprocessingPhase.TRANSFORMATIONS] if phase not in session.completed_phases and phase != session.current_phase])}
"""
        say(summary)
    
    def handle_override(self, session: PreprocessingSession, text: str, say):
        """Handle override/modification requests"""
        say(f"🔧 **Modification Request Noted**\n\nRequest: {text}\n\nCustom modifications will be applied in the current phase.")
        # Store the override for later use
        session.user_overrides[session.current_phase] = {"request": text, "timestamp": datetime.now().isoformat()}
    
    def handle_navigate(self, session: PreprocessingSession, text: str, say):
        """Handle navigation requests"""
        text_lower = text.lower()
        
        if "overview" in text_lower:
            session.current_phase = PreprocessingPhase.OVERVIEW
        elif "outlier" in text_lower:
            session.current_phase = PreprocessingPhase.OUTLIERS
        elif "missing" in text_lower:
            session.current_phase = PreprocessingPhase.MISSING_VALUES
        elif "encoding" in text_lower:
            session.current_phase = PreprocessingPhase.ENCODING
        elif "transformation" in text_lower:
            session.current_phase = PreprocessingPhase.TRANSFORMATIONS
        
        say(f"🧭 **Navigated to {session.current_phase.replace('_', ' ').title()} phase**")
        
        # Show current phase menu
        menu = PreprocessingMenuGenerator.generate_main_menu(session)
        say(menu)
        session.last_menu = menu
    
    def handle_exit(self, session: PreprocessingSession, say):
        """Handle exit requests"""
        say("👋 **Preprocessing session ended.**\n\nThank you for using Sequential Preprocessing Agent!")
        session.phase = "completed"
    
    def generate_final_summary(self, session: PreprocessingSession, say):
        """Generate final preprocessing summary"""
        original_shape = session.original_df.shape if session.original_df is not None else (0, 0)
        final_shape = session.current_df.shape if session.current_df is not None else (0, 0)
        
        summary = f"""🎉 **Preprocessing Complete!**

**📊 Final Results:**
• Original dataset: {original_shape[0]:,} rows × {original_shape[1]} columns
• Processed dataset: {final_shape[0]:,} rows × {final_shape[1]} columns
• Target column: {session.target_column}

**✅ Completed Phases:**
{chr(10).join([f"• {phase.replace('_', ' ').title()}" for phase in session.completed_phases])}

**📁 Your preprocessed data is ready for feature selection and model building!**

Would you like to proceed to feature selection or download the processed dataset?"""
        
        say(summary)
    
    def _convert_to_sequential_state(self, session: PreprocessingSession) -> SequentialState:
        """Convert PreprocessingSession to SequentialState for compatibility"""
        # Create a temporary file for the current dataframe
        temp_file = f"temp_sequential_{session.user_id}.csv"
        session.current_df.to_csv(temp_file, index=False)
        
        return SequentialState(
            df_path=temp_file,
            target_column=session.target_column or "target",
            model_name=session.model_name,
            current_phase=session.current_phase,
            completed_phases=session.completed_phases,
            phase_results=session.phase_results,
            column_analysis=session.column_analysis,
            outlier_analysis=session.outlier_analysis,
            missing_analysis=session.missing_analysis,
            encoding_analysis=session.encoding_analysis,
            transformation_analysis=session.transformation_analysis,
            user_messages=session.user_messages,
            user_overrides=session.user_overrides,
            missing_threshold=session.missing_threshold,
            outlier_threshold=session.outlier_threshold,
            high_cardinality_threshold=session.high_cardinality_threshold,
            onehot_top_categories=session.onehot_top_categories,
            df=session.current_df
        )
    
    def handle_app_mention(self, body, say, logger):
        """Handle app mentions - adapted from feature selection pattern"""
        try:
            event = body["event"]
            user_id = event["user"]
            text = event.get("text", "")
            
            # Clean the text (remove bot mention)
            import re
            cleaned_text = re.sub(r'<@[^>]+>', '', text).strip()
            
            logger.info(f"🔍 DEBUG: Cleaned text: '{cleaned_text}'")
            
            if not cleaned_text:
                if user_id not in self.users:
                    say("👋 How can I help? Upload a CSV file to start preprocessing.")
                    return
                else:
                    session = self.users[user_id]
                    if session.phase == "need_target":
                        say("🎯 Please specify your target column first.")
                    elif session.phase == "waiting_input":
                        if session.last_menu:
                            say(session.last_menu)
                        else:
                            menu = PreprocessingMenuGenerator.generate_main_menu(session)
                            say(menu)
                    else:
                        say("👋 How can I help? Upload a CSV file to start preprocessing.")
                return
            
            # Process as regular message
            fake_message = {"user": user_id, "text": cleaned_text}
            thread_ts = body["event"].get("thread_ts") or body["event"].get("ts")
            self.handle_message(fake_message, say, thread_ts)
            
        except Exception as e:
            logger.exception(f"Error handling app mention: {e}")
            say("❌ Error processing your mention. Please try again.")
    
    def run(self):
        """Start the bot"""
        handler = SocketModeHandler(self.app, os.environ["SLACK_APP_TOKEN"])
        print("🚀 Slack-Compatible Preprocessing Agent Started!")
        print("📡 Socket Mode activated - Bot is ready!")
        handler.start()

# Function to create and run the Slack preprocessing bot
def create_slack_preprocessing_bot():
    """Create and return Slack preprocessing bot instance"""
    return SlackPreprocessingBot()

def run_slack_preprocessing_agent():
    """Run the Slack preprocessing agent"""
    bot = create_slack_preprocessing_bot()
    bot.run()

if __name__ == "__main__":
    run_slack_preprocessing_agent()
