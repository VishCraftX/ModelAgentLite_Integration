#!/usr/bin/env python3
"""
Slack Bot Integration for Multi-Agent ML Pipeline
Provides Slack interface for the integrated ML system
"""

import os
import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict

try:
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler
    SLACK_BOLT_AVAILABLE = True
except ImportError:
    App = None
    SocketModeHandler = None
    SLACK_BOLT_AVAILABLE = False

from langgraph_pipeline import get_pipeline
from config import (
    SLACK_BOT_TOKEN, SLACK_APP_TOKEN, 
    SUPPORTED_FILE_TYPES, validate_config, print_config
)


class SlackMLBot:
    """
    Slack bot for the Multi-Agent ML Pipeline
    """
    
    def __init__(self):
        # Initialize Slack app
        if not SLACK_BOLT_AVAILABLE:
            raise ImportError("slack-bolt is required for Slack integration. Install with: pip install slack-bolt")
        self.app = App(token=SLACK_BOT_TOKEN)
        
        # Initialize ML pipeline
        self.ml_pipeline = get_pipeline(
            slack_token=SLACK_BOT_TOKEN,
            enable_persistence=True
        )
        
        # Session management
        self.chat_sessions = defaultdict(list)
        self.session_data = defaultdict(pd.DataFrame)
        self.user_file_history = defaultdict(list)
        
        # Set up event handlers
        self._setup_handlers()
        
        print("🤖 Slack ML Bot initialized")
    
    def _setup_handlers(self):
        """Set up Slack event handlers"""
        
        @self.app.event("app_mention")
        @self.app.event("message")
        def handle_message_events(body, say, event, context):
            """Handle Slack messages with file attachment support"""
            
            # Debug logging
            timestamp = datetime.now().strftime("%H:%M:%S")
            user_id = event.get('user', 'unknown')
            text = event.get('text', '')
            files = event.get('files', [])
            
            print(f"💬 SLACK MSG [{timestamp}] User: {user_id} | Text: {text[:50]}{'...' if len(text) > 50 else ''} | Files: {len(files)}")
            
            # Get bot's user ID from context
            bot_user_id = context["bot_user_id"]
            
            # Check if we should handle this message
            if not self._should_handle_message(event, bot_user_id):
                return
            
            # Extract message details
            user_id = event['user']
            channel = event['channel']
            thread_ts = event.get('thread_ts', event['ts'])
            session_id = self._get_session_id(user_id, thread_ts)
            
            # Register session with SlackManager for proper channel tracking
            self.ml_pipeline.slack_manager.register_session(session_id, channel, thread_ts)
            
            # Clean the message text (remove bot mention)
            if f"<@{bot_user_id}>" in text:
                text = text.replace(f"<@{bot_user_id}>", "").strip()
            
            # Handle file attachments first
            if files:
                self._handle_file_attachments(files, session_id, say, thread_ts)
            
            # Handle text query if present
            if text.strip():
                self._handle_text_query(text, session_id, say, thread_ts)
            elif not files:
                # No text and no files
                say("👋 Hi! I'm your AI ML pipeline assistant. Upload a data file and ask me to build models, preprocess data, or analyze your data!", 
                    thread_ts=thread_ts)
        
        @self.app.command("/pipeline_status")
        def pipeline_status_command(ack, respond, command):
            """Show pipeline status for the user"""
            ack()
            
            user_id = command['user_id']
            
            # Get user sessions
            user_sessions = [sid for sid in self.ml_pipeline.list_sessions() if sid.startswith(user_id)]
            
            if not user_sessions:
                respond("🤖 No active pipeline sessions found. Upload data and start asking questions!")
                return
            
            status_parts = ["🤖 *Pipeline Status:*\n"]
            
            for session_id in user_sessions[-3:]:  # Show last 3 sessions
                session_status = self.ml_pipeline.get_session_status(session_id)
                if session_status["exists"]:
                    data_summary = session_status["data_summary"]
                    status_parts.append(f"• Session `{session_id}`:")
                    status_parts.append(f"  - Raw data: {'✅' if data_summary['has_raw_data'] else '❌'}")
                    status_parts.append(f"  - Cleaned data: {'✅' if data_summary['has_cleaned_data'] else '❌'}")
                    status_parts.append(f"  - Features selected: {'✅' if data_summary['has_selected_features'] else '❌'}")
                    status_parts.append(f"  - Model trained: {'✅' if data_summary['has_trained_model'] else '❌'}")
            
            respond('\n'.join(status_parts))
        
        @self.app.command("/help")
        def help_command(ack, respond, command):
            """Show comprehensive help information"""
            ack()
            
            help_text = """🤖 *Multi-Agent ML Pipeline - AI Assistant*

*🚀 Getting Started:*
1. Upload a data file (CSV, Excel, JSON, TSV)
2. Ask me to analyze, preprocess, or build models with your data

*🔧 Example Commands:*
• `clean and preprocess my data` - Data preprocessing
• `select the best features` - Feature selection
• `train a machine learning model` - Model building
• `build a complete ML pipeline` - Full end-to-end pipeline
• `show data summary` - Analyze uploaded data
• `make predictions` - Use trained models

*📁 Supported File Formats:*
• CSV files (.csv)
• Excel files (.xlsx, .xls)  
• JSON files (.json)
• Tab-separated files (.tsv)

*🎯 Pipeline Stages:*
1. **Data Preprocessing** - Cleaning, missing values, outliers
2. **Feature Selection** - IV analysis, correlation, PCA
3. **Model Building** - Training, evaluation, predictions

*💡 Slash Commands:*
• `/pipeline_status` - Check your pipeline status
• `/help` - Show this help

*🏗️ Architecture:*
This bot uses LangGraph with specialized agents that intelligently route your requests and execute ML tasks.

Just upload your data and start asking questions in natural language! 🎉"""
            
            respond(help_text)
    
    def _should_handle_message(self, event: Dict[str, Any], bot_user_id: str) -> bool:
        """Determine if the bot should handle this message"""
        # Ignore bot's own messages
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return False

        # Get message properties
        channel_type = event.get("channel_type", "")
        text = event.get("text", "")
        thread_ts = event.get("thread_ts")
        files = event.get("files", [])

        # Case 1: Direct mention
        if f"<@{bot_user_id}>" in text:
            return True

        # Case 2: Direct message
        if channel_type == "im":
            return True

        # Case 3: Thread participation
        if thread_ts and thread_ts in self.chat_sessions:
            return True
        
        # Case 4: File attachments (always handle files)
        if files:
            return True

        return False
    
    def _get_session_id(self, user_id: str, thread_ts: Optional[str] = None) -> str:
        """Generate a unique session ID for tracking user conversations"""
        if thread_ts:
            return f"{user_id}_{thread_ts}"
        return user_id
    
    def _handle_file_attachments(self, files: List[Dict], session_id: str, say, thread_ts: str):
        """Handle file attachments"""
        say("📎 Processing uploaded file(s)...", thread_ts=thread_ts)
        
        for file_info in files:
            file_name = file_info.get('name', 'unknown_file')
            
            # Download and process file
            file_content = self._download_file_from_slack(file_info)
            if not file_content:
                say(f"❌ Failed to download file `{file_name}`", thread_ts=thread_ts)
                continue
            
            # Load as DataFrame
            data_frame = self._load_dataframe_from_file(file_content, file_name)
            if data_frame is None:
                say(f"❌ Could not load `{file_name}` as a data file. Supported formats: {', '.join(SUPPORTED_FILE_TYPES).upper()}", 
                    thread_ts=thread_ts)
                continue
            
            # Store the data for this session
            self.session_data[session_id] = data_frame
            self.user_file_history[session_id.split('_')[0]].append({
                'name': file_name,
                'timestamp': datetime.now(),
                'shape': data_frame.shape,
                'session_id': session_id
            })
            
            # Load data into the ML pipeline
            self.ml_pipeline.load_data(data_frame, session_id)
            
            # Send confirmation with data info
            file_info_msg = self._format_dataframe_info(data_frame, file_name)
            say(file_info_msg, thread_ts=thread_ts)
    
    def _handle_text_query(self, text: str, session_id: str, say, thread_ts: str):
        """Handle text queries"""
        try:
            # Working message
            say("🤔 Working on your request...", thread_ts=thread_ts)
            
            # Create progress callback for real-time updates
            def progress_callback(message: str, stage: str = ""):
                """Send progress updates to Slack"""
                if stage:
                    say(f"⏳ *{stage}:* {message}", thread_ts=thread_ts)
                else:
                    say(f"⏳ {message}", thread_ts=thread_ts)
            
            # Process query through ML pipeline
            result = self.ml_pipeline.process_query(
                text, 
                session_id, 
                progress_callback=progress_callback
            )
            
            # Send main response
            response_text = result["response"]
            say(response_text, thread_ts=thread_ts)
            
            # Send additional information if available
            if result.get("data_summary"):
                summary = result["data_summary"]
                if any(summary.values()):  # If any data is available
                    summary_text = self._format_pipeline_summary(summary)
                    say(summary_text, thread_ts=thread_ts)
            
        except Exception as e:
            error_msg = f"❌ Sorry, I encountered an error processing your request: {str(e)}"
            say(error_msg, thread_ts=thread_ts)
            print(f"Error processing query: {e}")
    
    def _download_file_from_slack(self, file_info: Dict[str, Any]) -> Optional[bytes]:
        """Download file content from Slack"""
        try:
            file_url = file_info.get("url_private_download")
            if not file_url:
                return None
                
            headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
            response = requests.get(file_url, headers=headers)
            response.raise_for_status()
            
            return response.content
        except Exception as e:
            print(f"Failed to download file: {e}")
            return None
    
    def _load_dataframe_from_file(self, file_content: bytes, file_name: str) -> Optional[pd.DataFrame]:
        """Load a pandas DataFrame from file content"""
        try:
            file_extension = file_name.lower().split('.')[-1]
            
            if file_extension == 'csv':
                # Try different encodings for CSV files
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        content_str = file_content.decode(encoding)
                        return pd.read_csv(StringIO(content_str))
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode CSV file with any supported encoding")
                
            elif file_extension in ['xlsx', 'xls']:
                return pd.read_excel(BytesIO(file_content))
                
            elif file_extension == 'json':
                content_str = file_content.decode('utf-8')
                return pd.read_json(StringIO(content_str))
                
            elif file_extension == 'tsv':
                content_str = file_content.decode('utf-8')
                return pd.read_csv(StringIO(content_str), sep='\t')
                
            else:
                # Try to read as CSV by default
                content_str = file_content.decode('utf-8')
                return pd.read_csv(StringIO(content_str))
                
        except Exception as e:
            print(f"Failed to load DataFrame from {file_name}: {e}")
            return None
    
    def _format_dataframe_info(self, data_frame: pd.DataFrame, file_name: str) -> str:
        """Format DataFrame information for Slack display"""
        
        # Analyze column types
        numeric_cols = data_frame.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        categorical_cols = data_frame.select_dtypes(include=['object', 'category']).columns
        boolean_cols = data_frame.select_dtypes(include=['bool']).columns
        datetime_cols = data_frame.select_dtypes(include=['datetime64']).columns
        
        info_parts = [
            f"📊 *File loaded successfully: `{file_name}`*",
            f"• *Shape:* {data_frame.shape[0]:,} rows × {data_frame.shape[1]} columns",
        ]
        
        # Add column type analysis
        column_summary = []
        if len(numeric_cols) > 0:
            column_summary.append(f"📊 {len(numeric_cols)} numeric")
        if len(categorical_cols) > 0:
            column_summary.append(f"🏷️ {len(categorical_cols)} categorical")
        if len(boolean_cols) > 0:
            column_summary.append(f"✅ {len(boolean_cols)} boolean")
        if len(datetime_cols) > 0:
            column_summary.append(f"📅 {len(datetime_cols)} datetime")
        
        if column_summary:
            info_parts.append(f"• *Features:* {', '.join(column_summary)}")
        
        # Add target column detection
        target_candidates = ['target', 'label', 'y', 'class', 'outcome']
        target_found = any(col.lower() in target_candidates for col in data_frame.columns)
        info_parts.append(f"• *Target column:* {'✅ Found' if target_found else '❌ Not detected'}")
        
        # Add memory usage
        memory_usage = data_frame.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        info_parts.append(f"• *Memory usage:* {memory_usage:.2f} MB")
        
        return '\n'.join(info_parts)
    
    def _format_pipeline_summary(self, summary: Dict[str, Any]) -> str:
        """Format pipeline summary for Slack"""
        parts = ["📋 *Pipeline Summary:*"]
        
        if summary.get("has_raw_data"):
            shape = summary.get("raw_data_shape", (0, 0))
            parts.append(f"• Raw data: ✅ {shape[0]:,} rows × {shape[1]} columns")
        
        if summary.get("has_cleaned_data"):
            shape = summary.get("cleaned_data_shape", (0, 0))
            parts.append(f"• Cleaned data: ✅ {shape[0]:,} rows × {shape[1]} columns")
        
        if summary.get("has_selected_features"):
            count = summary.get("selected_features_count", 0)
            parts.append(f"• Selected features: ✅ {count} features")
        
        if summary.get("has_trained_model"):
            parts.append("• Trained model: ✅ Available")
        
        return '\n'.join(parts)
    
    def start(self):
        """Start the Slack bot"""
        # Print current configuration
        print_config()
        
        # Validate configuration
        config_errors = validate_config()
        if config_errors:
            print(f"❌ Configuration errors:")
            for error in config_errors:
                print(f"  • {error}")
            print("Please fix these configuration issues before starting the bot.")
            return False
        
        print("🚀 Starting Multi-Agent ML Pipeline Slack Bot...")
        print("🏗️ Architecture: Slack UI -> LangGraph Pipeline -> Specialized Agents")
        print("📎 Features: File upload support, intelligent routing, persistent sessions")
        print("🤖 Ready to process your ML workflows!")
        
        try:
            # Start the bot
            handler = SocketModeHandler(self.app, SLACK_APP_TOKEN)
            print("✅ Bot started! Upload data files and send messages in Slack to get started.")
            handler.start()
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Slack bot: {e}")
            return False


if __name__ == "__main__":
    bot = SlackMLBot()
    bot.start()
