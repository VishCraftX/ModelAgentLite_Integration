#!/usr/bin/env python3
"""
Thread-Specific Logging System
Saves logs to user's thread folder and appends when user returns to the same thread.
"""

import os
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Import for username resolution
from agent_utils import get_username_for_user_id


class ThreadLogger:
    """Thread-specific logger that saves to user's thread folder using usernames"""
    
    def __init__(self, user_id: str, thread_id: str, base_dir: str = "user_data", slack_manager: Optional[SlackManager] = None):
        self.user_id = user_id
        self.thread_id = thread_id
        self.base_dir = base_dir
        self.slack_manager = slack_manager
        
        # Get username for directory naming
        self.username = get_username_for_user_id(user_id)
        
        # Create user thread directory using username
        self.thread_dir = os.path.join(base_dir, self.username, thread_id)
        os.makedirs(self.thread_dir, exist_ok=True)
        
        # Log file paths
        self.log_file = os.path.join(self.thread_dir, "thread.log")
        self.debug_log_file = os.path.join(self.thread_dir, "debug.log")
        self.error_log_file = os.path.join(self.thread_dir, "errors.log")
        self.session_log_file = os.path.join(self.thread_dir, "session_events.jsonl")
        
        # Initialize loggers
        self._setup_loggers()
        
        # Log session start
        self.log_session_event("session_start", {
            "user_id": user_id,
            "username": self.username,
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat()
        })
    
    
    def _setup_loggers(self):
        """Setup thread-specific loggers"""
        
        # Main logger
        self.logger = logging.getLogger(f"thread_{self.user_id}_{self.thread_id}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler for main logs
        main_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        main_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(main_formatter)
        main_handler.setLevel(logging.INFO)
        self.logger.addHandler(main_handler)
        
        # Debug logger
        self.debug_logger = logging.getLogger(f"debug_{self.user_id}_{self.thread_id}")
        self.debug_logger.setLevel(logging.DEBUG)
        
        if self.debug_logger.handlers:
            self.debug_logger.handlers.clear()
        
        debug_handler = logging.FileHandler(self.debug_log_file, mode='a', encoding='utf-8')
        debug_formatter = logging.Formatter(
            '%(asctime)s | DEBUG | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        debug_handler.setFormatter(debug_formatter)
        self.debug_logger.addHandler(debug_handler)
        
        # Error logger
        self.error_logger = logging.getLogger(f"error_{self.user_id}_{self.thread_id}")
        self.error_logger.setLevel(logging.ERROR)
        
        if self.error_logger.handlers:
            self.error_logger.handlers.clear()
        
        error_handler = logging.FileHandler(self.error_log_file, mode='a', encoding='utf-8')
        error_formatter = logging.Formatter(
            '%(asctime)s | ERROR | %(message)s | %(pathname)s:%(lineno)d',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        self.debug_logger.propagate = False
        self.error_logger.propagate = False
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log info message"""
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        self.logger.info(message)
    
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        self.debug_logger.debug(message)
    
    def error(self, message: str, error: Optional[Exception] = None, extra_data: Optional[Dict[str, Any]] = None):
        """Log error message"""
        error_msg = message
        if error:
            error_msg = f"{message} | Exception: {str(error)} | Type: {type(error).__name__}"
        if extra_data:
            error_msg = f"{error_msg} | Data: {json.dumps(extra_data, default=str)}"
        self.error_logger.error(error_msg)
    
    def log_query(self, query: str, intent: Optional[str] = None, agent: Optional[str] = None):
        """Log user query"""
        self.info(f"USER QUERY: {query}", {
            "intent": intent,
            "agent": agent,
            "query_length": len(query)
        })
    
    def log_response(self, response: str, agent: Optional[str] = None, success: bool = True):
        """Log system response"""
        status = "SUCCESS" if success else "FAILED"
        self.info(f"SYSTEM RESPONSE [{status}]: {response[:200]}{'...' if len(response) > 200 else ''}", {
            "agent": agent,
            "response_length": len(response),
            "success": success
        })
    
    def log_agent_switch(self, from_agent: Optional[str], to_agent: str, reason: str = ""):
        """Log agent switching"""
        self.info(f"AGENT SWITCH: {from_agent} → {to_agent}", {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "reason": reason
        })
    
    def log_analysis(self, analysis_type: str, parameters: Dict[str, Any], result: Dict[str, Any]):
        """Log analysis execution"""
        self.info(f"ANALYSIS: {analysis_type.upper()}", {
            "analysis_type": analysis_type,
            "parameters": parameters,
            "success": result.get("success", False),
            "features_before": result.get("features_before"),
            "features_after": result.get("features_after"),
            "features_removed": result.get("features_removed")
        })
    
    def log_data_operation(self, operation: str, details: Dict[str, Any]):
        """Log data operations (load, save, transform)"""
        self.info(f"DATA OPERATION: {operation.upper()}", details)
    
    def log_session_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log session events to JSONL file"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            **event_data
        }
        
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, default=str) + '\n')
    
    def log_performance(self, operation: str, duration_seconds: float, details: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        perf_data = {
            "operation": operation,
            "duration_seconds": duration_seconds,
            "details": details or {}
        }
        self.debug(f"PERFORMANCE: {operation} took {duration_seconds:.3f}s", perf_data)
    
    def log_classification(self, query: str, classification_results: Dict[str, Any]):
        """Log intent classification results"""
        self.debug(f"CLASSIFICATION: '{query}' → {classification_results}")
    
    def log_routing(self, query: str, route_decision: str, confidence: Optional[float] = None):
        """Log routing decisions"""
        route_data = {
            "query": query,
            "route": route_decision,
            "confidence": confidence
        }
        self.debug(f"ROUTING: '{query}' → {route_decision}", route_data)
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of logs for this thread"""
        summary = {
            "thread_dir": self.thread_dir,
            "log_files": {
                "main_log": self.log_file,
                "debug_log": self.debug_log_file,
                "error_log": self.error_log_file,
                "session_events": self.session_log_file
            },
            "file_sizes": {},
            "line_counts": {}
        }
        
        # Get file sizes and line counts
        for log_type, log_path in summary["log_files"].items():
            if os.path.exists(log_path):
                summary["file_sizes"][log_type] = os.path.getsize(log_path)
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        summary["line_counts"][log_type] = sum(1 for _ in f)
                except:
                    summary["line_counts"][log_type] = 0
            else:
                summary["file_sizes"][log_type] = 0
                summary["line_counts"][log_type] = 0
        
        return summary
    
    def close(self):
        """Close all handlers and log session end"""
        self.log_session_event("session_end", {
            "timestamp": datetime.now().isoformat()
        })
        
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        for handler in self.debug_logger.handlers[:]:
            handler.close()
            self.debug_logger.removeHandler(handler)
        
        for handler in self.error_logger.handlers[:]:
            handler.close()
            self.error_logger.removeHandler(handler)


class ThreadLoggerManager:
    """Manager for thread loggers - singleton pattern"""
    
    _instance = None
    _loggers: Dict[str, ThreadLogger] = {}
    _slack_manager: Optional[SlackManager] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ThreadLoggerManager, cls).__new__(cls)
        return cls._instance
    
    def set_slack_manager(self, slack_manager: SlackManager):
        """Set the slack manager for username resolution"""
        self._slack_manager = slack_manager
    
    def get_logger(self, user_id: str, thread_id: str) -> ThreadLogger:
        """Get or create thread logger"""
        logger_key = f"{user_id}_{thread_id}"
        
        if logger_key not in self._loggers:
            self._loggers[logger_key] = ThreadLogger(user_id, thread_id, slack_manager=self._slack_manager)
        
        return self._loggers[logger_key]
    
    def close_logger(self, user_id: str, thread_id: str):
        """Close and remove thread logger"""
        logger_key = f"{user_id}_{thread_id}"
        
        if logger_key in self._loggers:
            self._loggers[logger_key].close()
            del self._loggers[logger_key]
    
    def close_all_loggers(self):
        """Close all active loggers"""
        for logger in self._loggers.values():
            logger.close()
        self._loggers.clear()


# Global logger manager instance
logger_manager = ThreadLoggerManager()


def get_thread_logger(user_id: str, thread_id: str) -> ThreadLogger:
    """Convenience function to get thread logger"""
    return logger_manager.get_logger(user_id, thread_id)


def close_thread_logger(user_id: str, thread_id: str):
    """Convenience function to close thread logger"""
    logger_manager.close_logger(user_id, thread_id) 