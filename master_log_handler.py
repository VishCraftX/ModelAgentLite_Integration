#!/usr/bin/env python3
"""
Master Log Handler
Custom logging handler that sends all logger.info/debug/error calls to master.log
"""

import logging
import os
import inspect
from datetime import datetime
from typing import Optional


class MasterLogHandler(logging.Handler):
    """Custom logging handler that writes to master.log in thread directories"""
    
    def __init__(self):
        super().__init__()
        self.setLevel(logging.DEBUG)  # Capture all levels
        
    def emit(self, record):
        """Emit a log record to master.log"""
        try:
            # Format the log message
            message = self.format(record)
            
            # Try to get session context from global context first, then call stack
            user_id = "system"
            thread_id = "general"
            
            # Try global session context first
            try:
                from session_context import get_session_context, has_session_context
                if has_session_context():
                    user_id, thread_id = get_session_context()
                    # If we got valid context, skip stack inspection
                    if user_id != "system" or thread_id != "general":
                        # Create thread directory and write log
                        thread_dir = os.path.join("user_data", str(user_id), str(thread_id))
                        os.makedirs(thread_dir, exist_ok=True)
                        
                        log_file = os.path.join(thread_dir, "master.log")
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_entry = f"[{timestamp}] LOGGER-{record.levelname}: {message}\n"
                        
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(log_entry)
                        return  # Exit early, we're done
            except ImportError:
                pass
            
            # Look through the call stack for session information
            frame = inspect.currentframe()
            for i in range(10):  # Check up to 10 frames up
                try:
                    if frame:
                        frame = frame.f_back
                        if frame and frame.f_locals:
                            locals_dict = frame.f_locals
                            
                            # Look for session_id pattern
                            if 'session_id' in locals_dict:
                                session_id = locals_dict['session_id']
                                if isinstance(session_id, str) and '_' in session_id:
                                    parts = session_id.split('_')
                                    user_id = parts[0]
                                    thread_id = '_'.join(parts[1:])
                                    break
                            
                            # Look for state object with session_id
                            if 'state' in locals_dict:
                                state = locals_dict['state']
                                if hasattr(state, 'session_id') and state.session_id:
                                    session_id = state.session_id
                                    if isinstance(session_id, str) and '_' in session_id:
                                        parts = session_id.split('_')
                                        user_id = parts[0]
                                        thread_id = '_'.join(parts[1:])
                                        break
                            
                            # Look for session object with user_id
                            if 'session' in locals_dict:
                                session = locals_dict['session']
                                if hasattr(session, 'user_id'):
                                    user_id = str(session.user_id)
                                    thread_id = getattr(session, 'thread_id', 'main')
                                    break
                            
                            # Look for direct user_id, thread_id
                            if 'user_id' in locals_dict and 'thread_id' in locals_dict:
                                user_id = str(locals_dict['user_id'])
                                thread_id = str(locals_dict['thread_id'])
                                break
                            
                            # Look for user_id alone
                            if 'user_id' in locals_dict:
                                user_id = str(locals_dict['user_id'])
                                thread_id = 'main'
                                break
                                
                except:
                    continue
            
            # Create thread directory
            thread_dir = os.path.join("user_data", str(user_id), str(thread_id))
            os.makedirs(thread_dir, exist_ok=True)
            
            # Write to master.log
            log_file = os.path.join(thread_dir, "master.log")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] LOGGER-{record.levelname}: {message}\n"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            # Silently fail - don't disrupt the main logging flow
            pass


def setup_master_log_handler():
    """Set up the master log handler for all loggers"""
    
    # Create our custom handler
    master_handler = MasterLogHandler()
    
    # Set a simple format (the handler will add its own prefix)
    formatter = logging.Formatter('%(message)s')
    master_handler.setFormatter(formatter)
    
    # Add to the root logger so all loggers inherit it
    root_logger = logging.getLogger()
    root_logger.addHandler(master_handler)
    
    # Also add to specific loggers that are commonly used
    common_loggers = [
        'ModelAgentLite',
        'feature_selection_agent_impl',
        'preprocessing_agent_slack',
        'agents_wrapper',
        'model_building_agent_impl',
        'langgraph_pipeline'
    ]
    
    for logger_name in common_loggers:
        logger = logging.getLogger(logger_name)
        logger.addHandler(master_handler)
    
    print("✅ Master log handler installed - all logger calls will also go to master.log")


# Auto-setup when imported
def auto_setup():
    """Automatically set up master logging when this module is imported"""
    try:
        setup_master_log_handler()
    except Exception as e:
        # Don't break if setup fails
        pass

# Call auto-setup
auto_setup() 