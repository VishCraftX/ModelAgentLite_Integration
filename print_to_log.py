#!/usr/bin/env python3
"""
Print to Log Function
Replaces print() to automatically log to thread.log while preserving console output
"""

import os
from datetime import datetime
from typing import Any


def print_to_log(*args, **kwargs):
    """
    Drop-in replacement for print() that also logs to thread.log
    
    Usage: Replace print(...) with print_to_log(...)
    """
    # First, print to console exactly like normal print()
    print(*args, **kwargs)
    
    # Then, try to log to thread file
    try:
        # Convert args to string (same as print does)
        message = ' '.join(str(arg) for arg in args)
        
        # Try to get session context from globals/locals
        import inspect
        frame = inspect.currentframe()
        
        # Look for session_id, user_id, thread_id in calling context
        user_id = "system"
        thread_id = "general"
        
        # Check multiple frames up the call stack for session info
        for i in range(5):  # Check up to 5 frames up
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
                        
                        # Look for direct user_id, thread_id
                        if 'user_id' in locals_dict and 'thread_id' in locals_dict:
                            user_id = locals_dict['user_id']
                            thread_id = locals_dict['thread_id']
                            break
            except:
                continue
        
        # Create thread directory
        thread_dir = os.path.join("user_data", str(user_id), str(thread_id))
        os.makedirs(thread_dir, exist_ok=True)
        
        # Log to master.log
        log_file = os.path.join(thread_dir, "master.log")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
    except Exception as e:
        # Silently fail - don't disrupt the main flow
        pass 