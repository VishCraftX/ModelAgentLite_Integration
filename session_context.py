#!/usr/bin/env python3
"""
Global Session Context Manager
Allows setting and getting current session context across the application
"""

import threading
from typing import Optional, Tuple

# Thread-local storage for session context
_context = threading.local()

def set_session_context(user_id: str, thread_id: str):
    """Set the current session context for this thread"""
    _context.user_id = str(user_id)
    _context.thread_id = str(thread_id)

def get_session_context() -> Tuple[str, str]:
    """Get the current session context, returns (user_id, thread_id)"""
    user_id = getattr(_context, 'user_id', 'system')
    thread_id = getattr(_context, 'thread_id', 'general')
    return user_id, thread_id

def clear_session_context():
    """Clear the current session context"""
    if hasattr(_context, 'user_id'):
        delattr(_context, 'user_id')
    if hasattr(_context, 'thread_id'):
        delattr(_context, 'thread_id')

def has_session_context() -> bool:
    """Check if session context is set"""
    return hasattr(_context, 'user_id') and hasattr(_context, 'thread_id')

class SessionContext:
    """Context manager for setting session context"""
    
    def __init__(self, user_id: str, thread_id: str):
        self.user_id = str(user_id)
        self.thread_id = str(thread_id)
        self.previous_context = None
    
    def __enter__(self):
        # Save previous context
        self.previous_context = get_session_context()
        # Set new context
        set_session_context(self.user_id, self.thread_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        if self.previous_context:
            set_session_context(*self.previous_context)
        else:
            clear_session_context()

def extract_session_from_session_id(session_id: str) -> Tuple[str, str]:
    """Extract user_id and thread_id from session_id format"""
    if isinstance(session_id, str) and '_' in session_id:
        parts = session_id.split('_')
        user_id = parts[0]
        thread_id = '_'.join(parts[1:])
        return user_id, thread_id
    else:
        return str(session_id), 'main' 