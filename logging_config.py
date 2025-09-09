#!/usr/bin/env python3
"""
Logging configuration for Multi-Agent ML Integration System
"""

import logging
import os
from pathlib import Path
from datetime import datetime

# Import master log handler to activate it
try:
    import master_log_handler
except ImportError:
    pass

def setup_colored_logging(instance_name: str = None) -> str:
    """
    Setup colored logging for the application
    
    Args:
        instance_name: Optional instance name for log file
        
    Returns:
        str: Path to the log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_suffix = f"_{instance_name}" if instance_name else ""
    log_filename = f"mal_integration{instance_suffix}_{timestamp}.log"
    log_file_path = logs_dir / log_filename
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()  # Console output
        ]
    )
    
    # Create logger for this application
    logger = logging.getLogger("ModelAgentLite")
    logger.setLevel(logging.INFO)
    
    print(f"📝 Logging configured - File: {log_file_path}")
    
    return str(log_file_path) 