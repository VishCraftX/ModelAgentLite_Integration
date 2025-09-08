#!/usr/bin/env python3
"""
Configuration for Multi-Agent ML Pipeline
"""

import os

# Slack Configuration - Environment variables only (no hardcoded tokens)
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

# Ollama Configuration  
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model Configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
FALLBACK_MODEL_1 = os.getenv("FALLBACK_MODEL_1", DEFAULT_MODEL)
FALLBACK_MODEL_2 = os.getenv("FALLBACK_MODEL_2", "deepseek-coder-v2:latest")

# OpenAI Configuration removed - using Qwen models only

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# File Upload Configuration
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls", "json", "tsv", "txt"]

# Model Building Configuration
DEFAULT_TEST_SIZE = float(os.getenv("DEFAULT_TEST_SIZE", "0.2"))
DEFAULT_RANDOM_STATE = int(os.getenv("DEFAULT_RANDOM_STATE", "42"))

# Session Configuration
SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
MAX_SESSIONS_PER_USER = int(os.getenv("MAX_SESSIONS_PER_USER", "10"))

# Pipeline Configuration
ENABLE_PERSISTENCE = os.getenv("ENABLE_PERSISTENCE", "true").lower() == "true"
ARTIFACTS_BASE_DIR = os.getenv("ARTIFACTS_BASE_DIR", None)  # Uses temp dir if None
STATE_BASE_DIR = os.getenv("STATE_BASE_DIR", None)  # Uses temp dir if None

# Validation
def validate_config():
    """Validate required configuration"""
    errors = []
    
    if not SLACK_BOT_TOKEN:
        errors.append("SLACK_BOT_TOKEN is required")
    elif not SLACK_BOT_TOKEN.startswith("xoxb-"):
        errors.append("SLACK_BOT_TOKEN should start with 'xoxb-'")
    
    if not SLACK_APP_TOKEN:
        errors.append("SLACK_APP_TOKEN is required")
    elif not SLACK_APP_TOKEN.startswith("xapp-"):
        errors.append("SLACK_APP_TOKEN should start with 'xapp-'")
    
    # Check if at least one model endpoint is available
    model_available = False
    
    # Check Ollama
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            model_available = True
    except:
        pass
    
    # Check OpenAI
    if OPENAI_API_KEY:
        model_available = True
    
    if not model_available:
        errors.append("No model endpoint available (Ollama or OpenAI)")
    
    return errors

def print_config():
    """Print current configuration (masking sensitive values)"""
    print("🔧 Multi-Agent ML Pipeline Configuration:")
    print(f"  SLACK_BOT_TOKEN: {'✅ Set' if SLACK_BOT_TOKEN else '❌ Not set'}")
    print(f"  SLACK_APP_TOKEN: {'✅ Set' if SLACK_APP_TOKEN else '❌ Not set'}")
    print(f"  OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
    print(f"  DEFAULT_MODEL: {DEFAULT_MODEL}")
    print(f"  FALLBACK_MODEL_1: {FALLBACK_MODEL_1}")
    print(f"  FALLBACK_MODEL_2: {FALLBACK_MODEL_2}")
    print(f"  LOG_LEVEL: {LOG_LEVEL}")
    print(f"  MAX_FILE_SIZE_MB: {MAX_FILE_SIZE_MB}")
    print(f"  SUPPORTED_FILE_TYPES: {', '.join(SUPPORTED_FILE_TYPES)}")
    print(f"  DEFAULT_TEST_SIZE: {DEFAULT_TEST_SIZE}")
    print(f"  SESSION_TIMEOUT_HOURS: {SESSION_TIMEOUT_HOURS}")
    print(f"  ENABLE_PERSISTENCE: {ENABLE_PERSISTENCE}")
    print(f"  ARTIFACTS_BASE_DIR: {ARTIFACTS_BASE_DIR or 'Default (temp)'}")
    print(f"  STATE_BASE_DIR: {STATE_BASE_DIR or 'Default (temp)'}")
