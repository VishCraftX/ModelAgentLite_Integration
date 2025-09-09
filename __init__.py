#!/usr/bin/env python3
"""
Multi-Agent ML Integration System
A comprehensive ML pipeline using LangGraph with specialized agents
"""

from .pipeline_state import PipelineState, StateManager, state_manager
from .orchestrator import Orchestrator, orchestrator, AgentType
from .agents_wrapper import preprocessing_agent, feature_selection_agent, model_building_agent
from .toolbox import (
    SlackManager, ArtifactManager, ProgressTracker, ExecutionAgent,
    initialize_toolbox
)
from .langgraph_pipeline import MultiAgentMLPipeline, get_pipeline, initialize_pipeline
from .slack_bot import SlackMLBot
from .config import validate_config, print_config

__version__ = "1.0.0"
__author__ = "MAL Integration Team"

__all__ = [
    # Core classes
    "PipelineState",
    "StateManager", 
    "Orchestrator",
    "MultiAgentMLPipeline",
    "SlackMLBot",
    "SlackManager",
    "ArtifactManager",
    "ProgressTracker",
    "ExecutionAgent",
    
    # Agent types
    "AgentType",
    
    # Global instances
    "state_manager",
    "orchestrator", 
    "preprocessing_agent",
    "feature_selection_agent", 
    "model_building_agent",
    
    # Factory functions
    "get_pipeline",
    "initialize_pipeline",
    "initialize_toolbox",
    
    # Configuration
    "validate_config",
    "print_config"
]
