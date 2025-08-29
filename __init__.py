#!/usr/bin/env python3
"""
Multi-Agent ML Integration System
A comprehensive ML pipeline using LangGraph with specialized agents
"""

from .pipeline_state import PipelineState, StateManager, state_manager
from .orchestrator import Orchestrator, orchestrator, AgentType
from .agents_integrated import preprocessing_agent, feature_selection_agent, model_building_agent
from .toolbox import (
    SlackManager, ArtifactManager, ProgressTracker, ExecutionAgent,
    slack_manager, artifact_manager, progress_tracker, execution_agent,
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
    
    # Agent types
    "AgentType",
    
    # Global instances
    "state_manager",
    "orchestrator", 
    "preprocessing_agent",
    "feature_selection_agent", 
    "model_building_agent",
    "slack_manager",
    "artifact_manager",
    "progress_tracker",
    "execution_agent",
    
    # Factory functions
    "get_pipeline",
    "initialize_pipeline",
    "initialize_toolbox",
    
    # Configuration
    "validate_config",
    "print_config"
]
