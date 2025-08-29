#!/usr/bin/env python3
"""
Orchestrator for Multi-Agent ML System
Intelligent routing and coordination of preprocessing, feature selection, and model building agents
"""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pipeline_state import PipelineState


class AgentType(Enum):
    """Available agent types in the system"""
    PREPROCESSING = "preprocessing"
    FEATURE_SELECTION = "feature_selection"
    MODEL_BUILDING = "model_building"
    END = "END"


class Orchestrator:
    """
    Intelligent orchestrator that routes queries to appropriate agents
    and manages the overall pipeline flow
    """
    
    def __init__(self):
        # Keyword patterns for routing
        self.preprocessing_keywords = [
            "clean", "cleaning", "preprocess", "preprocessing", 
            "missing", "null", "nan", "impute", "imputation",
            "normalize", "normalization", "standardize", "scale", "scaling",
            "outlier", "outliers", "duplicate", "duplicates",
            "encode", "encoding", "categorical", "transform"
        ]
        
        self.feature_selection_keywords = [
            "feature", "features", "select", "selection", "variable",
            "iv", "information value", "woe", "weight of evidence",
            "correlation", "corr", "multicollinearity", "vif",
            "pca", "principal component", "dimensionality",
            "importance", "rank", "ranking", "filter", "wrapper"
        ]
        
        self.model_building_keywords = [
            "model", "train", "training", "fit", "fitting",
            "predict", "prediction", "forecast", "forecasting",
            "algorithm", "classifier", "regression", "clustering",
            "lgbm", "lightgbm", "xgboost", "xgb", "random forest",
            "decision tree", "logistic", "linear", "svm", "neural",
            "evaluate", "evaluation", "performance", "metrics",
            "accuracy", "precision", "recall", "f1", "auc", "roc"
        ]
        
        # Pipeline flow patterns
        self.full_pipeline_patterns = [
            r"train.*model.*on.*csv",
            r"build.*model.*from.*data",
            r"complete.*pipeline",
            r"end.*to.*end",
            r"full.*analysis"
        ]
        
        self.resume_patterns = [
            r"use.*cleaned.*data",
            r"continue.*from.*last",
            r"resume.*session",
            r"load.*previous"
        ]
    
    def route(self, state: PipelineState) -> str:
        """
        Main routing logic - determines which agent to call next
        """
        query = (state.user_query or "").lower().strip()
        
        if not query:
            print("[Orchestrator] No query provided, ending pipeline")
            return AgentType.END.value
        
        print(f"[Orchestrator] Routing query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        # Check for full pipeline requests
        if self._is_full_pipeline_request(query):
            return self._route_full_pipeline(state)
        
        # Check for resume/continuation requests
        if self._is_resume_request(query):
            return self._route_resume(state)
        
        # Direct agent routing based on keywords
        agent_scores = self._calculate_agent_scores(query)
        best_agent = max(agent_scores.items(), key=lambda x: x[1])
        
        print(f"[Orchestrator] Agent scores: {agent_scores}")
        print(f"[Orchestrator] Selected agent: {best_agent[0]} (score: {best_agent[1]:.2f})")
        
        # If no clear winner, use data-driven routing
        if best_agent[1] < 0.3:
            return self._route_by_data_state(state)
        
        return best_agent[0]
    
    def _is_full_pipeline_request(self, query: str) -> bool:
        """Check if query requests a full pipeline"""
        for pattern in self.full_pipeline_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _is_resume_request(self, query: str) -> bool:
        """Check if query is asking to resume from previous state"""
        for pattern in self.resume_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _calculate_agent_scores(self, query: str) -> Dict[str, float]:
        """Calculate relevance scores for each agent based on keywords"""
        scores = {
            AgentType.PREPROCESSING.value: 0.0,
            AgentType.FEATURE_SELECTION.value: 0.0,
            AgentType.MODEL_BUILDING.value: 0.0
        }
        
        query_words = query.split()
        total_words = len(query_words)
        
        if total_words == 0:
            return scores
        
        # Count keyword matches for each agent
        preprocessing_matches = sum(1 for word in query_words 
                                  if any(keyword in word for keyword in self.preprocessing_keywords))
        
        feature_selection_matches = sum(1 for word in query_words 
                                      if any(keyword in word for keyword in self.feature_selection_keywords))
        
        model_building_matches = sum(1 for word in query_words 
                                   if any(keyword in word for keyword in self.model_building_keywords))
        
        # Calculate normalized scores
        scores[AgentType.PREPROCESSING.value] = preprocessing_matches / total_words
        scores[AgentType.FEATURE_SELECTION.value] = feature_selection_matches / total_words
        scores[AgentType.MODEL_BUILDING.value] = model_building_matches / total_words
        
        return scores
    
    def _route_full_pipeline(self, state: PipelineState) -> str:
        """Route full pipeline requests based on current data state"""
        print("[Orchestrator] Full pipeline request detected")
        
        # Start from the beginning if no data
        if state.raw_data is None:
            print("[Orchestrator] No raw data, need to start with data loading")
            return AgentType.PREPROCESSING.value
        
        # If we have raw data but no cleaned data, start with preprocessing
        if state.cleaned_data is None:
            print("[Orchestrator] Raw data available, starting with preprocessing")
            return AgentType.PREPROCESSING.value
        
        # If we have cleaned data but no features selected, do feature selection
        if not state.selected_features:
            print("[Orchestrator] Cleaned data available, starting with feature selection")
            return AgentType.FEATURE_SELECTION.value
        
        # If we have features but no model, build model
        if state.trained_model is None:
            print("[Orchestrator] Features selected, starting with model building")
            return AgentType.MODEL_BUILDING.value
        
        # Full pipeline already complete
        print("[Orchestrator] Full pipeline already complete")
        return AgentType.END.value
    
    def _route_resume(self, state: PipelineState) -> str:
        """Route resume requests based on available state"""
        print("[Orchestrator] Resume request detected")
        
        # Determine where to resume based on available data
        if state.trained_model is not None:
            print("[Orchestrator] Model available, routing to model building for further operations")
            return AgentType.MODEL_BUILDING.value
        
        if state.selected_features:
            print("[Orchestrator] Features selected, routing to model building")
            return AgentType.MODEL_BUILDING.value
        
        if state.cleaned_data is not None:
            print("[Orchestrator] Cleaned data available, routing to feature selection")
            return AgentType.FEATURE_SELECTION.value
        
        if state.raw_data is not None:
            print("[Orchestrator] Raw data available, routing to preprocessing")
            return AgentType.PREPROCESSING.value
        
        print("[Orchestrator] No previous state to resume from")
        return AgentType.END.value
    
    def _route_by_data_state(self, state: PipelineState) -> str:
        """Route based on current data state when keywords are unclear"""
        print("[Orchestrator] Using data-driven routing")
        
        # If no data at all, suggest preprocessing (data loading)
        if state.raw_data is None:
            print("[Orchestrator] No data available, routing to preprocessing")
            return AgentType.PREPROCESSING.value
        
        # If raw data but no cleaned data, go to preprocessing
        if state.cleaned_data is None:
            print("[Orchestrator] Raw data needs cleaning, routing to preprocessing")
            return AgentType.PREPROCESSING.value
        
        # If cleaned data but no features, go to feature selection
        if not state.selected_features:
            print("[Orchestrator] Cleaned data needs feature selection")
            return AgentType.FEATURE_SELECTION.value
        
        # If features but no model, go to model building
        if state.trained_model is None:
            print("[Orchestrator] Features ready for model building")
            return AgentType.MODEL_BUILDING.value
        
        # Everything is done, route to model building for additional operations
        print("[Orchestrator] Pipeline complete, routing to model building for additional operations")
        return AgentType.MODEL_BUILDING.value
    
    def get_routing_explanation(self, state: PipelineState, selected_agent: str) -> str:
        """Get human-readable explanation of routing decision"""
        query = state.user_query or "No query"
        
        explanations = {
            AgentType.PREPROCESSING.value: self._get_preprocessing_explanation(state),
            AgentType.FEATURE_SELECTION.value: self._get_feature_selection_explanation(state),
            AgentType.MODEL_BUILDING.value: self._get_model_building_explanation(state),
            AgentType.END.value: "No further processing needed"
        }
        
        return explanations.get(selected_agent, f"Routed to {selected_agent}")
    
    def _get_preprocessing_explanation(self, state: PipelineState) -> str:
        """Get explanation for preprocessing routing"""
        if state.raw_data is None:
            return "No data available - preprocessing agent will handle data loading and cleaning"
        elif state.cleaned_data is None:
            return "Raw data needs cleaning - preprocessing agent will handle missing values, outliers, and transformations"
        else:
            return "Additional preprocessing requested"
    
    def _get_feature_selection_explanation(self, state: PipelineState) -> str:
        """Get explanation for feature selection routing"""
        if not state.selected_features:
            return "Feature selection needed - agent will analyze and select optimal features"
        else:
            return "Additional feature analysis requested"
    
    def _get_model_building_explanation(self, state: PipelineState) -> str:
        """Get explanation for model building routing"""
        if state.trained_model is None:
            return "Model building needed - agent will train and evaluate models"
        else:
            return "Model operations requested - predictions, evaluation, or model improvements"
    
    def suggest_next_steps(self, state: PipelineState) -> List[str]:
        """Suggest possible next steps based on current state"""
        suggestions = []
        
        if state.raw_data is None:
            suggestions.append("Upload or load a dataset to begin analysis")
        elif state.cleaned_data is None:
            suggestions.append("Clean and preprocess the raw data")
        elif not state.selected_features:
            suggestions.append("Select optimal features for modeling")
        elif state.trained_model is None:
            suggestions.append("Build and train a machine learning model")
        else:
            suggestions.extend([
                "Make predictions with the trained model",
                "Evaluate model performance",
                "Try different model algorithms",
                "Create visualizations and reports"
            ])
        
        return suggestions
    
    def validate_transition(self, current_agent: str, next_agent: str, state: PipelineState) -> Tuple[bool, str]:
        """Validate if transition between agents is valid"""
        # Define valid transitions
        valid_transitions = {
            AgentType.PREPROCESSING.value: [
                AgentType.FEATURE_SELECTION.value, 
                AgentType.MODEL_BUILDING.value,
                AgentType.END.value
            ],
            AgentType.FEATURE_SELECTION.value: [
                AgentType.MODEL_BUILDING.value,
                AgentType.PREPROCESSING.value,  # Back to preprocessing if needed
                AgentType.END.value
            ],
            AgentType.MODEL_BUILDING.value: [
                AgentType.PREPROCESSING.value,  # Back to preprocessing for new data
                AgentType.FEATURE_SELECTION.value,  # Back to feature selection
                AgentType.END.value
            ]
        }
        
        if current_agent not in valid_transitions:
            return True, "No current agent, transition allowed"
        
        if next_agent in valid_transitions[current_agent]:
            return True, f"Valid transition from {current_agent} to {next_agent}"
        
        return False, f"Invalid transition from {current_agent} to {next_agent}"


# Global orchestrator instance
orchestrator = Orchestrator()
