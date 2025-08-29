#!/usr/bin/env python3
"""
Unified Orchestrator for Multi-Agent ML System
Combines enhanced LLM-powered routing with hybrid keyword/LLM approach
"""

import os
import re
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from datetime import datetime
from pipeline_state import PipelineState

# LLM imports (with optional fallback)
try:
    import ollama
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ LLM libraries not available, using keyword-only classification")

# Text normalization imports
try:
    from nltk.stem import WordNetLemmatizer
    import nltk
    # Download required NLTK data if not present
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("📦 Downloading NLTK WordNet data...")
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print("📦 Downloading NLTK OMW data...")
        nltk.download('omw-1.4', quiet=True)
    
    lemmatizer = WordNetLemmatizer()
    LEMMATIZER_AVAILABLE = True
except ImportError:
    LEMMATIZER_AVAILABLE = False
    print("⚠️ NLTK not available, using basic text normalization")


class AgentType(Enum):
    """Available agent types in the system"""
    PREPROCESSING = "preprocessing"
    FEATURE_SELECTION = "feature_selection"
    MODEL_BUILDING = "model_building"
    END = "END"


class Orchestrator:
    """
    Unified orchestrator with hybrid keyword scoring + LLM fallback
    Fast keyword classification with intelligent LLM fallback for ambiguous cases
    """
    
    def __init__(self):
        self.default_model = os.getenv("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
        
        # Exhaustive keywords for fast classification
        self.preprocessing_keywords = [
            # Cleaning & general
            "clean", "cleaning", "preprocess", "preprocessing", "prepare", "preparation",
            "wrangle", "wrangling", "munging", "transform", "transformation",
            "pipeline", "etl", "extract", "load", "data prep", "data cleansing",

            # Missing values
            "missing", "null", "nan", "na", "impute", "imputation", "fillna", 
            "replace", "dropna", "interpolate",

            # Outliers & scaling
            "outlier", "outliers", "clip", "winsorize", "standardize", "standardization",
            "normalize", "normalization", "scaling", "rescale", "rescaling",
            "zscore", "minmax", "robust", "log transform",

            # Encoding
            "encode", "encoding", "categorical", "onehot", "one-hot", "label encode",
            "ordinal", "dummies", "binary encode", "hash encode",

            # Duplicates
            "duplicate", "duplicates", "deduplicate", "remove duplicates",

            # Other cleaning ops
            "noise", "denoise", "inconsistent", "typo", "text clean", "string clean",
            "parse", "tokenize", "stemming", "lemmatize", "lemmatization"
        ]
        
        self.feature_selection_keywords = [
            # General
            "feature", "features", "variable", "variables", "column", "columns",
            "dimension", "dimensions", "attribute", "attributes", "predictor", "predictors",

            # Selection methods
            "select", "selection", "filter", "wrapper", "embedded", "ranking", "rank",
            "importance", "relevance", "subset", "stepwise",

            # Statistical
            "iv", "information value", "woe", "weight of evidence",
            "correlation", "corr", "pearson", "spearman", "mutual information",
            "anova", "chi-square", "chisq", "f-test", "p-value", "vif", "variance inflation",
            "multicollinearity",

            # Dimensionality reduction
            "pca", "principal component", "factor analysis", "fa", "ica",
            "dimensionality", "reduction", "manifold", "t-sne", "umap", "svd",

            # Model-driven
            "gini importance", "permutation importance", "shap", "shapley", "lime",
            "gain", "split importance"
        ]
        
        self.model_building_keywords = [
            # General
            "model", "models", "train", "training", "fit", "fitting", 
            "predict", "prediction", "forecast", "forecasts",
            "inference", "deploy", "deployment", "evaluate", "evaluation",
            "validate", "validation", "cross-validation", "cv", "test set",

            # Algorithms (classical)
            "regression", "linear regression", "logistic regression", 
            "ridge", "lasso", "elasticnet", "glm",
            "decision tree", "random forest", "xgboost", "lgbm", "lightgbm",
            "catboost", "gradient boosting", "svm", "support vector", "knn",
            "naive bayes", "bayesian", "ensemble", "bagging", "boosting", "stacking",

            # Neural nets
            "neural", "network", "deep learning", "cnn", "rnn", "lstm",
            "transformer", "bert", "gpt",

            # Metrics
            "accuracy", "precision", "recall", "f1", "auc", "roc", "mse", "rmse",
            "mae", "logloss", "loss function", "objective",

            # Forecasting
            "arima", "sarima", "prophet", "timeseries", "time series", "seasonal",
            "trend", "holt winters"
        ]
        
        self.general_keywords = [
            "hello", "hi", "hey", "greetings", "morning", "afternoon", "evening",
            "good morning", "good evening", "good afternoon",
            "thanks", "thank you", "cheers", "appreciate",
            "what can you do", "capabilities", "features", "how it works",
            "status", "summary", "overview", "explain", "about", "help", "support"
        ]
        
        self.code_execution_keywords = [
            # General coding
            "code", "snippet", "script", "function", "method", "execute", "execution",
            "run", "running", "debug", "fix", "implement", "implementation",

            # Stats / analysis
            "calculate", "compute", "analyze", "analysis", "statistical", "statistics",
            "mean", "median", "mode", "std", "variance", "distribution",
            "correlation", "covariance",

            # Visualization
            "plot", "plots", "graph", "graphs", "chart", "charts", "visualize",
            "visualization", "show", "display", "scatter", "line chart", "bar chart",
            "histogram", "boxplot", "heatmap",

            # Data inspection
            "summary", "describe", "shape", "head", "tail", "info", "schema",
            "columns", "datatypes"
        ]

    def normalize_text(self, query: str) -> List[str]:
        """
        Normalize text to handle singular/plural/variants automatically
        Returns list of normalized words
        """
        # Extract words using regex
        words = re.findall(r"\w+", query.lower())
        
        if LEMMATIZER_AVAILABLE:
            # Use NLTK lemmatizer for better normalization
            return [lemmatizer.lemmatize(word) for word in words]
        else:
            # Basic normalization without NLTK
            normalized = []
            for word in words:
                # Handle common plural forms
                if word.endswith('ies') and len(word) > 4:
                    normalized.append(word[:-3] + 'y')  # categories -> category
                elif word.endswith('es') and len(word) > 3:
                    normalized.append(word[:-2])  # features -> feature
                elif word.endswith('s') and len(word) > 2:
                    normalized.append(word[:-1])  # models -> model
                else:
                    normalized.append(word)
            return normalized

    def _match_keywords_advanced(self, query: str, keywords: List[str]) -> int:
        """
        Advanced keyword matching with normalization and phrase support
        """
        query_lower = query.lower()
        normalized_words = self.normalize_text(query)
        
        matches = 0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Direct phrase match (highest priority)
            if keyword_lower in query_lower:
                matches += 1
                continue
            
            # Normalized word matching
            if LEMMATIZER_AVAILABLE:
                keyword_normalized = [lemmatizer.lemmatize(word) for word in re.findall(r"\w+", keyword_lower)]
            else:
                keyword_normalized = self.normalize_text(keyword)
            
            # Check if all words in the keyword phrase are present (normalized)
            if len(keyword_normalized) == 1:
                # Single word keyword
                if keyword_normalized[0] in normalized_words:
                    matches += 1
            else:
                # Multi-word keyword - check if all words are present
                if all(word in normalized_words for word in keyword_normalized):
                    matches += 1
        
        return matches

    def classify_intent_with_llm(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Use LLM to classify user intent (fallback for ambiguous cases)
        """
        if not LLM_AVAILABLE:
            print("⚠️ LLM not available, using fallback classification")
            return self._fallback_classify_intent(query)
        
        # Build context information
        context_parts = []
        if context:
            if context.get("has_raw_data"):
                context_parts.append("User has uploaded data")
            if context.get("has_cleaned_data"):
                context_parts.append("Data has been preprocessed")
            if context.get("has_selected_features"):
                context_parts.append("Features have been selected")
            if context.get("has_trained_model"):
                context_parts.append("Model has been trained")
        
        context_info = f"CONVERSATION CONTEXT: {'; '.join(context_parts)}" if context_parts else "CONVERSATION CONTEXT: None"
        
        classification_prompt = f"""You are an intent classifier for a multi-agent ML pipeline system. Your job is to classify user queries into exactly one of these categories:

1. "preprocessing" - User wants to clean, preprocess, or prepare data
2. "feature_selection" - User wants to select features, analyze feature importance, or reduce dimensionality  
3. "model_building" - User wants to work with models (build new, use existing, predict, evaluate, visualize)
4. "general_query" - General questions, greetings, capability questions, or unclear intent
5. "code_execution" - General Python code that needs to be executed (data analysis, calculations, plotting, etc.)
6. "full_pipeline" - User wants to run the complete ML pipeline from start to finish

{context_info}

PRIORITY CLASSIFICATION RULES (check in this order):

HIGHEST PRIORITY - "full_pipeline" if query contains:
- "complete pipeline", "full pipeline", "end to end", "start to finish"
- "build complete ML pipeline", "run entire pipeline", "do everything"

SECOND PRIORITY - "preprocessing" if query contains:
- "clean", "preprocess", "missing values", "outliers", "normalize", "standardize"
- "encode categorical", "handle nulls", "remove duplicates", "data cleaning"

THIRD PRIORITY - "feature_selection" if query contains:
- "select features", "feature selection", "feature importance", "IV analysis"
- "correlation analysis", "PCA", "dimensionality reduction", "VIF analysis"

FOURTH PRIORITY - "model_building" if query contains:
- "model", "train", "build", "predict", "score", "classify", "evaluate"
- "use model", "existing model", "new model", "create model"
- "lgbm", "xgboost", "random forest", "decision tree", "classifier", "regressor"
- "visualize model", "model plot", "tree plot", "feature importance"
- "segments", "deciles", "buckets", "rankings" (model-based operations)

FIFTH PRIORITY - "code_execution" if query contains:
- "calculate", "compute", "analyze", "plot", "graph", "chart", "visualize" (but NOT model-related)
- "show me", "create plot", "data analysis", "statistics", "correlation", "distribution"
- "write code", "python code", "code to", "script to"

LOWEST PRIORITY - "general_query" for:
- Greetings: "hello", "hi", "hey"
- Capability questions: "what can you do", "help", "capabilities"
- General questions without specific action requests

EXAMPLES:
- "Build a complete ML pipeline" → full_pipeline
- "Clean this data and handle missing values" → preprocessing
- "Select the most important features" → feature_selection
- "Train a new model" → model_building
- "Use existing model for predictions" → model_building
- "Show model performance" → model_building
- "Visualize decision tree" → model_building
- "Calculate correlation between features" → code_execution (data analysis code needed)
- "Hello, what can you do?" → general_query

USER QUERY: "{query}"

Respond with ONLY one word: preprocessing, feature_selection, model_building, general_query, code_execution, or full_pipeline"""

        try:
            response = ollama.chat(
                model=self.default_model,
                messages=[{"role": "user", "content": classification_prompt}]
            )
            
            intent = response["message"]["content"].strip().lower()
            
            # Validate response
            valid_intents = ["preprocessing", "feature_selection", "model_building", "general_query", "code_execution", "full_pipeline"]
            if intent not in valid_intents:
                # Fallback classification
                intent = self._fallback_classify_intent(query)
            
            print(f"🎯 LLM Intent classified: {intent}")
            return intent
            
        except Exception as e:
            print(f"⚠️ LLM intent classification failed: {e}")
            # If LLM fails, use fallback classification
            return self._fallback_classify_intent(query)

    def _fallback_classify_intent(self, query: str) -> str:
        """Fallback keyword-based classification"""
        query_lower = query.lower()
        
        # Count keyword matches for each category
        scores = {
            "preprocessing": sum(1 for kw in self.preprocessing_keywords if kw in query_lower),
            "feature_selection": sum(1 for kw in self.feature_selection_keywords if kw in query_lower),
            "model_building": sum(1 for kw in self.model_building_keywords if kw in query_lower),
            "code_execution": sum(1 for kw in self.code_execution_keywords if kw in query_lower),
            "general_query": sum(1 for kw in self.general_keywords if kw in query_lower)
        }
        
        # Check for full pipeline indicators
        full_pipeline_phrases = ["complete pipeline", "full pipeline", "end to end", "build complete", "entire pipeline"]
        if any(phrase in query_lower for phrase in full_pipeline_phrases):
            return "full_pipeline"
        
        # Find the category with the highest score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return "general_query"

    def _classify_with_keyword_scoring(self, query: str) -> Tuple[str, Dict]:
        """
        Fast keyword-based classification with confidence metrics and advanced matching
        Returns: (intent, confidence_info)
        """
        # Use advanced keyword matching with normalization
        scores = {
            "preprocessing": self._match_keywords_advanced(query, self.preprocessing_keywords),
            "feature_selection": self._match_keywords_advanced(query, self.feature_selection_keywords),
            "model_building": self._match_keywords_advanced(query, self.model_building_keywords),
            "code_execution": self._match_keywords_advanced(query, self.code_execution_keywords),
            "general_query": self._match_keywords_advanced(query, self.general_keywords)
        }
        
        # Check for full pipeline indicators
        query_lower = query.lower()
        full_pipeline_phrases = ["complete pipeline", "full pipeline", "end to end", "build complete", "entire pipeline"]
        if any(phrase in query_lower for phrase in full_pipeline_phrases):
            return "full_pipeline", {
                "max_score": 1.0,
                "score_diff": 1.0,
                "method": "phrase_match",
                "scores": scores
            }
        
        # Calculate confidence metrics
        total_words = len(query_lower.split())
        if total_words > 0:
            # Normalize scores by query length
            normalized_scores = {k: v / total_words for k, v in scores.items()}
        else:
            normalized_scores = scores
        
        # Find best and second-best scores
        sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, max_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
        
        score_diff = max_score - second_score
        
        confidence_info = {
            "max_score": max_score,
            "score_diff": score_diff,
            "method": "keyword_scoring",
            "scores": normalized_scores,
            "raw_scores": scores
        }
        
        # If no keywords matched, default to general_query
        if max_score == 0:
            return "general_query", confidence_info
        
        return best_intent, confidence_info

    def _route_by_intent(self, state: PipelineState, intent: str) -> str:
        """
        Route based on classified intent and current state
        """
        if intent == "full_pipeline":
            # Start from the beginning or continue from current state
            if state.raw_data is None:
                return "preprocessing"
            elif state.cleaned_data is None:
                return "preprocessing"
            elif state.selected_features is None:
                return "feature_selection"
            elif state.trained_model is None:
                return "model_building"
            else:
                return "model_building"  # Pipeline complete, can do additional operations
        
        elif intent == "preprocessing":
            return "preprocessing"
        
        elif intent == "feature_selection":
            if state.cleaned_data is None and state.raw_data is not None:
                print("[Orchestrator] Need to preprocess data first")
                return "preprocessing"
            return "feature_selection"
        
        elif intent == "model_building":
            # Check prerequisites
            if state.raw_data is None:
                print("[Orchestrator] No data available for model building")
                return "general_response"
            elif state.cleaned_data is None:
                print("[Orchestrator] Need to preprocess data first")
                return "preprocessing"
            elif state.selected_features is None:
                print("[Orchestrator] Need to select features first")
                return "feature_selection"
            return "model_building"
        
        elif intent == "code_execution":
            return "code_execution"
        
        elif intent == "general_query":
            # Handle general queries directly and end
            response = self.generate_general_response(state.user_query, state)
            state.last_response = response
            return AgentType.END.value
        
        else:
            # Fallback to data-driven routing
            return self._data_driven_routing(state)

    def _data_driven_routing(self, state: PipelineState) -> str:
        """Fallback data-driven routing logic"""
        if state.raw_data is None:
            return "general_response"
        elif state.cleaned_data is None:
            return "preprocessing"
        elif state.selected_features is None:
            return "feature_selection"
        elif state.trained_model is None:
            return "model_building"
        else:
            return "model_building"

    def generate_capability_response(self, state: PipelineState) -> str:
        """
        Generate a comprehensive capability response using LLM
        """
        # Build context about current pipeline state
        context_parts = []
        if state.raw_data is not None:
            context_parts.append(f"I have your dataset with {state.raw_data.shape[0]:,} rows and {state.raw_data.shape[1]} columns")
        if state.cleaned_data is not None:
            context_parts.append("Data has been preprocessed")
        if state.selected_features is not None:
            context_parts.append(f"{len(state.selected_features)} features have been selected")
        if state.trained_model is not None:
            context_parts.append("A model has been trained")
        
        context = ". ".join(context_parts) if context_parts else "No data has been uploaded yet"
        
        capability_prompt = f"""You are a Multi-Agent ML Integration System. The user is asking about your capabilities.

CURRENT STATUS: {context}

You are a comprehensive machine learning pipeline system with the following specialized agents:

🔧 PREPROCESSING AGENT:
- Data cleaning and validation
- Missing value imputation (mean, median, mode, forward fill, backward fill)
- Outlier detection and handling (IQR, Z-score, isolation forest)
- Data normalization and standardization
- Categorical encoding (one-hot, label, target encoding)
- Duplicate removal and data type optimization

🎯 FEATURE SELECTION AGENT:
- Information Value (IV) analysis for feature importance
- Correlation analysis and multicollinearity detection
- Variance Inflation Factor (VIF) analysis
- Principal Component Analysis (PCA) for dimensionality reduction
- Feature importance ranking and selection
- Statistical feature selection methods

🤖 MODEL BUILDING AGENT:
- Classification models: Random Forest, LightGBM, XGBoost, Decision Trees
- Regression models: Linear, Ridge, Lasso, Random Forest, Gradient Boosting
- Model training with cross-validation
- Hyperparameter tuning and optimization
- Model evaluation with comprehensive metrics
- Model persistence and artifact management

🎛️ ORCHESTRATOR CAPABILITIES:
- Intelligent query routing to appropriate agents
- Full pipeline execution (preprocessing → feature selection → model building)
- Direct entry to any pipeline stage
- Session management and state persistence
- Progress tracking and real-time updates

📊 GENERAL CAPABILITIES:
- File upload support (CSV, Excel, JSON, TSV)
- Data analysis and visualization
- Code execution with error recovery
- Session-based conversations
- Artifact management and storage

EXAMPLE QUERIES YOU CAN HANDLE:
- "Build a complete ML pipeline for this data"
- "Clean my dataset and handle missing values"
- "Select the most important features using IV analysis"
- "Train a LightGBM classifier"
- "Show me data statistics and correlations"
- "Resume my analysis from yesterday"

Respond naturally and conversationally. Be helpful and mention that you can handle both individual tasks and complete end-to-end ML workflows."""

        try:
            response = ollama.chat(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant specializing in machine learning and data science. Be conversational, friendly, and informative."},
                    {"role": "user", "content": capability_prompt}
                ]
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            print(f"⚠️ LLM capability response failed: {e}")
            # If LLM fails, use fallback response
            return self._fallback_capability_response(state)

    def _fallback_capability_response(self, state: PipelineState) -> str:
        """Fallback capability response without LLM"""
        status_parts = []
        if state.raw_data is not None:
            status_parts.append(f"📊 Dataset: {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns")
        if state.cleaned_data is not None:
            status_parts.append("✅ Data preprocessed")
        if state.selected_features is not None:
            status_parts.append(f"✅ {len(state.selected_features)} features selected")
        if state.trained_model is not None:
            status_parts.append("✅ Model trained")
        
        status = "\n".join(status_parts) if status_parts else "📝 No data uploaded yet"
        
        return f"""🤖 **Multi-Agent ML Integration System**

**Current Status:**
{status}

**My Capabilities:**

🔧 **Data Preprocessing**
• Clean and validate data
• Handle missing values and outliers
• Normalize and encode features

🎯 **Feature Selection**
• Information Value (IV) analysis
• Correlation and VIF analysis
• PCA and dimensionality reduction

🤖 **Model Building**
• Train classification/regression models
• LightGBM, XGBoost, Random Forest
• Model evaluation and optimization

🎛️ **Pipeline Management**
• Full end-to-end ML workflows
• Intelligent query routing
• Session persistence and resume

**Example queries:**
• "Build a complete ML pipeline"
• "Clean this data and select features"
• "Train a LightGBM classifier"
• "What's the current pipeline status?"

How can I help you with your ML workflow today?"""

    def generate_general_response(self, query: str, state: PipelineState) -> str:
        """
        Generate natural conversational responses for general queries
        """
        query_lower = query.lower()
        
        # Check if asking about capabilities
        capability_indicators = ["what can you do", "capabilities", "help", "what are you", "how can you help"]
        if any(indicator in query_lower for indicator in capability_indicators):
            return self.generate_capability_response(state)
        
        # Check if asking about status
        status_indicators = ["status", "progress", "current state", "where are we", "what's done"]
        if any(indicator in query_lower for indicator in status_indicators):
            return self._generate_status_response(state)
        
        # Use LLM for other general responses
        try:
            context = ""
            if state.raw_data is not None:
                context = f"I have the user's dataset with {state.raw_data.shape[0]:,} rows and {state.raw_data.shape[1]} columns. "
            
            response_prompt = f"{context}The user said: '{query}'. Respond naturally and conversationally as a helpful AI assistant specializing in machine learning and data science. Keep it brief and friendly."
            
            response = ollama.chat(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "You are a friendly AI assistant for machine learning and data science. Be conversational and helpful."},
                    {"role": "user", "content": response_prompt}
                ]
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            print(f"⚠️ LLM general response failed: {e}")
        
        # Fallback responses
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
        if any(greeting in query_lower for greeting in greetings):
            return "👋 Hello! I'm your Multi-Agent ML Integration System. I can help you build complete machine learning pipelines, from data preprocessing to model training. What would you like to work on today?"
        
        return "I'm here to help with your machine learning workflow! You can ask me to preprocess data, select features, build models, or run complete ML pipelines. What would you like to do?"

    def _generate_status_response(self, state: PipelineState) -> str:
        """Generate pipeline status response"""
        status_parts = ["📊 **Pipeline Status:**"]
        
        # Data status
        if state.raw_data is not None:
            status_parts.append(f"✅ **Raw Data**: {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns")
        else:
            status_parts.append("❌ **Raw Data**: Not loaded")
        
        if state.cleaned_data is not None:
            status_parts.append(f"✅ **Cleaned Data**: {state.cleaned_data.shape[0]:,} rows × {state.cleaned_data.shape[1]} columns")
        else:
            status_parts.append("❌ **Cleaned Data**: Not processed")
        
        if state.selected_features is not None:
            status_parts.append(f"✅ **Selected Features**: {len(state.selected_features)} features")
        else:
            status_parts.append("❌ **Selected Features**: Not selected")
        
        if state.trained_model is not None:
            model_type = type(state.trained_model).__name__
            status_parts.append(f"✅ **Trained Model**: {model_type}")
        else:
            status_parts.append("❌ **Trained Model**: Not trained")
        
        # Progress indicator
        progress_steps = [
            state.raw_data is not None,
            state.cleaned_data is not None, 
            state.selected_features is not None,
            state.trained_model is not None
        ]
        completed_steps = sum(progress_steps)
        progress_bar = "🟢" * completed_steps + "⚪" * (4 - completed_steps)
        status_parts.append(f"\n**Progress**: {progress_bar} ({completed_steps}/4 steps)")
        
        # Next step suggestion
        if completed_steps == 0:
            status_parts.append("\n💡 **Next**: Upload data to get started")
        elif completed_steps == 1:
            status_parts.append("\n💡 **Next**: Clean and preprocess the data")
        elif completed_steps == 2:
            status_parts.append("\n💡 **Next**: Select important features")
        elif completed_steps == 3:
            status_parts.append("\n💡 **Next**: Train a machine learning model")
        else:
            status_parts.append("\n🎉 **Pipeline Complete!** You can now make predictions or try different models.")
        
        return "\n".join(status_parts)

    def route(self, state: PipelineState) -> str:
        """
        Hybrid routing: Fast keyword scoring with LLM fallback for ambiguous cases
        """
        if not state.user_query:
            return "preprocessing"  # Default
        
        print(f"[Orchestrator] Processing query: '{state.user_query}'")
        
        # Step 1: Try fast keyword-based classification first
        intent, confidence_info = self._classify_with_keyword_scoring(state.user_query)
        
        print(f"[Orchestrator] Keyword classification: {intent}")
        print(f"[Orchestrator] Confidence: max_score={confidence_info['max_score']:.3f}, score_diff={confidence_info['score_diff']:.3f}")
        
        # Step 2: Check if we need LLM fallback
        needs_llm_fallback = (
            confidence_info["max_score"] < 0.25 or  # Low confidence
            confidence_info["score_diff"] < 0.1     # Ambiguous (scores too close)
        )
        
        if needs_llm_fallback:
            print(f"[Orchestrator] 🤖 Low confidence or ambiguous scores, using LLM fallback")
            # Build context for LLM
            context = {
                "has_raw_data": state.raw_data is not None,
                "has_cleaned_data": state.cleaned_data is not None,
                "has_selected_features": state.selected_features is not None,
                "has_trained_model": state.trained_model is not None
            }
            intent = self.classify_intent_with_llm(state.user_query, context)
            print(f"[Orchestrator] LLM classification: {intent}")
        else:
            print(f"[Orchestrator] ⚡ High confidence keyword classification, using: {intent}")
        
        # Step 3: Route based on classified intent
        return self._route_by_intent(state, intent)

    def get_routing_explanation(self, state: PipelineState, routing_decision: str) -> str:
        """Get explanation for routing decision"""
        explanations = {
            "preprocessing": "Routing to data preprocessing - will clean and prepare your data",
            "feature_selection": "Routing to feature selection - will identify the most important features",
            "model_building": "Routing to model building - will train and evaluate ML models",
            "general_response": "Generating conversational response",
            "code_execution": "Executing custom code analysis",
            "full_pipeline": "Running complete ML pipeline from start to finish"
        }
        
        return explanations.get(routing_decision, f"Routing to {routing_decision}")


# Create global instance
orchestrator = Orchestrator()
