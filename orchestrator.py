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
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    LLM_AVAILABLE = True
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    EMBEDDINGS_AVAILABLE = False
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
        
        # Semantic intent definitions for embedding-based classification
        self.intent_definitions = {
            "preprocessing": "Data cleaning, preprocessing, preparation, transformation, missing values, outliers, duplicates, data quality, normalization, scaling, encoding, sanitization, purification, data wrangling, cleansing, standardization, imputation, outlier detection, duplicate removal, data validation, quality assurance, clean my data, prepare my data, handle missing values, remove outliers, data preparation workflow, data cleaning pipeline, normalize features, standardize data, scale features, feature normalization, feature scaling, data normalization, feature standardization, analyze data, analyze dataset, data analysis, exploratory analysis, examine data, investigate data.",
            "feature_selection": "Feature selection, feature engineering, feature importance, correlation analysis, dimensionality reduction, variable selection, attribute ranking, predictor analysis, feature extraction, variable engineering, feature scoring, mutual information, chi-square test, recursive feature elimination, principal component analysis, feature correlation, variable importance, select best features, engineer features, reduce dimensions, feature analysis.",
            "model_building": "Train machine learning models, build predictive algorithms, create classification models, develop regression models, neural network training, ensemble model creation, deep learning model development, hyperparameter optimization, model fitting, algorithm training, predictive model construction, supervised model training, unsupervised learning, model evaluation, cross-validation, model selection, algorithm comparison, model deployment preparation, train classifier, build predictor, create model, develop algorithm, use this model and print rank order table, apply existing model, use trained model, model application, model usage, existing model analysis, use this model, use the model, with this model, for this model, use existing model, show model plot, visualize model, model visualization, tree plot, decision tree plot, model tree visualization, display model tree, show tree structure, plot model tree, model plotting, existing model plot, trained model visualization, built model plot, apply this model, utilize this model, employ this model, leverage this model, work with this model, operate this model, run this model, execute this model, this model predictions, this model analysis, this model evaluation, this model performance, this model metrics, this model results, existing model usage, trained model application, built model utilization, saved model usage, loaded model application.",
            "code_execution": "Execute Python code, run custom scripts, perform statistical calculations, compute descriptive statistics, calculate correlations, generate histograms, create scatter plots, box plots, statistical testing, mathematical computations, analytical programming, code-based analysis, run code, execute script, custom analysis, statistical computation, data summary, basic statistics, correlation analysis, distribution analysis, data profiling, general programming tasks, custom calculations.",
            "general_query": "Greetings, help requests, system capabilities, status inquiries, general questions, conversational interactions, explanations, assistance, guidance, information requests, hello, hi, what can you do, how does this work, explain, describe, tell me about, what is, how to use, system information, help me, what are your capabilities, system overview, introduction."
        }
        
        # Cache for embeddings to avoid recomputation
        self._embedding_cache = {}
        self._intent_embeddings = None
        
        # Initialize intent embeddings if possible
        if EMBEDDINGS_AVAILABLE:
            self._initialize_intent_embeddings()
        
        # Use global universal pattern classifier from toolbox
        from toolbox import pattern_classifier
        self.pattern_classifier = pattern_classifier
        
        # Fallback: Exhaustive keywords for compatibility (used when embeddings unavailable)
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
            
            # Direct model building
            "skip preprocessing", "skip feature selection", "raw data", "as-is",
            "direct", "without preprocessing", "existing features", "all columns",
            "use current data", "no preprocessing", "train directly", "bypass preprocessing",

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
    
    def _initialize_intent_embeddings(self):
        """Initialize embeddings for all intent definitions"""
        try:
            print("🧠 Initializing semantic intent embeddings...")
            self._intent_embeddings = {}
            
            for intent, definition in self.intent_definitions.items():
                embedding = self._get_embedding(definition)
                if embedding is not None:
                    self._intent_embeddings[intent] = embedding
                    
            if self._intent_embeddings:
                print(f"✅ Initialized embeddings for {len(self._intent_embeddings)} intents")
            else:
                print("❌ Failed to initialize intent embeddings, falling back to keywords")
                
        except Exception as e:
            print(f"⚠️ Error initializing embeddings: {e}")
            self._intent_embeddings = None
    
    def _get_embedding(self, text: str):
        """Get embedding for text using Ollama"""
        if not EMBEDDINGS_AVAILABLE:
            return None
            
        # Check cache first
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            # Try embedding models in order of preference (best to fallback)
            embedding_models = [
                "bge-large",           # Best accuracy for intent classification
                "mxbai-embed-large",   # Good balance of accuracy and speed
                "nomic-embed-text",    # Fast and lightweight
                "all-minilm"           # Universal fallback
            ]
            
            response = None
            successful_model = None
            
            for model in embedding_models:
                try:
                    response = ollama.embeddings(
                        model=model,
                        prompt=text
                    )
                    successful_model = model
                    break  # Success, use this model
                except Exception as model_error:
                    if "not found" in str(model_error).lower():
                        continue  # Try next model
                    else:
                        print(f"⚠️ Error with {model}: {model_error}")
                        continue
            
            if response is None:
                print("❌ No embedding models available. Try: ollama pull bge-large")
                return None
            
            # Cache successful model for future reference
            if not hasattr(self, '_active_embedding_model'):
                self._active_embedding_model = successful_model
                print(f"✅ Using embedding model: {successful_model}")
            
            if 'embedding' in response:
                embedding = np.array(response['embedding'])
                self._embedding_cache[text] = embedding
                return embedding
            else:
                print(f"⚠️ No embedding in response for: {text[:50]}...")
                return None
                
        except Exception as e:
            print(f"⚠️ Error getting embedding: {e}")
            return None
    
    def _classify_with_semantic_similarity(self, query: str) -> Tuple[str, Dict]:
        """
        Semantic classification using embedding similarity
        Returns: (intent, confidence_info)
        """
        if not EMBEDDINGS_AVAILABLE or not self._intent_embeddings:
            # Fallback to keyword classification
            return self._classify_with_keyword_scoring(query)
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return self._classify_with_keyword_scoring(query)
            
            # Calculate similarities with all intents
            similarities = {}
            for intent, intent_embedding in self._intent_embeddings.items():
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    intent_embedding.reshape(1, -1)
                )[0][0]
                similarities[intent] = float(similarity)
            
            # Find best match
            best_intent = max(similarities, key=similarities.get)
            max_similarity = similarities[best_intent]
            
            # Calculate confidence metrics
            sorted_similarities = sorted(similarities.values(), reverse=True)
            second_best = sorted_similarities[1] if len(sorted_similarities) > 1 else 0
            similarity_diff = max_similarity - second_best
            
            confidence_info = {
                "max_score": max_similarity,
                "score_diff": similarity_diff,
                "method": "semantic_similarity",
                "similarities": similarities,
                "threshold_met": max_similarity > 0.4 and similarity_diff > 0.08,  # Balanced threshold for ~60-70% semantic usage
                "confident": similarity_diff > 0.08  # Clear winner threshold
            }
            
            # Check for full pipeline indicators (still use phrase matching for these)
            query_lower = query.lower()
            full_pipeline_phrases = ["complete pipeline", "full pipeline", "end to end", "build complete", "entire pipeline"]
            if any(phrase in query_lower for phrase in full_pipeline_phrases):
                return "full_pipeline", {
                    **confidence_info,
                    "method": "phrase_match_override",
                    "max_score": 1.0
                }
            
            return best_intent, confidence_info
            
        except Exception as e:
            print(f"⚠️ Error in semantic classification: {e}")
            # Fallback to keyword classification
            return self._classify_with_keyword_scoring(query)

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

    def _should_analyze_skip_patterns(self, query: str) -> bool:
        """Determine if skip pattern analysis is relevant for this query"""
        
        # Skip patterns only apply to explicit skip language
        skip_indicators = [
            "skip preprocessing", "bypass preprocessing", 
            "skip cleaning", "go straight to", 
            "directly to modeling", "skip data preparation",
            "bypass data", "skip to", "direct to"
        ]
        
        has_skip_language = any(indicator in query.lower() for indicator in skip_indicators)
        
        # CRITICAL: Enhanced existing model detection to prevent misclassification
        existing_model_indicators = [
            "use this model", "use existing model", "use the model", "use current model",
            "with this model", "apply this model", "existing model", "current model",
            "apply existing", "apply current", "use trained", "apply trained",
            "this classifier", "existing classifier", "current classifier",
            "this predictor", "existing predictor", "current predictor",
            "saved model", "built model", "previous model", 
            "apply this", "use this", "apply the model", "apply current model",
            "apply this model to", "apply previous model", "apply this predictor"
        ]
        
        has_existing_model_ref = any(ref in query.lower() for ref in existing_model_indicators)
        
        # CRITICAL: If query has existing model language, NEVER analyze skip patterns
        if has_existing_model_ref:
            print(f"[Skip Analysis] Skipping pattern analysis - existing model detected: '{query}'")
            return False
            
        # Only apply skip patterns for explicit skip requests without existing model references
        return has_skip_language

    def _classify_skip_patterns(self, query: str) -> str:
        """
        Classify skip/bypass patterns using Universal Pattern Classifier
        Returns: routing destination or None if no skip patterns detected
        """
        if not query:
            return None
            
        # Check if this query warrants skip pattern analysis
        if not self._should_analyze_skip_patterns(query):
            return None
            
        # Define skip pattern semantic definitions
        skip_intent_definitions = {
            "skip_to_modeling": "Skip to modeling, go straight to modeling, bypass preprocessing and feature selection, direct to modeling, skip all preprocessing, skip everything and build new model, immediate new model training, direct new model building, bypass data preparation entirely",
            "skip_preprocessing_to_features": "Skip preprocessing but do feature selection, bypass data cleaning but select features, skip preprocessing and select features, feature selection without preprocessing, feature engineering without cleaning, skip data preparation but analyze features",
            "skip_preprocessing_to_modeling": "Skip preprocessing and go to new modeling, bypass preprocessing for new model building, skip data cleaning and train new model, new model building without preprocessing, train new model with raw data, build new classifier without cleaning, direct new model training",
            "no_skip": "Normal pipeline, full pipeline, complete workflow, do preprocessing, clean data first, prepare data, standard pipeline, regular workflow, full data preparation, use existing model, apply existing model, use this model, use trained model, existing model analysis, current model application, model usage, model application, existing model operations, use current model, apply current model, utilize existing model"
        }
        
        # Use universal pattern classifier
        skip_result, method_used = self.pattern_classifier.classify_pattern(
            query, 
            skip_intent_definitions,
            use_case="skip_patterns"
        )
        
        if skip_result and skip_result != "no_skip":
            print(f"[Orchestrator] Skip pattern detected: {skip_result} (method: {method_used})")
            return self._route_skip_pattern(skip_result, query)
        else:
            print(f"[Orchestrator] No skip pattern detected (result: {skip_result}, method: {method_used})")
            return None

    def _OBSOLETE_llm_classify_skip_patterns(self, query: str) -> str:
        """LLM-based skip pattern classification"""
        try:
            import requests
            
            prompt = f"""
            Classify this query's skip/bypass intent:
            
            Options:
            - "skip_to_modeling": Skip everything and go directly to model building
            - "skip_preprocessing_to_features": Skip preprocessing but do feature selection  
            - "skip_preprocessing_to_modeling": Skip preprocessing and go to model building
            - "no_skip": Normal pipeline, no skipping
            
            Query: "{query}"
            
            Respond with ONLY the option name.
            """
            
            # Try Ollama first
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "qwen2.5-coder:32b-instruct-q4_K_M",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    skip_type = result.get("response", "").strip().lower()
                    
                    valid_types = ["skip_to_modeling", "skip_preprocessing_to_features", 
                                 "skip_preprocessing_to_modeling", "no_skip"]
                    
                    for valid_type in valid_types:
                        if valid_type in skip_type:
                            return valid_type
                            
            except Exception as ollama_error:
                print(f"[Orchestrator] Ollama LLM error: {ollama_error}")
                
            # Fallback to OpenAI if available
            try:
                import openai
                client = openai.OpenAI()
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0
                )
                
                skip_type = response.choices[0].message.content.strip().lower()
                
                valid_types = ["skip_to_modeling", "skip_preprocessing_to_features", 
                             "skip_preprocessing_to_modeling", "no_skip"]
                
                for valid_type in valid_types:
                    if valid_type in skip_type:
                        return valid_type
                        
            except Exception as openai_error:
                print(f"[Orchestrator] OpenAI LLM error: {openai_error}")
                
        except Exception as e:
            print(f"[Orchestrator] LLM skip pattern classification failed: {e}")
        
        return None

    def _keyword_classify_skip_patterns(self, query: str) -> str:
        """Keyword-based skip pattern classification (fallback)"""
        query_lower = query.lower()
        
        # Different skip patterns with different implications
        skip_to_modeling_patterns = [
            "skip to modeling", "go straight to modeling", "skip preprocessing and feature selection",
            "bypass preprocessing and feature selection", "direct to modeling"
        ]
        
        skip_preprocessing_patterns = [
            "skip preprocessing", "bypass preprocessing", "skip data cleaning", 
            "bypass data cleaning", "bypass preprocessing and"
        ]
        
        # Check for explicit "skip to modeling" first
        if any(pattern in query_lower for pattern in skip_to_modeling_patterns):
            return "skip_to_modeling"
        
        # Check for "skip preprocessing" - need to decide next step intelligently
        if any(pattern in query_lower for pattern in skip_preprocessing_patterns):
            # Analyze what the user wants to do after skipping preprocessing
            feature_keywords = ["feature", "select", "selection", "engineering", "importance", "correlation"]
            model_keywords = ["model", "train", "build", "classifier", "regressor", "predict", "algorithm", 
                             "random forest", "decision tree", "lgbm", "xgboost", "neural network"]
            
            has_feature_intent = any(keyword in query_lower for keyword in feature_keywords)
            has_model_intent = any(keyword in query_lower for keyword in model_keywords)
            
            if has_model_intent and not has_feature_intent:
                return "skip_preprocessing_to_modeling"
            elif has_feature_intent and not has_model_intent:
                return "skip_preprocessing_to_features"
            elif has_model_intent and has_feature_intent:
                # Both mentioned - default to feature selection first (normal pipeline order)
                return "skip_preprocessing_to_features"
            else:
                # Ambiguous - default to feature selection (safer pipeline progression)
                return "skip_preprocessing_to_features"
        
        return None

    def _route_skip_pattern(self, skip_type: str, query: str) -> str:
        """Route based on classified skip pattern"""
        if skip_type == "skip_to_modeling":
            print(f"[Orchestrator] Skip to modeling detected - routing to model_building agent")
            return "model_building"
        elif skip_type == "skip_preprocessing_to_modeling":
            print(f"[Orchestrator] Skip preprocessing + model intent detected - routing to model_building agent")
            return "model_building"
        elif skip_type == "skip_preprocessing_to_features":
            print(f"[Orchestrator] Skip preprocessing + feature intent detected - routing to feature_selection agent")
            return "feature_selection"
        elif skip_type == "no_skip":
            print(f"[Orchestrator] No skip pattern detected - continuing with normal routing")
            return None
        else:
            print(f"[Orchestrator] Unknown skip pattern: {skip_type} - continuing with normal routing")
            return None

    def _route_by_intent(self, state: PipelineState, intent: str) -> str:
        """
        Route based on classified intent and current state
        """
        # Handle skip/bypass requests with semantic classification
        skip_routing = self._classify_skip_patterns(state.user_query)
        if skip_routing:
            return skip_routing
        
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
            # ✅ DIRECT FEATURE SELECTION FIX: Allow feature selection with raw data only for explicit keywords
            if state.cleaned_data is None and state.raw_data is not None:
                # Check for explicit direct feature selection keywords
                query_lower = (state.user_query or "").lower()
                direct_fs_keywords = [
                    "direct feature selection", "skip preprocessing", "raw data", "without preprocessing",
                    "bypass preprocessing", "use raw data", "no preprocessing", "direct fs"
                ]
                
                # Only bypass preprocessing if user explicitly requests it
                if any(keyword in query_lower for keyword in direct_fs_keywords):
                    print("[Orchestrator] 🚀 Direct feature selection requested - using raw data")
                    return "feature_selection"
                else:
                    print("[Orchestrator] Need to preprocess data first")
                    return "preprocessing"
            return "feature_selection"
        
        elif intent == "model_building":
            # Check if this is actually an educational/explanatory query about ML concepts
            query_lower = (state.user_query or "").lower()
            educational_patterns = [
                "tell me about", "tell me how", "explain", "what is", "how does", "how do",
                "describe", "definition of", "meaning of", "concept of"
            ]
            
            if any(pattern in query_lower for pattern in educational_patterns):
                print("[Orchestrator] Educational query detected - routing to general response")
                return "general_response"
            
            # Check prerequisites for actual model building
            if state.raw_data is None:
                print("[Orchestrator] No data available for model building - routing to model_building agent to handle")
                return "model_building"  # Let model building agent handle the "no data" case
            
            # Check for direct model building keywords
            query_lower = (state.user_query or "").lower()
            direct_keywords = [
                "skip preprocessing", "skip feature selection", "raw data", "as-is", 
                "direct", "without preprocessing", "existing features", "all columns",
                "use current data", "no preprocessing", "train directly", "bypass preprocessing"
            ]
            
            if any(keyword in query_lower for keyword in direct_keywords):
                print("[Orchestrator] 🚀 Direct model building requested - skipping preprocessing and feature selection")
                # Use raw data as cleaned data for direct model building
                if state.cleaned_data is None:
                    state.cleaned_data = state.raw_data.copy()
                    print("[Orchestrator] Using raw data as cleaned data")
                if state.selected_features is None:
                    state.selected_features = state.raw_data.copy()
                    print("[Orchestrator] Using all columns as selected features")
                return "model_building"
            
            # Normal pipeline flow
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
            # Handle general queries - route to general_response node
            response = self.generate_general_response(state.user_query, state)
            state.last_response = response
            return "general_response"
        
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
        Semantic-first routing: Embedding similarity with keyword and LLM fallbacks
        """
        if not state.user_query:
            return "general_response"  # Do nothing until user provides intent
        
        print(f"[Orchestrator] Processing query: '{state.user_query}'")
        # Minimal hardcoded override for local testing without embeddings/LLM
        ql = state.user_query.lower().strip()
        if ql == "preprocessing" or ql.startswith("preprocessing "):
            print("[Orchestrator] 🔧 Keyword override matched: preprocessing → routing to preprocessing agent")
            return "preprocessing"
        
        # Use universal pattern classifier for main intent classification
        intent, method_used = self.pattern_classifier.classify_pattern(
            state.user_query,
            self.intent_definitions,
            use_case="intent_classification"
        )
        
        print(f"[Orchestrator] Intent classification: {intent} (method: {method_used})")
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
