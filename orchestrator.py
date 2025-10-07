#!/usr/bin/env python3
from print_to_log import print_to_log
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

# Import thread logging system
from thread_logger import get_thread_logger
# LLM imports (with optional fallback)
try:
    import ollama
    # from langchain_openai import ChatOpenAI  # Removed - using Qwen models only
    from langchain_core.messages import HumanMessage
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    LLM_AVAILABLE = True
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    EMBEDDINGS_AVAILABLE = False
    print_to_log("⚠️ LLM libraries not available, using keyword-only classification")

# Text normalization imports
try:
    from nltk.stem import WordNetLemmatizer
    import nltk
    # Download required NLTK data if not present
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print_to_log("📦 Downloading NLTK WordNet data...")
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print_to_log("📦 Downloading NLTK OMW data...")
        nltk.download('omw-1.4', quiet=True)
    
    lemmatizer = WordNetLemmatizer()
    LEMMATIZER_AVAILABLE = True
except ImportError:
    LEMMATIZER_AVAILABLE = False
    print_to_log("⚠️ NLTK not available, using basic text normalization")


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
    
    def _get_thread_logger(self, state):
        """Helper function to get thread logger from state"""
        if hasattr(state, "session_id") and state.session_id:
            session_id = state.session_id
        elif hasattr(state, "chat_session") and state.chat_session:
            session_id = state.chat_session
        else:
            return None
        
        if "_" in session_id:
            parts = session_id.split("_")
            user_id = parts[0] if len(parts) >= 1 else session_id
            thread_id = "_".join(parts[1:]) if len(parts) > 1 else session_id
        else:
            user_id = session_id
            thread_id = session_id
        
        return get_thread_logger(user_id, thread_id)
    
    def _check_prerequisites(self, state: PipelineState, intent: str) -> str:
        """
        Check data upload and target column prerequisites for all non-general intents
        Returns: "proceed" if all good, or "general_response" if prerequisites missing
        """
        # Step 1: Check if data is uploaded
        if state.raw_data is None:
            print_to_log(f"[Orchestrator] No data uploaded for intent '{intent}' - prompting for upload")
            
            # Check if user is mentioning file upload
            query_lower = (state.user_query or "").lower()
            upload_patterns = ["upload", "file", "dataset", "data", "csv", "excel"]
            
            if any(pattern in query_lower for pattern in upload_patterns):
                # User is mentioning upload - provide specific upload guidance
                state.last_response = f"""📁 Ready to help with your {intent.replace('_', ' ')} task!

To get started, please upload your dataset first.

📊 Supported formats:
• CSV files (.csv)
• Excel files (.xlsx, .xls)

📤 How to upload:
• Drag and drop your file into this chat
• Or use the attachment button to select your file

Once you upload your data, I'll help you with {self._get_intent_description(intent)}.

Ready when you are! 🚀"""
            else:
                # User didn't mention upload - general data requirement message
                state.last_response = f"""📊 Data Required for {intent.replace('_', ' ').title()}

I need a dataset to help you with {self._get_intent_description(intent)}.

📤 Please upload your data:
• Drag and drop your CSV/Excel file into this chat
• Or use the attachment button to select your file

📋 Supported formats: CSV (.csv), Excel (.xlsx, .xls)

Once your data is uploaded, I'll be ready to assist! 🚀"""
            
            return "general_response"
        
        # Step 2: Check if target column is set (for intents that need it)
        target_required_intents = ["preprocessing", "feature_selection", "model_building", "full_pipeline", "code_execution"]
        if intent in target_required_intents:
            if not hasattr(state, 'target_column') or not state.target_column:
                print_to_log(f"[Orchestrator] No target column set for intent '{intent}' - prompting for target")
                target_result = self._prompt_for_target_column(state, intent)
                if target_result != "proceed":
                    return target_result  # Return general_response if target selection needed
                # If target_result == "proceed", continue to check if target is actually set
        
        # All prerequisites met
        return "proceed"
    
    def _get_intent_description(self, intent: str) -> str:
        """Get user-friendly description of what the intent does"""
        descriptions = {
            "preprocessing": "data cleaning, preprocessing, and preparation",
            "feature_selection": "feature selection and importance analysis", 
            "model_building": "machine learning model building and evaluation",
            "code_execution": "data analysis and code execution",
            "full_pipeline": "complete machine learning pipeline from data preparation to model building"
        }
        return descriptions.get(intent, intent.replace('_', ' '))
    
    def _prompt_for_target_column(self, state: PipelineState, intent: str) -> str:
        """
        Smart target column detection and selection
        """
        if state.raw_data is None:
            state.last_response = "❌ No data available for target column selection. Please upload data first."
            return "general_response"
        
        available_columns = list(state.raw_data.columns)
        print_to_log(f"📊 Available columns for target selection: {available_columns}")
        
        # FIRST: Check if "target" column exists - use it automatically
        if "target" in available_columns:
            state.target_column = "target"
            print_to_log("🎯 Found 'target' column - using automatically")
            print_to_log(f"🎯 Set state.target_column = '{state.target_column}'")
            return "proceed"  # Continue with original intent
        
        # SECOND: Set up interactive session for target selection
        state.interactive_session = {
            'agent_type': 'target_selection',
            'session_active': True,
            'session_id': state.chat_session,
            'phase': 'target_selection',
            'original_intent': intent,  # Store current intent
            'available_columns': available_columns,
            'needs_target': True
        }
        
        # Create column list with numbering for easy selection
        column_list = []
        for i, col in enumerate(available_columns, 1):
            column_list.append(f"{i}. `{col}`")
        
        columns_text = "\n".join(column_list)
        
        state.last_response = f"""🎯 Target Column Selection Required

📊 Dataset: {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns

📋 Available Columns:
{columns_text}

💬 Please type the EXACT column name you want to use as target:

❓ Which column should be used as the target variable for prediction?

⚠️ Important: Type only the exact column name (case-sensitive)."""
        
        return "general_response"
    
    def _handle_target_selection(self, state: PipelineState) -> str:
        """
        Handle target column selection from user input
        """
        user_input = state.user_query.strip()
        available_columns = state.interactive_session.get('available_columns', [])
        original_intent = state.interactive_session.get('original_intent')
        
        print_to_log(f"🎯 [Target Selection] User input: '{user_input}'")
        print_to_log(f"🎯 [Target Selection] Original intent: {original_intent}")
        print_to_log(f"🎯 [Target Selection] Available columns: {available_columns}")
        
        # Check if user input matches exactly any column name
        if user_input in available_columns:
            # SUCCESS: Set target column and continue with original intent
            state.target_column = user_input
            state.interactive_session = None  # Clear interactive session
            print_to_log(f"✅ [Target Selection] Target column set to: {user_input}")
            
            state.last_response = f"✅ Target column set to: `{user_input}`\n\nProceeding with {original_intent.replace('_', ' ')}..."
            
            # Continue with original intent
            return self._route_by_intent(state, original_intent)
        else:
            # FAILURE: Column not found, ask again
            print_to_log(f"❌ [Target Selection] Column '{user_input}' not found in dataset")
            
            state.last_response = f"""❌ Column not found in dataset

💬 Please type the EXACT column name (case-sensitive):

⚠️ Important: The column name must match exactly including case."""
            
            return "general_response"
    
    def _handle_mode_selection(self, state: PipelineState) -> str:
        """
        Handle mode selection from user input (fast vs slow)
        """
        user_input = state.user_query.strip().lower()
        original_intent = state.interactive_session.get('original_intent', 'full_pipeline')
        
        print_to_log(f"🚀 [Mode Selection] User input: '{user_input}'")
        print_to_log(f"🚀 [Mode Selection] Original intent: {original_intent}")
        
        if 'fast' in user_input or 'automated' in user_input:
            # SUCCESS: Fast mode selected - handle directly in orchestrator
            
            # CRITICAL: Preserve original user query before it gets overwritten
            original_query = state.interactive_session.get('original_query', state.user_query)
            print_to_log(f"⚡ [Mode Selection] Fast mode selected - preserving original query: '{original_query}'")
            
            # Store original query in preprocessing state for automated pipeline agent
            if not hasattr(state, 'preprocessing_state') or state.preprocessing_state is None:
                state.preprocessing_state = {}
            state.preprocessing_state['original_user_query'] = original_query
            
            state.interactive_session = None  # Clear interactive session
            print_to_log(f"⚡ [Mode Selection] Calling automated pipeline with preserved query")
            
            try:
                # Import and call automated pipeline agent directly
                from automated_pipeline_agent import automated_pipeline_agent
                
                # Call automated pipeline
                result_state = automated_pipeline_agent(state)
                
                # The automated pipeline agent handles its own response
                print_to_log(f"✅ [Mode Selection] Automated pipeline completed successfully")
                
                # Update the current state with results
                state.last_response = result_state.last_response
                state.last_error = result_state.last_error
                state.artifacts = result_state.artifacts
                state.pending_file_uploads = result_state.pending_file_uploads
                
                return "general_response"  # Return the automated pipeline's response
                
            except Exception as e:
                print_to_log(f"❌ [Mode Selection] Automated pipeline failed: {e}")
                state.last_response = f"❌ Fast pipeline failed: {str(e)}"
                return "general_response"
            
        elif 'slow' in user_input or 'interactive' in user_input:
            # SUCCESS: Slow mode selected - update interactive session for preprocessing
            state.interactive_session['phase'] = 'slow_mode_selected'
            state.interactive_session['mode_selected'] = 'slow'
            state.interactive_session['needs_mode_selection'] = False
            print_to_log(f"🎛️ [Mode Selection] Slow mode selected - updated interactive session")
            
            state.last_response = f"🎛️ Slow Mode Selected - Starting interactive preprocessing..."
            
            # Route to preprocessing for interactive mode
            return "preprocessing"
            
        else:
            # FAILURE: Invalid mode selection, ask again
            print_to_log(f"❌ [Mode Selection] Invalid mode selection: '{user_input}'")
            
            state.last_response = f"""❓ Please choose your pipeline mode:

⚡ Fast Mode: Type `fast` for automated pipeline
🎛️ Slow Mode: Type `slow` for interactive mode

💬 Choose: Type `fast` or `slow`"""
            
            return "general_response"
    
    def _prompt_for_mode_selection(self, state: PipelineState) -> str:
        """
        Prompt user for mode selection (fast vs slow)
        """
        # Set up interactive session for mode selection
        state.interactive_session = {
            'agent_type': 'mode_selection',
            'session_active': True,
            'session_id': state.chat_session,
            'phase': 'mode_selection',
            'original_intent': 'full_pipeline',
            'original_query': state.user_query,  # CRITICAL: Store original user query
            'needs_mode_selection': True
        }
        
        state.last_response = f"""🚀 Choose Your ML Pipeline Mode

⚡ Fast Mode (Automated):
• Complete ML pipeline without interaction
• AI handles all preprocessing decisions
• Get results in 2-3 minutes

🎛️ Slow Mode (Interactive):
• Step-by-step guided process
• Review and approve each phase
• Full control over decisions

💬 Choose: Type `fast` or `slow`"""
        
        return "general_response"
    
    def _prompt_for_preprocessing_confirmation(self, state: PipelineState) -> str:
        """
        Prompt user for confirmation when model building is requested but no cleaned data exists
        """
        # Set up interactive session for preprocessing confirmation
        state.interactive_session = {
            'agent_type': 'preprocessing_confirmation',
            'session_active': True,
            'session_id': state.chat_session,
            'phase': 'preprocessing_confirmation',
            'original_intent': 'model_building',
            'original_query': state.user_query,
            'needs_preprocessing_confirmation': True
        }
        
        state.last_response = f"""⚠️ Preprocessing Required for Better Results

🔍 Current Situation:
• You want to build a model, but your data hasn't been preprocessed yet
• Raw data may contain outliers, missing values, or unoptimized features
• This could lead to poor model performance or training issues

🎯 Recommendation:
It's highly recommended to preprocess your data first for better model accuracy and reliability.

💬 Do you want to still build model with out preprocessing?
• Type `Yes` - Build model with raw data anyway
• Type `No` - Clean data first (recommended)
"""
        
        return "general_response"
    
    def _handle_preprocessing_confirmation(self, state: PipelineState) -> str:
        """
        Handle user response to preprocessing confirmation
        """
        user_input = state.user_query.strip().lower()
        original_intent = state.interactive_session.get('original_intent', 'model_building')
        
        print_to_log(f"🔧 [Preprocessing Confirmation] User input: '{user_input}'")
        print_to_log(f"🔧 [Preprocessing Confirmation] Original intent: {original_intent}")
        
        if 'skip' in user_input or 'no' in user_input or 'clean' in user_input or 'No' in user_input:
            # SUCCESS: User wants to preprocess first
            state.interactive_session = None  # Clear interactive session
            print_to_log(f"✅ [Preprocessing Confirmation] User chose to preprocess first")
            
            state.last_response = f"✅ Great choice! Starting data preprocessing to optimize your model performance..."
            
            # Route to preprocessing
            return "preprocessing"
            
        elif 'Yes' in user_input or 'build' in user_input or 'yes' in user_input:
            # User wants to skip preprocessing and build model with raw data
            state.interactive_session = None  # Clear interactive session
            print_to_log(f"⚠️ [Preprocessing Confirmation] User chose to skip preprocessing")
            
            # Use raw data as cleaned data for model building
            if state.cleaned_data is None:
                state.cleaned_data = state.raw_data.copy()
                print_to_log("[Preprocessing Confirmation] Using raw data as cleaned data")
            
            # Use all columns as selected features
            if state.selected_features is None:
                state.selected_features = state.raw_data.copy()
                print_to_log("[Preprocessing Confirmation] Using all columns as selected features")
            
            state.last_response = f"⚠️ Proceeding with raw data modeling (preprocessing skipped)..."
            
            # Route to model building
            return "model_building"
            
        else:
            # FAILURE: Invalid response, ask again
            print_to_log(f"❌ [Preprocessing Confirmation] Invalid response: '{user_input}'")
            
            state.last_response = f"""❓ Please choose one of the options:

💬 Valid responses:
• Type `Yes` - Build model with raw data anyway
• Type `No` - Clean data first (recommended)

⚠️ Your response '{user_input}' was not recognized. Please try again."""
            
            return "general_response"
    
    def _extract_target_from_query(self, query: str, available_columns: list) -> str:
        """
        Extract target column from user query using pattern matching
        Handles: "use revenue as target", "target price", "predict sales", etc.
        """
        if not query or not available_columns:
            return None
        
        query_lower = query.lower().strip()
        
        # Pattern 1: "target column_name" - exact format
        if query_lower.startswith('target '):
            target_candidate = query[7:].strip().split()[0]  # Get first word after "target"
            match = self._fuzzy_match_column(target_candidate, available_columns)
            if match:
                print_to_log(f"🎯 [Target Extraction] Pattern 1 match: '{target_candidate}' → '{match}'")
                return match
        
        # Pattern 2: "set target to column_name" or "set target column_name"
        if 'set target' in query_lower:
            parts = query_lower.split('set target')
            if len(parts) > 1:
                target_part = parts[1].strip()
                if target_part.startswith('to '):
                    target_part = target_part[3:].strip()
                target_candidate = target_part.split()[0] if target_part.split() else ""
                match = self._fuzzy_match_column(target_candidate, available_columns)
                if match:
                    print_to_log(f"🎯 [Target Extraction] Pattern 2 match: '{target_candidate}' → '{match}'")
                    return match
        
        # Pattern 3: "use column_name as target"
        if 'as target' in query_lower:
            parts = query_lower.split('as target')[0]
            if 'use ' in parts:
                target_candidate = parts.split('use ')[-1].strip()
                match = self._fuzzy_match_column(target_candidate, available_columns)
                if match:
                    print_to_log(f"🎯 [Target Extraction] Pattern 3 match: '{target_candidate}' → '{match}'")
                    return match
        
        # Pattern 4: "predict column_name" or "predicting column_name"
        predict_patterns = ['predict ', 'predicting ', 'prediction of ', 'forecasting ', 'forecast ']
        for pattern in predict_patterns:
            if pattern in query_lower:
                parts = query_lower.split(pattern)
                if len(parts) > 1:
                    target_candidate = parts[1].strip().split()[0]  # Get first word after pattern
                    match = self._fuzzy_match_column(target_candidate, available_columns)
                    if match:
                        print_to_log(f"🎯 [Target Extraction] Pattern 4 match: '{target_candidate}' → '{match}'")
                        return match
        
        # Pattern 5: Direct column name mention with fuzzy matching
        # Find all columns that have fuzzy matches in the query
        fuzzy_matches = []
        for col in available_columns:
            if self._is_fuzzy_match(col.lower(), query_lower):
                fuzzy_matches.append(col)
        
        # If exactly one fuzzy match, use it
        if len(fuzzy_matches) == 1:
            print_to_log(f"🎯 [Target Extraction] Single fuzzy match: '{fuzzy_matches[0]}'")
            return fuzzy_matches[0]
        
        return None
    
    def _fuzzy_match_column(self, candidate: str, available_columns: list) -> str:
        """
        Find the best fuzzy match for a candidate column name
        """
        candidate_lower = candidate.lower().strip()
        
        # Exact match first
        for col in available_columns:
            if col.lower() == candidate_lower:
                return col
        
        # Partial match (candidate is substring of column)
        for col in available_columns:
            if candidate_lower in col.lower():
                return col
        
        # Reverse partial match (column is substring of candidate)
        for col in available_columns:
            if col.lower() in candidate_lower:
                return col
        
        return None
    
    def _is_fuzzy_match(self, column_name: str, query: str) -> bool:
        """
        Check if a column name has a fuzzy match in the query
        """
        # Direct substring match
        if column_name in query:
            return True
        
        # Clean column name and check
        clean_col = column_name.replace('_', ' ').replace('-', ' ')
        if clean_col in query:
            return True
        
        return False
    
    def _classify_with_single_llm(self, query: str) -> Dict[str, Any]:
        """
        Single LLM call for both data science relevance check and intent classification
        Returns JSON with: {is_data_science: bool, intent: string, confidence: float}
        """
        if not LLM_AVAILABLE:
            print_to_log("⚠️ LLM not available, using fallback")
            return {
                "is_data_science": True,
                "intent": "general_query", 
                "confidence": 0.5,
                "explanation":""
            }
        
        
        classification_prompt = f"""You are a data science assistant classifier. Your job is to:
1. Determine if the query is related to data science/machine learning
2. If yes, This step is of very high priority. classify the intent into one of the predefined categories but be very sure about this classification. 
3. Provide a confidence score
4. provide an explanation for classification

DATA SCIENCE RELEVANCE - Consider "yes" for:
- Data preprocessing, cleaning, transformation, missing values, outliers
- Feature selection, feature engineering, correlation analysis
- Machine learning model building, training, evaluation, prediction
- Statistical analysis, data visualization, plotting
- Data exploration, descriptive statistics
- ML algorithms (regression, classification, clustering, deep learning)
- Model evaluation metrics, performance analysis
- Data analysis programming (pandas, sklearn, etc.)
- Business analytics, A/B testing, hypothesis testing

Consider "no" for:
- General programming unrelated to data (web dev, mobile apps)
- Personal questions, casual conversation, weather, news
- General knowledge unrelated to data/ML
- Technical support for non-data tools
- Creative writing, literature, gaming (unless data analysis related)

INTENT CATEGORIES (if data science related):
1. "preprocessing" - Data cleaning, preparation, transformation
2. "feature_selection" - Feature selection, importance analysis, dimensionality reduction  
3. "model_building" - Model training, evaluation, prediction, visualization
4. "general_query" - Questions, explanations, greetings, capabilities
5. "code_execution" - Data analysis code, calculations, plotting
6. "full_pipeline" - Complete ML pipeline from start to finish

USER QUERY: "{query}"

Respond with ONLY a valid JSON object in this exact format:
{{"is_data_science": true/false, "intent": "category_name", "confidence": 0.0-1.0, "explanation": "explanation for classification"}}

Examples:
- "Hello, what can you do?" → {{"is_data_science": true, "intent": "general_query", "confidence": 0.9, "explanation": "Greetings, help requests, system capabilities, status inquiries, general questions, conversational interactions, explanations, assistance, guidance, information requests, hello, hi, what can you do, how does this work, explain, describe, tell me about, what is, how to use, system information, help me, what are your capabilities, system overview, introduction, educational questions about machine learning concepts, what are saddle points, what is gradient descent, explain overfitting, what is bias variance tradeoff, describe neural networks, what are support vector machines, explain random forests, what is cross validation, describe regularization, what are hyperparameters, explain backpropagation, what is feature engineering, describe ensemble methods, what are decision trees, explain logistic regression, what is clustering, describe dimensionality reduction, what are outliers, explain normalization, what is correlation, describe statistical concepts, machine learning theory questions, algorithm explanations, concept clarifications, theoretical understanding, academic explanations, educational content about AI ML concepts, correlation coefficient calculation, correlation coefficient value, calculate correlation, mathematical correlation, statistical correlation formula, pearson correlation, equation correlation, if I have equation, what would correlation be, correlation between x and y, theoretical correlation questions, statistical formula questions, mathematical calculation questions."}}
- "Clean this data" → {{"is_data_science": true, "intent": "preprocessing", "confidence": 0.95, "explanation": "Data cleaning, preprocessing, preparation, transformation, missing values, outliers, duplicates, data quality, normalization, scaling, encoding, sanitization, purification, data wrangling, cleansing, standardization, imputation, outlier detection, duplicate removal, data validation, quality assurance, clean my data, prepare my data, handle missing values, remove outliers, data preparation workflow, data cleaning pipeline, normalize features, standardize data, scale features, feature normalization, feature scaling, data normalization, feature standardization, analyze data, analyze dataset, data analysis, exploratory analysis, examine data, investigate data."}}
- "What's the weather?" → {{"is_data_science": false, "intent": "general_query", "confidence": 0.9, "explanation": "General questions, greetings, capability questions, or unclear intent"}}
- "Train a model" → {{"is_data_science": true, "intent": "model_building", "confidence": 0.95, "explanation": "Model training, evaluation, prediction, visualization"}}"""

        try:
            response = ollama.chat(
                model=self.default_model,
                messages=[{"role": "user", "content": classification_prompt}]
            )
            
            result_text = response["message"]["content"].strip()
            print_to_log(f"🤖 LLM Response: {result_text}")
            
            # Parse JSON response
            import json
            try:
                result = json.loads(result_text)
                
                # Validate required fields
                if not all(key in result for key in ["is_data_science", "intent", "confidence", "explanation"]):
                    raise ValueError("Missing required fields in JSON response")
                
                # Validate intent is in allowed list
                valid_intents = ["preprocessing", "feature_selection", "model_building", "general_query", "code_execution", "full_pipeline"]
                if result["intent"] not in valid_intents:
                    print_to_log(f"⚠️ Invalid intent '{result['intent']}', defaulting to general_query")
                    result["intent"] = "general_query"
                
                # Ensure confidence is between 0 and 1
                result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
                
                print_to_log(f"🎯 Parsed classification: {result}")
                return result
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print_to_log(f"⚠️ Failed to parse JSON response: {e}")
                # Fallback parsing
                return self._fallback_classify_intent(query)
                
        except Exception as e:
            print_to_log(f"⚠️ LLM classification failed: {e}")
            return self._fallback_classify_intent(query)
    
    
    
    def __init__(self):
        self.default_model = os.getenv("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
        
        # Semantic intent definitions for embedding-based classification
        self.intent_definitions = {
            "preprocessing": "Data cleaning, preprocessing, preparation, transformation, missing values, outliers, duplicates, data quality, normalization, scaling, encoding, sanitization, purification, data wrangling, cleansing, standardization, imputation, outlier detection, duplicate removal, data validation, quality assurance, clean my data, prepare my data, handle missing values, remove outliers, data preparation workflow, data cleaning pipeline, normalize features, standardize data, scale features, feature normalization, feature scaling, data normalization, feature standardization, analyze data, analyze dataset, data analysis, exploratory analysis, examine data, investigate data.",
            "feature_selection": "Feature selection, feature engineering, feature importance, data correlation analysis, dimensionality reduction, variable selection, attribute ranking, predictor analysis, feature extraction, variable engineering, feature scoring, mutual information, chi-square test, recursive feature elimination, principal component analysis, feature correlation matrix, variable importance, select best features, engineer features, reduce dimensions, feature analysis, analyze correlations in dataset, correlation matrix, feature correlation analysis, which features are correlated, find correlated features, remove correlated features, correlation between features.",
            "model_building": "Train machine learning models, build predictive algorithms, create classification models, develop regression models, neural network training, ensemble model creation, deep learning model development, hyperparameter optimization, model fitting, algorithm training, predictive model construction, supervised model training, unsupervised learning, model evaluation, cross-validation, model selection, algorithm comparison, model deployment preparation, train classifier, build predictor, create model, develop algorithm, use this model and print rank order table, apply existing model, use trained model, model application, model usage, existing model analysis, use this model, use the model, with this model, for this model, use existing model, show model plot, visualize model, model visualization, tree plot, decision tree plot, model tree visualization, display model tree, show tree structure, plot model tree, model plotting, existing model plot, trained model visualization, built model plot, apply this model, utilize this model, employ this model, leverage this model, work with this model, operate this model, run this model, execute this model, this model predictions, this model analysis, this model evaluation, this model performance, this model metrics, this model results, existing model usage, trained model application, built model utilization, saved model usage, loaded model application.",
            "code_execution": "Execute Python code, run custom scripts, perform statistical calculations, compute descriptive statistics, calculate correlations, generate histograms, create scatter plots, box plots, statistical testing, mathematical computations, analytical programming, code-based analysis, run code, execute script, custom analysis, statistical computation, data summary, basic statistics, correlation analysis, distribution analysis, data profiling, general programming tasks, custom calculations.",
            "general_query": "Greetings, help requests, system capabilities, status inquiries, general questions, conversational interactions, explanations, assistance, guidance, information requests, hello, hi, what can you do, how does this work, explain, describe, tell me about, what is, how to use, system information, help me, what are your capabilities, system overview, introduction, educational questions about machine learning concepts, what are saddle points, what is gradient descent, explain overfitting, what is bias variance tradeoff, describe neural networks, what are support vector machines, explain random forests, what is cross validation, describe regularization, what are hyperparameters, explain backpropagation, what is feature engineering, describe ensemble methods, what are decision trees, explain logistic regression, what is clustering, describe dimensionality reduction, what are outliers, explain normalization, what is correlation, describe statistical concepts, machine learning theory questions, algorithm explanations, concept clarifications, theoretical understanding, academic explanations, educational content about AI ML concepts, correlation coefficient calculation, correlation coefficient value, calculate correlation, mathematical correlation, statistical correlation formula, pearson correlation, equation correlation, if I have equation, what would correlation be, correlation between x and y, theoretical correlation questions, statistical formula questions, mathematical calculation questions."
        }
        
        # Cache for embeddings to avoid recomputation
        self._embedding_cache = {}
        self._intent_embeddings = None
        
        # Initialize intent embeddings if possible
        if EMBEDDINGS_AVAILABLE:
            self._initialize_intent_embeddings()
        
        # Use global universal pattern classifier from toolbox (lazy import)
        try:
            from toolbox import pattern_classifier
            if pattern_classifier is None:
                print_to_log("⚠️ Pattern classifier not initialized, creating fallback")
                from toolbox import UniversalPatternClassifier
                self.pattern_classifier = UniversalPatternClassifier()
            else:
                self.pattern_classifier = pattern_classifier
                print_to_log("✅ Pattern classifier initialized successfully")
        except ImportError:
            print_to_log("⚠️ Could not import pattern classifier, creating fallback")
            from toolbox import UniversalPatternClassifier
            self.pattern_classifier = UniversalPatternClassifier()
        
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
            print_to_log("🧠 Initializing semantic intent embeddings...")
            self._intent_embeddings = {}
            
            for intent, definition in self.intent_definitions.items():
                embedding = self._get_embedding(definition)
                if embedding is not None:
                    self._intent_embeddings[intent] = embedding
                    
            if self._intent_embeddings:
                print_to_log(f"✅ Initialized embeddings for {len(self._intent_embeddings)} intents")
            else:
                print_to_log("❌ Failed to initialize intent embeddings, falling back to keywords")
                
        except Exception as e:
            print_to_log(f"⚠️ Error initializing embeddings: {e}")
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
                        print_to_log(f"⚠️ Error with {model}: {model_error}")
                        continue
            
            if response is None:
                print_to_log("❌ No embedding models available. Try: ollama pull bge-large")
                return None
            
            # Cache successful model for future reference
            if not hasattr(self, '_active_embedding_model'):
                self._active_embedding_model = successful_model
                print_to_log(f"✅ Using embedding model: {successful_model}")
            
            if 'embedding' in response:
                embedding = np.array(response['embedding'])
                self._embedding_cache[text] = embedding
                return embedding
            else:
                print_to_log(f"⚠️ No embedding in response for: {text[:50]}...")
                return None
                
        except Exception as e:
            print_to_log(f"⚠️ Error getting embedding: {e}")
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
            print_to_log(f"⚠️ Error in semantic classification: {e}")
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


    def _fallback_classify_intent(self, query: str) -> Dict[str, Any]:
        """Fallback keyword-based classification returning JSON structure"""
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
            return {
                "is_data_science": True,
                "intent": "full_pipeline",
                "confidence": 0.9,
                "explanation": "Complete ML pipeline detected based on keyword matching"
            }
        
        # Find the category with the highest score
        best_category = "general_query"
        best_score = 0
        if max(scores.values()) > 0:
            best_category = max(scores, key=scores.get)
            best_score = scores[best_category]
        
        # Check for non-data science keywords
        non_data_science_keywords = ["weather", "news", "entertainment", "sports", "gaming", "cooking", "travel"]
        non_data_science_score = sum(1 for kw in non_data_science_keywords if kw in query_lower)
        
        # Determine if data science related
        total_data_science_score = sum(scores.values())
        is_data_science = total_data_science_score > non_data_science_score
        
        # Calculate confidence
        max_possible_score = 3  # Rough estimate
        confidence = min(0.8, max(0.3, best_score / max_possible_score)) if best_score > 0 else 0.5
        
        # Create explanation
        explanations = {
            "preprocessing": "Data cleaning, preprocessing, preparation, transformation based on keyword matching",
            "feature_selection": "Feature selection, importance analysis, dimensionality reduction based on keyword matching", 
            "model_building": "Model training, evaluation, prediction, visualization based on keyword matching",
            "general_query": "General questions, greetings, capability questions based on keyword matching",
            "code_execution": "Data analysis code, calculations, plotting based on keyword matching",
            "full_pipeline": "Complete ML pipeline from start to finish based on keyword matching"
        }
        
        result = {
            "is_data_science": is_data_science,
            "intent": best_category,
            "confidence": confidence,
            "explanation": explanations.get(best_category, "Fallback classification based on keyword matching")
        }
        
        print_to_log(f"🔧 Fallback classification: {result}")
        return result

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

    def _classify_direct_feature_selection(self, query: str) -> bool:
        """
        BGE-based classification to determine if a feature selection query should bypass preprocessing.
        
        Args:
            query: User query string
            
        Returns:
            bool: True if this is a direct feature selection request (skip preprocessing)
        """
        try:
            # Define examples for direct vs standard approaches (without "feature selection" noise)
            direct_fs_examples = [
                "skip preprocessing",
                "bypass preprocessing", 
                "directly on raw data",
                "use raw data",
                "without preprocessing",
                "no preprocessing needed",
                "skip data cleaning",
                "bypass data preparation",
                "on unprocessed data",
                "raw data analysis",
                "immediately",
                "skip cleaning step",
                "directly",
                "straight to analysis",
                "without cleaning",
                "raw analysis",
                "immediate analysis",
                "no cleaning required",
                "bypass cleaning"
            ]
            
            standard_fs_examples = [
                "analyze",
                "process data first", 
                "clean the data",
                "prepare data",
                "standard process",
                "normal workflow",
                "typical approach",
                "regular analysis",
                "standard procedure",
                "clean first",
                "prepare first",
                "process then analyze",
                "standard analysis",
                "normal process",
                "clean and analyze",
                "prepare and select"
            ]
            
            # Use BGE embeddings if available
            if EMBEDDINGS_AVAILABLE and hasattr(self, '_get_embedding'):
                print_to_log(f"🧠 [Direct FS BGE] Classifying query: '{query}'")
                
                # Get query embedding
                query_embedding = self._get_embedding(query)
                if query_embedding is None:
                    print_to_log("⚠️ [Direct FS BGE] Failed to get query embedding, using keyword fallback")
                    return self._classify_direct_fs_keywords(query)
                
                # Get embeddings for examples (with caching)
                direct_embeddings = []
                standard_embeddings = []
                
                for example in direct_fs_examples:
                    emb = self._get_embedding(example)
                    if emb is not None:
                        direct_embeddings.append(emb)
                
                for example in standard_fs_examples:
                    emb = self._get_embedding(example)
                    if emb is not None:
                        standard_embeddings.append(emb)
                
                if not direct_embeddings or not standard_embeddings:
                    print_to_log("⚠️ [Direct FS BGE] Failed to get example embeddings, using keyword fallback")
                    return self._classify_direct_fs_keywords(query)
                
                # Calculate average similarities
                direct_similarities = []
                for emb in direct_embeddings:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        emb.reshape(1, -1)
                    )[0][0]
                    direct_similarities.append(float(similarity))
                
                standard_similarities = []
                for emb in standard_embeddings:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        emb.reshape(1, -1)
                    )[0][0]
                    standard_similarities.append(float(similarity))
                
                # Calculate average similarities
                avg_direct_similarity = np.mean(direct_similarities)
                avg_standard_similarity = np.mean(standard_similarities)
                
                # Determine classification
                is_direct = avg_direct_similarity > avg_standard_similarity
                confidence_diff = abs(avg_direct_similarity - avg_standard_similarity)
                
                print_to_log(f"🔍 [Direct FS BGE] Similarity scores:")
                print_to_log(f"   Direct FS: {avg_direct_similarity:.3f}")
                print_to_log(f"   Standard FS: {avg_standard_similarity:.3f}")
                print_to_log(f"   Confidence diff: {confidence_diff:.3f}")
                
                # Use BGE result if confidence is high enough
                if confidence_diff > 0.01:  # Minimum confidence threshold (lowered from 0.05)
                    result = is_direct
                    method = "BGE"
                    print_to_log(f"🎯 [Direct FS BGE] Classified as {'DIRECT' if result else 'STANDARD'} (confidence: {confidence_diff:.3f})")
                else:
                    print_to_log(f"⚠️ [Direct FS BGE] Low confidence ({confidence_diff:.3f} < 0.05), using keyword fallback")
                    result = self._classify_direct_fs_keywords(query)
                    method = "BGE+Keyword"
                
                print_to_log(f"✅ [Direct FS {method}] Final result: {'DIRECT' if result else 'STANDARD'}")
                return result
                
            else:
                print_to_log("⚠️ [Direct FS BGE] BGE embeddings not available, using keyword fallback")
                return self._classify_direct_fs_keywords(query)
                
        except Exception as e:
            print_to_log(f"❌ [Direct FS BGE] Classification error: {e}")
            print_to_log("🔧 [Direct FS BGE] Falling back to keyword classification")
            return self._classify_direct_fs_keywords(query)
    
    def _classify_direct_fs_keywords(self, query: str) -> bool:
        """
        Keyword-based fallback for direct feature selection classification
        
        Args:
            query: User query string
            
        Returns:
            bool: True if this is a direct feature selection request
        """
        query_lower = query.lower()
        
        # Strong indicators for direct feature selection (skip preprocessing)
        direct_fs_keywords = [
            "direct feature selection", "skip preprocessing", "raw data", "without preprocessing",
            "bypass preprocessing", "use raw data", "no preprocessing", "direct fs",
            "skip cleaning", "bypass cleaning", "unprocessed data", "immediate feature",
            "skip data preparation", "raw feature analysis", "without cleaning",
            "skip data cleaning", "bypass data preparation", "raw analysis",
            "directly", "feature selection directly", "directly feature selection",
            "straight to features", "immediate analysis"
        ]
        
        # Check for explicit direct keywords
        has_direct_keywords = any(keyword in query_lower for keyword in direct_fs_keywords)
        
        print_to_log(f"🔍 [Direct FS Keyword] Query: '{query}'")
        print_to_log(f"🔍 [Direct FS Keyword] Has direct keywords: {has_direct_keywords}")
        
        if has_direct_keywords:
            matched_keywords = [kw for kw in direct_fs_keywords if kw in query_lower]
            print_to_log(f"🎯 [Direct FS Keyword] Matched keywords: {matched_keywords}")
        
        return has_direct_keywords

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
            print_to_log(f"[Skip Analysis] Skipping pattern analysis - existing model detected: '{query}'")
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
            "skip_preprocessing_to_modeling": "Skip preprocessing and go to new modeling, bypass preprocessing for new model building, skip data cleaning and train new model, new model building without preprocessing, train new model with raw data, build new classifier without cleaning, direct new model training, skip preprocessing and feature selection steps, skip preprocessing / feature selection, bypass preprocessing and feature selection, skip both preprocessing and features",
            "no_skip": "Normal pipeline, full pipeline, complete workflow, do preprocessing, clean data first, prepare data, standard pipeline, regular workflow, full data preparation, use existing model, apply existing model, use this model, use trained model, existing model analysis, current model application, model usage, model application, existing model operations, use current model, apply current model, utilize existing model, use this built tree, use this tree, use the tree, with this model, for this model, with this tree, for this tree, apply this model, apply existing model, use this decision tree, utilize this model, employ this model, leverage this model, work with this model, operate this model, run this model, execute this model, show plot, visualize tree, display tree, show decision tree plot, plot decision tree, tree visualization, model tree plot, existing model plot, trained model plot, built model plot, show model plot, visualize existing model, display existing model, model visualization, tree structure, decision tree structure, plot model tree, model plotting, existing model visualization, trained model visualization, built model visualization, show tree structure, display tree structure, plot existing model, visualize this model, display this model, show this model, plot this model, tree plot for existing, model plot for existing, existing tree plot, current model plot, previous model plot, saved model plot, use this built tree and show, use this tree and show, use this model and show, apply this model and show, use existing model and show, show me the decision tree plot, show me the tree plot, show me the model plot, show me the existing model plot, show me this model plot, show me this tree plot, show me this decision tree plot, use this built tree and show me, use this tree and show me, use this model and show me, apply this model and show me, use existing model and show me, use this built tree and show me the decision tree plot, use this tree and show me the decision tree plot, use this model and show me the decision tree plot, apply this model and show me the decision tree plot, use existing model and show me the decision tree plot, use this built tree and show me the tree plot, use this tree and show me the tree plot, use this model and show me the tree plot, apply this model and show me the tree plot, use existing model and show me the tree plot, use this built tree and show me the model plot, use this tree and show me the model plot, use this model and show me the model plot, apply this model and show me the model plot, use existing model and show me the model plot, use this built tree and show me the existing model plot, use this tree and show me the existing model plot, use this model and show me the existing model plot, apply this model and show me the existing model plot, use existing model and show me the existing model plot, use this built tree and show me the trained model plot, use this tree and show me the trained model plot, use this model and show me the trained model plot, apply this model and show me the trained model plot, use existing model and show me the trained model plot, use this built tree and show me the built model plot, use this tree and show me the built model plot, use this model and show me the built model plot, apply this model and show me the built model plot, use existing model and show me the built model plot, use this built tree and show me the current model plot, use this tree and show me the current model plot, use this model and show me the current model plot, apply this model and show me the current model plot, use existing model and show me the current model plot, use this built tree and show me the previous model plot, use this tree and show me the previous model plot, use this model and show me the previous model plot, apply this model and show me the previous model plot, use existing model and show me the previous model plot, use this built tree and show me the saved model plot, use this tree and show me the saved model plot, use this model and show me the saved model plot, apply this model and show me the saved model plot, use existing model and show me the saved model plot"
        }
        
        # Use universal pattern classifier
        if self.pattern_classifier is None:
            print_to_log("⚠️ Pattern classifier is None, using fallback skip detection")
            skip_result = False
            method_used = "fallback"
        else:
            skip_result, method_used = self.pattern_classifier.classify_pattern(
                query, 
                skip_intent_definitions,
                use_case="skip_patterns"
            )
        
        if skip_result and skip_result != "no_skip":
            print_to_log(f"[Orchestrator] Skip pattern detected: {skip_result} (method: {method_used})")
            return self._route_skip_pattern(skip_result, query)
        else:
            print_to_log(f"[Orchestrator] No skip pattern detected (result: {skip_result}, method: {method_used})")
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
                print_to_log(f"[Orchestrator] Ollama LLM error: {ollama_error}")
                
            # Use keyword fallback instead of OpenAI
            print_to_log(f"[Orchestrator] Using keyword fallback for skip pattern classification")
            return self._keyword_classify_skip_patterns(query)
                
        except Exception as e:
            print_to_log(f"[Orchestrator] LLM skip pattern classification failed: {e}")
        
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
            # CRITICAL FIX: Check if user explicitly wants to skip BOTH preprocessing AND feature selection
            explicit_skip_both = any(skip_both in query_lower for skip_both in [
                "skip preprocessing / feature selection", 
                "skip preprocessing and feature selection",
                "skip preprocessing / feature",
                "skip preprocessing and feature", 
                "bypass preprocessing / feature",
                "bypass preprocessing and feature",
                "skip both preprocessing",
                "skip preprocessing & feature"
            ])
            
            if explicit_skip_both:
                print_to_log(f"[Orchestrator] Explicit skip BOTH preprocessing AND feature selection detected")
                return "skip_preprocessing_to_modeling"
            
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
            print_to_log(f"[Orchestrator] Skip to modeling detected - routing to model_building agent")
            return "model_building"
        elif skip_type == "skip_preprocessing_to_modeling":
            print_to_log(f"[Orchestrator] Skip preprocessing + model intent detected - routing to model_building agent")
            return "model_building"
        elif skip_type == "skip_preprocessing_to_features":
            print_to_log(f"[Orchestrator] Skip preprocessing + feature intent detected - routing to feature_selection agent")
            return "feature_selection"
        elif skip_type == "no_skip":
            print_to_log(f"[Orchestrator] No skip pattern detected - continuing with normal routing")
            return None
        else:
            print_to_log(f"[Orchestrator] Unknown skip pattern: {skip_type} - continuing with normal routing")
            return None

    def _route_by_intent(self, state: PipelineState, intent: str) -> str:
        """
        Route based on classified intent and current state
        """
        # Handle skip/bypass requests with semantic classification
        skip_routing = self._classify_skip_patterns(state.user_query)
        if skip_routing:
            print_to_log(f"[Orchestrator] Skip pattern routing override: {skip_routing}")
            print_to_log(f"[Orchestrator] Returning early with skip routing: {skip_routing}")
            return skip_routing
        
        if intent == "full_pipeline":
            # Check if user specified mode preference in the query
            query_lower = (state.user_query or "").lower()
            
            # Direct mode detection from query
            if any(word in query_lower for word in ["fast", "automated", "quick"]):
                print_to_log("[Orchestrator] Fast mode detected in query - calling automated pipeline directly")
                
                # CRITICAL: Preserve original user query for automated pipeline
                if not hasattr(state, 'preprocessing_state') or state.preprocessing_state is None:
                    state.preprocessing_state = {}
                state.preprocessing_state['original_user_query'] = state.user_query
                print_to_log(f"⚡ [Orchestrator] Preserved original query: '{state.user_query}'")
                
                try:
                    # Import and call automated pipeline agent directly
                    from automated_pipeline_agent import automated_pipeline_agent
                    
                    # Call automated pipeline
                    result_state = automated_pipeline_agent(state)
                    
                    # Update the current state with results
                    state.last_response = result_state.last_response
                    state.last_error = result_state.last_error
                    state.artifacts = result_state.artifacts
                    state.pending_file_uploads = result_state.pending_file_uploads
                    
                    print_to_log(f"✅ [Orchestrator] Automated pipeline completed successfully")
                    return "general_response"  # Return the automated pipeline's response
                    
                except Exception as e:
                    print_to_log(f"❌ [Orchestrator] Automated pipeline failed: {e}")
                    state.last_response = f"❌ Fast pipeline failed: {str(e)}"
                    return "general_response"
            elif any(word in query_lower for word in ["slow", "interactive", "step"]):
                print_to_log("[Orchestrator] Slow mode detected in query")
                return "preprocessing"  # Start with interactive preprocessing
            else:
                # Need to prompt for mode selection
                print_to_log("[Orchestrator] No mode specified - prompting for mode selection")
                return self._prompt_for_mode_selection(state)
        
        elif intent == "preprocessing":
            return "preprocessing"
        
        elif intent == "feature_selection":
            # ✅ ENHANCED DIRECT FEATURE SELECTION: BGE model classification with keyword fallback
            if state.cleaned_data is None and state.raw_data is not None:
                # Use BGE model to classify if this is a direct feature selection request
                is_direct_fs = self._classify_direct_feature_selection(state.user_query or "")
                
                if is_direct_fs:
                    print_to_log("[Orchestrator] 🚀 Direct feature selection requested (BGE classified) - using raw data")
                    return "feature_selection"
                else:
                    print_to_log("[Orchestrator] 📊 Standard feature selection request (BGE classified) - preprocessing first")
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
                print_to_log("[Orchestrator] Educational query detected - routing to general response")
                return "general_response"
            
            # Check prerequisites for actual model building
            if state.raw_data is None:
                print_to_log("[Orchestrator] No data available for model building - routing to model_building agent to handle")
                return "model_building"  # Let model building agent handle the "no data" case
            
            # Check for direct model building keywords
            query_lower = (state.user_query or "").lower()
            direct_keywords = [
                "skip preprocessing", "skip feature selection", "raw data", "as-is", 
                "direct", "without preprocessing", "existing features", "all columns",
                "use current data", "no preprocessing", "train directly", "bypass preprocessing"
            ]
            
            if any(keyword in query_lower for keyword in direct_keywords):
                print_to_log("[Orchestrator] 🚀 Direct model building requested - skipping preprocessing and feature selection")
                # Use raw data as cleaned data for direct model building
                if state.cleaned_data is None:
                    state.cleaned_data = state.raw_data.copy()
                    print_to_log("[Orchestrator] Using raw data as cleaned data")
                if state.selected_features is None:
                    state.selected_features = state.raw_data.copy()
                    print_to_log("[Orchestrator] Using all columns as selected features")
                return "model_building"
            
            # Normal pipeline flow - check if preprocessing is needed
            elif state.cleaned_data is None:
                print_to_log("[Orchestrator] No cleaned data available - prompting for preprocessing confirmation")
                return self._prompt_for_preprocessing_confirmation(state)
            elif state.selected_features is None:
                print_to_log("[Orchestrator] Need to select features first")
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
            print_to_log(f"⚠️ LLM capability response failed: {e}")
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
        
        return f"""🤖 Multi-Agent ML Integration System

Current Status:
{status}

My Capabilities:

🔧 Data Preprocessing
• Clean and validate data
• Handle missing values and outliers
• Normalize and encode features

🎯 Feature Selection
• Information Value (IV) analysis
• Correlation and VIF analysis
• PCA and dimensionality reduction

🤖 Model Building
• Train classification/regression models
• LightGBM, XGBoost, Random Forest
• Model evaluation and optimization

🎛️ Pipeline Management
• Full end-to-end ML workflows
• Intelligent query routing
• Session persistence and resume

Example queries:
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
            print_to_log(f"⚠️ LLM general response failed: {e}")
        
        # Fallback responses
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
        if any(greeting in query_lower for greeting in greetings):
            return "👋 Hello! I'm your Multi-Agent ML Integration System. I can help you build complete machine learning pipelines, from data preprocessing to model training. What would you like to work on today?"
        
        return "I'm here to help with your machine learning workflow! You can ask me to preprocess data, select features, build models, or run complete ML pipelines. What would you like to do?"

    def _generate_status_response(self, state: PipelineState) -> str:
        """Generate pipeline status response"""
        status_parts = ["📊 Pipeline Status:"]
        
        # Data status
        if state.raw_data is not None:
            status_parts.append(f"✅ Raw Data: {state.raw_data.shape[0]:,} rows × {state.raw_data.shape[1]} columns")
        else:
            status_parts.append("❌ Raw Data: Not loaded")
        
        if state.cleaned_data is not None:
            status_parts.append(f"✅ Cleaned Data: {state.cleaned_data.shape[0]:,} rows × {state.cleaned_data.shape[1]} columns")
        else:
            status_parts.append("❌ Cleaned Data: Not processed")
        
        if state.selected_features is not None:
            status_parts.append(f"✅ Selected Features: {len(state.selected_features)} features")
        else:
            status_parts.append("❌ Selected Features: Not selected")
        
        if state.trained_model is not None:
            model_type = type(state.trained_model).__name__
            status_parts.append(f"✅ Trained Model: {model_type}")
        else:
            status_parts.append("❌ Trained Model: Not trained")
        
        # Progress indicator
        progress_steps = [
            state.raw_data is not None,
            state.cleaned_data is not None, 
            state.selected_features is not None,
            state.trained_model is not None
        ]
        completed_steps = sum(progress_steps)
        progress_bar = "🟢" * completed_steps + "⚪" * (4 - completed_steps)
        status_parts.append(f"\nProgress: {progress_bar} ({completed_steps}/4 steps)")
        
        # Next step suggestion
        if completed_steps == 0:
            status_parts.append("\n💡 Next: Upload data to get started")
        elif completed_steps == 1:
            status_parts.append("\n💡 Next: Clean and preprocess the data")
        elif completed_steps == 2:
            status_parts.append("\n💡 Next: Select important features")
        elif completed_steps == 3:
            status_parts.append("\n💡 Next: Train a machine learning model")
        else:
            status_parts.append("\n🎉 Pipeline Complete! You can now make predictions or try different models.")
        
        return "\n".join(status_parts)



    def route(self, state: PipelineState) -> str:
        """
        Single LLM routing: Combined data science relevance + intent classification
        """
        if not state.user_query:
            return "general_response"  # Do nothing until user provides intent
        
        print_to_log(f"[Orchestrator] Processing query: '{state.user_query}'")
        
        # DEBUG: Check interactive session state
        print_to_log(f"🔍 [DEBUG] Has interactive_session: {hasattr(state, 'interactive_session')}")
        if hasattr(state, 'interactive_session'):
            print_to_log(f"🔍 [DEBUG] Interactive session: {state.interactive_session}")
            if state.interactive_session:
                print_to_log(f"🔍 [DEBUG] Phase: {state.interactive_session.get('phase')}")
        
        # CRITICAL: Check if we're in mode selection mode
        if (hasattr(state, 'interactive_session') and 
            state.interactive_session and 
            state.interactive_session.get('phase') == 'mode_selection'):
            print_to_log(f"🚀 [DEBUG] Entering mode selection handler")
            return self._handle_mode_selection(state)
        
        # CRITICAL: Check if we're in target selection mode
        if (hasattr(state, 'interactive_session') and 
            state.interactive_session and 
            state.interactive_session.get('phase') == 'target_selection'):
            print_to_log(f"🎯 [DEBUG] Entering target selection handler")
            return self._handle_target_selection(state)
        
        # CRITICAL: Check if we're in preprocessing confirmation mode
        if (hasattr(state, 'interactive_session') and 
            state.interactive_session and 
            state.interactive_session.get('phase') == 'preprocessing_confirmation'):
            print_to_log(f"🔧 [DEBUG] Entering preprocessing confirmation handler")
            return self._handle_preprocessing_confirmation(state)
        
        # Get thread logger
        thread_logger = self._get_thread_logger(state)
        if thread_logger:
            thread_logger.log_query(state.user_query, agent="orchestrator")
        
        # Single LLM call for both relevance check and intent classification
        classification_result = self._classify_with_single_llm(state.user_query)
        
        # Check if query is data science related
        if not classification_result["is_data_science"]:
            print_to_log(f"[Orchestrator] Query blocked: Not data science related (confidence: {classification_result['confidence']:.2f})")
            explanation = classification_result.get('explanation', '')
            state.last_response = f"""❌ This query is not related to data science or machine learning.

🤖 I'm a specialized data science and machine learning assistant. I can help you with:

📊 Data Analysis & Processing:
• Data cleaning, preprocessing, and transformation
• Handling missing values, outliers, and duplicates
• Data exploration and statistical analysis

🔍 Feature Engineering & Selection:
• Feature selection and importance analysis
• Correlation analysis and dimensionality reduction
• Feature engineering techniques

🧠 Machine Learning:
• Model building, training, and evaluation
• Predictions and model performance analysis
• Algorithm selection and hyperparameter tuning

💬 Please reframe your question to focus on data science, machine learning, statistics, or data analysis tasks."""
            return "general_response"
        
        # Use the classified intent
        intent = classification_result["intent"]
        confidence = classification_result["confidence"]
        
        print_to_log(f"[Orchestrator] Intent classification: {intent} (confidence: {confidence:.2f}) - {classification_result.get('explanation', '')[:100]}...")
        
        # Log classification results
        if thread_logger:
            thread_logger.log_classification(state.user_query, {
                "intent": intent,
                "method": "single_llm",
                "confidence": confidence,
                "is_data_science": classification_result["is_data_science"],
                "explanation": classification_result.get("explanation", "")
            })
        
        # CRITICAL: For any intent except general_query, check data and target prerequisites
        if intent != "general_query":
            # Store current intent for interactive session
            state._current_intent = intent
            
            # First, try to extract target column from current query if not set
            if state.raw_data is not None and (not hasattr(state, 'target_column') or not state.target_column):
                extracted_target = self._extract_target_from_query(state.user_query, state.raw_data.columns.tolist())
                if extracted_target:
                    state.target_column = extracted_target
                    print_to_log(f"🎯 [Orchestrator] Extracted target column from query: {extracted_target}")
            
            prerequisite_check = self._check_prerequisites(state, intent)
            print_to_log(f"🔍 [Orchestrator] Prerequisite check result: {prerequisite_check}")
            print_to_log(f"🔍 [Orchestrator] Target column after check: {getattr(state, 'target_column', 'NOT_SET')}")
            if prerequisite_check == "proceed":
                print_to_log(f"[Orchestrator] All prerequisites met, proceeding with {intent}")
            else:
                print_to_log(f"[Orchestrator] Prerequisites not met, returning: {prerequisite_check}")
                return prerequisite_check
        
        # Route by intent
        route_decision = self._route_by_intent(state, intent)
        
        # Log routing decision
        if thread_logger:
            thread_logger.log_routing(state.user_query, route_decision)
        
        return route_decision

    def get_routing_explanation(self, state: PipelineState, routing_decision: str) -> str:
        """Get explanation for routing decision"""
        explanations = {
            "preprocessing": "Routing to data preprocessing - will clean and prepare your data",
            "feature_selection": "Routing to feature selection - will identify the most important features",
            "model_building": "Routing to model building - will train and evaluate ML models",
            "general_response": "Generating conversational response",
            "code_execution": "Executing custom code analysis",
            "full_pipeline": "Running complete ML pipeline from start to finish",
            "fast_pipeline": "Running automated ML pipeline - fast mode"
        }
        
        return explanations.get(routing_decision, f"Routing to {routing_decision}")


# Create global instance
orchestrator = Orchestrator()
