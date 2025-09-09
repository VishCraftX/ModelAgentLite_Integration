#!/usr/bin/env python3
from print_to_log import print_to_log
# Import master log handler to capture logger.info calls
try:
    import master_log_handler
except ImportError:
    pass
"""
NEW AGENTIC FEATURE SELECTION BOT - Clean Architecture
Designed for reliable state management and proper Slack integration
"""

import os
import pandas as pd
import numpy as np
import tempfile
import json
import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# Import thread logging system
from thread_logger import get_thread_logger

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Slack imports
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# LLM imports
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Analysis imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2

# Global configuration
GLOBAL_DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")

@dataclass
class AnalysisStep:
    """Represents a single analysis step in the pipeline"""
    type: str
    parameters: Dict[str, Any]
    features_before: int
    features_after: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserSession:
    """Complete user session state - simple and reliable"""
    # File and target info
    file_path: str
    file_name: str
    user_id: Optional[str] = None  # For logging purposes
    target_column: Optional[str] = None
    datetime_column: Optional[str] = None
    oot_month: Optional[str] = None
    
    # Data states
    original_df: Optional[pd.DataFrame] = None
    current_df: Optional[pd.DataFrame] = None
    current_features: List[str] = field(default_factory=list)
    
    # Analysis pipeline
    analysis_chain: List[AnalysisStep] = field(default_factory=list)
    snapshots: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Conversation state
    phase: str = "need_target"  # need_target, analyzing, waiting_input, completed
    last_menu: Optional[str] = None
    pending_message: Optional[str] = None
    pending_csi_threshold: Optional[float] = None
    thread_ts: Optional[str] = None  # For Slack thread context
    csi_completed: bool = False  # Track if CSI analysis has been completed
    datetime_setup_completed: bool = False  # Track if datetime setup has been completed
    
    # Model configuration
    model_name: str = GLOBAL_DEFAULT_MODEL
    
    # Analysis results
    analysis_results: Dict[str, Any] = field(default_factory=dict)

class LLMManager:
    """Manages LLM interactions - supports multiple model providers"""
    
    @staticmethod
    def get_llm(model_name: str):
        """Get appropriate LLM instance for different model providers"""
        model_name_lower = model_name.lower()
        
        # Ollama models (local)
        if any(name in model_name_lower for name in ["qwen", "llama", "mistral", "codellama", "gemma"]):
            return ChatOllama(model=model_name, temperature=0)
        
        # Default to Ollama for unknown models
        else:
            print_to_log(f"⚠️ Unknown model {model_name}, defaulting to Ollama")
            return ChatOllama(model=model_name, temperature=0)
    
    @staticmethod
    def extract_json_robust(content: str) -> Dict[str, Any]:
        """Robust JSON extraction that works with different model outputs"""
        import json
        
        # Try direct JSON parsing first
        try:
            return json.loads(content.strip())
        except:
            pass
        
        # Remove markdown code blocks
        if "```" in content:
            lines = content.split('\n')
            content = '\n'.join([line for line in lines 
                               if not line.strip().startswith('```') 
                               and not line.strip().startswith('json')])
        
        # Extract JSON from text using regex
        import re
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested JSON
            r'\{.*?\}',  # Basic JSON pattern
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        # Fallback: construct JSON from keywords
        intent = "QUERY"
        if any(word in content.lower() for word in ["analysis", "filter", "run", "apply"]):
            intent = "STANDARD_ANALYSIS"
        elif any(word in content.lower() for word in ["vif", "lasso", "pca"]):
            intent = "CUSTOM_ANALYSIS"
        elif any(word in content.lower() for word in ["revert", "back"]):
            intent = "REVERT"
        
        return {
            "intent": intent,
            "query_details": content,
            "analysis_type": "unknown",
            "threshold": 0.05
        }
    
    @staticmethod
    def parse_user_intent(message: str, session: UserSession) -> Dict[str, Any]:
        """Parse user intent using LLM - fully conversational approach"""
        try:
            llm = LLMManager.get_llm(session.model_name)
            
            current_features = len(session.current_features)
            pipeline_summary = " -> ".join([step.type for step in session.analysis_chain]) if session.analysis_chain else "No analyses yet"
            
            prompt = f"""You are an expert feature selection assistant. Analyze the user's request and determine their intent.

CURRENT CONTEXT:
• Dataset: {current_features} features available
• Target column: {session.target_column}
• Current pipeline: {pipeline_summary}

USER REQUEST: "{message}"

Based on this request, determine the user's intent and extract relevant parameters. Consider these possibilities:

1. **STANDARD_ANALYSIS** - They want to RUN/APPLY IV analysis, correlation analysis, or CSI analysis to FILTER features
2. **STANDARD_ANALYSIS_QUERY** - They want to QUERY results from standard analyses (IV, SHAP, correlation, CSI, VIF) like "top 10 SHAP features", "how many IV > 0.4", etc.
3. **CUSTOM_ANALYSIS** - They want VIF filtering, PCA, LASSO, custom code analysis, or advanced techniques  
4. **QUERY** - They're asking for general information about existing data (like feature counts, current state, etc.)
5. **GENERAL_QUERY** - They're asking theoretical questions about feature selection concepts, bot capabilities, or general ML topics (no data needed)
6. **REVERT** - They want to go back to initial/original state (after cleaning)

IMPORTANT DISTINCTION:
- "run correlation analysis" or "filter highly correlated features" = STANDARD_ANALYSIS (removes correlated feature pairs)
- "filter features with correlation > 0.8" = STANDARD_ANALYSIS (removes correlated feature pairs)
- "show me top 10 SHAP features" or "give me IV scores" = STANDARD_ANALYSIS_QUERY (query built-in analysis results)
- "top 20 features by correlation with target" or "correlation with target column" = QUERY (needs code execution)
- "top 20 correlation pairs" or "show correlation pairs" = QUERY (needs code execution - displays info only)
- "correlation matrix" or "feature correlation pairs" = QUERY (needs code execution - displays info only)
- "which features correlate most with target" or "correlation scores with target" = QUERY (needs code execution)  
- "train decision tree and show importance" or "feature importance from model" = QUERY (computational analysis)
- "keep only top 10 SHAP features" or "filter with SHAP > 0.01" = CUSTOM_ANALYSIS (custom filtering)
- "what are top 10 features by RFE" or "show me RFE ranking" = QUERY (display results, don't modify data)
- "keep top 10 RFE features" or "filter with RFE" = CUSTOM_ANALYSIS (modify/filter data)
- "how many features remain" or "current state" = QUERY (general info)

CORRELATION ANALYSIS TYPES:
- "correlation analysis" = finds correlated FEATURE PAIRS for removal (STANDARD_ANALYSIS)
- "correlation with target" = ranks features by TARGET relationship (QUERY - needs code)
- "correlation pairs" or "correlation matrix" = shows correlation information (QUERY - needs code)
- "top X correlation pairs" = displays highest correlations (QUERY - needs code)

CLASSIFICATION EXAMPLES:



**STANDARD_ANALYSIS_QUERY** examples:
- "show me top 10 SHAP features" → STANDARD_ANALYSIS_QUERY
- "give me top 5 IV values" → STANDARD_ANALYSIS_QUERY
- "how many features have SHAP > 0.01" → STANDARD_ANALYSIS_QUERY
- "what are the highest CSI values" → STANDARD_ANALYSIS_QUERY
- "show me VIF scores" → STANDARD_ANALYSIS_QUERY
- "top 10 features by IV analysis" → STANDARD_ANALYSIS_QUERY
- "best features by correlation analysis" → STANDARD_ANALYSIS_QUERY

**QUERY** examples:
- "top 20 features by correlation with target" → QUERY
- "correlation with target column" → QUERY
- "which features correlate most with target" → QUERY
- "show me correlation scores with target" → QUERY
- "top 20 correlation pairs" → QUERY
- "show correlation pairs" → QUERY
- "highest correlation pairs" → QUERY
- "correlation matrix" → QUERY
- "feature correlation pairs" → QUERY
- "most correlated features" → QUERY
- "train a decision tree and tell me feature importance" → QUERY
- "give me top 10 important features after training decision tree" → QUERY
- "show me feature importance from random forest" → QUERY
- "train XGBoost and show feature rankings" → QUERY
- "what are top 10 features by RFE method" → QUERY
- "top features by RFE" → QUERY
- "what are top 10 features by LASSO" → QUERY
- "top features by LASSO" → QUERY
- "what are top 10 features by PCA" → QUERY
- "what are top 10 features based on PCA analysis" → QUERY
- "top features by PCA" → QUERY
- "features by PCA" → QUERY
- "show me RFE ranking" → QUERY
- "display LASSO coefficients" → QUERY
- "show PCA results" → QUERY
- "display PCA analysis" → QUERY
- "what are RFE results" → QUERY
- "show LASSO feature importance" → QUERY
- "PCA feature importance" → QUERY
- "how many features remain" → QUERY
- "what analyses have been done" → QUERY
- "current dataset info" → QUERY



**STANDARD_ANALYSIS** examples:
- "run IV analysis" → STANDARD_ANALYSIS
- "apply correlation filter" → STANDARD_ANALYSIS
- "do CSI analysis" → STANDARD_ANALYSIS
- "IV 0.02" → STANDARD_ANALYSIS
- "IV 0.2" → STANDARD_ANALYSIS
- "CSI 0.05" → STANDARD_ANALYSIS
- "CSI 0.5" → STANDARD_ANALYSIS
- "correlation 0.8" → STANDARD_ANALYSIS
- "SHAP 0.01" → STANDARD_ANALYSIS
- "VIF 5" → STANDARD_ANALYSIS

**CUSTOM_ANALYSIS** examples:
- "filter features with VIF > 5" → CUSTOM_ANALYSIS
- "run LASSO feature selection" → CUSTOM_ANALYSIS
- "keep only top 10 SHAP features" → CUSTOM_ANALYSIS
- "filter features with SHAP > 0.01" → CUSTOM_ANALYSIS
- "apply SHAP filtering with threshold 0.05" → CUSTOM_ANALYSIS
- "run RFE with 10 features" → CUSTOM_ANALYSIS
- "keep top 10 RFE features" → CUSTOM_ANALYSIS
- "filter with RFE" → CUSTOM_ANALYSIS
- "apply RFE selection" → CUSTOM_ANALYSIS
- "run LASSO regularization" → CUSTOM_ANALYSIS
- "keep top 20 LASSO features" → CUSTOM_ANALYSIS
- "filter with LASSO" → CUSTOM_ANALYSIS
- "apply LASSO selection" → CUSTOM_ANALYSIS
- "PCA analysis" → CUSTOM_ANALYSIS
- "principal component analysis" → CUSTOM_ANALYSIS
- "run PCA with 95% variance" → CUSTOM_ANALYSIS

**GENERAL_QUERY** examples:
- "what can you do" → GENERAL_QUERY
- "what are your capabilities" → GENERAL_QUERY
- "what is IV analysis" → GENERAL_QUERY
- "explain CSI" → GENERAL_QUERY
- "what is correlation analysis" → GENERAL_QUERY
- "what is VIF" → GENERAL_QUERY
- "how does feature selection work" → GENERAL_QUERY
- "what is SHAP" → GENERAL_QUERY
- "explain feature importance" → GENERAL_QUERY
- "help me understand" → GENERAL_QUERY

**REVERT** examples:
- "revert to original" → REVERT
- "go back to initial state" → REVERT
- "start from start" → REVERT
- "reset to beginning" → REVERT
- "go back to clean data" → REVERT
- "restart from cleaned dataset" → REVERT
- "undo all changes" → REVERT
- "back to square one" → REVERT
- "return to initial" → REVERT



CRITICAL DISTINCTIONS:
- Analysis requests should specify the exact analysis type and any thresholds
- Query requests should be classified based on whether they need code execution or not

For analysis requests, carefully extract:
- The analysis type (iv, correlation, csi, vif, pca, lasso, rfe, shap)
- Any threshold values mentioned (look for numbers like 0.04, 0.05, 0.8, 0.1, 0.2, etc.)
- Any comparison operators (>, <, greater than, less than, above, below)

THRESHOLD EXTRACTION EXAMPLES:
- "CSI analysis with threshold < 0.05" → threshold: 0.05, comparison: "<"
- "do CSI with features < 0.1" → threshold: 0.1, comparison: "<"  
- "run CSI cutoff 0.15" → threshold: 0.15, comparison: ">"
- "CSI analysis threshold 0.08" → threshold: 0.08, comparison: ">"
- "filter using CSI > 0.2" → threshold: 0.2, comparison: ">"

Respond with ONLY this JSON format (no markdown, no explanations):
{{
    "intent": "STANDARD_ANALYSIS",
    "analysis_type": "iv", 
    "threshold": 0.04,
    "comparison": "<",
    "query_details": "Brief description of what user wants",
    "extracted_info": "Any specific parameters or details mentioned"
}}

Be very careful with threshold extraction:
- "IV < 0.04" → threshold: 0.04, comparison: "<"
- "correlation > 0.8" → threshold: 0.8, comparison: ">"
- "CSI threshold < 0.05" → threshold: 0.05, comparison: "<"
- "CSI features > 0.1" → threshold: 0.1, comparison: ">"

IMPORTANT: Respond with ONLY the JSON object. No explanations, no additional text, just the JSON."""

            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Clean and parse response
            content = response.content.strip()
            
            # Use robust JSON extraction
            result = LLMManager.extract_json_robust(content)
            
            # Validate and set defaults
            if "intent" not in result:
                result["intent"] = "QUERY"
            if "query_details" not in result:
                result["query_details"] = message
            if "threshold" not in result and result.get("analysis_type"):
                # Set reasonable defaults
                defaults = {"iv": 0.05, "correlation": 0.8, "csi": 0.2}
                result["threshold"] = defaults.get(result["analysis_type"], 0.05)
            elif "threshold" in result and result["threshold"] == 0 and result.get("analysis_type") == "correlation":
                # Safety check: Never use 0 threshold for correlation (would remove all features)
                print_to_log("⚠️ WARNING: Correlation threshold was 0, setting to safe default 0.8")
                result["threshold"] = 0.8
                
            return result
            
        except Exception as e:
            print_to_log(f"❌ Error parsing intent with LLM: {e}")
            # Minimal fallback - just return as query
            return {
                "intent": "QUERY",
                "query_details": message,
                "error": f"LLM parsing failed: {str(e)}"
            }

class DataProcessor:
    """Handles data loading and basic cleaning"""
    
    @staticmethod
    def load_and_clean_data(session: UserSession) -> bool:
        """Load data and perform intelligent cleaning with numeric conversion"""
        try:
            # Load data
            df = pd.read_csv(session.file_path)
            session.original_df = df.copy()
            
            print_to_log(f"📊 Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Step 1: Remove single value columns
            single_value_cols = []
            for col in df.columns:
                if df[col].nunique() <= 1:
                    single_value_cols.append(col)
            
            print_to_log(f"🔍 Found {len(single_value_cols)} single-value columns to remove")
            
            # Step 2: Smart object column handling - try to convert to numeric first
            object_cols = [col for col in df.columns if col not in single_value_cols and df[col].dtype == 'object']
            converted_cols = []
            failed_conversion_cols = []
            
            print_to_log(f"🔍 Found {len(object_cols)} object columns, attempting numeric conversion...")
            
            for col in object_cols:
                try:
                    # Try to convert to numeric, handling common string number formats
                    # This handles cases like: "123", "45.67", "1,234", etc.
                    original_series = df[col].copy()
                    
                    # First, try direct conversion
                    converted = pd.to_numeric(original_series, errors='coerce')
                    
                    # If that fails for many values, try cleaning common string formats
                    if converted.isna().sum() > len(original_series) * 0.5:  # More than 50% NaN
                        # Try removing commas, spaces, and other common formatting
                        cleaned_series = original_series.astype(str).str.replace(',', '').str.replace(' ', '').str.strip()
                        converted = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # If conversion is successful for most values (less than 20% NaN)
                    non_null_before = original_series.notna().sum()
                    non_null_after = converted.notna().sum()
                    
                    if non_null_after >= non_null_before * 0.8:  # At least 80% successfully converted
                        df[col] = converted
                        converted_cols.append(col)
                        print_to_log(f"   ✅ Converted '{col}' to numeric ({non_null_after}/{non_null_before} values)")
                    else:
                        failed_conversion_cols.append(col)
                        print_to_log(f"   ❌ Failed to convert '{col}' (only {non_null_after}/{non_null_before} values convertible)")
                        
                except Exception as e:
                    failed_conversion_cols.append(col)
                    print_to_log(f"   ❌ Error converting '{col}': {str(e)[:50]}")
            
            print_to_log(f"📈 Conversion summary: {len(converted_cols)} converted, {len(failed_conversion_cols)} remained as objects")
            
            # Step 3: Remove remaining non-numeric columns
            cols_to_remove = single_value_cols + failed_conversion_cols
            clean_df = df.drop(columns=cols_to_remove)
            
            session.current_df = clean_df.copy()
            session.current_features = list(clean_df.columns)
            
            # Add cleaning step to pipeline
            cleaning_step = AnalysisStep(
                type="intelligent_data_cleaning",
                parameters={"removed_cols": cols_to_remove},
                features_before=df.shape[1],
                features_after=clean_df.shape[1],
                timestamp=datetime.now().isoformat(),
                metadata={
                    "single_value_cols": single_value_cols,
                    "converted_to_numeric": converted_cols,
                    "failed_conversion_cols": failed_conversion_cols,
                    "conversion_strategy": "smart_numeric_conversion"
                }
            )
            session.analysis_chain.append(cleaning_step)
            
            # Create initial snapshot - this is the clean starting state for revert
            session.snapshots["after_cleaning"] = {
                "df": clean_df.copy(),
                "features": list(clean_df.columns),
                "timestamp": datetime.now().isoformat()
            }
            
            print_to_log(f"✅ Intelligently cleaned dataset: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns")
            print_to_log(f"   📊 Summary:")
            print_to_log(f"   • Removed: {len(single_value_cols)} single-value columns")
            print_to_log(f"   • Converted: {len(converted_cols)} object → numeric columns")
            print_to_log(f"   • Removed: {len(failed_conversion_cols)} non-convertible object columns")
            print_to_log(f"   • Final: {clean_df.shape[1]} numeric columns ready for analysis")
            
            return True
            
        except Exception as e:
            print_to_log(f"❌ Error loading data: {e}")
            return False
    
    @staticmethod
    def load_and_clean_data_with_progress(session: UserSession, say) -> bool:
        """Load data and perform initial cleaning with user progress updates"""
        try:
            # Load data
            say("📊 **Loading dataset...**")
            df = pd.read_csv(session.file_path)
            session.original_df = df.copy()
            
            rows, cols = df.shape
            say(f"✅ **Dataset loaded:** {rows:,} rows, {cols} columns")
            
            # Show cleaning progress for large datasets
            if rows > 10000 or cols > 50:
                say("🧹 **Cleaning data (removing single-value and non-numeric columns)...**")
            
            # Remove single value columns
            single_value_cols = []
            for col in df.columns:
                if df[col].nunique() <= 1:
                    single_value_cols.append(col)
            
            # Remove object columns (non-numeric)  
            non_numeric_cols = []
            for col in df.columns:
                if col not in single_value_cols and df[col].dtype == 'object':
                    non_numeric_cols.append(col)
            
            # Create clean dataset
            cols_to_remove = single_value_cols + non_numeric_cols
            clean_df = df.drop(columns=cols_to_remove)
            
            session.current_df = clean_df.copy()
            session.current_features = list(clean_df.columns)
            
            # Add cleaning step to pipeline
            cleaning_step = AnalysisStep(
                type="data_cleaning",
                parameters={"removed_cols": cols_to_remove},
                features_before=df.shape[1],
                features_after=clean_df.shape[1],
                timestamp=datetime.now().isoformat(),
                metadata={
                    "single_value_cols": single_value_cols,
                    "non_numeric_cols": non_numeric_cols
                }
            )
            session.analysis_chain.append(cleaning_step)
            
            # Create initial snapshot
            session.snapshots["after_cleaning"] = {
                "df": clean_df.copy(),
                "features": list(clean_df.columns),
                "timestamp": datetime.now().isoformat()
            }
            
            # Show cleaning results
            removed_count = len(cols_to_remove)
            if removed_count > 0:
                say(f"🧹 **Data cleaned:** Removed {len(single_value_cols)} single-value + {len(non_numeric_cols)} object columns")
            
            say(f"✅ **Ready for analysis:** {clean_df.shape[0]:,} rows, {clean_df.shape[1]} features")
            
            return True
            
        except Exception as e:
            say(f"❌ **Error processing dataset:** {str(e)}")
            return False

class AnalysisEngine:
    """Handles all feature selection analyses"""
    
    @staticmethod
    def run_iv_analysis(session: UserSession, threshold: float = 0.02) -> Dict[str, Any]:
        """Run Information Value analysis"""
        if session.target_column not in session.current_df.columns:
            return {"error": "Target column not found"}
        
        try:
            df = session.current_df
            target = session.target_column
            feature_cols = [col for col in df.columns if col != target]
            
            iv_scores = {}
            for col in feature_cols:
                try:
                    # Simple IV calculation (binning approach)
                    bins = pd.qcut(df[col], q=5, duplicates='drop')
                    crosstab = pd.crosstab(bins, df[target])
                    
                    if crosstab.shape[1] >= 2:
                        # Calculate IV
                        crosstab_pct = crosstab.div(crosstab.sum(axis=0), axis=1)
                        iv = 0
                        for i in range(len(crosstab)):
                            if crosstab_pct.iloc[i, 0] > 0 and crosstab_pct.iloc[i, 1] > 0:
                                woe = np.log(crosstab_pct.iloc[i, 1] / crosstab_pct.iloc[i, 0])
                                iv += (crosstab_pct.iloc[i, 1] - crosstab_pct.iloc[i, 0]) * woe
                        
                        iv_scores[col] = abs(iv)
                    else:
                        iv_scores[col] = 0
                except:
                    iv_scores[col] = 0
            
            # Filter features based on threshold
            selected_features = [col for col, score in iv_scores.items() if score >= threshold]
            selected_features.append(target)  # Keep target
            
            # Update session
            features_before = len(session.current_features)
            session.current_df = session.current_df[selected_features]
            session.current_features = selected_features
            
            # Add to pipeline
            analysis_step = AnalysisStep(
                type="iv_analysis",
                parameters={"threshold": threshold},
                features_before=features_before,
                features_after=len(selected_features),
                timestamp=datetime.now().isoformat(),
                metadata={"iv_scores": iv_scores, "removed_count": features_before - len(selected_features)}
            )
            session.analysis_chain.append(analysis_step)
            
            # Create snapshot
            session.snapshots["after_iv"] = {
                "df": session.current_df.copy(),
                "features": session.current_features.copy(),
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "features_removed": features_before - len(selected_features),
                "remaining_features": len(selected_features),
                "iv_scores": iv_scores
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def run_correlation_analysis(session: UserSession, threshold: float = 0.8) -> Dict[str, Any]:
        """Run correlation analysis"""
        try:
            # Safety check: Ensure threshold is reasonable
            if threshold <= 0 or threshold >= 1:
                print_to_log(f"⚠️ WARNING: Invalid correlation threshold {threshold}, using default 0.8")
                threshold = 0.8
            
            df = session.current_df
            feature_cols = [col for col in df.columns if col != session.target_column]
            
            # Calculate correlation matrix
            corr_matrix = df[feature_cols].corr().abs()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            cols_to_remove = set()
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] >= threshold:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
                        cols_to_remove.add(col2)  # Remove second column
            
            # Update session
            features_before = len(session.current_features)
            remaining_cols = [col for col in session.current_features if col not in cols_to_remove]
            session.current_df = session.current_df[remaining_cols]
            session.current_features = remaining_cols
            
            # Add to pipeline
            analysis_step = AnalysisStep(
                type="correlation_analysis",
                parameters={"threshold": threshold},
                features_before=features_before,
                features_after=len(remaining_cols),
                timestamp=datetime.now().isoformat(),
                metadata={"high_corr_pairs": high_corr_pairs, "removed_features": list(cols_to_remove)}
            )
            session.analysis_chain.append(analysis_step)
            
            # Create snapshot
            session.snapshots["after_correlation"] = {
                "df": session.current_df.copy(),
                "features": session.current_features.copy(),
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "features_removed": len(cols_to_remove),
                "remaining_features": len(remaining_cols),
                "high_corr_pairs": high_corr_pairs
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def run_csi_analysis(session: UserSession, threshold: float = 0.2):
        """Run Characteristic Stability Index (CSI) analysis"""
        try:
            if not session.datetime_column:
                return {"error": "Datetime column not specified"}
            
            if not session.oot_month:
                return {"error": "OOT month not specified"}
            
            # Use original dataframe for CSI analysis since it contains all columns including datetime
            if session.original_df is None:
                return {"error": "Original dataframe not available"}
                
            if session.datetime_column not in session.original_df.columns:
                return {"error": f"Datetime column '{session.datetime_column}' not found in original data"}
            
            # Convert datetime column - use original dataframe
            df = session.original_df.copy()
            try:
                df[session.datetime_column] = pd.to_datetime(df[session.datetime_column])
            except:
                return {"error": f"Cannot convert '{session.datetime_column}' to datetime"}
            
            # Parse OOT month (handle YYYYMXX format like 2023M09)
            try:
                if 'M' in session.oot_month:
                    # Handle YYYYMXX format (e.g., 2023M09)
                    year, month = session.oot_month.split('M')
                    oot_start = pd.to_datetime(f"{year}-{month.zfill(2)}-01")
                else:
                    # Handle other formats
                    oot_start = pd.to_datetime(session.oot_month)
            except:
                return {"error": f"Invalid OOT month format: '{session.oot_month}'. Use YYYYMXX format (e.g., 2023M09)"}
            
            # Split data into development and OOT periods
            dev_data = df[df[session.datetime_column] < oot_start]
            oot_data = df[df[session.datetime_column] >= oot_start]
            
            if len(dev_data) == 0:
                return {"error": "No development data found before OOT month"}
            
            if len(oot_data) == 0:
                return {"error": "No OOT data found after OOT month"}
            
            # Calculate CSI for each feature - analyze current features on original dataframe
            csi_scores = {}
            features_to_remove = []
            
            # Only analyze current features (after cleaning) but use original dataframe for data
            for feature in session.current_features:
                if feature == session.target_column or feature == session.datetime_column:
                    continue
                
                # Ensure feature exists in original dataframe
                if feature not in df.columns:
                    continue
                
                try:
                    # Calculate CSI
                    dev_series = dev_data[feature].dropna()
                    oot_series = oot_data[feature].dropna()
                    
                    if len(dev_series) == 0 or len(oot_series) == 0:
                        csi_scores[feature] = float('inf')
                        features_to_remove.append(feature)
                        continue
                    
                    # Create bins based on development data quantiles
                    bins = np.unique(np.percentile(dev_series, np.linspace(0, 100, 11)))
                    if len(bins) < 2:
                        csi_scores[feature] = float('inf')
                        features_to_remove.append(feature)
                        continue
                    
                    # Calculate distributions
                    dev_dist, _ = np.histogram(dev_series, bins=bins)
                    oot_dist, _ = np.histogram(oot_series, bins=bins)
                    
                    # Convert to proportions
                    dev_prop = dev_dist / dev_dist.sum()
                    oot_prop = oot_dist / oot_dist.sum()
                    
                    # Calculate CSI
                    csi = 0
                    for i in range(len(dev_prop)):
                        if dev_prop[i] > 0 and oot_prop[i] > 0:
                            csi += (oot_prop[i] - dev_prop[i]) * np.log(oot_prop[i] / dev_prop[i])
                    
                    csi_scores[feature] = csi
                    
                    if csi > threshold:
                        features_to_remove.append(feature)
                        
                except Exception:
                    csi_scores[feature] = float('inf')
                    features_to_remove.append(feature)
            
            # Update session
            features_before = len(session.current_features)
            remaining_cols = [col for col in session.current_features if col not in features_to_remove]
            session.current_df = session.current_df[remaining_cols]
            session.current_features = remaining_cols
            
            # Add to pipeline
            analysis_step = AnalysisStep(
                type="csi_analysis",
                parameters={"threshold": threshold, "datetime_column": session.datetime_column, "oot_month": session.oot_month},
                features_before=features_before,
                features_after=len(remaining_cols),
                timestamp=datetime.now().isoformat(),
                metadata={"csi_scores": csi_scores, "removed_features": features_to_remove}
            )
            session.analysis_chain.append(analysis_step)
            
            # Create snapshot
            session.snapshots[f"after_csi"] = {
                "features": remaining_cols.copy(),
                "df": session.current_df.copy(),
                "step": len(session.analysis_chain) - 1
            }
            
            return {
                "success": True,
                "features_removed": len(features_to_remove),
                "remaining_features": len(remaining_cols),
                "csi_scores": csi_scores,
                "dev_samples": len(dev_data),
                "oot_samples": len(oot_data)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def run_vif_analysis(session: UserSession, threshold: float = 5.0):
        """Run Variance Inflation Factor (VIF) analysis"""
        try:
            print_to_log(f"🔧 DEBUG VIF ENGINE: Starting VIF analysis with threshold {threshold}")
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            from sklearn.preprocessing import StandardScaler
            
            # Get numeric features only (exclude target)
            numeric_features = [f for f in session.current_features if f != session.target_column]
            print_to_log(f"🔧 DEBUG VIF ENGINE: Found {len(numeric_features)} numeric features")
            
            if len(numeric_features) < 2:
                error_msg = "Need at least 2 features for VIF analysis"
                print_to_log(f"🔧 DEBUG VIF ENGINE: ERROR - {error_msg}")
                return {"error": error_msg}
            
            # Prepare data
            df = session.current_df[numeric_features].copy()
            print_to_log(f"🔧 DEBUG VIF ENGINE: Data shape: {df.shape}")
            
            # Handle missing values
            missing_before = df.isnull().sum().sum()
            df = df.fillna(df.mean())
            print_to_log(f"🔧 DEBUG VIF ENGINE: Filled {missing_before} missing values")
            
            # Scale features for stability
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            print_to_log(f"🔧 DEBUG VIF ENGINE: Scaled data shape: {df_scaled.shape}")
            
            # Calculate VIF for each feature
            vif_scores = {}
            features_to_remove = []
            
            for i, feature in enumerate(df_scaled.columns):
                try:
                    vif_value = variance_inflation_factor(df_scaled.values, i)
                    vif_scores[feature] = vif_value
                    print_to_log(f"🔧 DEBUG VIF ENGINE: {feature} VIF = {vif_value:.4f}")
                    
                    if vif_value > threshold:
                        features_to_remove.append(feature)
                        print_to_log(f"🔧 DEBUG VIF ENGINE: {feature} marked for removal (VIF {vif_value:.4f} > {threshold})")
                except Exception as e:
                    print_to_log(f"🔧 DEBUG VIF ENGINE: Error calculating VIF for {feature}: {e}")
                    vif_scores[feature] = float('inf')
                    features_to_remove.append(feature)
            
            # Update session
            features_before = len(session.current_features)
            remaining_cols = [col for col in session.current_features if col not in features_to_remove]
            print_to_log(f"🔧 DEBUG VIF ENGINE: Features before: {features_before}, removing: {len(features_to_remove)}, remaining: {len(remaining_cols)}")
            
            session.current_df = session.current_df[remaining_cols]
            session.current_features = remaining_cols
            
            # Add to pipeline
            analysis_step = AnalysisStep(
                type="vif_analysis",
                parameters={"threshold": threshold},
                features_before=features_before,
                features_after=len(remaining_cols),
                timestamp=datetime.now().isoformat(),
                metadata={"vif_scores": vif_scores, "removed_features": features_to_remove}
            )
            session.analysis_chain.append(analysis_step)
            print_to_log(f"🔧 DEBUG VIF ENGINE: Added analysis step to pipeline")
            
            # Create snapshot
            session.snapshots[f"after_vif"] = {
                "features": remaining_cols.copy(),
                "df": session.current_df.copy(),
                "step": len(session.analysis_chain) - 1
            }
            print_to_log(f"🔧 DEBUG VIF ENGINE: Created snapshot")
            
            result = {
                "success": True,
                "features_removed": len(features_to_remove),
                "remaining_features": len(remaining_cols),
                "vif_scores": vif_scores
            }
            print_to_log(f"🔧 DEBUG VIF ENGINE: Returning success result: {result}")
            return result
            
        except Exception as e:
            error_msg = f"VIF analysis failed: {str(e)}"
            print_to_log(f"🔧 DEBUG VIF ENGINE: EXCEPTION - {error_msg}")
            import traceback
            traceback.print_exc()
            return {"error": error_msg}

    @staticmethod
    def run_shap_analysis(session: UserSession, threshold: float = 0.01, model_type: str = "randomforest", top_n: int = None):
        """Run SHAP feature importance analysis using RandomForest (or other models)"""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Get numeric features only (exclude target)
            numeric_features = [f for f in session.current_features if f != session.target_column]
            
            if len(numeric_features) < 2:
                return {"error": "Need at least 2 features for SHAP analysis"}
            
            # Prepare data
            X = session.current_df[numeric_features].copy()
            y = session.current_df[session.target_column].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
            
            # Determine if classification or regression
            is_classification = len(y.unique()) <= 10 and y.dtype in ['object', 'int64', 'bool']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train model
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            
            model.fit(X_train, y_train)
            
            # Get feature importances (proxy for SHAP values)
            feature_importances = model.feature_importances_
            
            # Create SHAP-like scores dictionary
            shap_scores = {}
            for i, feature in enumerate(numeric_features):
                shap_scores[feature] = feature_importances[i]
            
            # Determine features to remove
            if top_n:
                # Keep top N features, remove the rest
                sorted_features = sorted(shap_scores.items(), key=lambda x: x[1], reverse=True)
                top_features = [f[0] for f in sorted_features[:top_n]]
                features_to_remove = [f for f in numeric_features if f not in top_features]
            elif threshold is not None:
                # Threshold-based removal
                features_to_remove = [f for f, score in shap_scores.items() if score < threshold]
            else:
                # No filtering criteria provided, don't remove any features
                features_to_remove = []
            
            # Update session
            remaining_cols = [f for f in session.current_features if f not in features_to_remove]
            session.current_features = remaining_cols
            session.current_df = session.current_df[remaining_cols]
            
            # Add to analysis chain
            analysis_step = AnalysisStep(
                type="shap_analysis",
                parameters={"threshold": threshold, "model_type": model_type, "top_n": top_n},
                features_before=len(numeric_features) + 1,  # +1 for target
                features_after=len(remaining_cols),
                timestamp=datetime.now().isoformat(),
                metadata={"shap_scores": shap_scores, "removed_features": features_to_remove, "is_classification": is_classification}
            )
            session.analysis_chain.append(analysis_step)
            
            # Create snapshot
            session.snapshots[f"after_shap"] = {
                "features": remaining_cols.copy(),
                "df": session.current_df.copy(),
                "step": len(session.analysis_chain) - 1
            }
            
            return {
                "success": True,
                "features_removed": len(features_to_remove),
                "remaining_features": len(remaining_cols),
                "shap_scores": shap_scores,
                "model_type": model_type,
                "is_classification": is_classification,
                "top_features": sorted(shap_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
        except Exception as e:
            return {"error": str(e)}

class MenuGenerator:
    """Generates dynamic menus and responses"""
    
    @staticmethod
    def show_crisp_summary(session: UserSession, say) -> None:
        """Show crisp summary after analysis instead of full menu"""
        if not session.analysis_chain:
            summary = f"""📊 **Current State:** {len(session.current_features)} features
💬 **Ready for analysis**"""
            say(summary)
            return
        
        # Build detailed pipeline with thresholds and feature counts
        pipeline_steps = []
        total_removed = 0
        original_features = session.analysis_chain[0].features_before
        
        for i, step in enumerate(session.analysis_chain):
            features_removed = step.features_before - step.features_after
            
            # Skip query steps that don't actually filter features
            # These are typically analysis queries with threshold=0.0 and no features removed
            is_query_step = (
                features_removed == 0 and 
                step.parameters and 
                step.parameters.get('threshold') == 0.0
            )
            
            if is_query_step:
                continue  # Skip this step in the summary
            
            total_removed += features_removed
            
            # Extract key parameters for display
            step_details = []
            if step.parameters:
                if 'threshold' in step.parameters:
                    threshold = step.parameters['threshold']
                    step_details.append(f"thresh={threshold}")
                if 'datetime_column' in step.parameters:
                    step_details.append(f"dt_col={step.parameters['datetime_column']}")
                if 'oot_month' in step.parameters:
                    step_details.append(f"oot={step.parameters['oot_month']}")
            
            step_params = f" ({', '.join(step_details)})" if step_details else ""
            
            if step.type == "data_cleaning":
                pipeline_steps.append(f"🧹 **{step.type}**: {step.features_before}→{step.features_after} (-{features_removed})")
            elif step.type == "csi_analysis":
                pipeline_steps.append(f"📅 **CSI{step_params}**: {step.features_before}→{step.features_after} (-{features_removed})")
            elif step.type == "iv_analysis":
                pipeline_steps.append(f"📊 **IV{step_params}**: {step.features_before}→{step.features_after} (-{features_removed})")
            elif step.type == "correlation_analysis":
                pipeline_steps.append(f"🔗 **Corr{step_params}**: {step.features_before}→{step.features_after} (-{features_removed})")
            elif step.type == "vif_analysis":
                pipeline_steps.append(f"📈 **VIF{step_params}**: {step.features_before}→{step.features_after} (-{features_removed})")
            elif step.type == "shap_analysis":
                pipeline_steps.append(f"🎯 **SHAP{step_params}**: {step.features_before}→{step.features_after} (-{features_removed})")
            else:
                pipeline_steps.append(f"⚙️ **{step.type}{step_params}**: {step.features_before}→{step.features_after} (-{features_removed})")
        
        # Calculate reduction percentage
        reduction_pct = (total_removed / original_features * 100) if original_features > 0 else 0
        
        newline = "\n"
        summary = f"""📊 **Pipeline Summary** | {len(session.current_features)} features remaining

📈 **Feature Reduction:** {original_features} → {len(session.current_features)} (-{total_removed}, {reduction_pct:.1f}%)

🔄 **Analysis Steps:**
{newline.join(pipeline_steps)}

💬 **Continue:** Ask questions or request more analyses"""
        
        say(summary)
    
    @staticmethod
    def generate_main_menu(session: UserSession) -> str:
        """Generate the main interactive menu"""
        current_features_count = len(session.current_features)
        pipeline_text = " -> ".join([step.type for step in session.analysis_chain])
        available_snapshots = list(session.snapshots.keys())
        
        menu = f"""🎯 **Feature Selection Assistant**

📊 **Current Dataset:** {current_features_count} features
📈 **Progress:** {pipeline_text or 'Ready to start'}

**🔬 Available Analyses:**
• `IV analysis 0.05` - Filter by predictive power
• `Correlation analysis 0.8` - Remove redundant features  
• `CSI analysis 0.2` - Check feature stability over time
• `VIF analysis 5` - Reduce multicollinearity
• `SHAP analysis` - Feature importance ranking
• `PCA analysis` - Dimensionality reduction
• `LASSO selection` - Regularized feature selection

**❓ Ask Me Anything:**
• Data questions: `how many features remain?`, `show me top features`
• Analysis queries: `top 10 IV scores`, `correlation with target`
• Learn concepts: `what is IV?`, `explain correlation analysis`
• Get suggestions: `what analysis should I run next?`

**🔄 Navigation:**
• Start over: `revert`, `go back to clean data`, `reset`
• Check progress: `show pipeline`, `current summary`
• Complete: `proceed`, `finalize analysis`

💬 **What would you like to do next?**"""
        
        return menu
    
    @staticmethod
    def format_analysis_result(analysis_type: str, result: Dict[str, Any]) -> str:
        """Format analysis results for display"""
        if result.get("error"):
            return f"❌ **{analysis_type.upper()} Analysis Failed:** {result['error']}"
        
        if analysis_type == "iv_analysis":
            return f"""✅ **IV Analysis Complete**

📊 **Results:**
• Features removed: {result['features_removed']}
• Remaining features: {result['remaining_features']}

📈 **Top IV Scores:**"""
        
        elif analysis_type == "correlation_analysis":
            return f"""✅ **Correlation Analysis Complete**

📊 **Results:**
• Features removed: {result['features_removed']} 
• Remaining features: {result['remaining_features']}
• High correlation pairs found: {len(result.get('high_corr_pairs', []))}"""
        
        return f"✅ **{analysis_type.upper()} Analysis Complete**"

class AgenticFeatureSelectionBot:
    """Main bot controller with clean state management"""
    
    def __init__(self):
        self.app = App(token=os.environ.get("SLACK_BOT_TOKEN_FS"))
        self.users: Dict[str, UserSession] = {}
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup all Slack event handlers"""
        
        # File upload handler
        @self.app.event("file_shared")
        def handle_file_upload(event, say, client):
            self.handle_file_upload(event, say, client)
        
        # Message handler - check if bot is mentioned
        @self.app.event("message")
        def handle_message_event(body, say, logger):
            event = body["event"]
            text = event.get("text", "")
            
            # Check if this message mentions the bot
            import re
            if re.search(r'<@[^>]+>', text):
                # Extract bot ID and check if it's the correct bot
                bot_match = re.search(r'<@([^>]+)>', text)
                if bot_match:
                    bot_id = bot_match.group(1)
                    # Only respond to the correct bot ID (U099J68BABU)
                    if bot_id == "U099J68BABU":
                        # This is a mention, process it
                        self.handle_app_mention(body, say, logger)
                    else:
                        # Wrong bot mentioned, ignore
                        logger.info(f"🔍 DEBUG: Wrong bot mentioned ({bot_id}), ignoring")
        

    
    def handle_file_upload(self, event, say, client):
        """Handle CSV file uploads"""
        try:
            user_id = event.get("user_id") or event.get("user")
            file_id = event["file_id"]
            
            logger.info(f"📁 FILE UPLOAD: User={user_id}, File={file_id}")
            
            # Get file info
            file_info = client.files_info(file=file_id)["file"]
            
            if not file_info["name"].endswith('.csv'):
                say("❌ Please upload a CSV file for feature selection analysis.")
                return
            
            # Show immediate welcome message
            say(f"👋 **Welcome to Agentic Feature Selection!**\n🔄 **Processing your file:** {file_info['name']}...")
            
            # Download file
            import requests
            headers = {"Authorization": f"Bearer {os.environ.get('SLACK_BOT_TOKEN_FS')}"}
            response = requests.get(file_info["url_private_download"], headers=headers)
            
            # Save file temporarily
            file_path = f"temp_{user_id}_{file_info['name']}"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Show data scanning message
            say("🔍 **Scanning dataset and preparing for analysis...**")
            
            # Create user session
            session = UserSession(
                file_path=file_path,
                file_name=file_info["name"],
                model_name=GLOBAL_DEFAULT_MODEL
            )
            
            # Load and clean data with progress updates
            if DataProcessor.load_and_clean_data_with_progress(session, say):
                self.users[user_id] = session
                logger.info(f"✅ SESSION CREATED: User={user_id}, File={file_info['name']}, Phase={session.phase}")
                logger.info(f"✅ TOTAL SESSIONS: {len(self.users)}")
                
                # Show target column selection
                columns = session.current_features
                say(f"""📁 **File uploaded:** {file_info['name']}

🎯 **Target Column Selection**

Available columns: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}

📝 **Please specify your target column:**
• Type the column name directly (e.g., `is_fraud`)  
• Or use: `target column_name`

**Example:** `is_fraud` or `target is_fraud`""")
            else:
                say("❌ Error processing the uploaded file. Please check the file format and try again.")
                
        except Exception as e:
            say(f"❌ Error processing file: {str(e)}")
    
    def handle_message(self, message, say, thread_ts=None):
        """Handle all user messages"""
        user_id = message["user"]
        text = message["text"].strip()
        
        logger.info(f"📨 RECEIVED MESSAGE | User: {user_id} | Text: '{text}' | Thread: {thread_ts}")
        
        # Skip bot messages
        if message.get("bot_id"):
            logger.info(f"⏭️ SKIPPING BOT MESSAGE | Bot ID: {message.get('bot_id')}")
            return
        
        # Check if user has an active session
        if user_id not in self.users:
            logger.info(f"❌ NO ACTIVE SESSION | User: {user_id} | No file uploaded yet")
            return
        
        session = self.users[user_id]
        logger.info(f"✅ ACTIVE SESSION FOUND | User: {user_id} | Phase: {session.phase} | Features: {len(session.current_features)}")
        
        # Add user_id to session for logging if not present
        if not hasattr(session, 'user_id') or session.user_id is None:
            session.user_id = user_id
        
        # Store thread_ts in session for use in say() calls
        session.thread_ts = thread_ts
        
        try:
            if session.phase == "need_target":
                logger.info(f"🎯 ROUTING TO TARGET SELECTION | User: {user_id}")
                self.handle_target_selection(session, text, say)
            elif session.phase == "waiting_datetime":
                logger.info(f"📅 ROUTING TO DATETIME SETUP | User: {user_id}")
                self.handle_datetime_setup(session, text, say)
            elif session.phase == "waiting_input":
                logger.info(f"🔄 ROUTING TO ANALYSIS REQUEST | User: {user_id}")
                self.handle_analysis_request(session, text, say)
            else:
                logger.warning(f"⚠️ UNKNOWN PHASE | User: {user_id} | Phase: {session.phase}")
                say("🤔 I'm not sure what to do right now. Try uploading a new file to start over.", thread_ts=thread_ts)
                
        except Exception as e:
            logger.error(f"💥 ERROR IN MESSAGE HANDLING | User: {user_id} | Error: {str(e)}")
            say(f"❌ Error processing your request: {str(e)}", thread_ts=thread_ts)
    
    def handle_target_selection(self, session: UserSession, text: str, say):
        """Handle target column selection"""
        # Extract target column name
        target_column = text
        if text.startswith("target"):
            parts = text.replace(":", "").split(maxsplit=1)
            if len(parts) > 1:
                target_column = parts[1].strip()
            else:
                say("🎯 Please specify the target column name.\nExample: `target is_fraud` or just `is_fraud`", thread_ts=None)
                return
        
        # Validate target column exists
        if target_column not in session.current_features:
            available_cols = ', '.join(session.current_features[:5])
            newline = "\n"
            say(f"❌ Column '{target_column}' not found.{newline}{newline}📋 Available columns: {available_cols}...{newline}{newline}🎯 Please try again:", thread_ts=None)
            return
        
        # Set target column and show menu
        session.target_column = target_column
        session.phase = "waiting_input"
        
        say(f"🎯 **Target column set:** {target_column}", thread_ts=None)
        
        # Show main menu
        menu = MenuGenerator.generate_main_menu(session)
        session.last_menu = menu
        say(menu, thread_ts=None)
    
    def handle_analysis_request(self, session: UserSession, text: str, say):
        """Handle analysis requests and queries"""
        logger.info(f"🧠 PARSING USER INTENT | File: {session.file_name} | Text: '{text}'")
        
        # Parse user intent
        intent_data = LLMManager.parse_user_intent(text, session)
        intent = intent_data.get("intent", "QUERY")
        
        logger.info(f"🎯 INTENT CLASSIFIED | User: {session.user_id} | Intent: {intent} | Data: {intent_data}")
        
        if intent == "STANDARD_ANALYSIS":
            logger.info(f"⚙️ ROUTING TO STANDARD ANALYSIS | User: {session.user_id} | Type: {intent_data.get('analysis_type', 'unknown')}")
            self.run_standard_analysis(session, intent_data, say)
        elif intent == "STANDARD_ANALYSIS_QUERY":
            logger.info(f"🔍 ROUTING TO STANDARD ANALYSIS QUERY | User: {session.user_id} | Type: {intent_data.get('analysis_type', 'unknown')}")
            self.handle_standard_analysis_query(session, intent_data, say)
        elif intent == "CUSTOM_ANALYSIS":
            logger.info(f"🔧 ROUTING TO CUSTOM ANALYSIS | User: {session.user_id} | Type: {intent_data.get('analysis_type', 'unknown')}")
            self.run_custom_analysis(session, intent_data, say)
        elif intent == "QUERY":
            logger.info(f"❓ ROUTING TO QUERY HANDLER | User: {session.user_id} | Query: {intent_data.get('query_details', 'unknown')}")
            self.handle_query(session, intent_data, say)
        elif intent == "GENERAL_QUERY":
            logger.info(f"💬 ROUTING TO GENERAL QUERY | User: {session.user_id} | Query: {intent_data.get('query_details', 'unknown')}")
            self.handle_general_query(session, intent_data, say)
        elif intent == "REVERT":
            logger.info(f"↩️ ROUTING TO REVERT | User: {session.user_id}")
            self.handle_revert(session, say)
        else:
            logger.warning(f"⚠️ UNKNOWN INTENT | User: {session.user_id} | Intent: {intent}")
            # Fallback
            say("🤔 I'm not sure how to help with that. Please try a specific analysis command or ask a question.")
            
        # Menu display is handled in individual methods to avoid duplication
    
    def run_standard_analysis(self, session: UserSession, intent_data: Dict[str, Any], say):
        """Run standard analyses"""
        analysis_type = intent_data.get("analysis_type", "").lower()
        threshold = intent_data.get("threshold", 0.05)
        
        # Show what we're doing
        say(f"🔄 **Running {analysis_type.upper()} Analysis** with threshold {threshold}...")
        
        if analysis_type == "iv":
            result = AnalysisEngine.run_iv_analysis(session, threshold)
            if result.get("success"):
                # Show detailed IV analysis results
                iv_scores = result.get("iv_scores", {})
                top_features = sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                
                response = f"""✅ **IV Analysis Complete**

📊 **Results:**
• Features before: {result.get('features_removed', 0) + result.get('remaining_features', 0)}
• Features removed: {result.get('features_removed', 0)} (IV < {threshold})
• Remaining features: {result.get('remaining_features', 0)}

📈 **Top IV Scores:**"""
                
                for feature, score in top_features:
                    if feature != session.target_column:  # Don't show target column IV
                        response += f"{chr(10)}• {feature}: {score:.4f}"
                
                response += f"\n\n🎯 **Analysis added to pipeline!**"
                say(response)
            else:
                say(f"❌ IV Analysis failed: {result.get('error', 'Unknown error')}")
            
        elif analysis_type == "correlation":
            result = AnalysisEngine.run_correlation_analysis(session, threshold)
            if result.get("success"):
                high_corr_pairs = result.get("high_corr_pairs", [])
                response = f"""✅ **Correlation Analysis Complete**

📊 **Results:**
• Features before: {result.get('features_removed', 0) + result.get('remaining_features', 0)}
• Features removed: {result.get('features_removed', 0)} (correlation > {threshold})
• Remaining features: {result.get('remaining_features', 0)}
• High correlation pairs found: {len(high_corr_pairs)}"""

                if high_corr_pairs:
                    response += f"{chr(10)}{chr(10)}📈 **Removed Correlations:**"
                    for col1, col2, corr_val in high_corr_pairs[:3]:  # Show top 3
                        response += f"{chr(10)}• {col1} ↔ {col2}: {corr_val:.3f}"
                    if len(high_corr_pairs) > 3:
                        response += f"{chr(10)}• ... and {len(high_corr_pairs) - 3} more"

                response += f"{chr(10)}{chr(10)}🎯 **Analysis added to pipeline!**"
                say(response)
            else:
                say(f"❌ Correlation Analysis failed: {result.get('error', 'Unknown error')}")
                
        elif analysis_type == "csi":
            # Use default CSI threshold if LLM didn't extract it
            if threshold is None:
                threshold = 0.2  # Default CSI threshold
            
            # Store the CSI threshold for later use
            session.pending_csi_threshold = threshold
            
            # CSI analysis requires datetime column
            if not session.datetime_column:
                # Ask for datetime column
                datetime_candidates = self._find_datetime_columns(session)
                
                if datetime_candidates:
                    response = f"""📅 **CSI Analysis Setup**

🔍 **Detected potential datetime columns:**
{chr(10).join([f'• {col}' for col in datetime_candidates[:5]])}

📝 **Please specify (in natural language):**
• **Datetime column**: e.g., "use transaction_date as datetime column"
• **OOT start month**: e.g., "set oot month to 2023M09"

**Example:** 
"use transaction_date as datetime column and set oot month to 2023M09\""""
                else:
                    # Get original columns for display
                    orig_cols = list(session.original_df.columns) if session.original_df is not None else session.current_features
                    response = f"""📅 **CSI Analysis Setup**

⚠️ **No obvious datetime columns detected.**

📝 **Please specify (in natural language):**
• **Datetime column**: e.g., "use your_date_column as datetime column"
• **OOT start month**: e.g., "set oot month to 2023M09"

📋 **Available columns:** {', '.join(orig_cols[:10])}{'...' if len(orig_cols) > 10 else ''}

**Example:** 
"use your_date_column as datetime column and set oot month to 2023M09\""""
                
                say(response)
                # Set phase to collect datetime info
                session.phase = "waiting_datetime"
                return
            else:
                # Run CSI analysis with existing datetime column
                result = AnalysisEngine.run_csi_analysis(session, threshold)
                if result.get("success"):
                    response = f"""✅ **CSI Analysis Complete**

📊 **Results:**
• Features before: {result.get('features_removed', 0) + result.get('remaining_features', 0)}
• Features removed: {result.get('features_removed', 0)} (CSI > {threshold})
• Remaining features: {result.get('remaining_features', 0)}
• Datetime column: {session.datetime_column}
• OOT month: {session.oot_month}

🎯 **Analysis added to pipeline!**"""
                    say(response)
                else:
                    say(f"❌ CSI Analysis failed: {result.get('error', 'Unknown error')}")
            
        elif analysis_type == "vif":
            # Extract threshold from intent data or use default
            if threshold is None:
                threshold = 5.0  # Default VIF threshold
            
            # Get thread logger
            thread_logger = get_thread_logger(session.user_id, session.user_id)
            
            print_to_log(f"🔧 DEBUG VIF: About to run VIF analysis with threshold {threshold}")
            thread_logger.log_analysis("vif", {"threshold": threshold}, {"status": "starting"})
            
            result = AnalysisEngine.run_vif_analysis(session, threshold)
            print_to_log(f"🔧 DEBUG VIF: Analysis result: {result}")
            
            # Log analysis completion
            thread_logger.log_analysis("vif", {"threshold": threshold}, result)
            
            if result.get("success"):
                vif_scores = result.get("vif_scores", {})
                top_features = sorted(vif_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                
                response = f"""✅ **VIF Analysis Complete**

📊 **Results:**
• Features before: {result.get('features_removed', 0) + result.get('remaining_features', 0)}
• Features removed: {result.get('features_removed', 0)} (VIF > {threshold})
• Remaining features: {result.get('remaining_features', 0)}

📈 **Top VIF Scores:**"""
                
                for feature, score in top_features:
                    if feature != session.target_column:  # Don't show target column VIF
                        response += f"{chr(10)}• {feature}: {score:.4f}"
                
                response += f"\n\n🎯 **Analysis added to pipeline!**"
                print_to_log(f"🔧 DEBUG VIF: Sending success response to Slack")
                say(response)
            elif result.get("error"):
                error_msg = f"❌ VIF Analysis failed: {result.get('error', 'Unknown error')}"
                print_to_log(f"🔧 DEBUG VIF: Sending error response to Slack: {error_msg}")
                say(error_msg)
            else:
                error_msg = f"❌ VIF Analysis failed: Unexpected result format: {result}"
                print_to_log(f"🔧 DEBUG VIF: Sending unexpected result error to Slack: {error_msg}")
                say(error_msg)
            
        else:
            say(f"🔧 Analysis type '{analysis_type}' is not yet implemented. Coming soon!")
    
    def run_custom_analysis(self, session: UserSession, intent_data: Dict[str, Any], say):
        """Run custom analyses like PCA, LASSO, advanced techniques"""
        analysis_type = intent_data.get("analysis_type", "").lower()
        query_details = intent_data.get("query_details", "")
        
        say(f"🔄 **Running {analysis_type.upper()} Analysis**...")
        
        if analysis_type == "shap":
            # Handle SHAP analysis as a direct tool
            threshold = intent_data.get("threshold", 0.01)
            top_n = None
            
            # Check if this is a "top N" request
            import re
            top_match = re.search(r'top\s+(\d+)', query_details.lower())
            if top_match:
                top_n = int(top_match.group(1))
                # Keep threshold for parameter logging, but top_n takes precedence
            
            result = AnalysisEngine.run_shap_analysis(session, threshold=threshold, top_n=top_n)
            if result.get("success"):
                shap_scores = result.get("shap_scores", {})
                top_features = sorted(shap_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                
                if top_n:
                    response = f"""✅ **SHAP Analysis Complete**

📊 **Results:**
• Features before: {result.get('features_removed', 0) + result.get('remaining_features', 0)}
• Kept top: {top_n} most important features
• Features removed: {result.get('features_removed', 0)}
• Remaining features: {result.get('remaining_features', 0)}

📈 **Top SHAP Scores:**"""
                else:
                    response = f"""✅ **SHAP Analysis Complete**

📊 **Results:**
• Features before: {result.get('features_removed', 0) + result.get('remaining_features', 0)}
• Features removed: {result.get('features_removed', 0)} (SHAP < {threshold})
• Remaining features: {result.get('remaining_features', 0)}

📈 **Top SHAP Scores:**"""
                
                for feature, score in top_features:
                    if feature != session.target_column:  # Don't show target column SHAP
                        response += f"{chr(10)}• {feature}: {score:.4f}"
                
                response += f"\n\n🎯 **Analysis added to pipeline!**"
                say(response)
            else:
                say(f"❌ SHAP Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            # Generate and execute custom code for analysis
            self._generate_and_execute_custom_analysis(session, intent_data, say)
    
    def _generate_and_execute_custom_analysis(self, session: UserSession, intent_data: Dict[str, Any], say):
        """Generate and execute custom analysis code using LLM"""
        analysis_type = intent_data.get("analysis_type", "").lower()
        query_details = intent_data.get("query_details", "")
        threshold = intent_data.get("threshold")
        
        say(f"🔄 **Running {analysis_type.upper()} Analysis**...")
        
        try:
            llm = LLMManager.get_llm(session.model_name)
            
            # Prepare context for code generation
            current_features = [f for f in session.current_features if f != session.target_column]
            feature_list = ', '.join([f"'{f}'" for f in current_features])
            
            # Generate analysis-specific prompt
            if "shap" in analysis_type.lower():
                prompt = self._generate_shap_code_prompt(session, query_details, threshold, feature_list)
            else:
                prompt = self._generate_general_analysis_prompt(analysis_type, session, query_details, threshold, feature_list)
            
            max_retries = 3
            fallback_model = "gpt-3.5-turbo"  # Fallback model for different approach
            
            for attempt in range(max_retries):
                # Use fallback model for final attempt if primary model keeps failing
                current_model = session.model_name
                if attempt == max_retries - 1 and session.model_name != fallback_model:
                    current_model = fallback_model
                    logger.info(f"🔄 CUSTOM ANALYSIS FALLBACK MODEL | User: {session.user_id} | Attempt: {attempt + 1} | Model: {current_model}")
                    llm = LLMManager.get_llm(current_model)
                    # Fallback model usage logged only, not shown to user
                
                try:
                    logger.info(f"🤖 CUSTOM ANALYSIS ATTEMPT {attempt + 1}/{max_retries} | User: {session.user_id} | Model: {current_model}")
                    response = llm.invoke([HumanMessage(content=prompt)])
                    logger.info(f"📝 CUSTOM ANALYSIS RESPONSE | User: {session.user_id} | Model: {current_model} | Length: {len(response.content)} chars")
                    
                    # Extract code from response
                    code = self._extract_code_from_response(response.content)
                    
                    if not code:
                        raise ValueError("No code found in LLM response")
                    
                    # Code generation logged in backend only, not shown to user
                    
                    # Execute the code
                    result = self._execute_custom_analysis_code(session, code)
                    
                    if result.get("success"):
                        # Update session with results
                        features_to_remove = result.get("features_to_remove", [])
                        analysis_scores = result.get("analysis_scores", {})
                        
                        # Apply feature removal
                        features_before = len(session.current_features)
                        remaining_features = [f for f in session.current_features if f not in features_to_remove]
                        session.current_df = session.current_df[remaining_features]
                        session.current_features = remaining_features
                        
                        # Add to pipeline
                        analysis_step = AnalysisStep(
                            type=f"{analysis_type}_analysis",
                            parameters={"threshold": threshold, "query": query_details},
                            features_before=features_before,
                            features_after=len(remaining_features),
                            timestamp=datetime.now().isoformat(),
                            metadata={"analysis_scores": analysis_scores, "removed_features": features_to_remove, "generated_code": True}
                        )
                        session.analysis_chain.append(analysis_step)
                        
                        # Create snapshot
                        session.snapshots[f"after_{analysis_type}"] = {
                            "features": remaining_features.copy(),
                            "df": session.current_df.copy(),
                            "step": len(session.analysis_chain) - 1
                        }
                        
                        response_msg = f"""✅ **{analysis_type.upper()} Analysis Complete**

📊 **Results:**
• Features before: {features_before}
• Features removed: {len(features_to_remove)} 
• Remaining features: {len(remaining_features)}

🎯 **Analysis added to pipeline!**"""
                        
                        say(response_msg)
                        
                        # Ask LLM to provide top features from the analysis instead of showing pipeline
                        if result.get("analysis_scores"):
                            self._generate_top_features_summary(session, analysis_type, result.get("analysis_scores", {}), say)
                        
                        return
                    else:
                        error_msg = result.get("error", "Unknown execution error")
                        logger.warning(f"⚠️ CUSTOM ANALYSIS ATTEMPT {attempt + 1} FAILED | User: {session.user_id} | Error: {error_msg}")
                        
                        # Update prompt with error for next attempt
                        prompt += f"\n\nPREVIOUS ATTEMPT FAILED WITH ERROR: {error_msg}\nPlease fix the code and avoid this error."
                        
                except Exception as e:
                    logger.warning(f"⚠️ CUSTOM ANALYSIS ATTEMPT {attempt + 1} EXCEPTION | User: {session.user_id} | Error: {str(e)}")
                    
                    # Update prompt with error for next attempt
                    prompt += f"\n\nPREVIOUS ATTEMPT FAILED WITH ERROR: {str(e)}\nPlease fix the code and avoid this error."
            
            # All attempts failed
            say(f"🔧 **{analysis_type.upper()} Analysis Coming Soon**: This analysis is temporarily unavailable. Please try other analyses or check back later.")
            
        except Exception as e:
            say(f"🔧 **Analysis Temporarily Unavailable**: Please try other analyses or check back later.")
    
    def _generate_top_features_summary(self, session: UserSession, analysis_type: str, analysis_scores: dict, say):
        """Generate LLM-based summary of top features from custom analysis"""
        try:
            llm = LLMManager.get_llm(session.model_name)
            
            # Format scores for LLM
            sorted_scores = sorted(analysis_scores.items(), key=lambda x: x[1], reverse=True)
            scores_text = "\n".join([f"• {feature}: {score:.4f}" for feature, score in sorted_scores[:15]])  # Top 15 for context
            
            prompt = f"""Based on the {analysis_type.upper()} analysis results, provide a helpful summary of the top features.

{analysis_type.upper()} SCORES FOR FEATURES:
{scores_text}

INSTRUCTIONS:
- Show the TOP 5-10 most important features with their scores
- Explain what these scores mean for {analysis_type.upper()} analysis
- Be concise and practical
- Use format: "feature_name: score_value"
- Add a brief interpretation of what high/low scores indicate

Example format:
🏆 **Top {analysis_type.upper()} Features:**
• feature_A: 0.85 - Highest {analysis_type.lower()} score
• feature_B: 0.72 - Strong {analysis_type.lower()} importance  
• feature_C: 0.68 - Good {analysis_type.lower()} value
...

💡 **Interpretation:** Higher {analysis_type.lower()} scores indicate [brief explanation]
"""
            
            logger.info(f"🤖 GENERATING TOP FEATURES SUMMARY | User: {session.user_id} | Analysis: {analysis_type}")
            response = llm.invoke([HumanMessage(content=prompt)])
            
            logger.info(f"✅ TOP FEATURES SUMMARY GENERATED | User: {session.user_id} | Length: {len(response.content)}")
            say(f"{response.content}")
            
        except Exception as e:
            logger.error(f"💥 TOP FEATURES SUMMARY FAILED | User: {session.user_id} | Error: {str(e)}")
            # Fallback to simple top 5 list
            sorted_scores = sorted(analysis_scores.items(), key=lambda x: x[1], reverse=True)
            top_5 = sorted_scores[:5]
            fallback_msg = f"🏆 **Top 5 {analysis_type.upper()} Features:**\n"
            for i, (feature, score) in enumerate(top_5, 1):
                fallback_msg += f"{i}. {feature}: {score:.4f}\n"
            say(fallback_msg)
    
    def _extract_code_from_response(self, response_content: str) -> str:
        """Extract Python code from LLM response - enhanced for robustness"""
        import re
        
        # Remove any "python" language identifier that might be on its own line
        response_content = re.sub(r'^python\s*$', '', response_content, flags=re.MULTILINE)
        
        # Look for code blocks with python specification
        code_blocks = re.findall(r'```python\n(.*?)\n```', response_content, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Look for code blocks without language specification  
        code_blocks = re.findall(r'```\n(.*?)\n```', response_content, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Look for code blocks with just ``` at start and end
        code_blocks = re.findall(r'```(.*?)```', response_content, re.DOTALL)
        if code_blocks:
            code = code_blocks[0].strip()
            # Remove "python" if it's the first line
            if code.startswith('python\n'):
                code = code[7:]
            return code.strip()
        
        # If no code blocks, extract everything that looks like Python code
        lines = response_content.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Start collecting when we see imports or variable assignments
            if (line.strip().startswith('import ') or 
                line.strip().startswith('from ') or
                line.strip().startswith('# ') or
                (not in_code and '=' in line and any(keyword in line for keyword in ['df', 'feature', 'X', 'y']))):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            code = '\n'.join(code_lines)
            # Ensure we have the required variables
            if 'features_to_remove' not in code:
                code += '\n\n# Ensure required variables exist\nif "features_to_remove" not in locals():\n    features_to_remove = []\nif "analysis_scores" not in locals():\n    analysis_scores = {}'
            return code
        
        # Fallback: return the whole response and let execution fail gracefully
        return response_content.strip()
    
    def _generate_shap_code_prompt(self, session: UserSession, query_details: str, threshold: float, feature_list: str) -> str:
        """Generate SHAP-specific code prompt"""
        return f"""Generate Python code for SHAP feature importance analysis.

CONTEXT:
- DataFrame variable name: df
- Target column: '{session.target_column}'
- Available features: [{feature_list}]
- Analysis request: "{query_details}"
- Threshold: {threshold if threshold else 'auto-detect from request or use default'}

MANDATORY OUTPUT VARIABLES:
You MUST define these two variables at the end:
1. features_to_remove = [list of feature names to remove]
2. analysis_scores = {{feature_name: absolute_shap_importance, ...}}

COMPLETE SHAP CODE TEMPLATE:
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

# Prepare data
feature_cols = [{feature_list}]
X = df[feature_cols].fillna(df[feature_cols].mean())
y = df['{session.target_column}']

# Train model (RandomForest by default, can be customized)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Handle multi-class case
if isinstance(shap_values, list):
    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

# Calculate mean absolute SHAP values per feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Create analysis_scores dictionary
analysis_scores = {{}}
for i, feature in enumerate(feature_cols):
    analysis_scores[feature] = mean_abs_shap[i]

# Determine features to remove based on threshold
threshold_value = {threshold if threshold else '0.01'}  # Default SHAP threshold
features_to_remove = []

# Handle different query types
if "top" in "{query_details}".lower():
    # Extract top N features (remove the rest)
    try:
        import re
        top_match = re.search(r'top\\s+(\\d+)', "{query_details}".lower())
        if top_match:
            top_n = int(top_match.group(1))
            sorted_features = sorted(analysis_scores.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_features[:top_n]]
            features_to_remove = [f for f in feature_cols if f not in top_features]
        else:
            # Default: remove features below threshold
            for feature, score in analysis_scores.items():
                if score < threshold_value:
                    features_to_remove.append(feature)
    except:
        # Fallback to threshold filtering
        for feature, score in analysis_scores.items():
            if score < threshold_value:
                features_to_remove.append(feature)
else:
    # Threshold-based filtering
    for feature, score in analysis_scores.items():
        if score < threshold_value:
            features_to_remove.append(feature)

print_to_log(f"SHAP analysis complete. Features to remove: {{len(features_to_remove)}}")
```

CRITICAL REQUIREMENTS:
1. MUST define features_to_remove as a list
2. MUST define analysis_scores as a dictionary with SHAP importance values
3. Handle missing values with .fillna()
4. Support both threshold filtering and top-N selection
5. Use RandomForest or specified model
6. No markdown formatting - just pure Python code

Generate the complete code now:"""

    def _generate_general_analysis_prompt(self, analysis_type: str, session: UserSession, query_details: str, threshold: float, feature_list: str) -> str:
        """Generate general analysis code prompt for LASSO, PCA, etc."""
        return f"""Generate Python code for {analysis_type.upper()} feature selection analysis.

CONTEXT:
- DataFrame variable name: df
- Target column: '{session.target_column}'
- Available features: [{feature_list}]
- Analysis request: "{query_details}"
- Threshold: {threshold if threshold else 'auto-detect from request or use default'}

MANDATORY OUTPUT VARIABLES:
You MUST define these two variables at the end:
1. features_to_remove = [list of feature names to remove]
2. analysis_scores = {{feature_name: score, ...}}

COMPLETE CODE TEMPLATE FOR {analysis_type.upper()}:
```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Prepare data
feature_cols = [{feature_list}]
X = df[feature_cols].fillna(df[feature_cols].mean())
y = df['{session.target_column}']

# Scale features for stability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply LASSO with cross-validation
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=1000)
lasso_cv.fit(X_scaled, y)

# Get feature importance (absolute coefficients)
feature_importance = np.abs(lasso_cv.coef_)

# Create analysis_scores dictionary
analysis_scores = {{}}
for i, feature in enumerate(feature_cols):
    analysis_scores[feature] = feature_importance[i]

# Determine features to remove based on threshold
threshold_value = {threshold if threshold else '0.01'}  # Default threshold
features_to_remove = []
for feature, score in analysis_scores.items():
    if score < threshold_value:
        features_to_remove.append(feature)

print_to_log(f"{analysis_type.upper()} analysis complete. Features to remove: {{len(features_to_remove)}}")
```

CRITICAL REQUIREMENTS:
1. MUST define features_to_remove as a list
2. MUST define analysis_scores as a dictionary
3. Handle missing values with .fillna()
4. Use appropriate threshold for feature selection
5. Include proper imports
6. No markdown formatting - just pure Python code

Generate the complete code now:"""
    
    def _execute_custom_analysis_code(self, session: UserSession, code: str) -> Dict[str, Any]:
        """Execute custom analysis code safely"""
        try:
            # Create execution environment
            exec_globals = {
                'df': session.current_df.copy(),
                'pd': pd,
                'np': np,
                'session': session,  # In case code needs session info
                '__builtins__': __builtins__
            }
            
            # Add common imports to globals
            try:
                import sklearn
                import scipy
                exec_globals['sklearn'] = sklearn
                exec_globals['scipy'] = scipy
            except ImportError:
                pass
            
            exec_locals = {}
            
            # Execute the code
            exec(code, exec_globals, exec_locals)
            
            # Extract results
            features_to_remove = exec_locals.get('features_to_remove', [])
            analysis_scores = exec_locals.get('analysis_scores', {})
            
            # Validate results
            if not isinstance(features_to_remove, list):
                return {"error": "features_to_remove must be a list"}
            
            if not isinstance(analysis_scores, dict):
                return {"error": "analysis_scores must be a dictionary"}
            
            # Ensure features_to_remove are valid feature names
            valid_features = set(session.current_features)
            invalid_features = [f for f in features_to_remove if f not in valid_features]
            if invalid_features:
                return {"error": f"Invalid features to remove: {invalid_features}"}
            
            return {
                "success": True,
                "features_to_remove": features_to_remove,
                "analysis_scores": analysis_scores
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def handle_revert(self, session: UserSession, say):
        """Handle revert to cleaned state (after removing single-value and object columns)"""
        if "after_cleaning" in session.snapshots:
            # Revert to cleaned state (after single-value and object column removal)
            snapshot = session.snapshots["after_cleaning"]
            session.current_df = snapshot["df"].copy()
            session.current_features = snapshot["features"].copy()
            
            # Clear analysis chain except cleaning
            session.analysis_chain = session.analysis_chain[:1]  # Keep only cleaning step
            
            # Clear other snapshots except the cleaning snapshot
            session.snapshots = {"after_cleaning": snapshot}
            
            # Reset datetime settings if they were set
            session.datetime_column = None
            session.oot_month = None
            session.phase = "waiting_input"
            
            say(f"""✅ **Reverted to Cleaned Dataset**

📊 **Current State:**
• Features: {len(session.current_features)} (after removing single-value & object columns)
• Pipeline: data_cleaning only
• Removed analysis steps: {len(session.analysis_chain) - 1 if len(session.analysis_chain) > 1 else 0}

🔄 **Ready for fresh analysis!**""")
            
            # Show crisp summary
            MenuGenerator.show_crisp_summary(session, say)
        else:
            say("❌ No cleaned state available to revert to.")
        
        # Show updated menu (only if not waiting for datetime and first analysis)
        if session.phase != "waiting_datetime":
            # Show crisp summary instead of full menu after first analysis
            if len(session.analysis_chain) > 1:
                MenuGenerator.show_crisp_summary(session, say)
            else:
                menu = MenuGenerator.generate_main_menu(session)
                say(menu)
    
    def _find_datetime_columns(self, session: UserSession) -> List[str]:
        """Find potential datetime columns in the original dataset"""
        datetime_candidates = []
        
        # Look in original dataframe since datetime columns might have been removed during cleaning
        columns_to_check = list(session.original_df.columns) if session.original_df is not None else session.current_features
        
        for col in columns_to_check:
            col_lower = col.lower()
            # Check for common datetime column patterns
            if any(pattern in col_lower for pattern in [
                'date', 'time', 'timestamp', 'created', 'updated', 
                'modified', 'transaction', 'event', 'log', 'month', 'year'
            ]):
                datetime_candidates.append(col)
        
        return datetime_candidates
    
    def handle_datetime_setup(self, session: UserSession, text: str, say):
        """Handle datetime column and OOT month setup for CSI analysis using LLM"""
        try:
            llm = LLMManager.get_llm(session.model_name)
            
            # Show original columns for datetime selection since datetime columns might have been removed
            original_columns = list(session.original_df.columns) if session.original_df is not None else session.current_features
            available_cols = ', '.join(original_columns[:10])
            current_datetime = session.datetime_column or "Not set"
            current_oot = session.oot_month or "Not set"
            
            prompt = f"""The user is setting up CSI analysis and needs to specify datetime column and OOT month.

AVAILABLE COLUMNS: {available_cols}{'...' if len(original_columns) > 10 else ''}
CURRENT STATE:
- Datetime column: {current_datetime}
- OOT month: {current_oot}

USER INPUT: "{text}"

Extract datetime column name and/or OOT month from their message. They might say things like:
- "use transaction_date as datetime column"
- "datetime column is created_at"
- "oot month is 2023M09"
- "set oot to September 2023"
- "use created_at and oot month 2023M12"
- "datetime is transaction_date, oot is 2024M01"

IMPORTANT: OOT month format should be YYYYMXX (like 2023M09, 2024M01, etc.)

Respond with ONLY this JSON format:
{{
    "datetime_column": "column_name_if_mentioned_or_null",
    "oot_month": "YYYYMXX_format_if_mentioned_or_null", 
    "interpretation": "What you understood from their message"
}}

Convert date formats to YYYYMXX:
- "January 2024" -> "2024M01"
- "September 2023" -> "2023M09" 
- "2023-06" -> "2023M06"
- "2023M09" -> keep as "2023M09"
- "06/2023" -> "2023M06\""""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Parse LLM response
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split('\n')
                content = '\n'.join([line for line in lines if not line.strip().startswith('```') and not line.strip().startswith('json')])
            
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            result = json.loads(content)
            
            # Process datetime column - check in ORIGINAL dataset, not cleaned features
            datetime_col = result.get("datetime_column")
            if datetime_col and datetime_col != "null":
                # Check in original dataframe since datetime columns might have been removed during cleaning
                original_columns = list(session.original_df.columns) if session.original_df is not None else session.current_features
                if datetime_col in original_columns:
                    # Only show confirmation if it's actually changing
                    if session.datetime_column != datetime_col:
                        session.datetime_column = datetime_col
                        say(f"✅ **Datetime column set:** {datetime_col}")
                    else:
                        session.datetime_column = datetime_col  # Set silently if same value
                else:
                    available_cols_short = ', '.join(original_columns[:5])
                    newline = "\n"
                    say(f"❌ Column '{datetime_col}' not found.{newline}{newline}📋 Available columns: {available_cols_short}...{newline}{newline}Please specify an existing column name from the original dataset.")
                    return
            
            # Process OOT month
            oot_month = result.get("oot_month")
            if oot_month and oot_month != "null":
                # Only show confirmation if it's actually changing
                if session.oot_month != oot_month:
                    session.oot_month = oot_month
                    say(f"✅ **OOT month set:** {oot_month}")
                else:
                    session.oot_month = oot_month  # Set silently if same value
            
            # Check if we have both and can run CSI
            if session.datetime_column and session.oot_month and not getattr(session, 'csi_completed', False):
                # Use the previously stored threshold
                threshold = session.pending_csi_threshold or 0.2
                say(f"🔄 **Running CSI Analysis** with threshold {threshold}...")
                self._run_csi_analysis(session, say, threshold)
                # Clear the pending threshold after use
                session.pending_csi_threshold = None
                # Mark CSI as completed to prevent re-execution
                session.csi_completed = True
            elif session.datetime_column and session.oot_month and getattr(session, 'csi_completed', False):
                # CSI already completed, just continue without any message
                pass
            else:
                # Show what's still needed
                still_need = []
                if not session.datetime_column:
                    still_need.append("datetime column")
                if not session.oot_month:
                    still_need.append("OOT month")
                
                say(f"📅 **Still need:** {' and '.join(still_need)}")
                
                if not session.datetime_column:
                    say(f"💡 **Available columns:** {available_cols}")
                    say("📝 **Example:** 'use transaction_date as datetime column'")
                
                if not session.oot_month:
                    say("📝 **Example:** 'set oot month to 2023M09'")
                    
        except Exception as e:
            say(f"❌ Error parsing datetime setup: {str(e)}")
            say("📝 **Please specify:**\n• Datetime column from your data\n• OOT month in YYYYMXX format (e.g., 2023M09)\n\n**Examples:**\n• 'use transaction_date as datetime column'\n• 'set oot month to 2023M09'")
            
    
    def _run_csi_analysis(self, session: UserSession, say, threshold: float = 0.2):
        """Run CSI analysis after datetime setup is complete"""
        try:
            result = AnalysisEngine.run_csi_analysis(session, threshold=threshold)
            if result.get("success"):
                response = f"""✅ **CSI Analysis Complete**

📊 **Results:**
• Features before: {result.get('features_removed', 0) + result.get('remaining_features', 0)}
• Features removed: {result.get('features_removed', 0)} (CSI > {threshold})
• Remaining features: {result.get('remaining_features', 0)}
• Datetime column: {session.datetime_column}
• OOT month: {session.oot_month}

🎯 **Analysis added to pipeline!**"""
                say(response)
                
                # Reset phase - no waterfall summary
                session.phase = "waiting_input"
            else:
                say(f"❌ CSI Analysis failed: {result.get('error', 'Unknown error')}")
                session.phase = "waiting_input"
        except Exception as e:
            say(f"❌ CSI Analysis error: {str(e)}")
            session.phase = "waiting_input"
    
    def handle_query(self, session: UserSession, intent_data: Dict[str, Any], say):
        """Handle user queries - both text responses and code-based analytical queries"""
        query = intent_data.get("query_details", "")
        
        logger.info(f"❓ HANDLING QUERY | User: {session.user_id} | Query: '{query}'")
        
        # Let LLM handle all query routing - no keyword matching
        
        # Check if this is a computational query that needs code execution
        logger.info(f"🤔 CHECKING EXECUTION REQUIREMENT | User: {session.user_id} | Query: '{query}'")
        if self._requires_code_execution(session, query):
            logger.info(f"💻 CODE EXECUTION REQUIRED | User: {session.user_id} | Routing to code-based query handler")
            self._handle_code_based_query(session, query, say)
            return
        else:
            logger.info(f"💬 TEXT RESPONSE SUFFICIENT | User: {session.user_id} | Routing to LLM text response")
        
        try:
            llm = LLMManager.get_llm(session.model_name)
            
            # Prepare context information
            features_info = f"{len(session.current_features)} features: {', '.join(session.current_features[:10])}{'...' if len(session.current_features) > 10 else ''}"
            pipeline_info = " -> ".join([step.type for step in session.analysis_chain]) if session.analysis_chain else "No analyses performed yet"
            
            # Get analysis details
            analysis_details = []
            for step in session.analysis_chain:
                details = f"{step.type}: {step.features_before} -> {step.features_after} features"
                if step.parameters:
                    params = ", ".join([f"{k}={v}" for k, v in step.parameters.items()])
                    details += f" ({params})"
                analysis_details.append(details)
            
            snapshots_info = list(session.snapshots.keys()) if session.snapshots else ["No snapshots available"]
            
            prompt = f"""Answer the user's specific question with a DIRECT, CRISP response. No menus, no general information.

CURRENT DATA:
• Target: {session.target_column}
• Features ({len(session.current_features)}): {', '.join(session.current_features[:15])}{'...' if len(session.current_features) > 15 else ''}
• Pipeline: {pipeline_info}

ANALYSIS DETAILS:
{chr(10).join(analysis_details) if analysis_details else 'No analyses performed yet'}

USER QUESTION: "{query}"

INSTRUCTIONS:
- If they ask "how many features have IV > X" - give exact count and list them
- If they ask "which features" - list them directly  
- If they ask about specific analysis - give specific numbers
- If they ask about pipeline - show the sequence
- Keep it SHORT and DIRECT - max 2-3 lines
- NO menus, NO general suggestions
- Just answer their specific question

Example good responses:
- "5 features have IV > 0.04: feature_A (0.67), feature_B (0.45), feature_C (0.39), feature_D (0.12), feature_E (0.05)"
- "Current features: feature_A, feature_B, target_col (3 total)"
- "Pipeline removed: 12 features via IV analysis, 3 via correlation analysis\""""

            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Send the LLM's response
            say(response.content)
            
        except Exception as e:
            print_to_log(f"❌ Error in LLM query handling: {e}")
            # Minimal fallback
            if "features" in query.lower():
                newline = "\n"
                say(f"📊 **Current Features:** {len(session.current_features)} features{newline}{newline}Features: {', '.join(session.current_features[:10])}{'...' if len(session.current_features) > 10 else ''}")
            elif "pipeline" in query.lower() or "summary" in query.lower():
                self._generate_detailed_summary(session, say)
            else:
                newline = "\n"
                say(f"💡 **Query about:** {query}{newline}{newline}I can help you with feature information, pipeline status, and analysis suggestions.")
    
    def handle_general_query(self, session: UserSession, intent_data: dict, say):
        """Handle general theoretical questions about feature selection and bot capabilities"""
        try:
            user_query = intent_data.get('query_details', '')
            llm = LLMManager.get_llm(session.model_name)
            
            prompt = f"""You are an expert feature selection assistant. Answer this theoretical question briefly and clearly in 1-2 lines.

USER QUESTION: "{user_query}"

CONTEXT: You are a conversational feature selection bot that helps users:
- Run standard analyses: IV (Information Value), CSI (Characteristic Stability Index), Correlation
- Apply custom filtering: VIF, PCA, LASSO, SHAP
- Generate code for complex analytical queries
- Provide interactive feature selection workflow

INSTRUCTIONS:
- Give a concise, practical answer in 1-2 lines
- Focus on feature selection and modeling concepts
- If asked about capabilities, mention your key analyses briefly
- Use simple, clear language
- No need for detailed explanations unless specifically asked

EXAMPLES:
Q: "What is IV analysis?" 
A: "IV (Information Value) measures how well a feature predicts the target variable. Higher IV values (>0.1) indicate stronger predictive features."

Q: "What can you do?"
A: "I help with feature selection using IV, correlation, CSI, VIF, PCA, LASSO, and SHAP analyses. I can filter features, answer data queries, and generate custom analysis code."

Q: "What is CSI?"
A: "CSI (Characteristic Stability Index) measures feature stability over time periods. Values >0.2 indicate unstable features that should be removed."
"""
            
            logger.info(f"🤖 GENERATING GENERAL RESPONSE | User: {session.user_id} | Query: {user_query[:50]}...")
            response = llm.invoke([HumanMessage(content=prompt)])
            
            logger.info(f"✅ GENERAL RESPONSE GENERATED | User: {session.user_id} | Length: {len(response.content)}")
            say(f"💡 {response.content}")
            
        except Exception as e:
            logger.error(f"💥 GENERAL QUERY FAILED | User: {session.user_id} | Error: {str(e)}")
            say(f"❌ **Error answering your question**: {str(e)}")
    
    def handle_suggestion_request(self, session: UserSession, intent_data: dict, say):
        """Handle requests for analysis suggestions based on current progress"""
        try:
            llm = LLMManager.get_llm(session.model_name)
            
            # Prepare current state context
            current_features = len(session.current_features)
            original_features = len(session.original_df.columns) - 1  # -1 for target
            
            # Analyze completed analyses
            completed_analyses = []
            for step in session.analysis_chain:
                if step.type != "data_cleaning":
                    params = ""
                    if step.parameters:
                        threshold = step.parameters.get('threshold')
                        if threshold:
                            params = f" (threshold: {threshold})"
                    completed_analyses.append(f"{step.type}{params}: {step.features_before} → {step.features_after}")
            
            # Determine data characteristics
            has_datetime_cols = any(col for col in session.original_df.columns 
                                  if session.original_df[col].dtype == 'object' 
                                  and 'date' in col.lower() or 'time' in col.lower())
            
            feature_count_status = "many" if current_features > 50 else "moderate" if current_features > 20 else "few"
            
            prompt = f"""You are an expert data scientist providing personalized feature selection recommendations.

CURRENT SITUATION:
- Dataset: {current_features} features remaining (started with {original_features})
- Target column: {session.target_column}
- Completed analyses: {', '.join(completed_analyses) if completed_analyses else 'None yet'}
- Feature count: {feature_count_status} features
- Has datetime columns: {has_datetime_cols}

AVAILABLE ANALYSES:
1. IV Analysis - Filter by predictive power (good for removing weak predictors)
2. Correlation Analysis - Remove redundant highly correlated features  
3. CSI Analysis - Check feature stability over time (needs datetime column)
4. VIF Analysis - Reduce multicollinearity between features
5. SHAP Analysis - Feature importance ranking using tree models
6. PCA Analysis - Dimensionality reduction while preserving variance
7. LASSO Analysis - L1 regularization for automatic feature selection

USER REQUEST: "{intent_data.get('query_details', '')}"

INSTRUCTIONS:
- Recommend 2-3 specific next steps based on current progress
- Explain WHY each analysis makes sense given the current state
- Consider what analyses haven't been done yet
- Suggest commands in NATURAL CONVERSATIONAL language (like talking to the bot)
- Use phrases like "IV analysis 0.05", "correlation analysis 0.8" - simple and human
- Be encouraging and helpful like a senior data scientist mentor
- Keep suggestions practical and actionable

COMMAND EXAMPLES:
- "IV analysis 0.05" (NOT "run_iv_analysis(threshold=0.05)")
- "correlation analysis 0.8" (NOT "run_correlation_analysis(threshold=0.8)")
- "CSI analysis 0.2" (NOT "run_csi_analysis()")
- "VIF analysis 5" (NOT "vif_analysis(threshold=5)")
- "SHAP analysis" (NOT "run_shap_analysis()")
- "PCA analysis 95% variance" (NOT "pca_analysis(variance=0.95)")

RESPONSE FORMAT:
🎯 **Recommended Next Steps:**

1. **[Analysis Name]** - [Brief explanation why this makes sense now]
   • Just say: `[natural conversational command]`
   • Why now: [specific reason based on current state]

2. **[Analysis Name]** - [Brief explanation]
   • Just say: `[natural conversational command]`
   • Why now: [reason]

💡 **Data Science Insight:** [Brief insight about the overall strategy]
"""
            
            logger.info(f"🤖 GENERATING ANALYSIS SUGGESTIONS | User: {session.user_id} | Features: {current_features}")
            response = llm.invoke([HumanMessage(content=prompt)])
            
            logger.info(f"✅ SUGGESTIONS GENERATED | User: {session.user_id} | Length: {len(response.content)}")
            say(f"{response.content}")
            
        except Exception as e:
            logger.error(f"💥 SUGGESTION REQUEST FAILED | User: {session.user_id} | Error: {str(e)}")
            # Fallback suggestions based on simple rules
            fallback_msg = "💡 **Suggestions based on your current progress:**\n\n"
            
            if not any(step.type == "iv_analysis" for step in session.analysis_chain):
                fallback_msg += "1. **IV Analysis** - Start with `IV analysis 0.05` to remove weak predictors\n"
            
            if not any(step.type == "correlation_analysis" for step in session.analysis_chain):
                fallback_msg += "2. **Correlation Analysis** - Try `correlation analysis 0.8` to remove redundant features\n"
            
            if current_features > 20:
                fallback_msg += "3. **Dimensionality Reduction** - Consider `PCA analysis` to reduce feature count\n"
            
            fallback_msg += "\n💡 These are standard next steps for feature selection!"
            say(fallback_msg)
    
    def handle_standard_analysis_query(self, session: UserSession, intent_data: Dict[str, Any], say):
        """Handle queries about standard analysis results - run analysis first, then answer query"""
        analysis_type = intent_data.get("analysis_type", "").lower()
        query_details = intent_data.get("query_details", "")
        
        logger.info(f"🔬 STANDARD ANALYSIS QUERY | User: {session.user_id} | Type: {analysis_type} | Query: '{query_details}'")
        
        # Run the appropriate standard analysis without filtering to get all values
        analysis_results = None
        
        if analysis_type == "iv":
            logger.info(f"📊 RUNNING IV ANALYSIS FOR QUERY | User: {session.user_id}")
            say("🔍 **Computing IV values for all features...**")
            analysis_results = AnalysisEngine.run_iv_analysis(session, threshold=0.0)
        elif analysis_type == "correlation":
            logger.info(f"📊 RUNNING CORRELATION ANALYSIS FOR QUERY | User: {session.user_id}")
            say("🔍 **Computing correlation values for all features...**")
            analysis_results = AnalysisEngine.run_correlation_analysis(session, threshold=0.0)
        elif analysis_type == "shap":
            logger.info(f"📊 RUNNING SHAP ANALYSIS FOR QUERY | User: {session.user_id}")
            say("🔍 **Computing SHAP values for all features...**")
            analysis_results = AnalysisEngine.run_shap_analysis(session, threshold=0.0)
        elif analysis_type == "csi":
            logger.info(f"📊 RUNNING CSI ANALYSIS FOR QUERY | User: {session.user_id}")
            say("🔍 **Computing CSI values for all features...**")
            if session.datetime_column and session.oot_month:
                analysis_results = AnalysisEngine.run_csi_analysis(session, threshold=0.0)
            else:
                say("❌ **CSI analysis requires datetime column and OOT month setup first.**")
                return
        elif analysis_type == "vif":
            logger.info(f"📊 RUNNING VIF ANALYSIS FOR QUERY | User: {session.user_id}")
            say("🔍 **Computing VIF values for all features...**")
            # VIF is in custom analysis, but we can still compute it
            from sklearn.feature_selection import VarianceThreshold
            import pandas as pd
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            # Simple VIF calculation
            numeric_features = [f for f in session.current_features if f != session.target_column]
            X = session.current_df[numeric_features].fillna(0)
            
            vif_scores = {}
            for i, feature in enumerate(numeric_features):
                try:
                    vif = variance_inflation_factor(X.values, i)
                    vif_scores[feature] = vif if not pd.isna(vif) else 0
                except:
                    vif_scores[feature] = 0
            
            analysis_results = {"success": True, "vif_scores": vif_scores}
        else:
            logger.warning(f"⚠️ UNKNOWN ANALYSIS TYPE | User: {session.user_id} | Type: {analysis_type}")
            logger.info(f"🔄 FALLBACK TO CUSTOM ANALYSIS | User: {session.user_id} | Type: {analysis_type}")
            say(f"🔄 **{analysis_type.upper()} not available in standard analyses, routing to custom code execution...**")
            
            # Fallback: Route to CUSTOM_ANALYSIS for code execution
            fallback_intent_data = {
                "intent": "CUSTOM_ANALYSIS",
                "analysis_type": analysis_type,
                "threshold": intent_data.get("threshold"),
                "comparison": intent_data.get("comparison"),
                "query_details": intent_data.get("query_details", f"Custom {analysis_type} analysis"),
                "extracted_info": intent_data.get("extracted_info", "")
            }
            
            # Route to custom analysis handler
            self.run_custom_analysis(session, fallback_intent_data, say)
            return
        
        if not analysis_results or not analysis_results.get("success"):
            logger.error(f"💥 ANALYSIS FAILED | User: {session.user_id} | Type: {analysis_type}")
            say(f"❌ **{analysis_type.upper()} analysis failed.**")
            return
        
        # Extract the scores based on analysis type
        scores = {}
        if analysis_type == "iv":
            scores = analysis_results.get("iv_scores", {})
        elif analysis_type == "correlation":
            scores = analysis_results.get("correlation_scores", {})
        elif analysis_type == "shap":
            scores = analysis_results.get("shap_scores", {})
        elif analysis_type == "csi":
            scores = analysis_results.get("csi_scores", {})
        elif analysis_type == "vif":
            scores = analysis_results.get("vif_scores", {})
        
        if not scores:
            logger.warning(f"⚠️ NO SCORES FOUND | User: {session.user_id} | Type: {analysis_type}")
            say(f"❌ **No {analysis_type.upper()} scores found.**")
            return
        
        logger.info(f"✅ ANALYSIS COMPLETE | User: {session.user_id} | Type: {analysis_type} | Features: {len(scores)}")
        
        # Now use LLM to answer the specific query using these scores
        try:
            llm = LLMManager.get_llm(session.model_name)
            
            # Format scores for LLM
            scores_text = "\n".join([f"• {feature}: {score:.4f}" for feature, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)])
            
            prompt = f"""Answer the user's specific question about {analysis_type.upper()} analysis results.

USER QUERY: "{query_details}"

{analysis_type.upper()} SCORES FOR ALL FEATURES:
{scores_text}

INSTRUCTIONS:
- Answer the user's exact question using these scores
- If they ask for "top N", list them in order with scores
- If they ask "how many > X", give exact count and list them
- If they ask about specific cutoffs, calculate and respond
- Be direct and concise
- Use the format: "feature_name: score_value"

Example responses:
- "Top 5 features by {analysis_type.upper()}: feature_A: 0.67, feature_B: 0.45, feature_C: 0.39, feature_D: 0.25, feature_E: 0.18"
- "3 features have {analysis_type.upper()} > 0.5: feature_A (0.67), feature_B (0.55), feature_C (0.52)"
"""

            logger.info(f"🤖 CALLING LLM TO ANSWER QUERY | User: {session.user_id} | Type: {analysis_type}")
            response = llm.invoke([HumanMessage(content=prompt)])
            
            logger.info(f"✅ LLM RESPONSE GENERATED | User: {session.user_id} | Length: {len(response.content)}")
            say(f"📊 **{analysis_type.upper()} Query Results:**\n{response.content}")
            
        except Exception as e:
            logger.error(f"💥 LLM QUERY RESPONSE FAILED | User: {session.user_id} | Error: {str(e)}")
            say(f"❌ **Error answering query**: {str(e)}")
    
    def _requires_code_execution(self, session: UserSession, query: str) -> bool:
        """Use LLM to determine if query requires computational analysis"""
        logger.info(f"🔍 LLM EXECUTION DETECTION | User: {session.user_id} | Query: '{query}'")
        try:
            llm = LLMManager.get_llm(session.model_name)
            logger.info(f"🤖 CALLING LLM FOR EXECUTION DETECTION | User: {session.user_id} | Model: {session.model_name}")
            
            prompt = f"""Determine if this user query requires COMPUTATIONAL ANALYSIS (code execution) or can be answered with TEXT RESPONSE only.

USER QUERY: "{query}"

COMPUTATIONAL ANALYSIS means:
- Calculating/computing values from data (correlations, statistics, rankings)
- Sorting/ranking features by some metric
- Performing mathematical operations on the dataset
- Generating lists of top/bottom features with scores
- Creating visualizations or plots

TEXT RESPONSE means:
- Asking about current state/status
- General questions about the process
- Questions that can be answered from existing information
- Requests for explanations or guidance

Examples:
- "give me top 10 features by correlation" → COMPUTATIONAL (needs to calculate correlations)
- "show me SHAP values" → COMPUTATIONAL (needs to compute SHAP scores)
- "how many features do I have?" → TEXT (simple count from existing data)
- "what analyses have I done?" → TEXT (can answer from analysis history)
- "calculate feature importance" → COMPUTATIONAL (needs computation)
- "what is correlation analysis?" → TEXT (explanation)

Respond with ONLY: "COMPUTATIONAL" or "TEXT\""""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip().upper()
            
            logger.info(f"🎯 LLM EXECUTION DECISION | User: {session.user_id} | Decision: {result} | Query: '{query}'")
            
            is_computational = result == "COMPUTATIONAL"
            logger.info(f"{'💻' if is_computational else '💬'} EXECUTION TYPE DETERMINED | User: {session.user_id} | Computational: {is_computational}")
            
            return is_computational
            
        except Exception as e:
            # Fallback: if LLM fails, use simple heuristics
            logger.warning(f"⚠️ LLM EXECUTION DETECTION FAILED | User: {session.user_id} | Error: {e} | Using fallback")
            computational_keywords = ["top", "calculate", "show me", "correlation", "importance", "shap"]
            is_computational = any(keyword in query.lower() for keyword in computational_keywords)
            logger.info(f"🔄 FALLBACK DECISION | User: {session.user_id} | Computational: {is_computational} | Keywords matched: {[k for k in computational_keywords if k in query.lower()]}")
            return is_computational
    
    def _handle_code_based_query(self, session: UserSession, query: str, say):
        """Handle queries that require code execution for computation"""
        logger.info(f"💻 STARTING CODE-BASED QUERY | User: {session.user_id} | Query: '{query}'")
        say(f"🔍 **Analyzing Data**: {query}")
        
        try:
            llm = LLMManager.get_llm(session.model_name)
            
            # Prepare context for code generation
            current_features = [f for f in session.current_features if f != session.target_column]
            feature_list = ', '.join([f"'{f}'" for f in current_features])
            
            prompt = f"""Generate Python code to answer the user's analytical query about the dataset.

CONTEXT:
- DataFrame variable name: df
- Target column: '{session.target_column}'
- Available features: [{feature_list}]
- User query: "{query}"

REQUIREMENTS:
1. Import necessary libraries (pandas, numpy, etc.)
2. Use 'df' as the dataframe variable name
3. Generate code that answers their specific question
4. Store the result in a variable called 'result'
5. The result should be a formatted string or simple data structure
6. Handle missing values appropriately
7. Keep the output concise and readable

EXAMPLES:
For "top 10 features with highest correlation with target":
```python
import pandas as pd
import numpy as np

# Calculate correlations with target
feature_cols = [{feature_list}]
correlations = {{}}
for col in feature_cols:
    corr = df[col].corr(df['{session.target_column}'])
    correlations[col] = abs(corr) if not pd.isna(corr) else 0

# Sort by absolute correlation
sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
top_10 = sorted_corr[:10]

# Format result
result = "Top 10 features by correlation with {session.target_column}:\\n"
for i, (feature, corr) in enumerate(top_10, 1):
    result += f"{{i}}. {{feature}}: {{corr:.4f}}\\n"
```

For "give me top 10 SHAP features" or "show me SHAP values":
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# Prepare data
feature_cols = [{feature_list}]
X = df[feature_cols].fillna(0)
y = df['{session.target_column}']

# Train Random Forest model
if y.nunique() <= 10:  # Classification
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:  # Regression
    model = RandomForestRegressor(n_estimators=100, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)

# Get feature importance (proxy for SHAP values)
importance_scores = model.feature_importances_
shap_scores = {{feature: importance for feature, importance in zip(feature_cols, importance_scores)}}

# Sort by importance and get top N
sorted_features = sorted(shap_scores.items(), key=lambda x: x[1], reverse=True)
top_features = sorted_features[:10]  # Adjust number based on query

# Format result
result = "Top 10 SHAP Features (using RandomForest importance):\\n"
for i, (feature, score) in enumerate(top_features, 1):
    result += f"{{i:2d}}. {{feature}}: {{score:.4f}}\\n"
```

Generate COMPLETE, EXECUTABLE Python code for: "{query}"
"""
            
            max_retries = 3
            fallback_model = "gpt-3.5-turbo"  # Fallback model for different approach
            logger.info(f"🔄 STARTING CODE GENERATION LOOP | User: {session.user_id} | Max retries: {max_retries}")
            
            for attempt in range(max_retries):
                # Use fallback model for final attempt if primary model keeps failing
                current_model = session.model_name
                if attempt == max_retries - 1 and session.model_name != fallback_model:
                    current_model = fallback_model
                    logger.info(f"🔄 USING FALLBACK MODEL | User: {session.user_id} | Attempt: {attempt + 1} | Model: {current_model}")
                    llm = LLMManager.get_llm(current_model)
                
                logger.info(f"🤖 CODE GENERATION ATTEMPT {attempt + 1}/{max_retries} | User: {session.user_id} | Model: {current_model}")
                try:
                    response = llm.invoke([HumanMessage(content=prompt)])
                    logger.info(f"📝 LLM RESPONSE RECEIVED | User: {session.user_id} | Model: {current_model} | Length: {len(response.content)} chars")
                    
                    # Extract code from response
                    code = self._extract_code_from_response(response.content)
                    
                    if not code:
                        logger.error(f"❌ NO CODE EXTRACTED | User: {session.user_id} | Attempt: {attempt + 1}")
                        raise ValueError("No code found in LLM response")
                    
                    logger.info(f"✅ CODE EXTRACTED | User: {session.user_id} | Attempt: {attempt + 1} | Code length: {len(code)} chars")
                    
                    # Execute the code
                    logger.info(f"⚡ EXECUTING CODE | User: {session.user_id} | Attempt: {attempt + 1}")
                    exec_result = self._execute_query_code(session, code)
                    
                    if exec_result.get("success"):
                        result = exec_result.get("result", "Analysis completed successfully")
                        logger.info(f"🎉 CODE EXECUTION SUCCESS | User: {session.user_id} | Attempt: {attempt + 1}")
                        say(f"📊 **Results:**\n{result}")
                        return
                    else:
                        error_msg = exec_result.get("error", "Unknown execution error")
                        logger.warning(f"⚠️ CODE EXECUTION FAILED | User: {session.user_id} | Attempt: {attempt + 1} | Error: {error_msg}")
                        if attempt < max_retries - 1:
                            # Update prompt with error for next attempt
                            logger.info(f"🔄 RETRYING WITH ERROR FEEDBACK | User: {session.user_id} | Attempt: {attempt + 1}")
                            prompt += f"\n\nPREVIOUS ATTEMPT FAILED WITH ERROR: {error_msg}\nPlease fix the code and avoid this error."
                        else:
                            logger.error(f"💥 ALL ATTEMPTS FAILED | User: {session.user_id} | Final error: {error_msg}")
                            say(f"🔧 **Analysis Coming Soon**: This type of analysis is temporarily unavailable. Please try other analyses.")
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        # Update prompt with error for next attempt
                        prompt += f"\n\nPREVIOUS ATTEMPT FAILED WITH ERROR: {str(e)}\nPlease fix the code and avoid this error."
                    else:
                        say(f"🔧 **Query Coming Soon**: This type of query is temporarily unavailable. Please try other questions.")
            
        except Exception as e:
            say(f"🔧 **Query Temporarily Unavailable**: Please try other questions or analyses.")
    
    def _execute_query_code(self, session: UserSession, code: str) -> Dict[str, Any]:
        """Execute query code safely and extract result"""
        try:
            # Create execution environment
            exec_globals = {
                'df': session.current_df.copy(),
                'pd': pd,
                'np': np,
                'session': session,
                '__builtins__': __builtins__
            }
            
            # Add common imports to globals
            try:
                import sklearn
                import scipy
                exec_globals['sklearn'] = sklearn
                exec_globals['scipy'] = scipy
            except ImportError:
                pass
            
            exec_locals = {}
            
            # Execute the code
            exec(code, exec_globals, exec_locals)
            
            # Extract result
            result = exec_locals.get('result', 'Code executed successfully but no result variable found')
            
            return {
                "success": True,
                "result": str(result)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_detailed_summary(self, session: UserSession, say):
        """Generate detailed pipeline summary using LLM"""
        try:
            llm = LLMManager.get_llm(session.model_name)
            
            # Prepare analysis chain details
            analysis_details = []
            total_features_removed = 0
            
            for i, step in enumerate(session.analysis_chain):
                features_removed = step.features_before - step.features_after
                total_features_removed += features_removed
                
                # Extract key parameters
                key_params = []
                if step.parameters:
                    if 'threshold' in step.parameters:
                        key_params.append(f"threshold={step.parameters['threshold']}")
                    if 'datetime_column' in step.parameters:
                        key_params.append(f"datetime_col={step.parameters['datetime_column']}")
                    if 'oot_month' in step.parameters:
                        key_params.append(f"oot_month={step.parameters['oot_month']}")
                
                param_str = f" ({', '.join(key_params)})" if key_params else ""
                
                detail = f"Step {i+1}: {step.type}{param_str} | {step.features_before} → {step.features_after} features | Removed: {features_removed}"
                analysis_details.append(detail)
            
            # Current state
            original_features = session.analysis_chain[0].features_before if session.analysis_chain else len(session.current_features)
            current_features = len(session.current_features)
            
            prompt = f"""Generate a comprehensive, informative analysis pipeline summary.

PIPELINE DETAILS:
Original features: {original_features}
Current features: {current_features}
Total features removed: {total_features_removed}
Target column: {session.target_column}

ANALYSIS STEPS PERFORMED:
{chr(10).join(analysis_details) if analysis_details else 'No analyses performed yet'}

REQUIREMENTS:
1. Create an executive summary of the feature selection pipeline
2. Highlight key statistics (original → current features, % reduction)
3. Explain what each analysis step accomplished 
4. Mention specific thresholds and parameters used
5. Assess the overall feature reduction effectiveness
6. Keep it informative but concise (max 10-12 lines)
7. Use emojis and formatting for clarity

Generate a professional analysis summary:"""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            say(f"📊 **Pipeline Summary**\n\n{response.content}")
            
        except Exception as e:
            # Fallback to simple summary
            pipeline = " → ".join([step.type for step in session.analysis_chain]) if session.analysis_chain else "No analyses performed"
            original_count = session.analysis_chain[0].features_before if session.analysis_chain else len(session.current_features)
            
            say(f"""📊 **Pipeline Summary**

🔄 **Analysis Chain:** {pipeline}
📈 **Feature Reduction:** {original_count} → {len(session.current_features)} features
📉 **Total Removed:** {original_count - len(session.current_features)} features
🎯 **Target:** {session.target_column}
📋 **Steps Completed:** {len(session.analysis_chain)}""")
                
    def generate_final_summary(self, session: UserSession, say):
        """Generate detailed waterfall final analysis summary"""
        if not session.analysis_chain:
            say("📋 **No analyses performed yet.**")
            return
        
        # Build waterfall breakdown
        waterfall_steps = []
        original_features = session.analysis_chain[0].features_before
        total_removed = 0
        
        for step in session.analysis_chain:
            features_removed = step.features_before - step.features_after
            total_removed += features_removed
            
            # Extract key parameters for display
            step_details = []
            if step.parameters:
                if 'threshold' in step.parameters:
                    threshold = step.parameters['threshold']
                    step_details.append(f"cutoff={threshold}")
                if 'datetime_column' in step.parameters:
                    step_details.append(f"datetime={step.parameters['datetime_column']}")
                if 'oot_month' in step.parameters:
                    step_details.append(f"oot={step.parameters['oot_month']}")
            
            step_params = f" ({', '.join(step_details)})" if step_details else ""
            
            if step.type == "data_cleaning":
                waterfall_steps.append(f"🧹 **Data Cleaning**: {step.features_before} → {step.features_after} features (removed {features_removed} single-value/object columns)")
            elif step.type == "csi_analysis":
                waterfall_steps.append(f"📅 **CSI Analysis{step_params}**: {step.features_before} → {step.features_after} features (removed {features_removed} unstable features)")
            elif step.type == "iv_analysis":
                waterfall_steps.append(f"📊 **IV Analysis{step_params}**: {step.features_before} → {step.features_after} features (removed {features_removed} low-predictive features)")
            elif step.type == "correlation_analysis":
                waterfall_steps.append(f"🔗 **Correlation Analysis{step_params}**: {step.features_before} → {step.features_after} features (removed {features_removed} highly correlated features)")
            elif step.type == "vif_analysis":
                waterfall_steps.append(f"📈 **VIF Analysis{step_params}**: {step.features_before} → {step.features_after} features (removed {features_removed} multicollinear features)")
            elif step.type == "shap_analysis":
                waterfall_steps.append(f"🎯 **SHAP Analysis{step_params}**: {step.features_before} → {step.features_after} features (removed {features_removed} low-importance features)")
            else:
                waterfall_steps.append(f"⚙️ **{step.type.title()}{step_params}**: {step.features_before} → {step.features_after} features (removed {features_removed} features)")
        
        # Calculate reduction percentage
        reduction_pct = (total_removed / original_features * 100) if original_features > 0 else 0
        
        newline = "\n"
        summary = f"""📋 **Feature Selection Analysis Complete**

📊 **Final Results:**
• Started with: {original_features} features
• Ended with: {len(session.current_features)} features  
• Total removed: {total_removed} features ({reduction_pct:.1f}% reduction)
• Target column: {session.target_column}

🏗️ **Waterfall Breakdown:**
{newline.join(waterfall_steps)}

✅ **Analysis complete!** Your optimized dataset is ready for modeling."""
        
        say(summary)
        session.phase = "completed"
    
    def handle_app_mention(self, body, say, logger):
        """Handle app mentions"""
        try:
            user_id = body["event"]["user"]
            text = body["event"]["text"]
            
            # Debug: Show active sessions
            logger.info(f"🔍 DEBUG: Active sessions: {list(self.users.keys())}")
            logger.info(f"🔍 DEBUG: Current user: {user_id}")
            logger.info(f"🔍 DEBUG: Total sessions: {len(self.users)}")
            
            # Show details of each session
            for session_user_id, session in self.users.items():
                logger.info(f"🔍 DEBUG: Session for {session_user_id}: file={session.file_name}, phase={session.phase}")
            
            # Debug: Show original text
            logger.info(f"🔍 DEBUG: Original text: '{text}'")
            
            # Remove bot mention from text
            import re
            cleaned_text = re.sub(r'<@[^>]+>', '', text).strip()
            
            # Debug: Show cleaned text
            logger.info(f"🔍 DEBUG: Cleaned text: '{cleaned_text}'")
            
            if not cleaned_text:
                if user_id not in self.users:
                    # No session yet, user should upload file first
                    logger.info(f"🔍 DEBUG: User {user_id} not found in sessions")
                    return
                else:
                    session = self.users[user_id]
                    if session.phase == "need_target":
                        say("🎯 Please specify your target column first.")
                    elif session.phase == "waiting_input":
                        if session.last_menu:
                            say(session.last_menu)
                        else:
                            menu = MenuGenerator.generate_main_menu(session)
                            say(menu)
                    else:
                        say("👋 How can I help? Upload a CSV file to start analysis.")
                return
            
            # Process as regular message
            fake_message = {"user": user_id, "text": cleaned_text}
            thread_ts = body["event"].get("thread_ts") or body["event"].get("ts")
            self.handle_message(fake_message, say, thread_ts)
            
        except Exception as e:
            logger.exception(f"Error handling app mention: {e}")
            say("❌ Error processing your mention. Please try again.")
    
    def run(self):
        """Start the bot"""
        handler = SocketModeHandler(self.app, os.environ["SLACK_APP_TOKEN"])
        print_to_log("🚀 New Agentic Feature Selection Bot Started!")
        print_to_log("📡 Socket Mode activated - Bot is ready!")
        handler.start()

if __name__ == "__main__":
    # Check environment variables
    required_vars = ["SLACK_BOT_TOKEN_FS", "SLACK_APP_TOKEN"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print_to_log(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    # Start the new bot
    bot = AgenticFeatureSelectionBot()
    bot.run() 