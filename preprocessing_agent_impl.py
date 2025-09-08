#!/usr/bin/env python3
"""
Sequential Preprocessing Agent - Part 1: Core State Management
Educational preprocessing agent with phase-by-phase workflow
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import json
import os
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import tempfile
import time
import warnings
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Load environment variables
load_dotenv()

print(f"🔧 DEBUG: Environment loaded - DEFAULT_MODEL: {os.getenv('DEFAULT_MODEL', 'Not set')}")
print(f"🔧 DEBUG: OPENAI_API_KEY available: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")

class PreprocessingPhase:
    """Enum-like class for preprocessing phases"""
    OVERVIEW = "overview"
    OUTLIERS = "outliers" 
    MISSING_VALUES = "missing_values"
    ENCODING = "encoding"
    TRANSFORMATIONS = "transformations"
    SCALING = "scaling"
    COMPLETION = "completion"

class SequentialState(BaseModel):
    """Enhanced state for sequential preprocessing workflow"""
    # Data management
    df: Optional[pd.DataFrame] = None
    df_path: str
    target_column: str
    model_name: str = os.environ.get("DEFAULT_MODEL", "gpt-4o")
    
    # Phase management
    current_phase: str = PreprocessingPhase.OVERVIEW
    completed_phases: List[str] = Field(default_factory=list)
    phase_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis results
    column_analysis: Dict[str, Dict] = Field(default_factory=dict)
    outlier_analysis: Dict[str, Dict] = Field(default_factory=dict)
    missing_analysis: Dict[str, Dict] = Field(default_factory=dict)
    encoding_analysis: Dict[str, Dict] = Field(default_factory=dict)
    transformation_analysis: Dict[str, Dict] = Field(default_factory=dict)
    
    # User interactions
    user_messages: List[str] = Field(default_factory=list)
    user_overrides: Dict[str, Dict] = Field(default_factory=dict)
    phase_approved: bool = False
    is_query: bool = False
    query_response: Optional[str] = None
    
    # Configuration
    missing_threshold: float = 50.0
    outlier_threshold: float = 10.0
    high_cardinality_threshold: int = 50
    onehot_top_categories: int = 10
    
    # Progress tracking
    total_columns: int = 0
    current_step: str = ""
    suggestions_enabled: bool = True
    
    class Config:
        arbitrary_types_allowed = True

def get_llm_from_state(state: SequentialState):
    """Get LLM instance based on state configuration"""
    try:
        # Get default model from environment
        default_model = os.getenv("DEFAULT_MODEL", "gpt-4o")
        print(f"🔧 DEBUG: Using model: {default_model}")
        
        # Handle None state
        if state is None:
            print("🔧 DEBUG: State is None, using default LLM configuration")
            if default_model.startswith("gpt-"):
                openai_key = os.getenv("OPENAI_API_KEY")
                print(f"🔧 DEBUG: OpenAI key available: {'Yes' if openai_key else 'No'}")
                return ChatOpenAI(
                    model=default_model,
                    temperature=0,
                    openai_api_key=openai_key
                )
            else:
                return ChatOllama(
                    model=default_model,
                    temperature=0
                )
        
        # Handle missing model_name
        if not hasattr(state, 'model_name') or state.model_name is None:
            print("🔧 DEBUG: State has no model_name, using default")
            if default_model.startswith("gpt-"):
                openai_key = os.getenv("OPENAI_API_KEY")
                print(f"🔧 DEBUG: OpenAI key available: {'Yes' if openai_key else 'No'}")
                return ChatOpenAI(
                    model=default_model,
                    temperature=0,
                    openai_api_key=openai_key
                )
            else:
                return ChatOllama(
                    model=default_model,
                    temperature=0
                )
        
        # Check if it's an OpenAI model (starts with gpt-)
        if state.model_name.startswith("gpt-"):
            openai_key = os.getenv("OPENAI_API_KEY")
            print(f"🔧 DEBUG: Using ChatOpenAI with key: {'Available' if openai_key else 'Missing'}")
            return ChatOpenAI(
                model=state.model_name,
                temperature=0,
                openai_api_key=openai_key
            )
        else:
            print(f"🔧 DEBUG: Using ChatOllama with model: {state.model_name}")
            return ChatOllama(
                model=state.model_name,
                temperature=0
            )
    except Exception as e:
        print(f"❌ Error creating LLM: {e}")
        # Fallback to default model
        default_model = os.getenv("DEFAULT_MODEL", "gpt-4o")
        print(f"🔧 DEBUG: Fallback to default model: {default_model}")
        if default_model.startswith("gpt-"):
            openai_key = os.getenv("OPENAI_API_KEY")
            return ChatOpenAI(model=default_model, temperature=0, openai_api_key=openai_key)
        else:
            return ChatOllama(model=default_model, temperature=0)

# Part 2: Enhanced Statistical Analysis Functions

def analyze_column_comprehensive(series: pd.Series, target: pd.Series = None, column_name: str = None) -> Dict[str, Any]:
    """
    Comprehensive column analysis for better preprocessing decisions
    All decisions will be made by LLM based on these statistics
    """
    analysis = {}
    
    # Basic statistics
    analysis['dtype'] = str(series.dtype)
    analysis['total_count'] = len(series)
    analysis['missing_count'] = series.isnull().sum()
    analysis['missing_percentage'] = (series.isnull().sum() / len(series)) * 100
    analysis['unique_count'] = series.nunique()
    analysis['unique_ratio'] = series.nunique() / len(series)
    analysis['sample_values'] = series.dropna().unique()[:5].tolist()
    
    # For numeric columns
    if pd.api.types.is_numeric_dtype(series) and series.dtype != 'bool':
        clean_series = series.dropna()
        if len(clean_series) > 0:
            # Distribution analysis
            analysis['mean'] = float(clean_series.mean())
            analysis['median'] = float(clean_series.median())
            analysis['std'] = float(clean_series.std())
            
            # Suppress harmless precision loss warnings for nearly identical data
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Precision loss occurred in moment calculation.*")
                analysis['skewness'] = float(stats.skew(clean_series))
                analysis['kurtosis'] = float(stats.kurtosis(clean_series))
            
            analysis['coefficient_of_variation'] = float(clean_series.std() / clean_series.mean()) if clean_series.mean() != 0 else float('inf')
            
            # Percentiles for outlier analysis
            analysis['percentile_1'] = float(clean_series.quantile(0.01))
            analysis['percentile_5'] = float(clean_series.quantile(0.05))
            analysis['percentile_95'] = float(clean_series.quantile(0.95))
            analysis['percentile_99'] = float(clean_series.quantile(0.99))
            analysis['q1'] = float(clean_series.quantile(0.25))
            analysis['q3'] = float(clean_series.quantile(0.75))
            analysis['iqr'] = analysis['q3'] - analysis['q1']
            
            # Outlier detection (multiple methods)
            if len(clean_series) > 4:
                # IQR method
                iqr_lower = analysis['q1'] - 1.5 * analysis['iqr']
                iqr_upper = analysis['q3'] + 1.5 * analysis['iqr']
                # Fix: Use np.logical_or instead of | for boolean arrays
                iqr_outliers = np.logical_or(clean_series < iqr_lower, clean_series > iqr_upper).sum()
                analysis['outliers_iqr_count'] = int(iqr_outliers)
                analysis['outliers_iqr_percentage'] = float((iqr_outliers / len(clean_series)) * 100)
                
                # Z-score method
                z_scores = np.abs(stats.zscore(clean_series))
                zscore_outliers = (z_scores > 3).sum()
                analysis['outliers_zscore_count'] = int(zscore_outliers)
                analysis['outliers_zscore_percentage'] = float((zscore_outliers / len(clean_series)) * 100)
                
                # Extreme outliers (very far from normal range)
                extreme_outliers = (z_scores > 5).sum()
                analysis['extreme_outliers_count'] = int(extreme_outliers)
                analysis['extreme_outliers_percentage'] = float((extreme_outliers / len(clean_series)) * 100)
            
            # Normality test
            try:
                if len(clean_series) > 8:
                    _, p_normal = stats.normaltest(clean_series)
                    analysis['normality_p_value'] = float(p_normal)
                    analysis['is_likely_normal'] = p_normal > 0.05
                else:
                    analysis['normality_p_value'] = None
                    analysis['is_likely_normal'] = False
            except:
                analysis['normality_p_value'] = None
                analysis['is_likely_normal'] = False
    
    # For boolean columns
    elif series.dtype == 'bool':
        # Boolean-specific analysis
        analysis['true_count'] = int(series.sum())
        analysis['false_count'] = int((~series).sum())
        analysis['true_percentage'] = float((series.sum() / len(series)) * 100)
        analysis['false_percentage'] = float(((~series).sum() / len(series)) * 100)
        analysis['is_balanced'] = abs(analysis['true_percentage'] - 50) < 10  # Within 10% of 50/50
    
    # For categorical columns
    if series.dtype in ['object', 'category']:
        value_counts = series.value_counts()
        analysis['cardinality'] = len(value_counts)
        if len(value_counts) > 0:
            analysis['most_frequent_value'] = str(value_counts.index[0])
            analysis['most_frequent_count'] = int(value_counts.iloc[0])
            analysis['most_frequent_percentage'] = float((value_counts.iloc[0] / len(series)) * 100)
            
            # Cardinality analysis for encoding decisions
            analysis['top_5_categories'] = value_counts.head(5).to_dict()
            analysis['category_distribution'] = {
                'low_cardinality': analysis['cardinality'] <= 5,
                'medium_cardinality': 5 < analysis['cardinality'] <= 20,
                'high_cardinality': analysis['cardinality'] > 20,
                'very_high_cardinality': analysis['cardinality'] > 50
            }
    
    # Target correlation analysis
    if target is not None:
        try:
            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target) and series.dtype != 'bool':
                # Numeric-numeric correlation
                combined = pd.concat([series, target], axis=1).dropna()
                if len(combined) > 1:
                    correlation = combined.corr().iloc[0, 1]
                    analysis['target_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
                    analysis['target_correlation_strength'] = 'strong' if abs(analysis['target_correlation']) > 0.7 else 'moderate' if abs(analysis['target_correlation']) > 0.3 else 'weak'
            elif series.dtype == 'bool' and pd.api.types.is_numeric_dtype(target):
                # Boolean-numeric correlation (point-biserial correlation)
                combined = pd.concat([series, target], axis=1).dropna()
                if len(combined) > 1:
                    # Point-biserial correlation for boolean-numeric
                    true_values = combined[target.name][combined[series.name] == True]
                    false_values = combined[target.name][combined[series.name] == False]
                    if len(true_values) > 0 and len(false_values) > 0:
                        correlation = (true_values.mean() - false_values.mean()) / target.std() * np.sqrt(len(true_values) * len(false_values) / len(combined))
                        analysis['target_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
                        analysis['target_correlation_strength'] = 'strong' if abs(analysis['target_correlation']) > 0.7 else 'moderate' if abs(analysis['target_correlation']) > 0.3 else 'weak'
            elif series.dtype in ['object', 'category'] and pd.api.types.is_numeric_dtype(target):
                # Categorical-numeric relationship (ANOVA)
                combined = pd.concat([series, target], axis=1).dropna()
                if len(combined) > 0:
                    groups = [group[target.name].values for name, group in combined.groupby(series.name) if len(group) > 0]
                    if len(groups) > 1:
                        f_stat, p_value = stats.f_oneway(*groups)
                        analysis['anova_f_statistic'] = float(f_stat)
                        analysis['anova_p_value'] = float(p_value)
                        analysis['significant_relationship'] = p_value < 0.05
        except Exception as e:
            print(f"Error in target correlation analysis for {column_name}: {e}")
    
    return analysis

def detect_patterns_llm_ready(series: pd.Series, column_name: str = None) -> Dict[str, Any]:
    """
    Detect patterns in data for LLM analysis
    No hardcoded rules - just statistical evidence for LLM to interpret
    """
    patterns = {}
    
    if series.dtype == 'object':
        str_series = series.astype(str).str.strip()
        total_non_null = len(str_series.dropna())
        
        if total_non_null > 0:
            # Pattern confidence scores (for LLM to interpret)
            patterns['contains_at_symbol'] = (str_series.str.contains('@', na=False).sum() / total_non_null)
            patterns['contains_digits'] = (str_series.str.contains(r'\d', na=False).sum() / total_non_null)
            patterns['contains_spaces'] = (str_series.str.contains(r'\s', na=False).sum() / total_non_null)
            patterns['contains_special_chars'] = (str_series.str.contains(r'[^a-zA-Z0-9\s]', na=False).sum() / total_non_null)
            patterns['all_uppercase'] = (str_series.str.isupper().sum() / total_non_null)
            patterns['all_lowercase'] = (str_series.str.islower().sum() / total_non_null)
            # Fix: Use np.logical_and instead of & for boolean arrays
            patterns['mixed_case'] = (np.logical_and(~str_series.str.isupper(), ~str_series.str.islower()).sum() / total_non_null)
            
            # Length analysis
            lengths = str_series.str.len()
            patterns['avg_length'] = float(lengths.mean())
            patterns['length_std'] = float(lengths.std())
            patterns['min_length'] = int(lengths.min())
            patterns['max_length'] = int(lengths.max())
            patterns['consistent_length'] = patterns['length_std'] < 2.0  # Low variance in length
    
    return patterns

# Part 3: LLM-driven Phase Analysis Functions

def analyze_outliers_with_llm(state: SequentialState, progress_callback=None) -> Dict[str, Any]:
    """
    LLM-driven outlier analysis for all numeric columns
    Uses chunked processing for large datasets to avoid token limits
    """
    llm = get_llm_from_state(state)
    df = state.df
    target = df[state.target_column] if state.target_column in df.columns else None
    
    # Determine if we need chunked processing based on dataset size
    total_columns = len(df.columns)
    chunk_size = 10  # Optimized for open-source models
    
    if total_columns > 100:
        print(f"📊 Large dataset detected ({total_columns} columns). Using chunked processing...")
        return analyze_outliers_chunked(state, chunk_size, progress_callback)
    else:
        return analyze_outliers_single_batch(state)

def analyze_outliers_single_batch(state: SequentialState) -> Dict[str, Any]:
    """Original single-batch outlier analysis for smaller datasets"""
    llm = get_llm_from_state(state)
    df = state.df
    target = df[state.target_column] if state.target_column in df.columns else None
    
    outlier_results = {}
    numeric_columns = []
    
    # Analyze each numeric column
    for col in df.columns:
        if col != state.target_column and pd.api.types.is_numeric_dtype(df[col]):
            analysis = analyze_column_comprehensive(df[col], target, col)
            patterns = detect_patterns_llm_ready(df[col], col)
            
            # Only include columns with actual outliers
            if analysis.get('outliers_iqr_percentage', 0) > 0:
                numeric_columns.append({
                    'column': col,
                    'analysis': analysis,
                    'patterns': patterns
                })
    
    if not numeric_columns:
        return {'outlier_columns': [], 'llm_recommendations': {}}
    
    # LLM prompt for outlier analysis
    prompt = f"""
You are an expert data scientist analyzing outliers in a dataset. Based on the statistical analysis provided, recommend the best outlier treatment strategy for each column.

TARGET COLUMN: {state.target_column}
TOTAL COLUMNS WITH OUTLIERS: {len(numeric_columns)}

For each column, consider:
1. The severity of outliers (percentage and methods)
2. Distribution shape (skewness, normality)
3. Relationship with target variable
4. Data type and domain context

COLUMNS TO ANALYZE:
"""
    
    for col_info in numeric_columns:
        col = col_info['column']
        analysis = col_info['analysis']
        prompt += f"""

Column: {col}
- Data type: {analysis['dtype']}
- Missing: {analysis['missing_percentage']:.1f}%
- Distribution: mean={analysis.get('mean', 0):.2f}, median={analysis.get('median', 0):.2f}, skewness={analysis.get('skewness', 0):.2f}
- Outliers (IQR): {analysis.get('outliers_iqr_percentage', 0):.1f}% ({analysis.get('outliers_iqr_count', 0)} values)
- Outliers (Z-score): {analysis.get('outliers_zscore_percentage', 0):.1f}% ({analysis.get('outliers_zscore_count', 0)} values)
- Extreme outliers: {analysis.get('extreme_outliers_percentage', 0):.1f}%
- Target correlation: {analysis.get('target_correlation', 0):.3f}
- Normality: {analysis.get('is_likely_normal', False)}
- Sample values: {analysis['sample_values']}
"""
    
    prompt += f"""

OUTLIER TREATMENT OPTIONS:
- "keep": Keep all outliers (if they are legitimate values)
- "winsorize": Cap at 1st/99th percentiles (for income, amounts, skewed data, bounded data)
- "remove": Remove outliers (if they are measurement errors)
- "mark_missing": Convert outliers to NaN for later imputation

Return JSON with treatment recommendations:
{{
  "column_name": {{
    "treatment": "winsorize/remove/keep/mark_missing",
    "reasoning": "Brief explanation of why this treatment is best",
    "severity": "mild/moderate/severe",
    "impact_on_missing": "percentage increase in missing values if applicable"
  }}
}}

Consider domain context:
- Financial amounts (income, credit_score) → winsorize
- Bounded values (age, percentages) → winsorize
- Measurements with errors → remove
- Legitimate extreme values → keep
- High skewness → winsorize
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            recommendations = json.loads(json_match.group())
        else:
            recommendations = {}
        
        outlier_results = {
            'outlier_columns': [col_info['column'] for col_info in numeric_columns],
            'llm_recommendations': recommendations,
            'analysis_details': {col_info['column']: col_info['analysis'] for col_info in numeric_columns}
        }
        
    except Exception as e:
        print(f"Error in LLM outlier analysis: {e}")
        # Fallback recommendations
        recommendations = {}
        for col_info in numeric_columns:
            col = col_info['column']
            analysis = col_info['analysis']
            
            # Simple rule-based fallback
            if analysis.get('outliers_iqr_percentage', 0) > 20:
                treatment = "winsorize"  # Changed from "remove"
            elif analysis.get('skewness', 0) > 2.0:
                treatment = "winsorize"
            elif analysis.get('outliers_iqr_percentage', 0) > 10:
                treatment = "winsorize"  # Changed from "clip" to "winsorize"
            else:
                treatment = "keep"
                
            recommendations[col] = {
                "treatment": treatment,
                "reasoning": f"Fallback rule based on outlier percentage and skewness",
                "severity": "medium",
                "impact_on_missing": "0%"
            }
        
        outlier_results = {
            'outlier_columns': [col_info['column'] for col_info in numeric_columns],
            'llm_recommendations': recommendations,
            'analysis_details': {col_info['column']: col_info['analysis'] for col_info in numeric_columns}
        }
    
    return outlier_results

def analyze_outliers_chunked(state: SequentialState, chunk_size: int = 10, progress_callback=None) -> Dict[str, Any]:
    """
    Chunked outlier analysis for large datasets to avoid token limits
    Processes columns in batches and combines results
    """
    import time
    
    df = state.df
    target = df[state.target_column] if state.target_column in df.columns else None
    
    # Get all numeric columns with outliers
    numeric_columns_with_outliers = []
    
    print("🔍 Scanning for columns with outliers...")
    for col in df.columns:
        if col != state.target_column and pd.api.types.is_numeric_dtype(df[col]):
            analysis = analyze_column_comprehensive(df[col], target, col)
            
            # Only include columns with actual outliers
            if analysis.get('outliers_iqr_percentage', 0) > 0:
                numeric_columns_with_outliers.append({
                    'column': col,
                    'analysis': analysis,
                    'patterns': detect_patterns_llm_ready(df[col], col)
                })
    
    if not numeric_columns_with_outliers:
        return {'outlier_columns': [], 'llm_recommendations': {}}
    
    print(f"📊 Processing {len(numeric_columns_with_outliers)} columns with outliers in chunks of {chunk_size}")
    
    # Process in chunks
    all_recommendations = {}
    all_analysis_details = {}
    start_time = time.time()
    total_chunks = (len(numeric_columns_with_outliers) + chunk_size - 1) // chunk_size
    
    # Send initial loading message if callback provided
    if progress_callback:
        progress_callback("outliers", 0, total_chunks, 0, len(numeric_columns_with_outliers), start_time, None, is_initial=True)
    
    for i in range(0, len(numeric_columns_with_outliers), chunk_size):
        chunk = numeric_columns_with_outliers[i:i + chunk_size]
        chunk_num = (i // chunk_size) + 1
        
        # Send progress update if callback provided
        if progress_callback:
            current_columns = len(all_recommendations) + len(chunk)
            progress_callback("outliers", chunk_num, total_chunks, current_columns, len(numeric_columns_with_outliers), start_time, None, is_initial=False)
        
        print(f"📊 Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} columns)")
        
        try:
            chunk_recommendations = analyze_outlier_chunk(state, chunk)
            all_recommendations.update(chunk_recommendations)
            
            # Store analysis details
            for col_info in chunk:
                all_analysis_details[col_info['column']] = col_info['analysis']
                
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_num}: {e}")
            # Fallback for this chunk
            for col_info in chunk:
                col = col_info['column']
                analysis = col_info['analysis']
                
                # Simple rule-based fallback
                if analysis.get('outliers_iqr_percentage', 0) > 20:
                    treatment = "winsorize"  # Changed from "remove"
                elif analysis.get('skewness', 0) > 2.0:
                    treatment = "winsorize"
                elif analysis.get('outliers_iqr_percentage', 0) > 10:
                    treatment = "winsorize"  # Changed from "clip" to "winsorize"
                else:
                    treatment = "keep"
                    
                all_recommendations[col] = {
                    "treatment": treatment,
                    "reasoning": f"Fallback rule for chunk {chunk_num}",
                    "severity": "medium",
                    "impact_on_missing": "0%"
                }
                all_analysis_details[col] = analysis
    
    total_time = time.time() - start_time
    print(f"✅ Completed outlier analysis for {len(all_recommendations)} columns in {total_time:.2f}s")
    return {
        'outlier_columns': list(all_recommendations.keys()),
        'llm_recommendations': all_recommendations,
        'analysis_details': all_analysis_details
    }

def analyze_outlier_chunk(state: SequentialState, chunk: list) -> Dict[str, Any]:
    """Analyze a single chunk of columns for outliers"""
    llm = get_llm_from_state(state)
    df = state.df
    target = df[state.target_column] if state.target_column in df.columns else None
    
    # Create optimized prompt for this chunk
    prompt = f"""Analyze outliers for {len(chunk)} columns. Target: {state.target_column}

TREATMENT OPTIONS: winsorize (recommended), keep

COLUMNS:"""
    
    for col_info in chunk:
        col = col_info['column']
        analysis = col_info['analysis']
        prompt += f"""

{col}:
- Missing: {analysis['missing_percentage']:.1f}%
- Outliers (IQR): {analysis.get('outliers_iqr_percentage', 0):.1f}%
- Skewness: {analysis.get('skewness', 0):.2f}
- Target corr: {analysis.get('target_correlation', 0):.3f}
- Sample: {analysis['sample_values'][:3]}"""
    
    prompt += f"""

Return JSON: {{"column_name": {{"treatment": "winsorize/keep", "reasoning": "brief reason", "severity": "low/medium/high"}}}}"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        
        # Extract JSON from response
        import json
        import re
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            recommendations = json.loads(json_match.group())
        else:
            # Fallback parsing
            recommendations = {}
            lines = response.split('\n')
            for line in lines:
                if ':' in line and 'treatment' in line.lower():
                    parts = line.split(':')
                    if len(parts) >= 2:
                        col_name = parts[0].strip().strip('"{}')
                        treatment = parts[1].strip().strip('"{}')
                        if treatment in ['winsorize', 'keep']:
                            recommendations[col_name] = {
                                "treatment": treatment,
                                "reasoning": "Fallback recommendation",
                                "severity": "medium",
                                "impact_on_missing": "0%"
                            }
        
        return recommendations
        
    except Exception as e:
        print(f"❌ Error in outlier chunk analysis: {e}")
        # Fallback recommendations
        fallback_recommendations = {}
        for col_info in chunk:
            col = col_info['column']
            analysis = col_info['analysis']
            
            # Simple rule-based fallback
            if analysis.get('outliers_iqr_percentage', 0) > 20:
                treatment = "winsorize"  # Changed from "remove"
            elif analysis.get('skewness', 0) > 2.0:
                treatment = "winsorize"
            elif analysis.get('outliers_iqr_percentage', 0) > 10:
                treatment = "winsorize"  # Changed from "clip" to "winsorize"
            else:
                treatment = "keep"
                
            fallback_recommendations[col] = {
                "treatment": treatment,
                "reasoning": f"Fallback rule for {col}",
                "severity": "medium",
                "impact_on_missing": "0%"
            }
        
        return fallback_recommendations

def analyze_missing_values_with_llm(state: SequentialState, progress_callback=None) -> Dict[str, Any]:
    """
    LLM-driven missing value analysis using current data state
    Uses chunked processing for large datasets to avoid token limits
    """
    llm = get_llm_from_state(state)
    
    # Get current data state (after outliers treatment if completed)
    current_df = get_current_data_state(state)
    
    target = current_df[state.target_column] if state.target_column in current_df.columns else None
    
    # Determine if we need chunked processing based on dataset size
    total_columns = len(current_df.columns)
    chunk_size = 10  # Optimized for open-source models
    
    if total_columns > 100:
        print(f"📊 Large dataset detected ({total_columns} columns). Using chunked processing...")
        return analyze_missing_values_chunked(state, current_df, chunk_size, progress_callback)
    else:
        return analyze_missing_values_single_batch(state, current_df)

def analyze_missing_values_single_batch(state: SequentialState, current_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Original single-batch missing value analysis for smaller datasets
    """
    llm = get_llm_from_state(state)
    
    # Get target from current_df
    target = current_df[state.target_column] if state.target_column in current_df.columns else None
    
    missing_columns = []
    
    # Analyze columns with missing values
    for col in current_df.columns:
        if current_df[col].isnull().sum() > 0:
            analysis = analyze_column_comprehensive(current_df[col], target, col)
            patterns = detect_patterns_llm_ready(current_df[col], col)
            
            missing_columns.append({
                'column': col,
                'analysis': analysis,
                'patterns': patterns
            })
    
    if not missing_columns:
        return {'missing_columns': [], 'llm_recommendations': {}}
    
    # Check if outlier treatment affected missing percentages
    outlier_impact = ""
    if PreprocessingPhase.OUTLIERS in state.completed_phases:
        outlier_impact = "\nIMPACT OF OUTLIER TREATMENT:\n"
        for col, details in state.phase_results.get('outliers', {}).get('analysis_details', {}).items():
            if col in [mc['column'] for mc in missing_columns]:
                original_missing = details.get('original_missing_percentage', 0)
                current_missing = current_df[col].isnull().sum() / len(current_df) * 100
                if current_missing > original_missing:
                    outlier_impact += f"- {col}: Missing increased from {original_missing:.1f}% to {current_missing:.1f}% due to outlier treatment\n"
    
    prompt = f"""
You are an expert data scientist analyzing missing values. Recommend the best imputation strategy for each column based on statistical analysis.

TARGET COLUMN: {state.target_column}
TOTAL COLUMNS WITH MISSING VALUES: {len(missing_columns)}
{outlier_impact}

COLUMNS TO ANALYZE:
"""
    
    for col_info in missing_columns:
        col = col_info['column']
        analysis = col_info['analysis']
        patterns = col_info['patterns']
        
        prompt += f"""

Column: {col}
- Data type: {analysis['dtype']}
- Missing: {analysis['missing_percentage']:.1f}% ({analysis['missing_count']} values)
- Unique values: {analysis['unique_count']} (ratio: {analysis['unique_ratio']:.3f})
- Sample values: {analysis['sample_values']}
"""
        
        if pd.api.types.is_numeric_dtype(current_df[col]):
            prompt += f"""- Distribution: mean={analysis.get('mean', 0):.2f}, median={analysis.get('median', 0):.2f}, skewness={analysis.get('skewness', 0):.2f}
- Normality: {analysis.get('is_likely_normal', False)}
- Target correlation: {analysis.get('target_correlation', 0):.3f}"""
        else:
            prompt += f"""- Cardinality: {analysis.get('cardinality', 0)}
- Most frequent: {analysis.get('most_frequent_value', 'N/A')} ({analysis.get('most_frequent_percentage', 0):.1f}%)
- Pattern analysis: {patterns}"""
    
    prompt += f"""

IMPUTATION STRATEGIES:
- "mean": For normally distributed numeric data with low missing%
- "median": For skewed numeric data or data with outliers
- "mode": For categorical data or highly skewed numeric
- "forward_fill": For time series data
- "model_based": For high correlation with other features
- "constant": For specific business logic (e.g., 0 for amounts)
- "drop_column": If missing > 70% and low importance
- "keep_missing": If missingness itself is informative

Return JSON with imputation recommendations:
{{
  "column_name": {{
    "strategy": "mean/median/mode/model_based/constant/drop_column/keep_missing",
    "reasoning": "Statistical justification for this choice",
    "priority": "high/medium/low",
    "constant_value": "if strategy is constant, specify the value"
  }}
}}

Consider:
- Distribution shape for numeric imputation choice
- Cardinality for categorical imputation
- Target correlation for importance assessment
- Missing percentage for drop decisions
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            recommendations = json.loads(json_match.group())
        else:
            recommendations = {}
        
        missing_results = {
            'missing_columns': [col_info['column'] for col_info in missing_columns],
            'llm_recommendations': recommendations,
            'analysis_details': {col_info['column']: col_info['analysis'] for col_info in missing_columns}
        }
        
    except Exception as e:
        print(f"Error in LLM missing value analysis: {e}")
        # Fallback recommendations
        recommendations = {}
        for col_info in missing_columns:
            col = col_info['column']
            analysis = col_info['analysis']
            
            if analysis['missing_percentage'] > 70:
                strategy = "drop_column"
            elif analysis['dtype'] in ['object', 'category']:
                strategy = "mode"
            elif analysis.get('skewness', 0) > 1.0:
                strategy = "median"
            else:
                strategy = "mean"
                
            recommendations[col] = {
                "strategy": strategy,
                "reasoning": f"Fallback rule based on missing% and data type",
                "priority": "medium"
            }
        
        missing_results = {
            'missing_columns': [col_info['column'] for col_info in missing_columns],
            'llm_recommendations': recommendations,
            'analysis_details': {col_info['column']: col_info['analysis'] for col_info in missing_columns}
        }
    
    return missing_results

def analyze_missing_values_chunked(state: SequentialState, current_df: pd.DataFrame, chunk_size: int = 10, progress_callback=None) -> Dict[str, Any]:
    """
    Chunked missing value analysis for large datasets to avoid token limits
    Processes columns in batches and combines results
    """
    import time
    
    target = current_df[state.target_column] if state.target_column in current_df.columns else None
    
    # Get all columns with missing values
    missing_columns_list = []
    
    print("🔍 Scanning for columns with missing values...")
    for col in current_df.columns:
        if current_df[col].isnull().sum() > 0:
            analysis = analyze_column_comprehensive(current_df[col], target, col)
            patterns = detect_patterns_llm_ready(current_df[col], col)
            
            missing_columns_list.append({
                'column': col,
                'analysis': analysis,
                'patterns': patterns
            })
    
    if not missing_columns_list:
        return {'missing_columns': [], 'llm_recommendations': {}}
    
    print(f"📊 Processing {len(missing_columns_list)} columns with missing values in chunks of {chunk_size}")
    
    # Process in chunks
    all_recommendations = {}
    all_analysis_details = {}
    
    for i in range(0, len(missing_columns_list), chunk_size):
        chunk = missing_columns_list[i:i + chunk_size]
        chunk_num = (i // chunk_size) + 1
        total_chunks = (len(missing_columns_list) + chunk_size - 1) // chunk_size
        
        # Send progress update if callback provided
        if progress_callback:
            current_columns = len(all_recommendations) + len(chunk)
            progress_callback("missing_values", chunk_num, total_chunks, current_columns, len(missing_columns_list))
        
        print(f"📊 Processing missing values chunk {chunk_num}/{total_chunks} ({len(chunk)} columns)")
        
        try:
            chunk_recommendations = analyze_missing_values_chunk(state, chunk, current_df)
            all_recommendations.update(chunk_recommendations)
            
            # Store analysis details
            for col_info in chunk:
                all_analysis_details[col_info['column']] = col_info['analysis']
                
        except Exception as e:
            print(f"❌ Error processing missing values chunk {chunk_num}: {e}")
            # Fallback for this chunk
            for col_info in chunk:
                col = col_info['column']
                analysis = col_info['analysis']
                
                # Simple rule-based fallback
                if analysis['missing_percentage'] > 70:
                    strategy = "drop_column"
                elif analysis['dtype'] in ['object', 'category']:
                    strategy = "mode"
                elif analysis.get('skewness', 0) > 1.0:
                    strategy = "median"
                else:
                    strategy = "mean"
                    
                all_recommendations[col] = {
                    "strategy": strategy,
                    "reasoning": f"Fallback rule for missing values chunk {chunk_num}",
                    "priority": "medium"
                }
                all_analysis_details[col] = analysis
    
    print(f"✅ Completed missing values analysis for {len(all_recommendations)} columns")
    return {
        'missing_columns': list(all_recommendations.keys()),
        'llm_recommendations': all_recommendations,
        'analysis_details': all_analysis_details
    }

def analyze_missing_values_chunk(state: SequentialState, chunk: list, current_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze a single chunk of columns for missing values"""
    llm = get_llm_from_state(state)
    
    target = current_df[state.target_column] if state.target_column in current_df.columns else None
    
    # Create optimized prompt for this chunk
    prompt = f"""Analyze missing values for {len(chunk)} columns. Target: {state.target_column}

STRATEGIES: mean, median, mode, model_based, constant, drop_column, keep_missing

COLUMNS:"""
    
    for col_info in chunk:
        col = col_info['column']
        analysis = col_info['analysis']
        patterns = col_info['patterns']
        
        prompt += f"""

{col}:
- Missing: {analysis['missing_percentage']:.1f}% ({analysis['missing_count']} values)
- Type: {analysis['dtype']}
- Unique: {analysis['unique_count']} (ratio: {analysis['unique_ratio']:.3f})"""
        
        if pd.api.types.is_numeric_dtype(current_df[col]):
            prompt += f"""
- Mean: {analysis.get('mean', 0):.2f}, Median: {analysis.get('median', 0):.2f}
- Skewness: {analysis.get('skewness', 0):.2f}
- Target corr: {analysis.get('target_correlation', 0):.3f}"""
        else:
            prompt += f"""
- Cardinality: {analysis.get('cardinality', 0)}
- Most frequent: {analysis.get('most_frequent_value', 'N/A')} ({analysis.get('most_frequent_percentage', 0):.1f}%)"""
    
    prompt += f"""

Return JSON: {{"column_name": {{"strategy": "strategy_name", "reasoning": "brief reason", "priority": "high/medium/low"}}}}"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            recommendations = json.loads(json_match.group())
        else:
            recommendations = {}
        
        return recommendations
        
    except Exception as e:
        print(f"Error in missing values chunk analysis: {e}")
        # Return empty dict, will be handled by fallback in parent function
        return {}

def analyze_encoding_with_llm(state: SequentialState, progress_callback=None) -> Dict[str, Any]:
    """
    LLM-driven categorical encoding analysis using current data state
    Uses chunked processing for large datasets to avoid token limits
    """
    llm = get_llm_from_state(state)
    
    # Get current data state (after outliers and missing values treatment if completed)
    current_df = get_current_data_state(state)
    
    target = current_df[state.target_column] if state.target_column in current_df.columns else None
    
    # Determine if we need chunked processing based on dataset size
    total_columns = len(current_df.columns)
    chunk_size = 10  # Optimized for open-source models
    
    if total_columns > 100:
        print(f"📊 Large dataset detected ({total_columns} columns). Using chunked processing...")
        return analyze_encoding_chunked(state, current_df, chunk_size, progress_callback)
    else:
        return analyze_encoding_single_batch(state, current_df)

def analyze_encoding_single_batch(state: SequentialState, current_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Original single-batch categorical encoding analysis for smaller datasets
    """
    llm = get_llm_from_state(state)
    
    # Get categorical columns
    categorical_columns = []
    high_cardinality_columns = []
    
    for col in current_df.columns:
        if col != state.target_column and current_df[col].dtype == 'object':
            unique_count = current_df[col].nunique()
            categorical_columns.append(col)
            
            # Detect high cardinality (more than 50 unique values)
            if unique_count > 50:
                high_cardinality_columns.append({
                    'column': col,
                    'unique_count': unique_count,
                    'unique_ratio': unique_count / len(current_df)
                })
    
    if not categorical_columns:
        return {'categorical_columns': [], 'llm_recommendations': {}}
    
    # Create optimized prompt for encoding analysis
    prompt = f"""Analyze encoding for {len(categorical_columns)} categorical columns. Target: {state.target_column}

STRATEGIES: label_encoding, onehot_encoding, target_encoding, binary_encoding, drop_column

COLUMNS:"""
    
    for col in categorical_columns:
        unique_count = current_df[col].nunique()
        unique_ratio = unique_count / len(current_df)
        sample_values = current_df[col].dropna().head(3).tolist()
        
        prompt += f"""

{col}:
- Unique: {unique_count} ({unique_ratio:.1%})
- Sample: {sample_values}"""
    
    prompt += f"""

Return JSON: {{"column_name": {{"strategy": "strategy_name", "reasoning": "brief reason", "cardinality_level": "low/medium/high"}}}}"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        
        # Parse JSON response
        import json
        import re
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            recommendations = json.loads(json_match.group())
        else:
            # Fallback recommendations
            recommendations = {}
            for col in categorical_columns:
                unique_count = current_df[col].nunique()
                
                if unique_count > 1000:
                    strategy = "drop_column"
                    reasoning = "Extremely high cardinality"
                elif unique_count > 50:
                    strategy = "target_encoding"
                    reasoning = "High cardinality, use target encoding"
                elif unique_count > 10:
                    strategy = "onehot_encoding"
                    reasoning = "Medium cardinality, use one-hot encoding"
                else:
                    strategy = "label_encoding"
                    reasoning = "Low cardinality, use label encoding"
                
                recommendations[col] = {
                    "strategy": strategy,
                    "reasoning": reasoning,
                    "cardinality_level": "high" if unique_count > 50 else "medium" if unique_count > 10 else "low",
                    "target_relationship": "moderate"
                }
        
        return {
            'categorical_columns': categorical_columns,
            'high_cardinality_columns': [h['column'] for h in high_cardinality_columns],
            'llm_recommendations': recommendations
        }
        
    except Exception as e:
        print(f"❌ Error in encoding analysis: {e}")
        # Fallback recommendations
        fallback_recommendations = {}
        for col in categorical_columns:
            unique_count = current_df[col].nunique()
            
            if unique_count > 1000:
                strategy = "drop_column"
            elif unique_count > 50:
                strategy = "target_encoding"
            elif unique_count > 10:
                strategy = "onehot_encoding"
            else:
                strategy = "label_encoding"
            
            fallback_recommendations[col] = {
                "strategy": strategy,
                "reasoning": f"Fallback rule for {col}",
                "cardinality_level": "high" if unique_count > 50 else "medium" if unique_count > 10 else "low",
                "target_relationship": "moderate"
            }
        
        return {
            'categorical_columns': categorical_columns,
            'high_cardinality_columns': [h['column'] for h in high_cardinality_columns],
            'llm_recommendations': fallback_recommendations
        }

def analyze_transformations_with_llm(state: SequentialState, progress_callback=None) -> Dict[str, Any]:
    """
    LLM-driven transformation analysis using current data state
    Uses chunked processing for large datasets to avoid token limits
    """
    llm = get_llm_from_state(state)
    
    # Get current data state (after outliers, missing values, and encoding treatment if completed)
    current_df = get_current_data_state(state)
    
    target = current_df[state.target_column] if state.target_column in current_df.columns else None
    
    # Determine if we need chunked processing based on dataset size
    total_columns = len(current_df.columns)
    chunk_size = 10  # Optimized for open-source models
    
    if total_columns > 100:
        print(f"📊 Large dataset detected ({total_columns} columns). Using chunked processing...")
        return analyze_transformations_chunked(state, current_df, chunk_size, progress_callback)
    else:
        return analyze_transformations_single_batch(state, current_df)

def analyze_transformations_single_batch(state: SequentialState, current_df: pd.DataFrame) -> Dict[str, Any]:
    """Original single-batch transformation analysis for smaller datasets"""
    llm = get_llm_from_state(state)
    
    # Get target from current_df
    target = current_df[state.target_column] if state.target_column in current_df.columns else None
    
    numeric_columns = []
    
    # Analyze numeric columns for transformation needs
    for col in current_df.columns:
        if col != state.target_column and pd.api.types.is_numeric_dtype(current_df[col]):
            analysis = analyze_column_comprehensive(current_df[col], target, col)
            
            # Only consider columns that might benefit from transformation
            skewness = abs(analysis.get('skewness', 0))
            kurtosis = abs(analysis.get('kurtosis', 0))
            
            if skewness > 1.0 or kurtosis > 5.0 or not analysis.get('is_likely_normal', False):
                numeric_columns.append({
                    'column': col,
                    'analysis': analysis
                })
    
    if not numeric_columns:
        return {'transformation_columns': [], 'llm_recommendations': {}}
    
    # Create optimized prompt for transformation analysis
    prompt = f"""Analyze transformations for {len(numeric_columns)} numeric columns. Target: {state.target_column}

TRANSFORMATIONS: log, log1p, sqrt, box_cox, yeo_johnson, standardize, robust_scale, quantile, none

COLUMNS:"""
    
    for col_info in numeric_columns:
        col = col_info['column']
        analysis = col_info['analysis']
        
        prompt += f"""

{col}:
- Skewness: {analysis.get('skewness', 0):.2f}
- Kurtosis: {analysis.get('kurtosis', 0):.2f}
- Normal: {analysis.get('is_likely_normal', False)}
- Target corr: {analysis.get('target_correlation', 0):.3f}
- Range: {analysis.get('percentile_1', 0):.2f} to {analysis.get('percentile_99', 0):.2f}"""
    
    prompt += f"""

Return JSON: {{"column_name": {{"transformation": "transformation_name", "reasoning": "brief reason", "priority": "high/medium/low"}}}}"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            recommendations = json.loads(json_match.group())
        else:
            recommendations = {}
        
        transformation_results = {
            'transformation_columns': [col_info['column'] for col_info in numeric_columns],
            'llm_recommendations': recommendations,
            'analysis_details': {col_info['column']: col_info['analysis'] for col_info in numeric_columns}
        }
        
    except Exception as e:
        print(f"Error in LLM transformation analysis: {e}")
        # Fallback transformation recommendations
        recommendations = {}
        for col_info in numeric_columns:
            col = col_info['column']
            analysis = col_info['analysis']
            skewness = analysis.get('skewness', 0)
            
            if skewness > 2:
                transformation = "log1p"
            elif skewness > 1:
                transformation = "sqrt"
            elif skewness < -1:
                transformation = "square"
            else:
                transformation = "standardize"
                
            recommendations[col] = {
                "transformation": transformation,
                "reasoning": f"Fallback rule based on skewness {skewness:.2f}",
                "priority": "medium"
            }
        
        transformation_results = {
            'transformation_columns': [col_info['column'] for col_info in numeric_columns],
            'llm_recommendations': recommendations,
            'analysis_details': {col_info['column']: col_info['analysis'] for col_info in numeric_columns}
        }
    
    return transformation_results

def analyze_transformations_chunked(state: SequentialState, current_df: pd.DataFrame, chunk_size: int = 10, progress_callback=None) -> Dict[str, Any]:
    """
    Chunked transformation analysis for large datasets to avoid token limits
    Processes numeric columns in batches and combines results
    """
    import time
    
    # Get target from current_df
    target = current_df[state.target_column] if state.target_column in current_df.columns else None
    
    numeric_columns = []
    
    # Analyze numeric columns for transformation needs
    for col in current_df.columns:
        if col != state.target_column and pd.api.types.is_numeric_dtype(current_df[col]):
            analysis = analyze_column_comprehensive(current_df[col], target, col)
            
            # Only consider columns that might benefit from transformation
            skewness = abs(analysis.get('skewness', 0))
            kurtosis = abs(analysis.get('kurtosis', 0))
            
            if skewness > 1.0 or kurtosis > 5.0 or not analysis.get('is_likely_normal', False):
                numeric_columns.append({
                    'column': col,
                    'analysis': analysis
                })
    
    if not numeric_columns:
        return {'transformation_columns': [], 'llm_recommendations': {}}
    
    print(f"📊 Processing {len(numeric_columns)} transformation columns in chunks of {chunk_size}")
    
    # Process in chunks
    all_recommendations = {}
    all_analysis_details = {}
    start_time = time.time()
    total_chunks = (len(numeric_columns) + chunk_size - 1) // chunk_size
    
    # Send initial loading message if callback provided
    if progress_callback:
        progress_callback("transformations", 0, total_chunks, 0, len(numeric_columns), start_time, None, is_initial=True)
    
    for i in range(0, len(numeric_columns), chunk_size):
        chunk = numeric_columns[i:i + chunk_size]
        chunk_num = (i // chunk_size) + 1
        
        # Send progress update if callback provided
        if progress_callback:
            current_columns = len(all_recommendations) + len(chunk)
            progress_callback("transformations", chunk_num, total_chunks, current_columns, len(numeric_columns), start_time, None, is_initial=False)
        
        print(f"📊 Processing transformations chunk {chunk_num}/{total_chunks} ({len(chunk)} columns)")
        
        try:
            chunk_recommendations = analyze_transformations_chunk(state, chunk, current_df)
            all_recommendations.update(chunk_recommendations)
            
            # Store analysis details
            for col_info in chunk:
                all_analysis_details[col_info['column']] = col_info['analysis']
                
        except Exception as e:
            print(f"❌ Error processing transformations chunk {chunk_num}: {e}")
            # Fallback for this chunk
            for col_info in chunk:
                col = col_info['column']
                analysis = col_info['analysis']
                skewness = analysis.get('skewness', 0)
                
                if skewness > 2:
                    transformation = "log1p"
                elif skewness > 1:
                    transformation = "sqrt"
                elif skewness < -1:
                    transformation = "square"
                else:
                    transformation = "standardize"
                    
                all_recommendations[col] = {
                    "transformation": transformation,
                    "reasoning": f"Fallback rule for transformations chunk {chunk_num}",
                    "priority": "medium"
                }
                all_analysis_details[col] = analysis
    
    total_time = time.time() - start_time
    print(f"✅ Completed transformations analysis for {len(all_recommendations)} columns in {total_time:.2f}s")
    
    return {
        'transformation_columns': list(all_recommendations.keys()),
        'llm_recommendations': all_recommendations,
        'analysis_details': all_analysis_details
    }

def analyze_transformations_chunk(state: SequentialState, chunk: list, current_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze a single chunk of numeric columns for transformations"""
    llm = get_llm_from_state(state)
    
    # Create optimized prompt for this chunk
    prompt = f"""Analyze transformations for {len(chunk)} numeric columns. Target: {state.target_column}

TRANSFORMATIONS: log, log1p, sqrt, box_cox, yeo_johnson, standardize, robust_scale, quantile, none

COLUMNS:"""
    
    for col_info in chunk:
        col = col_info['column']
        analysis = col_info['analysis']
        
        prompt += f"""

{col}:
- Skewness: {analysis.get('skewness', 0):.2f}
- Kurtosis: {analysis.get('kurtosis', 0):.2f}
- Normal: {analysis.get('is_likely_normal', False)}
- Target corr: {analysis.get('target_correlation', 0):.3f}
- Range: {analysis.get('percentile_1', 0):.2f} to {analysis.get('percentile_99', 0):.2f}"""
    
    prompt += f"""

Return JSON: {{"column_name": {{"transformation": "transformation_name", "reasoning": "brief reason", "priority": "high/medium/low"}}}}"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            recommendations = json.loads(json_match.group())
        else:
            recommendations = {}
        
        # If no recommendations were parsed, use fallback
        if not recommendations:
            print("⚠️ No valid JSON parsed from LLM response, using fallback")
            fallback_recommendations = {}
            for col_info in chunk:
                col = col_info['column']
                analysis = col_info['analysis']
                
                # Simple rule-based fallback for transformations
                skewness = abs(analysis.get('skewness', 0))
                kurtosis = abs(analysis.get('kurtosis', 0))
                
                if skewness > 2.0:
                    transformation = "log1p"  # For highly skewed data
                elif skewness > 1.0:
                    transformation = "sqrt"   # For moderately skewed data
                elif kurtosis > 5.0:
                    transformation = "standardize"  # For high kurtosis
                else:
                    transformation = "none"  # No transformation needed
                    
                fallback_recommendations[col] = {
                    "transformation": transformation,
                    "reasoning": f"Fallback rule: skewness={skewness:.2f}, kurtosis={kurtosis:.2f}",
                    "priority": "medium"
                }
            
            recommendations = fallback_recommendations
        
        return recommendations
        
    except Exception as e:
        print(f"Error in transformations chunk analysis: {e}")
        # Return empty dict, will be handled by fallback in parent function
        return {}

# Part 5: Core Sequential Workflow Functions

def initialize_dataset_analysis(state: SequentialState) -> SequentialState:
    """
    Initial dataset analysis and overview generation
    """
    print(f"🔍 Initializing dataset analysis for {state.df_path}")
    
    # Load data if not already loaded
    if state.df is None:
        try:
            state.df = pd.read_csv(state.df_path)
            print(f"📊 Loaded dataset: {state.df.shape[0]} rows, {state.df.shape[1]} columns")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return state.copy(update={"current_step": "error"})
    
    # Basic dataset validation
    print(f"🔧 DEBUG: Checking target column '{state.target_column}' in dataset columns: {list(state.df.columns)}")
    if state.target_column not in state.df.columns:
        print(f"❌ Target column '{state.target_column}' not found in dataset")
        print(f"🔧 DEBUG: Available columns: {list(state.df.columns)}")
        print(f"🔧 DEBUG: Target column type: {type(state.target_column)}")
        return state.copy(update={"current_step": "error"})
    else:
        print(f"✅ Target column '{state.target_column}' found in dataset")
    
    # Quick overview analysis
    total_columns = len(state.df.columns) - 1  # Exclude target
    numeric_cols = state.df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = state.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target from counts
    if state.target_column in numeric_cols:
        numeric_cols.remove(state.target_column)
    if state.target_column in categorical_cols:
        categorical_cols.remove(state.target_column)
    
    # Quick outlier scan
    outlier_columns = []
    for col in numeric_cols:
        if pd.api.types.is_numeric_dtype(state.df[col]):
            clean_series = state.df[col].dropna()
            if len(clean_series) > 4:
                Q1 = clean_series.quantile(0.25)
                Q3 = clean_series.quantile(0.75)
                IQR = Q3 - Q1
                # Fix: Use np.logical_or instead of | for boolean arrays
                outliers = np.logical_or(clean_series < (Q1 - 1.5 * IQR), clean_series > (Q3 + 1.5 * IQR)).sum()
                if outliers > 0:
                    outlier_columns.append(col)
    
    # Quick missing value scan
    missing_columns = []
    for col in state.df.columns:
        if col != state.target_column and state.df[col].isnull().sum() > 0:
            missing_columns.append(col)
    
    # Update state with overview
    overview_results = {
        'total_columns': total_columns,
        'numeric_columns': len(numeric_cols),
        'categorical_columns': len(categorical_cols),
        'outlier_columns': len(outlier_columns),
        'missing_columns': len(missing_columns),
        'transformation_candidates': 0  # Will be calculated later
    }
    
    return state.copy(update={
        "total_columns": total_columns,
        "current_phase": PreprocessingPhase.OVERVIEW,
        "phase_results": {"overview": overview_results},
        "current_step": "overview_ready"
    })

def generate_overview_summary(state: SequentialState) -> str:
    """
    Generate user-friendly overview summary with educational suggestions
    """
    overview = state.phase_results.get('overview', {})
    
    summary = f"""📊 **Dataset Analysis Complete**

🎯 **{overview.get('total_columns', 0)} columns analyzed** targeting `{state.target_column}`

📋 **Preprocessing Phases Identified:**
⚠️  **Phase 1: Outliers** - {overview.get('outlier_columns', 0)} columns need attention
📈 **Phase 2: Missing Values** - {overview.get('missing_columns', 0)} columns need imputation  
🏷️  **Phase 3: Encoding** - {overview.get('categorical_columns', 0)} categorical columns
🔄 **Phase 4: Transformations** - Will analyze distribution shapes

💡 **What You Can Do:**
• `proceed` - Start with Phase 1 (Outliers)
• `skip outliers` - Jump to missing values
• `overview details` - See column breakdown
• `help` - Get preprocessing guidance
• `set config` - Adjust thresholds

**Ready to start Phase 1 (Outliers)?**"""

    if state.suggestions_enabled:
        summary += f"""

🧠 **Educational Note:**
Outliers can significantly impact model performance. We'll analyze {overview.get('outlier_columns', 0)} columns using multiple detection methods (IQR, Z-score) and recommend treatment strategies based on statistical properties and domain context."""

    return summary

def classify_user_intent_with_llm(user_input: str, current_phase: str, available_actions: list, state: SequentialState) -> dict:
    """Specialized intent classifier using enhanced LLM prompt"""
    # Use the same LLM initialization pattern as other functions
    llm = get_llm_from_state(state)
    
    prompt = f"""Classify this user input into exactly one category based on their clear intent.

USER INPUT: "{user_input}"
CURRENT PHASE: {current_phase}

CLASSIFICATION CATEGORIES WITH CLEAR EXAMPLES:

1. PROCEED (user wants to apply current phase and continue):
   • "proceed" → proceed
   • "yes" → proceed (clear agreement to proceed)
   • "okay" → proceed (clear agreement)
   • "ok" → proceed (clear agreement)
   • "sure" → proceed (clear agreement)
   • "go ahead" → proceed (clear agreement to continue)
   • "continue" → proceed
   • "apply these changes" → proceed
   • "looks good, continue" → proceed
   • "let's proceed" → proceed
   • "next" → proceed
   • "sounds good" → proceed (agreement to proceed)
   • "yeah" → proceed (agreement)
   • "proceed to next" → proceed
   • "let's go" → proceed
   • "do it" → proceed
   • "apply" → proceed
   • "start" → proceed
   • "begin" → proceed
   • "move forward" → proceed
   • "continue to next" → proceed

2. SKIP (user wants to skip current phase entirely):
   • "skip" → skip
   • "skip this phase" → skip
   • "skip outlier detection" → skip
   • "skip to next" → skip
   • "skip to next phase" → skip
   • "bypass this step" → skip
   • "move to next phase" → skip
   • "let's skip this" → skip
   • "move on" → skip
   • "skip this" → skip
   • "skip current phase" → skip
   • "skip this step" → skip
   • "skip to missing values" → skip (any skip command should just skip to next)
   • "skip to encoding" → skip (any skip command should just skip to next)
   • "skip to transformations" → skip (any skip command should just skip to next)
   • "jump to next" → skip
   • "go to next" → skip

3. OVERRIDE (user wants to modify/change current strategy):
   • "modify income treatment" → override
   • "change strategy for age column" → override
   • "use median imputation instead" → override
   • "apply different encoding" → override
   • "change" → override
   • "modify" → override
   • "use different strategy" → override
   • "I want to change the approach" → override
   • "do mean imputation for all" → override
   • "use median for all columns" → override
   • "apply winsorize to all" → override
   • "change all to keep" → override
   • "do [strategy] for all" → override
   • "use [strategy] for all" → override
   • "apply [strategy] to all" → override
   • "move all [strategy1] to [strategy2]" → override
   • "dont do any transformation to [column]" → override
   • "dont transform [column]" → override
   • "no transformation for [column]" → override
   • "skip transformation for [column]" → override
   • "dont apply any transformation to [column]" → override
   • "keep [column] as is" → override
   • "leave [column] unchanged" → override
   • "dont do anything to [column]" → override

4. QUERY (user asking questions or wants information):
   • "what are outliers?" → query
   • "explain missing value strategies" → query
   • "how does target encoding work?" → query
   • "why is median better than mean?" → query
   • "what will happen to the income column?" → query
   • "?" → query
   • "what?" → query
   • "explain" → query
   • "tell me more" → query
   • "I don't understand" → query

5. SUMMARY (user wants to see current phase summary/strategies):
   • "show current strategies" → summary
   • "give summary" → summary
   • "what are current strategies" → summary
   • "show me the plan" → summary
   • "current plan" → summary
   • "what's the current plan" → summary
   • "show strategies" → summary
   • "current strategies" → summary

6. CANCEL (user wants to cancel current action/confirmation):
   • "cancel" → cancel
   • "stay" → cancel (user wants to stay in current phase)
   • "don't skip" → cancel (user doesn't want to skip)
   • "no" → cancel (user doesn't want to proceed)
   • "stop" → cancel
   • "nevermind" → cancel
   • "forget it" → cancel
   • "I changed my mind" → cancel
   • "let me think about it" → cancel
   • "not now" → cancel
   • "wait" → cancel
   • "hold on" → cancel
   • "I want to stay" → cancel
   • "keep me here" → cancel
   • "don't proceed" → cancel
   • "abort" → cancel
   • "back out" → cancel

7. NAVIGATE (user wants to jump to specific phase - NOT skipping):
   • "jump to missing values" → navigate
   • "go to encoding phase" → navigate
   • "take me to transformations" → navigate
   • "navigate to outliers" → navigate
   • "go to [phase]" → navigate (when not skipping)

8. EXIT (user wants to end session):
   • "quit" → exit
   • "stop the process" → exit
   • "exit session" → exit
   • "I'm done" → exit

KEY CLASSIFICATION RULES:
• Simple agreements ("yes", "okay", "ok", "sure", "go ahead") → PROCEED
• Question words ("what", "how", "why", "explain") → QUERY
• Action commands ("modify", "change", "use", "do", "apply") → OVERRIDE
• Bulk commands ("do [strategy] for all", "use [strategy] for all", "apply [strategy] to all") → OVERRIDE
• Skip commands ("skip", "bypass", "move on") → SKIP
• Help requests ("help", "guide me", "options") → HELP
• Exit commands ("quit", "stop", "done") → EXIT

When in doubt, favor PROCEED for positive responses and QUERY for unclear requests.

Return ONLY this JSON format:
{{
  "intent": "proceed|skip|override|query|summary|cancel|navigate|exit"
}}"""
    
    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)]).content
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            import json
            result = json.loads(json_match.group())
            intent = result.get('intent', 'query')
            
            # Map intent to action and set defaults
            action_mapping = {
                'proceed': 'next_phase',
                'skip': 'skip_phase', 
                'query': 'show_details',
                'override': 'modify_config',
                'help': 'show_help',
                'navigate': 'navigate_phase',
                'exit': 'exit'
            }
            
            response_type_mapping = {
                'proceed': 'action',
                'skip': 'action',
                'query': 'answer', 
                'override': 'action',
                'help': 'suggestion',
                'navigate': 'action',
                'exit': 'exit'
            }
            
            # Build complete response with defaults
            complete_result = {
                'intent': intent,
                'action': action_mapping.get(intent, 'show_details'),
                'response_type': response_type_mapping.get(intent, 'answer'),
                'target_phase': result.get('target_phase'),
                'explanation': f"Classified as {intent}",
                'suggested_response': result.get('suggested_response', "I'll help you with that.")
            }
            
            return complete_result
        else:
            return {'intent': 'query', 'action': 'show_details', 'response_type': 'answer'}
            
    except Exception as e:
        print(f"Intent classification error: {e}")
        return {'intent': 'query', 'action': 'show_details', 'response_type': 'answer'}

def process_user_input_with_llm(state: SequentialState, user_input: str) -> SequentialState:
    """Process user input using the enhanced intent classifier and return updated state"""
    current_phase = state.current_phase
    available_actions = get_available_actions(state)
    
    # Use specialized intent classifier
    analysis = classify_user_intent_with_llm(user_input, current_phase, available_actions, state)
    
    # Update state based on analysis
    updates = {
        "user_messages": state.user_messages + [user_input],
        "current_step": "user_input_processed"
    }
    
    # Handle different intents
    intent = analysis.get("intent", "query")
    
    # Special handling for navigation choices
    if hasattr(state, 'navigation_context') and state.navigation_context and state.current_step == "navigation_choice_required":
        from enhanced_navigation import handle_navigation_choice
        
        # Handle navigation choice
        choice_result = handle_navigation_choice(
            user_input,
            state.navigation_context["analysis"],
            state.navigation_context["target_phase"],
            state.current_phase,
            getattr(state, 'completed_phases', set())
        )
        
        if choice_result["success"]:
            updates["current_phase"] = choice_result["new_phase"]
            updates["current_step"] = "phase_navigated"
            updates["query_response"] = choice_result["response"]
            updates["completed_phases"] = choice_result["updated_completed_phases"]
        else:
            updates["query_response"] = choice_result["response"]
        
        # Clear navigation context
        updates["navigation_context"] = None
        updates["is_query"] = False
        
        return state.copy(update=updates)
    
    if intent == "proceed":
        updates["phase_approved"] = True
    elif intent == "skip":
        target_phase = analysis.get("target_phase")
        if target_phase:
            updates["current_phase"] = target_phase
        else:
            # Skip current phase - advance to next
            next_phase = get_next_phase(state.current_phase)
            updates["current_phase"] = next_phase
        updates["current_step"] = "phase_skipped"
    elif intent == "query":
        updates["is_query"] = True
        updates["query_response"] = analysis.get("suggested_response", "I can help with that.")
    elif intent == "override":
        updates["is_query"] = True
        updates["current_step"] = "override_requested"
        updates["query_response"] = analysis.get("suggested_response", "I'll help you modify the preprocessing strategy.")
    elif intent == "help":
        updates["is_query"] = True
        updates["current_step"] = "help_requested"
        updates["query_response"] = analysis.get("suggested_response", "I'm here to guide you through preprocessing.")
    elif intent == "navigate":
        # Enhanced navigation with dependency checking
        from enhanced_navigation import enhance_navigation_in_bot
        
        # Get completed phases from state
        completed_phases = getattr(state, 'completed_phases', set())
        if hasattr(state, 'phase_results'):
            # Convert phase_results keys to completed phases
            completed_phases = set(state.phase_results.keys())
        
        # Use enhanced navigation
        nav_result = enhance_navigation_in_bot(
            user_input, 
            state.current_phase, 
            completed_phases
        )
        
        if nav_result["success"]:
            # Safe navigation - execute immediately
            updates["current_phase"] = nav_result["new_phase"]
            updates["current_step"] = "phase_navigated"
            updates["query_response"] = nav_result["response"]
        elif nav_result["requires_user_choice"]:
            # Requires user choice - store navigation context
            updates["is_query"] = True
            updates["current_step"] = "navigation_choice_required"
            updates["query_response"] = nav_result["response"]
            updates["navigation_context"] = {
                "analysis": nav_result["analysis"],
                "target_phase": nav_result["target_phase"]
            }
        else:
            # Navigation failed
            updates["is_query"] = True
            updates["current_step"] = "navigation_failed"
            updates["query_response"] = nav_result["response"]
    elif intent == "config":
        updates["is_query"] = True
        updates["current_step"] = "config_requested"
        updates["query_response"] = analysis.get("suggested_response", "I can help you adjust configuration settings.")
    elif intent == "exit":
        updates["current_step"] = "exit_requested"
        updates["query_response"] = "Thank you for using the Sequential Preprocessing Agent. Your session has ended."
        return state.copy(update=updates)
    
    return state.copy(update=updates)

def get_available_actions(state: SequentialState) -> List[str]:
    """Get list of available actions based on current state"""
    actions = ["proceed", "skip", "help", "overview"]
    
    if state.current_phase == PreprocessingPhase.OVERVIEW:
        actions.extend(["start outliers", "start missing", "start encoding"])
    elif state.current_phase == PreprocessingPhase.OUTLIERS:
        actions.extend(["show outliers", "modify treatment"])
    elif state.current_phase == PreprocessingPhase.MISSING_VALUES:
        actions.extend(["show missing", "modify imputation"])
    elif state.current_phase == PreprocessingPhase.ENCODING:
        actions.extend(["show encoding", "modify strategy"])
    
    return actions

def get_next_phase(current_phase: str) -> str:
    """Get the next phase in sequence"""
    phase_order = [
        PreprocessingPhase.OVERVIEW,
        PreprocessingPhase.OUTLIERS,
        PreprocessingPhase.MISSING_VALUES,
        PreprocessingPhase.ENCODING,
        PreprocessingPhase.TRANSFORMATIONS,
        PreprocessingPhase.COMPLETION
    ]
    
    try:
        current_index = phase_order.index(current_phase)
        if current_index < len(phase_order) - 1:
            return phase_order[current_index + 1]
        else:
            return PreprocessingPhase.COMPLETION
    except ValueError:
        return PreprocessingPhase.OUTLIERS 

# Part 6: LangGraph Agent Implementation

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph = None
    END = None
    LANGGRAPH_AVAILABLE = False

def create_sequential_preprocessing_agent():
    """
    Create the LangGraph agent for sequential preprocessing
    Returns compiled StateGraph if available, None otherwise
    """
    if not LANGGRAPH_AVAILABLE:
        print("⚠️ LangGraph not available, cannot create preprocessing agent")
        return None
    
    def overview_node(state: SequentialState) -> SequentialState:
        """Generate overview and wait for user input"""
        print("📊 Phase 0: Dataset Overview")
        
        if state.current_step != "overview_ready":
            state = initialize_dataset_analysis(state)
        
        if state.current_step == "error":
            return state
        
        summary = generate_overview_summary(state)
        return state.copy(update={
            "current_step": "awaiting_user_input",
            "query_response": summary
        })
    
    def outliers_node(state: SequentialState) -> SequentialState:
        """Analyze and present outlier treatment recommendations"""
        print("⚠️  Phase 1: Outlier Analysis")
        
        # Run outlier analysis
        outlier_results = analyze_outliers_with_llm(state)
        
        if not outlier_results['outlier_columns']:
            summary = "✅ **No Outliers Detected**\n\nAll numeric columns are within normal ranges. Ready to proceed to Phase 2 (Missing Values)?"
        else:
            outlier_cols = outlier_results['outlier_columns']
            recommendations = outlier_results['llm_recommendations']
            
            # Group by severity
            severe_cols = [col for col, rec in recommendations.items() if rec.get('severity') == 'severe']
            moderate_cols = [col for col, rec in recommendations.items() if rec.get('severity') == 'moderate']
            mild_cols = [col for col, rec in recommendations.items() if rec.get('severity') == 'mild']
            
            summary = f"""⚠️  **Outlier Detection Results ({len(outlier_cols)} columns):**

🔴 **Severe (>15%):** {len(severe_cols)} columns
{chr(10).join([f'• {col}: {recommendations.get(col, {}).get("treatment", "N/A")} - {recommendations.get(col, {}).get("reasoning", "N/A")[:50]}...' for col in severe_cols[:5]])}

🟡 **Moderate (5-15%):** {len(moderate_cols)} columns  
{chr(10).join([f'• {col}: {recommendations.get(col, {}).get("treatment", "N/A")} - {recommendations.get(col, {}).get("reasoning", "N/A")[:50]}...' for col in moderate_cols[:5]])}

🟢 **Mild (<5%):** {len(mild_cols)} columns
{chr(10).join([f'• {col}: {recommendations.get(col, {}).get("treatment", "N/A")}' for col in mild_cols[:3]])}

💡 **Recommended Strategy:**
• Winsorize/clip severe outliers to preserve data while reducing impact
• Keep mild outliers as they may contain valuable information  
• Transform highly skewed distributions

**Options:**
• `proceed` - Apply recommended treatments
• `show details` - See full analysis for specific columns
• `modify [column]` - Override treatment for specific columns
• `skip outliers` - Skip this phase"""

        # Update state with results
        updated_phase_results = state.phase_results.copy()
        updated_phase_results['outliers'] = outlier_results
        
        # Mark current phase as completed and advance
        completed_phases = state.completed_phases.copy()
        if state.current_phase not in completed_phases:
            completed_phases.append(state.current_phase)
        
        return state.copy(update={
            "current_phase": PreprocessingPhase.OUTLIERS,
            "phase_results": updated_phase_results,
            "completed_phases": completed_phases,
            "phase_approved": False,  # Reset for next user input
            "current_step": "awaiting_user_input",
            "query_response": summary
        })
    
    def missing_values_node(state: SequentialState) -> SequentialState:
        """Analyze and present missing value imputation strategies"""
        print("📈 Phase 2: Missing Values Analysis")
        
        # Run missing value analysis
        missing_results = analyze_missing_values_with_llm(state)
        
        if not missing_results['missing_columns']:
            summary = "✅ **No Missing Values Detected**\n\nAll columns are complete. Ready to proceed to Phase 3 (Encoding)?"
        else:
            missing_cols = missing_results['missing_columns']
            recommendations = missing_results['llm_recommendations']
            
            # Group by strategy
            drop_cols = [col for col, rec in recommendations.items() if rec.get('strategy') == 'drop_column']
            mean_cols = [col for col, rec in recommendations.items() if rec.get('strategy') == 'mean']
            median_cols = [col for col, rec in recommendations.items() if rec.get('strategy') == 'median']
            mode_cols = [col for col, rec in recommendations.items() if rec.get('strategy') == 'mode']
            model_cols = [col for col, rec in recommendations.items() if rec.get('strategy') == 'model_based']
            
            summary = f"""📈 **Missing Value Analysis ({len(missing_cols)} columns):**

🗑️ **Drop Columns (>70% missing):** {len(drop_cols)}
{', '.join(drop_cols[:5]) + ('...' if len(drop_cols) > 5 else '')}

📊 **Mean Imputation (Normal distribution):** {len(mean_cols)}
{', '.join(mean_cols[:5]) + ('...' if len(mean_cols) > 5 else '')}

📊 **Median Imputation (Skewed distribution):** {len(median_cols)}
{', '.join(median_cols[:5]) + ('...' if len(median_cols) > 5 else '')}

📊 **Mode Imputation (Categorical):** {len(mode_cols)}
{', '.join(mode_cols[:5]) + ('...' if len(mode_cols) > 5 else '')}

🤖 **Model-based Imputation (High correlation):** {len(model_cols)}
{', '.join(model_cols[:3])}

💡 **Strategy Summary:**
Imputation choices based on distribution shape, missing percentage, and target correlation.

**Options:**
• `proceed` - Apply recommended imputation
• `show details` - See reasoning for specific columns  
• `modify [column]` - Change strategy for specific columns
• `skip missing` - Skip this phase"""

        # Update state with results
        updated_phase_results = state.phase_results.copy()
        updated_phase_results['missing_values'] = missing_results
        
        # Mark current phase as completed and advance
        completed_phases = state.completed_phases.copy()
        if state.current_phase not in completed_phases:
            completed_phases.append(state.current_phase)
        
        return state.copy(update={
            "current_phase": PreprocessingPhase.MISSING_VALUES,
            "phase_results": updated_phase_results,
            "completed_phases": completed_phases,
            "phase_approved": False,  # Reset for next user input
            "current_step": "awaiting_user_input",
            "query_response": summary
        })
    
    def encoding_node(state: SequentialState) -> SequentialState:
        """Analyze and present categorical encoding strategies"""
        print("🏷️  Phase 3: Categorical Encoding Analysis")
        
        
        # Run encoding analysis
        encoding_results = analyze_encoding_with_llm(state)
        
        if not encoding_results['categorical_columns']:
            summary = "✅ **No Categorical Columns Detected**\n\nAll features are numeric. Ready to proceed to Phase 4 (Transformations)?"
        else:
            cat_cols = encoding_results['categorical_columns']
            recommendations = encoding_results['llm_recommendations']
            
            # Group by encoding type
            onehot_cols = [col for col, rec in recommendations.items() if rec.get('encoding') == 'onehot']
            onehot_top_cols = [col for col, rec in recommendations.items() if rec.get('encoding') == 'onehot_top']
            target_cols = [col for col, rec in recommendations.items() if rec.get('encoding') == 'target']
            label_cols = [col for col, rec in recommendations.items() if rec.get('encoding') == 'label']
            drop_cols = [col for col, rec in recommendations.items() if rec.get('encoding') == 'drop']
            
            # Calculate dimensionality impact
            total_new_cols = 0
            for col, rec in recommendations.items():
                dim = rec.get('expected_dimensionality', 1)
                if isinstance(dim, str) and dim.isdigit():
                    total_new_cols += int(dim)
                elif isinstance(dim, int):
                    total_new_cols += dim
                else:
                    total_new_cols += 1
            
            summary = f"""🏷️  **Categorical Encoding Strategy ({len(cat_cols)} columns):**

🎯 **One-Hot Encoding (Low cardinality):** {len(onehot_cols)}
{', '.join(onehot_cols[:5]) + ('...' if len(onehot_cols) > 5 else '')}

🎯 **One-Hot + "Other" (Medium cardinality):** {len(onehot_top_cols)}
{', '.join(onehot_top_cols[:3])} - Keep top {state.onehot_top_categories} categories

🎯 **Target Encoding (High cardinality):** {len(target_cols)}
{', '.join(target_cols[:3])}

🔢 **Label Encoding (Ordinal):** {len(label_cols)}
{', '.join(label_cols[:3])}

🗑️ **Drop (Very high cardinality/ID):** {len(drop_cols)}
{', '.join(drop_cols[:3])}

📊 **Dimensionality Impact:** ~{total_new_cols} total encoded columns

💡 **Smart One-Hot Strategy:**
For medium cardinality columns, we keep the most frequent categories and group the rest as "Other" to control dimensionality while preserving information.

**Options:**
• `proceed` - Apply encoding strategies
• `show details` - See cardinality analysis
• `modify [column]` - Change encoding for specific columns  
• `skip encoding` - Skip this phase"""

        # Update state with results
        updated_phase_results = state.phase_results.copy()
        updated_phase_results['encoding'] = encoding_results
        
        # Mark current phase as completed and advance
        completed_phases = state.completed_phases.copy()
        if state.current_phase not in completed_phases:
            completed_phases.append(state.current_phase)
        
        return state.copy(update={
            "current_phase": PreprocessingPhase.ENCODING,
            "phase_results": updated_phase_results,
            "completed_phases": completed_phases,
            "phase_approved": False,  # Reset for next user input
            "current_step": "awaiting_user_input",
            "query_response": summary
        })
    
    def transformations_node(state: SequentialState) -> SequentialState:
        """Analyze and present distribution transformation strategies"""
        print("🔄 Phase 4: Distribution Transformations Analysis")
        
        # Run transformation analysis
        transformation_results = analyze_transformations_with_llm(state)
        
        if not transformation_results['transformation_columns']:
            summary = "✅ **No Transformations Needed**\n\nAll numeric distributions are acceptable. Ready to complete preprocessing!"
        else:
            transform_cols = transformation_results['transformation_columns']
            recommendations = transformation_results['llm_recommendations']
            
            # Group by transformation type
            log_cols = [col for col, rec in recommendations.items() if 'log' in rec.get('transformation', '')]
            sqrt_cols = [col for col, rec in recommendations.items() if rec.get('transformation') == 'sqrt']
            standardize_cols = [col for col, rec in recommendations.items() if rec.get('transformation') == 'standardize']
            robust_cols = [col for col, rec in recommendations.items() if rec.get('transformation') == 'robust_scale']
            scale_cols = [col for col, rec in recommendations.items() if rec.get("transformation") in ["standardize", "robust_scale"]]
            other_cols = [col for col, rec in recommendations.items() if rec.get('transformation') not in ['log', 'log1p', 'sqrt', 'standardize', 'robust_scale']]
            
            summary = f"""🔄 **Distribution Transformation Analysis ({len(transform_cols)} columns):**

📈 **Log Transformations (Right skew):** {len(log_cols)}
{', '.join(log_cols[:5]) + ('...' if len(log_cols) > 5 else '')}

📈 **Square Root (Moderate skew):** {len(sqrt_cols)}
{', '.join(sqrt_cols[:3])}

⚖️ **Scaling (Different magnitudes):** {len(scale_cols)}
{', '.join(scale_cols[:3])}

🔄 **Other Transformations:** {len(other_cols)}
{', '.join(other_cols[:3])}

💡 **Transformation Benefits:**
• Improve normality for better model performance
• Handle different scales between features
• Reduce impact of outliers through scaling
• Better linear relationships

**Options:**
• `proceed` - Apply transformation strategies
• `show details` - See skewness analysis
• `modify [column]` - Change transformation for specific columns
• `complete` - Finish preprocessing"""

        # Update state with results
        updated_phase_results = state.phase_results.copy()
        updated_phase_results['transformations'] = transformation_results
        
        # Mark current phase as completed and advance
        completed_phases = state.completed_phases.copy()
        if state.current_phase not in completed_phases:
            completed_phases.append(state.current_phase)
        
        return state.copy(update={
            "current_phase": PreprocessingPhase.TRANSFORMATIONS,
            "phase_results": updated_phase_results,
            "completed_phases": completed_phases,
            "phase_approved": False,  # Reset for next user input
            "current_step": "awaiting_user_input",
            "query_response": summary
        })
    
    def completion_node(state: SequentialState) -> SequentialState:
        """Generate final preprocessing summary and cleaned dataset"""
        print("✅ Phase 5: Preprocessing Complete")
        
        completed = state.completed_phases
        total_phases = len([p for p in [PreprocessingPhase.OUTLIERS, PreprocessingPhase.MISSING_VALUES, 
                           PreprocessingPhase.ENCODING, PreprocessingPhase.TRANSFORMATIONS] if p in completed])
        
        summary = f"""🎉 **Preprocessing Complete!**

✅ **Phases Completed:** {len(completed_phases)}
📊 **Dataset:** {df.shape[0]:,} rows × {df.shape[1]:,} columns
🎯 **Target:** {state.target_column}

**Next Steps:**
• Your data is now ready for machine learning
• All preprocessing steps have been applied

*Thank you for using the Interactive Preprocessing Agent!* 🚀"""

        return state.copy(update={
            "current_phase": PreprocessingPhase.COMPLETION,
            "current_step": "complete",
            "query_response": summary
        })
    
    def user_input_node(state: SequentialState) -> SequentialState:
        """Handle user input with improved LLM-based intent classification"""
        if not state.user_messages:
            return state.copy(update={"current_step": "awaiting_user_input"})
        
        user_input = state.user_messages[-1]
        current_phase = state.current_phase
        available_actions = get_available_actions(state)
        
        # Use specialized intent classifier
        analysis = classify_user_intent_with_llm(user_input, current_phase, available_actions, state)
        
        # Update state based on analysis
        updates = {
            "user_messages": state.user_messages + [user_input],
            "current_step": "user_input_processed"
        }
        
        # Handle different intents
        intent = analysis.get("intent", "query")
        
        # Special handling for navigation choices
        if hasattr(state, 'navigation_context') and state.navigation_context and state.current_step == "navigation_choice_required":
            from enhanced_navigation import handle_navigation_choice
            
            # Handle navigation choice
            choice_result = handle_navigation_choice(
                user_input,
                state.navigation_context["analysis"],
                state.navigation_context["target_phase"],
                state.current_phase,
                getattr(state, 'completed_phases', set())
            )
            
            if choice_result["success"]:
                updates["current_phase"] = choice_result["new_phase"]
                updates["current_step"] = "phase_navigated"
                updates["query_response"] = choice_result["response"]
                updates["completed_phases"] = choice_result["updated_completed_phases"]
            else:
                updates["query_response"] = choice_result["response"]
            
            # Clear navigation context
            updates["navigation_context"] = None
            updates["is_query"] = False
            
            return state.copy(update=updates)
        
        if intent == "proceed":
            updates["phase_approved"] = True
        elif intent == "skip":
            target_phase = analysis.get("target_phase")
            if target_phase:
                updates["current_phase"] = target_phase
            else:
                # Skip current phase - advance to next
                next_phase = get_next_phase(state.current_phase)
                updates["current_phase"] = next_phase
            updates["current_step"] = "phase_skipped"
        elif intent == "query":
            updates["is_query"] = True
            updates["query_response"] = analysis.get("suggested_response", "I can help with that.")
        elif intent == "override":
            updates["is_query"] = True
            updates["current_step"] = "override_requested"
            updates["query_response"] = analysis.get("suggested_response", "I'll help you modify the preprocessing strategy.")
        elif intent == "help":
            updates["is_query"] = True
            updates["current_step"] = "help_requested"
            updates["query_response"] = analysis.get("suggested_response", "I'm here to guide you through preprocessing.")
        elif intent == "navigate":
            # Enhanced navigation with dependency checking
            from enhanced_navigation import enhance_navigation_in_bot
            
            # Get completed phases from state
            completed_phases = getattr(state, 'completed_phases', set())
            if hasattr(state, 'phase_results'):
                # Convert phase_results keys to completed phases
                completed_phases = set(state.phase_results.keys())
            
            # Use enhanced navigation
            nav_result = enhance_navigation_in_bot(
                user_input, 
                state.current_phase, 
                completed_phases
            )
            
            if nav_result["success"]:
                # Safe navigation - execute immediately
                updates["current_phase"] = nav_result["new_phase"]
                updates["current_step"] = "phase_navigated"
                updates["query_response"] = nav_result["response"]
            elif nav_result["requires_user_choice"]:
                # Requires user choice - store navigation context
                updates["is_query"] = True
                updates["current_step"] = "navigation_choice_required"
                updates["query_response"] = nav_result["response"]
                updates["navigation_context"] = {
                    "analysis": nav_result["analysis"],
                    "target_phase": nav_result["target_phase"]
                }
            else:
                # Navigation failed
                updates["is_query"] = True
                updates["current_step"] = "navigation_failed"
                updates["query_response"] = nav_result["response"]
        elif intent == "config":
            updates["is_query"] = True
            updates["current_step"] = "config_requested"
            updates["query_response"] = analysis.get("suggested_response", "I can help you adjust configuration settings.")
        elif intent == "exit":
            updates["current_step"] = "exit_requested"
            updates["query_response"] = "Thank you for using the Sequential Preprocessing Agent. Your session has ended."
            return state.copy(update=updates)
        
        return state.copy(update=updates)
    
    def query_response_node(state: SequentialState) -> SequentialState:
        """Handle user queries about data or process"""
        # Reset query state and return response
        return state.copy(update={
            "is_query": False,
            "current_step": "query_answered"
        })
    
    # Build the graph
    workflow = StateGraph(SequentialState)
    
    # Add nodes
    workflow.add_node("overview", overview_node)
    workflow.add_node("outliers", outliers_node)  
    workflow.add_node("missing_values", missing_values_node)
    workflow.add_node("encoding", encoding_node)
    workflow.add_node("transformations", transformations_node)
    workflow.add_node("completion", completion_node)
    workflow.add_node("user_input", user_input_node)
    workflow.add_node("handle_query", query_response_node)
    
    # Set entry point
    workflow.set_entry_point("overview")
    
    # Direct sequential flow with interrupts
    workflow.add_edge("overview", "user_input")
    
    # Conditional routing from user_input
    def route_from_user_input(state: SequentialState) -> str:
        """Route from user input to appropriate next step"""
        # For initial analysis without user interaction, end at overview
        if not state.user_messages and state.current_step == "awaiting_user_input":
            return "__end__"
            
        if state.is_query:
            return "handle_query"
        elif state.phase_approved:
            # Route to next phase based on current phase and reset phase_approved
            if state.current_phase == PreprocessingPhase.OVERVIEW:
                return "outliers"
            elif state.current_phase == PreprocessingPhase.OUTLIERS:
                return "missing_values" 
            elif state.current_phase == PreprocessingPhase.MISSING_VALUES:
                return "encoding"
            elif state.current_phase == PreprocessingPhase.ENCODING:
                return "transformations"
            elif state.current_phase == PreprocessingPhase.TRANSFORMATIONS:
                return "completion"
            else:
                return "completion"
        else:
            # Stay in current phase - this creates the loop we want for interactive sessions
            return "user_input"
    
    workflow.add_conditional_edges(
        "user_input",
        route_from_user_input,
        {
            "outliers": "outliers",
            "missing_values": "missing_values",
            "encoding": "encoding", 
            "transformations": "transformations",
            "completion": "completion",
            "handle_query": "handle_query",
            "user_input": "user_input",  # Allow staying in user_input for interactive sessions
            "__end__": "__end__"  # Allow termination for initial analysis
        }
    )
    
    # Connect all other phases back to user_input
    workflow.add_edge("outliers", "user_input")
    workflow.add_edge("missing_values", "user_input")
    workflow.add_edge("encoding", "user_input")
    workflow.add_edge("transformations", "user_input") 
    workflow.add_edge("handle_query", "user_input")
    
    # Compile the graph
    return workflow.compile()

def run_sequential_agent(df_path: str, target_column: str, model_name: str = None):
    """
    Main function to run the sequential preprocessing agent
    """
    print("🚀 Starting Sequential Preprocessing Agent")
    print("=" * 50)
    
    # Use provided model_name or get from environment
    if model_name is None:
        model_name = os.environ.get("DEFAULT_MODEL", "gpt-4o")
    
    # Initialize state
    initial_state = SequentialState(
        df_path=df_path,
        target_column=target_column,
        model_name=model_name
    )
    
    # Create agent
    agent = create_sequential_preprocessing_agent()
    
    # Main interaction loop
    current_state = initial_state
    while current_state.current_phase != PreprocessingPhase.COMPLETION and current_state.current_step != "exit_requested":
        try:
            # Run one step of the agent
            result = agent.invoke(current_state)
            current_state = result
            
            # Check for exit request
            if current_state.current_step == "exit_requested":
                if current_state.query_response:
                    print("\n" + current_state.query_response)
                print("👋 Exiting preprocessing agent.")
                break
            
            # Display response to user
            if current_state.query_response:
                print("\n" + current_state.query_response)
                print("\n" + "─" * 50)
            
            # If waiting for user input, get it
            if current_state.current_step == "awaiting_user_input":
                user_input = input("\n💬 Your response: ").strip()
                # Let LLM handle all input processing, including exit requests
                # Add user input to state
                current_state = current_state.copy(update={
                    "user_messages": current_state.user_messages + [user_input],
                    "phase_approved": False,
                    "is_query": False
                })
            
        except KeyboardInterrupt:
            print("\n\n👋 Preprocessing interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error in agent execution: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n🎉 Sequential preprocessing agent session ended.")
    return current_state

def detect_and_handle_extreme_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect and handle extreme outliers by converting them to missing data.
    Returns cleaned dataframe and handling report.
    """
    import numpy as np
    
    handling_report = {
        'columns_processed': [],
        'extreme_outliers_found': {},
        'total_extreme_outliers': 0
    }
    
    df_cleaned = df.copy()
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Detect extreme outliers
            extreme_patterns = [
                np.isinf(df[col]),                    # ±infinity
                np.abs(df[col]) > 1e300,              # Default double values
                df[col] == -1.7e308,                  # Common default
                df[col] < -1e100,                     # Extreme negative
                df[col] > 1e100                       # Extreme positive
            ]
            
            extreme_mask = np.any(extreme_patterns, axis=0)
            extreme_count = extreme_mask.sum()
            
            if extreme_count > 0:
                # Convert extreme outliers to NaN
                df_cleaned.loc[extreme_mask, col] = np.nan
                
                handling_report['columns_processed'].append(col)
                handling_report['extreme_outliers_found'][col] = {
                    'count': extreme_count,
                    'percentage': (extreme_count / len(df)) * 100,
                    'patterns': {
                        'infinity': np.isinf(df[col]).sum(),
                        'extreme_negative': (df[col] < -1e100).sum(),
                        'extreme_positive': (df[col] > 1e100).sum(),
                        'default_double': (np.abs(df[col]) > 1e300).sum()
                    }
                }
                handling_report['total_extreme_outliers'] += extreme_count
    
    return df_cleaned, handling_report

def apply_preprocessing_pipeline(df: pd.DataFrame, state: SequentialState) -> pd.DataFrame:
    """Apply all preprocessing steps to the dataset based on state"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    
    df_processed = df.copy()
    
    # Apply phase results in order
    phase_results = getattr(state, 'phase_results', {})
    user_overrides = getattr(state, 'user_overrides', {})
    
    # Phase 1: Outliers
    if 'outliers' in phase_results:
        outlier_results = phase_results['outliers']
        recommendations = outlier_results.get('llm_recommendations', {})
        
        # Apply user overrides if any
        if 'outliers' in user_overrides:
            for col, strategy in user_overrides['outliers'].items():
                if col in recommendations:
                    recommendations[col]['treatment'] = strategy
        
        for col, rec in recommendations.items():
            if col in df_processed.columns:
                treatment = rec.get('treatment', 'keep')
                
                if treatment == 'winsorize':
                    # Winsorize outliers (1st and 99th percentiles)
                    q01 = df_processed[col].quantile(0.01)
                    q99 = df_processed[col].quantile(0.99)
                    df_processed[col] = df_processed[col].clip(lower=q01, upper=q99)
                
                elif treatment == 'mark_missing':
                    # Convert outliers to NaN
                    q01 = df_processed[col].quantile(0.01)
                    q99 = df_processed[col].quantile(0.99)
                    # Fix: Use np.logical_or instead of | for boolean arrays
                    outlier_mask = np.logical_or(df_processed[col] < q01, df_processed[col] > q99)
                    df_processed.loc[outlier_mask, col] = np.nan
                
                # 'keep' means no change
    
    # Phase 2: Missing Values
    if 'missing_values' in phase_results:
        missing_results = phase_results['missing_values']
        recommendations = missing_results.get('llm_recommendations', {})
        
        # Apply user overrides if any
        if 'missing_values' in user_overrides:
            for col, strategy in user_overrides['missing_values'].items():
                if col in recommendations:
                    recommendations[col]['treatment'] = strategy
        
        for col, rec in recommendations.items():
            if col in df_processed.columns:
                treatment = rec.get('treatment', 'mean')
                
                if treatment == 'mean':
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                elif treatment == 'median':
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                elif treatment == 'mode':
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                elif treatment == 'drop_column':
                    df_processed = df_processed.drop(columns=[col])
                elif treatment == 'forward_fill':
                    df_processed[col] = df_processed[col].fillna(method='ffill')
    
    # Phase 3: Encoding
    if 'encoding' in phase_results:
        encoding_results = phase_results['encoding']
        recommendations = encoding_results.get('llm_recommendations', {})
        
        # Apply user overrides if any
        if 'encoding' in user_overrides:
            for col, strategy in user_overrides['encoding'].items():
                if col in recommendations:
                    recommendations[col]['encoding'] = strategy
        
        for col, rec in recommendations.items():
            if col in df_processed.columns:
                # Handle both 'strategy' and 'encoding' keys from LLM
                encoding = rec.get('encoding', rec.get('strategy', 'label'))
                
                # Map strategy names to encoding names
                if encoding == 'onehot_encoding':
                    encoding = 'onehot'
                elif encoding == 'label_encoding':
                    encoding = 'label'
                elif encoding == 'target_encoding':
                    encoding = 'target'
                elif encoding == 'binary_encoding':
                    encoding = 'label'  # Fallback to label encoding
                elif encoding == 'drop_column':
                    encoding = 'drop'
                
                if encoding == 'label':
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                elif encoding == 'onehot':
                    # Create dummy variables
                    dummies = pd.get_dummies(df_processed[col], prefix=col)
                    df_processed = pd.concat([df_processed, dummies], axis=1)
                    df_processed = df_processed.drop(columns=[col])
                elif encoding == 'target':
                    # Target encoding - replace categories with mean target values
                    target_col = state.target_column
                    if target_col in df_processed.columns:
                        target_means = df_processed.groupby(col)[target_col].mean()
                        df_processed[col] = df_processed[col].map(target_means)
                        # Fill any missing values with overall mean
                        overall_mean = df_processed[target_col].mean()
                        df_processed[col] = df_processed[col].fillna(overall_mean)
                elif encoding == 'drop':
                    df_processed = df_processed.drop(columns=[col])
    
    # Phase 4: Transformations
    if 'transformations' in phase_results:
        transform_results = phase_results['transformations']
        recommendations = transform_results.get('llm_recommendations', {})
        
        # Apply user overrides if any
        if 'transformations' in user_overrides:
            for col, strategy in user_overrides['transformations'].items():
                if col in recommendations:
                    recommendations[col]['transformation'] = strategy
        
        for col, rec in recommendations.items():
            if col in df_processed.columns:
                transformation = rec.get('transformation', 'keep')
                
                if transformation == 'log_transform':
                    # Add small constant to avoid log(0)
                    df_processed[col] = np.log1p(df_processed[col] - df_processed[col].min() + 1)
                elif transformation == 'sqrt':
                    df_processed[col] = np.sqrt(df_processed[col] - df_processed[col].min())
                elif transformation == 'standardize':
                    scaler = StandardScaler()
                    df_processed[col] = scaler.fit_transform(df_processed[col].values.reshape(-1, 1))
                elif transformation == 'robust_scale':
                    scaler = RobustScaler()
                    df_processed[col] = scaler.fit_transform(df_processed[col].values.reshape(-1, 1))
                
                # 'keep' means no change
    
    return df_processed

def export_cleaned_dataset(state: SequentialState, output_path: str = None) -> str:
    """Export the cleaned dataset after applying all preprocessing"""
    import pandas as pd
    import os
    from datetime import datetime
    
    # Load original dataset
    df = pd.read_csv(state.df_path)
    
    # Apply preprocessing pipeline
    df_cleaned = apply_preprocessing_pipeline(df, state)
    
    # Generate output filename
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(state.df_path))[0]
        output_path = f"cleaned_{base_name}_{timestamp}.csv"
    
    # Save cleaned dataset
    df_cleaned.to_csv(output_path, index=False)
    
    # Calculate detailed preprocessing summary
    original_cols = len(df.columns)
    cleaned_cols = len(df_cleaned.columns)
    added_cols = cleaned_cols - original_cols
    
    # Count one-hot encoded columns
    onehot_count = 0
    if 'encoding' in state.phase_results:
        encoding_results = state.phase_results['encoding']
        if 'llm_recommendations' in encoding_results:
            recommendations = encoding_results['llm_recommendations']
            for col, rec in recommendations.items():
                strategy = rec.get('strategy', rec.get('encoding', ''))
                if 'onehot' in strategy:
                    # Count how many one-hot columns were created for this column
                    onehot_cols = [c for c in df_cleaned.columns if c.startswith(f'{col}_')]
                    onehot_count += len(onehot_cols)
    
    # Generate detailed summary report
    summary = f"""📊 **Dataset Export Summary**

📁 **File:** {output_path}
📈 **Original Shape:** {df.shape[0]} rows, {df.shape[1]} columns
📊 **Cleaned Shape:** {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns
🎯 **Target Column:** {state.target_column}

**Column Changes:**
• Original columns: {original_cols}
• Final columns: {cleaned_cols}
• Columns added: {added_cols} (including {onehot_count} one-hot encoded)

**Preprocessing Applied:**
• Outliers: {'✓' if 'outliers' in state.completed_phases else '✗'}
• Missing Values: {'✓' if 'missing_values' in state.completed_phases else '✗'}
• Encoding: {'✓' if 'encoding' in state.completed_phases else '✗'}
• Transformations: {'✓' if 'transformations' in state.completed_phases else '✗'}

✅ **Dataset ready for model training!**"""
    
    return output_path, summary

def get_current_data_state(state: SequentialState) -> pd.DataFrame:
    """
    Get the current state of data after applying all completed phases
    This ensures each phase analyzes data with previous treatments applied
    """
    df = state.df.copy()
    
    # Apply outliers treatment if completed
    if 'outliers' in state.phase_results:
        outliers_results = state.phase_results['outliers']
        if 'llm_recommendations' in outliers_results:
            df = apply_outliers_treatment(df, outliers_results['llm_recommendations'])
    
    # Apply missing values treatment if completed
    if 'missing_values' in state.phase_results:
        missing_results = state.phase_results['missing_values']
        if 'llm_recommendations' in missing_results:
            df = apply_missing_values_treatment(df, missing_results['llm_recommendations'])
    
    # Apply encoding treatment if completed
    if 'encoding' in state.phase_results:
        encoding_results = state.phase_results['encoding']
        if 'llm_recommendations' in encoding_results:
            df = apply_encoding_treatment(df, encoding_results['llm_recommendations'])
    
    # Apply transformations treatment if completed
    if 'transformations' in state.phase_results:
        transformation_results = state.phase_results['transformations']
        if 'llm_recommendations' in transformation_results:
            df = apply_transformations_treatment(df, transformation_results['llm_recommendations'])
    
    return df

def apply_outliers_treatment(df: pd.DataFrame, recommendations: Dict[str, Any]) -> pd.DataFrame:
    """Apply outlier treatments to dataframe"""
    df_processed = df.copy()
    
    for col, rec in recommendations.items():
        if col in df_processed.columns and isinstance(rec, dict):
            treatment = rec.get('treatment', 'keep')
            
            if treatment == 'winsorize':
                # Winsorize at 5th and 95th percentiles
                q05 = df_processed[col].quantile(0.05)
                q95 = df_processed[col].quantile(0.95)
                df_processed[col] = df_processed[col].clip(lower=q05, upper=q95)
            
            elif treatment == 'clip':
                # Clip at reasonable bounds based on data
                q01 = df_processed[col].quantile(0.01)
                q99 = df_processed[col].quantile(0.99)
                df_processed[col] = df_processed[col].clip(lower=q01, upper=q99)
            
            elif treatment == 'mark_missing':
                # Convert outliers to NaN
                q01 = df_processed[col].quantile(0.01)
                q99 = df_processed[col].quantile(0.99)
                # Fix: Use np.logical_or instead of | for boolean arrays
                outlier_mask = np.logical_or(df_processed[col] < q01, df_processed[col] > q99)
                df_processed.loc[outlier_mask, col] = np.nan
    
    return df_processed

def apply_missing_values_treatment(df: pd.DataFrame, recommendations: Dict[str, Any]) -> pd.DataFrame:
    """Apply missing value treatments to dataframe"""
    df_processed = df.copy()
    
    for col, rec in recommendations.items():
        if col in df_processed.columns and isinstance(rec, dict):
            strategy = rec.get('strategy', 'mean')
            
            if strategy == 'mean':
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            elif strategy == 'median':
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            elif strategy == 'mode':
                mode_value = df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else 0
                df_processed[col] = df_processed[col].fillna(mode_value)
            elif strategy == 'drop':
                df_processed = df_processed.drop(columns=[col])
    
    return df_processed

def apply_encoding_treatment(df: pd.DataFrame, recommendations: Dict[str, Any]) -> pd.DataFrame:
    """Apply encoding treatments to dataframe"""
    df_processed = df.copy()
    
    for col, rec in recommendations.items():
        if col in df_processed.columns and isinstance(rec, dict):
            # Fix: Look for 'strategy' key (what LLM outputs) instead of 'encoding'
            strategy = rec.get('strategy', 'label_encoding')
            
            # Map strategy values to encoding types
            if strategy == 'onehot_encoding' or strategy == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df_processed[col], prefix=col)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed.drop(columns=[col], inplace=True)
            
            elif strategy == 'label_encoding' or strategy == 'label':
                # Label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            
            elif strategy == 'target_encoding' or strategy == 'target':
                # Target encoding (simplified)
                target_col = [c for c in df_processed.columns if c != col][0]  # Assume first other column is target
                if target_col in df_processed.columns:
                    target_means = df_processed.groupby(col)[target_col].mean()
                    df_processed[col] = df_processed[col].map(target_means)
            
            elif strategy == 'drop_column' or strategy == 'drop':
                df_processed.drop(columns=[col], inplace=True)
    
    return df_processed

def apply_transformations_treatment(df: pd.DataFrame, recommendations: Dict[str, Any]) -> pd.DataFrame:
    """Apply transformation treatments to dataframe"""
    df_processed = df.copy()
    
    for col, rec in recommendations.items():
        if col in df_processed.columns and isinstance(rec, dict):
            # Look for 'transformation' key (what LLM outputs)
            transformation = rec.get('transformation', 'standardize')
            
            if transformation == 'log1p':
                # Log transformation (log1p for safety with zeros)
                df_processed[col] = np.log1p(df_processed[col] - df_processed[col].min() + 1)
            
            elif transformation == 'sqrt':
                # Square root transformation
                df_processed[col] = np.sqrt(df_processed[col] - df_processed[col].min() + 1)
            
            elif transformation == 'square':
                # Square transformation
                df_processed[col] = df_processed[col] ** 2
            
            elif transformation == 'standardize':
                # Standardization (z-score)
                mean_val = df_processed[col].mean()
                std_val = df_processed[col].std()
                if std_val > 0:
                    df_processed[col] = (df_processed[col] - mean_val) / std_val
            
            elif transformation == 'normalize':
                # Min-max normalization
                min_val = df_processed[col].min()
                max_val = df_processed[col].max()
                if max_val > min_val:
                    df_processed[col] = (df_processed[col] - min_val) / (max_val - min_val)
            
            elif transformation == 'robust_scale':
                # Robust scaling using median and IQR
                median_val = df_processed[col].median()
                q75 = df_processed[col].quantile(0.75)
                q25 = df_processed[col].quantile(0.25)
                iqr = q75 - q25
                if iqr > 0:
                    df_processed[col] = (df_processed[col] - median_val) / iqr
    
    return df_processed

def analyze_encoding_chunked(state: SequentialState, current_df: pd.DataFrame, chunk_size: int = 10, progress_callback=None) -> Dict[str, Any]:
    """
    Chunked encoding analysis for large datasets to avoid token limits
    Processes categorical columns in batches and combines results
    """
    import time
    
    # Get categorical columns
    categorical_columns = []
    high_cardinality_columns = []
    
    for col in current_df.columns:
        if col != state.target_column and current_df[col].dtype == 'object':
            unique_count = current_df[col].nunique()
            categorical_columns.append(col)
            
            # Detect high cardinality (more than 50 unique values)
            if unique_count > 50:
                high_cardinality_columns.append({
                    'column': col,
                    'unique_count': unique_count,
                    'unique_ratio': unique_count / len(current_df)
                })
    
    if not categorical_columns:
        return {'categorical_columns': [], 'llm_recommendations': {}}
    
    print(f"📊 Processing {len(categorical_columns)} categorical columns in chunks of {chunk_size}")
    
    # Process in chunks
    all_recommendations = {}
    start_time = time.time()
    total_chunks = (len(categorical_columns) + chunk_size - 1) // chunk_size
    
    # Send initial loading message if callback provided
    if progress_callback:
        progress_callback("encoding", 0, total_chunks, 0, len(categorical_columns), start_time, None, is_initial=True)
    
    for i in range(0, len(categorical_columns), chunk_size):
        chunk = categorical_columns[i:i + chunk_size]
        chunk_num = (i // chunk_size) + 1
        
        # Send progress update if callback provided
        if progress_callback:
            current_columns = len(all_recommendations) + len(chunk)
            progress_callback("encoding", chunk_num, total_chunks, current_columns, len(categorical_columns), start_time, None, is_initial=False)
        
        print(f"📊 Processing encoding chunk {chunk_num}/{total_chunks} ({len(chunk)} columns)")
        
        try:
            chunk_recommendations = analyze_encoding_chunk(state, chunk, current_df)
            all_recommendations.update(chunk_recommendations)
                
        except Exception as e:
            print(f"❌ Error processing encoding chunk {chunk_num}: {e}")
            # Fallback for this chunk
            for col in chunk:
                unique_count = current_df[col].nunique()
                
                if unique_count > 1000:
                    strategy = "drop_column"
                elif unique_count > 50:
                    strategy = "target_encoding"
                elif unique_count > 10:
                    strategy = "onehot_encoding"
                else:
                    strategy = "label_encoding"
                    
                all_recommendations[col] = {
                    "strategy": strategy,
                    "reasoning": f"Fallback rule for encoding chunk {chunk_num}",
                    "cardinality_level": "high" if unique_count > 50 else "medium" if unique_count > 10 else "low",
                    "target_relationship": "moderate"
                }
    
    total_time = time.time() - start_time
    print(f"✅ Completed encoding analysis for {len(all_recommendations)} columns in {total_time:.2f}s")
    
    return {
        'categorical_columns': list(all_recommendations.keys()),
        'high_cardinality_columns': [h['column'] for h in high_cardinality_columns],
        'llm_recommendations': all_recommendations
    }

def analyze_encoding_chunk(state: SequentialState, chunk: list, current_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze a single chunk of categorical columns for encoding"""
    llm = get_llm_from_state(state)
    
    # Create optimized prompt for this chunk
    prompt = f"""Analyze encoding for {len(chunk)} categorical columns. Target: {state.target_column}

STRATEGIES: label_encoding, onehot_encoding, target_encoding, binary_encoding, drop_column

COLUMNS:"""
    
    for col in chunk:
        unique_count = current_df[col].nunique()
        unique_ratio = unique_count / len(current_df)
        sample_values = current_df[col].dropna().head(3).tolist()
        
        prompt += f"""

{col}:
- Unique: {unique_count} ({unique_ratio:.1%})
- Sample: {sample_values}"""
    
    prompt += f"""

Return JSON: {{"column_name": {{"strategy": "strategy_name", "reasoning": "brief reason", "cardinality_level": "low/medium/high"}}}}"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            recommendations = json.loads(json_match.group())
        else:
            recommendations = {}
        
        return recommendations
        
    except Exception as e:
        print(f"Error in encoding chunk analysis: {e}")
        return {}

class ConfidenceBasedPreprocessor:
    """
    2-Step confidence-based preprocessing with timeout fallback:
    STEP 1: Rule-based pre-decisions for all columns
    STEP 2: LLM chunk processing for uncertain columns only (with 2-minute timeout)
    """
    
    def __init__(self, confidence_threshold: float = 0.8, timeout_minutes: int = 2):
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_minutes * 60
        self.stats = {
            'rule_based_decisions': 0,
            'llm_decisions': 0,
            'timeout_fallbacks': 0,
            'total_columns': 0
        }
    
    def analyze_phase_with_confidence(self, state: SequentialState, phase: str) -> Dict[str, Any]:
        """
        Main entry point for confidence-based analysis with timeout fallback
        """
        start_time = time.time()
        print(f"🎯 Starting 2-step confidence analysis for {phase} (timeout: {self.timeout_seconds/60:.0f}min)...")
        
        # STEP 1: Rule-based pre-analysis (always fast)
        rule_results = self._analyze_columns_with_rules(state, phase)
        high_conf_decisions = rule_results['high_confidence']
        uncertain_columns = rule_results['uncertain_columns']
        
        # STEP 1 LOGS
        total_cols = len(high_conf_decisions) + len(uncertain_columns)
        high_conf_pct = (len(high_conf_decisions) / total_cols * 100) if total_cols > 0 else 0
        
        print(f"📊 STEP 1 PRE-DECISION RESULTS:")
        print(f"   ✅ High Confidence: {len(high_conf_decisions)}/{total_cols} columns ({high_conf_pct:.1f}%)")
        print(f"   ❓ Uncertain: {len(uncertain_columns)}/{total_cols} columns ({100-high_conf_pct:.1f}%)")
        
        # STEP 2: LLM processing with timeout monitoring
        llm_decisions = {}
        if uncertain_columns:
            elapsed = time.time() - start_time
            remaining_time = self.timeout_seconds - elapsed
            
            if remaining_time > 30:  # At least 30 seconds left
                chunk_size = 12  # Optimized for uncertain cases
                num_chunks = (len(uncertain_columns) + chunk_size - 1) // chunk_size
                
                print(f"🤖 STEP 2 LLM PROCESSING:")
                print(f"   📦 Chunking {len(uncertain_columns)} uncertain columns into {num_chunks} chunks")
                print(f"   ⏰ Timeout in {remaining_time:.0f}s")
                
                llm_decisions = self._process_with_timeout(state, phase, uncertain_columns, remaining_time)
            else:
                print(f"⚠️ Insufficient time remaining ({remaining_time:.0f}s), using fallback strategies...")
                llm_decisions = self._apply_fallback_strategies(uncertain_columns, phase)
                self.stats['timeout_fallbacks'] += len(uncertain_columns)
        
        # Update stats - track what actually happened
        self.stats['rule_based_decisions'] += len(high_conf_decisions)
        
        # Count LLM decisions vs timeout fallbacks
        current_timeout_fallbacks = self.stats.get('timeout_fallbacks', 0)
        llm_processed = len(llm_decisions) - current_timeout_fallbacks
        if llm_processed > 0:
            self.stats['llm_decisions'] += llm_processed
            
        self.stats['total_columns'] += total_cols
        
        # Performance summary
        total_time = time.time() - start_time
        print(f"⚡ PERFORMANCE SUMMARY:")
        print(f"   🚀 Rule-based efficiency: {high_conf_pct:.1f}%")
        print(f"   ⏱️  Total processing time: {total_time:.1f}s")
        print(f"   🎯 LLM calls saved: ~{(total_cols//10) - (len(uncertain_columns)//12 if uncertain_columns else 0)}")
        
        # Combine results
        all_recommendations = {**high_conf_decisions, **llm_decisions}
        
        return {
            f'{phase}_columns': list(all_recommendations.keys()),
            'llm_recommendations': all_recommendations,
            'confidence_stats': self.stats.copy(),
            'processing_time': total_time
        }
    
    def _analyze_columns_with_rules(self, state: SequentialState, phase: str) -> Dict[str, Any]:
        """STEP 1: Apply rule-based confidence scoring to all columns"""
        
        if phase == 'outliers':
            return self._analyze_outliers_rules(state)
        elif phase == 'missing_values':
            return self._analyze_missing_values_rules(state)
        elif phase == 'encoding':
            return self._analyze_encoding_rules(state)
        elif phase == 'transformations':
            return self._analyze_transformations_rules(state)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def _analyze_outliers_rules(self, state: SequentialState) -> Dict[str, Any]:
        """Rule-based confidence analysis for outliers"""
        df = state.df
        target = df[state.target_column] if state.target_column in df.columns else None
        
        high_confidence = {}
        uncertain_columns = []
        
        print("🔍 STEP 1: Analyzing outliers with confidence rules...")
        
        for col in df.columns:
            if col == state.target_column or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Pre-analysis: Check if column actually has outliers
            analysis = analyze_column_comprehensive(df[col], target, col)
            outlier_pct = analysis.get('outliers_iqr_percentage', 0)
            
            # ✅ CRITICAL FIX: Only process columns WITH outliers
            if outlier_pct > 0:
                print(f"🎯 {col}: {outlier_pct:.1f}% outliers detected - analyzing...")
                confidence_result = self._outlier_confidence_rules(analysis, col)
                
                if confidence_result['confidence'] >= self.confidence_threshold:
                    high_confidence[col] = {
                        'treatment': confidence_result['strategy'],
                        'reasoning': confidence_result['reasoning'],
                        'confidence': confidence_result['confidence'],
                        'rule_based': True
                    }
                    print(f"✅ {col}: {confidence_result['strategy']} (conf: {confidence_result['confidence']:.2f}) - RULE DECISION")
                else:
                    uncertain_columns.append({
                        'column': col,
                        'analysis': analysis,
                        'patterns': detect_patterns_llm_ready(df[col], col),
                        'rule_confidence': confidence_result['confidence']
                    })
                    print(f"❓ {col}: uncertain (conf: {confidence_result['confidence']:.2f}) - NEEDS LLM")
            else:
                print(f"⚪ {col}: No outliers ({outlier_pct:.1f}%) - skipping")
        
        return {'high_confidence': high_confidence, 'uncertain_columns': uncertain_columns}
    
    def _outlier_confidence_rules(self, analysis: Dict, column: str) -> Dict[str, Any]:
        """High-confidence rules for outlier treatment"""
        outlier_pct = analysis.get('outliers_iqr_percentage', 0)
        extreme_pct = analysis.get('extreme_outliers_percentage', 0)
        
        # Rule 1: No outliers → Keep (Very High Confidence)
        if outlier_pct == 0:
            return {
                'strategy': 'keep',
                'confidence': 0.95,
                'reasoning': 'No outliers detected using IQR method'
            }
        
        # Rule 2: Extreme outliers (>5% extreme) → Winsorize (High Confidence)
        if extreme_pct > 5.0:
            return {
                'strategy': 'winsorize',
                'confidence': 0.90,
                'reasoning': f'High extreme outlier rate ({extreme_pct:.1f}%) indicates data quality issues'
            }
        
        # Rule 3: High outliers → Winsorize (High Confidence)
        if outlier_pct > 15.0:
            return {
                'strategy': 'winsorize',
                'confidence': 0.90,
                'reasoning': f'Very high outlier rate ({outlier_pct:.1f}%) suggests measurement errors'
            }
        
        # Rule 4: Very low outliers (<2%) → Keep (High Confidence)
        if outlier_pct < 2.0:
            return {
                'strategy': 'keep',
                'confidence': 0.85,
                'reasoning': f'Low outlier rate ({outlier_pct:.1f}%) - likely legitimate values'
            }
        
        # Rule 5: Financial/ID-like columns → Keep (Medium-High Confidence)
        if any(keyword in column.lower() for keyword in ['id', 'account', 'balance', 'amount', 'price', 'income', 'revenue', 'cost']):
            return {
                'strategy': 'keep',
                'confidence': 0.80,
                'reasoning': f'Financial/ID column - outliers likely legitimate extreme values'
            }
        
        # Gray area - needs LLM
        return {
            'strategy': 'uncertain',
            'confidence': 0.5,
            'reasoning': f'Moderate outlier rate ({outlier_pct:.1f}%) requires contextual analysis'
        }
    
    def _analyze_missing_values_rules(self, state: SequentialState) -> Dict[str, Any]:
        """Rule-based confidence analysis for missing values"""
        current_df = get_current_data_state(state)
        target = current_df[state.target_column] if state.target_column in current_df.columns else None
        
        high_confidence = {}
        uncertain_columns = []
        
        print("🔍 STEP 1: Analyzing missing values with confidence rules...")
        
        for col in current_df.columns:
            if current_df[col].isnull().sum() > 0:  # Process columns WITH missing values
                analysis = analyze_column_comprehensive(current_df[col], target, col)
                confidence_result = self._missing_values_confidence_rules(analysis, col)
                
                if confidence_result['confidence'] >= self.confidence_threshold:
                    high_confidence[col] = {
                        'strategy': confidence_result['strategy'],
                        'reasoning': confidence_result['reasoning'],
                        'confidence': confidence_result['confidence'],
                        'rule_based': True
                    }
                    print(f"✅ {col}: {confidence_result['strategy']} (conf: {confidence_result['confidence']:.2f}) - RULE DECISION")
                else:
                    uncertain_columns.append({
                        'column': col,
                        'analysis': analysis,
                        'patterns': detect_patterns_llm_ready(current_df[col], col),
                        'rule_confidence': confidence_result['confidence']
                    })
                    print(f"❓ {col}: uncertain (conf: {confidence_result['confidence']:.2f}) - NEEDS LLM")
            else:
                print(f"⚪ {col}: No missing values - skipping")
        
        return {'high_confidence': high_confidence, 'uncertain_columns': uncertain_columns}
    
    def _missing_values_confidence_rules(self, analysis: Dict, column: str) -> Dict[str, Any]:
        """High-confidence rules for missing value treatment"""
        missing_pct = analysis.get('missing_percentage', 0)
        unique_ratio = analysis.get('unique_ratio', 0)
        is_numeric = analysis.get('dtype', '').startswith(('int', 'float'))
        skewness = abs(analysis.get('skewness', 0)) if is_numeric else 0
        
        # Rule 1: Very high missing rate (>80%) → Drop column (High Confidence)
        if missing_pct > 80.0:
            return {
                'strategy': 'drop_column',
                'confidence': 0.95,
                'reasoning': f'Very high missing rate ({missing_pct:.1f}%) - insufficient data'
            }
        
        # Rule 2: ID columns → Drop missing (Very High Confidence)
        if any(keyword in column.lower() for keyword in ['id', 'key', 'uuid', 'guid']):
            return {
                'strategy': 'drop_missing',
                'confidence': 0.95,
                'reasoning': 'ID column - missing values should be excluded'
            }
        
        # Rule 3: Low missing rate (<5%) + Numeric + Low skewness → Mean (High Confidence)
        if missing_pct < 5.0 and is_numeric and skewness < 1.0:
            return {
                'strategy': 'mean',
                'confidence': 0.85,
                'reasoning': f'Low missing rate ({missing_pct:.1f}%) with normal distribution'
            }
        
        # Rule 4: Low missing rate (<5%) + Numeric + High skewness → Median (High Confidence)
        if missing_pct < 5.0 and is_numeric and skewness >= 1.0:
            return {
                'strategy': 'median',
                'confidence': 0.85,
                'reasoning': f'Low missing rate ({missing_pct:.1f}%) with skewed distribution'
            }
        
        # Rule 5: Categorical + Low missing rate (<10%) + Low cardinality → Mode (High Confidence)
        if not is_numeric and missing_pct < 10.0 and unique_ratio < 0.1:
            return {
                'strategy': 'mode',
                'confidence': 0.80,
                'reasoning': f'Low missing rate ({missing_pct:.1f}%) categorical with low cardinality'
            }
        
        # Rule 6: High cardinality categorical (likely IDs) → Drop missing (High Confidence)
        if not is_numeric and unique_ratio > 0.8:
            return {
                'strategy': 'drop_missing',
                'confidence': 0.85,
                'reasoning': f'High cardinality ({unique_ratio:.2f}) - likely unique identifiers'
            }
        
        # Gray area - needs LLM
        return {
            'strategy': 'uncertain',
            'confidence': 0.5,
            'reasoning': f'Missing rate ({missing_pct:.1f}%) requires contextual analysis'
        }

    def _process_with_timeout(self, state: SequentialState, phase: str, uncertain_columns: List[Dict], timeout_seconds: float) -> Dict[str, Any]:
        """Process uncertain columns with timeout fallback - FIXED VERSION"""
        
        # ✅ CRITICAL FIX: Pre-check timeout before starting LLM calls
        # Only use immediate fallback if we have very little time (less than 30 seconds for large datasets)
        min_time_needed = min(30, len(uncertain_columns) * 2)  # 2 seconds per column, max 30s
        if timeout_seconds <= min_time_needed:
            print(f"⏰ TIMEOUT PREVENTION: Only {timeout_seconds:.0f}s remaining (need {min_time_needed}s) - using fallback strategies...")
            print(f"🔄 Applying conservative strategies for {len(uncertain_columns)} columns...")
            
            fallback_result = self._apply_fallback_strategies(uncertain_columns, phase)
            self.stats['timeout_fallbacks'] += len(uncertain_columns)
            return fallback_result
        
        def llm_processing_task():
            """The actual LLM processing task with per-chunk timeout checking"""
            return self._process_uncertain_columns_with_llm_safe(state, phase, uncertain_columns, timeout_seconds)
        
        try:
            # Use ThreadPoolExecutor with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(llm_processing_task)
                
                print(f"🤖 Starting LLM processing with {timeout_seconds:.0f}s timeout...")
                result = future.result(timeout=timeout_seconds)
                
                print(f"✅ LLM processing completed within timeout")
                return result
                
        except TimeoutError:
            print(f"⏰ TIMEOUT: LLM processing exceeded {timeout_seconds:.0f}s limit")
            print(f"🔄 Falling back to conservative strategies for {len(uncertain_columns)} columns...")
            
            fallback_result = self._apply_fallback_strategies(uncertain_columns, phase)
            self.stats['timeout_fallbacks'] += len(uncertain_columns)
            return fallback_result
        
        except Exception as e:
            print(f"❌ LLM processing failed: {e}")
            print(f"🔄 Using fallback strategies...")
            return self._apply_fallback_strategies(uncertain_columns, phase)
    
    def _process_uncertain_columns_with_llm(self, state: SequentialState, phase: str, uncertain_columns: List[Dict]) -> Dict[str, Any]:
        """Process uncertain columns with LLM in optimized chunks"""
        if not uncertain_columns:
            return {}
        
        chunk_size = 12  # Optimized for uncertain cases
        all_recommendations = {}
        
        for i in range(0, len(uncertain_columns), chunk_size):
            chunk = uncertain_columns[i:i+chunk_size]
            chunk_num = (i // chunk_size) + 1
            total_chunks = (len(uncertain_columns) + chunk_size - 1) // chunk_size
            
            print(f"🔧 Processing chunk {chunk_num}/{total_chunks}: {len(chunk)} columns")
            print(f"   Columns: {[col_info['column'] for col_info in chunk]}")
            
            start_time = time.time()
            
            try:
                if phase == 'outliers':
                    chunk_results = analyze_outlier_chunk(state, chunk)
                elif phase == 'missing_values':
                    current_df = get_current_data_state(state)
                    chunk_results = analyze_missing_values_chunk(state, chunk, current_df)
                elif phase == 'encoding':
                    chunk_results = self._process_encoding_chunk_with_llm(state, chunk)
                elif phase == 'transformations':
                    chunk_results = self._process_transformations_chunk_with_llm(state, chunk)
                else:
                    chunk_results = {}
                
                processing_time = time.time() - start_time
                print(f"   ✅ Chunk {chunk_num} completed in {processing_time:.1f}s")
                
                # Handle different return formats for different phases
                if phase == 'missing_values':
                    # analyze_missing_values_chunk returns recommendations directly
                    all_recommendations.update(chunk_results)
                else:
                    # Other phases return wrapped in 'llm_recommendations'
                    all_recommendations.update(chunk_results.get('llm_recommendations', {}))
                
            except Exception as e:
                print(f"   ❌ Chunk {chunk_num} failed: {e}")
                # Apply fallback for this chunk
                chunk_fallback = self._apply_fallback_strategies(chunk, phase)
                all_recommendations.update(chunk_fallback)
        
        return all_recommendations
    
    def _process_uncertain_columns_with_llm_safe(self, state: SequentialState, phase: str, uncertain_columns: List[Dict], total_timeout: float) -> Dict[str, Any]:
        """Process uncertain columns with LLM in optimized chunks - TIMEOUT SAFE VERSION"""
        if not uncertain_columns:
            return {}
        
        chunk_size = 12  # Optimized for uncertain cases
        all_recommendations = {}
        start_total_time = time.time()
        
        for i in range(0, len(uncertain_columns), chunk_size):
            # ✅ CRITICAL: Check timeout before each chunk
            elapsed_time = time.time() - start_total_time
            remaining_time = total_timeout - elapsed_time
            
            # Need enough time for at least one more chunk (estimate 10s per chunk)
            if remaining_time <= 10:  
                print(f"⏰ CHUNK TIMEOUT: Only {remaining_time:.1f}s remaining - stopping LLM processing")
                print(f"🔄 Processed {i}/{len(uncertain_columns)} columns, applying fallback for remaining...")
                
                # Apply fallback for remaining columns
                remaining_columns = uncertain_columns[i:]
                if remaining_columns:
                    fallback_result = self._apply_fallback_strategies(remaining_columns, phase)
                    all_recommendations.update(fallback_result)
                    self.stats['timeout_fallbacks'] += len(remaining_columns)
                break
            
            chunk = uncertain_columns[i:i+chunk_size]
            chunk_num = (i // chunk_size) + 1
            total_chunks = (len(uncertain_columns) + chunk_size - 1) // chunk_size
            
            print(f"🔧 Processing chunk {chunk_num}/{total_chunks}: {len(chunk)} columns (⏰ {remaining_time:.0f}s left)")
            print(f"   Columns: {[col_info['column'] for col_info in chunk]}")
            
            start_time = time.time()
            
            try:
                if phase == 'outliers':
                    chunk_results = analyze_outlier_chunk(state, chunk)
                elif phase == 'missing_values':
                    current_df = get_current_data_state(state)
                    chunk_results = analyze_missing_values_chunk(state, chunk, current_df)
                elif phase == 'encoding':
                    chunk_results = self._process_encoding_chunk_with_llm(state, chunk)
                elif phase == 'transformations':
                    chunk_results = self._process_transformations_chunk_with_llm(state, chunk)
                else:
                    chunk_results = {}
                
                processing_time = time.time() - start_time
                print(f"   ✅ Chunk {chunk_num} completed in {processing_time:.1f}s")
                
                # Handle different return formats for different phases
                if phase == 'missing_values':
                    # analyze_missing_values_chunk returns recommendations directly
                    all_recommendations.update(chunk_results)
                else:
                    # Other phases return wrapped in 'llm_recommendations'
                    all_recommendations.update(chunk_results.get('llm_recommendations', {}))
                
            except Exception as e:
                print(f"   ❌ Chunk {chunk_num} failed: {e}")
                print(f"   🔄 Applying fallback for this chunk...")
                
                # Apply fallback for this chunk only
                chunk_fallback = self._apply_fallback_strategies(chunk, phase)
                all_recommendations.update(chunk_fallback)
                self.stats['timeout_fallbacks'] += len(chunk)
                continue
        
        return all_recommendations
    
    def _process_encoding_chunk_with_llm(self, state: SequentialState, chunk: List[Dict]) -> Dict[str, Any]:
        """Process encoding chunk with LLM"""
        try:
            from langchain_core.messages import HumanMessage
            llm = get_llm_from_state(state)
            
            prompt = self._create_encoding_prompt(chunk, state.target_column)
            response = llm.invoke([HumanMessage(content=prompt)])
            
            return {'llm_recommendations': self._parse_encoding_response(response.content, chunk)}
            
        except Exception as e:
            print(f"❌ Encoding LLM processing failed: {e}")
            return {'llm_recommendations': self._apply_fallback_strategies(chunk, 'encoding')}
    
    def _process_transformations_chunk_with_llm(self, state: SequentialState, chunk: List[Dict]) -> Dict[str, Any]:
        """Process transformations chunk with LLM"""
        try:
            from langchain_core.messages import HumanMessage
            llm = get_llm_from_state(state)
            
            prompt = self._create_transformations_prompt(chunk, state.target_column)
            response = llm.invoke([HumanMessage(content=prompt)])
            
            return {'llm_recommendations': self._parse_transformations_response(response.content, chunk)}
            
        except Exception as e:
            print(f"❌ Transformations LLM processing failed: {e}")
            return {'llm_recommendations': self._apply_fallback_strategies(chunk, 'transformations')}
    
    def _create_encoding_prompt(self, chunk: List[Dict], target_column: str) -> str:
        """Create focused LLM prompt for encoding uncertain cases"""
        prompt = f"""Analyze categorical encoding strategies for {len(chunk)} UNCERTAIN columns.
Focus on cardinality, ordinality, and target relationship.

TARGET COLUMN: {target_column}

UNCERTAIN COLUMNS:
"""
        for col_info in chunk:
            analysis = col_info['analysis']
            prompt += f"""
Column: {col_info['column']}
- Unique values: {analysis.get('unique_count', 0)}
- Cardinality ratio: {analysis.get('unique_ratio', 0):.3f}
- Sample values: {analysis.get('sample_values', [])}
- Target correlation: {analysis.get('target_correlation', 0):.3f}
"""
        
        prompt += """
Provide encoding strategies in JSON format:
{
    "column_name": {
        "strategy": "label_encoding|onehot_encoding|target_encoding|binary_encoding",
        "reasoning": "Brief justification"
    }
}

GUIDELINES:
- label_encoding: Natural ordering exists
- onehot_encoding: No ordering, low-medium cardinality (<15)
- target_encoding: High cardinality or strong target relationship
- binary_encoding: Medium-high cardinality (10-50 categories)
"""
        return prompt
    
    def _create_transformations_prompt(self, chunk: List[Dict], target_column: str) -> str:
        """Create focused LLM prompt for transformations uncertain cases"""
        prompt = f"""Analyze transformation strategies for {len(chunk)} UNCERTAIN numeric columns.
Focus on distribution characteristics and business context.

TARGET COLUMN: {target_column}

UNCERTAIN COLUMNS:
"""
        for col_info in chunk:
            analysis = col_info['analysis']
            prompt += f"""
Column: {col_info['column']}
- Skewness: {analysis.get('skewness', 0):.2f}
- Kurtosis: {analysis.get('kurtosis', 0):.2f}
- Min/Max: {analysis.get('percentile_1', 0):.2f} / {analysis.get('percentile_99', 0):.2f}
- Has zeros: {'Yes' if analysis.get('percentile_1', 1) <= 0 else 'No'}
- Has negatives: {'Yes' if analysis.get('percentile_1', 0) < 0 else 'No'}
"""
        
        prompt += """
Provide transformation strategies in JSON format:
{
    "column_name": {
        "strategy": "log|log1p|sqrt|box_cox|yeo_johnson|quantile|none",
        "reasoning": "Brief explanation"
    }
}

GUIDELINES:
- log: Right-skewed, positive values, no zeros
- log1p: Right-skewed, positive values, has zeros  
- sqrt: Moderate right-skew, count/rate data
- box_cox: Right-skewed, positive values, needs normality
- yeo_johnson: Any skewness, handles negatives and zeros
- quantile: Non-parametric, extreme distributions
- none: Already well-distributed
"""
        return prompt
    
    def _parse_encoding_response(self, response: str, chunk: List[Dict]) -> Dict[str, Any]:
        """Parse LLM response for encoding recommendations"""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group())
                return recommendations
            
        except Exception as e:
            print(f"❌ Failed to parse encoding response: {e}")
        
        # Fallback
        return {col_info['column']: {'strategy': 'onehot_encoding', 'reasoning': 'Parse error fallback'} 
                for col_info in chunk}
    
    def _parse_transformations_response(self, response: str, chunk: List[Dict]) -> Dict[str, Any]:
        """Parse LLM response for transformation recommendations"""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group())
                return recommendations
            
        except Exception as e:
            print(f"❌ Failed to parse transformation response: {e}")
        
        # Fallback based on skewness
        fallback = {}
        for col_info in chunk:
            col = col_info['column']
            skewness = abs(col_info['analysis'].get('skewness', 0))
            strategy = 'log1p' if skewness > 2.0 else 'none'
            fallback[col] = {'strategy': strategy, 'reasoning': 'Parse error, skewness-based fallback'}
        
        return fallback
    
    def _apply_fallback_strategies(self, uncertain_columns: List[Dict], phase: str) -> Dict[str, Any]:
        """Apply conservative fallback strategies when LLM fails or times out"""
        fallback_decisions = {}
        
        for col_info in uncertain_columns:
            col = col_info['column']
            analysis = col_info['analysis']
            
            if phase == 'outliers':
                outlier_pct = analysis.get('outliers_iqr_percentage', 0)
                if outlier_pct > 5.0:
                    strategy = 'winsorize'
                    reasoning = f'Timeout fallback: {outlier_pct:.1f}% outliers → conservative winsorization'
                else:
                    strategy = 'keep'
                    reasoning = f'Timeout fallback: {outlier_pct:.1f}% outliers → keep as legitimate values'
                fallback_decisions[col] = {'treatment': strategy, 'reasoning': reasoning, 'fallback_used': True}
                
            elif phase == 'missing_values':
                missing_pct = analysis.get('missing_percentage', 0)
                is_numeric = analysis.get('dtype', '').startswith(('int', 'float'))
                
                if missing_pct > 50:
                    strategy = 'drop_column'
                    reasoning = f'Timeout fallback: {missing_pct:.1f}% missing → drop column (safe choice)'
                elif is_numeric:
                    strategy = 'median'
                    reasoning = f'Timeout fallback: numeric column → median imputation (robust)'
                else:
                    strategy = 'mode'
                    reasoning = f'Timeout fallback: categorical column → mode imputation (safe)'
                fallback_decisions[col] = {'strategy': strategy, 'reasoning': reasoning, 'fallback_used': True}
                
            elif phase == 'encoding':
                unique_count = analysis.get('unique_count', 0)
                if unique_count <= 10:
                    strategy = 'onehot_encoding'
                    reasoning = f'Timeout fallback: {unique_count} categories → one-hot encoding (safe)'
                else:
                    strategy = 'target_encoding'
                    reasoning = f'Timeout fallback: {unique_count} categories → target encoding (dimensionality)'
                fallback_decisions[col] = {'strategy': strategy, 'reasoning': reasoning, 'fallback_used': True}
                
            elif phase == 'transformations':
                skewness = abs(analysis.get('skewness', 0))
                if skewness > 2.0:
                    strategy = 'log1p'
                    reasoning = f'Timeout fallback: skewness {skewness:.2f} → log1p transformation (safe)'
                else:
                    strategy = 'none'
                    reasoning = f'Timeout fallback: skewness {skewness:.2f} → no transformation (conservative)'
                fallback_decisions[col] = {'strategy': strategy, 'reasoning': reasoning, 'fallback_used': True}
        
        return fallback_decisions

    def _analyze_encoding_rules(self, state: SequentialState) -> Dict[str, Any]:
        """Rule-based confidence analysis for encoding"""
        current_df = get_current_data_state(state)
        
        high_confidence = {}
        uncertain_columns = []
        
        print("🔍 STEP 1: Analyzing encoding with confidence rules...")
        
        # Get categorical columns
        for col in current_df.columns:
            if col != state.target_column and (current_df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(current_df[col])):
                analysis = analyze_column_comprehensive(current_df[col], 
                                                     current_df[state.target_column] if state.target_column in current_df.columns else None, 
                                                     col)
                confidence_result = self._encoding_confidence_rules(analysis, col, current_df)
                
                if confidence_result['confidence'] >= self.confidence_threshold:
                    high_confidence[col] = {
                        'strategy': confidence_result['strategy'],
                        'reasoning': confidence_result['reasoning'],
                        'confidence': confidence_result['confidence'],
                        'rule_based': True
                    }
                    print(f"✅ {col}: {confidence_result['strategy']} (conf: {confidence_result['confidence']:.2f}) - RULE DECISION")
                else:
                    uncertain_columns.append({
                        'column': col,
                        'analysis': analysis,
                        'patterns': detect_patterns_llm_ready(current_df[col], col),
                        'rule_confidence': confidence_result['confidence']
                    })
                    print(f"❓ {col}: uncertain (conf: {confidence_result['confidence']:.2f}) - NEEDS LLM")
        
        return {'high_confidence': high_confidence, 'uncertain_columns': uncertain_columns}
    
    def _encoding_confidence_rules(self, analysis: Dict, column: str, df: pd.DataFrame) -> Dict[str, Any]:
        """High-confidence rules for categorical encoding"""
        unique_count = analysis.get('unique_count', 0)
        unique_ratio = analysis.get('unique_ratio', 0)
        
        # Rule 1: Binary categorical (2 unique values) → Label encoding (Very High Confidence)
        if unique_count == 2:
            return {
                'strategy': 'label_encoding',
                'confidence': 0.95,
                'reasoning': f'Binary categorical with {unique_count} unique values'
            }
        
        # Rule 2: Very low cardinality (≤3 unique values) → Label encoding (High Confidence)
        if unique_count <= 3:
            return {
                'strategy': 'label_encoding',
                'confidence': 0.90,
                'reasoning': f'Low cardinality with {unique_count} unique values'
            }
        
        # Rule 3: Very high cardinality (>50% unique) → Target encoding (High Confidence)
        if unique_ratio > 0.5:
            return {
                'strategy': 'target_encoding',
                'confidence': 0.85,
                'reasoning': f'High cardinality ({unique_count} unique, {unique_ratio:.2f} ratio)'
            }
        
        # Rule 4: Geographic columns → Target encoding (Medium-High Confidence)
        if any(keyword in column.lower() for keyword in ['city', 'state', 'country', 'region', 'location', 'zip', 'postal']):
            return {
                'strategy': 'target_encoding',
                'confidence': 0.80,
                'reasoning': 'Geographic column - target encoding captures regional patterns'
            }
        
        # Rule 5: Ordinal-like patterns → Label encoding (Medium-High Confidence)
        if self._detect_ordinal_pattern(df[column]):
            return {
                'strategy': 'label_encoding',
                'confidence': 0.80,
                'reasoning': 'Detected ordinal pattern in categorical values'
            }
        
        # Gray area - needs LLM
        return {
            'strategy': 'uncertain',
            'confidence': 0.5,
            'reasoning': f'Moderate cardinality ({unique_count} unique) requires contextual analysis'
        }
    
    def _detect_ordinal_pattern(self, series: pd.Series) -> bool:
        """Detect if categorical values have ordinal pattern"""
        unique_values = series.dropna().unique()
        if len(unique_values) < 3:
            return False
        
        # Check for size patterns
        size_patterns = ['small', 'medium', 'large', 'xs', 's', 'm', 'l', 'xl', 'low', 'high']
        if any(str(val).lower() in size_patterns for val in unique_values):
            return True
        
        # Check for rating patterns
        rating_patterns = ['poor', 'fair', 'good', 'excellent', 'bad', 'average', 'great']
        if any(str(val).lower() in rating_patterns for val in unique_values):
            return True
        
        # Check for numeric-like strings
        try:
            numeric_values = [float(val) for val in unique_values if str(val).replace('.', '').replace('-', '').isdigit()]
            if len(numeric_values) == len(unique_values):
                return True
        except:
            pass
        
        return False
    
    def _analyze_transformations_rules(self, state: SequentialState) -> Dict[str, Any]:
        """Rule-based confidence analysis for transformations"""
        current_df = get_current_data_state(state)
        target = current_df[state.target_column] if state.target_column in current_df.columns else None
        
        high_confidence = {}
        uncertain_columns = []
        
        print("🔍 STEP 1: Analyzing transformations with confidence rules...")
        
        for col in current_df.columns:
            if col == state.target_column or not pd.api.types.is_numeric_dtype(current_df[col]):
                continue
                
            analysis = analyze_column_comprehensive(current_df[col], target, col)
            
            # Only consider columns that might benefit from transformation
            skewness = abs(analysis.get('skewness', 0))
            kurtosis = abs(analysis.get('kurtosis', 0))
            is_normal = analysis.get('is_likely_normal', False)
            
            if skewness < 1.0 and kurtosis < 5.0 and is_normal:
                continue  # Skip normal distributions
            
            confidence_result = self._transformation_confidence_rules(analysis, col)
            
            if confidence_result['confidence'] >= self.confidence_threshold:
                high_confidence[col] = {
                    'strategy': confidence_result['strategy'],
                    'reasoning': confidence_result['reasoning'],
                    'confidence': confidence_result['confidence'],
                    'rule_based': True
                }
                print(f"✅ {col}: {confidence_result['strategy']} (conf: {confidence_result['confidence']:.2f}) - RULE DECISION")
            else:
                uncertain_columns.append({
                    'column': col,
                    'analysis': analysis,
                    'rule_confidence': confidence_result['confidence']
                })
                print(f"❓ {col}: uncertain (conf: {confidence_result['confidence']:.2f}) - NEEDS LLM")
        
        return {'high_confidence': high_confidence, 'uncertain_columns': uncertain_columns}
    
    def _transformation_confidence_rules(self, analysis: Dict, column: str) -> Dict[str, Any]:
        """High-confidence rules for transformation selection"""
        skewness = abs(analysis.get('skewness', 0))
        kurtosis = abs(analysis.get('kurtosis', 0))
        has_negative = analysis.get('percentile_1', 0) < 0
        has_zero = analysis.get('percentile_1', 1) <= 0
        
        # Rule 1: Near-normal distribution → No transformation (High Confidence)
        if skewness < 1.5 and kurtosis < 7.0:
            return {
                'strategy': 'none',
                'confidence': 0.90,
                'reasoning': f'Near-normal distribution (skew: {skewness:.2f}, kurtosis: {kurtosis:.2f})'
            }
        
        # Rule 2: Extreme skewness (>5) → Yeo-Johnson (High Confidence)
        if skewness > 5.0:
            return {
                'strategy': 'yeo_johnson',
                'confidence': 0.90,
                'reasoning': f'Extreme skewness ({skewness:.2f}) requires robust transformation'
            }
        
        # Rule 3: Moderate right skew + no zeros/negatives → Log transform (High Confidence)
        if 1.0 < skewness < 3.0 and not has_zero and not has_negative:
            return {
                'strategy': 'log',
                'confidence': 0.85,
                'reasoning': f'Moderate right skew ({skewness:.2f}) with positive values'
            }
        
        # Rule 4: High right skew + has zeros → Log1p transform (High Confidence)
        if skewness > 2.0 and has_zero and not has_negative:
            return {
                'strategy': 'log1p',
                'confidence': 0.85,
                'reasoning': f'High right skew ({skewness:.2f}) with zero values'
            }
        
        # Rule 5: Financial/measurement columns + moderate skew → Square root (Medium-High Confidence)
        if any(keyword in column.lower() for keyword in ['amount', 'price', 'income', 'balance', 'cost', 'revenue']):
            if 1.0 < skewness < 4.0 and not has_negative:
                return {
                    'strategy': 'sqrt',
                    'confidence': 0.80,
                    'reasoning': f'Financial column with moderate skew ({skewness:.2f})'
                }
        
        # Gray area - needs LLM
        return {
            'strategy': 'uncertain',
            'confidence': 0.5,
            'reasoning': f'Complex distribution (skew: {skewness:.2f}, kurtosis: {kurtosis:.2f}) requires analysis'
        }

# Integration functions to replace existing LLM-only analysis
def analyze_outliers_with_confidence(state: SequentialState, progress_callback=None) -> Dict[str, Any]:
    """Replace analyze_outliers_with_llm with confidence-based approach"""
    processor = ConfidenceBasedPreprocessor()
    return processor.analyze_phase_with_confidence(state, 'outliers')

def analyze_missing_values_with_confidence(state: SequentialState, progress_callback=None) -> Dict[str, Any]:
    """Replace analyze_missing_values_with_llm with confidence-based approach"""
    processor = ConfidenceBasedPreprocessor()
    return processor.analyze_phase_with_confidence(state, 'missing_values')

def analyze_encoding_with_confidence(state: SequentialState, progress_callback=None) -> Dict[str, Any]:
    """Replace analyze_encoding_with_llm with confidence-based approach"""
    processor = ConfidenceBasedPreprocessor()
    return processor.analyze_phase_with_confidence(state, 'encoding')

def analyze_transformations_with_confidence(state: SequentialState, progress_callback=None) -> Dict[str, Any]:
    """Replace analyze_transformations_with_llm with confidence-based approach"""
    processor = ConfidenceBasedPreprocessor()
    return processor.analyze_phase_with_confidence(state, 'transformations')

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python SequentialPreprocessingAgent.py <csv_path> <target_column> [model_name]")
        print("Example: python SequentialPreprocessingAgent.py data.csv target gpt-4o")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    target_col = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("DEFAULT_MODEL", "gpt-4o")
    
    run_sequential_agent(csv_path, target_col, model) 