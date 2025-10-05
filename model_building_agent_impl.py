#!/usr/bin/env python3
from print_to_log import print_to_log
# Import master log handler to capture logger.info calls
try:
    import master_log_handler
except ImportError:
    pass

# Import for username resolution
from agent_utils import get_username_for_user_id, get_username_artifacts_dir


"""
LangGraph Implementation of ModelAgentLite
Architecture: UI -> Prompt Understanding Agent -> Controller Agent -> Model Building Agent
"""

import json
import os
import time
import math
import tempfile
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import ollama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# Import utilities
from agent_utils import extract_first_code_block, safe_joblib_dump, safe_plt_savefig, diagnose_io_error

# Import thread logging system
from thread_logger import get_thread_logger

# Global model states registry
global_model_states = {}

# Model configuration
MAIN_MODEL = os.getenv("MAIN_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")  # Main code generation model
ERROR_FIXING_MODEL_1 = MAIN_MODEL  # First fallback: Same model for consistency
ERROR_FIXING_MODEL_2 = os.getenv("ERROR_FIXING_MODEL_2", "deepseek-coder-v2:latest")  # Second fallback: Different model perspective

def extract_failing_code_block(full_code: str, error_msg: str) -> tuple[str, int, int, str]:
    """Extract the specific failing code block from the full code with imports context"""
    lines = full_code.split('\n')
    
    # Extract imports section (first 20 lines or until first non-import/comment)
    imports_section = []
    for i, line in enumerate(lines[:20]):
        stripped = line.strip()
        if (stripped.startswith('import ') or stripped.startswith('from ') or 
            stripped.startswith('#') or stripped.startswith('"""') or 
            stripped.startswith("'''") or not stripped):
            imports_section.append(line)
        else:
            break
    imports_context = '\n'.join(imports_section)
    
    # Try to find line number from traceback
    line_number = None
    if "line " in error_msg:
        import re
        line_match = re.search(r'line (\d+)', error_msg)
        if line_match:
            line_number = int(line_match.group(1)) - 1  # Convert to 0-based index
    
    # If we found a line number, extract context around it
    if line_number is not None and 0 <= line_number < len(lines):
        # Extract 5 lines before and after the error line
        start_line = max(0, line_number - 5)
        end_line = min(len(lines), line_number + 6)
        failing_block = '\n'.join(lines[start_line:end_line])
        return failing_block, start_line, end_line, imports_context
    
    # Fallback: look for specific error patterns
    if "pd.qcut" in error_msg and "Bin edges must be unique" in error_msg:
        # Find the pd.qcut line
        for i, line in enumerate(lines):
            if "pd.qcut" in line:
                start_line = max(0, i - 3)
                end_line = min(len(lines), i + 4)
                failing_block = '\n'.join(lines[start_line:end_line])
                return failing_block, start_line, end_line, imports_context
    
    # If we can't identify specific block, return a reasonable chunk
    # Take middle section to avoid imports and final result
    total_lines = len(lines)
    if total_lines > 20:
        start_line = total_lines // 4
        end_line = 3 * total_lines // 4
        failing_block = '\n'.join(lines[start_line:end_line])
        return failing_block, start_line, end_line, imports_context
    
    # For short code, return everything
    return full_code, 0, len(lines), imports_context

def fix_code_with_complete_rewrite(original_code: str, error_msg: str, error_type: str, user_query: str = "", intent: str = "", attempt: int = 1, original_system_prompt: str = "") -> str:
    """Complete code rewrite approach - pass full code, error, AND original system prompt to LLM"""
    
    # Select model based on attempt
    if attempt == 1:
        model_to_use = ERROR_FIXING_MODEL_1
        model_description = "same model (consistency)"
    elif attempt == 2:
        model_to_use = ERROR_FIXING_MODEL_2
        model_description = "different model (fresh perspective)"
    else:
        print_to_log("⚠️ Maximum attempts reached")
        return None
    
    print_to_log(f"🔧 Attempt {attempt}: Complete rewrite using {model_description}")
    print_to_log(f"🔧 Using original system prompt ({len(original_system_prompt)} chars) to maintain requirements")
    
    try:
        # Create error fixing prompt that preserves original system requirements
        error_fixing_prompt = f"""The following code was generated based on specific system requirements but failed with an error. 
Please rewrite the COMPLETE code to fix the error while maintaining ALL original requirements.

ORIGINAL USER REQUEST: {user_query}
INTENT: {intent}

FAILING CODE:
```python
{original_code}
```

ERROR MESSAGE: {error_msg}
ERROR TYPE: {error_type}

CRITICAL REQUIREMENTS:
1. Fix the error but maintain ALL the original system requirements and code structure
2. Keep the EXACT same result dictionary structure and keys as the original code
3. If the original code had 'models', 'user_config', or other specific keys, maintain them
4. Preserve all variable names and result format exactly as specified in the system prompt
5. Only fix the specific error, don't change the overall logic or result structure

REWRITTEN CODE:"""

        # Use the ORIGINAL system prompt as the system message (this is the key!)
        # This ensures the fixing LLM follows the same rules as the original generation
        system_message = original_system_prompt if original_system_prompt else BASE_SYSTEM_PROMPT

        # Call LLM for complete rewrite with ORIGINAL system prompt
        response = ollama.chat(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": error_fixing_prompt}
            ],
            options={
                "num_predict": -1,
                "temperature": 0.1,
                "top_p": 0.9
            }
        )
        
        if response and "message" in response and "content" in response["message"]:
            reply = response["message"]["content"]
            # Extract code from response
            rewritten_code = extract_first_code_block(reply)
            
            if rewritten_code and rewritten_code.strip():
                print_to_log(f"📝 Complete rewrite successful: {len(original_code)} → {len(rewritten_code)} chars")
                return rewritten_code
            else:
                print_to_log(f"⚠️ No valid code extracted from LLM response")
                return None
        else:
            print_to_log(f"⚠️ Invalid LLM response format")
            return None
            
    except Exception as e:
        print_to_log(f"❌ Error in complete rewrite: {e}")
        return None

def fix_code_with_tiered_llm_DEPRECATED(original_code: str, error_msg: str, error_type: str, data_shape: tuple, user_query: str = "", intent: str = "", attempt: int = 1) -> str:
    """Use tiered LLM approach for surgical error fixing"""
    
    # Select model based on attempt
    if attempt == 1:
        model_to_use = ERROR_FIXING_MODEL_1
        model_description = "same model (consistency)"
    elif attempt == 2:
        model_to_use = ERROR_FIXING_MODEL_2
        model_description = "different model (fresh perspective)"
    else:
        print_to_log("⚠️ Maximum attempts reached, using fallback rules")
        return auto_fix_code_errors_fallback(original_code, error_msg, error_type)
    
    # Check model availability
    if not ensure_model_available(model_to_use):
        print_to_log(f"❌ Model {model_to_use} not available, trying fallback")
        if attempt == 1:
            return fix_code_with_tiered_llm(original_code, error_msg, error_type, data_shape, user_query, intent, 2)
        else:
            return auto_fix_code_errors_fallback(original_code, error_msg, error_type)
    
    # Extract only the failing code block for surgical fixing
    failing_block, start_line, end_line, imports_context = extract_failing_code_block(original_code, error_msg)
    
    # Create surgical fixing prompt with imports context
    surgical_prompt = f"""Fix only this code block to resolve the error. Do not rename functions, do not change other logic, do not introduce new APIs. Keep style & structure intact.

ERROR MESSAGE:
{error_msg}

ORIGINAL USER QUERY: {user_query}
INTENT: {intent}

IMPORTS CONTEXT (for reference):
```python
{imports_context}
```

FAILING CODE BLOCK (lines {start_line+1}-{end_line}):
```python
{failing_block}
```

SURGICAL FIX RULES:
1. Fix ONLY the specific error - do not rewrite unrelated code
2. Preserve all function names, variable names, and structure
3. Keep the same coding style and patterns
4. Do not add new imports unless absolutely necessary
5. Return ONLY the fixed code block (no explanations)
6. CRITICAL: If error is about tree visualization for non-tree models (LGBM, XGB), REMOVE the tree plotting code entirely

COMMON QUICK FIXES:
- pd.qcut error → add duplicates='drop' parameter
- Import error → remove problematic import
- Undefined variable → add missing calculation
- Dict iteration → use list(dict.items())
- Tree plotting for LGBM/XGB → Remove tree plotting code, keep only rank ordering
- AttributeError 'tree_' → Remove tree-specific code for non-tree models

FIXED CODE BLOCK:"""

    try:
        print_to_log(f"🔧 Attempt {attempt}: Surgical fix using {model_description}")
        print_to_log(f"📍 Targeting lines {start_line+1}-{end_line} ({end_line-start_line} lines)")
        
        response = ollama.chat(
            model=model_to_use,
            messages=[{"role": "user", "content": surgical_prompt}],
            keep_alive="10m"
        )
        
        fixed_block = response["message"]["content"].strip()
        
        # Extract code block if wrapped in markdown
        if "```python" in fixed_block:
            start = fixed_block.find("```python") + 9
            end = fixed_block.find("```", start)
            if end != -1:
                fixed_block = fixed_block[start:end].strip()
        elif "```" in fixed_block:
            start = fixed_block.find("```") + 3
            end = fixed_block.find("```", start)
            if end != -1:
                fixed_block = fixed_block[start:end].strip()
        
        # Reconstruct the full code with the fixed block
        lines = original_code.split('\n')
        fixed_lines = lines[:start_line] + fixed_block.split('\n') + lines[end_line:]
        reconstructed_code = '\n'.join(fixed_lines)
        
        print_to_log(f"🔧 Surgical fix applied: {len(failing_block)} → {len(fixed_block)} chars in block")
        return reconstructed_code
        
    except Exception as e:
        print_to_log(f"⚠️ Surgical fix attempt {attempt} failed: {e}")
        if attempt == 1:
            return fix_code_with_tiered_llm(original_code, error_msg, error_type, data_shape, user_query, intent, 2)
        else:
            return auto_fix_code_errors_fallback(original_code, error_msg, error_type)

def auto_fix_code_errors_fallback(code: str, error_msg: str, error_type: str) -> str:
    """Fallback rule-based error fixing (original implementation)"""
    
    # Fix missing X_test, y_test variables
    if "name 'X_test' is not defined" in error_msg or "name 'y_test' is not defined" in error_msg:
        print_to_log("🔧 Fallback: Adding missing train/test split...")
        # Add train/test split after data splitting
        if "X = sample_data.drop(" in code and "train_test_split" not in code:
            # Find the line with X = sample_data.drop and add train/test split after it
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if "X = sample_data.drop('TARGET_COLUMN', axis=1)" in line:
                    # Insert train/test split after the next line (y = sample_data['TARGET_COLUMN'])
                    if i + 1 < len(lines) and "y = sample_data['TARGET_COLUMN']" in lines[i + 1]:
                        lines.insert(i + 2, "")
                        lines.insert(i + 3, "# Train/test split for validation")
                        lines.insert(i + 4, "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
                        return '\n'.join(lines)
    
    # Fix graphviz ImportError for tree plotting
    if "graphviz" in error_msg.lower() and "plot_tree" in code:
        print_to_log("🔧 Fallback: Fixing graphviz ImportError by removing tree plotting...")
        # Remove the problematic plot_tree lines
        lines = code.split('\n')
        filtered_lines = []
        skip_next = False
        for line in lines:
            if "plot_tree" in line or "safe_plt_savefig" in line:
                # Skip tree plotting lines
                continue
            elif "import matplotlib.pyplot as plt" in line and "plot_tree" in code:
                # Skip matplotlib import if only used for tree plotting
                continue
            else:
                filtered_lines.append(line)
        return '\n'.join(filtered_lines)
    
    # Fix pd.qcut "Bin edges must be unique" error
    if "Bin edges must be unique" in error_msg and "pd.qcut" in code:
        print_to_log("🔧 Fallback: Fixing pd.qcut duplicate edges error...")
        # Replace pd.qcut with pd.qcut(..., duplicates='drop')
        fixed_code = code.replace(
            "pd.qcut(test_df['probability'], q=10, labels=False)",
            "pd.qcut(test_df['probability'], q=10, labels=False, duplicates='drop')"
        )
        # Also handle other variations
        import re
        fixed_code = re.sub(
            r"pd\.qcut\(([^,]+),\s*q=(\d+),\s*labels=False\)",
            r"pd.qcut(\1, q=\2, labels=False, duplicates='drop')",
            fixed_code
        )
        return fixed_code
    
    # Fix "cannot import name" errors by removing problematic imports
    if "cannot import name" in error_msg and "ImportError" in error_type:
        print_to_log("🔧 Fallback: Fixing import error...")
        lines = code.split('\n')
        fixed_lines = []
        for line in lines:
            # Skip lines that import the problematic function
            if "from sklearn.metrics import" in line and any(prob in error_msg for prob in ["huber_loss", "average_precision_score"]):
                # Remove the problematic import from the line
                if "huber_loss" in error_msg:
                    line = line.replace(", huber_loss", "").replace("huber_loss, ", "").replace("huber_loss", "")
                if "average_precision_score" in error_msg:
                    line = line.replace(", average_precision_score", "").replace("average_precision_score, ", "").replace("average_precision_score", "")
                # Clean up any double commas or trailing commas
                import re
                line = re.sub(r',\s*,', ',', line)
                line = re.sub(r',\s*\)', ')', line)
                if line.strip().endswith("import"):
                    continue  # Skip empty import line
            fixed_lines.append(line)
        return '\n'.join(fixed_lines)
    
    # Fix "name 'X_test' is not defined" when using existing models
    if "name 'X_test' is not defined" in error_msg:
        print_to_log("🔧 Fallback: Fixing undefined X_test error...")
        # Add data preparation code at the beginning
        # Get target column from pipeline state or use default
        target_col = 'target'  # Default fallback
        try:
            from pipeline_state import state_manager
            current_state = state_manager.load_state(user_id)
            if current_state and hasattr(current_state, 'target_column') and current_state.target_column:
                target_col = current_state.target_column
        except:
            pass
        
        data_prep = f"""
# Prepare data for existing model usage
X = sample_data.drop('{target_col}', axis=1)
y = sample_data['{target_col}']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
        return data_prep + code
    
    return code  # Return original code if no fix available

def check_model_availability(model_name: str) -> bool:
    """Check if a model is available in Ollama"""
    try:
        # Try a simple warmup call to check availability
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            keep_alive="1m"
        )
        return True
    except Exception as e:
        print_to_log(f"⚠️ Model {model_name} not available: {e}")
        return False

def ensure_model_available(model_name: str) -> bool:
    """Ensure model is available, try to pull if not"""
    if check_model_availability(model_name):
        return True
    
    try:
        print_to_log(f"📥 Attempting to pull model: {model_name}")
        # Note: ollama.pull() might not be available in all versions
        # This is a placeholder - you might need to use subprocess or ollama CLI
        import subprocess
        result = subprocess.run(['ollama', 'pull', model_name], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print_to_log(f"✅ Successfully pulled {model_name}")
            return check_model_availability(model_name)
        else:
            print_to_log(f"❌ Failed to pull {model_name}: {result.stderr}")
            return False
    except Exception as e:
        print_to_log(f"❌ Error pulling model {model_name}: {e}")
        return False

def preload_ollama_models():
    """Preload main and error-fixing models to avoid cold starts"""
    success_count = 0
    total_models = 3
    
    # Preload main model
    try:
        print_to_log(f"🔄 Preloading main model ({MAIN_MODEL})...")
        if ensure_model_available(MAIN_MODEL):
            response = ollama.chat(
                model=MAIN_MODEL,
                messages=[{"role": "user", "content": "warmup"}],
                keep_alive="10m"
            )
            print_to_log(f"✅ Main model preloaded")
            success_count += 1
        else:
            print_to_log(f"❌ Main model not available")
    except Exception as e:
        print_to_log(f"⚠️ Failed to preload main model: {e}")
    
    # Preload first error-fixing model (same as main)
    if ERROR_FIXING_MODEL_1 != MAIN_MODEL:
        try:
            print_to_log(f"🔄 Preloading error-fixing model 1 ({ERROR_FIXING_MODEL_1})...")
            if ensure_model_available(ERROR_FIXING_MODEL_1):
                response = ollama.chat(
                    model=ERROR_FIXING_MODEL_1,
                    messages=[{"role": "user", "content": "warmup"}],
                    keep_alive="10m"
                )
                print_to_log(f"✅ Error-fixing model 1 preloaded")
                success_count += 1
            else:
                print_to_log(f"❌ Error-fixing model 1 not available")
        except Exception as e:
            print_to_log(f"⚠️ Failed to preload error-fixing model 1: {e}")
    else:
        print_to_log(f"✅ Error-fixing model 1 same as main model")
        success_count += 1
    
    # Preload second error-fixing model (DeepSeek)
    try:
        print_to_log(f"🔄 Preloading error-fixing model 2 ({ERROR_FIXING_MODEL_2})...")
        if ensure_model_available(ERROR_FIXING_MODEL_2):
            response = ollama.chat(
                model=ERROR_FIXING_MODEL_2,
                messages=[{"role": "user", "content": "warmup"}],
                keep_alive="10m"
            )
            print_to_log(f"✅ Error-fixing model 2 preloaded")
            success_count += 1
        else:
            print_to_log(f"❌ Error-fixing model 2 not available")
    except Exception as e:
        print_to_log(f"⚠️ Failed to preload error-fixing model 2: {e}")
    
    print_to_log(f"📊 Model preloading summary: {success_count}/{total_models} models ready")
    return success_count > 0  # Return True if at least one model is available

# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """State shared across all agents in the graph"""
    # User inputs
    user_id: str
    query: str
    data: Optional[pd.DataFrame]
    
    # Messages and communication
    messages: List[Dict[str, Any]]
    
    # Agent outputs
    intent: str  # From prompt understanding agent
    routing_decision: str  # From controller agent
    code: str  # Generated code
    execution_result: Any  # Execution result
    
    # Model state
    model_path: Optional[str]
    has_existing_model: bool
    
    # Final output
    response: str
    artifacts: Dict[str, Any]
    
    # Progress callback for real-time updates
    progress_callback: Optional[callable]

# =============================================================================
# LIBRARY MANAGEMENT UTILITIES
# =============================================================================

def extract_missing_library(error_msg: str) -> str:
    """Extract the missing library name from import error message"""
    import re
    
    # Common patterns for missing libraries
    patterns = [
        r"No module named '([^']+)'",
        r"No module named ([^\s]+)",
        r"cannot import name '([^']+)'",
        r"ImportError: (.+)",
        r"ModuleNotFoundError: (.+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, error_msg)
        if match:
            library_name = match.group(1).strip()
            # Clean up library name
            if library_name.startswith("No module named "):
                library_name = library_name.replace("No module named ", "").strip("'\"")
            return library_name.split('.')[0]  # Get base library name
    
    return None

def get_library_install_guidance(library_name: str) -> dict:
    """Get installation guidance for missing libraries"""
    
    # Common ML/Data Science libraries and their install commands
    library_guidance = {
        "lightgbm": {
            "install_cmd": "pip install lightgbm",
            "description": "LightGBM - Gradient boosting framework",
            "alternatives": ["sklearn.ensemble.GradientBoostingClassifier", "xgboost"]
        },
        "xgboost": {
            "install_cmd": "pip install xgboost", 
            "description": "XGBoost - Gradient boosting library",
            "alternatives": ["lightgbm", "sklearn.ensemble.GradientBoostingClassifier"]
        },
        "catboost": {
            "install_cmd": "pip install catboost",
            "description": "CatBoost - Gradient boosting library",
            "alternatives": ["lightgbm", "xgboost"]
        },
        "seaborn": {
            "install_cmd": "pip install seaborn",
            "description": "Seaborn - Statistical data visualization",
            "alternatives": ["matplotlib.pyplot"]
        },
        "plotly": {
            "install_cmd": "pip install plotly",
            "description": "Plotly - Interactive plotting library",
            "alternatives": ["matplotlib.pyplot", "seaborn"]
        },
        "shap": {
            "install_cmd": "pip install shap",
            "description": "SHAP - Model explainability library",
            "alternatives": ["sklearn.inspection.permutation_importance"]
        },
        "optuna": {
            "install_cmd": "pip install optuna",
            "description": "Optuna - Hyperparameter optimization",
            "alternatives": ["sklearn.model_selection.GridSearchCV", "sklearn.model_selection.RandomizedSearchCV"]
        },
        "bayes_opt": {
            "install_cmd": "pip install bayesian-optimization",
            "description": "Bayesian Optimization - Hyperparameter tuning using Gaussian processes",
            "alternatives": ["sklearn.model_selection.GridSearchCV", "sklearn.model_selection.RandomizedSearchCV", "optuna"]
        },
        "bayesian-optimization": {
            "install_cmd": "pip install bayesian-optimization",
            "description": "Bayesian optimization library",
            "alternatives": ["sklearn.model_selection.RandomizedSearchCV", "optuna"]
        },
        "skopt": {
            "install_cmd": "pip install scikit-optimize",
            "description": "Scikit-optimize - Bayesian optimization",
            "alternatives": ["sklearn.model_selection.RandomizedSearchCV"]
        }
    }
    
    # Default guidance for unknown libraries
    default_guidance = {
        "install_cmd": f"pip install {library_name}",
        "description": f"{library_name} - External library",
        "alternatives": ["Check if there's a sklearn equivalent"]
    }
    
    return library_guidance.get(library_name.lower(), default_guidance)

def get_library_fallback_code(original_code: str, missing_library: str) -> str:
    """Generate fallback code when a library is missing"""
    
    # Define fallback replacements
    fallbacks = {
        "lightgbm": {
            "import lightgbm as lgb": "# lightgbm not available - using sklearn GradientBoosting\nfrom sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor\nclass lgb:\n    LGBMClassifier = GradientBoostingClassifier\n    LGBMRegressor = GradientBoostingRegressor",
            "from lightgbm import LGBMClassifier": "from sklearn.ensemble import GradientBoostingClassifier as LGBMClassifier",
            "from lightgbm import LGBMRegressor": "from sklearn.ensemble import GradientBoostingRegressor as LGBMRegressor",
            "lgb.LGBMClassifier(": "GradientBoostingClassifier(",
            "lgb.LGBMRegressor(": "GradientBoostingRegressor(",
            "LGBMClassifier(": "GradientBoostingClassifier(",
            "LGBRegressor(": "GradientBoostingRegressor("
        },
        "xgboost": {
            "from xgboost import XGBClassifier": "from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier",
            "from xgboost import XGBRegressor": "from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor",
            "XGBClassifier(": "GradientBoostingClassifier(",
            "XGBRegressor(": "GradientBoostingRegressor("
        },
        "bayes_opt": {
            "from bayes_opt import BayesianOptimization": "# FALLBACK: Using sklearn GridSearchCV instead of Bayesian Optimization\nfrom sklearn.model_selection import GridSearchCV",
            "BayesianOptimization(": "GridSearchCV(",
            "optimizer.maximize(": "# Using GridSearchCV instead of Bayesian optimization\n# optimizer.maximize(",
            "optimizer.probe(": "# optimizer.probe("
        },
        "optuna": {
            "import optuna": "# FALLBACK: Using sklearn GridSearchCV instead of Optuna\nfrom sklearn.model_selection import GridSearchCV",
            "optuna.create_study": "# Using GridSearchCV instead of Optuna study",
            "study.optimize": "# Using GridSearchCV instead of study.optimize"
        },
        "seaborn": {
            "import seaborn as sns": "# seaborn not available - using matplotlib",
            "sns.": "plt.",
            "seaborn": "matplotlib.pyplot"
        },
        "plotly": {
            "import plotly": "# plotly not available - using matplotlib",
            "plotly.": "plt."
        }
    }
    
    library_key = missing_library.lower()
    if library_key in fallbacks:
        modified_code = original_code
        for old_pattern, new_pattern in fallbacks[library_key].items():
            modified_code = modified_code.replace(old_pattern, new_pattern)
        
        # Add a comment about the fallback
        modified_code = f"# FALLBACK: {missing_library} not available, using alternative\n" + modified_code
        return modified_code
    
    return None

# =============================================================================
# EXECUTION AGENT (From original core.py)
# =============================================================================

def ExecutionAgent(code: str, df: pd.DataFrame, user_id="default_user", max_retries=2, verbose=True, model_states=None, user_query="", intent="", artifacts_dir=None, progress_callback=None, original_system_prompt=""):
    """Execute generated code with proper environment and error handling"""
    
    # Get thread logger
    if '_' in user_id:
        parts = user_id.split('_')
        actual_user_id = parts[0] if len(parts) >= 1 else user_id
        thread_id = '_'.join(parts[1:]) if len(parts) > 1 else user_id
    else:
        actual_user_id = user_id
        thread_id = user_id
    
    thread_logger = get_thread_logger(actual_user_id, thread_id)
    
    # Single line execution start
    timestamp = datetime.now().strftime("%H:%M:%S")
    print_to_log(f"⚙️ EXEC START [{timestamp}] Session: {user_id} | Data: {df.shape} | Code: {len(code)} chars")
    print_to_log(f"🔍 ExecutionAgent called with verbose={verbose}, max_retries={max_retries}")
    print_to_log(f"🔍 Complete code rewrite error fixing is ENABLED (replaced broken surgical approach)")
    
    # Log code execution start
    thread_logger.log_data_operation("code_execution_start", {
        "user_query": user_query,
        "intent": intent,
        "data_shape": df.shape,
        "code_length": len(code)
    })
    
    # Setup is fast - no progress needed
    
    # Initialize environment with data and current model if it exists
    env = {
        "pd": pd, 
        "sample_data": df, 
        "np": np, 
        "joblib": joblib,
        "intent": intent,  # Add intent to environment
        "user_query": user_query  # Also add user_query for context
    }
    
    # Add sklearn modules and functions to environment
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,
        roc_auc_score, average_precision_score, log_loss,
        mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Additional imports for general data analysis
    import seaborn as sns
    from scipy import stats
    import warnings
    import math
    import tempfile
    import time
    import os
    warnings.filterwarnings('ignore')
    
    # Add all imports to environment
    env.update({
        "train_test_split": train_test_split,
        "accuracy_score": accuracy_score, "precision_score": precision_score,
        "recall_score": recall_score, "f1_score": f1_score,
        "confusion_matrix": confusion_matrix, "classification_report": classification_report,
        "roc_auc_score": roc_auc_score, "average_precision_score": average_precision_score,
        "log_loss": log_loss, "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error, "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "r2_score": r2_score, "RandomForestClassifier": RandomForestClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier, "plot_tree": plot_tree,
        "OneHotEncoder": OneHotEncoder, "LabelEncoder": LabelEncoder,
        "StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler,
        "ColumnTransformer": ColumnTransformer, "Pipeline": Pipeline,
        "LGBMClassifier": LGBMClassifier, "XGBClassifier": XGBClassifier,
        "plt": plt, "math": math, "sqrt": math.sqrt, "tempfile": tempfile,
        "sns": sns, "stats": stats, "warnings": warnings, "time": time, "os": os
    })
    
    # Add thread-aware safe wrapper functions
    def thread_safe_joblib_dump(obj, filename):
        if artifacts_dir:
            filepath = os.path.join(artifacts_dir, filename)
        else:
            filepath = filename
        return safe_joblib_dump(obj, filepath)
    
    def thread_safe_plt_savefig(filename, **kwargs):
        if artifacts_dir:
            filepath = os.path.join(artifacts_dir, filename)
        else:
            filepath = filename
        return safe_plt_savefig(filepath, **kwargs)
    
    env["safe_joblib_dump"] = thread_safe_joblib_dump
    env["safe_plt_savefig"] = thread_safe_plt_savefig
    
    # Load existing model if available
    if model_states is None:
        model_states = {}  # Fallback for standalone usage
    state = model_states.get(user_id, {})
    if state.get('model_path') and os.path.exists(state['model_path']):
        try:
            env['current_model'] = joblib.load(state['model_path'])
            if verbose:
                print_to_log(f"✅ Loaded existing model from {state['model_path']}")
                print_to_log(f"📊 Model type: {type(env['current_model']).__name__}")
                # Log model parameters to help debug performance issues
                if hasattr(env['current_model'], 'get_params'):
                    params = env['current_model'].get_params()
                    print_to_log(f"🔧 Model parameters: {params}")
        except Exception as e:
            if verbose:
                print_to_log(f"⚠️ Failed to load existing model: {e}")
    else:
        if verbose:
            model_path = state.get('model_path', 'None')
            path_exists = os.path.exists(model_path) if model_path else False
            print_to_log(f"🔍 No existing model found - Path: {model_path}, Exists: {path_exists}")
            print_to_log(f"🔍 Available model states: {list(model_states.keys())}")
            print_to_log(f"🔍 Global model states: {list(global_model_states.keys())}")
            # Try to find any recent model files in artifacts directory first, then root
            import glob
            search_paths = []
            username = get_username_for_user_id(user_id)
            if artifacts_dir and os.path.exists(artifacts_dir):
                search_paths.append(os.path.join(artifacts_dir, f"model_{username}_*.joblib"))
                search_paths.append(os.path.join(artifacts_dir, "*model*.joblib"))  # Any model file
            search_paths.append(f"model_{username}_*.joblib")  # Fallback to root
            
            recent_models = []
            for pattern in search_paths:
                found_models = glob.glob(pattern)
                if found_models:
                    recent_models.extend(found_models)
                    print_to_log(f"🔍 Found {len(found_models)} model(s) with pattern: {pattern}")
            
            if recent_models:
                latest_model = max(recent_models, key=os.path.getctime)
                print_to_log(f"🔍 Found recent model file: {latest_model}")
                # Load it manually if we found one
                try:
                    env['current_model'] = joblib.load(latest_model)
                    print_to_log(f"✅ Manually loaded model from {latest_model}")
                    print_to_log(f"📊 Model type: {type(env['current_model']).__name__}")
                except Exception as e:
                    print_to_log(f"⚠️ Failed to manually load model: {e}")
    
    # Execute code with retries
    for attempts in range(1, max_retries + 1):
        # Send progress update only for first execution attempt
        if progress_callback and attempts == 1:
            progress_callback("⚡ Running your code...", "Code Execution")
        
        try:
            if verbose:
                print_to_log(f"🔄 Execution attempt {attempts}/{max_retries}")
                print_to_log(f"📝 DEBUG - Generated Code (Attempt {attempts}):")
                print_to_log("=" * 60)
                print_to_log(code)
                print_to_log("=" * 60)
                print_to_log(f"🔍 About to execute {len(code)} characters of code...")
                print_to_log(f"🔍 Code starts with: {code[:100]}...")
                print_to_log(f"🔍 Environment has sample_data shape: {df.shape}")
                print_to_log("🔍 Starting execution now...")
            
            # Execution start - no additional progress needed
            
            # Execute the code with timeout (if supported)
            print_to_log("🔍 About to call exec()...")
            print_to_log(f"🔍 Environment variables available: {list(env.keys())}")
            print_to_log(f"🔍 'intent' in environment: {'intent' in env}")
            print_to_log(f"🔍 'intent' value: {env.get('intent', 'NOT_FOUND')}")
            
            import sys, threading
            try:
                import signal
                can_use_alarm = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread() and sys.platform != "win32"
            except Exception:
                can_use_alarm = False

            if can_use_alarm:
                def timeout_handler(signum, frame):
                    raise TimeoutError("Code execution timed out after 60 seconds")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
                try:
                    exec(code, env)
                    print_to_log("🔍 exec() completed successfully!")
                except TimeoutError as e:
                    print_to_log(f"⏰ Execution timed out: {e}")
                    raise e
                finally:
                    signal.alarm(0)  # Disable alarm
            else:
                exec(code, env)
                print_to_log("🔍 exec() completed successfully!")
            
            # Success - show final completion only
            if progress_callback:
                progress_callback("✅ Analysis complete!", "Completed")
            
            # Get result
            if 'result' not in env:
                result = "Code executed successfully but no 'result' variable was set"
            else:
                result = env['result']
            
            # Save model if any model-like object is present in the result
            if isinstance(result, dict):
                print_to_log(f"🔍 Checking result for model objects. Result keys: {list(result.keys())}")
                model_objects = []
                for key, value in list(result.items()):
                    print_to_log(f"🔍 Checking key '{key}': type={type(value)}, has_predict={hasattr(value, 'predict')}, has_fit={hasattr(value, 'fit')}")
                    if hasattr(value, 'predict') and hasattr(value, 'fit'):
                        model_objects.append((key, value))
                        print_to_log(f"✅ Found model object: {key} -> {type(value)}")
                
                print_to_log(f"🔍 Total model objects found: {len(model_objects)}")
                if model_objects:
                    save_key, model_obj = model_objects[0]
                    try:
                        username = get_username_for_user_id(user_id)
                        model_filename = f"model_{username}_{int(time.time())}.joblib"
                        print_to_log(f"🔍 Attempting to save model to: {model_filename}")
                        # Use the thread-safe wrapper that saves to artifacts directory
                        model_path = env["safe_joblib_dump"](model_obj, model_filename)
                        result['model_path'] = model_path
                        
                        # Update model state
                        model_states[user_id] = {
                            'model_path': model_path,
                            'last_result': result
                        }
                        # Also update global model states
                        global_model_states[user_id] = model_states[user_id]
                        if verbose:
                            print_to_log(f"✅ Saved new model to {model_path} (from result key: {save_key})")
                    except Exception as e:
                        if verbose:
                            print_to_log(f"⚠️ Failed to save model: {e}")
                else:
                    print_to_log(f"⚠️ No model objects found in result. Available keys: {list(result.keys())}")
            
            # Single line execution success
            result_info = f"Dict: {len(result)} keys" if isinstance(result, dict) else f"Type: {type(result).__name__}"
            print_to_log(f"✅ EXEC SUCCESS [{user_id}] {result_info}")
            
            # Log successful execution
            if thread_logger:
                thread_logger.log_data_operation("code_execution_success", {
                    "user_query": user_query,
                    "intent": intent,
                    "result_type": type(result).__name__,
                    "result_info": result_info
                })
            
            # Final completion already sent above
            
            return result
            
        except (OSError, IOError) as io_exc:
            # Handle I/O errors specifically with enhanced fallback
            error_msg = f"I/O Error (Errno {getattr(io_exc, 'errno', 'unknown')}): {str(io_exc)}"
            if verbose:
                print_to_log(f"💾 I/O Error during execution: {error_msg}")
            
            # Log I/O error
            if thread_logger:
                thread_logger.error("Code execution I/O error", io_exc, {
                    "user_query": user_query,
                    "intent": intent,
                    "error_type": "IOError"
                })
            
            # Get diagnostic information
            diagnosis = diagnose_io_error(error_msg)
            
            if attempts == max_retries:
                # Create fallback result with diagnostic info
                fallback_result = {
                    "error": "I/O Error encountered during execution",
                    "error_details": error_msg,
                    "diagnostic_info": diagnosis,
                    "execution_status": "failed_with_io_error",
                    "suggested_actions": diagnosis['suggested_actions']
                }
                if verbose:
                    print_to_log(f"🔍 Diagnosis: {diagnosis['likely_causes']}")
                    print_to_log(f"💡 Suggestions: {diagnosis['suggested_actions']}")
                
                # Single line fallback
                print_to_log(f"⚠️ EXEC FALLBACK [{user_id}] I/O Error - using fallback result")
                
                return fallback_result
            
            # Add delay before retry for I/O errors
            time.sleep(1 + attempts)
            
        except Exception as exc:
            error_msg = str(exc)
            error_type = type(exc).__name__
            print_to_log(f"🚨 EXCEPTION CAUGHT in ExecutionAgent!")
            print_to_log(f"🔍 Error type: {error_type}")
            print_to_log(f"🔍 Error message: {error_msg}")
            print_to_log(f"🔍 Current attempt: {attempts}/{max_retries}")
            print_to_log(f"🔍 Should try tiered LLM? {attempts <= 2}")
            
            # Handle import/library errors specifically
            if error_type in ["ImportError", "ModuleNotFoundError"] or "No module named" in error_msg:
                missing_library = extract_missing_library(error_msg)
                if missing_library:
                    print_to_log(f"📦 MISSING LIBRARY DETECTED: {missing_library}")
                    
                    # Try to provide installation guidance and fallback
                    install_guidance = get_library_install_guidance(missing_library)
                    fallback_code = get_library_fallback_code(code, missing_library)
                    
                    if fallback_code and attempts == 1:
                        print_to_log(f"🔧 ATTEMPTING FALLBACK: Replacing {missing_library} with alternative...")
                        
                        # Notify user about fallback via progress callback
                        if progress_callback:
                            alternatives = install_guidance.get('alternatives', [])
                            alt_text = alternatives[0] if alternatives else "sklearn alternative"
                            progress_callback(f"📦 {missing_library} not found, using {alt_text} instead", "Library Fallback")
                        
                        code = fallback_code
                        continue  # Retry with fallback code
                    else:
                        # Return helpful error with admin request message
                        return {
                            "error": f"Missing library: {missing_library}",
                            "error_type": "ImportError",
                            "admin_message": f"Please contact your system administrator to install: {install_guidance['install_cmd']}",
                            "description": install_guidance["description"],
                            "alternatives": install_guidance["alternatives"],
                            "execution_status": "failed_missing_library",
                            "user_friendly_message": f"The code requires '{missing_library}' library which is not installed. Please contact your admin to install it, or I can try using an alternative approach."
                        }
            
            if verbose:
                print_to_log(f"❌ {error_type} during execution: {error_msg}")
                if "dictionary changed size during iteration" in error_msg:
                    print_to_log("🔍 DICT ITERATION ERROR DETECTED:")
                    print_to_log("   This usually happens when modifying a dict while iterating over it")
                    print_to_log("   Common causes: for k,v in dict.items(): dict[new_key] = value")
                    print_to_log("   Solution: Create a copy first or collect changes separately")
            
            # NEW APPROACH: Complete code rewrite with full context (much more reliable)
            if attempts <= 2:  # Try LLM complete rewrite for first two attempts
                print_to_log(f"🔧 STARTING complete code rewrite for attempt {attempts}...")
                print_to_log(f"🔧 Passing full code and error to LLM for complete rewrite...")
                fixed_code = fix_code_with_complete_rewrite(code, error_msg, error_type, user_query, intent, attempts, original_system_prompt)
                if fixed_code and fixed_code != code:
                    print_to_log(f"✅ Complete rewrite provided a solution")
                    print_to_log(f"📝 Code length: {len(code)} → {len(fixed_code)} characters")
                    code = fixed_code
                    print_to_log(f"🔄 RETRYING with rewritten code immediately...")
                    continue  # Retry with rewritten code immediately
                else:
                    print_to_log(f"⚠️ Complete rewrite couldn't provide a different solution")
                    # Try rule-based fallback when LLM rewrite fails
                    print_to_log(f"🔧 Falling back to rule-based error fixing...")
                    fallback_code = auto_fix_code_errors_fallback(code, error_msg, error_type)
                    if fallback_code and fallback_code != code:
                        print_to_log(f"✅ Rule-based fallback provided a solution")
                        print_to_log(f"📝 Code length: {len(code)} → {len(fallback_code)} characters")
                        code = fallback_code
                        print_to_log(f"🔄 RETRYING with fallback-fixed code immediately...")
                        continue  # Retry with fallback-fixed code immediately
            else:
                print_to_log(f"🚫 Skipping LLM rewrite (attempt {attempts} > 2)")
                # For attempts > 2, go straight to rule-based fallback
                print_to_log(f"🔧 Using rule-based error fixing for attempt {attempts}...")
                fallback_code = auto_fix_code_errors_fallback(code, error_msg, error_type)
                if fallback_code and fallback_code != code:
                    print_to_log(f"✅ Rule-based fallback provided a solution")
                    print_to_log(f"📝 Code length: {len(code)} → {len(fallback_code)} characters")
                    code = fallback_code
                    print_to_log(f"🔄 RETRYING with fallback-fixed code immediately...")
                    continue  # Retry with fallback-fixed code immediately
            
            if attempts == max_retries:
                if verbose:
                    print_to_log(f"💥 Final failure after {max_retries} attempts.")
                print_to_log(f"❌ EXEC FAILED [{user_id}] {error_type}: {error_msg}")
                return f"Final error after {max_retries} attempts: {error_msg}"
            
            # Add delay before retry for any error (only if not auto-fixed)
            if verbose:
                print_to_log(f"⏳ Waiting {attempts} seconds before retry...")
            time.sleep(attempts)

# =============================================================================
# AGENT 1: PROMPT UNDERSTANDING AGENT
# =============================================================================

def prompt_understanding_agent(state: AgentState) -> AgentState:
    """
    Classifies user intent using Qwen 2.5 model
    Maps to your current classify_user_intent function
    """
    print_to_log(f"🧠 PROMPT UNDERSTANDING AGENT - Processing query: {state['query'][:60]}...")
    
    query = state["query"]
    user_id = state["user_id"]
    
    # Get conversation context from messages
    messages = state.get("messages", [])
    if messages:
        # Build context from recent relevant messages for intent classification
        # Focus on last few interactions but include all data uploads and model builds
        context_parts = []
        
        # Always include data uploads (regardless of position)
        data_uploads = [msg for msg in messages if msg.get("type") == "data_upload"]
        for msg in data_uploads[-2:]:  # Last 2 data uploads
            context_parts.append(f"User uploaded {msg.get('content', 'data')}")
        
        # Always include successful model builds (regardless of position) 
        model_builds = [msg for msg in messages if msg.get("type") == "model_built"]
        for msg in model_builds[-2:]:  # Last 2 model builds
            context_parts.append(f"Built {msg.get('content', 'model')}")
        
        # Include recent user queries (last 3)
        recent_queries = [msg for msg in messages if msg.get("type") == "user_query"]
        for msg in recent_queries[-3:]:
            query_content = msg.get('content', '')[:50]  # Truncate long queries
            context_parts.append(f"User asked: {query_content}")
        
        context = "; ".join(context_parts)
    else:
        context = ""
    
    context_info = f"CONVERSATION CONTEXT: {context}" if context else "CONVERSATION CONTEXT: None"
    
    classification_prompt = f"""You are an intent classifier for a machine learning model building agent. Your job is to classify user queries into exactly one of these categories:

1. "new_model" - User wants to build/create/train a SINGLE NEW model from scratch
2. "multi_model" - User wants to build/train/compare MULTIPLE models and find the best one
3. "use_existing" - User wants to use/modify/visualize an EXISTING model that was previously built

{context_info}

CRITICAL CONTEXT RULES:
- If context shows user just uploaded data or said "here is data" and current query says "use this", it means "use this DATA to build a model" → new_model
- "use this data", "use this file", "here is data, use this" → new_model (not use_existing)
- Only classify as "use_existing" if user explicitly refers to a previously BUILT MODEL
- CRITICAL: "use this tree model", "use this model", "use the tree" = use_existing (refers to existing model)
- CRITICAL: "use this tree" in context of model analysis = use_existing (not new model building)

PRIORITY CLASSIFICATION RULES (check in this order):

HIGHEST PRIORITY - "use_existing" if query contains:
- "use this model", "use the model", "with this model", "for this model" (explicit model reference)
- "use the best model", "use best model", "apply the best model", "apply best model" (best model reference)
- "use this tree", "use the tree", "with this tree", "for this tree" (tree model reference)  
- "use this tree model", "use this decision tree", "apply this model", "apply existing model"
- "existing model", "current model", "built model", "trained model", "saved model"
- "show plot", "visualize tree", "display tree" (when model exists)
- "build segments", "build deciles", "build buckets", "build rankings" (these use existing models)
- "rank ordering", "score", "predict", "classify" with existing model context
- ANY combination of "use" + "this" + model terms (tree, model, classifier, etc.)
- ANY combination of "use" + "best" + model terms (use best model, use the best model, etc.)

SECOND PRIORITY - "multi_model" if query contains:
- "build multiple models", "train several models", "create multiple", "build three models", "build 3 models"
- "compare models", "model comparison", "best model", "choose best", "pick best", "select best"
- Multiple algorithm names: "lgbm random forest", "xgboost random forest", "decision tree lgbm", etc.
- "all models", "different models", "various models", "several algorithms"
- "comparison plot", "compare performance", "roc curves", "model metrics"
- "benchmark models", "evaluate models", "test multiple", "try different"

DEFAULT PRIORITY - "new_model" for single model requests:
- Context shows recent data upload/mention AND current query says "use this"
- Query contains "build [model_type]", "create [model_type]", "train [model_type]" where model_type is: model, classifier, regressor, tree, forest, lgbm, xgboost
- "use this data", "use this file", "with this data"
- Any other single model building, training, or creation requests

EXAMPLES:
- Context: "uploaded data" + Query: "use this" → new_model (use this data to build model)
- Context: "built lgbm model" + Query: "use this model for segments" → use_existing
- "build lgbm model" → new_model (creating single new model)
- "show plot for the model" → use_existing (visualizing existing model)
- "train random forest" → new_model (creating single new model)
- "build three models lgbm random forest xgboost" → multi_model (multiple models)
- "compare models and pick the best" → multi_model (model comparison)
- "build lgbm and random forest and compare" → multi_model (multiple + comparison)

USER QUERY: "{query}"

Respond with ONLY one word: new_model, multi_model, or use_existing"""

    try:
        response = ollama.chat(
            # model="krith/qwen2.5-coder-14b-instruct:IQ2_M",
            model=MAIN_MODEL,
            messages=[{"role": "user", "content": classification_prompt}]
        )
        
        intent = response["message"]["content"].strip().lower()
        
        # Validate response
        if intent not in ["new_model", "multi_model", "use_existing"]:
            # Fallback classification
            intent = fallback_classify_intent(query)
        
        print_to_log(f"🎯 Intent classified: {intent}")
        
        # Update state
        state["intent"] = intent
        state["messages"].append({
            "agent": "prompt_understanding", 
            "content": f"Classified intent as: {intent}",
            "timestamp": datetime.now().isoformat()
        })
        
        return state
        
    except Exception as e:
        print_to_log(f"⚠️ Prompt understanding failed: {e}")
        intent = fallback_classify_intent(query)
        state["intent"] = intent
        return state

def semantic_classify_model_intent(query: str) -> str:
    """Semantic classification for model-specific intents using embeddings"""
    
    try:
        # Import orchestrator for semantic classification
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import pattern classifier dynamically to avoid initialization issues
        try:
            from toolbox import pattern_classifier
            if not pattern_classifier:
                print_to_log("⚠️ Pattern classifier not initialized, using fallback classification")
                return "new_model", "fallback"
        except ImportError:
            print_to_log("⚠️ Could not import pattern classifier, using fallback classification")
            return "new_model", "fallback"
        
        # Define model-specific intent definitions
        model_intent_definitions = {
            "use_existing": "Use existing model, apply current model, utilize trained model, work with built model, use this model, apply this classifier, use previous model, existing model analysis, current model evaluation, built model application, trained model usage, model reuse, apply saved model, use the best model, use best model, apply the best model, apply best model, show plot, visualize tree, display tree, build segments, build deciles, build buckets, build rankings, rank ordering, score, predict, classify, create rank order table, create segments, create buckets, generate rankings, generate segments, generate buckets, generate rank order, create deciles, build rank order, create ranking table, generate decile table, score data, predict outcomes, classify records, apply model for scoring, use model for ranking, apply for segmentation, use this tree model, use this tree, use the tree, with this model, for this model, with this tree, for this tree, apply this model, apply existing model, use this decision tree, utilize this model, employ this model, leverage this model, work with this model, operate this model, run this model, execute this model",
            "multi_model": "Build multiple models, train several models, create multiple models, build three models, build 3 models, compare models, model comparison, best model, choose best model, pick best model, select best model, multiple algorithms, lgbm random forest xgboost, decision tree lgbm, all models, different models, various models, several algorithms, comparison plot, compare performance, roc curves, model metrics, benchmark models, evaluate models, test multiple models, try different models, multi model training, ensemble comparison, algorithm comparison, model benchmarking, performance comparison, multiple classifier training, various algorithm testing, comprehensive model evaluation, model selection process, algorithm evaluation, cross model comparison",
            "new_model": "Train single new model, build single new classifier, create single new predictor, develop single new algorithm, train fresh model, build from scratch, new model training, create classifier, develop predictor, train algorithm, build new, create new, fresh training, single machine learning model, single model development, single algorithm training, train fresh algorithm, develop fresh model, build brand new model, create entirely new model, individual model training, single predictor development"
        }
        
        # Use universal pattern classifier from toolbox
        result, method_used = pattern_classifier.classify_pattern(
            query,
            model_intent_definitions,
            use_case="model_sub_classification"
        )
        
        print_to_log(f"[ModelAgent] Model intent classification: {result} (method: {method_used})")
        return result
        
    except Exception as e:
        print_to_log(f"[ModelAgent] Universal classifier error: {e}, using keyword fallback")
        return fallback_classify_intent_keywords(query)


def llm_classify_model_intent(query: str) -> str:
    """LLM-based classification for model-specific intents"""
    try:
        import requests
        import json
        
        # Use the same LLM setup as the main orchestrator
        prompt = f"""
        Classify this query into one of these model intents:
        - "use_existing": User wants to use/apply an existing trained model
        - "multi_model": User wants to build/train/compare multiple models
        - "new_model": User wants to train/build a single new model
        
        Query: "{query}"
        
        Respond with ONLY the intent name (use_existing, multi_model, or new_model).
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
                intent = result.get("response", "").strip().lower()
                
                # Validate and clean response
                if "use_existing" in intent:
                    return "use_existing"
                elif "multi_model" in intent:
                    return "multi_model"
                elif "new_model" in intent:
                    return "new_model"
                    
        except Exception as ollama_error:
            print_to_log(f"[ModelAgent] Ollama LLM error: {ollama_error}")
            
        # Use keyword fallback instead of OpenAI
        print_to_log(f"[ModelAgent] Using keyword fallback for intent classification")
        return fallback_classify_intent_keywords(query)
            
    except Exception as e:
        print_to_log(f"[ModelAgent] LLM classification failed: {e}")
    
    return None

def fallback_classify_intent_keywords(query: str) -> str:
    """Keyword fallback classification logic - handles all model-specific intents"""
    query_lower = query.lower()
    
    # Use existing patterns for existing models (ENHANCED)
    use_existing_patterns = [
        "use this model", "use the model", "with this model", "for this model",
        "use the best model", "use best model", "apply the best model", "apply best model",
        "use this tree", "use the tree", "with this tree", "for this tree",
        "use this tree model", "use this decision tree", "apply this model",
        "existing model", "current model", "built model", "trained model",
        "show plot", "visualize tree", "display tree",
        "build segments", "build deciles", "build buckets", "build rankings",
        "rank ordering", "score", "predict", "classify"
    ]
    
    # Multi-model patterns
    multi_model_patterns = [
        "build multiple", "train several", "create multiple", "build three", "build 3",
        "compare models", "model comparison", "best model", "choose best", "pick best", "select best",
        "lgbm random forest", "xgboost random forest", "decision tree lgbm", "multiple algorithms",
        "all models", "different models", "various models", "several algorithms",
        "comparison plot", "compare performance", "roc curves", "model metrics",
        "benchmark models", "evaluate models", "test multiple", "try different"
    ]
    
    if any(pattern in query_lower for pattern in use_existing_patterns):
        return "use_existing"
    elif any(pattern in query_lower for pattern in multi_model_patterns):
        return "multi_model"
    
    # Default to new_model for single model cases
    return "new_model"

# Keep the original function name for backward compatibility
def fallback_classify_intent(query: str) -> str:
    """Wrapper for backward compatibility - now uses semantic classification"""
    return semantic_classify_model_intent(query)

def semantic_detect_plot_request(query: str) -> bool:
    """Semantic detection of plot/visualization requests"""
    
    try:
        # Import pattern classifier dynamically to avoid initialization issues
        try:
            from toolbox import pattern_classifier
            if not pattern_classifier:
                print_to_log("⚠️ Pattern classifier not initialized, using fallback classification")
                return False, "fallback"
        except ImportError:
            print_to_log("⚠️ Could not import pattern classifier, using fallback classification")
            return False, "fallback"
        
        # Define plot detection intent definitions
        plot_intent_definitions = {
            "plot_request": "Show plot, visualize tree, display tree, generate plot, create visualization, show visualization, plot tree, display model, visualize model, show decision tree, plot decision tree, tree visualization, model visualization, graphical representation, visual display",
            "no_plot": "Train model, build classifier, create predictor, develop algorithm, model training, algorithm development, machine learning, predictive modeling, classification, regression, model building, model creation"
        }
        
        # Use universal pattern classifier from toolbox
        result, method_used = pattern_classifier.classify_pattern(
            query,
            plot_intent_definitions,
            use_case="feature_detection"
        )
        
        print_to_log(f"[ModelAgent] Plot detection: {result} (method: {method_used})")
        return result == "plot_request"
        
    except Exception as e:
        print_to_log(f"[ModelAgent] Universal classifier plot detection error: {e}, using keyword fallback")
        # Keyword fallback
        plot_keywords = ['show', 'plot', 'visualize', 'display']
        tree_keywords = ['tree', 'decision tree', 'model']
        return any(pk in query.lower() for pk in plot_keywords) and any(tk in query.lower() for tk in tree_keywords)

def semantic_detect_financial_analysis(query: str) -> bool:
    """Semantic detection of financial analysis/rank ordering requests"""
    
    try:
        # Import pattern classifier dynamically to avoid initialization issues
        try:
            from toolbox import pattern_classifier
            if not pattern_classifier:
                print_to_log("⚠️ Pattern classifier not initialized, using fallback classification")
                return False, "fallback"
        except ImportError:
            print_to_log("⚠️ Could not import pattern classifier, using fallback classification")
            return False, "fallback"
        
        # Define financial analysis intent definitions
        financial_intent_definitions = {
            "financial_analysis": "Segment analysis, decile analysis, rank ordering, bucket analysis, bad rate analysis, coverage analysis, segmentation, ranking, financial segmentation, risk segmentation, score segmentation, decile buckets, rank order, performance buckets, risk buckets, score buckets, population segmentation",
            "regular_modeling": "Train model, build classifier, create predictor, develop algorithm, model training, algorithm development, machine learning, predictive modeling, classification, regression, model building, model creation, model development"
        }
        
        # Use universal pattern classifier from toolbox
        result, method_used = pattern_classifier.classify_pattern(
            query,
            financial_intent_definitions,
            use_case="feature_detection"
        )
        
        print_to_log(f"[ModelAgent] Financial analysis detection: {result} (method: {method_used})")
        return result == "financial_analysis"
        
    except Exception as e:
        print_to_log(f"[ModelAgent] Universal classifier financial analysis error: {e}, using keyword fallback")
        # Keyword fallback
        financial_keywords = ['segment', 'decile', 'rank', 'bucket', 'badrate', 'coverage', 'rank ordering', 'segmentation']
        return any(fk in query.lower() for fk in financial_keywords)

def llm_detect_plot_request(query: str) -> bool:
    """LLM-based detection of plot/visualization requests"""
    try:
        import requests
        
        prompt = f"""
        Does this query request a plot, visualization, or graphical display?
        
        Query: "{query}"
        
        Respond with ONLY "yes" or "no".
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
                answer = result.get("response", "").strip().lower()
                
                if "yes" in answer:
                    return True
                elif "no" in answer:
                    return False
                    
        except Exception as ollama_error:
            print_to_log(f"[ModelAgent] Ollama LLM error: {ollama_error}")
            
        # Use keyword fallback instead of OpenAI
        print_to_log(f"[ModelAgent] Using keyword fallback for plot detection")
        return keyword_detect_plot(query)
            
    except Exception as e:
        print_to_log(f"[ModelAgent] LLM plot detection failed: {e}")
    
    return None

def llm_detect_financial_analysis(query: str) -> bool:
    """LLM-based detection of financial analysis/rank ordering requests"""
    try:
        import requests
        
        prompt = f"""
        Does this query request financial analysis, segmentation, rank ordering, or decile analysis?
        
        Query: "{query}"
        
        Respond with ONLY "yes" or "no".
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
                answer = result.get("response", "").strip().lower()
                
                if "yes" in answer:
                    return True
                elif "no" in answer:
                    return False
                    
        except Exception as ollama_error:
            print_to_log(f"[ModelAgent] Ollama LLM error: {ollama_error}")
            
        # Use keyword fallback instead of OpenAI
        print_to_log(f"[ModelAgent] Using keyword fallback for financial analysis")
        financial_keywords = ['segment', 'decile', 'rank', 'bucket', 'badrate', 'coverage', 'rank ordering', 'segmentation']
        return any(fk in query.lower() for fk in financial_keywords)
            
    except Exception as e:
        print_to_log(f"[ModelAgent] LLM financial analysis failed: {e}")
    
    return None

# =============================================================================
# AGENT 2: CONTROLLER AGENT (ROUTER)
# =============================================================================

def controller_agent(state: AgentState) -> AgentState:
    """
    Routes requests based on intent and model state
    Acts as the central controller in your architecture
    """
    print_to_log(f"🎛️ CONTROLLER AGENT - Routing based on intent: {state['intent']}")
    
    # No progress update needed - internal routing is fast
    
    intent = state["intent"]
    user_id = state["user_id"]
    has_existing_model = state.get("has_existing_model", False)
    
    # Determine routing decision
    if intent == "new_model":
        routing_decision = "build_new_model"
        if has_existing_model:
            print_to_log("ℹ️ Building new model (existing model will be replaced)")
    
    elif intent == "multi_model":
        routing_decision = "build_multi_model"
        print_to_log(f"🔄 Multi-model comparison requested - will build and compare multiple models")
    
    elif intent == "use_existing":
        if has_existing_model:
            routing_decision = "use_existing_model"
        else:
            routing_decision = "no_model_available"
    
    elif intent == "code_execution":
        routing_decision = "execute_code"
    
    elif intent == "general_query":
        routing_decision = "general_response"
    
    else:
        routing_decision = "general_response"
    
    print_to_log(f"🚦 Routing decision: {routing_decision}")
    
    # Update state
    state["routing_decision"] = routing_decision
    state["messages"].append({
        "agent": "controller", 
        "content": f"Routing decision: {routing_decision}",
        "timestamp": datetime.now().isoformat()
    })
    
    return state

# =============================================================================
# AGENT 3: MODEL BUILDING AGENT
# =============================================================================

def model_building_agent(state: AgentState) -> AgentState:
    """
    Handles all model building, using, and visualization tasks
    Maps to your current send_to_llm and ExecutionAgent functions
    """
    print_to_log(f"🏗️ MODEL BUILDING AGENT - Processing: {state['routing_decision']}")
    
    # Get thread logger
    user_id = state["user_id"]
    if '_' in user_id:
        parts = user_id.split('_')
        actual_user_id = parts[0] if len(parts) >= 1 else user_id
        thread_id = '_'.join(parts[1:]) if len(parts) > 1 else user_id
    else:
        actual_user_id = user_id
        thread_id = user_id
    
    thread_logger = get_thread_logger(actual_user_id, thread_id)
    thread_logger.log_query(state["query"], intent=state.get("intent"), agent="model_building")
    
    # Get progress callback from state
    progress_callback = state.get("progress_callback")
    routing_decision = state.get("routing_decision", "")
    
    query = state["query"]
    routing_decision = state["routing_decision"]
    data = state["data"]
    intent = state["intent"]  # Extract intent from state
    
    # Handle cases where no data is available based on routing decision
    if data is None:
        if routing_decision == "general_response":
            # For general queries like "Hi", respond naturally using direct LLM call
            try:
                print_to_log(f"🔍 DEBUG: Generating conversational response for query: '{query}'")
                response = ollama.chat(
                    # model="krith/qwen2.5-coder-14b-instruct:IQ2_M",
                    model=MAIN_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a specialized AI assistant for data science and machine learning. You help users build models, analyze data, and work with datasets. When greeting users, be friendly and natural. When asked about capabilities, mention your ML/data science skills like building models, data analysis, visualization, etc. Keep responses conversational and concise."},
                        {"role": "user", "content": f"The user said: '{query}'. Respond in a friendly, natural way as an AI assistant. Do not list capabilities unless specifically asked. Keep it conversational and brief."}
                    ]
                )
                print_to_log(f"🔍 DEBUG: Raw LLM response: {response}")
                generated_response = response["message"]["content"].strip()
                print_to_log(f"🔍 DEBUG: Extracted response: '{generated_response}'")
                state["response"] = generated_response
                return state
            except Exception as e:
                print_to_log(f"🔥 Error generating conversational response: {e}")
                # Show error instead of hiding it
                state["response"] = f"❌ Error generating response: {str(e)}"
                return state
        else:
            # For model-specific requests, ask for data
            state["response"] = """📊 I need data to work with! Please upload a data file first.

Supported formats: CSV, Excel (.xlsx/.xls), JSON, TSV

Once you upload your data, I can help you build models and analyze it! 🎯"""
            return state
    
    # Modify prompt based on routing decision
    if routing_decision == "use_existing_model" and state.get("has_existing_model"):
        # Direct handling for existing model operations (no LLM codegen)
        print_to_log("🔧 Using existing model path without code generation...")
        user_model_path = state.get("model_path") or global_model_states.get(user_id, {}).get("model_path")
        # Auto-discover a recently saved model if path is missing or file doesn't exist
        if not user_model_path or not os.path.exists(user_model_path):
            try:
                import glob
                candidates: list[str] = []
                # 1) From last_result
                last_result = global_model_states.get(user_id, {}).get('last_result', {}) if global_model_states.get(user_id) else {}
                maybe_path = last_result.get('model_path')
                if maybe_path and os.path.exists(maybe_path):
                    candidates.append(maybe_path)
                
                # 2) Search in artifacts directory first, then root
                if '_' in user_id:
                    user_id_part, thread_id = user_id.split('_', 1)
                    artifacts_dir = get_username_artifacts_dir(user_id_part, thread_id)
                else:
                    artifacts_dir = None
                search_locations = []
                if artifacts_dir and os.path.exists(artifacts_dir):
                    search_locations.append(artifacts_dir)
                search_locations.append(".")  # Current directory as fallback
                
                username = get_username_for_user_id(user_id)
                for location in search_locations:
                    # Named by generator
                    candidates.extend(glob.glob(os.path.join(location, f"model_{username}_*.joblib")))
                    # Common fallback names
                    candidates.extend(glob.glob(os.path.join(location, "*model*.joblib")))
                    candidates.extend(glob.glob(os.path.join(location, "*.joblib")))
                # Pick the most recent existing file
                candidates = [p for p in candidates if os.path.exists(p)]
                if candidates:
                    user_model_path = max(candidates, key=os.path.getctime)
                    # Sync to state and global
                    state["model_path"] = user_model_path
                    if user_id not in global_model_states:
                        global_model_states[user_id] = {}
                    global_model_states[user_id]['model_path'] = user_model_path
                    if 'last_result' not in global_model_states[user_id]:
                        global_model_states[user_id]['last_result'] = {}
                    print_to_log(f"🔍 Auto-discovered model file: {user_model_path}")
            except Exception as _:
                pass
        if not user_model_path or not os.path.exists(user_model_path):
            state["response"] = "❌ No saved model found for this session. Please build a model first."
            return state

        try:
            current_model = joblib.load(user_model_path)
            print_to_log(f"✅ Loaded model from {user_model_path} | Type: {type(current_model).__name__}")
        except Exception as e:
            state["response"] = f"❌ Failed to load existing model: {e}"
            return state

        # Check for analysis/computation requests first (higher priority than just visualization)
        analysis_intent = any(k in query.lower() for k in ["rank ordering", "ranking", "decile", "bucket", "segment", "predict", "classify", "score", "table"])
        
        # If the request is to show/plot/visualize the decision tree, generate plot directly
        # BUT only if it's not asking for analysis/computation
        plot_intent = (not analysis_intent) and any(k in query.lower() for k in ["show", "plot", "visualize", "display"]) and any(
            t in query.lower() for t in ["tree", "decision tree", "decision-tree", "decisiontree"]
        )

        if plot_intent:
            # First try to return an existing saved plot without regenerating
            try:
                # 1) From last_result
                last_result = global_model_states.get(user_id, {}).get('last_result', {}) if global_model_states.get(user_id) else {}
                existing_path = last_result.get('plot_path')
                if existing_path and os.path.exists(existing_path):
                    state["execution_result"] = {"plot_path": existing_path}
                    state["artifacts"] = {"files": [existing_path]}
                    state["response"] = "📊 Decision tree visualization retrieved."
                    return state

                # 2) Search for a user-specific recent decision tree image
                import glob
                candidates = glob.glob(f"decision_tree*{user_id}*.png")
                if not candidates:
                    candidates = glob.glob("decision_tree*.png")
                if candidates:
                    latest_img = max(candidates, key=os.path.getctime)
                    state["execution_result"] = {"plot_path": latest_img}
                    state["artifacts"] = {"files": [latest_img]}
                    state["response"] = "📊 Decision tree visualization retrieved."
                    return state
            except Exception:
                pass

            # If no plot found on disk, generate from the loaded model
            # Prepare feature names from data if available (optional)
            feature_names = None
            if data is not None:
                try:
                    X_cols = data.drop('target', axis=1).columns if 'target' in data.columns else data.columns
                    feature_names = list(X_cols)
                except Exception:
                    feature_names = None

            # Only plot if it's a decision tree-like model
            if hasattr(current_model, 'tree_') or type(current_model).__name__.lower().startswith('decisiontree'):
                try:
                    # Local imports for plotting
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    from sklearn.tree import plot_tree

                    # Dynamic sizing
                    tree_depth = current_model.get_depth() if hasattr(current_model, 'get_depth') else current_model.tree_.max_depth
                    max_feature_len = max(len(name) for name in (feature_names or [])) if (feature_names and len(feature_names) > 0) else 10
                    safe_tree_depth = min(tree_depth, 15)
                    nodes_at_max_level = 2 ** safe_tree_depth
                    width = int(max(25, min(100, nodes_at_max_level * 0.1 * max_feature_len)))
                    height = int(max(15, min(50, safe_tree_depth * 3)))
                    font_size = int(max(6, min(12, 80 / (safe_tree_depth + (max_feature_len or 10)/10))))
                    max_depth_plot = min(5, safe_tree_depth)

                    plt.figure(figsize=(width, height))
                    plot_tree(
                        current_model,
                        filled=True,
                        feature_names=feature_names,
                        fontsize=font_size,
                        proportion=True,
                        rounded=True,
                        precision=2,
                        max_depth=max_depth_plot,
                    )
                    plt.tight_layout(pad=2.0)
                    ts = int(time.time())
                    plot_path = safe_plt_savefig(f"decision_tree_{user_id}_{ts}.png", bbox_inches='tight', dpi=300, facecolor='white')

                    state["execution_result"] = {"plot_path": plot_path}
                    state["artifacts"] = {"files": [plot_path]}
                    state["response"] = "📊 Decision tree visualization generated."
                    return state
                except Exception as e:
                    state["response"] = f"❌ Failed to generate decision tree plot: {e}"
                    return state
            else:
                state["response"] = "❌ The existing model is not a DecisionTree model, so a tree plot cannot be generated."
                return state

        # If not a plot intent, fall back to code generation path (e.g., predictions/segments)
        modified_prompt = f"""CRITICAL INSTRUCTION: THERE IS AN EXISTING TRAINED MODEL ALREADY AVAILABLE.

ABSOLUTELY FORBIDDEN - DO NOT DO ANY OF THESE:
- DO NOT create DecisionTreeClassifier() or any new model
- DO NOT use .fit() method
- DO NOT import or use any model creation code
- DO NOT retrain anything
- DO NOT generate any plots or visualizations unless specifically requested

REQUIRED - YOU MUST DO THIS:
- Use the variable 'current_model' which contains the already trained model
- For any predictions: current_model.predict() or current_model.predict_proba()
- The model is already fitted and ready to use

🚨 CRITICAL VALIDATION METRICS REQUIREMENTS:
- For VALIDATION METRICS: MUST use train_test_split() to get proper test data
- For RANK ORDERING/SEGMENTATION: Can use full dataset for business analysis
- NEVER calculate validation metrics on training data or full dataset
- Validation metrics measure model performance on UNSEEN data

VALIDATION METRICS PROCESS (when user asks for validation metrics, model performance, accuracy, etc.):
1. Split data: X = sample_data.drop('TARGET_COLUMN', axis=1); y = sample_data['TARGET_COLUMN']
2. Create test split: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
3. Use current_model.predict(X_test) and current_model.predict_proba(X_test) for test predictions
4. Calculate metrics ONLY on test data: accuracy_score(y_test, y_pred), etc.
5. This ensures proper validation on unseen data

RANK ORDERING/SEGMENTATION PROCESS (when user asks for rank ordering, deciles, buckets):
🚨 CRITICAL DEFAULT: ALWAYS CREATE RANK ORDERING ON TEST DATA UNLESS USER EXPLICITLY ASKS FOR FULL DATASET

DEFAULT BEHAVIOR (most queries):
1. Split data: X = sample_data.drop('TARGET_COLUMN', axis=1); y = sample_data['TARGET_COLUMN']
2. Create test split: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
3. Get TEST probabilities: test_probabilities = current_model.predict_proba(X_test)[:,1]
4. Create rank_df with TEST data only: pd.DataFrame({{'actual': y_test.values, 'probability': test_probabilities}})
5. Create segments with pd.qcut(test_probabilities, q=N, duplicates='drop')
6. Calculate badrate, coverage, cumulative metrics per the RANK_ORDERING_PROMPT guidelines
7. Store ONLY in 'rank_ordering_table': result['rank_ordering_table'] = rank_metrics.to_dict('records')

ONLY IF USER EXPLICITLY ASKS FOR FULL DATASET (mentions "full data", "full dataset", "all data", "entire dataset"):
1. Split data: X = sample_data.drop('TARGET_COLUMN', axis=1); y = sample_data['TARGET_COLUMN']
2. Get FULL dataset probabilities: full_probabilities = current_model.predict_proba(X)[:,1]
3. Create rank_df with FULL data: pd.DataFrame({{'actual': y.values, 'probability': full_probabilities}})
4. Create segments with pd.qcut(full_probabilities, q=N, duplicates='drop')
5. Calculate badrate, coverage, cumulative metrics per the RANK_ORDERING_PROMPT guidelines
6. Store in 'rank_ordering_table_full': result['rank_ordering_table_full'] = rank_metrics.to_dict('records')

🚨 DO NOT CREATE BOTH TABLES UNLESS EXPLICITLY REQUESTED
🚨 DEFAULT: Use TEST data only and store as 'rank_ordering_table' (without _test or _full suffix)

ALWAYS include ALL required columns: bucket, badrate, totalUsersCount, cum_badrate, coverage, avg_probability, min_threshold, max_threshold

Your specific task: {query}

REMEMBER: Use 'current_model' for everything. Do not create any new models. DEFAULT to TEST data rank ordering only."""
        
    elif routing_decision == "no_model_available":
        state["response"] = """❌ You're asking to use an existing model, but no model has been built yet in this session.

Please build a model first with commands like:
- "build a decision tree model"
- "create a random forest classifier"
- "train an LGBM model"

Then you can use it for predictions, visualizations, and analysis."""
        return state
        
    elif routing_decision == "general_response":
        # Use LLM to generate natural conversational responses
        try:
            if data is not None:
                context_prompt = f"The user said: '{query}'. I have their dataset with {data.shape[0]:,} rows and {data.shape[1]} columns. Respond naturally and conversationally. Only mention specific capabilities if they ask 'what can you do' or similar questions."
            else:
                context_prompt = f"The user said: '{query}'. Respond naturally and conversationally as an AI assistant. Don't list capabilities unless they specifically ask what you can do."
            
            print_to_log(f"🔍 DEBUG: Generating conversational response (with data: {data is not None}) for query: '{query}'")
            # Use ollama directly for conversational responses (not code generation)
            response = ollama.chat(
                # model="krith/qwen2.5-coder-14b-instruct:IQ2_M",
                model=MAIN_MODEL,
                messages=[
                    {"role": "system", "content": "You are a specialized AI assistant for data science and machine learning. You help users build models, analyze data, and work with datasets. When greeting users, be friendly and natural. When asked about capabilities, mention your ML/data science skills like building models, data analysis, visualization, etc. Keep responses conversational and concise."},
                    {"role": "user", "content": context_prompt}
                ]
            )
            
            print_to_log(f"🔍 DEBUG: Raw LLM response: {response}")
            generated_response = response["message"]["content"].strip()
            print_to_log(f"🔍 DEBUG: Extracted response: '{generated_response}'")
            state["response"] = generated_response
            return state
        except Exception as e:
            print_to_log(f"🔥 Error generating conversational response: {e}")
            # Show error instead of hiding it
            state["response"] = f"❌ Error generating conversational response: {str(e)}"
            return state
    
    elif routing_decision == "execute_code":
        # Generate and execute Python code for general data analysis
        has_data = data is not None and not data.empty
        data_info = f"shape {data.shape}" if has_data else "No data available"
        
        code_prompt = f"""Generate Python code to fulfill this request. DataFrame status: {data_info}.

User request: {query}

CRITICAL DATA HANDLING:
- DataFrame variable name: 'sample_data'
- Data available: {has_data}
- If no data: provide informative error message and return early
- If data available: proceed with analysis

Requirements:
- Write complete, executable Python code
- ALWAYS check if sample_data is empty before proceeding
- For model building: verify required columns exist
- Assign final results to a variable called 'result'  
- For plots, save them using safe_plt_savefig() and add path to result
- Import only model and metrics libraries explicitly (no scalers/pipelines)
- Convert numpy types to Python native types (float(), int(), .tolist())

Available libraries: pandas, numpy, matplotlib, seaborn, sklearn, etc.

PREPROCESSED DATA ASSUMPTION:
- The dataset is already preprocessed. Do NOT add StandardScaler, MinMaxScaler, ColumnTransformer, or Pipeline unless user explicitly requests preprocessing.
  * For tree/ensemble models, NEVER scale features.
  * Fit models directly on X_train, y_train.

Example handling for no data:
```python
if sample_data.empty:
    result = {{
        'error': 'No data available. Please upload a dataset first.',
        'suggestion': 'Use the file upload feature to load CSV, Excel, or other data files.'
    }}
else:
    # Proceed with analysis
    pass
```"""

        try:
            print_to_log("🤔 Generating code for general analysis...")
            # Start thinking animation for LLM call
            if progress_callback:
                progress_callback("🤔 Generating code using AI...", "Code Generation")
            
            reply, code, system_prompt = generate_model_code(code_prompt, user_id, query)
            
            if not code.strip():
                state["response"] = reply
                return state
            
            state["code"] = code
            print_to_log(f"📝 GENERAL CODE EXECUTION - Generated code ({len(code)} chars):")
            print_to_log("=" * 60)
            print_to_log(code)
            print_to_log("=" * 60)
            
            # Execute the code
            print_to_log("⚙️ Executing generated code...")
            # Get artifacts directory for this thread
            artifacts_dir = None
            try:
                # Extract user and thread from user_id format: user_threadts
                if "_" in user_id:
                    user, thread_ts = user_id.split("_", 1)
                    # Get username for proper directory structure
                    username = get_username_for_user_id(user)
                    thread_dir = os.path.join("user_data", username, thread_ts)
                    artifacts_dir = os.path.join(thread_dir, "artifacts")
                    if not os.path.exists(artifacts_dir):
                        os.makedirs(artifacts_dir, exist_ok=True)
                        print_to_log(f"📁 Created artifacts directory: {artifacts_dir}")
            except Exception as e:
                print_to_log(f"⚠️ Failed to create artifacts directory: {e}")
                artifacts_dir = None
            
            # ExecutionAgent will handle progress updates
            
            result = ExecutionAgent(
                code, 
                data if data is not None else pd.DataFrame(), 
                user_id=user_id,
                verbose=True,
                user_query=query,
                intent=intent,
                model_states=global_model_states,
                artifacts_dir=artifacts_dir,
                progress_callback=progress_callback,  # Pass progress callback to ExecutionAgent
                original_system_prompt=system_prompt  # Pass original system prompt for error fixing
            )
            
            print_to_log(f"🔍 EXECUTION RESULT DEBUG:")
            print_to_log(f"   📝 Result type: {type(result)}")
            print_to_log(f"   📝 Result value: {repr(result)}")
            print_to_log(f"   📝 Is string: {isinstance(result, str)}")
            if isinstance(result, str):
                print_to_log(f"   📝 Contains 'error': {'error' in result.lower()}")
            
            if isinstance(result, str) and "error" in result.lower():
                print_to_log(f"🚨 DETECTED ERROR STRING - Setting execution_result to None")
                state["execution_result"] = None  # Signal execution failure to Slack
                state["error_message"] = result  # Store actual error for logging
                state["response"] = f"❌ {result}"
                return state
            
            # Store results - set to None if execution failed (error string)
            if isinstance(result, str) and ("error" in result.lower() or "failed" in result.lower() or "exception" in result.lower()):
                state["execution_result"] = None  # Signal execution failure to Slack
                state["error_message"] = result  # Store actual error for logging
            else:
                state["execution_result"] = result
            
            # Format response based on result type
            if isinstance(result, dict):
                response_parts = []
                if 'plot_path' in result:
                    response_parts.append("📊 Visualization created successfully!")
                if any(key for key in result.keys() if key != 'plot_path'):
                    response_parts.append("✅ Analysis completed successfully!")
                state["response"] = " ".join(response_parts) if response_parts else "✅ Code executed successfully!"
            else:
                state["response"] = "✅ Code executed successfully!"
            
            return state
            
        except Exception as e:
            print_to_log(f"💥 Code execution failed: {e}")
            state["response"] = f"❌ Code execution failed: {str(e)}"
            return state
    
    elif routing_decision == "build_multi_model":
        # NEW: Multi-model comparison workflow
        print_to_log("🔄 Starting multi-model comparison workflow...")
        
        if progress_callback:
            progress_callback("🔄 Processing your request...", "Multi-Model Training")
        
        # Check if data is available for model building
        if data is None:
            state["response"] = """📊 I need data to work with! Please upload a data file first.

Supported formats: CSV, Excel (.xlsx/.xls), JSON, TSV

Once you upload your data, I can build multiple models and compare them! 🎯"""
            return state
        
        # NEW MODULAR ORCHESTRATOR APPROACH - No more massive prompts!
        try:
            result = handle_multi_model_orchestrator(query, data, user_id, progress_callback)
            
            if result and isinstance(result, dict):
                # Use format_multi_model_response for rich, detailed output
                response = format_multi_model_response(result, query)
                state["response"] = response
                state["model_building_state"] = result
                state["execution_result"] = result
                
                # Handle plot uploads for multi-model comparison
                plots = result.get('comparison_plots', {})
                if plots:
                    plot_files = []
                    for plot_type, plot_path in plots.items():
                        if plot_path and os.path.exists(plot_path):
                            plot_files.append({
                                "path": plot_path, 
                                "title": f"Multi-Model {plot_type.replace('_', ' ').title()}", 
                                "type": "comparison_plot"
                            })
                    
                    if plot_files:
                        state["artifacts"] = state.get("artifacts", {})
                        state["artifacts"]["files"] = plot_files
                        result["artifacts"] = {"files": plot_files}
                        print_to_log(f"📊 {len(plot_files)} comparison plots ready for upload")
                
                return state
            else:
                state["response"] = "❌ Multi-model comparison failed: Orchestrator returned no results"
                return state
                
        except Exception as e:
            print_to_log(f"❌ Multi-model orchestrator failed: {str(e)}")
            import traceback
            print_to_log(f"💥 Full traceback: {traceback.format_exc()}")
            state["response"] = f"❌ Multi-model comparison failed: {str(e)}"
            return state
        
        # DEAD CODE BELOW - TO BE REMOVED IN NEXT COMMIT
        multi_model_prompt = f"""DEAD CODE - This entire prompt is dead code and will never execute.

You are building a comprehensive multi-model comparison system. 

USER REQUEST: {query}

🚨 CRITICAL SUCCESS REQUIREMENTS:
1. Build multiple ML models based on USER'S SPECIFIC REQUEST (minimum 2 models)
2. Train all models with identical train/test splits 
3. Generate comprehensive metrics for each model
4. Create comparison visualizations (ROC curves, metric comparison table)
5. Automatically select the best model based on USER-SPECIFIED or default metric
6. Return detailed results for all models + best model selection
7. MANDATORY: Code must be complete, executable, and return the exact result format specified

DYNAMIC USER INPUT PARSING:
- MODELS: Extract specific model names from user query. If none specified, use: RandomForest, XGBoost, LightGBM                                                                                                    
- TEST SIZE: Look for "test", "split", "validation" mentions. Default: 0.2 (20%)
- BEST MODEL METRIC: Look for "best based on", "select by", "choose using". Default: accuracy for classification, r2 for regression                                                                                     
- Handle ANY model names user mentions (LogisticRegression, SVM, XGBoost, Neural Network, etc.)

🚨 CRITICAL: USE THIS EXACT MODEL PARSING CODE (DO NOT MODIFY):
```python
# MANDATORY: Use this exact model parsing approach - DO NOT use split() or regex
user_request = "{query}"
user_request_lower = user_request.lower()
models_requested = []

# Define model aliases and keywords - EXACTLY as shown
model_keywords = {{
    'lgbm': ['lgbm', 'lightgbm', 'light gbm'],
    'xgboost': ['xgboost', 'xgb', 'extreme gradient boosting'],
    'random_forest': ['random forest', 'randomforest', 'rf'],
    'decision_tree': ['decision tree', 'decisiontree', 'dt'],
    'neural_network': ['neural network', 'nn', 'mlp', 'neural net'],
    'logistic_regression': ['logistic regression', 'logistic', 'lr'],
    'svm': ['svm', 'support vector machine']
}}

# Extract models mentioned in query - EXACTLY as shown
for model_name, keywords in model_keywords.items():
    if any(keyword in user_request_lower for keyword in keywords):
        models_requested.append(model_name)

# If no models found, use defaults - EXACTLY as shown
if not models_requested:
    models_requested = ['random_forest', 'xgboost', 'lgbm']

print(f"Models requested: {{models_requested}}")
```

🚨 FORBIDDEN: DO NOT use user_request.split('models') or .split('and') - these approaches are broken!

🚨 CRITICAL: USE THIS EXACT MODEL AVAILABILITY CODE (DO NOT MODIFY):
```python
# Build available models dictionary - EXACTLY as shown
available_models = {{}}

try:
    from sklearn.ensemble import RandomForestClassifier
    available_models['Random Forest'] = RandomForestClassifier(random_state=42)
except ImportError:
    print("Warning: sklearn RandomForest not available")

try:
    from lightgbm import LGBMClassifier
    available_models['LightGBM'] = LGBMClassifier(verbosity=-1, random_state=42)
except ImportError:
    print("Warning: LightGBM not available")

try:
    from xgboost import XGBClassifier
    available_models['XGBoost'] = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
except ImportError:
    print("Warning: XGBoost not available")

try:
    from sklearn.neural_network import MLPClassifier
    available_models['Neural Network'] = MLPClassifier(random_state=42, max_iter=500)
except ImportError:
    print("Warning: Neural Network not available")

try:
    from sklearn.tree import DecisionTreeClassifier
    available_models['Decision Tree'] = DecisionTreeClassifier(max_depth=5, random_state=42)
except ImportError:
    print("Warning: Decision Tree not available")

try:
    from sklearn.linear_model import LogisticRegression
    available_models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
except ImportError:
    print("Warning: Logistic Regression not available")

try:
    from sklearn.svm import SVC
    available_models['SVM'] = SVC(random_state=42, probability=True)
except ImportError:
    print("Warning: SVM not available")

print(f"Available models: {{list(available_models.keys())}}")
```

🚨 CRITICAL: USE THIS EXACT MODEL MAPPING CODE (DO NOT MODIFY):
```python
# Map requested models to available models - EXACTLY as shown
model_name_mapping = {{
    'lgbm': 'LightGBM',
    'xgboost': 'XGBoost',
    'random_forest': 'Random Forest',
    'decision_tree': 'Decision Tree',
    'neural_network': 'Neural Network',
    'logistic_regression': 'Logistic Regression',
    'svm': 'SVM'
}}

# Build final models to train - EXACTLY as shown
models_to_train = {{}}
for requested_model in models_requested:
    mapped_name = model_name_mapping.get(requested_model)
    if mapped_name and mapped_name in available_models:
        models_to_train[mapped_name] = available_models[mapped_name]
    else:
        print(f"Warning: {{requested_model}} -> {{mapped_name}} not available")

# Check if we have enough models for comparison
if len(models_to_train) < 2:
    available_list = list(available_models.keys())
    requested_list = models_requested
    error_msg = f"Insufficient Models for Comparison. Requested: {{requested_list}}, Available: {{available_list}}, Found: {{len(models_to_train)}} (need minimum 2)"
    raise ValueError(error_msg)

print(f"Final models to build: {{list(models_to_train.keys())}}")
```

🚨 CRITICAL: USE EXISTING DATA - DO NOT CREATE FAKE DATA:
```python
# MANDATORY: Use existing sample_data - DO NOT create fake data
X = sample_data.drop('target', axis=1)
y = sample_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Using existing data: {{sample_data.shape}}")
print(f"Features: {{X.shape[1]}}, Target: {{y.name}}")
```

🚨 CRITICAL: USE SAFE ARTIFACT FUNCTIONS:
```python
# MANDATORY: Use safe functions for saving - DO NOT use plt.savefig directly
model_path = safe_joblib_dump(model, f'{{model_name}}_model.joblib')
plot_path = safe_plt_savefig('roc_comparison.png')

# These functions automatically handle artifacts directory structure
```

DATA REQUIREMENTS:
- Use 'sample_data' DataFrame (already loaded in environment)
- Target column: 'target'  
- Data is preprocessed (no scalers/encoders needed)
- Use train_test_split with user-specified test_size or default 0.2, random_state=42

🚨 MANDATORY IMPORTS (include ALL of these):
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, log_loss, roc_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
```

ROBUST LIBRARY HANDLING WITH FALLBACKS:
1. Import libraries with try-except blocks
2. If a model library is missing, skip that model with warning message
3. Continue with available models (minimum 2 required)
4. ALWAYS ensure at least 2 models are available
5. Example:
```python
models = {{}}
try:
    from sklearn.ensemble import RandomForestClassifier
    models['Random Forest'] = RandomForestClassifier()
except ImportError:
    print_to_log("Warning: sklearn RandomForest not available")

try:
    from lightgbm import LGBMClassifier
    # 🚨 FIX: Use verbosity=-1 to prevent verbose_eval errors
    models['LightGBM'] = LGBMClassifier(verbosity=-1, random_state=42)
except ImportError:
    print_to_log("Warning: LightGBM not available")
    

try:
    from xgboost import XGBClassifier
    # 🚨 FIX: Use verbosity=0 for clean output
    models['XGBoost'] = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
except ImportError:
    print("Warning: XGBoost not available")
```

DYNAMIC MODEL DICTIONARY:
- Build models dictionary based on user's specific request
- Handle model aliases (e.g., "rf" = RandomForest, "lgbm" = LightGBM, "xgb" = XGBoost)
- If user says "3 models" or "5 models", include that many
- If user lists specific models, use exactly those models

FLEXIBLE BEST MODEL SELECTION:
- Parse user's preference from query for selection criteria
- Common metrics: accuracy, precision, recall, f1, roc_auc, r2, mse, mae
- Default: accuracy (classification) or r2 (regression)

🚨 CRITICAL ARTIFACT PATH HANDLING:
- Use safe_joblib_dump() for model saving (automatically handles paths)
- Use safe_plt_savefig() for plot saving (automatically handles paths)
- DO NOT use hardcoded paths like '/path/to/model.joblib'
- Let the safe functions handle proper directory structure
- Example: model_path = safe_joblib_dump(model, f'{{model_name}}_model.joblib')
- Example: plot_path = safe_plt_savefig('roc_comparison.png')

MANDATORY CODE STRUCTURE:
1. Import libraries with error handling
2. Parse user requirements (models, test_size, metric)
3. Data splitting (X/y, train/test with dynamic test_size)
4. Build dynamic models dictionary based on user input
5. Train all available models and collect metrics
6. Create comparison visualizations with safe_plt_savefig()
7. Select best model based on user-specified or default metric
8. Save all artifacts using safe functions (models + plots)

🚨 CRITICAL ARRAY HANDLING FIX: For ROC curve generation, convert probabilities to numpy array:
```python
# Create comparison visualizations
plt.figure(figsize=(10, 8))
for model_name, result in results.items():
    if result['probabilities'] is not None:
        # 🚨 FIX: Convert list to numpy array to prevent indexing errors
        y_proba_array = np.array(result['probabilities'])
        if y_proba_array.ndim == 2 and y_proba_array.shape[1] >= 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba_array[:, 1])
        else:
            fpr, tpr, _ = roc_curve(y_test, y_proba_array)
        plt.plot(fpr, tpr, label=f'{{model_name}} (AUC = {{result["metrics"]["roc_auc"]:.2f}})')
```

RESULT DICTIONARY REQUIREMENTS:
result = {{
    'user_config': {{
        'models_requested': ['model1', 'model2', ...],
        'test_size': float,
        'selection_metric': 'metric_name',
        'total_models_built': int
    }},
    'models': {{model_name: {{
        'model': model_object,
        'model_path': saved_path,
        'metrics': {{accuracy, precision, recall, f1, roc_auc, mse, mae, r2, etc.}},
        'predictions': y_pred,
        'probabilities': y_proba_or_none,
        'training_time': float,
        'model_type': 'classification_or_regression'
    }}}},
    'best_model': {{
        'name': best_model_name,
        'model': best_model_object, 
        'model_path': best_model_path,
        'metrics': best_model_metrics,
        'selection_criteria': 'accuracy: 0.XX (user requested)' or 'roc_auc: 0.XX (default)',
        'improvement_over_worst': 'X% better than worst model'
    }},
    'comparison_plots': {{
        'roc_curves': 'path/to/roc_comparison.png',
        'metrics_table': 'path/to/metrics_table.png'
    }},
    'model_ranking': [{{
        'rank': 1,
        'model_name': 'best_model',
        'score': float,
        'metric_used': 'metric_name'
    }}],
    'summary': {{
        'total_models': int,
        'best_model': 'model_name',
        'best_score': float,
        'worst_model': 'model_name', 
        'worst_score': float,
        'performance_spread': 'X% difference between best and worst',
        'recommendation': 'Detailed recommendation text'
    }},
    'detailed_comparison': 'Comprehensive textual comparison of all models with strengths/weaknesses'
}}

🚨 CRITICAL: FINAL RESULT MUST BE NAMED 'result' (NOT 'results'):
```python
# MANDATORY: Create result dictionary with exact structure
result = {{
    'user_config': {{
        'models_requested': models_requested,
        'test_size': test_size,
        'selection_metric': selection_metric,
        'total_models_built': len(models_to_train)
    }},
    'models': models_results,  # Dictionary of all trained models
    'best_model': {{
        'name': best_model_name,
        'model': best_model_object,
        'model_path': best_model_path,
        'metrics': best_model_metrics,
        'selection_criteria': f'{{selection_metric}}: {{best_score:.4f}} (user requested)',
        'improvement_over_worst': f'{{improvement_percentage:.2f}}% better than worst model'
    }},
    'comparison_plots': {{
        'roc_curves': roc_plot_path,
        'metrics_table': metrics_plot_path
    }},
    'model_ranking': ranking_list,
    'summary': summary_dict,
    'detailed_comparison': detailed_text
}}

# MANDATORY: Validate result format before returning
required_keys = ['user_config', 'models', 'best_model', 'comparison_plots', 'model_ranking', 'summary', 'detailed_comparison']
for key in required_keys:
    if key not in result:
        print(f"ERROR: Missing required key: {{key}}")
        result[key] = {{}} if key != 'detailed_comparison' else "Analysis not available"

print("✅ Multi-model comparison result format validated")
print(f"✅ Result keys: {{list(result.keys())}}")
print(f"✅ Models trained: {{list(result['models'].keys())}}")
```

ENHANCED RESPONSE FORMATTING:
- Print detailed model performance table to console
- Show training time for each model
- Display model ranking with scores
- Include recommendation for model selection
- Show performance improvement percentages
- CRITICAL: Always validate result format before returning

🚨 EXECUTION ROBUSTNESS REQUIREMENTS:
1. Use try-except blocks around model training
2. If a model fails, continue with other models
3. Ensure at least 2 models succeed
4. Always return the complete result dictionary
5. Include error handling for plotting
6. Validate all required keys exist in result

Generate complete, executable Python code that implements this dynamic multi-model comparison system with robust error handling."""
        try:
            print_to_log("🤔 Generating multi-model comparison code...")
            if progress_callback:
                progress_callback("🤔 Generating code...", "Code Generation")
            
            reply, code, system_prompt = generate_model_code(multi_model_prompt, user_id, query)
            
            if not code.strip():
                state["response"] = f"❌ Failed to generate multi-model code: {reply}"
                return state
            
            state["code"] = code
            print_to_log(f"📝 MULTI-MODEL CODE - Generated ({len(code)} chars)")
            
            # Execute the multi-model comparison code
            print_to_log("⚙️ Executing multi-model comparison...")
            
            # Get artifacts directory
            artifacts_dir = None
            try:
                if "_" in user_id:
                    user, thread_ts = user_id.split("_", 1)
                    # Get username for proper directory structure
                    username = get_username_for_user_id(user)
                    thread_dir = os.path.join("user_data", username, thread_ts)
                    artifacts_dir = os.path.join(thread_dir, "artifacts")
                    if not os.path.exists(artifacts_dir):
                        os.makedirs(artifacts_dir, exist_ok=True)
            except Exception as e:
                print_to_log(f"⚠️ Failed to create artifacts directory: {e}")
            
            # ExecutionAgent will handle progress
            
            result = ExecutionAgent(
                code, 
                data, 
                user_id=user_id,
                verbose=True,
                user_query=query,
                intent=intent,
                model_states=global_model_states,
                artifacts_dir=artifacts_dir,
                progress_callback=progress_callback,
                original_system_prompt=system_prompt  # Pass original system prompt for error fixing
            )
            
            # Store results and format comprehensive response
            state["execution_result"] = result
            
            # Enhanced result validation and diagnostic logging
            print_to_log(f"🔍 Multi-model result type: {type(result)}")
            if isinstance(result, dict):
                print_to_log(f"🔍 Multi-model result keys: {list(result.keys())}")
            else:
                print_to_log(f"🔍 Multi-model result content: {str(result)[:200]}...")
            
            if isinstance(result, dict) and 'models' in result:
                # Format comprehensive multi-model response
                response_parts = ["✅ 🎯 Multi-Model Comparison Completed Successfully!\n"]
                
                # User configuration
                user_config = result.get('user_config', {})
                if user_config:
                    response_parts.append(f"⚙️ Configuration:")
                    response_parts.append(f"   • Models Requested: {', '.join(user_config.get('models_requested', []))}")
                    response_parts.append(f"   • Test Split: {user_config.get('test_size', 0.2):.0%}")
                    response_parts.append(f"   • Selection Metric: {user_config.get('selection_metric', 'accuracy')}")
                    response_parts.append(f"   • Models Built: {user_config.get('total_models_built', 0)}\n")
                
                # Model performance summary
                models = result.get('models', {})
                response_parts.append(f"📊 Model Performance Summary:")
                for model_name, model_data in models.items():
                    metrics = model_data.get('metrics', {})
                    training_time = model_data.get('training_time', 0)
                    model_type = model_data.get('model_type', 'unknown')
                    
                    # Show relevant metrics based on problem type
                    if model_type == 'classification':
                        acc = metrics.get('accuracy', 0)
                        auc = metrics.get('roc_auc', 0)
                        f1 = metrics.get('f1_score', 0)
                        response_parts.append(f"   🤖 {model_name}: Acc: {acc:.3f} | AUC: {auc:.3f} | F1: {f1:.3f} | Time: {training_time:.2f}s")
                    else:  # regression
                        r2 = metrics.get('r2_score', 0)
                        mae = metrics.get('mae', 0)
                        rmse = metrics.get('rmse', 0)
                        response_parts.append(f"   🤖 {model_name}: R²: {r2:.3f} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | Time: {training_time:.2f}s")
                
                # Model ranking
                ranking = result.get('model_ranking', [])
                if ranking and len(ranking) > 1:
                    response_parts.append(f"\n🏅 Model Ranking:")
                    for rank_info in ranking[:3]:  # Show top 3
                        rank = rank_info.get('rank', 0)
                        model_name = rank_info.get('model_name', 'Unknown')
                        score = rank_info.get('score', 0)
                        metric = rank_info.get('metric_used', 'score')
                        response_parts.append(f"   #{rank}. {model_name}: {metric.upper()} = {score:.3f}")
                
                # Best model details
                best_model = result.get('best_model', {})
                if best_model:
                    response_parts.append(f"\n🏆 Winner: {best_model.get('name', 'Unknown')}")
                    response_parts.append(f"   📈 Selection: {best_model.get('selection_criteria', 'Not specified')}")
                    improvement = best_model.get('improvement_over_worst', '')
                    if improvement:
                        response_parts.append(f"   📊 Performance: {improvement}")
                
                # Summary insights
                summary_data = result.get('summary', {})
                if isinstance(summary_data, dict):
                    response_parts.append(f"\n📋 Analysis:")
                    response_parts.append(f"   • Best: {summary_data.get('best_model', 'Unknown')} ({summary_data.get('best_score', 0):.3f})")
                    response_parts.append(f"   • Worst: {summary_data.get('worst_model', 'Unknown')} ({summary_data.get('worst_score', 0):.3f})")
                    spread = summary_data.get('performance_spread', '')
                    if spread:
                        response_parts.append(f"   • Performance Spread: {spread}")
                    
                    recommendation = summary_data.get('recommendation', '')
                    if recommendation:
                        response_parts.append(f"\n💡 Recommendation: {recommendation}")
                elif isinstance(summary_data, str) and summary_data:
                    response_parts.append(f"\n📋 Summary: {summary_data}")
                
                # Visualizations
                plots = result.get('comparison_plots', {})
                if plots:
                    response_parts.append(f"\n📊 Visualizations Generated:")
                    if plots.get('roc_curves'):
                        response_parts.append(f"   • ROC Curves Comparison Plot")
                    if plots.get('metrics_table'):
                        response_parts.append(f"   • Metrics Comparison Table")
                
                # Detailed comparison
                detailed_comparison = result.get('detailed_comparison', '')
                if detailed_comparison:
                    response_parts.append(f"\n🔍 Detailed Analysis:")
                    response_parts.append(f"{detailed_comparison}")
                
                state["response"] = "\n".join(response_parts)
                
                # Handle plot uploads for multi-model comparison
                if plots:
                    plot_files = []
                    for plot_type, plot_path in plots.items():
                        if plot_path and os.path.exists(plot_path):
                            plot_files.append({
                                "path": plot_path, 
                                "title": f"Multi-Model {plot_type.replace('_', ' ').title()}", 
                                "type": "comparison_plot"
                            })
                    
                    if plot_files:
                        state["artifacts"] = state.get("artifacts", {})
                        state["artifacts"]["files"] = plot_files
                        result["artifacts"] = {"files": plot_files}
                        print_to_log(f"📊 {len(plot_files)} comparison plots ready for upload")
            elif isinstance(result, dict):
                # Result is a dict but missing 'models' key - provide detailed diagnostic
                available_keys = list(result.keys())
                print_to_log(f"⚠️ Multi-model result missing 'models' key. Available keys: {available_keys}")
                
                # Try to extract any useful information from the result
                error_info = result.get('error', 'Unknown execution error')
                execution_status = result.get('execution_status', 'unknown')
                
                state["response"] = f"""❌ Multi-model comparison failed: Result format issue

🔍 Diagnostic Information:
• Result type: Dictionary
• Available keys: {', '.join(available_keys)}
• Missing required key: 'models'
• Execution status: {execution_status}
• Error details: {error_info}

💡 This suggests the generated code didn't complete successfully or didn't return the expected result format."""
                
            else:
                # Result is not a dict at all
                result_type = type(result).__name__
                result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                print_to_log(f"⚠️ Multi-model result is not a dictionary. Type: {result_type}")
                
                state["response"] = f"""❌ Multi-model comparison failed: Unexpected result format

🔍 Diagnostic Information:
• Expected: Dictionary with 'models' key
• Received: {result_type}
• Content preview: {result_preview}

💡 This suggests the code execution failed or returned an unexpected data type."""
            
            return state
            
        except Exception as e:
            print_to_log(f"💥 Multi-model comparison failed: {e}")
            import traceback
            print_to_log(f"💥 Full traceback: {traceback.format_exc()}")
            state["response"] = f"❌ Multi-model comparison failed: {str(e)}"
            return state
        
    else:  # build_new_model
        # Check if data is available for model building
        if data is None:
            # For model building requests without data, ask for data upload
            state["response"] = """📊 I need data to work with! Please upload a data file first.

Supported formats: CSV, Excel (.xlsx/.xls), JSON, TSV

Once you upload your data, I can help you build models and analyze it! 🎯"""
            return state
        
        # Check if plot/visualization is requested using semantic classification
        should_generate_plot = semantic_detect_plot_request(query)
        
        # AUTOMATIC DECISION TREE PLOTTING - Always generate plot for decision trees
        decision_tree_keywords = ['decision tree', 'decision-tree', 'decisiontree', 'tree classifier', 'tree regressor']
        is_decision_tree_request = any(dt_keyword in query.lower() for dt_keyword in decision_tree_keywords)
        
        # Auto-generate plot for decision trees regardless of explicit request
        if is_decision_tree_request:
            should_generate_plot = True
        
        # Check if financial segmentation/rank ordering is requested using semantic classification
        should_generate_ranking = semantic_detect_financial_analysis(query)
        
        # Check if multiple models comparison is requested
        comparison_keywords = ['compare', 'comparison', 'multiple models', 'best model', 'model comparison']
        should_compare_models = any(ck in query.lower() for ck in comparison_keywords)
        
        modified_prompt = query
        # Enforce preprocessed-data rules in generation
        modified_prompt += "\n\nASSUME DATA IS PREPROCESSED. Do NOT add scalers, encoders, ColumnTransformer, or Pipelines. Fit models directly.\n\nLIBRARY USAGE POLICY:\n- Use any library the user requests (lightgbm, xgboost, catboost, optuna, bayes_opt, etc.)\n- Always wrap imports in try-except blocks with informative error messages\n- If a library is not available, provide sklearn alternatives as fallback\n- For missing libraries, print a clear message: 'Library [name] not installed. Please ask admin to install: pip install [name]'"
        if should_generate_plot:
            if is_decision_tree_request:
                modified_prompt += "\n\nCRITICAL: You MUST generate a decision tree visualization plot. This is mandatory for decision tree models. Use the dynamic sizing code provided in the guidelines and add 'plot_path' to the result dictionary."
            else:
                modified_prompt += "\n\nIMPORTANT: Also generate a visualization plot of the model and add 'plot_path' to the result dictionary."
        
        if should_generate_ranking:
            modified_prompt += "\n\nIMPORTANT: Generate rank ordering table with 10 deciles showing badrate, coverage, and cumulative metrics for financial analysis. CRITICAL: Use TEST SET predictions (y_proba and y_test) to create the rank ordering table - this evaluates model performance on unseen data."
        
        if should_compare_models:
            modified_prompt += "\n\nIMPORTANT: Compare multiple models (Random Forest, Decision Tree, LightGBM) and provide comprehensive metrics comparison including rank ordering for each model."
    
    # Generate code using LLM with thinking animation
    try:
        print_to_log("🤔 Generating code...")
        if progress_callback:
            progress_callback("🤔 Generating code...", "Code Generation")
        
        reply, code, system_prompt = generate_model_code(modified_prompt, user_id, query)
        
        if not code.strip():
            state["response"] = reply
            return state
        
        state["code"] = code
        print_to_log(f"📝 MODEL BUILDING AGENT - Generated code ({len(code)} chars)")
        
        # Execute the code
        print_to_log("⚙️ Executing generated code...")
        print_to_log(f"🔍 About to call ExecutionAgent with {len(code)} chars of code")
        # Get artifacts directory for this thread
        artifacts_dir = None
        try:
            # Extract user and thread from user_id format: user_threadts
            if "_" in user_id:
                user, thread_ts = user_id.split("_", 1)
                # Get username for proper directory structure
                username = get_username_for_user_id(user)
                thread_dir = os.path.join("user_data", username, thread_ts)
                artifacts_dir = os.path.join(thread_dir, "artifacts")
                if not os.path.exists(artifacts_dir):
                    os.makedirs(artifacts_dir, exist_ok=True)
                    print_to_log(f"📁 Created artifacts directory: {artifacts_dir}")
        except Exception as e:
            print_to_log(f"⚠️ Failed to create artifacts directory: {e}")
            artifacts_dir = None
        
        try:
            # ExecutionAgent will handle progress updates
            
            result = ExecutionAgent(
                code, 
                data, 
                user_id=user_id,
                verbose=True,
                model_states=global_model_states,
                user_query=query,
                intent=intent,
                artifacts_dir=artifacts_dir,
                progress_callback=progress_callback,  # Pass progress callback to ExecutionAgent
                original_system_prompt=system_prompt  # Pass original system prompt for error fixing
            )
            
            if isinstance(result, str) and "error" in result.lower():
                state["response"] = f"❌ {result}"
                return state
        except Exception as exec_error:
            # ExecutionAgent handles its own errors with tiered LLM
            # If we get here, it means ExecutionAgent gave up after all retries
            print_to_log(f"🚨 ExecutionAgent failed after all retries: {exec_error}")
            state["response"] = f"❌ Code execution failed: {str(exec_error)}"
            return state
        
        # Store results and set appropriate response
        state["execution_result"] = result
        
        # Check if execution actually succeeded
        if result is None:
            state["response"] = "❌ Code execution failed - no results generated"
        elif isinstance(result, str) and ("error" in result.lower() or "failed" in result.lower()):
            state["response"] = f"❌ Code execution failed: {result}"
        elif isinstance(result, dict) and result.get("execution_status") == "failed_missing_library":
            # Handle missing library case with Slack notification
            missing_lib = result.get("error", "").replace("Missing library: ", "")
            admin_message = result.get('admin_message', 'Please install the required library')
            
            # Send Slack notification about missing library
            if progress_callback:
                progress_callback(f"❌ Missing library: {missing_lib}. {admin_message}", "Error")
            
            # Format detailed response with metrics
            state["response"] = format_model_response(result, routing_decision, query)
        else:
            # Format detailed response with metrics
            state["response"] = format_model_response(result, routing_decision, query)
            
            # Note: Plot uploads are now handled by the wrapper after response completion
            # The plot_path in the execution result will be processed by agents_wrapper.py
            if isinstance(result, dict) and 'plot_path' in result and result['plot_path']:
                plot_path = result['plot_path']
                if os.path.exists(plot_path):
                    # Store in both state and result for compatibility
                    state["artifacts"] = state.get("artifacts", {})
                    state["artifacts"]["files"] = [{"path": plot_path, "title": "Decision Tree Plot", "type": "plot"}]
                    # Also add to result so wrapper can access it
                    result["artifacts"] = {"files": [{"path": plot_path, "title": "Decision Tree Plot", "type": "plot"}]}
                    print_to_log(f"📊 Plot ready for upload: {plot_path}")
                else:
                    print_to_log(f"⚠️ Plot file not found: {plot_path}")
        
        # Update model state if new model was built
        print_to_log(f"🔍 MODEL STATE TRACKING:")
        print_to_log(f"🔍 routing_decision: {routing_decision}")
        print_to_log(f"🔍 result type: {type(result)}")
        print_to_log(f"🔍 result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        
        if routing_decision == "build_new_model" and isinstance(result, dict):
            # Check if model was successfully built (either has model_path or performance metrics)
            has_model_path = result.get('model_path')
            has_performance_metrics = any(key in result for key in ['accuracy', 'precision', 'model_performance', 'classification_report'])
            
            print_to_log(f"🔍 has_model_path: {has_model_path}")
            print_to_log(f"🔍 has_performance_metrics: {has_performance_metrics}")
            print_to_log(f"🔍 Performance keys found: {[k for k in result.keys() if k in ['accuracy', 'precision', 'model_performance', 'classification_report']]}")
            
            if has_model_path or has_performance_metrics:
                if has_model_path:
                    state["model_path"] = result['model_path']
                    print_to_log(f"✅ New model saved: {result['model_path']}")
                    # Update global model states (preserve existing data)
                    if user_id not in global_model_states:
                        global_model_states[user_id] = {}
                    global_model_states[user_id]['model_path'] = result['model_path']
                    global_model_states[user_id]['last_result'] = result
                    # Preserve sample_data if it exists
                    print_to_log(f"🔍 Updated global_model_states for {user_id}")
                else:
                    # Model was built but not saved - still mark as having a model
                    print_to_log(f"✅ New model built successfully (performance metrics detected)")
                
                state["has_existing_model"] = True
                print_to_log(f"🔍 Set has_existing_model = True")
                
                # Handle full dataset predictions and probabilities
                if 'full_predictions' in result:
                    full_predictions = result['full_predictions']
                    full_probabilities = result.get('full_probabilities', None)
                    print_to_log(f"🔍 Full predictions received: {len(full_predictions)} predictions")
                    if full_probabilities is not None:
                        print_to_log(f"🔍 Full probabilities received: {len(full_probabilities)} probability arrays")
                    
                    # Add predictions and probabilities to pipeline state
                    if hasattr(state, 'add_predictions_to_dataset'):
                        success = state.add_predictions_to_dataset(full_predictions, "predictions", full_probabilities)
                        if success:
                            print_to_log(f"✅ Added predictions and probabilities to dataset")
                            
                            # Save predictions dataset to artifacts
                            artifacts_dir = os.path.join(os.path.dirname(result.get('model_path', '')), '..')
                            if os.path.exists(artifacts_dir):
                                # Extract model name from result or model_path
                                model_name = "unknown_model"
                                
                                # Try to get from result first
                                if 'model' in result:
                                    model = result['model']
                                    if hasattr(model, '__class__'):
                                        model_name = model.__class__.__name__.lower()
                                    elif hasattr(model, 'name'):
                                        model_name = model.name.lower()
                                
                                # If still unknown, try to extract from model_path
                                if model_name == "unknown_model":
                                    model_path = result.get('model_path') or global_model_states.get(user_id, {}).get('model_path', '')
                                    if model_path:
                                        # Extract from filename: "LightGBM_model.joblib" -> "lightgbm"
                                        filename = os.path.basename(model_path)
                                        if '_model.joblib' in filename:
                                            model_name = filename.replace('_model.joblib', '').lower()
                                        elif '.joblib' in filename:
                                            model_name = filename.replace('.joblib', '').lower()
                                
                                # Create filename with model name
                                timestamp = int(time.time())
                                predictions_file = state.save_predictions_dataset(
                                    os.path.join(artifacts_dir, f"predictions_dataset_{model_name}_{timestamp}.csv")
                                )
                                if predictions_file:
                                    print_to_log(f"✅ Predictions dataset saved to: {predictions_file}")
                        else:
                            print_to_log(f"⚠️ Failed to add predictions to dataset")
                    else:
                        print_to_log(f"⚠️ PipelineState does not have add_predictions_to_dataset method")
                else:
                    print_to_log(f"⚠️ No full_predictions found in result")
            else:
                print_to_log(f"⚠️ Model not recognized - no model_path and no performance metrics")
        else:
            print_to_log(f"⚠️ Model state not updated - routing_decision={routing_decision}, result_is_dict={isinstance(result, dict)}")
        
        state["messages"].append({
            "agent": "model_building", 
            "content": "Model operation completed",
            "timestamp": datetime.now().isoformat()
        })
        
        return state
        
    except Exception as e:
        print_to_log(f"💥 Model building failed: {e}")
        print_to_log(f"🔍 Error type: {type(e).__name__}")
        print_to_log(f"🔍 Error occurred in model building agent, not ExecutionAgent")
        state["response"] = f"❌ Model building failed: {str(e)}"
        return state



BASE_SYSTEM_PROMPT = """
You are a Python + modelling expert. 
Return ONLY executable Python code (no markdown, no explanations, no function definitions).

🚨 CRITICAL: Generate DIRECT executable code, NOT function definitions!
🚨 CRITICAL: You MUST include model training, predictions, and result dictionary!
🚨 CRITICAL: The code must be COMPLETE and EXECUTABLE immediately!
🚨 CRITICAL: Your code MUST end with a 'result' dictionary containing all metrics and model_path!
🚨 CRITICAL: DO NOT stop after model.fit() - continue with predictions, metrics, and result dictionary!

🚨 METRICS CALCULATION RULES:
- For BINARY classification: Use average='binary' (positive class metrics)
- For MULTI-CLASS classification: Use average='macro' by default (better for imbalanced data)
- If user explicitly requests 'weighted', 'micro', or 'macro' averaging, use their preference
- Always include 'metrics_averaging' in result dictionary to show which method was used

🚨 CRITICAL DATA AVAILABILITY:
The variable `sample_data` is ALREADY LOADED and available in your environment.
The variable `current_model` is ALREADY LOADED (for existing model operations).
❌ DO NOT import datasets (sklearn.datasets, load_iris, etc.)
❌ DO NOT load external data files
❌ DO NOT import safe_utils (safe_joblib_dump and safe_plt_savefig are already available)
✅ USE the existing `sample_data` DataFrame directly
✅ USE safe_joblib_dump() and safe_plt_savefig() directly (no imports needed)

MANDATORY STRUCTURE:
1. Import statements (NO data loading imports!)
2. Data splitting: X = sample_data.drop('TARGET_COLUMN', axis=1); y = sample_data['TARGET_COLUMN']
3. Train/test split: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
4. Model training: model = [UserRequestedModel](); model.fit(X_train, y_train)
   ✅ FOLLOW USER'S SPECIFIC REQUEST for model type and parameters
   ✅ FOR DECISION TREES: Use max_depth=5 as default if user doesn't specify depth
5. Predictions: y_pred = model.predict(X_test); y_proba = model.predict_proba(X_test)
6. Metrics calculation
7. Model saving: model_path = safe_joblib_dump(model, 'model.joblib') (Not required for existing models)
8. Result dictionary: result = {'model': model, 'model_path': model_path, ...}

FORBIDDEN:
❌ DO NOT define functions (especially safe_joblib_dump - it's already available)
❌ DO NOT use def statements
❌ DO NOT import joblib (use safe_joblib_dump directly)
❌ DO NOT import safe_utils (safe_joblib_dump and safe_plt_savefig are already available)
❌ DO NOT generate tree plots for LGBM, XGBoost, or RandomForest models (only for DecisionTree models)
❌ DO NOT use plot_tree() for non-DecisionTree models
❌ DO NOT use pipelines, ColumnTransformer, scalers unless explicitly asked

REQUIRED:
✅ Use `sample_data` (already preprocessed)
✅ Always evaluate on X_test, y_test (not train)
✅ Convert numpy objects to Python native types: float(), int(), .tolist()
✅ Save plots with safe_plt_savefig(), never plt.show()
✅ For DecisionTreeClassifier/Regressor → automatically generate plot
✅ Final result dictionary with model, model_path, and all metrics

🌲 DECISION TREE DEFAULTS:
- If user requests "decision tree" without depth: DecisionTreeClassifier(max_depth=5, random_state=42)
- If user specifies depth: DecisionTreeClassifier(max_depth=USER_SPECIFIED, random_state=42)
- Always include random_state=42 for reproducibility

🚨 MANDATORY CODE COMPLETION TEMPLATE:
Your code MUST follow this exact structure and be COMPLETE:

1. Import statements
2. Data splitting: X = sample_data.drop('TARGET_COLUMN', axis=1); y = sample_data['TARGET_COLUMN']
3. Train/test split: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
4. Model training: model = [ModelType](...); model.fit(X_train, y_train)
5. Test predictions: y_pred = model.predict(X_test); y_proba = model.predict_proba(X_test)
6. Full dataset predictions: full_predictions = model.predict(X); full_probabilities = model.predict_proba(X)
7. Metrics calculation: accuracy, precision, recall, f1, etc.
8. Model saving: model_path = safe_joblib_dump(model, 'model_name.joblib')
9. Plot generation (if applicable): safe_plt_savefig('plot_name.png')
10. Result dictionary: result = {'model': model, 'model_path': model_path, 'accuracy': accuracy, 'full_predictions': full_predictions, 'full_probabilities': full_probabilities, ...}

DO NOT STOP EARLY! Complete ALL 10 steps!
"""

CLASSIFICATION_METRICS_PROMPT = """
🚨 MANDATORY: After model training, you MUST complete these steps:

# Step 1: Make predictions on TEST set
   y_pred = model.predict(X_test)
   y_proba = model.predict_proba(X_test)
   
# Step 1.5: Make predictions on TRAINING set (for overfitting detection)
   y_train_pred = model.predict(X_train)
   y_train_proba = model.predict_proba(X_train)
   
# Step 1.6: Make predictions on full dataset
   full_predictions = model.predict(X)
   full_probabilities = model.predict_proba(X)

# Step 2: Calculate TEST set metrics
   cm = confusion_matrix(y_test, y_pred)
   tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
   specificity = tn/(tn+fp) if (tn+fp) > 0 else 0.0
   
# Step 2.5: Calculate TRAINING set metrics (for overfitting detection)
   cm_train = confusion_matrix(y_train, y_train_pred)
   tn_train, fp_train, fn_train, tp_train = cm_train.ravel() if cm_train.shape == (2,2) else (0,0,0,0)
   specificity_train = tn_train/(tn_train+fp_train) if (tn_train+fp_train) > 0 else 0.0
   
# Step 3: Save the model
model_path = safe_joblib_dump(model, 'decision_tree_model.joblib')

# Step 4: Create result dictionary (MANDATORY!)
   # Determine averaging method based on number of classes
   if len(np.unique(y_test)) == 2:
       # Binary classification - use positive class (class 1)
       avg_method = 'binary'
   else:
       # Multi-class - use macro average by default (better than weighted for imbalanced data)
       avg_method = 'macro'
   
   result = {
       # TEST SET METRICS
       'accuracy': float(accuracy_score(y_test, y_pred)),
       'precision': float(precision_score(y_test, y_pred, average=avg_method)),
       'recall': float(recall_score(y_test, y_pred, average=avg_method)),
       'f1_score': float(f1_score(y_test, y_pred, average=avg_method)),
       'specificity': float(specificity),
       'roc_auc': float(roc_auc_score(y_test, y_proba[:,1])) if y_proba.shape[1] == 2 else float(roc_auc_score(y_test, y_proba, multi_class='ovr')),
       'confusion_matrix': cm.tolist(),
       'log_loss': float(log_loss(y_test, y_proba)),
       
       # TRAINING SET METRICS (for overfitting detection)
       'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
       'train_precision': float(precision_score(y_train, y_train_pred, average=avg_method)),
       'train_recall': float(recall_score(y_train, y_train_pred, average=avg_method)),
       'train_f1_score': float(f1_score(y_train, y_train_pred, average=avg_method)),
       'train_specificity': float(specificity_train),
       'train_roc_auc': float(roc_auc_score(y_train, y_train_proba[:,1])) if y_train_proba.shape[1] == 2 else float(roc_auc_score(y_train, y_train_proba, multi_class='ovr')),
       'train_confusion_matrix': cm_train.tolist(),
       'train_log_loss': float(log_loss(y_train, y_train_proba)),
       
       # OTHER METADATA
       'model': model,
       'model_path': model_path,
       'full_predictions': full_predictions.tolist(),
       'full_probabilities': full_probabilities.tolist(),
       'metrics_averaging': avg_method
   }

🚨 CRITICAL: The code MUST end with the complete result dictionary!
"""

REGRESSION_METRICS_PROMPT = """
🚨 MANDATORY: After model training, you MUST complete these steps:

# Step 1: Make predictions on TEST set
y_pred = model.predict(X_test)

# Step 1.5: Make predictions on TRAINING set (for overfitting detection)
y_train_pred = model.predict(X_train)

# Step 1.6: Make predictions on full dataset
full_predictions = model.predict(X)

# Step 2: Calculate TEST set regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

n, k = X_test.shape
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1-r2)*(n-1)/(n-k-1)

# Calculate MAPE safely (avoid division by zero)
mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100

# Calculate Huber loss
delta = 1.35
residual = np.abs(y_test - y_pred)
huber_loss = np.where(residual <= delta, 0.5*residual**2, delta*residual - 0.5*delta**2).mean()

# Step 2.5: Calculate TRAINING set regression metrics (for overfitting detection)
n_train, k_train = X_train.shape
r2_train = r2_score(y_train, y_train_pred)
adj_r2_train = 1 - (1-r2_train)*(n_train-1)/(n_train-k_train-1)

# Calculate MAPE safely for training set
mape_train = np.mean(np.abs((y_train - y_train_pred) / np.where(y_train != 0, y_train, 1))) * 100

# Calculate Huber loss for training set
residual_train = np.abs(y_train - y_train_pred)
huber_loss_train = np.where(residual_train <= delta, 0.5*residual_train**2, delta*residual_train - 0.5*delta**2).mean()

# Step 3: Save the model
model_path = safe_joblib_dump(model, 'regression_model.joblib')

# Step 4: Create result dictionary (MANDATORY!)
result = {
    # TEST SET METRICS
    'mae': float(mean_absolute_error(y_test, y_pred)),
    'mse': float(mean_squared_error(y_test, y_pred)),
    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
    'mape': float(mape),
    'r2': float(r2),
    'adjusted_r2': float(adj_r2),
    'huber_loss': float(huber_loss),
    
    # TRAINING SET METRICS (for overfitting detection)
    'train_mae': float(mean_absolute_error(y_train, y_train_pred)),
    'train_mse': float(mean_squared_error(y_train, y_train_pred)),
    'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
    'train_mape': float(mape_train),
    'train_r2': float(r2_train),
    'train_adjusted_r2': float(adj_r2_train),
    'train_huber_loss': float(huber_loss_train),
    
    # OTHER METADATA
    'model': model,
    'model_path': model_path,
    'full_predictions': full_predictions.tolist()
}

🚨 CRITICAL: The code MUST end with the complete result dictionary!
"""

RANK_ORDERING_PROMPT = """
🚨 MANDATORY: Add rank ordering/segmentation analysis:
🚨 CRITICAL: Use TEST SET ONLY for rank ordering (NEVER mix test and full dataset sizes!)
🚨 CRITICAL: NEVER assign test predictions to full dataset - they have different sizes!

COMPLETE STEP-BY-STEP PROCESS:

# Step 1: Get predictions for segmentation (use test set only)
y_proba = model.predict_proba(X_test)[:,1]  # Get positive class probabilities

# Step 2: Create segmentation dataframe
test_df = pd.DataFrame({{
    'actual': y_test.values,
    'probability': y_proba
}})

# Step 3: Create buckets/deciles (CRITICAL: use duplicates='drop')
# Parse user query for number of segments
user_query_lower = "{query}".lower()
num_segments = 10  # Default to 10 deciles
import re
if "segment" in user_query_lower:
    segments_match = re.search(r'(\\d+)\\s*segment', user_query_lower)
    if segments_match:
        num_segments = int(segments_match.group(1))

test_df['bucket'] = pd.qcut(test_df['probability'], q=num_segments, labels=False, duplicates='drop')

# Step 4: Calculate rank ordering metrics
rank_metrics = test_df.groupby('bucket').agg({{
    'actual': ['sum','count'],
    'probability': ['mean','min','max']
}}).reset_index()
rank_metrics.columns = ['bucket','numBads','totalUsersCount','avg_probability','min_threshold','max_threshold']

# Step 5: Calculate rates and cumulative metrics
rank_metrics['badrate'] = rank_metrics['numBads'] / rank_metrics['totalUsersCount']
rank_metrics['bucket'] = rank_metrics['bucket'] + 1  # Start from 1, not 0
rank_metrics = rank_metrics.sort_values('bucket')

# Cumulative calculations
rank_metrics['cum_numBads'] = rank_metrics['numBads'].cumsum()
rank_metrics['cum_totalUsers'] = rank_metrics['totalUsersCount'].cumsum()
rank_metrics['cum_badrate'] = rank_metrics['cum_numBads'] / rank_metrics['cum_totalUsers']
rank_metrics['coverage'] = (rank_metrics['cum_totalUsers']/len(test_df))*100

# Step 6: Format results (round to appropriate decimal places)
for col in ['badrate','cum_badrate','avg_probability','min_threshold','max_threshold']:
    rank_metrics[col] = rank_metrics[col].round(4)
rank_metrics['coverage'] = rank_metrics['coverage'].round(2)

# Step 7: Add to result dictionary
result['rank_ordering_table'] = rank_metrics.to_dict('records')

🚨 CRITICAL: Always add rank_ordering_table to the existing result dictionary!
"""

DECISION_TREE_PLOT_PROMPT = """
🚨 MANDATORY: For DecisionTreeClassifier/Regressor - Generate visualization plot:

COMPLETE STEP-BY-STEP PROCESS:

# Step 1: Import required plotting libraries
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Step 2: Calculate dynamic sizing parameters
tree_depth = model.get_depth()
max_feature_len = max(len(name) for name in X.columns)
safe_tree_depth = min(tree_depth, 15)  # Cap at 15 to prevent huge plots
nodes_at_max_level = 2 ** safe_tree_depth
width = int(max(25, min(100, nodes_at_max_level * 0.1 * max_feature_len)))
height = int(max(15, min(50, safe_tree_depth * 3)))
font_size = int(max(5, min(10, 80 / (safe_tree_depth + max_feature_len/10))))

# Step 3: Set plot depth (limit for readability)
max_depth_plot = min(5, safe_tree_depth)

# Step 4: Create and configure the plot
plt.figure(figsize=(width, height))
plot_tree(model, filled=True, feature_names=X.columns.tolist(),
          fontsize=font_size, proportion=True, rounded=True, precision=2,
          max_depth=max_depth_plot)
plt.tight_layout(pad=2.0)

# Step 5: Save plot and add to result dictionary
plot_path = safe_plt_savefig('decision_tree.png', bbox_inches='tight', dpi=300, facecolor='white')
result['plot_path'] = plot_path

🚨 CRITICAL: Always add plot_path to the existing result dictionary!
"""

def detect_problem_type(y: pd.Series) -> str:
    """Detect whether the task is regression or classification based on target column."""
    # If y is numeric but has few unique values, treat as classification (e.g. {0,1})
    if pd.api.types.is_numeric_dtype(y):
        unique_vals = y.nunique()
        if unique_vals <= 20:  # threshold can be tuned
            return "classification"
        else:
            return "regression"
    else:
        return "classification"


def find_target_column(user_specified: str, available_columns: List[str]) -> tuple[str, str]:
    """
    Find target column using fuzzy matching.
    
    Args:
        user_specified: Column name specified by user
        available_columns: List of available column names in data
        
    Returns:
        tuple: (matched_column_name, match_type)
        match_type: 'exact', 'case_insensitive', 'partial', 'none'
    """
    if not user_specified or not available_columns:
        return None, 'none'
    
    user_specified = user_specified.strip()
    available_columns = [col.strip() for col in available_columns]
    
    # 1. Exact match
    if user_specified in available_columns:
        return user_specified, 'exact'
    
    # 2. Case-insensitive match
    user_lower = user_specified.lower()
    for col in available_columns:
        if col.lower() == user_lower:
            return col, 'case_insensitive'
    
    # 3. Partial match (user input is substring of column name)
    for col in available_columns:
        if user_lower in col.lower():
            return col, 'partial'
    
    # 4. Reverse partial match (column name is substring of user input)
    for col in available_columns:
        if col.lower() in user_lower:
            return col, 'partial'
    
    return None, 'none'


def extract_target_column_from_prompt(prompt: str) -> Optional[str]:
    """
    Extract target column name from user prompt.
    
    Looks for patterns like:
    - "target column is X"
    - "use X as target"
    - "predict X"
    - "X as target column"
    """
    import re
    
    prompt_lower = prompt.lower()
    
    # Pattern 1: "target column is X" or "target column X"
    patterns = [
        r'target\s+column\s+is\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'target\s+column\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'use\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s+target',
        r'predict\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s+target\s+column',
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s+target',
        r'with\s+target\s+column\s+["'"]([a-zA-Z_][a-zA-Z0-9_]*)["'"]',  # "with target column 'f_segment'"
        r'with\s+target\s+column\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # "with target column f_segment"
        r"with target column '([^']+)'",  # Simple pattern for single quotes
        r'with\s+target\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # "with target f_segment" - CRITICAL MISSING PATTERN
        r'target\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # "target f_segment"
    ]    
    for pattern in patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            return match.group(1)
    
    return None


def generate_model_code(prompt: str, user_id: str, original_query: str = "") -> tuple[str, str, str]:
    """Generate model code using modular LLM prompts - returns (reply, code, system_prompt)"""
    
    print_to_log(f"🔍 DEBUG - generate_model_code called with user_id: {user_id}")
    print_to_log(f"🔍 DEBUG - global_model_states keys: {list(global_model_states.keys())}")
    
    # Get sample_data from global model states to detect problem type
    try:
        # Try to get data from global model states first
        global_info = global_model_states.get(user_id, {})
        sample_data = global_info.get('sample_data')
        
        print_to_log(f"🔍 DEBUG - global_info keys: {list(global_info.keys())}")
        print_to_log(f"🔍 DEBUG - sample_data is None: {sample_data is None}")
        if sample_data is not None:
            print_to_log(f"🔍 DEBUG - sample_data shape: {sample_data.shape}")
            
            # Analyze column types instead of listing all columns
            numeric_cols = sample_data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
            categorical_cols = sample_data.select_dtypes(include=['object', 'category']).columns
            boolean_cols = sample_data.select_dtypes(include=['bool']).columns
            datetime_cols = sample_data.select_dtypes(include=['datetime64']).columns
            
            print_to_log(f"🔍 DEBUG - Column analysis:")
            print_to_log(f"   📊 Numeric features: {len(numeric_cols)}")
            print_to_log(f"   🏷️  Categorical features: {len(categorical_cols)}")
            if len(boolean_cols) > 0:
                print_to_log(f"   ✅ Boolean features: {len(boolean_cols)}")
            if len(datetime_cols) > 0:
                print_to_log(f"   📅 DateTime features: {len(datetime_cols)}")
            print_to_log(f"   🎯 Target column: {'target' if 'target' in sample_data.columns else 'Not found'}")
        
        # If not found, we'll assume classification for now
        problem_type = "classification"
        
        # If we have access to the data, detect the actual problem type
        target_column = extract_target_column_from_prompt(prompt) or global_info.get('target_column')
        if sample_data is not None and target_column and target_column in sample_data.columns:
            y = sample_data[target_column]
            problem_type = detect_problem_type(y)
            print_to_log(f"🔍 PROBLEM TYPE DETECTION:")
            print_to_log(f"   📊 Target column '{target_column}' found: {y.nunique()} unique values")
            print_to_log(f"   🎯 Detected problem type: {problem_type}")
        else:
            print_to_log(f"🔍 PROBLEM TYPE DETECTION:")
            if sample_data is not None:
                print_to_log(f"   📊 Sample data available: {sample_data.shape}")
                print_to_log(f"   🎯 Target column sought: '{target_column if target_column else 'None'}'")
                print_to_log(f"   📋 Available columns: {list(sample_data.columns)}")
                if target_column:
                    print_to_log(f"   ❓ Target in columns: {target_column in sample_data.columns}")
            else:
                print_to_log(f"   ⚠️ No sample_data available")
            print_to_log(f"   🎯 Defaulting to: {problem_type}")
    except Exception as e:
        print_to_log(f"🔍 PROBLEM TYPE DETECTION:")
        print_to_log(f"   ❌ Error during detection: {e}")
        print_to_log(f"   🎯 Defaulting to: classification")
        problem_type = "classification"
    
    # =============================================================================
    # TARGET COLUMN SELECTION LOGIC
    # =============================================================================
    target_column = None
    target_column_source = None
    
    # Step 1: Check if target column is specified in pipeline state
    try:
        from pipeline_state import state_manager
        current_state = state_manager.load_state(user_id)
        if current_state and hasattr(current_state, 'target_column') and current_state.target_column:
            target_column = current_state.target_column
            target_column_source = "pipeline_state"
            print_to_log(f"🎯 TARGET COLUMN: Found in pipeline state: '{target_column}'")
    except Exception as e:
        print_to_log(f"🔍 TARGET COLUMN: Could not load pipeline state: {e}")
    
    # Step 2: If no target column in state, try to extract from user prompt
    if not target_column:
        extracted_target = extract_target_column_from_prompt(original_query or prompt)
        if extracted_target:
            print_to_log(f"🎯 TARGET COLUMN: Extracted from prompt: '{extracted_target}'")
            
            # Try to match with available columns
            if sample_data is not None:
                available_columns = list(sample_data.columns)
                matched_column, match_type = find_target_column(extracted_target, available_columns)
                
                if matched_column:
                    target_column = matched_column
                    target_column_source = f"prompt_extracted_{match_type}"
                    print_to_log(f"🎯 TARGET COLUMN: Matched '{extracted_target}' -> '{matched_column}' ({match_type})")
                    
                    # Update pipeline state with the matched target column
                    try:
                        if current_state:
                            current_state.target_column = target_column
                            state_manager.save_state(current_state)
                            print_to_log(f"🎯 TARGET COLUMN: Updated pipeline state with '{target_column}'")
                    except Exception as e:
                        print_to_log(f"⚠️ TARGET COLUMN: Could not update pipeline state: {e}")
                else:
                    # Target column specified but not found in data
                    available_cols_str = ", ".join(available_columns[:10])  # Show first 10 columns
                    if len(available_columns) > 10:
                        available_cols_str += f" ... and {len(available_columns) - 10} more"
                    
                    error_msg = f"❌ Target column '{extracted_target}' not found in data. Available columns: {available_cols_str}"
                    print_to_log(f"🎯 TARGET COLUMN: {error_msg}")
                    return error_msg, "", ""
    
    # Step 3: If still no target column, check if 'target' column exists in data
    if not target_column and sample_data is not None:
        if 'target' in sample_data.columns:
            target_column = 'target'
            target_column_source = "default_target_column"
            print_to_log(f"🎯 TARGET COLUMN: Using default 'target' column")
        else:
            # FALLBACK: Try to read target column from session state
            print_to_log(f"🔍 TARGET COLUMN: Attempting session state fallback for user: {user_id}")
            try:
                from langgraph_pipeline import MultiAgentMLPipeline
                pipeline = MultiAgentMLPipeline()
                
                # Try to load session state
                session_state = pipeline._load_session_state(user_id)
                if session_state and isinstance(session_state, dict) and session_state.get('target_column'):
                    # Verify the target column exists in the current data
                    session_target = session_state.get('target_column')
                    if session_target in sample_data.columns:
                        target_column = session_target
                        target_column_source = "session_state_fallback"
                        print_to_log(f"✅ TARGET COLUMN: Found in session state: '{target_column}'")
                    else:
                        print_to_log(f"⚠️ TARGET COLUMN: Session state target '{session_target}' not in current data columns")
                else:
                    print_to_log(f"🔍 TARGET COLUMN: No target found in session state (session_state type: {type(session_state)})")
                    
            except Exception as e:
                print_to_log(f"⚠️ TARGET COLUMN: Session state fallback failed: {e}")
            
            # If still no target column found anywhere - ask user to specify
            if not target_column:
                available_cols_str = ", ".join(sample_data.columns[:10])  # Show first 10 columns
                if len(sample_data.columns) > 10:
                    available_cols_str += f" ... and {len(sample_data.columns) - 10} more"
                
                error_msg = f"❌ No target column found. Please specify the target column in your request. Available columns: {available_cols_str}"
                print_to_log(f"🎯 TARGET COLUMN: {error_msg}")
                return error_msg, "", ""
            
    
    # Final target column validation
    if target_column and sample_data is not None:
        if target_column not in sample_data.columns:
            available_cols_str = ", ".join(sample_data.columns[:10])
            if len(sample_data.columns) > 10:
                available_cols_str += f" ... and {len(sample_data.columns) - 10} more"
            
            error_msg = f"❌ Target column '{target_column}' not found in data. Available columns: {available_cols_str}"
            print_to_log(f"🎯 TARGET COLUMN: {error_msg}")
            return error_msg, "", ""
        
        # Update problem type detection with the correct target column
        y = sample_data[target_column]
        problem_type = detect_problem_type(y)
        print_to_log(f"🎯 TARGET COLUMN: Final selection: '{target_column}' (source: {target_column_source})")
        print_to_log(f"🎯 TARGET COLUMN: Problem type: {problem_type}")
    else:
        print_to_log(f"🎯 TARGET COLUMN: No target column determined")
        return "❌ No target column available for modeling", "", ""

    # Use original_query if available to avoid false matches in system instructions
    detection_text = original_query.lower() if original_query else prompt.lower()
    
    # Rank-ordering detection (from user query only, not system instructions)
    is_rank_ordering_request = any(
        k in detection_text for k in ['rank ordering','bucket','decile','segment']
    )
    print_to_log(f"🔍 KEYWORD DETECTION:")
    print_to_log(f"   📝 User prompt: '{prompt}'")
    print_to_log(f"   🎯 Rank ordering keywords found: {is_rank_ordering_request}")
    print_to_log(f"   🌳 Tree keywords found: {'tree' in prompt.lower()}")
    
    # Use original_query if available to avoid false matches in system instructions
    detection_text = original_query.lower() if original_query else prompt.lower()
    
    # 🧠 SEMANTIC + KEYWORD HYBRID APPROACH for Decision Tree Detection
    from toolbox import UniversalPatternClassifier
    
    # Define decision tree patterns for semantic classification
    decision_tree_patterns = {
        "decision_tree_request": "train decision tree, build decision tree, create decision tree, decision tree model, classification tree, tree classifier, dt model, tree algorithm, train tree, build tree, make tree, fit tree, tree model, decision tree classification",
        "other_tree_models": "random forest, extra trees, gradient boosting trees, lgbm tree, xgboost tree, lightgbm, xgboost, catboost, random forest classifier, ensemble trees",
        "general_modeling": "train model, build model, create classifier, machine learning model, fit algorithm"
    }
    
    # Use semantic classification with fallback
    classifier = UniversalPatternClassifier()
    detection_query = original_query if original_query else prompt
    
    pattern_result, method = classifier.classify_pattern(
        detection_query, 
        decision_tree_patterns,
        use_case="model_sub_classification"  # Liberal thresholds for better detection
    )
    
    # Semantic decision tree detection
    semantic_decision_tree = (pattern_result == "decision_tree_request")
    is_other_tree_model = (pattern_result == "other_tree_models")
    
    # Keyword fallback for edge cases
    keyword_decision_tree = any(phrase in detection_text for phrase in [
        "decision tree", "decisiontree", "decision-tree", "dt", "classification tree"
    ]) and not any(phrase in detection_text for phrase in [
        "random forest", "lgbm", "lightgbm", "xgb", "xgboost", "catboost"
    ])
    
    # Context-aware "tree" detection (when semantic isn't confident)
    contextual_tree = (
        "tree" in detection_text and 
        any(verb in detection_text for verb in ["train", "build", "create", "make", "fit", "model"]) and
        not any(exclude in detection_text for exclude in ["random", "forest", "gradient", "boost"])
    )
    
    # Final decision: Semantic primary, keyword + contextual as fallback
    if method == "semantic":  # Trust semantic if it was used
        is_decision_tree_request = semantic_decision_tree and not is_other_tree_model
        detection_method = f"semantic (method: {method})"
    else:
        is_decision_tree_request = keyword_decision_tree or contextual_tree
        detection_method = f"keyword_fallback (method: {method})"
    
    # Tree plot detection logic
    tree_in_prompt = is_decision_tree_request  # Use the robust detection
    # AUTOMATIC DECISION TREE PLOTTING: Always generate plots for decision trees (as per BASE_SYSTEM_PROMPT line 2700)
    wants_tree_plot = tree_in_prompt  # Auto-generate plots for all decision tree requests
    using_existing_tree = any(phrase in detection_text for phrase in [
        "use this tree", "use the tree", "with this tree", "for this tree",
        "existing tree", "current tree", "saved tree"
    ])
    
    print_to_log(f"   🌳 Tree keywords found: {tree_in_prompt}")
    print_to_log(f"   🌳 Wants tree plot: {wants_tree_plot}")
    print_to_log(f"   🌳 Using existing tree: {using_existing_tree}")
    print_to_log(f"   🌳 Is DecisionTree request: {is_decision_tree_request}")
    
    # Enhanced debug logging for decision tree detection
    print_to_log(f"   🔍 Detection method: {detection_method}")
    print_to_log(f"   🧠 Semantic result: {pattern_result} (method: {method})")
    print_to_log(f"   🎯 Semantic decision tree: {semantic_decision_tree}")
    print_to_log(f"   ❌ Other tree model detected: {is_other_tree_model}")
    print_to_log(f"   🔤 Keyword fallback: {keyword_decision_tree}")
    print_to_log(f"   📝 Contextual tree: {contextual_tree}")
    if original_query:
        print_to_log(f"   📋 Detection text: '{original_query}'")
    else:
        print_to_log(f"   ⚠️ Using modified prompt for detection")

    # Build system prompt
    print_to_log(f"🔍 PROMPT ASSEMBLY:")
    system_prompt = BASE_SYSTEM_PROMPT
    print_to_log(f"   📋 BASE_SYSTEM_PROMPT added ({len(BASE_SYSTEM_PROMPT)} chars)")
    
    if problem_type == "regression":
        system_prompt += "\n" + REGRESSION_METRICS_PROMPT
        print_to_log(f"   📊 REGRESSION_METRICS_PROMPT added ({len(REGRESSION_METRICS_PROMPT)} chars)")
    else:
        system_prompt += "\n" + CLASSIFICATION_METRICS_PROMPT
        print_to_log(f"   📊 CLASSIFICATION_METRICS_PROMPT added ({len(CLASSIFICATION_METRICS_PROMPT)} chars)")

    if is_rank_ordering_request:
        system_prompt += "\n" + RANK_ORDERING_PROMPT
        print_to_log(f"   📈 RANK_ORDERING_PROMPT added ({len(RANK_ORDERING_PROMPT)} chars)")

    if wants_tree_plot and not using_existing_tree and is_decision_tree_request:
        system_prompt += "\n" + DECISION_TREE_PLOT_PROMPT
        print_to_log(f"   🌳 DECISION_TREE_PLOT_PROMPT added ({len(DECISION_TREE_PLOT_PROMPT)} chars)")
    elif wants_tree_plot and not is_decision_tree_request:
        print_to_log(f"   🚫 Skipping tree plot for non-DecisionTree model (LGBM/XGB/etc.)")
    elif tree_in_prompt:
        print_to_log(f"   🌳 Tree keyword detected but plot not needed (using existing tree for other purposes)")
    
    # Detect averaging method from user query
    user_query_lower = (original_query or prompt).lower()
    avg_method = None
    if 'weighted' in user_query_lower:
        avg_method = 'weighted'
    elif 'micro' in user_query_lower:
        avg_method = 'micro'
    elif 'macro' in user_query_lower:
        avg_method = 'macro'
    elif 'binary' in user_query_lower:
        avg_method = 'binary'
    
    if avg_method:
        print_to_log(f"📊 METRICS AVERAGING: User specified '{avg_method}' averaging method")
        # Replace the averaging method in the system prompt
        system_prompt = system_prompt.replace("avg_method = 'binary'", f"avg_method = '{avg_method}'")
        system_prompt = system_prompt.replace("avg_method = 'macro'", f"avg_method = '{avg_method}'")
    else:
        print_to_log(f"📊 METRICS AVERAGING: Using default method (binary for 2-class, macro for multi-class)")
    
    # Replace TARGET_COLUMN placeholder with actual target column in system prompt
    if target_column:
        system_prompt = system_prompt.replace("'TARGET_COLUMN'", f"'{target_column}'")
        system_prompt = system_prompt.replace('"TARGET_COLUMN"', f'"{target_column}"')
        system_prompt = system_prompt.replace("drop('TARGET_COLUMN'", f"drop('{target_column}'")
        system_prompt = system_prompt.replace("['TARGET_COLUMN']", f"['{target_column}']")
        print_to_log(f"🎯 TARGET COLUMN: Updated system prompt to use '{target_column}'")
    
    print_to_log(f"🔍 FINAL PROMPT STATS:")
    print_to_log(f"   📏 Total system prompt length: {len(system_prompt)} characters")
    print_to_log(f"   📝 User prompt length: {len(prompt)} characters")
    print_to_log(f"   📊 Sample data shape: {sample_data.shape}")

    # Call LLM with much smaller, focused prompt
    try:
        response = ollama.chat(
            model=MAIN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={
                "num_predict": -1,    # No limit on response length (unlimited)
                "temperature": 0.1,   # Low temperature for consistent code generation
                "top_p": 0.9,         # Nucleus sampling
                "stop": []            # No early stopping
            }
        )
        
        reply = response["message"]["content"]
        code = extract_first_code_block(reply)
        return reply, code, system_prompt
        
    except Exception as e:
        print_to_log(f"🔥 Error in generate_model_code: {e}")
        import traceback
        print_to_log(f"🔥 Full traceback: {traceback.format_exc()}")
        return f"LLM error: {e}", "", ""


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_agent_graph() -> StateGraph:
    """Create the LangGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes (agents)
    workflow.add_node("prompt_understanding", prompt_understanding_agent)
    workflow.add_node("controller", controller_agent)
    workflow.add_node("model_building", model_building_agent)
    
    # Define the flow
    workflow.set_entry_point("prompt_understanding")
    
    # From prompt understanding -> controller (always)
    workflow.add_edge("prompt_understanding", "controller")
    
    # From controller -> model_building or END
    def should_use_model_building(state: AgentState) -> str:
        routing_decision = state.get("routing_decision", "")
        if routing_decision in ["build_new_model", "build_multi_model", "use_existing_model", "general_response", "no_model_available", "execute_code"]:
            return "model_building"
        else:
            return END
    
    workflow.add_conditional_edges(
        "controller",
        should_use_model_building,
        {
            "model_building": "model_building",
            END: END
        }
    )
    
    # From model_building -> END
    workflow.add_edge("model_building", END)
    
    return workflow.compile()

# =============================================================================
# MODULAR MULTI-MODEL ORCHESTRATOR
# =============================================================================

CONFIG_PARSER_PROMPT = """You are a CONFIG PARSING agent for a model comparison system.

INPUT:
- USER_QUERY: "{query}"

TASK:
1. Parse and extract these fields:
   - models_requested: list of canonical model keys (see mapping below)
   - test_size: float between 0 and 1 (default 0.2)
   - selection_metric: string (default: accuracy for classification, r2 for regression)
   - problem_type: "classification" or "regression"

CANONICAL KEYS:
- 'lgbm', 'xgboost', 'random_forest', 'decision_tree', 'neural_network', 'logistic_regression', 'svm'

OUTPUT:
Return STRICT JSON only (no explanation) with exactly these keys:
{{
  "models_requested": ["random_forest", "xgboost"],
  "test_size": 0.2,
  "selection_metric": "accuracy",
  "problem_type": "classification"
}}
"""

def handle_multi_model_orchestrator(query: str, sample_data, user_id: str, progress_callback=None):
    """
    Modular orchestrator that breaks multi-model comparison into focused steps.
    Each step uses ExecutionAgent to run small, focused code blocks.
    """
    print_to_log("🎯 Starting modular multi-model orchestrator...")
    
    try:
        # Step 0: Create artifacts directory
        artifacts_dir = None
        try:
            if "_" in user_id:
                from agent_utils import get_username_for_user_id
                user, thread_ts = user_id.split("_", 1)
                username = get_username_for_user_id(user)
                thread_dir = os.path.join("user_data", username, thread_ts)
                artifacts_dir = os.path.join(thread_dir, "artifacts")
                if not os.path.exists(artifacts_dir):
                    os.makedirs(artifacts_dir, exist_ok=True)
                    print_to_log(f"📁 Created artifacts directory: {artifacts_dir}")
                else:
                    print_to_log(f"📁 Using artifacts directory: {artifacts_dir}")
        except Exception as e:
            print_to_log(f"⚠️ Failed to create artifacts directory: {e}")
            artifacts_dir = "."  # Fallback to current directory
        
        # Step A: Config parsing using focused LLM prompt
        print_to_log("📋 Step A: Parsing configuration...")
        if progress_callback:
            progress_callback("📋 Parsing request configuration...", "Config Parsing")
        
        config_prompt = CONFIG_PARSER_PROMPT.format(query=query)
        
        try:
            response = ollama.chat(
                model=MAIN_MODEL,
                messages=[{"role": "user", "content": config_prompt}]
            )
            config_json = response["message"]["content"].strip()
            # Extract JSON from markdown code blocks if present
            if "```json" in config_json:
                config_json = config_json.split("```json")[1].split("```")[0].strip()
            elif "```" in config_json:
                config_json = config_json.split("```")[1].split("```")[0].strip()
            
            config = json.loads(config_json)
            print_to_log(f"✅ Config parsed: {config}")
            
        except Exception as e:
            print_to_log(f"⚠️ Config parsing failed, using defaults: {e}")
            config = {
                "models_requested": ["random_forest", "xgboost", "lgbm"],
                "test_size": 0.2,
                "selection_metric": "accuracy",
                "problem_type": "classification"
            }
        
        # Step B: Model availability check (DIRECT CODE EXECUTION - NO LLM)
        print_to_log("🔧 Step B: Checking model availability...")
        if progress_callback:
            progress_callback("🔧 Checking available models...", "Model Setup")
        
        model_availability_code = f"""
import re

user_request = "{query}"
user_request_lower = user_request.lower()
models_requested = {config['models_requested']}

available_models = {{}}

try:
    from sklearn.ensemble import RandomForestClassifier
    available_models['Random Forest'] = RandomForestClassifier(random_state=42)
except ImportError:
    print("Warning: sklearn RandomForest not available")

try:
    from lightgbm import LGBMClassifier
    available_models['LightGBM'] = LGBMClassifier(verbosity=-1, random_state=42)
except ImportError:
    print("Warning: LightGBM not available")

try:
    from xgboost import XGBClassifier
    available_models['XGBoost'] = XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42)
except ImportError:
    print("Warning: XGBoost not available")

try:
    from sklearn.neural_network import MLPClassifier
    available_models['Neural Network'] = MLPClassifier(random_state=42, max_iter=500)
except ImportError:
    print("Warning: Neural Network not available")

try:
    from sklearn.tree import DecisionTreeClassifier
    available_models['Decision Tree'] = DecisionTreeClassifier(max_depth=5, random_state=42)
except ImportError:
    print("Warning: Decision Tree not available")

try:
    from sklearn.linear_model import LogisticRegression
    available_models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
except ImportError:
    print("Warning: Logistic Regression not available")

try:
    from sklearn.svm import SVC
    available_models['SVM'] = SVC(random_state=42, probability=True)
except ImportError:
    print("Warning: SVM not available")

print(f"Available models: {{list(available_models.keys())}}")

model_name_mapping = {{
    'lgbm': 'LightGBM',
    'xgboost': 'XGBoost',
    'random_forest': 'Random Forest',
    'decision_tree': 'Decision Tree',
    'neural_network': 'Neural Network',
    'logistic_regression': 'Logistic Regression',
    'svm': 'SVM'
}}

models_to_train = {{}}
for requested_model in models_requested:
    mapped_name = model_name_mapping.get(requested_model)
    if mapped_name and mapped_name in available_models:
        models_to_train[mapped_name] = available_models[mapped_name]
    else:
        print(f"Warning: {{requested_model}} -> {{mapped_name}} not available")

if len(models_to_train) < 2:
    available_list = list(available_models.keys())
    requested_list = models_requested
    error_msg = f"Insufficient Models for Comparison. Requested: {{requested_list}}, Available: {{available_list}}, Found: {{len(models_to_train)}} (need minimum 2)"
    raise ValueError(error_msg)

print(f"Final models to build: {{list(models_to_train.keys())}}")
"""
        
        # Execute model availability code
        env_b = {'sample_data': sample_data}
        exec(model_availability_code, env_b)
        models_to_train = env_b['models_to_train']
        print_to_log(f"✅ Models to train: {list(models_to_train.keys())}")
        
        # Step C: Training (DIRECT CODE EXECUTION - NO LLM)
        print_to_log("🏋️ Step C: Training models...")
        if progress_callback:
            progress_callback(f"🏋️ Training {len(models_to_train)} models...", "Model Training")
        
        training_code = f"""
import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, log_loss, roc_curve)
import warnings
warnings.filterwarnings('ignore')

X = sample_data.drop('target', axis=1)
y = sample_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={config['test_size']}, random_state=42)

models_results = {{}}

for model_name, model in models_to_train.items():
    try:
        print(f"Training {{model_name}}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        metrics = {{
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba[:, 1]) if y_proba is not None and y_proba.shape[1] == 2 else None,
        }}

        # Save model to artifacts directory with full path
        model_filename = f'{{model_name.replace(" ", "_")}}_model.joblib'
        model_full_path = os.path.join(artifacts_dir, model_filename)
        model_path = safe_joblib_dump(model, model_full_path)

        models_results[model_name] = {{
            'model': model,
            'model_path': model_path,
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist() if y_proba is not None else None,
            'training_time': training_time,
            'model_type': 'classification'
        }}

        print(f"✅ {{model_name}} completed - accuracy: {{metrics['accuracy']:.4f}}")
        
    except Exception as e:
        print(f"❌ Error training {{model_name}}: {{e}}")
        continue

print(f"✅ Trained {{len(models_results)}} models successfully")
"""
        
        # Execute training code
        env_c = {
            'sample_data': sample_data, 
            'models_to_train': models_to_train, 
            'safe_joblib_dump': safe_joblib_dump,
            'artifacts_dir': artifacts_dir
        }
        exec(training_code, env_c)
        models_results = env_c['models_results']
        X_test = env_c['X_test']
        y_test = env_c['y_test']
        print_to_log(f"✅ Trained {len(models_results)} models")
        
        # Step D: Visualization (DIRECT CODE EXECUTION - NO LLM)  
        print_to_log("📊 Step D: Creating visualizations...")
        if progress_callback:
            progress_callback("📊 Generating comparison plots...", "Visualization")
        
        plotting_code = """
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve

comparison_plots = {}

# ROC Curves
try:
    plt.figure(figsize=(10, 8))
    plotted = False
    for model_name, result in models_results.items():
        if result['probabilities'] is not None:
            y_proba_array = np.array(result['probabilities'])
            if y_proba_array.ndim == 2 and y_proba_array.shape[1] >= 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba_array[:, 1])
            else:
                fpr, tpr, _ = roc_curve(y_test, y_proba_array)
            roc_auc = result['metrics'].get('roc_auc', 0)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
            plotted = True
    
    if plotted:
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        
        # Save to artifacts directory with full path
        roc_filename = 'roc_comparison.png'
        roc_full_path = os.path.join(artifacts_dir, roc_filename)
        roc_plot_path = safe_plt_savefig(roc_full_path)
        comparison_plots['roc_curves'] = roc_plot_path
        print(f"✅ Saved ROC curves to: {roc_plot_path}")
        
except Exception as e:
    print(f"Error creating ROC curves: {e}")
    comparison_plots['roc_curves'] = None

# Metrics table
try:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC', 'Time (s)']
    
    for model_name, result in models_results.items():
        metrics = result['metrics']
        row = [
            model_name,
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            f"{metrics.get('f1', 0):.4f}",
            f"{metrics.get('roc_auc', 0):.4f}" if metrics.get('roc_auc') else 'N/A',
            f"{result.get('training_time', 0):.2f}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    plt.title('Model Performance Comparison', fontsize=14, pad=20)
    
    # Save to artifacts directory with full path
    metrics_filename = 'metrics_table.png'
    metrics_full_path = os.path.join(artifacts_dir, metrics_filename)
    metrics_plot_path = safe_plt_savefig(metrics_full_path)
    comparison_plots['metrics_table'] = metrics_plot_path
    print(f"✅ Saved metrics table to: {metrics_plot_path}")
    
except Exception as e:
    print(f"Error creating metrics table: {e}")
    comparison_plots['metrics_table'] = None

print(f"✅ Created {len(comparison_plots)} visualizations")
"""
        
        # Execute plotting code
        env_d = {
            'models_results': models_results,
            'X_test': X_test,
            'y_test': y_test,
            'safe_plt_savefig': safe_plt_savefig,
            'artifacts_dir': artifacts_dir
        }
        exec(plotting_code, env_d)
        comparison_plots = env_d['comparison_plots']
        print_to_log(f"✅ Created {len(comparison_plots)} visualizations")
        
        # Step E: Best model selection (DIRECT CODE EXECUTION - NO LLM)
        print_to_log("🏆 Step E: Selecting best model...")
        if progress_callback:
            progress_callback("🏆 Selecting best model...", "Model Selection")
        
        selection_code = f"""
selection_metric = '{config['selection_metric']}'

# Get scores for all models
scores = {{model_name: result['metrics'].get(selection_metric, 0) 
          for model_name, result in models_results.items()}}

# Find best and worst
best_model_name = max(scores, key=scores.get)
worst_model_name = min(scores, key=scores.get)
best_score = scores[best_model_name]
worst_score = scores[worst_model_name]

# Calculate improvement
improvement_percent = ((best_score - worst_score) / worst_score * 100) if worst_score != 0 else 0

# Create ranking
ranking_list = []
for rank, (model_name, score) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True), 1):
    ranking_list.append({{
        'rank': rank,
        'model_name': model_name,
        'score': score,
        'metric_used': selection_metric
    }})

# Build final result
result = {{
    'user_config': {{
        'models_requested': {config['models_requested']},
        'test_size': {config['test_size']},
        'selection_metric': selection_metric,
        'total_models_built': len(models_results)
    }},
    'models': models_results,
    'best_model': {{
        'name': best_model_name,
        'model': models_results[best_model_name]['model'],
        'model_path': models_results[best_model_name]['model_path'],
        'metrics': models_results[best_model_name]['metrics'],
        'selection_criteria': f'{{selection_metric}}: {{best_score:.4f}}',
        'improvement_over_worst': f'{{improvement_percent:.2f}}% better than worst model'
    }},
    'comparison_plots': comparison_plots,
    'model_ranking': ranking_list,
    'summary': {{
        'total_models': len(models_results),
        'best_model': best_model_name,
        'best_score': best_score,
        'worst_model': worst_model_name,
        'worst_score': worst_score,
        'performance_spread': f'{{improvement_percent:.2f}}% difference',
        'recommendation': f'Use {{best_model_name}} with {{selection_metric}} of {{best_score:.4f}}'
    }},
    'detailed_comparison': '\\n'.join([f"{{r['rank']}}. {{r['model_name']}}: {{r['score']:.4f}}" for r in ranking_list])
}}

print("✅ Best model selection completed")
"""
        
        # Execute selection code
        env_e = {
            'models_results': models_results,
            'comparison_plots': comparison_plots,
            'config': config
        }
        exec(selection_code, env_e)
        result = env_e['result']
        
        print_to_log("✅ Multi-model orchestrator completed successfully!")
        return result
        
    except Exception as e:
        print_to_log(f"❌ Orchestrator failed: {e}")
        import traceback
        print_to_log(f"💥 Traceback: {traceback.format_exc()}")
        return None

def format_multi_model_response(result: Dict, query: str) -> str:
    """Format comprehensive multi-model comparison response with rich details"""
    try:
        response_parts = ["✅ 🎯 Multi-Model Comparison Completed Successfully!\n"]
        
        # User configuration
        user_config = result.get('user_config', {})
        if user_config:
            response_parts.append("⚙️ Configuration:")
            models_req = user_config.get('models_requested', [])
            response_parts.append(f"   • Models Requested: {', '.join([m.title() for m in models_req])}")
            response_parts.append(f"   • Test Split: {user_config.get('test_size', 0.2):.0%}")
            response_parts.append(f"   • Selection Metric: {user_config.get('selection_metric', 'accuracy')}")
            response_parts.append(f"   • Models Built: {user_config.get('total_models_built', 0)}\n")
        
        # Model performance summary
        models = result.get('models', {})
        if models:
            response_parts.append("📊 Model Performance Summary:")
            for model_name, model_data in models.items():
                metrics = model_data.get('metrics', {})
                training_time = model_data.get('training_time', 0)
                model_type = model_data.get('model_type', 'classification')
                
                # Show relevant metrics based on problem type
                if model_type == 'classification':
                    acc = metrics.get('accuracy', 0)
                    auc = metrics.get('roc_auc', 0)
                    f1 = metrics.get('f1', 0)
                    response_parts.append(f"   🤖 {model_name}: Acc: {acc:.3f} | AUC: {auc:.3f} | F1: {f1:.3f} | Time: {training_time:.2f}s")
                else:  # regression
                    r2 = metrics.get('r2_score', 0)
                    mae = metrics.get('mae', 0)
                    rmse = metrics.get('rmse', 0)
                    response_parts.append(f"   🤖 {model_name}: R²: {r2:.3f} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | Time: {training_time:.2f}s")
        
        # Model ranking
        ranking = result.get('model_ranking', [])
        if ranking and len(ranking) > 1:
            response_parts.append("\n🏅 Model Ranking:")
            for rank_info in ranking:
                rank = rank_info.get('rank', 0)
                model_name = rank_info.get('model_name', 'Unknown')
                score = rank_info.get('score', 0)
                metric = rank_info.get('metric_used', 'score')
                response_parts.append(f"   #{rank}. {model_name}: {metric.upper()} = {score:.3f}")
        
        # Best model details
        best_model = result.get('best_model', {})
        if best_model:
            response_parts.append(f"\n🏆 Winner: {best_model.get('name', 'Unknown')}")
            response_parts.append(f"   📈 Selection: {best_model.get('selection_criteria', 'Not specified')}")
            improvement = best_model.get('improvement_over_worst', '')
            if improvement:
                response_parts.append(f"   📊 Performance: {improvement}")
        
        # Summary insights
        summary_data = result.get('summary', {})
        if isinstance(summary_data, dict):
            response_parts.append("\n📋 Analysis:")
            response_parts.append(f"   • Best: {summary_data.get('best_model', 'Unknown')} ({summary_data.get('best_score', 0):.3f})")
            response_parts.append(f"   • Worst: {summary_data.get('worst_model', 'Unknown')} ({summary_data.get('worst_score', 0):.3f})")
            spread = summary_data.get('performance_spread', '')
            if spread:
                response_parts.append(f"   • Performance Spread: {spread}")
            
            recommendation = summary_data.get('recommendation', '')
            if recommendation:
                response_parts.append(f"\n💡 Recommendation: {recommendation}")
        
        # Visualizations
        plots = result.get('comparison_plots', {})
        if plots:
            response_parts.append("\n📊 Visualizations Generated:")
            if plots.get('roc_curves'):
                response_parts.append("   • ROC Curves Comparison Plot")
            if plots.get('metrics_table'):
                response_parts.append("   • Metrics Comparison Table")
        
        # Detailed comparison
        detailed_comparison = result.get('detailed_comparison', '')
        if detailed_comparison:
            response_parts.append("\n🔍 Detailed Analysis:")
            response_parts.append(f"Comprehensive textual comparison of all models with strengths/weaknesses")
            response_parts.append(detailed_comparison)
        
        return "\n".join(response_parts)
        
    except Exception as e:
        print_to_log(f"⚠️ Error formatting multi-model response: {e}")
        return f"✅ Multi-model comparison completed with {len(result.get('models', {}))} models"

def format_model_response(result: Dict, routing_decision: str, query: str) -> str:
    """Format detailed response with model metrics for Slack display"""
    try:
        if not isinstance(result, dict):
            return "✅ Model operation completed successfully!"
        
        # Handle missing library errors
        if result.get("execution_status") == "failed_missing_library":
            missing_lib = result.get("error", "").replace("Missing library: ", "")
            response_parts = [
                f"📦 Missing Library: {missing_lib}",
                f"",
                f"🔒 For System Administrator:",
                f"• {result.get('admin_message', 'Please install the required library')}",
                f"• Description: {result.get('description', 'External library required')}",
                f"",
                f"🔄 Available Alternatives:"
            ]
            
            alternatives = result.get('alternatives', [])
            if alternatives:
                for alt in alternatives:
                    response_parts.append(f"• {alt}")
            else:
                response_parts.append("• No direct alternatives available")
            
            response_parts.extend([
                f"",
                f"💡 Next Steps:",
                f"1. Contact your system administrator to install the library",
                f"2. Or ask me to try a different approach using available libraries",
                f"3. Or specify which libraries you have available"
            ])
            
            return "\n".join(response_parts)
        
        # Check if we have model metrics (classification or regression)
        if routing_decision == "build_new_model" and any(key in result for key in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score', 'mean_squared_error', 'mean_absolute_error']):
            # Determine model type and problem type
            model_type = "Model"
            model_emoji = "🤖"
            if "RandomForest" in str(type(result.get('model', ''))):
                model_type = "Random Forest"
                model_emoji = "🌳"
            elif "DecisionTree" in str(type(result.get('model', ''))):
                model_type = "Decision Tree"
                model_emoji = "🌲"
            elif "LGBM" in str(type(result.get('model', ''))):
                model_type = "LightGBM"
                model_emoji = "🚀"
            elif "XGB" in str(type(result.get('model', ''))):
                model_type = "XGBoost"
                model_emoji = "⚡"
            elif "LinearRegression" in str(type(result.get('model', ''))):
                model_type = "Linear Regression"
                model_emoji = "📈"
            elif "Ridge" in str(type(result.get('model', ''))):
                model_type = "Ridge Regression"
                model_emoji = "📈"
            elif "Lasso" in str(type(result.get('model', ''))):
                model_type = "Lasso Regression"
                model_emoji = "📈"
            elif "ElasticNet" in str(type(result.get('model', ''))):
                model_type = "ElasticNet Regression"
                model_emoji = "📈"
            elif "RandomForestRegressor" in str(type(result.get('model', ''))):
                model_type = "Random Forest Regressor"
                model_emoji = "🌳"
            elif "DecisionTreeRegressor" in str(type(result.get('model', ''))):
                model_type = "Decision Tree Regressor"
                model_emoji = "🌲"
            elif "GradientBoosting" in str(type(result.get('model', ''))):
                model_type = "Gradient Boosting Regressor"
                model_emoji = "🚀"
            elif "SVR" in str(type(result.get('model', ''))):
                model_type = "Support Vector Regressor"
                model_emoji = "⚙️"
            
            # Check if it's classification or regression
            is_classification = any(key in result for key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'])
            is_regression = any(key in result for key in ['r2_score', 'mean_squared_error', 'mean_absolute_error'])
            
            # Format header with model name (clean style like original)
            response_parts = [f"✅ {model_emoji} {model_type} Model Training Completed Successfully!"]
            
            # Check if we have both training and test metrics for overfitting detection
            has_train_metrics = any(key.startswith('train_') for key in result.keys())
            
            if has_train_metrics:
                if is_classification:
                    response_parts.append(f"📊 Classification Performance (Train vs Test):")
                    response_parts.append(f"*Compare metrics to detect overfitting*")
                elif is_regression:
                    response_parts.append(f"📊 Regression Performance (Train vs Test):")
                    response_parts.append(f"*Compare metrics to detect overfitting*")
                else:
                    response_parts.append(f"📊 Model Performance (Train vs Test):")
                    response_parts.append(f"*Compare metrics to detect overfitting*")
                
                # Classification metrics with train vs test comparison
                if 'accuracy' in result and 'train_accuracy' in result:
                    train_acc = result['train_accuracy']
                    test_acc = result['accuracy']
                    diff = train_acc - test_acc
                    flag = "🚨 OVERFITTING!" if diff > 0.05 else "✅ Good" if diff < 0.02 else "⚠️ Monitor"
                    response_parts.append(f"• Accuracy: Train: {train_acc:.4f} | Test: {test_acc:.4f} | Diff: {diff:+.4f} {flag}")
                
                if 'f1_score' in result and 'train_f1_score' in result:
                    train_f1 = result['train_f1_score']
                    test_f1 = result['f1_score']
                    diff = train_f1 - test_f1
                    flag = "🚨 OVERFITTING!" if diff > 0.05 else "✅ Good" if diff < 0.02 else "⚠️ Monitor"
                    response_parts.append(f"• F1 Score: Train: {train_f1:.4f} | Test: {test_f1:.4f} | Diff: {diff:+.4f} {flag}")
                
                if 'roc_auc' in result and 'train_roc_auc' in result:
                    train_auc = result['train_roc_auc']
                    test_auc = result['roc_auc']
                    diff = train_auc - test_auc
                    flag = "🚨 OVERFITTING!" if diff > 0.05 else "✅ Good" if diff < 0.02 else "⚠️ Monitor"
                    response_parts.append(f"• ROC AUC: Train: {train_auc:.4f} | Test: {test_auc:.4f} | Diff: {diff:+.4f} {flag}")
                
                if 'precision' in result and 'train_precision' in result:
                    response_parts.append(f"• Precision: Train: {result['train_precision']:.4f} | Test: {result['precision']:.4f}")
                
                if 'recall' in result and 'train_recall' in result:
                    response_parts.append(f"• Recall: Train: {result['train_recall']:.4f} | Test: {result['recall']:.4f}")
                
                # Regression metrics with train vs test comparison
                if 'r2' in result and 'train_r2' in result:
                    train_r2 = result['train_r2']
                    test_r2 = result['r2']
                    diff = train_r2 - test_r2
                    flag = "🚨 OVERFITTING!" if diff > 0.05 else "✅ Good" if diff < 0.02 else "⚠️ Monitor"
                    response_parts.append(f"• R² Score: Train: {train_r2:.4f} | Test: {test_r2:.4f} | Diff: {diff:+.4f} {flag}")
                
                if 'mae' in result and 'train_mae' in result:
                    train_mae = result['train_mae']
                    test_mae = result['mae']
                    diff = test_mae - train_mae  # For error metrics, higher test error indicates overfitting
                    flag = "🚨 OVERFITTING!" if diff > 0.05 else "✅ Good" if diff < 0.02 else "⚠️ Monitor"
                    response_parts.append(f"• MAE: Train: {train_mae:.4f} | Test: {test_mae:.4f} | Diff: {diff:+.4f} {flag}")
                
                if 'rmse' in result and 'train_rmse' in result:
                    train_rmse = result['train_rmse']
                    test_rmse = result['rmse']
                    diff = test_rmse - train_rmse  # For error metrics, higher test error indicates overfitting
                    flag = "🚨 OVERFITTING!" if diff > 0.05 else "✅ Good" if diff < 0.02 else "⚠️ Monitor"
                    response_parts.append(f"• RMSE: Train: {train_rmse:.4f} | Test: {test_rmse:.4f} | Diff: {diff:+.4f} {flag}")
                
                # Additional metrics without comparison
                if 'specificity' in result:
                    response_parts.append(f"• Specificity (Test): {result['specificity']:.4f}")
                if 'log_loss' in result:
                    response_parts.append(f"• Log Loss (Test): {result['log_loss']:.4f}")
                
            else:
                # Fallback to original display if no training metrics available
                if is_classification:
                    response_parts.append(f"📊 Classification Performance:")
                elif is_regression:
                    response_parts.append(f"📊 Regression Performance:")
                else:
                    response_parts.append(f"📊 {model_emoji} {model_type} Model Performance:")
                
                # Add classification metrics (clean format)
                if 'accuracy' in result:
                    response_parts.append(f"• Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
                if 'precision' in result:
                    response_parts.append(f"• Precision: {result['precision']:.4f}")
                if 'recall' in result:
                    response_parts.append(f"• Recall: {result['recall']:.4f}")
                if 'f1_score' in result:
                    response_parts.append(f"• F1 Score: {result['f1_score']:.4f}")
                if 'roc_auc' in result:
                    response_parts.append(f"• ROC AUC: {result['roc_auc']:.4f}")
                if 'specificity' in result:
                    response_parts.append(f"• Specificity: {result['specificity']:.4f}")
                
                # Add regression metrics (clean format)
                if 'r2' in result:
                    response_parts.append(f"• R² Score: {result['r2']:.4f} ({result['r2']*100:.2f}% variance explained)")
                elif 'r2_score' in result:
                    response_parts.append(f"• R² Score: {result['r2_score']:.4f} ({result['r2_score']*100:.2f}% variance explained)")
                if 'mae' in result:
                    response_parts.append(f"• Mean Absolute Error (MAE): {result['mae']:.4f}")
                elif 'mean_absolute_error' in result:
                    response_parts.append(f"• Mean Absolute Error (MAE): {result['mean_absolute_error']:.4f}")
                if 'rmse' in result:
                    response_parts.append(f"• Root Mean Squared Error (RMSE): {result['rmse']:.4f}")
                elif 'mean_squared_error' in result:
                    # Calculate RMSE if not provided
                    import math
                    rmse = math.sqrt(result['mean_squared_error'])
                    response_parts.append(f"• Root Mean Squared Error (RMSE): {rmse:.4f}")
                if 'mape' in result:
                    response_parts.append(f"• Mean Absolute Percentage Error (MAPE): {result['mape']:.4f}%")
                elif 'mean_absolute_percentage_error' in result:
                    response_parts.append(f"• Mean Absolute Percentage Error (MAPE): {result['mean_absolute_percentage_error']:.4f}%")
            
            # Add confusion matrices if available (enhanced format with train vs test)
            if 'confusion_matrix' in result:
                cm_test = result['confusion_matrix']
                cm_train = result.get('train_confusion_matrix')
                
                if isinstance(cm_test, list) and len(cm_test) == 2 and len(cm_test[0]) == 2:
                    if cm_train and isinstance(cm_train, list) and len(cm_train) == 2 and len(cm_train[0]) == 2:
                        # Show both training and test confusion matrices
                        response_parts.append(f"\n📋 *Confusion Matrices (Train vs Test):*")
                        
                        tn_test, fp_test = cm_test[0]
                        fn_test, tp_test = cm_test[1]
                        tn_train, fp_train = cm_train[0]
                        fn_train, tp_train = cm_train[1]
                        
                        response_parts.append(f"```")
                        response_parts.append(f"🏋️ TRAINING SET:          📊 TEST SET:")
                        response_parts.append(f"         Predicted                Predicted")
                        response_parts.append(f"Actual   0     1           Actual   0     1")
                        response_parts.append(f"   0   {tn_train:4}  {fp_train:4}             0   {tn_test:4}  {fp_test:4}")
                        response_parts.append(f"   1   {fn_train:4}  {tp_train:4}             1   {fn_test:4}  {tp_test:4}")
                        response_parts.append(f"```")
                        
                        # Calculate and show key differences
                        train_accuracy = (tp_train + tn_train) / (tp_train + tn_train + fp_train + fn_train)
                        test_accuracy = (tp_test + tn_test) / (tp_test + tn_test + fp_test + fn_test)
                        auc_diff = train_auc - test_auc
                        
                        if auc_diff > 0.05:
                            response_parts.append(f"⚠️ Overfitting Alert: Training AUC ({train_auc:.3f}) significantly higher than test ({test_auc:.3f})")
                        elif auc_diff > 0.025:
                            response_parts.append(f"�� Mild Overfitting: Small gap between train ({train_auc:.3f}) and test ({test_auc:.3f}) auc")
                        else:
                            response_parts.append(f"✅ Good Fit: Similar performance on train ({train_auc:.3f}) and test ({test_auc:.3f})")
                    else:
                        # Show only test confusion matrix if training not available
                        tn_test, fp_test = cm_test[0]
                        fn_test, tp_test = cm_test[1]
                        response_parts.append(f"\n📋 Confusion Matrix (Test Set):")
                        response_parts.append(f"```")
                        response_parts.append(f"         Predicted")
                        response_parts.append(f"Actual   0     1")
                        response_parts.append(f"   0   {tn_test:4}  {fp_test:4}")
                        response_parts.append(f"   1   {fn_test:4}  {tp_test:4}")
                        response_parts.append(f"```")
            
            # Add model save information (clean format)
            if 'model_path' in result and result['model_path']:
                response_parts.append(f"\n💾 Model saved: {result['model_path'].split('/')[-1]}")
            
            # Check for rank ordering table (SAME logic as use_existing_model)
            if 'rank_ordering_table' in result:
                rank_table = result['rank_ordering_table']
                if isinstance(rank_table, list) and len(rank_table) > 0:
                    response_parts.append(f"\n📊 *Rank Ordering Table:*")
                    
                    # Create header with optimized column widths
                    headers = ["Bucket", "Threshold", "BucketCount", "EventCount", "Event%", "CumEvent%", "Coverage%"]
                    # Optimized widths: [6, 11, 11, 10, 7, 9, 9]
                    widths = [6, 11, 11, 10, 7, 9, 9]
                    header_line = "|" + "|".join(f" {h:^{w}} " for h, w in zip(headers, widths)) + "|"
                    separator_line = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
                    
                    table_lines = [
                        "```",
                        header_line,
                        separator_line
                    ]
                    
                    # Add data rows
                    for i, row in enumerate(rank_table):
                        bucket = row.get('bucket', 0)
                        total_users = row.get('totalUsersCount', 0)
                        no_of_events = row.get('numBads', 0)  # numBads = no of events
                        event_rate = row.get('badrate', 0)    # badrate = event rate
                        cum_event_rate = row.get('cum_badrate', 0)  # cum_badrate = cumulative event rate
                        coverage = row.get('coverage', 0)
                        
                        # Get actual min/max probabilities for threshold range
                        min_threshold = row.get('min_threshold', 0)
                        max_threshold = row.get('max_threshold', 0)
                        
                        # If min/max not available, use avg_probability as fallback
                        if min_threshold == 0 and max_threshold == 0:
                            avg_prob = row.get('avg_probability', 0)
                            # Estimate range based on bucket position (rough approximation)
                            if i == 0:
                                threshold = f"0.000-{avg_prob:.3f}"
                            elif i == len(rank_table) - 1:
                                threshold = f"{avg_prob:.3f}-1.000"
                            else:
                                # Use a small range around average
                                range_size = 0.02  # Approximate range
                                threshold = f"{max(0, avg_prob - range_size/2):.3f}-{min(1, avg_prob + range_size/2):.3f}"
                        else:
                            threshold = f"{min_threshold:.3f}-{max_threshold:.3f}"
                        
                        # Format with optimized widths: [6, 11, 11, 10, 7, 9, 9]
                        data_line = f"| {bucket:^6} | {threshold:^11} | {total_users:^11,} | {no_of_events:^10,} | {event_rate:^7.4f} | {cum_event_rate:^9.4f} | {coverage:^8.1f}% |"
                        table_lines.append(data_line)
                    
                    table_lines.append("```")
                    response_parts.extend(table_lines)
            
            response_parts.append(f"\n🎯 You can now use this model for predictions, visualizations, or further analysis!")
            
            return "\n".join(response_parts)
        
        elif routing_decision == "use_existing_model":
            response_parts = []
            
            # Check if there are validation metrics to display
            if any(key in result for key in ['accuracy', 'precision', 'recall', 'f1_score']):
                response_parts.append("✅ Existing Model Analysis Completed!\n")
                response_parts.append("📊 Model Performance:")
                
                if 'accuracy' in result:
                    response_parts.append(f"• Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
                if 'precision' in result:
                    response_parts.append(f"• Precision: {result['precision']:.4f}")
                if 'recall' in result:
                    response_parts.append(f"• Recall: {result['recall']:.4f}")
                if 'f1_score' in result:
                    response_parts.append(f"• F1 Score: {result['f1_score']:.4f}")
                if 'roc_auc' in result:
                    response_parts.append(f"• ROC AUC: {result['roc_auc']:.4f}")
                if 'specificity' in result:
                    response_parts.append(f"• Specificity: {result['specificity']:.4f}")
                
                # Add confusion matrix if available
                if 'confusion_matrix' in result:
                    cm = result['confusion_matrix']
                    if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
                        tn, fp = cm[0]
                        fn, tp = cm[1]
                        response_parts.append(f"\n📋 Confusion Matrix:")
                        response_parts.append(f"```")
                        response_parts.append(f"         Predicted")
                        response_parts.append(f"Actual   0     1")
                        response_parts.append(f"   0   {tn:4}  {fp:4}")
                        response_parts.append(f"   1   {fn:4}  {tp:4}")
                        response_parts.append(f"```")
            
            # Check if there's a rank ordering table to display
            if 'rank_ordering_table' in result:
                rank_table = result['rank_ordering_table']
                if isinstance(rank_table, list) and len(rank_table) > 0:
                    if response_parts:  # Add spacing if validation metrics were shown above
                        response_parts.append("\n")
                    response_parts.append("📊 *Rank Ordering Table:*")
                    
                    # Create header with optimized column widths
                    headers = ["Bucket", "Threshold", "BucketCount", "EventCount", "Event%", "CumEvent%", "Coverage%"]
                    # Optimized widths: [6, 11, 11, 10, 7, 9, 9]
                    widths = [6, 11, 11, 10, 7, 9, 9]
                    header_line = "|" + "|".join(f" {h:^{w}} " for h, w in zip(headers, widths)) + "|"
                    separator_line = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
                    
                    table_lines = [
                        "```",
                        header_line,
                        separator_line
                    ]
                    
                    # Add data rows
                    for i, row in enumerate(rank_table):
                        bucket = row.get('bucket', 0)
                        total_users = row.get('totalUsersCount', 0)
                        no_of_events = row.get('numBads', 0)  # numBads = no of events
                        event_rate = row.get('badrate', 0)    # badrate = event rate
                        cum_event_rate = row.get('cum_badrate', 0)  # cum_badrate = cumulative event rate
                        coverage = row.get('coverage', 0)
                        
                        # Get actual min/max probabilities for threshold range
                        min_threshold = row.get('min_threshold', 0)
                        max_threshold = row.get('max_threshold', 0)
                        
                        # If min/max not available, use avg_probability as fallback
                        if min_threshold == 0 and max_threshold == 0:
                            avg_prob = row.get('avg_probability', 0)
                            # Estimate range based on bucket position (rough approximation)
                            if i == 0:
                                threshold = f"0.000-{avg_prob:.3f}"
                            elif i == len(rank_table) - 1:
                                threshold = f"{avg_prob:.3f}-1.000"
                            else:
                                # Use a small range around average
                                range_size = 0.02  # Approximate range
                                threshold = f"{max(0, avg_prob - range_size/2):.3f}-{min(1, avg_prob + range_size/2):.3f}"
                        else:
                            threshold = f"{min_threshold:.3f}-{max_threshold:.3f}"
                        
                        # Format with optimized widths: [6, 11, 11, 10, 7, 9, 9]
                        data_line = f"| {bucket:^6} | {threshold:^11} | {total_users:^11,} | {no_of_events:^10,} | {event_rate:^7.4f} | {cum_event_rate:^9.4f} | {coverage:^8.1f}% |"
                        table_lines.append(data_line)
                    
                    table_lines.append("```")
                    response_parts.extend(table_lines)
            
            # Return combined response or default
            if response_parts:
                response_parts.append(f"\n🎯 Model analysis completed successfully!")
                return "\n".join(response_parts)
            else:
                return "✅ Existing Model Operation Completed!\n\n🎯 Your existing model has been used successfully for the requested operation."
        
        elif routing_decision == "execute_code":
            return "✅ Code Execution Completed!\n\n📊 Your custom analysis has been executed successfully. Check the results above."
        
        else:
            # Generic success for other cases
            return "✅ Model operation completed successfully!"
            
    except Exception as e:
        print_to_log(f"⚠️ Error formatting model response: {e}")
        return f"❌ Error formatting response: {str(e)}"

# =============================================================================
# MAIN INTERFACE
# =============================================================================

class LangGraphModelAgent:
    """Main interface for the LangGraph-based model agent"""
    
    def __init__(self):
        self.graph = create_agent_graph()
        self.user_states = {}  # Store per-user-thread state
        self.base_data_dir = "user_data"  # Base directory for all user data
        self._ensure_base_directory()
    
    def _ensure_base_directory(self):
        """Ensure base user data directory exists"""
        if not os.path.exists(self.base_data_dir):
            os.makedirs(self.base_data_dir)
            print_to_log(f"📁 Created base directory: {self.base_data_dir}")
    
    def _get_thread_id(self, user_id: str) -> tuple[str, str]:
        """Extract user and thread from user_id format: user_threadts"""
        if "_" in user_id:
            parts = user_id.split("_", 1)  # Split only on first underscore
            return parts[0], parts[1]  # user, thread_ts
        return user_id, "main"  # fallback to main thread
    
    def _get_user_thread_dir(self, user_id: str) -> str:
        """Get directory path for specific user thread"""
        user, thread_ts = self._get_thread_id(user_id)
        # Get username for proper directory structure
        username = get_username_for_user_id(user)
        thread_dir = os.path.join(self.base_data_dir, username, thread_ts)
        if not os.path.exists(thread_dir):
            os.makedirs(thread_dir, exist_ok=True)
            print_to_log(f"📁 Created thread directory: {thread_dir}")
        return thread_dir
    
    # Conversation history is now handled by the main pipeline - methods removed
    
    def _get_artifacts_dir(self, user_id: str) -> str:
        """Get artifacts directory for specific thread"""
        thread_dir = self._get_user_thread_dir(user_id)
        artifacts_dir = os.path.join(thread_dir, "artifacts")
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir, exist_ok=True)
        return artifacts_dir
    
    def _get_data_file(self, user_id: str) -> str:
        """Get data file path for specific thread"""
        thread_dir = self._get_user_thread_dir(user_id)
        return os.path.join(thread_dir, "session_data.pkl")
    
    def _save_session_data(self, user_id: str):
        """Save DataFrame data to disk for specific thread"""
        try:
            if user_id in self.user_states and "data" in self.user_states[user_id]:
                data_file = self._get_data_file(user_id)
                data = self.user_states[user_id]["data"]
                data.to_pickle(data_file)
                user, thread_ts = self._get_thread_id(user_id)
                print_to_log(f"💾 Saved session data for user {user}, thread {thread_ts}: {data.shape}")
        except Exception as e:
            print_to_log(f"⚠️ Failed to save session data for {user_id}: {e}")
    
    def _load_session_data(self, user_id: str) -> pd.DataFrame:
        """Load DataFrame data from disk for specific thread"""
        try:
            data_file = self._get_data_file(user_id)
            if os.path.exists(data_file):
                data = pd.read_pickle(data_file)
                user, thread_ts = self._get_thread_id(user_id)
                print_to_log(f"📊 Restored session data for user {user}, thread {thread_ts}: {data.shape}")
                return data
        except Exception as e:
            print_to_log(f"⚠️ Failed to load session data for {user_id}: {e}")
        return pd.DataFrame()  # Return empty DataFrame if loading fails
        
    def load_data(self, data: pd.DataFrame, user_id: str = "default_user"):
        """Load data for a user session"""
        if user_id not in self.user_states:
            self.user_states[user_id] = {}
        
        self.user_states[user_id]["data"] = data
        self.user_states[user_id]["has_existing_model"] = False
        self.user_states[user_id]["model_path"] = None
        
        # Also store in global_model_states for access by generate_model_code
        if user_id not in global_model_states:
            global_model_states[user_id] = {}
        global_model_states[user_id]['sample_data'] = data
        
        # Initialize messages if not present
        if "messages" not in self.user_states[user_id]:
            self.user_states[user_id]["messages"] = []
        
        # Add data upload message
        self.user_states[user_id]["messages"].append({
            "type": "data_upload",
            "content": f"data with shape {data.shape}",
            "timestamp": datetime.now().isoformat()
        })
        
        # Analyze column types for better logging
        numeric_cols = data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        boolean_cols = data.select_dtypes(include=['bool']).columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        
        print_to_log(f"📊 Data loaded for user {user_id}: {data.shape}")
        print_to_log(f"   📊 Numeric features: {len(numeric_cols)}, 🏷️ Categorical: {len(categorical_cols)}")
        if len(boolean_cols) > 0:
            print_to_log(f"   ✅ Boolean features: {len(boolean_cols)}")
        if len(datetime_cols) > 0:
            print_to_log(f"   📅 DateTime features: {len(datetime_cols)}")
        print_to_log(f"   🎯 Target column: {'target' if 'target' in data.columns else 'Not found'}")
        self._save_session_data(user_id)  # Also save the DataFrame data
    
        # Conversation history methods removed - handled by main pipeline

    def process_query(self, query: str, user_id: str = "default_user", progress_callback=None) -> Dict[str, Any]:
        """Process a user query through the agent graph with optional progress updates"""
        
        print_to_log(f"🔍 PROCESS_QUERY DEBUG - progress_callback received: {progress_callback}")
        print_to_log(f"🔍 PROCESS_QUERY DEBUG - progress_callback type: {type(progress_callback)}")
        print_to_log(f"🔍 PROCESS_QUERY DEBUG - progress_callback is None: {progress_callback is None}")
        
        # Initialize user state if not present and load conversation history
        if user_id not in self.user_states:
            self.user_states[user_id] = {}
        if "messages" not in self.user_states[user_id]:
            self.user_states[user_id]["messages"] = []
        
        # Add current query to messages
        self.user_states[user_id]["messages"].append({
            "type": "user_query",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep conversation history but limit to reasonable size (last 20 messages)
        # This preserves important context while preventing unbounded memory growth
        if len(self.user_states[user_id]["messages"]) > 20:
            # Keep first message (often data upload) and last 19 messages
            first_msg = self.user_states[user_id]["messages"][0]
            recent_msgs = self.user_states[user_id]["messages"][-19:]
            self.user_states[user_id]["messages"] = [first_msg] + recent_msgs
        
        # Get model info from global model states (more up-to-date than user_states)
        global_model_info = global_model_states.get(user_id, {})
        user_model_path = global_model_info.get('model_path') or self.user_states.get(user_id, {}).get("model_path")
        
        # If no model path in memory, check for existing model files
        if not user_model_path:
            import glob
            recent_models = []
            
            # First, search in artifacts directory
            if '_' in user_id:
                user_id_part, thread_id = user_id.split('_', 1)
                artifacts_dir = get_username_artifacts_dir(user_id_part, thread_id)
                username = get_username_for_user_id(user_id_part)
            else:
                username = get_username_for_user_id(user_id)
                artifacts_dir = None
            if artifacts_dir and os.path.exists(artifacts_dir):
                artifacts_models = glob.glob(os.path.join(artifacts_dir, f"model_{username}_*.joblib"))
                artifacts_models.extend(glob.glob(os.path.join(artifacts_dir, "*model*.joblib")))
                recent_models.extend(artifacts_models)
                if artifacts_models:
                    print_to_log(f"🔍 Found {len(artifacts_models)} model(s) in artifacts directory")
            
            # Fallback to current directory
            root_models = glob.glob(f"model_{username}_*.joblib")
            recent_models.extend(root_models)
            if root_models:
                print_to_log(f"🔍 Found {len(root_models)} model(s) in root directory")
            
            if recent_models:
                # PRIORITY 1: Check if we have best model info from previous multi-model result
                best_model_path = None
                last_result = global_model_states.get(user_id, {}).get('last_result', {})
                
                if isinstance(last_result, dict) and 'best_model' in last_result:
                    best_model_info = last_result['best_model']
                    if isinstance(best_model_info, dict) and 'model_path' in best_model_info:
                        candidate_path = best_model_info['model_path']
                        if candidate_path in recent_models and os.path.exists(candidate_path):
                            best_model_path = candidate_path
                            best_model_name = best_model_info.get('name', 'Unknown')
                            print_to_log(f"🏆 Found best model from previous multi-model result: {best_model_name}")
                            print_to_log(f"🔍 Selected best model: {best_model_path}")
                
                # PRIORITY 2: If no best model info, fall back to most recent
                if not best_model_path:
                    user_model_path = max(recent_models, key=os.path.getctime)
                    print_to_log(f"🔍 No best model info found - selected most recent model: {user_model_path}")
                else:
                    user_model_path = best_model_path
                
                # Update global state (preserve existing data)
                if user_id not in global_model_states:
                    global_model_states[user_id] = {}
                global_model_states[user_id]['model_path'] = user_model_path
                if 'last_result' not in global_model_states[user_id]:
                    global_model_states[user_id]['last_result'] = {}
                # Preserve sample_data if it exists
        
        has_model = bool(user_model_path) or self.user_states.get(user_id, {}).get("has_existing_model", False)
        
        # Get data for this user session (try memory first, then disk)
        data = self.user_states.get(user_id, {}).get("data")
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            restored_data = self._load_session_data(user_id)
            if not restored_data.empty:
                if user_id not in self.user_states:
                    self.user_states[user_id] = {}
                self.user_states[user_id]["data"] = restored_data
                data = restored_data
                
                # Also store in global_model_states for access by generate_model_code
                if user_id not in global_model_states:
                    global_model_states[user_id] = {}
                global_model_states[user_id]['sample_data'] = restored_data
                
                print_to_log(f"🔄 Restored session data from disk: {data.shape}")
                print_to_log(f"🔄 Updated global_model_states with restored sample_data")
        
        # Initialize state
        initial_state = AgentState(
            user_id=user_id,
            query=query,
            data=data,
            messages=self.user_states.get(user_id, {}).get("messages", []),
            intent="",
            routing_decision="",
            code="",
            execution_result=None,
            model_path=user_model_path,
            has_existing_model=has_model,
            response="",
            artifacts={},
            progress_callback=progress_callback  # Add progress callback to state
        )
        
        print_to_log(f"🚀 NEW QUERY [{datetime.now().strftime('%H:%M:%S')}] User: {user_id} | Query: {query[:60]}{'...' if len(query) > 60 else ''}")
        
        # Send initial progress update
        print_to_log(f"🔍 DEBUG - progress_callback is None: {progress_callback is None}")
        if progress_callback:
            print_to_log(f"📡 CALLING progress_callback: Starting query analysis...")
            progress_callback("Starting query analysis...", "Intent Classification")
        else:
            print_to_log(f"⚠️ progress_callback is None - no progress updates will be sent")
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Send completion progress update
        if progress_callback:
            progress_callback("Processing completed!", "Finished")
        
        # Update user state
        if user_id not in self.user_states:
            self.user_states[user_id] = {}
            
        self.user_states[user_id]["has_existing_model"] = final_state.get("has_existing_model", False)
        self.user_states[user_id]["model_path"] = final_state.get("model_path")
        
        # Add model built message if a new model was created
        if final_state.get("routing_decision") == "build_new_model" and final_state.get("has_existing_model"):
            if "messages" not in self.user_states[user_id]:
                self.user_states[user_id]["messages"] = []
            self.user_states[user_id]["messages"].append({
                "type": "model_built",
                "content": "model successfully built",
                "timestamp": datetime.now().isoformat()
            })
        
        # Also sync with global model states for consistency (preserve existing data)
        execution_result = final_state.get("execution_result")
        
        # SPECIAL HANDLING: Multi-model results need different storage logic
        if isinstance(execution_result, dict) and 'models' in execution_result and 'best_model' in execution_result:
            # This is a multi-model result - store the entire result and use best model path
            if user_id not in global_model_states:
                global_model_states[user_id] = {}
            
            best_model_info = execution_result.get('best_model', {})
            best_model_path = best_model_info.get('model_path')
            
            if best_model_path:
                global_model_states[user_id]['model_path'] = best_model_path
                print_to_log(f"🏆 Synced best model from multi-model result: {best_model_info.get('name', 'Unknown')}")
            
            # Store the complete multi-model result for future "use best model" queries
            global_model_states[user_id]['last_result'] = execution_result
            print_to_log(f"🔄 Stored complete multi-model result for {user_id}")
            
        elif final_state.get("model_path"):
            # Regular single model result
            if user_id not in global_model_states:
                global_model_states[user_id] = {}
            global_model_states[user_id]['model_path'] = final_state["model_path"]
            global_model_states[user_id]['last_result'] = execution_result
            # Preserve sample_data if it exists
            print_to_log(f"🔄 Synced model state for {user_id}: {final_state['model_path']}")
        
        print_to_log(f"✅ COMPLETED [{user_id}] Response ready")
        
        # Conversation history is now handled by the main pipeline
        # self._save_conversation_history(user_id)  # Disabled to avoid duplicates
        
        return {
            "response": final_state["response"],
            "intent": final_state["intent"],
            "routing_decision": final_state["routing_decision"],
            "execution_result": final_state.get("execution_result"),
            "messages": final_state["messages"]
        }

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create agent
    agent = LangGraphModelAgent()
    
    # Load sample data
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    })
    
    agent.load_data(data, "test_user")
    
    # Test queries
    test_queries = [
        "build lgbm model",
        "use this model and build 10 segments",
        "show me data shape"
    ]
    
    for query in test_queries:
        print_to_log(f"\n{'='*80}")
        print_to_log(f"Testing: {query}")
        print_to_log('='*80)
        
        result = agent.process_query(query, "test_user")
        
        print_to_log(f"Response: {result['response']}")
        print_to_log(f"Intent: {result['intent']}")
        print_to_log(f"Routing: {result['routing_decision']}")
