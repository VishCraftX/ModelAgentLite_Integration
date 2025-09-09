"""
Utility functions for ModelAgentLite_LG and Agent System
"""

import re
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any

# Import for username resolution
try:
    from toolbox import SlackManager
    SLACK_MANAGER_AVAILABLE = True
except ImportError:
    SLACK_MANAGER_AVAILABLE = False

def extract_first_code_block(text: str) -> str:
    """
    Extract the first Python code block from LLM response.
    """
    # Look for ```python or ``` code blocks
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # Fallback: look for any code-like content
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if 'import ' in line or 'from ' in line or line.strip().startswith('def '):
            in_code = True
        
        if in_code:
            code_lines.append(line)
            
        # Stop if we hit a non-code line after starting
        if in_code and line.strip() and not any(keyword in line for keyword in ['import', 'from', 'def', '=', 'if', 'for', 'while', 'try', 'except', 'class', '#']):
            if not line.strip().startswith((' ', '\t')):
                break
    
    return '\n'.join(code_lines)

def safe_joblib_dump(obj, filepath: str) -> str:
    """
    Safely dump object using joblib with fallback mechanisms.
    """
    import joblib
    
    try:
        joblib.dump(obj, filepath)
        return filepath
    except (OSError, IOError) as e:
        # Fallback to temporary directory
        temp_dir = tempfile.gettempdir()
        filename = os.path.basename(filepath)
        fallback_path = os.path.join(temp_dir, filename)
        
        try:
            joblib.dump(obj, fallback_path)
            return fallback_path
        except Exception:
            # Final fallback with timestamp
            timestamp = int(time.time())
            final_path = os.path.join(temp_dir, f"model_{timestamp}.joblib")
            joblib.dump(obj, final_path)
            return final_path

def safe_plt_savefig(filepath: str, **kwargs) -> str:
    """
    Safely save matplotlib figure with fallback mechanisms.
    """
    import matplotlib.pyplot as plt
    
    try:
        plt.savefig(filepath, **kwargs)
        return filepath
    except (OSError, IOError) as e:
        # Fallback to temporary directory
        temp_dir = tempfile.gettempdir()
        filename = os.path.basename(filepath)
        fallback_path = os.path.join(temp_dir, filename)
        
        try:
            plt.savefig(fallback_path, **kwargs)
            return fallback_path
        except Exception:
            # Final fallback with timestamp
            timestamp = int(time.time())
            final_path = os.path.join(temp_dir, f"plot_{timestamp}.png")
            plt.savefig(final_path, **kwargs)
            return final_path

def diagnose_io_error(error_msg: str) -> Dict[str, List[str]]:
    """
    Diagnose I/O errors and provide actionable suggestions.
    """
    diagnosis = {
        "likely_causes": [],
        "suggested_actions": []
    }
    
    error_lower = error_msg.lower()
    
    if "errno 5" in error_lower or "input/output error" in error_lower:
        diagnosis["likely_causes"].extend([
            "Hardware I/O issue (disk, USB, network)",
            "File system corruption or disk errors",
            "Network connectivity issues for remote storage",
            "Insufficient disk space or permissions"
        ])
        
        diagnosis["suggested_actions"].extend([
            "Check available disk space (need >100MB free)",
            "Verify file permissions in current directory",
            "Try restarting the application",
            "Check system logs for hardware errors",
            "Use fallback temporary directory"
        ])
    
    elif "permission" in error_lower:
        diagnosis["likely_causes"].append("File permission restrictions")
        diagnosis["suggested_actions"].extend([
            "Check file/directory permissions",
            "Ensure write access to target directory",
            "Try using temporary directory as fallback"
        ])
    
    elif "no space" in error_lower or "disk full" in error_lower:
        diagnosis["likely_causes"].append("Insufficient disk space")
        diagnosis["suggested_actions"].extend([
            "Free up disk space",
            "Use alternative storage location",
            "Clean up temporary files"
        ])
    
    return diagnosis

# =============================================================================
# SLACK-SPECIFIC UTILITIES
# =============================================================================

def find_plot_files(directory: str = ".", pattern: str = "*.png") -> List[str]:
    """Find plot files in a directory"""
    import glob
    
    search_pattern = os.path.join(directory, pattern)
    plot_files = glob.glob(search_pattern)
    
    # Also check common plot file extensions
    for ext in ['*.jpg', '*.jpeg', '*.pdf', '*.svg']:
        if ext != pattern:
            additional_files = glob.glob(os.path.join(directory, ext))
            plot_files.extend(additional_files)
    
    return plot_files

def upload_plot_modern(file_path: str, title: str, channel: str, thread_ts: str, bot_token: str) -> bool:
    """Modern method to upload plots to Slack using files.upload API"""
    import requests
    
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ Plot file not found: {file_path}")
            return False
        
        with open(file_path, 'rb') as file_content:
            response = requests.post(
                'https://slack.com/api/files.upload',
                headers={'Authorization': f'Bearer {bot_token}'},
                data={
                    'channels': channel,
                    'thread_ts': thread_ts,
                    'title': title,
                    'initial_comment': f"📈 {title}"
                },
                files={'file': file_content}
            )
        
        result = response.json()
        if result.get('ok'):
            print(f"✅ Successfully uploaded plot: {title}")
            return True
        else:
            print(f"❌ Failed to upload plot: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"🔥 Error uploading plot to Slack: {e}")
        return False

def format_output_for_slack(result: Any) -> Tuple[str, List[Dict[str, str]]]:
    """
    Format execution results for Slack display and return files to upload.
    Returns (formatted_message, files_to_upload)
    """
    files_to_upload = []
    
    if isinstance(result, dict):
        message_parts = []
        
        # Handle model performance metrics
        if 'model_performance' in result:
            perf = result['model_performance']
            message_parts.append("🎯 *Model Performance:*")
            for metric, value in perf.items():
                if isinstance(value, (int, float)):
                    message_parts.append(f"• {metric}: {value:.4f}")
                else:
                    message_parts.append(f"• {metric}: {value}")
        
        # Handle model path
        if 'model_path' in result and result['model_path']:
            message_parts.append(f"💾 *Model saved:* `{result['model_path']}`")
        
        # Handle plot files
        if 'plot_path' in result and result['plot_path']:
            files_to_upload.append({
                "path": result['plot_path'],
                "title": "Generated Plot",
                "type": "plot"
            })
        
        # Handle multiple plots
        if 'plots' in result and isinstance(result['plots'], list):
            for i, plot_path in enumerate(result['plots']):
                files_to_upload.append({
                    "path": plot_path,
                    "title": f"Plot {i+1}",
                    "type": "plot"
                })
        
        # Handle summary
        if 'summary' in result:
            message_parts.append(f"📋 *Summary:* {result['summary']}")
        
        # Handle data insights
        if 'insights' in result:
            insights = result['insights']
            if isinstance(insights, list):
                message_parts.append("💡 *Key Insights:*")
                for insight in insights[:5]:  # Limit to top 5
                    message_parts.append(f"• {insight}")
            else:
                message_parts.append(f"💡 *Insights:* {insights}")
        
        # Handle feature importance
        if 'feature_importance' in result:
            importance = result['feature_importance']
            if isinstance(importance, dict):
                message_parts.append("🔍 *Top Features:*")
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feature, score in sorted_features[:5]:  # Top 5 features
                    message_parts.append(f"• {feature}: {score:.4f}")
        
        # Handle execution output
        if 'output' in result:
            output = result['output']
            output_str = str(output)
            if len(output_str) > 1000:
                message_parts.append(f"📄 *Output (truncated):*\n```{output_str[:1000]}...```")
            else:
                message_parts.append(f"📄 *Output:*\n```{output_str}```")
        
        # Handle errors
        if 'error' in result:
            message_parts.append(f"❌ *Error:* {result['error']}")
        
        formatted_message = '\n'.join(message_parts) if message_parts else "✅ Task completed successfully!"
        return formatted_message, files_to_upload
    
    else:
        # Handle string or other result types
        result_str = str(result)
        if len(result_str) > 1500:
            formatted_message = f"📄 *Result (truncated):*\n```{result_str[:1500]}...```"
        else:
            formatted_message = f"📄 *Result:*\n```{result_str}```"
        
        return formatted_message, files_to_upload

def validate_dataframe(data_frame) -> Tuple[bool, str]:
    """Validate a DataFrame for ML tasks"""
    import pandas as pd
    import numpy as np
    
    if not isinstance(data_frame, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if data_frame.empty:
        return False, "DataFrame is empty"
    
    if data_frame.shape[0] < 10:
        return False, "DataFrame has too few rows (minimum 10 required)"
    
    if data_frame.shape[1] < 2:
        return False, "DataFrame has too few columns (minimum 2 required)"
    
    # Check for all-null columns
    null_columns = data_frame.columns[data_frame.isnull().all()].tolist()
    if null_columns:
        return False, f"Columns with all null values: {null_columns}"
    
    # Memory usage check removed - no limits for now
    
    return True, "DataFrame is valid for ML tasks"

def detect_file_type(file_content: bytes, file_name: str) -> str:
    """Detect file type from content and filename"""
    file_extension = file_name.lower().split('.')[-1] if '.' in file_name else ''
    
    # Check by extension first
    if file_extension in ['csv']:
        return 'csv'
    elif file_extension in ['xlsx', 'xls']:
        return 'excel'
    elif file_extension in ['json']:
        return 'json'
    elif file_extension in ['tsv', 'txt']:
        return 'tsv'
    
    # Try to detect from content
    try:
        content_sample = file_content[:1000].decode('utf-8', errors='ignore')
        
        # Check for CSV patterns
        if ',' in content_sample and '\n' in content_sample:
            lines = content_sample.split('\n')[:3]
            if all(',' in line for line in lines if line.strip()):
                return 'csv'
        
        # Check for TSV patterns
        if '\t' in content_sample and '\n' in content_sample:
            return 'tsv'
        
        # Check for JSON patterns
        if content_sample.strip().startswith(('{', '[')):
            return 'json'
            
    except:
        pass
    
    # Default to CSV
    return 'csv'

def get_file_stats(data_frame) -> Dict[str, Any]:
    """Get comprehensive statistics about a DataFrame"""
    import pandas as pd
    import numpy as np
    
    stats = {
        'shape': data_frame.shape,
        'columns': data_frame.columns.tolist(),
        'dtypes': data_frame.dtypes.to_dict(),
        'memory_usage_mb': data_frame.memory_usage(deep=True).sum() / 1024 / 1024,
        'null_counts': data_frame.isnull().sum().to_dict(),
        'numeric_columns': data_frame.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data_frame.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': data_frame.select_dtypes(include=['datetime64']).columns.tolist()
    }
    
    # Add basic statistics for numeric columns
    if stats['numeric_columns']:
        numeric_stats = data_frame[stats['numeric_columns']].describe()
        stats['numeric_summary'] = numeric_stats.to_dict()
    
    # Add unique value counts for categorical columns
    if stats['categorical_columns']:
        categorical_stats = {}
        for col in stats['categorical_columns'][:5]:  # Limit to first 5 categorical columns
            categorical_stats[col] = {
                'unique_count': data_frame[col].nunique(),
                'top_values': data_frame[col].value_counts().head(3).to_dict()
            }
        stats['categorical_summary'] = categorical_stats
    
    return stats


# =============================================================================
# USERNAME UTILITIES
# =============================================================================

def get_username_for_user_id(user_id: str) -> str:
    """
    Get username for user_id, using Slack API if available.
    This is the centralized function used across the entire project.
    
    Args:
        user_id: The Slack user ID
        
    Returns:
        str: Username suitable for folder naming
    """
    try:
        if SLACK_MANAGER_AVAILABLE:
            # Try to get the global slack_manager instance
            from toolbox import slack_manager
            if slack_manager:
                return slack_manager.get_username_from_user_id(user_id)
    except:
        pass
    
    # Fallback to sanitized user_id
    return sanitize_for_folder_name(user_id)


def sanitize_for_folder_name(name: str) -> str:
    """
    Sanitize a name to be safe for use as a folder name.
    Removes or replaces characters that are not allowed in folder names.
    
    Args:
        name: The name to sanitize
        
    Returns:
        str: Sanitized name safe for folder use
    """
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    sanitized = re.sub(r'[^\w\-_.]', '_', sanitized)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure it's not empty and not too long
    if not sanitized:
        sanitized = "unknown_user"
    elif len(sanitized) > 50:
        sanitized = sanitized[:50]
    
    return sanitized


def get_username_session_id(session_id: str) -> str:
    """
    Convert session_id to use username instead of user_id.
    Handles both user_id_thread_ts and single user_id formats.
    
    Args:
        session_id: Session ID in format user_id_thread_ts or just user_id
        
    Returns:
        str: Session ID with username instead of user_id
    """
    if "_" in session_id:
        user_id_part, thread_ts = session_id.split("_", 1)
        username = get_username_for_user_id(user_id_part)
        return f"{username}_{thread_ts}"
    else:
        # Single user_id case
        username = get_username_for_user_id(session_id)
        return username


def get_username_artifacts_dir(user_id: str, thread_id: str = None) -> str:
    """
    Get artifacts directory path using username instead of user_id.
    
    Args:
        user_id: The Slack user ID
        thread_id: Optional thread ID
        
    Returns:
        str: Artifacts directory path with username
    """
    username = get_username_for_user_id(user_id)
    if thread_id:
        return f"user_data/{username}/{thread_id}/artifacts"
    else:
        return f"user_data/{username}/artifacts"


def get_username_user_data_dir(user_id: str, thread_id: str = None) -> str:
    """
    Get user data directory path using username instead of user_id.
    
    Args:
        user_id: The Slack user ID
        thread_id: Optional thread ID
        
    Returns:
        str: User data directory path with username
    """
    username = get_username_for_user_id(user_id)
    if thread_id:
        return f"user_data/{username}/{thread_id}"
    else:
        return f"user_data/{username}" 