# Multi-Agent ML Integration System - Setup Guide

## 🚀 Quick Installation

### 1. Install All Dependencies

```bash
# Install complete dependencies for full functionality
pip install -r requirements_complete.txt
```

### 2. Alternative: Minimal Installation

```bash
# Core functionality only
pip install pandas numpy scikit-learn pydantic

# Add LangGraph support
pip install langgraph langchain langchain-core

# Add Slack integration (optional)
pip install slack-bolt slack-sdk

# Add LLM support (optional)
pip install ollama openai langchain-openai langchain-ollama
```

## 🔧 Environment Configuration

Create a `.env` file in the project root:

```bash
# Slack Integration (required for Slack bot)
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here

# Model Configuration
DEFAULT_MODEL=qwen2.5-coder:32b-instruct-q4_K_M
FALLBACK_MODEL_1=qwen2.5-coder:32b-instruct-q4_K_M
FALLBACK_MODEL_2=deepseek-coder-v2:latest

# Ollama Configuration (for local models)
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI Configuration (optional fallback)
OPENAI_API_KEY=your-openai-api-key-here

# Pipeline Configuration
ENABLE_PERSISTENCE=true
SESSION_TIMEOUT_HOURS=24
MAX_FILE_SIZE_MB=100
```

## 📋 Dependency Breakdown

### Core Dependencies (Always Required)
- `pandas>=2.1.4` - Data manipulation
- `numpy>=1.24.3` - Numerical computing
- `scikit-learn>=1.3.2` - Machine learning
- `pydantic>=2.0.0` - Data validation

### LangGraph Dependencies (For Full Pipeline)
- `langgraph>=0.1.15` - Graph orchestration
- `langchain>=0.1.0` - LLM framework
- `langchain-core>=0.1.0` - Core LangChain
- `langchain-openai>=0.1.0` - OpenAI integration
- `langchain-ollama>=0.1.0` - Ollama integration

### Slack Dependencies (For Bot Interface)
- `slack-bolt>=1.18.1` - Slack app framework
- `slack-sdk>=3.28.0` - Slack API client

### ML Libraries (For Advanced Models)
- `lightgbm>=4.1.0` - Gradient boosting
- `xgboost>=2.0.3` - Extreme gradient boosting
- `catboost>=1.2.0` - Categorical boosting

### LLM Providers (Choose One or Both)
- `ollama>=0.3.3` - Local LLM support
- `openai>=1.12.0` - OpenAI API support

## 🎯 Usage Examples

### 1. Python API Usage

```python
from langgraph_pipeline import initialize_pipeline
import pandas as pd

# Initialize pipeline
pipeline = initialize_pipeline(enable_persistence=True)

# Load your data
data = pd.read_csv("your_dataset.csv")
session_id = "my_analysis_session"
pipeline.load_data(data, session_id)

# Run queries
result = pipeline.process_query("Clean and preprocess this data", session_id)
print(result["response"])

result = pipeline.process_query("Select the best features", session_id)
print(result["response"])

result = pipeline.process_query("Train a machine learning model", session_id)
print(result["response"])
```

### 2. Slack Bot Usage

```python
from slack_bot import SlackMLBot

# Start the Slack bot
bot = SlackMLBot()
bot.start()
```

Then in Slack:
1. Upload a CSV/Excel file
2. Ask: "Clean and preprocess this data"
3. Ask: "Select the best features for modeling"
4. Ask: "Train a LightGBM model"

### 3. Direct Agent Usage

```python
from agents_integrated import preprocessing_agent, feature_selection_agent, model_building_agent
from pipeline_state import PipelineState
import pandas as pd

# Create state
state = PipelineState(
    raw_data=pd.read_csv("data.csv"),
    session_id="direct_usage",
    user_query="preprocess data"
)

# Run agents directly
state = preprocessing_agent.run(state)
state = feature_selection_agent.run(state)
state = model_building_agent.run(state)

print(f"Final state: {state.get_data_summary()}")
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'langchain_core'
   ```
   **Solution**: Install missing dependencies
   ```bash
   pip install langchain langchain-core langchain-openai
   ```

2. **Slack Integration Issues**
   ```
   ModuleNotFoundError: No module named 'slack_bolt'
   ```
   **Solution**: Install Slack dependencies
   ```bash
   pip install slack-bolt slack-sdk
   ```

3. **LLM Connection Issues**
   ```
   Error: No model endpoint available
   ```
   **Solution**: 
   - For Ollama: Start Ollama server (`ollama serve`)
   - For OpenAI: Set `OPENAI_API_KEY` environment variable

### Graceful Degradation

The system is designed to work even with missing dependencies:

- **Without LangGraph**: Uses simplified pipeline mode
- **Without Slack**: Core functionality still works
- **Without LLM libraries**: Uses basic implementations
- **Without advanced ML libraries**: Falls back to scikit-learn

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent ML Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│  Slack Bot  │  Python API  │  Direct Agent Usage           │
├─────────────────────────────────────────────────────────────┤
│                   LangGraph Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  Orchestrator  │  State Manager  │  Progress Tracker       │
├─────────────────────────────────────────────────────────────┤
│ Preprocessing │ Feature Selection │ Model Building Agent    │
│    Agent      │      Agent        │                         │
├─────────────────────────────────────────────────────────────┤
│        Shared Toolbox (Slack, Artifacts, Execution)        │
├─────────────────────────────────────────────────────────────┤
│  Existing Agent Implementations (Integrated)               │
│  • DataPreprocessingAgent                                   │
│  • FeatureSelectionAgent                                    │
│  • ModelBuildingAgent                                       │
└─────────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
MAL_Integration/
├── agents_integrated.py           # Integrated agent implementations
├── langgraph_pipeline.py         # Main pipeline orchestration
├── pipeline_state.py             # Global state management
├── orchestrator.py               # Intelligent routing
├── toolbox.py                    # Shared utilities
├── slack_bot.py                  # Slack bot interface
├── config.py                     # Configuration management
├── requirements_complete.txt     # All dependencies
├── preprocessing_agent_impl.py   # Copied preprocessing implementation
├── feature_selection_agent_impl.py # Copied feature selection implementation
├── model_building_agent_impl.py # Copied model building implementation
├── model_agent_utils.py          # Model agent utilities
├── example_usage.py              # Usage examples
├── README.md                     # Documentation
├── SETUP_GUIDE.md               # This file
└── IMPLEMENTATION_SUMMARY.md    # Implementation details
```

## 🎉 Ready to Use!

The system is now fully integrated with the actual agent implementations and ready for production use. Choose your installation level based on your needs:

- **Minimal**: Core ML pipeline functionality
- **Standard**: Full pipeline with LangGraph orchestration
- **Complete**: All features including Slack integration and advanced models

Start with the minimal installation and add components as needed!
