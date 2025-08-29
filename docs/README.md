# Multi-Agent ML Integration System

A comprehensive machine learning pipeline system built with LangGraph that integrates preprocessing, feature selection, and model building agents into a unified, intelligent workflow.

## 🏗️ Architecture

The system consists of several key components:

### Core Components

1. **PipelineState** - Global state management with persistence
2. **Orchestrator** - Intelligent query routing and workflow coordination  
3. **Agents** - Specialized ML agents for different tasks
4. **Toolbox** - Shared utilities (Slack, artifacts, execution, progress tracking)
5. **LangGraph Pipeline** - Main orchestration system
6. **Slack Bot** - User interface layer

### Agent Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ PreprocessingAgent │    │ FeatureSelectionAgent │    │ ModelBuildingAgent │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Data cleaning │    │ • IV analysis    │    │ • Model training│
│ • Missing values│    │ • Correlation    │    │ • Evaluation    │
│ • Outliers      │    │ • VIF analysis   │    │ • Predictions   │
│ • Encoding      │    │ • PCA            │    │ • Visualization │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Flow Examples

**Full Pipeline:**
```
Query: "Train XGBoost on this CSV"
Flow: Orchestrator → PreprocessingAgent → FeatureSelectionAgent → ModelBuildingAgent
```

**Direct Entry:**
```
Query: "Select features using IV > 0.02"
Flow: Orchestrator → FeatureSelectionAgent only
```

**Resume Session:**
```
Query: "Use cleaned data from last session and run PCA"
Flow: Orchestrator → (loads state) → FeatureSelectionAgent
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or create the project directory
cd MAL_Integration

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file with your configuration:

```bash
# Slack Configuration (required for Slack integration)
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token

# Model Configuration
DEFAULT_MODEL=qwen2.5-coder:32b-instruct-q4_K_M
FALLBACK_MODEL_2=deepseek-coder-v2:latest

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI Configuration (optional fallback)
OPENAI_API_KEY=your-openai-key

# Pipeline Configuration
ENABLE_PERSISTENCE=true
SESSION_TIMEOUT_HOURS=24
MAX_FILE_SIZE_MB=100
```

### 3. Basic Usage

#### Python API

```python
from langgraph_pipeline import initialize_pipeline
import pandas as pd

# Initialize the pipeline
pipeline = initialize_pipeline(enable_persistence=True)

# Load sample data
data = pd.read_csv("your_data.csv")
session_id = "my_session"
pipeline.load_data(data, session_id)

# Process queries
result = pipeline.process_query("Clean and preprocess this data", session_id)
print(result["response"])

result = pipeline.process_query("Select the best features", session_id)
print(result["response"])

result = pipeline.process_query("Train a machine learning model", session_id)
print(result["response"])
```

#### Slack Bot

```python
from slack_bot import SlackMLBot

# Start the Slack bot
bot = SlackMLBot()
bot.start()
```

## 📊 Features

### Intelligent Routing
- **Keyword-based routing** - Automatically routes queries to appropriate agents
- **Context-aware decisions** - Uses current pipeline state to determine next steps
- **Flexible entry points** - Can start at any stage of the pipeline

### Session Management
- **Persistent state** - Sessions survive restarts and can be resumed
- **Multi-user support** - Isolated sessions per user/thread
- **Artifact storage** - All generated files stored per session

### Slack Integration
- **File upload support** - CSV, Excel, JSON, TSV files
- **Real-time progress updates** - Live feedback during processing
- **Thread-based conversations** - Organized discussion per dataset
- **Rich formatting** - Tables, metrics, and visualizations

### Fallback Mechanism
- **LLM-powered error recovery** - Automatic code fixing using multiple models
- **Tiered fallback approach** - Multiple models for different perspectives
- **Surgical error fixing** - Targeted fixes for specific error types

## 🔧 Configuration

### Model Configuration

The system supports multiple LLM providers:

- **Ollama** (local models) - Primary choice for privacy and control
- **OpenAI** (cloud models) - Fallback option for complex tasks

### File Support

Supported data formats:
- CSV files (.csv)
- Excel files (.xlsx, .xls)
- JSON files (.json)
- Tab-separated files (.tsv)

### Persistence

The system provides multiple persistence layers:
- **State persistence** - Pipeline states saved to disk
- **Session management** - User sessions with timeout
- **Artifact storage** - Generated files organized by session
- **LangGraph checkpoints** - Workflow state for resumption

## 🎯 Usage Examples

### Data Preprocessing
```
"Clean my dataset and handle missing values"
"Remove outliers and normalize the data"
"Encode categorical variables"
```

### Feature Selection
```
"Select features with IV > 0.02"
"Run correlation analysis and remove multicollinear features"
"Apply PCA for dimensionality reduction"
```

### Model Building
```
"Train a LightGBM model"
"Build a random forest classifier"
"Create customer segments using clustering"
"Evaluate model performance"
```

### Pipeline Operations
```
"Build a complete ML pipeline from this data"
"Resume my analysis from yesterday"
"Show me the current pipeline status"
```

## 🏗️ Architecture Details

### State Management

The `PipelineState` class maintains:
- **Data states** - raw_data, cleaned_data, selected_features, trained_model
- **Session info** - chat_session, artifacts, progress
- **Execution context** - user_query, last_code, last_error
- **Agent states** - preprocessing_state, feature_selection_state, model_building_state

### Agent Integration

Each agent wraps existing implementations:
- **PreprocessingAgent** - Integrates with DataPreprocessingAgent
- **FeatureSelectionAgent** - Integrates with FeatureSelection system
- **ModelBuildingAgent** - Integrates with ModelAgentLite_LG

### Orchestrator Logic

The orchestrator uses multiple routing strategies:
1. **Keyword matching** - Scores queries against agent vocabularies
2. **Pattern recognition** - Identifies full pipeline vs. single-step requests
3. **State-driven routing** - Routes based on available data
4. **Resume detection** - Handles session continuation requests

## 🔍 Monitoring and Debugging

### Progress Tracking
- Real-time progress updates in Slack
- Console logging with timestamps
- Execution history in pipeline state

### Error Handling
- Comprehensive error capture and reporting
- LLM-powered error analysis and fixing
- Graceful degradation with fallback options

### Session Management
- Session status commands (`/pipeline_status`)
- Automatic cleanup of old sessions
- Session data persistence and recovery

## 🚀 Deployment

### Local Development
```bash
# Start Ollama (if using local models)
ollama serve

# Start the Slack bot
python slack_bot.py
```

### Production Deployment
- Set up environment variables
- Configure persistent storage paths
- Set up monitoring and logging
- Configure model endpoints

## 🤝 Integration Points

The system integrates with existing agent implementations:

1. **DataPreprocessingAgent** (`/Users/10321/Vishwas/CV/GenAI/CursorProjects/DataPreprocessingAgent`)
2. **FeatureSelectionAgent** (`/Users/10321/Vishwas/CV/GenAI/CursorProjects/FeatureSelcetion`)
3. **ModelBuildingAgent** (`/Users/10321/Vishwas/CV/GenAI/CursorProjects/ModelAgentLite_LG`)

Each agent maintains its own specialized logic while participating in the unified pipeline.

## 📈 Extensibility

The system is designed for easy extension:

### Adding New Agents
1. Create agent class inheriting from `BaseAgent`
2. Implement the `run(state: PipelineState)` method
3. Add routing keywords to orchestrator
4. Register agent in LangGraph pipeline

### Custom Routing Logic
- Extend `Orchestrator` class with new routing methods
- Add custom keyword patterns
- Implement domain-specific routing rules

### Additional Integrations
- Add new file format support
- Integrate with other messaging platforms
- Connect to different model providers
- Add custom artifact types

## 🔒 Security and Privacy

- **Local model support** - Keep data processing on-premises
- **Session isolation** - User data separated by session
- **Configurable storage** - Control where data is stored
- **Token-based authentication** - Secure Slack integration

## 📚 API Reference

See individual module documentation for detailed API information:
- `pipeline_state.py` - State management
- `orchestrator.py` - Query routing
- `agents.py` - Agent implementations
- `toolbox.py` - Shared utilities
- `langgraph_pipeline.py` - Main pipeline
- `slack_bot.py` - Slack integration
