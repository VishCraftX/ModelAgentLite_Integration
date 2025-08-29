# Multi-Agent ML Integration System - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### 1. Install Dependencies
```bash
# Install all dependencies
pip install -r requirements_complete.txt

# Or minimal installation
pip install pandas numpy scikit-learn pydantic
```

### 2. Quick Launch Options

#### Option A: Run Demo (Recommended for first time)
```bash
python run.py demo
```
This will run an automated demo showing the complete pipeline in action.

#### Option B: Start Slack Bot
```bash
python run.py slack
```
Starts the Slack bot for interactive file upload and natural language queries.

#### Option C: Interactive API Testing
```bash
python run.py api
```
Interactive Python API testing with sample data.

#### Option D: Full Launcher
```bash
python start_pipeline.py
```
Full interactive launcher with all options.

### 3. For Slack Integration (Optional)

Set environment variables:
```bash
export SLACK_BOT_TOKEN=xoxb-your-bot-token
export SLACK_APP_TOKEN=xapp-your-app-token
```

Or create a `.env` file:
```
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
DEFAULT_MODEL=qwen2.5-coder:32b-instruct-q4_K_M
ENABLE_PERSISTENCE=true
```

## 🎯 Usage Examples

### Python API Usage
```python
from langgraph_pipeline import initialize_pipeline
import pandas as pd

# Initialize pipeline
pipeline = initialize_pipeline()

# Load your data
data = pd.read_csv("your_data.csv")
pipeline.load_data(data, "my_session")

# Run queries
result = pipeline.process_query("Clean and preprocess this data", "my_session")
print(result["response"])

result = pipeline.process_query("Select the best features", "my_session")
print(result["response"])

result = pipeline.process_query("Train a machine learning model", "my_session")
print(result["response"])
```

### Slack Bot Usage
1. Upload a CSV/Excel file to Slack
2. Ask: "Build a complete ML pipeline"
3. Or ask specific questions:
   - "Clean and preprocess this data"
   - "Select the most important features"
   - "Train a random forest model"

### Direct Agent Usage
```python
from agents_integrated import preprocessing_agent, feature_selection_agent, model_building_agent
from pipeline_state import PipelineState
import pandas as pd

# Create state
state = PipelineState(
    raw_data=pd.read_csv("data.csv"),
    session_id="direct_usage"
)

# Run agents
state = preprocessing_agent.run(state)
state = feature_selection_agent.run(state)
state = model_building_agent.run(state)

print(f"Pipeline complete: {state.get_data_summary()}")
```

## 🔧 Command Line Options

### Full Launcher
```bash
python start_pipeline.py [OPTIONS]

Options:
  --bot-token TOKEN     Slack Bot Token
  --app-token TOKEN     Slack App Token  
  --mode MODE          Run mode: slack, api, test, demo
  --model MODEL        Default LLM model
  --persistence        Enable state persistence
```

### Quick Launcher
```bash
python run.py [MODE]

Modes:
  demo    - Automated demo
  slack   - Slack bot
  api     - API testing
  test    - Direct tests
```

## 📊 What Each Mode Does

### 🎬 Demo Mode
- Creates realistic sample dataset (200 rows, 7 features)
- Runs complete pipeline: Preprocessing → Feature Selection → Model Building
- Shows progress and results for each step
- Perfect for understanding system capabilities

### 🤖 Slack Bot Mode
- Starts Slack bot with file upload support
- Supports CSV, Excel, JSON, TSV files
- Natural language query processing
- Real-time progress updates
- Session-based conversations

### 🧪 API Testing Mode
- Interactive Python API testing
- Sample data provided
- Type queries and see results immediately
- Shows pipeline status
- Great for development and testing

### 🔧 Direct Testing Mode
- Tests individual agents directly
- Verifies each component works
- Useful for debugging and development

## 🎉 Expected Output

### Demo Mode Output
```
🚀 Multi-Agent ML Integration System
============================================================
✅ Core Dependencies: Available
🤖 Default model: qwen2.5-coder:32b-instruct-q4_K_M
💾 Persistence: Enabled

🎬 Running Automated Demo...
✅ Pipeline initialized
📊 Demo data loaded: (200, 7)

==================================================
STEP 1: Data Preprocessing
Query: 'Clean and preprocess this data'
==================================================
✅ Success: True
📝 Response: ✅ Data preprocessing completed (200 rows × 7 columns)

==================================================
STEP 2: Feature Selection  
Query: 'Select the most important features'
==================================================
✅ Success: True
📝 Response: ✅ Feature selection completed (6 features selected)

==================================================
STEP 3: Model Building
Query: 'Train a machine learning model for classification'
==================================================
✅ Success: True
📝 Response: ✅ Model training completed

🎉 Demo completed successfully!
```

## 🆘 Troubleshooting

### Missing Dependencies
```bash
❌ Missing Required Dependencies:
   • Core ML libraries: No module named 'pandas'

💡 Install with: pip install -r requirements_complete.txt
```

### Slack Token Issues
```bash
❌ Missing Slack tokens. Please restart and provide valid tokens.

💡 Set environment variables or use command line:
python start_pipeline.py --bot-token xoxb-... --app-token xapp-...
```

### Import Errors
```bash
⚠️ Optional Dependencies:
   • LangGraph: No module named 'langgraph'
   • Slack integration: No module named 'slack_bolt'

💡 Some features may use fallback implementations
```

## 🎯 Next Steps

1. **Start with Demo**: `python run.py demo`
2. **Try API Testing**: `python run.py api`
3. **Set up Slack Bot**: Configure tokens and run `python run.py slack`
4. **Explore Code**: Check `ARCHITECTURE_DOCUMENTATION.md` for details
5. **Customize**: Modify agents or add new functionality

The system is designed to work with graceful degradation - even without all dependencies, core functionality will work using fallback implementations!
