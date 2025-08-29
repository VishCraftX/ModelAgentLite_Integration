# How to Run the Multi-Agent ML Integration System

## 🚀 Quick Start (Recommended)

### 1. **Run Demo** (Best for first time)
```bash
python run.py demo
```
**What it does:**
- Creates realistic sample dataset (200 rows, 7 features)
- Runs complete pipeline: Preprocessing → Feature Selection → Model Building
- Shows progress and results for each step
- Perfect for understanding system capabilities

### 2. **Interactive API Testing**
```bash
python run.py api
```
**What it does:**
- Starts interactive Python API testing
- Provides sample data automatically
- Type queries and see results immediately
- Shows pipeline status with `status` command
- Type `quit` to exit

### 3. **Start Slack Bot**
```bash
python run.py slack
```
**What it does:**
- Starts Slack bot with file upload support
- Supports CSV, Excel, JSON, TSV files
- Natural language query processing
- Real-time progress updates
- Session-based conversations

### 4. **Direct Agent Testing**
```bash
python run.py test
```
**What it does:**
- Tests individual agents directly
- Verifies each component works
- Useful for debugging and development

## 🔧 Advanced Usage

### Full Interactive Launcher
```bash
python start_pipeline.py
```
**Features:**
- Interactive menu with all options
- Dependency checking and status
- Environment setup
- Token configuration for Slack

### Command Line Options
```bash
python start_pipeline.py [OPTIONS]

Options:
  --bot-token TOKEN     Slack Bot Token (xoxb-...)
  --app-token TOKEN     Slack App Token (xapp-...)
  --mode MODE          Run mode: slack, api, test, demo
  --model MODEL        Default LLM model
  --persistence        Enable state persistence

Examples:
  python start_pipeline.py --mode demo
  python start_pipeline.py --mode slack --bot-token xoxb-... --app-token xapp-...
  python start_pipeline.py --mode api --model gpt-4o
```

## 📊 Expected Output Examples

### Demo Mode Success
```
🚀 Multi-Agent ML Integration System
============================================================
✅ Core Dependencies: Available
🤖 Default model: qwen2.5-coder:32b-instruct-q4_K_M
💾 Persistence: Enabled

🎬 Running Automated Demo...
✅ Pipeline initialized
📊 Demo data loaded: (200, 7)
   Missing values: 10
   Target distribution: {0: 146, 1: 54}

==================================================
STEP 1: Data Preprocessing
Query: 'Clean and preprocess this data'
==================================================
✅ Success: True
📝 Response: ✅ Data preprocessing completed (200 rows × 7 columns)
📊 Pipeline Progress: ✅ → ✅ → ❌ → ❌

==================================================
STEP 2: Feature Selection
Query: 'Select the most important features'
==================================================
✅ Success: True
📝 Response: ✅ Feature selection completed (7 features selected)
📊 Pipeline Progress: ✅ → ✅ → ✅ → ❌

==================================================
STEP 3: Model Building
Query: 'Train a machine learning model for classification'
==================================================
✅ Success: True
📝 Response: ✅ Model training completed
📊 Pipeline Progress: ✅ → ✅ → ✅ → ✅

🎉 Demo completed successfully!
💡 The system is ready for production use!
```

### API Testing Mode
```
🧪 Starting Interactive API Testing...
✅ Pipeline initialized
📊 Creating sample dataset...
✅ Sample data loaded: (100, 5)

🎯 Interactive Testing Mode
Available commands:
  • 'preprocess' or 'clean data'
  • 'select features' or 'feature selection'
  • 'train model' or 'build model'
  • 'status' - show current pipeline status
  • 'quit' or 'exit' - exit testing

🤖 Enter your query: preprocess data
🔄 Processing: 'preprocess data'
✅ Success: True
📝 Response: ✅ Data preprocessing completed (100 rows × 5 columns)

🤖 Enter your query: status
📊 Pipeline Status:
   Raw Data: ✅
   Cleaned Data: ✅
   Selected Features: ❌
   Trained Model: ❌

🤖 Enter your query: select features
🔄 Processing: 'select features'
✅ Success: True
📝 Response: ✅ Feature selection completed (5 features selected)
```

### Slack Bot Mode
```
🤖 Starting Multi-Agent ML Slack Bot...
Features:
  • File upload support (CSV, Excel, JSON, TSV)
  • Intelligent query routing
  • Real-time progress updates
  • Session-based conversations
  • Artifact management

🔑 Slack Bot Token Required
Enter SLACK_BOT_TOKEN (or press Enter to skip): xoxb-your-token

🔑 Slack App Token Required
Enter SLACK_APP_TOKEN (or press Enter to skip): xapp-your-token

✅ Slack bot initialized successfully
🚀 Starting bot... (Press Ctrl+C to stop)
✅ Bot started! Upload data files and send messages in Slack to get started.
```

## 🔍 System Status Indicators

### ✅ **Fully Functional**
```
✅ Core Dependencies: Available
🤖 Default model: qwen2.5-coder:32b-instruct-q4_K_M
💾 Persistence: Enabled
```

### ⚠️ **Fallback Mode** (Still Works!)
```
✅ Core Dependencies: Available

⚠️ Optional Dependencies:
   • LangGraph: No module named 'langgraph'
   • Slack integration: No module named 'slack_bolt'
   • Ollama: Not available (local LLM support)
   • OpenAI: Not available (cloud LLM support)

💡 Some features may use fallback implementations
```

### ❌ **Missing Core Dependencies**
```
❌ Missing Required Dependencies:
   • Core ML libraries: No module named 'pandas'

💡 Install with: pip install -r requirements_complete.txt
```

## 🛠️ Installation Options

### Complete Installation (Recommended)
```bash
pip install -r requirements_complete.txt
```
**Includes:** All features, real agent implementations, Slack integration, LLM support

### Minimal Installation
```bash
pip install pandas numpy scikit-learn pydantic
```
**Includes:** Core functionality with fallback implementations

### Slack Integration
```bash
pip install slack-bolt slack-sdk
```
**Adds:** Slack bot functionality

### LangGraph Support
```bash
pip install langgraph langchain langchain-core
```
**Adds:** Full pipeline orchestration

## 🎯 Usage Patterns

### 1. **First Time Users**
```bash
# Start here
python run.py demo

# Then try interactive testing
python run.py api
```

### 2. **Development/Testing**
```bash
# Test individual components
python run.py test

# Interactive development
python run.py api
```

### 3. **Production Slack Bot**
```bash
# Set up environment
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_APP_TOKEN=xapp-...

# Start bot
python run.py slack
```

### 4. **Python Integration**
```python
from langgraph_pipeline import initialize_pipeline

pipeline = initialize_pipeline()
# Use in your code...
```

## 🆘 Troubleshooting

### Common Issues and Solutions

**Issue:** `No module named 'pandas'`
**Solution:** `pip install pandas numpy scikit-learn`

**Issue:** `Missing Slack tokens`
**Solution:** Set environment variables or use command line flags

**Issue:** `LangGraph not available, using simplified pipeline`
**Solution:** This is normal! The system works with fallback implementations

**Issue:** `Import error: No module named 'langchain_core'`
**Solution:** This is expected - the system uses fallback implementations

## 🎉 Success Indicators

You know the system is working when you see:
- ✅ **Success: True** in responses
- 📊 **Pipeline Progress** showing ✅ → ✅ → ✅ → ✅
- 🎉 **Demo completed successfully!**
- Artifacts being saved (you'll see "Saved artifact for session_id: filename")

The system is designed to work even with missing dependencies using intelligent fallbacks!
