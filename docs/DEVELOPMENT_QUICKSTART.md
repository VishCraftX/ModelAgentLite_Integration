# Development Quick Start Guide

## 🚀 **Immediate Setup (5 minutes)**

```bash
# 1. Clone and setup
git clone <repository>
cd MAL_Integration
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# 2. Environment variables (create .env file)
SLACK_BOT_TOKEN=xoxb-your-token-here
SLACK_APP_TOKEN=xapp-your-token-here
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=qwen2.5-coder:32b-instruct-q4_K_M

# 3. Start Ollama (separate terminal)
ollama serve
ollama pull qwen2.5-coder:32b-instruct-q4_K_M

# 4. Run system
python start_pipeline.py --mode slack --bot-token <token> --app-token <token>
```

## 🎯 **Core Files to Understand**

| Priority | File | Purpose | Modify? |
|----------|------|---------|---------|
| 🔴 **CRITICAL** | `pipeline_state.py` | Core state schema | ❌ NO |
| 🔴 **CRITICAL** | `langgraph_pipeline.py` | Main orchestrator | ⚠️ CAREFUL |
| 🟡 **IMPORTANT** | `orchestrator.py` | Intent classification | ✅ EXTEND |
| 🟡 **IMPORTANT** | `agents_wrapper.py` | Agent interfaces | ✅ ADD NEW |
| 🟢 **SAFE** | `*_agent_impl.py` | Agent implementations | ✅ ENHANCE |
| 🟢 **SAFE** | `toolbox.py` | Utilities | ✅ ADD UTILS |

## 🔒 **DO NOT TOUCH (Will Break Everything)**

```python
# PipelineState core fields
raw_data: Optional[pd.DataFrame] = None
cleaned_data: Optional[pd.DataFrame] = None
processed_data: Optional[pd.DataFrame] = None
target_column: Optional[str] = None
session_id: Optional[str] = None
chat_session: Optional[str] = None

# LangGraph node names
"orchestrator_node"
"preprocessing_node" 
"feature_selection_node"
"model_building_node"

# Agent wrapper interface
def run(self, state: PipelineState) -> PipelineState:
```

## ✅ **Safe to Modify/Add**

```python
# Add new state fields (APPEND ONLY)
class PipelineState(BaseModel):
    # ... existing fields ...
    your_new_field: Optional[Dict] = Field(default_factory=dict)

# Add new agent wrapper
class YourAgentWrapper:
    def run(self, state: PipelineState) -> PipelineState:
        # Your implementation
        return state

# Add new utility functions
def your_utility_function(data, params):
    # Your implementation
    return result
```

## 🧪 **Testing Your Changes**

```bash
# 1. Test locally
python start_pipeline.py --mode slack --bot-token <token> --app-token <token>

# 2. Test in Slack
# Upload a CSV file
# Send: "@bot_name clean my data"
# Follow interactive prompts

# 3. Check logs for errors
# Look for session registration
# Verify state persistence
# Check Slack message delivery
```

## 🐛 **Common Issues & Fixes**

| Issue | Symptom | Fix |
|-------|---------|-----|
| Import Error | `ModuleNotFoundError` | Check `requirements.txt`, install missing packages |
| Slack Not Responding | No bot response | Verify tokens, check bot permissions |
| State Error | `'dict' object has no attribute` | Check PipelineState field names |
| Menu Missing | No interactive menu in Slack | Debug session registration |

## 📝 **Development Patterns**

### **Adding New Analysis Function**
```python
# 1. In agent implementation
def new_analysis(data: pd.DataFrame, params: Dict) -> Dict:
    """Your new analysis function"""
    result = your_analysis_logic(data, params)
    return {"analysis_type": "new", "results": result}

# 2. In agent wrapper
def run(self, state: PipelineState) -> PipelineState:
    if "new analysis" in state.user_query.lower():
        result = new_analysis(state.raw_data, {})
        state.preprocessing_state["new_analysis"] = result
    return state
```

### **Adding Interactive Command**
```python
# In agent implementation
def handle_user_command(self, query: str, session_data: Dict, say_function):
    if "new command" in query.lower():
        result = process_new_command(session_data)
        say_function(f"✅ New command executed: {result}")
        return True
    return False
```

## 🎯 **Suggested First Tasks**

### **Easy (1-2 hours)**
1. **Add new keywords** to orchestrator classification
2. **Enhance error messages** with more helpful text
3. **Add logging** to existing functions
4. **Fix typos** in user-facing messages

### **Medium (1-2 days)**
1. **Add new analysis function** to existing agent
2. **Create data visualization** for results
3. **Enhance interactive menus** with more options
4. **Add export functionality** for results

### **Advanced (1 week+)**
1. **Create new agent** following wrapper pattern
2. **Add batch processing** for multiple files
3. **Implement model deployment** features
4. **Add advanced ML algorithms**

## 🔍 **Debugging Tips**

```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def your_function():
    logger.debug(f"Processing data: {data.shape}")
    # Your code
    logger.debug(f"Result: {result}")

# Check state at any point
def debug_state(state: PipelineState):
    print(f"Session: {state.session_id}")
    print(f"Data: {state.raw_data.shape if state.raw_data is not None else 'None'}")
    print(f"Interactive: {state.interactive_session}")
```

## 📚 **Key Concepts**

### **State Flow**
```
User Input → Slack Bot → Pipeline → Orchestrator → Agent → State Update → Persistence → Response
```

### **Agent Pattern**
```
Wrapper (Interface) → Implementation (Logic) → State (Results)
```

### **Session Management**
```
user_data/{user_id}/{thread_id}/
├── conversation_history.json
├── session_state.json
└── artifacts/
```

---

**Ready to contribute? Pick a task, create a branch, and start coding! The architecture is designed to be extensible while maintaining stability.** 🚀
