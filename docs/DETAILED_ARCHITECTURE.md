# Detailed Architecture Documentation

## 🏗️ **System Overview**

The Multi-Agent ML Integration System is built with a **layered architecture** that orchestrates three independent ML agents through a unified interface while preserving their original functionality.

## 📋 **Architecture Layers**

### **1. Entry Points Layer**
- **`start_pipeline.py`** - Main startup script with multiple modes
  - **Slack Bot Mode**: Launches interactive Slack bot
  - **API Mode**: Provides programmatic interface
  - **Test Mode**: Runs direct agent tests
  - **Demo Mode**: Demonstration workflows

### **2. Core Pipeline Layer**
- **`langgraph_pipeline.py`** - Central orchestration engine
  - **`MultiAgentMLPipeline`** class manages the entire workflow
  - **LangGraph integration** for stateful agent workflows
  - **Session management** with persistence
  - **Query processing** and response generation
  - **State transitions** between agents

### **3. Orchestrator Layer**
- **`orchestrator.py`** - Hybrid intent classification system
  - **Keyword-based classification** (primary, fast)
  - **LLM-powered fallback** (when keyword confidence is low)
  - **Intent categories**:
    - `preprocessing` - Data cleaning and preparation
    - `feature_selection` - Feature engineering and selection
    - `model_building` - Model training and evaluation
    - `general_response` - Conversational queries
    - `code_execution` - Code generation and execution

### **4. Agent Wrapper Layer**
- **`agents_wrapper.py`** - Minimal wrappers for original agents
  - **`PreprocessingAgentWrapper`** - Routes to sequential preprocessing
  - **`FeatureSelectionAgentWrapper`** - Routes to agentic feature selection
  - **`ModelBuildingAgentWrapper`** - Routes to LangGraph model building
  - **No modification** of original agent code
  - **Data format conversion** between pipeline and agent states

### **5. Original Working Agents**
These are your existing, fully-functional agents used AS-IS:

#### **`preprocessing_agent_impl.py`**
- **`SequentialState`** - Comprehensive state management
- **LangGraph workflow** with interactive phases:
  - Overview → Outliers → Missing Values → Encoding → Transformations
- **LLM-powered intent classification** for user inputs
- **Interactive user prompts** and decision points
- **Phase-by-phase progression** with user approval

#### **`feature_selection_agent_impl.py`**
- **`AgenticFeatureSelectionBot`** - Main bot controller
- **`UserSession`** - Session state management
- **Analysis components**:
  - `DataProcessor` - Data loading and cleaning
  - `AnalysisEngine` - Statistical analysis execution
  - `MenuGenerator` - Interactive menu generation
  - `LLMManager` - Multi-provider LLM support
- **Analysis types**: IV, Correlation, CSI, VIF, SHAP, PCA, LASSO
- **Slack integration** with file upload and interactive menus

#### **`model_building_agent_impl.py`**
- **`LangGraphModelAgent`** - LangGraph-based workflow
- **Prompt understanding** and intent classification
- **Model building workflows** with LLM guidance
- **Evaluation and metrics** generation
- **Conversation history** and session persistence

### **6. State Management Layer**
- **`pipeline_state.py`** - Global state schema
  - **Core data fields**: `raw_data`, `cleaned_data`, `processed_data`
  - **Agent states**: `preprocessing_state`, `feature_selection_state`, `model_building_state`
  - **Multi-model storage**: `models` dict with metadata
  - **Session management**: `interactive_session`, `chat_session`
  - **Execution tracking**: `execution_history`, `current_agent`

### **7. Shared Utilities Layer**
- **`toolbox.py`** - Global utility classes
  - **`SlackManager`** - Multi-session Slack integration
  - **`ArtifactManager`** - File and artifact management
  - **`ProgressTracker`** - Progress updates with deduplication
  - **`UserDirectoryManager`** - User directory structure management
  - **`ExecutionAgent`** - Code execution with LLM fallback

### **8. Slack Integration Layer**
- **`slack_bot.py`** - Slack bot interface
  - **Message handling** with bot mentions
  - **File upload processing** for CSV data
  - **Progress callbacks** and real-time updates
  - **Thread management** for conversations

### **9. Persistence Layer**
- **User Directory Structure**: `user_data/{user}/{thread}/`
  - **`conversation_history.json`** - Chat history persistence
  - **`session_state.json`** - Pipeline state persistence
  - **DataFrames as CSV files** - `raw_data.csv`, `cleaned_data.csv`
  - **`artifacts/`** - Generated files and outputs
  - **`models/`** - Trained model storage with metadata

## 🔄 **Data Flow Architecture**

### **Request Processing Flow**
```
User Query → Slack Bot → Pipeline → Orchestrator → Agent Wrapper → Original Agent → Response
```

### **State Persistence Flow**
```
Pipeline State → Session Serialization → User Directory → CSV Files + JSON Metadata
```

### **Interactive Session Flow**
```
Agent Start → Interactive Menu → User Input → Agent Processing → Continue/Complete
```

## 🎯 **Key Design Principles**

### **1. Preservation of Working Code**
- **No modification** of your original agent implementations
- **Wrapper pattern** for integration without changes
- **AS-IS usage** of existing functionality

### **2. Layered Abstraction**
- **Clear separation** between orchestration and execution
- **Minimal coupling** between layers
- **Independent operation** of each agent

### **3. State Management**
- **Centralized state** in `PipelineState`
- **Session persistence** across conversations
- **DataFrame serialization** for data continuity

### **4. Hybrid Intelligence**
- **Fast keyword classification** for common queries
- **LLM fallback** for complex intent detection
- **Multi-provider LLM support** (OpenAI, Ollama)

### **5. Robust Error Handling**
- **Graceful degradation** when components unavailable
- **Fallback mechanisms** at every layer
- **Error isolation** to prevent system-wide failures

## 📊 **Component Interactions**

### **Orchestrator Decision Matrix**
| Query Type | Primary Method | Fallback | Target Agent |
|------------|---------------|----------|--------------|
| "clean my data" | Keyword (preprocessing) | - | PreprocessingAgent |
| "select features" | Keyword (feature_selection) | - | FeatureSelectionAgent |
| "build model" | Keyword (model_building) | - | ModelBuildingAgent |
| "explain LGBM" | LLM Classification | Keyword | GeneralResponse |
| Complex queries | LLM Classification | Keyword | Appropriate Agent |

### **Agent Wrapper Responsibilities**
| Wrapper | Input Conversion | Output Extraction | Error Handling |
|---------|-----------------|-------------------|----------------|
| Preprocessing | PipelineState → temp CSV + SequentialState | SequentialState.df → PipelineState.cleaned_data | Basic preprocessing fallback |
| Feature Selection | PipelineState → UserSession | UserSession results → PipelineState.selected_features | Session setup only |
| Model Building | PipelineState → agent parameters | Agent results → PipelineState.models | Basic model fallback |

### **State Transitions**
```
Initial → Data Upload → Preprocessing → Feature Selection → Model Building → Complete
    ↓         ↓              ↓               ↓                ↓
 Raw Data  Cleaned Data  Selected Features  Trained Models  Final Results
```

## 🔧 **Configuration & Extensibility**

### **Environment Variables**
- `SLACK_BOT_TOKEN` - Slack bot authentication
- `SLACK_APP_TOKEN` - Slack app authentication  
- `OPENAI_API_KEY` - OpenAI API access
- `DEFAULT_MODEL` - Default LLM model selection

### **Adding New Agents**
1. Create agent wrapper in `agents_wrapper.py`
2. Add intent classification in `orchestrator.py`
3. Add routing logic in `langgraph_pipeline.py`
4. Update `PipelineState` if needed

### **Customizing Workflows**
- **Orchestrator keywords** - Modify classification patterns
- **Agent parameters** - Adjust wrapper conversion logic
- **State fields** - Extend `PipelineState` schema
- **Persistence** - Modify serialization in `langgraph_pipeline.py`

## 🚀 **Deployment Architecture**

### **Development Setup**
```
Local Machine → Python Environment → Slack Workspace → Ollama/OpenAI
```

### **Production Considerations**
- **Containerization** with Docker
- **Environment isolation** with proper secrets management
- **Scalability** through agent instance pooling
- **Monitoring** with comprehensive logging
- **Backup** of user directory structures

This architecture ensures your working agents remain untouched while providing a unified, scalable interface for multi-agent ML workflows.
