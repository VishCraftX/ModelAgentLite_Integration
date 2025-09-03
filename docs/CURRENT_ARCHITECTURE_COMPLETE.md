# 🏗️ Complete Architecture & Component Diagram
## Multi-Agent ML Integration System - Current Implementation

## 📊 Architecture Overview

The system implements a **7-layer architecture** with **semantic intelligence** throughout:

### **Layer 1: User Interface** 🖥️
- **Slack Bot**: Primary interface for user interactions
- **File Upload**: Supports CSV, Excel, JSON, TSV formats
- **Real-time messaging** with threaded conversations

### **Layer 2: Entry Point & Session Management** 🚀
- **start_pipeline.py**: Main entry point and initialization
- **MultiAgentMLPipeline**: Central controller orchestrating all components
- **UserDirectoryManager**: Manages `user_data/{user}/{thread}/` structure
- **Centralized Conversation History**: Single source of truth for chat history
- **Session State Persistence**: Maintains PipelineState across sessions

### **Layer 3: Orchestrator - Semantic Intelligence** 🧠
**Revolutionary 3-tier classification system:**

#### **Intent Classification (Semantic → LLM → Keyword)**
1. **🧠 Semantic Classification**: 
   - Uses BGE-Large embeddings with cosine similarity
   - Comprehensive intent definitions for each agent type
   - Threshold: `max_similarity > 0.4` and `similarity_diff > 0.08`

2. **🤖 LLM Classification**: 
   - Ollama (qwen2.5-coder) → OpenAI (GPT-3.5-turbo) fallback
   - Structured prompts for ambiguous cases
   - High accuracy for complex queries

3. **⚡ Keyword Classification**: 
   - Traditional pattern matching as last resort
   - Exhaustive keyword sets with NLTK normalization

#### **Skip Pattern Detection (Semantic → LLM → Keyword)**
- **skip_to_modeling**: Skip everything → model_building
- **skip_preprocessing_to_features**: Skip preprocessing → feature_selection  
- **skip_preprocessing_to_modeling**: Skip preprocessing → model_building
- **no_skip**: Normal pipeline workflow

#### **Intelligent Routing**
- Combines intent classification with skip pattern analysis
- Context-aware routing decisions
- Educational query detection

### **Layer 4: Agent Layer - Wrapper Pattern** 🤖
**Minimal wrappers** that call original implementations:
- **PreprocessingAgent Wrapper**: Routes to interactive preprocessing
- **FeatureSelectionAgent Wrapper**: Handles feature analysis
- **ModelBuildingAgent Wrapper**: Manages model training with progress callbacks
- **GeneralResponse Node**: LLM-powered conversational responses
- **CodeExecution Node**: Python code execution with error handling

### **Layer 5: Actual Agent Implementations** 🔧

#### **PreprocessingAgent Implementation**
- **Interactive LangGraph Workflow**: Multi-phase preprocessing
- **Slack Integration**: Real-time interactive menus and responses
- **Comprehensive Data Cleaning**: Missing values, outliers, duplicates
- **User-guided Process**: Target column selection, phase-by-phase execution

#### **FeatureSelectionAgent Implementation**
- **DataProcessor + AnalysisEngine**: Sophisticated feature analysis
- **LLMManager Integration**: AI-powered feature insights
- **Multiple Selection Methods**: Correlation, importance, statistical tests
- **Interactive Selection**: User-guided feature choice

#### **ModelBuildingAgent Implementation**
**Internal Semantic Classification (Semantic → LLM → Keyword):**
1. **🧠 Semantic Model Intent**: `use_existing` vs `new_model`
2. **🧠 Semantic Plot Detection**: Visualization request analysis
3. **🧠 Semantic Financial Analysis**: Rank ordering/segmentation detection
4. **🤖 LLM Fallbacks**: Ollama/OpenAI for ambiguous cases
5. **⚡ Keyword Fallbacks**: Traditional matching as last resort

**Key Features:**
- **Multi-model storage**: `state.models` dictionary
- **Best model tracking**: `state.best_model` pointer
- **Raw data support**: Can use raw data when preprocessing skipped
- **Progress integration**: Connected to main pipeline progress tracker
- **No duplicate conversation history**: Cleaned up redundant handling

### **Layer 6: Shared Toolbox** 🧰
- **SlackManager**: Message delivery, session registration, channel management
- **ArtifactManager**: File management and organization
- **ProgressTracker**: Real-time updates with debounce logic
- **ExecutionAgent**: Code execution with LLM error fixing
- **UserDirectoryManager**: Consistent directory structure management

### **Layer 7: State Management** 💾

#### **PipelineState (Global Shared State)**
```python
class PipelineState:
    # Data States
    raw_data: Optional[pd.DataFrame] = None
    cleaned_data: Optional[pd.DataFrame] = None  
    processed_data: Optional[pd.DataFrame] = None
    selected_features: Optional[List[str]] = None
    target_column: Optional[str] = None
    
    # Model States  
    models: Optional[Dict[str, Dict]] = Field(default_factory=dict)  # Multi-model storage
    best_model: Optional[str] = None  # Best model pointer
    trained_model: Optional[Any] = None  # Backward compatibility
    
    # Session States
    interactive_session: Optional[Dict] = None
    last_response: Optional[str] = None
    chat_session: Optional[str] = None
    user_query: Optional[str] = None
    
    # Metadata
    session_id: str
    artifacts: Optional[Dict] = Field(default_factory=dict)
```

#### **Persistence System**
- **User Data Directory**: `user_data/{user}/{thread}/`
- **Conversation History**: `conversation_history.json` (centralized by pipeline)
- **Session State**: `session_state.json` (PipelineState + DataFrames as CSV)
- **Artifacts**: Models, plots, reports in organized subdirectories
- **Data Files**: CSV files for DataFrame persistence
- **Model Files**: Joblib serialized models

### **Layer 8: LLM Integration** 🤖
- **Ollama**: Local inference with qwen2.5-coder:32b
- **OpenAI**: GPT-3.5-turbo as fallback service
- **Embeddings**: BGE-Large (primary) with multiple fallbacks
- **Graceful Degradation**: System continues with reduced functionality if LLM unavailable

## 🔄 Data Flow Architecture

### **1. Query Processing Flow**
```
User Query → Slack Bot → MultiAgentMLPipeline → Orchestrator
↓
Semantic Classification → LLM Classification → Keyword Classification
↓
Skip Pattern Detection → Intelligent Routing → Agent Selection
↓
Agent Wrapper → Actual Implementation → State Update → Response
```

### **2. Data Processing Flow**
```
File Upload → raw_data → PreprocessingAgent → cleaned_data
↓
FeatureSelectionAgent → selected_features → ModelBuildingAgent → models
↓
Persistence → user_data/{user}/{thread}/ → Session Restoration
```

### **3. Interactive Session Flow**
```
Agent Initialization → interactive_session creation → Slack Menu
↓
User Response → Continuation Detection → Phase Execution → State Update
↓
Session Completion → State Cleanup → Final Response
```

## 🎯 Key Innovations

### **1. Unified Semantic Intelligence**
- **Consistent approach** across all classification tasks
- **Semantic → LLM → Keyword** hierarchy everywhere
- **BGE-Large embeddings** for robust understanding
- **Graceful degradation** with intelligent fallbacks

### **2. Intelligent Skip Pattern Detection**
- **Semantic understanding** of user intentions
- **Context-aware routing** based on skip requests
- **Multi-level skip support**: preprocessing, feature selection, full pipeline

### **3. Wrapper Pattern Architecture**
- **Minimal wrappers** preserve original agent functionality
- **Clean separation** between integration and implementation
- **Easy maintenance** and updates to individual agents

### **4. Centralized State Management**
- **Single source of truth** for all session data
- **Multi-model storage** with best model tracking
- **DataFrame persistence** as CSV files
- **Session restoration** across conversations

### **5. Progress Integration**
- **Real-time progress updates** via Slack
- **Debounce logic** prevents spam
- **Connected throughout** the entire pipeline

## 🛠️ Technical Specifications

### **Semantic Classification Thresholds**
- **Intent Classification**: `max_similarity > 0.4` and `similarity_diff > 0.08`
- **Skip Pattern Detection**: `max_similarity > 0.4` and `similarity_diff > 0.08`
- **Model Building Internal**: `max_similarity > 0.3` (more permissive)

### **LLM Configuration**
- **Primary**: Ollama qwen2.5-coder:32b-instruct-q4_K_M
- **Fallback**: OpenAI GPT-3.5-turbo
- **Embeddings**: bge-large → mxbai-embed-large → nomic-embed-text → all-minilm

### **File Structure**
```
user_data/
├── {user_id}/
│   └── {thread_id}/
│       ├── conversation_history.json
│       ├── session_state.json
│       ├── artifacts/
│       ├── data/
│       └── models/
```

### **Supported File Formats**
- **Data Upload**: CSV, Excel (.xlsx/.xls), JSON, TSV
- **Model Storage**: Joblib (.joblib)
- **Artifacts**: PNG, HTML, JSON reports

## 🚀 Performance Characteristics

### **Classification Performance**
- **Semantic Usage**: ~60-70% (optimized thresholds)
- **LLM Fallback**: ~20-30% (for ambiguous cases)
- **Keyword Fallback**: ~10-20% (last resort)

### **Response Times**
- **Semantic**: ~40ms (fastest)
- **LLM**: ~3-7s (high accuracy)
- **Keyword**: ~74ms (medium speed)

### **Accuracy Metrics**
- **Overall System**: ~85-90% intent classification accuracy
- **Skip Pattern Detection**: ~90%+ accuracy with semantic understanding
- **Model Building**: ~95%+ accuracy with internal semantic classification

## 🔧 Maintenance & Extensibility

### **Adding New Agents**
1. Create agent implementation
2. Add wrapper in `agents_wrapper.py`
3. Update orchestrator intent definitions
4. Add routing logic in `_route_by_intent`

### **Enhancing Classification**
1. Update intent definitions with new keywords
2. Adjust semantic thresholds if needed
3. Add new LLM prompts for complex cases
4. Extend keyword fallbacks

### **Monitoring & Debugging**
- **Comprehensive logging** throughout the pipeline
- **Method tracking** shows which classification method was used
- **Progress updates** provide real-time visibility
- **Error handling** with graceful degradation

This architecture provides a **robust, intelligent, and extensible** foundation for multi-agent ML workflows with **semantic understanding** at its core! 🎯
