# Component Breakdown - Multi-Agent ML Integration System

## 🏗️ **Core Components**

### **1. Slack Interface (`slack_bot.py`)**

#### **SlackMLBot Class**
```python
class SlackMLBot:
    def __init__(self):
        self.app = App(token=SLACK_BOT_TOKEN)
        self.ml_pipeline = get_pipeline()
        self.chat_sessions = defaultdict(list)
        self.session_data = defaultdict(pd.DataFrame)
```

**Key Methods**:
- `_setup_handlers()`: Configure Slack event listeners
- `_handle_file_attachments()`: Process uploaded files
- `_handle_text_query()`: Route text queries to pipeline
- `_get_session_id()`: Generate unique session identifiers

**Event Handlers**:
- `@app.event("app_mention")`: Handle bot mentions
- `@app.event("message")`: Process direct messages
- `@app.command("/pipeline_status")`: Status command

---

### **2. Pipeline Orchestration (`langgraph_pipeline.py`)**

#### **MultiAgentMLPipeline Class**
```python
class MultiAgentMLPipeline:
    def __init__(self, slack_token, artifacts_dir, user_data_dir, enable_persistence):
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        self.user_sessions = {}
```

**Key Methods**:
- `process_query()`: Main query processing entry point
- `_build_graph()`: Construct LangGraph state machine
- `_handle_preprocessing_interaction()`: Interactive preprocessing commands
- `_save_session_state()` / `_load_session_state()`: State persistence
- `_save_conversation_history()`: Chat history management

**LangGraph Nodes**:
- `_orchestrator_node()`: Intent classification
- `_preprocessing_node()`: Data preprocessing
- `_feature_selection_node()`: Feature engineering
- `_model_building_node()`: Model training
- `_code_execution_node()`: Code execution
- `_general_response_node()`: Conversational responses

---

### **3. Intent Classification (`orchestrator.py`)**

#### **Orchestrator Class**
```python
class Orchestrator:
    def __init__(self):
        self.preprocessing_keywords = {...}
        self.feature_selection_keywords = {...}
        self.model_building_keywords = {...}
```

**Classification Methods**:
- `classify_intent()`: Main classification entry point
- `_classify_with_keyword_scoring()`: Fast keyword-based classification
- `_fallback_classify_intent()`: LLM-powered fallback
- `_normalize_text()`: Text preprocessing with lemmatization

**Intent Categories**:
- `preprocessing`: Data cleaning and preparation
- `feature_selection`: Feature engineering and selection
- `model_building`: Model training and evaluation
- `general_query`: Conversational queries
- `code_execution`: Code execution requests

---

### **4. Agent Wrappers (`agents_wrapper.py`)**

#### **PreprocessingAgentWrapper**
```python
class PreprocessingAgentWrapper:
    def __init__(self):
        self.available = PREPROCESSING_AVAILABLE
        self.slack_bot = create_slack_preprocessing_bot()
    
    def run(self, state: PipelineState) -> PipelineState:
        # Interactive preprocessing workflow
```

**Features**:
- Interactive Slack menus for preprocessing options
- Target column selection
- Phase-by-phase processing (Overview → Outliers → Missing → Encoding → Transformations)
- Real-time progress updates

#### **FeatureSelectionAgentWrapper**
```python
class FeatureSelectionAgentWrapper:
    def __init__(self):
        self.available = FEATURE_SELECTION_AVAILABLE
        self.bot = AgenticFeatureSelectionBot()
    
    def run(self, state: PipelineState) -> PipelineState:
        # Feature selection workflow
```

**Features**:
- Statistical analysis (IV, CSI, Correlation)
- Advanced techniques (SHAP, VIF, PCA, LASSO)
- Interactive feature filtering
- Waterfall analysis pipeline

#### **ModelBuildingAgentWrapper**
```python
class ModelBuildingAgentWrapper:
    def __init__(self):
        self.available = MODEL_BUILDING_AVAILABLE
        self.agent = LangGraphModelAgent()
    
    def run(self, state: PipelineState) -> PipelineState:
        # Model building workflow
```

**Features**:
- Multiple algorithm support
- Automated hyperparameter tuning
- Comprehensive evaluation metrics
- Model persistence and versioning

---

### **5. State Management (`pipeline_state.py`)**

#### **PipelineState Class**
```python
class PipelineState(BaseModel):
    # Core data fields
    raw_data: Optional[pd.DataFrame] = None
    cleaned_data: Optional[pd.DataFrame] = None
    processed_data: Optional[pd.DataFrame] = None
    selected_features: Optional[List[str]] = None
    target_column: Optional[str] = None
    
    # Multi-model storage
    models: Optional[Dict[str, Dict]] = Field(default_factory=dict)
    best_model: Optional[str] = None
    
    # Session management
    session_id: Optional[str] = None
    chat_session: Optional[str] = None
    interactive_session: Optional[Dict] = None
```

**Key Features**:
- Pydantic validation and serialization
- DataFrame handling with custom encoders
- Execution history tracking
- Progress monitoring
- Agent-specific state extensions

#### **StateManager Class**
```python
class StateManager:
    def save_state(self, state: PipelineState) -> str:
        # Save state with DataFrame serialization
    
    def load_state(self, session_id: str) -> Optional[PipelineState]:
        # Load and reconstruct state
```

---

### **6. Utility Services (`toolbox.py`)**

#### **SlackManager**
```python
class SlackManager:
    def __init__(self, token: str):
        self.client = WebClient(token=token)
        self.session_channels = {}
        self.session_threads = {}
    
    def send_message(self, session_id: str, text: str):
        # Send message to registered channel/thread
```

**Features**:
- Session-based channel management
- Thread-aware messaging
- Error handling and fallbacks
- Message formatting and delivery

#### **ProgressTracker**
```python
class ProgressTracker:
    def update(self, stage: str, message: str, session_id: str = None):
        # Update progress with debouncing
```

**Features**:
- Real-time progress updates
- Duplicate message prevention
- Slack integration
- Stage-based tracking

#### **ArtifactManager**
```python
class ArtifactManager:
    def save_artifact(self, data: Any, filename: str, session_id: str):
        # Save artifacts with session organization
```

**Features**:
- Session-based artifact organization
- Multiple format support (CSV, JSON, pickle)
- Automatic cleanup
- Metadata tracking

#### **UserDirectoryManager**
```python
class UserDirectoryManager:
    def ensure_user_directory(self, user_id: str, thread_id: str):
        # Create user-specific directory structure
```

**Directory Structure**:
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

---

### **7. Agent Implementations**

#### **Preprocessing Agent (`preprocessing_agent_impl.py`)**
```python
class SequentialState:
    df_path: str
    target_column: str
    current_phase: str = PreprocessingPhase.OVERVIEW
    completed_phases: List[str] = []
```

**Key Functions**:
- `run_sequential_agent()`: Main preprocessing workflow
- `analyze_*_with_llm()`: LLM-powered analysis functions
- `apply_*_treatment()`: Data transformation functions
- `classify_user_intent_with_llm()`: Intent classification

**Processing Phases**:
1. **Overview**: Dataset analysis and summary
2. **Outliers**: Detection and handling strategies
3. **Missing Values**: Imputation and removal strategies
4. **Encoding**: Categorical variable encoding
5. **Transformations**: Feature scaling and transformations

#### **Feature Selection Agent (`feature_selection_agent_impl.py`)**
```python
class AgenticFeatureSelectionBot:
    def __init__(self):
        self.app = App(token=os.environ.get("SLACK_BOT_TOKEN_FS"))
        self.users: Dict[str, UserSession] = {}
```

**Analysis Types**:
- **IV Analysis**: Information Value calculation
- **Correlation Analysis**: Feature correlation detection
- **CSI Analysis**: Characteristic Stability Index
- **VIF Analysis**: Variance Inflation Factor
- **SHAP Analysis**: Feature importance via SHAP values
- **Custom Analysis**: User-defined analysis code

#### **Model Building Agent (`model_building_agent_impl.py`)**
```python
class LangGraphModelAgent:
    def __init__(self):
        self.graph = self._create_graph()
        self.app = self.graph.compile()
```

**Capabilities**:
- Algorithm selection and configuration
- Automated hyperparameter tuning
- Cross-validation and evaluation
- Model persistence and loading
- Prediction and inference
- Rank-order metrics for classification

---

## 🔄 **Interaction Patterns**

### **1. File Upload Flow**
```
User uploads file → SlackMLBot processes → Extract data → Store in session → Notify pipeline
```

### **2. Query Processing Flow**
```
User query → SlackMLBot → MultiAgentMLPipeline → Orchestrator → Agent → State update → Response
```

### **3. Interactive Session Flow**
```
Initial request → Agent setup → Interactive menu → User commands → Agent processing → Results
```

### **4. State Persistence Flow**
```
State changes → Serialize DataFrames → Save to user directory → Load on session resume
```

---

## 🎯 **Key Design Patterns**

### **1. Wrapper Pattern**
- Agent wrappers provide consistent interface
- Original agent implementations remain unchanged
- Enables gradual integration and testing

### **2. State Machine Pattern**
- LangGraph manages workflow states
- Conditional routing based on state
- Resumable and recoverable workflows

### **3. Session Management Pattern**
- User-thread isolation
- Persistent state across interactions
- Conversation history tracking

### **4. Hybrid Classification Pattern**
- Fast keyword scoring for common cases
- LLM fallback for complex queries
- Confidence-based routing decisions

---

## 📊 **Performance Characteristics**

### **Response Times**
- **Keyword Classification**: ~10ms
- **LLM Classification**: ~2-5s
- **File Processing**: ~100ms per MB
- **State Persistence**: ~50ms

### **Memory Usage**
- **Base System**: ~200MB
- **Per Session**: ~50MB + data size
- **LLM Models**: ~4-8GB (depending on model)

### **Scalability Limits**
- **Concurrent Sessions**: 50-100 (memory dependent)
- **File Size**: Up to 100MB (configurable)
- **Session Duration**: 24 hours (configurable)

---

*This component breakdown provides a comprehensive view of the system's architecture and implementation details.*
