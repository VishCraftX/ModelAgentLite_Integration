# Multi-Agent ML Integration System - Architecture Documentation

## 🏗️ System Architecture Overview

The Multi-Agent ML Integration System is built with a layered architecture that provides intelligent routing, state management, and seamless integration of existing ML agents.

## 📊 Architecture Layers

### 1. **User Interface Layer**
- **Slack Bot Interface**: File upload, natural language queries, real-time progress
- **Python API**: Direct programmatic access to the pipeline
- **Direct Agent Access**: Individual agent execution for specific tasks

### 2. **Pipeline Orchestration Layer**
- **MultiAgentMLPipeline**: Main pipeline coordinator
- **Orchestrator**: Intelligent query routing and decision making
- **State Manager**: Persistent state management across sessions

### 3. **Agent Layer**
- **IntegratedPreprocessingAgent**: Real implementation + fallback
- **IntegratedFeatureSelectionAgent**: Real implementation + fallback  
- **IntegratedModelBuildingAgent**: Real implementation + fallback

### 4. **Shared Toolbox Layer**
- **SlackManager**: Multi-session messaging
- **ArtifactManager**: Session-isolated file storage
- **ProgressTracker**: Real-time progress updates
- **ExecutionAgent**: Code execution with LLM fallback

### 5. **State Management Layer**
- **PipelineState**: Global shared state object
- **Session Storage**: Persistent state across restarts
- **Artifact Storage**: Files, models, and generated content

### 6. **External Integration Layer**
- **LLM Providers**: Ollama (local) and OpenAI (cloud)
- **Slack API**: Bot integration and messaging
- **File System**: Artifact and state persistence

## 🔄 Control Flow Architecture

### Query Processing Flow

1. **Entry Point**: User submits query via Slack, API, or direct access
2. **Data Loading**: CSV/Excel files processed and loaded into pipeline state
3. **Orchestrator Analysis**: Query analyzed for intent and routing decision
4. **Agent Execution**: Appropriate agent(s) executed based on routing
5. **State Updates**: PipelineState updated with results
6. **Artifact Management**: Generated files saved with session isolation
7. **Progress Tracking**: Real-time updates sent to user
8. **Response Generation**: Results formatted and returned

### Orchestrator Intelligence

The orchestrator uses multiple strategies for routing:

#### 1. **Keyword-Based Routing**
```python
preprocessing_keywords = ["clean", "preprocess", "missing", "outlier", "normalize"]
feature_selection_keywords = ["feature", "select", "iv", "correlation", "pca"]
model_building_keywords = ["model", "train", "predict", "algorithm", "evaluate"]
```

#### 2. **Pattern Recognition**
- **Full Pipeline**: "Train XGBoost on this CSV" → Complete workflow
- **Direct Entry**: "Select features using IV > 0.02" → Feature selection only
- **Resume**: "Continue from last session" → State-driven routing

#### 3. **Data-Driven Routing**
When keywords are unclear, routing based on current state:
- No raw data → Preprocessing
- No cleaned data → Preprocessing  
- No selected features → Feature Selection
- No trained model → Model Building
- Complete pipeline → Additional operations

#### 4. **Score-Based Decision Making**
```python
scores = {
    "preprocessing": keyword_matches / total_words,
    "feature_selection": keyword_matches / total_words,
    "model_building": keyword_matches / total_words
}
selected_agent = max(scores, key=scores.get)
```

## 📊 State Management Architecture

### PipelineState Structure

```python
class PipelineState(BaseModel):
    # Core Data
    raw_data: Optional[pd.DataFrame] = None
    cleaned_data: Optional[pd.DataFrame] = None  
    selected_features: Optional[List[str]] = None
    trained_model: Optional[Any] = None
    
    # Session Management
    session_id: Optional[str] = None
    chat_session: Optional[str] = None
    artifacts: Optional[Dict] = Field(default_factory=dict)
    
    # Agent-Specific States
    preprocessing_state: Optional[Dict] = Field(default_factory=dict)
    feature_selection_state: Optional[Dict] = Field(default_factory=dict)
    model_building_state: Optional[Dict] = Field(default_factory=dict)
    
    # Execution Context
    current_agent: Optional[str] = None
    execution_history: List[Dict] = Field(default_factory=list)
    progress: Optional[str] = None
    last_code: Optional[str] = None
    last_error: Optional[str] = None
```

### State Transitions

The system follows a clear state progression:

1. **Initial State**: Empty state with session identifiers
2. **Raw Data Loaded**: DataFrame loaded, ready for processing
3. **Preprocessing Complete**: Cleaned data available, preprocessing metadata stored
4. **Feature Selection Complete**: Features selected, selection metadata stored
5. **Model Building Complete**: Model trained, all artifacts saved

### State Update Mechanisms

#### 1. **Agent State Updates**
Each agent updates specific parts of the global state:

```python
# Preprocessing Agent
state.cleaned_data = processed_dataframe
state.preprocessing_state = {
    "completed": True,
    "timestamp": datetime.now().isoformat(),
    "original_shape": original_shape,
    "cleaned_shape": cleaned_shape,
    "phases_completed": ["overview", "missing_values", "outliers"]
}

# Feature Selection Agent  
state.selected_features = selected_feature_list
state.feature_selection_state = {
    "completed": True,
    "timestamp": datetime.now().isoformat(),
    "total_features": total_count,
    "selected_features": selected_count,
    "selection_method": "iv_analysis"
}

# Model Building Agent
state.trained_model = trained_model_object
state.model_building_state = {
    "completed": True,
    "timestamp": datetime.now().isoformat(),
    "model_type": "RandomForestClassifier",
    "features_used": feature_count
}
```

#### 2. **Artifact Management**
Each state update includes artifact saving:

```python
# Save cleaned data
artifact_path = artifact_manager.save_artifact(
    session_id, "cleaned_data.csv", cleaned_data, "dataframe"
)
state.artifacts["cleaned_data_path"] = artifact_path

# Save selected features
artifact_path = artifact_manager.save_artifact(
    session_id, "selected_features.json", 
    {"selected_features": selected_features}, "json"
)
state.artifacts["selected_features_path"] = artifact_path

# Save trained model
artifact_path = artifact_manager.save_artifact(
    session_id, "trained_model.joblib", model_binary, "binary"
)
state.artifacts["trained_model_path"] = artifact_path
```

#### 3. **Progress Tracking**
Real-time progress updates throughout execution:

```python
def _update_progress(self, state: PipelineState, message: str, stage: str = None):
    state.current_agent = self.agent_name
    self.progress_tracker.update(state, message, stage)
    # Automatically sends to Slack if session configured
```

## 🔧 Agent Integration Architecture

### Real Implementation Integration

Each agent integrates with existing implementations:

#### 1. **Preprocessing Agent Integration**
```python
class IntegratedPreprocessingAgent(BaseAgent):
    def __init__(self):
        # Try to import real implementation
        try:
            from preprocessing_agent_impl import SequentialState, PreprocessingPhase
            self.preprocessing_available = True
        except ImportError:
            self.preprocessing_available = False
    
    def run(self, state: PipelineState) -> PipelineState:
        if self.preprocessing_available:
            return self._run_comprehensive_preprocessing(state)
        else:
            return self._run_basic_preprocessing(state)
```

#### 2. **Fallback Mechanism**
When real implementations aren't available:

```python
def _run_basic_preprocessing(self, state: PipelineState) -> pd.DataFrame:
    df = state.raw_data.copy()
    
    # Basic operations
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    return df
```

### Error Handling and Recovery

#### 1. **LLM-Powered Error Recovery**
```python
def fallback_fix(self, state: PipelineState, code: str, error: str, attempt: int = 1):
    if attempt > 2:
        return state  # Max attempts reached
    
    # Select model based on attempt
    model = self.fallback_model_1 if attempt == 1 else self.fallback_model_2
    
    # Generate fix using LLM
    llm = self._get_llm(model)
    fixed_code = llm.invoke([HumanMessage(content=fallback_prompt)])
    
    # Try executing fixed code
    try:
        exec(fixed_code, execution_globals)
        state.last_code = fixed_code
        return state
    except Exception as e:
        return self.fallback_fix(state, fixed_code, str(e), attempt + 1)
```

#### 2. **Graceful Degradation**
- **Missing Dependencies**: Use fallback implementations
- **LLM Unavailable**: Skip error recovery, return original error
- **Slack Unavailable**: Log to console instead
- **LangGraph Unavailable**: Use simplified pipeline mode

## 🔄 Session and Persistence Architecture

### Session Management

#### 1. **Session Isolation**
```python
def get_session_id(user_id: str, thread_ts: Optional[str] = None) -> str:
    if thread_ts:
        return f"{user_id}_{thread_ts}"  # Slack thread-based
    return user_id  # Direct API usage
```

#### 2. **State Persistence**
```python
class StateManager:
    def save_state(self, state: PipelineState) -> str:
        session_dir = os.path.join(self.base_dir, state.session_id)
        
        # Save DataFrames separately
        if state.raw_data is not None:
            state.raw_data.to_pickle(os.path.join(session_dir, "raw_data.pkl"))
        
        # Save state metadata
        state_dict = state.dict()
        with open(os.path.join(session_dir, "state.json"), 'w') as f:
            json.dump(state_dict, f, default=str)
```

#### 3. **Artifact Storage**
```python
class ArtifactManager:
    def save_artifact(self, session_id: str, filename: str, content: Any):
        session_dir = self.base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        if artifact_type == "dataframe":
            content.to_csv(session_dir / filename, index=False)
        elif artifact_type == "json":
            with open(session_dir / filename, 'w') as f:
                json.dump(content, f, indent=2)
```

## 🚀 Deployment Architecture

### Scalability Considerations

1. **Horizontal Scaling**: Multiple pipeline instances with shared state storage
2. **Session Isolation**: Each user/thread has independent state
3. **Resource Management**: Configurable timeouts and cleanup
4. **Caching**: Artifact caching for frequently accessed data

### Configuration Management

```python
# Environment-based configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5-coder:32b-instruct-q4_K_M")
ENABLE_PERSISTENCE = os.getenv("ENABLE_PERSISTENCE", "true").lower() == "true"
SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
```

### Monitoring and Observability

1. **Progress Tracking**: Real-time updates to users
2. **Execution History**: Complete audit trail in state
3. **Error Logging**: Comprehensive error capture and reporting
4. **Performance Metrics**: Execution times and resource usage

## 🔍 Key Architectural Benefits

### 1. **Modularity**
- Each agent is independently replaceable
- Shared toolbox prevents code duplication
- Clear separation of concerns

### 2. **Resilience**
- Graceful degradation when components unavailable
- Multiple fallback mechanisms
- Comprehensive error handling

### 3. **Extensibility**
- Easy to add new agents
- Pluggable LLM providers
- Configurable routing logic

### 4. **User Experience**
- Multiple interaction modes
- Real-time progress feedback
- Session continuity across restarts

### 5. **Production Ready**
- Comprehensive logging and monitoring
- Configurable deployment options
- Scalable session management

This architecture provides a robust, scalable, and maintainable foundation for the multi-agent ML system while preserving the functionality of existing agent implementations.
