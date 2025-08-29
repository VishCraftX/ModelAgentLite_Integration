# Orchestrator Architecture - Separation of Concerns

## 🎯 **Architectural Principle**

The orchestrator's job is to determine **WHICH AGENT** should handle a query, not **HOW** the agent should handle it.

## 🔄 **Corrected Flow**

### **Before (Incorrect)**
```
User Query → Orchestrator → Classifies as "use_existing" → Routes to ModelBuildingAgent
```
**Problem**: Orchestrator making agent-internal decisions

### **After (Correct)**
```
User Query → Orchestrator → Classifies as "model_building" → Routes to ModelBuildingAgent
                                                           ↓
ModelBuildingAgent → Analyzes query → Determines "use_existing" vs "build_new"
```

## 📋 **Orchestrator Responsibilities**

### ✅ **What Orchestrator SHOULD Do:**
- **Agent Selection**: Which agent can handle this type of request?
- **High-Level Classification**: Is this about data, features, or models?
- **Pipeline Flow**: What's the next logical step in the workflow?

### ❌ **What Orchestrator SHOULD NOT Do:**
- **Agent-Internal Logic**: How should the agent handle the request?
- **Model-Specific Decisions**: Use existing vs build new model
- **Implementation Details**: Which algorithm, which parameters, etc.

## 🎛️ **Updated Intent Categories**

### **Orchestrator Level (6 categories):**
1. **`preprocessing`** - Route to PreprocessingAgent
2. **`feature_selection`** - Route to FeatureSelectionAgent  
3. **`model_building`** - Route to ModelBuildingAgent (any model operation)
4. **`general_query`** - Handle conversationally
5. **`code_execution`** - Route to appropriate agent for code execution
6. **`full_pipeline`** - Sequential routing through all agents

### **ModelBuildingAgent Level (internal classification):**
- **`build_new_model`** - Train a new model from scratch
- **`use_existing_model`** - Operations on existing model
- **`model_evaluation`** - Evaluate model performance
- **`model_visualization`** - Show plots, trees, etc.
- **`prediction`** - Make predictions using model

## 🔍 **Example Routing Decisions**

### **All Route to ModelBuildingAgent:**
```python
"Train a new LightGBM model" → model_building → MBA decides: build_new_model
"Use existing model for predictions" → model_building → MBA decides: use_existing_model  
"Show model performance metrics" → model_building → MBA decides: model_evaluation
"Visualize decision tree" → model_building → MBA decides: model_visualization
"Score this dataset" → model_building → MBA decides: prediction
```

### **Other Agent Routing:**
```python
"Clean missing values" → preprocessing → PreprocessingAgent handles
"Select important features" → feature_selection → FeatureSelectionAgent handles
"Build complete pipeline" → full_pipeline → Sequential routing
"What can you do?" → general_query → Orchestrator handles directly
```

## 🏗️ **Benefits of This Architecture**

### **1. Clear Separation of Concerns**
- **Orchestrator**: High-level routing logic
- **Agents**: Domain-specific implementation logic

### **2. Maintainability**
- Changes to model operations don't affect orchestrator
- Each component has single responsibility

### **3. Extensibility**
- Easy to add new model operations to ModelBuildingAgent
- Orchestrator routing remains stable

### **4. Consistency**
- All model-related queries go through same path
- ModelBuildingAgent has complete context for decisions

## 🎯 **Implementation Result**

Now the system has **proper separation of concerns**:

- **Orchestrator**: "This is about models" → Route to ModelBuildingAgent
- **ModelBuildingAgent**: "This is about using existing model" → Handle accordingly

This matches the **ModelBuildingAgent's internal architecture** where it has its own prompt understanding and controller logic for model-specific operations!
