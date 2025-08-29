# Multi-Agent ML Integration System

A unified orchestrator system that intelligently routes ML workflow queries to specialized agents using hybrid keyword + LLM classification.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive launcher
python start_pipeline.py

# Or use quick launcher
python run.py --mode demo
```

## 🎯 System Overview

- **🔧 PreprocessingAgent**: Data cleaning, missing values, outliers, encoding
- **🎯 FeatureSelectionAgent**: IV analysis, correlation, PCA, feature importance  
- **🤖 ModelBuildingAgent**: Train models, evaluate, predict, visualize
- **🎛️ Orchestrator**: Intelligent query routing with hybrid classification

## 📚 Documentation

All detailed documentation is organized in the [`docs/`](docs/) folder:

- **[📖 Documentation Index](docs/INDEX.md)** - Complete documentation guide
- **[🚀 Quick Start](docs/QUICK_START.md)** - Get up and running fast  
- **[📋 How to Run](docs/HOW_TO_RUN.md)** - Comprehensive usage guide
- **[🏗️ Architecture](docs/ARCHITECTURE_DOCUMENTATION.md)** - System design and flow
- **[⚙️ Setup Guide](docs/SETUP_GUIDE.md)** - Installation and configuration

## 🔧 Key Features

- **Hybrid Classification**: Fast keyword scoring + LLM fallback for ambiguous cases
- **Exhaustive Keywords**: 125+ domain-specific ML terms with text normalization
- **Multi-Session Support**: Slack integration with persistent state
- **Graceful Degradation**: Works without LLM libraries (keyword-only mode)
- **Pipeline Flexibility**: Full pipeline or direct agent entry points

## 🏗️ Architecture

```
User Query → Orchestrator → [Preprocessing|FeatureSelection|ModelBuilding] → Response
              ↓
         Hybrid Router
         ├── ⚡ Keyword Scoring (90%+ queries, <1ms)
         └── 🤖 LLM Fallback (ambiguous cases, 100-500ms)
```

## 📊 Performance

- **90%+ queries**: Sub-millisecond keyword classification
- **< 10% queries**: LLM fallback for complex/ambiguous cases
- **Scalable**: Enterprise-grade performance with research-grade intelligence

---

For detailed documentation, examples, and advanced usage, see the [`docs/`](docs/) folder.
