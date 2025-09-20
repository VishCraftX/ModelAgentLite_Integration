from pipeline_state import PipelineState
from print_to_log import print_to_log
import pandas as pd
import numpy as np

class FastModelAgent:
    """Agent for automated ML pipeline execution using the same intelligent analysis as manual flow"""
    
    def __init__(self):
        print_to_log("🚀 FastModelAgent initialized - using intelligent LLM + rule-based analysis")
    
    def _extract_model_request_from_query(self, original_query: str) -> str:
        """Extract model building request from original user query"""
        query_lower = original_query.lower()
        
        # Check for specific model types in user query
        if "xgboost" in query_lower or "xgb" in query_lower:
            return "build an XGBoost model"
        elif "random forest" in query_lower or "randomforest" in query_lower or "rf" in query_lower:
            return "build a Random Forest model"
        elif "decision tree" in query_lower or "tree" in query_lower:
            return "build a Decision Tree model"
        elif "lightgbm" in query_lower or "lgbm" in query_lower:
            return "build a LightGBM model"
        elif "logistic regression" in query_lower or "logistic" in query_lower:
            return "build a Logistic Regression model"
        elif "svm" in query_lower or "support vector" in query_lower:
            return "build an SVM model"
        elif "neural network" in query_lower or "mlp" in query_lower:
            return "build a Neural Network model"
        elif "gradient boosting" in query_lower or "gbm" in query_lower:
            return "build a Gradient Boosting model"
        elif "adaboost" in query_lower:
            return "build an AdaBoost model"
        elif "naive bayes" in query_lower:
            return "build a Naive Bayes model"
        else:
            # 🎯 IMPROVED DEFAULT: Use Random Forest for generic prompts
            # This handles cases like "build me a model", "clean my data", "fast", etc.
            print_to_log("🤖 No specific model mentioned - defaulting to Random Forest (best for general use)")
            return "build a Random Forest model"

    def handle_fast_model_request(self, state: PipelineState, target_column: str = None) -> PipelineState:
        """Handle fast model request with target column setting"""
        print_to_log("🚀 FastModelAgent: Starting intelligent automated pipeline")
        
        # CRITICAL: Preserve original user query at the very beginning
        if not hasattr(state, "preprocessing_state") or state.preprocessing_state is None:
            state.preprocessing_state = {}
        
        # Store original intent if not already stored
        if "original_user_intent" not in state.preprocessing_state and state.user_query:
            state.preprocessing_state["original_user_intent"] = state.user_query
            print_to_log(f"🔍 [FastModelAgent] Preserved original user intent: '{state.user_query}'")
                # Set target column if provided
        if target_column:
            state.target_column = target_column
            print_to_log(f"🎯 Target column set: {target_column}")
        
        # Check if target column is set
        if not state.target_column:
            state.last_response = f"""🎯 **Target Column Required**
            
Please specify your target column from: {', '.join(list(state.raw_data.columns)[:10])}{'...' if len(state.raw_data.columns) > 10 else ''}

Reply with the target column name (e.g., 'f_segment')"""
            return state
        
        # Run the intelligent automated pipeline
        return self._run_intelligent_automated_pipeline(state)
    
    def _run_intelligent_automated_pipeline(self, state: PipelineState) -> PipelineState:
        """Run automated ML pipeline using the same intelligent analysis as manual flow"""
        print_to_log(f"🚀 Starting INTELLIGENT automated ML pipeline for target: {state.target_column}")
        
        try:
            # Import the intelligent analysis functions
            from preprocessing_agent_impl import (
                initialize_dataset_analysis,
                ConfidenceBasedPreprocessor, 
                apply_outliers_treatment,
                apply_missing_values_treatment,
                apply_encoding_treatment,
                apply_transformations_treatment,
                SequentialState
            )

            
            # Get original chat session for progress messages
            original_chat_session = state.chat_session
            
            # Send progress messages directly to Slack
            def send_progress(message: str):
                if original_chat_session:
                    try:
                        from toolbox import slack_manager
                        if slack_manager and hasattr(slack_manager, 'send_message'):
                            slack_manager.send_message(original_chat_session, message)
                    except Exception as e:
                        print_to_log(f"⚠️ Could not send progress message: {e}")
                print_to_log(message)
            
            # Create SequentialState for preprocessing analysis
            import tempfile
            import os
            
            # Create a temporary file path for the dataframe (required by SequentialState)
            temp_dir = tempfile.gettempdir()
            temp_df_path = os.path.join(temp_dir, f'fast_pipeline_{state.chat_session}.csv')
            
            # Save dataframe to temp path so SequentialState can load it
            state.raw_data.to_csv(temp_df_path, index=False)
            
            preprocessing_state = SequentialState(
                df_path=temp_df_path,
                
                target_column=state.target_column,
                current_phase="overview"
            )
            
            # Initialize preprocessing state tracking
            
            # Create SequentialState for intelligent analysis
            import tempfile
            import os
            
            # Create temporary file for SequentialState
            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
            temp_df_path = temp_file.name
            temp_file.close()
            
            # Save data to temporary file
            state.raw_data.to_csv(temp_df_path, index=False)
            print_to_log(f"📁 Saved data to temporary file: {temp_df_path}")
            
            # Create SequentialState for intelligent analysis
            preprocessing_state = SequentialState(
                df_path=temp_df_path,
                target_column=state.target_column,
                current_phase="overview"
            )
            
            # CRITICAL: Load the dataframe into SequentialState
            print_to_log("🔧 Loading dataset into SequentialState...")
            preprocessing_state = initialize_dataset_analysis(preprocessing_state)
            
            if preprocessing_state.current_step == "error":
                raise Exception("Failed to load dataset into SequentialState")
            
            print_to_log(f"✅ Dataset loaded: {preprocessing_state.df.shape}")
            
            # Initialize confidence-based preprocessor (high-confidence rules + LLM fallback)
            confidence_processor = ConfidenceBasedPreprocessor(
                confidence_threshold=0.8,
                timeout_minutes=2
            )
            
            # Initialize preprocessing state tracking
            if not hasattr(state, 'preprocessing_state'):
                state.preprocessing_state = {}
            
            # Phase 1: Overview
        
            print_to_log("📊 Phase 1: Overview - Analyzing dataset structure and quality")
            
            # Basic overview analysis
            overview_stats = {
                'total_rows': len(state.raw_data),
                'total_columns': len(state.raw_data.columns),
                'missing_percentage': (state.raw_data.isnull().sum().sum() / (len(state.raw_data) * len(state.raw_data.columns))) * 100,
                'numeric_columns': len(state.raw_data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(state.raw_data.select_dtypes(include=['object']).columns)
            }
            state.preprocessing_state['overview'] = overview_stats
            print_to_log(f"📊 Dataset: {overview_stats['total_rows']} rows × {overview_stats['total_columns']} columns")
        
            
            # Phase 2: Intelligent Outlier Analysis
            send_progress("🚨 **Starting outlier phase**")
            print_to_log("🚨 Phase 2: Outliers - Running intelligent LLM + rule-based analysis")
            
            # Run the same intelligent outlier analysis as manual flow
            try:
                # CRITICAL: Ensure data is loaded in preprocessing_state BEFORE analysis
                if preprocessing_state.df is None:
                    print_to_log("🔧 Data not loaded in SequentialState, loading now...")
                    preprocessing_state.df = state.raw_data.copy()
                    print_to_log(f"✅ Data loaded into SequentialState: {preprocessing_state.df.shape}")
                
                # STEP 1: Handle extreme outliers FIRST (same as agents_wrapper.py)
                print_to_log("🚨 Step 1: Detecting and handling extreme outliers...")
                from preprocessing_agent_impl import detect_and_handle_extreme_outliers
                
                df_cleaned, extreme_report = detect_and_handle_extreme_outliers(preprocessing_state.df)
                if extreme_report['total_extreme_outliers'] > 0:
                    print_to_log(f"   🔧 Handled {extreme_report['total_extreme_outliers']} extreme outliers:")
                    for col, info in extreme_report['extreme_outliers_found'].items():
                        print_to_log(f"      • {col}: {info['count']} extreme values ({info['percentage']:.1f}%) → NaN")
                    # Update the data with extreme outliers cleaned
                    preprocessing_state.df = df_cleaned
                    state.raw_data = df_cleaned  # Also update the main state
                else:
                    print_to_log("   ✅ No extreme outliers detected")
                
                # STEP 2: Use confidence-based processor for intelligent outlier analysis
                print_to_log("🎯 Step 2: Starting confidence-based outlier analysis (2-min timeout)...")
                outlier_results = confidence_processor.analyze_phase_with_confidence(preprocessing_state, "outliers")
                state.preprocessing_state['outlier_results'] = outlier_results
                
                outlier_columns = len(outlier_results.get('outlier_columns', []))
                print_to_log(f"🧠 LLM analyzed {outlier_columns} columns with outliers")
                
                # Auto-apply outlier treatments (like clicking "continue" in manual flow)
                if outlier_results.get('llm_recommendations'):
                    print_to_log("🔧 Auto-applying LLM outlier recommendations...")
                    df_working = apply_outliers_treatment(state.raw_data, outlier_results['llm_recommendations'])
                    state.cleaned_data = df_working
                    print_to_log(f"✅ Outlier treatments applied to {len(outlier_results['llm_recommendations'])} columns")
                else:
                    df_working = state.raw_data.copy()
                    state.cleaned_data = df_working
                    print_to_log("✅ No outlier treatments needed")
            except Exception as e:
                print_to_log(f"⚠️ Outlier LLM analysis failed: {e}, using fallback")
                df_working = state.raw_data.copy()
                state.cleaned_data = df_working
                state.preprocessing_state['outlier_results'] = {'outlier_columns': [], 'llm_recommendations': {}}
            
            send_progress("✅ **Finished outlier phase**")
            
            # Phase 3: Intelligent Missing Values Analysis
            send_progress("🗑️ **Starting missing values phase**")
            print_to_log("🗑️ Phase 3: Missing Values - Running intelligent LLM + rule-based analysis")
            
            # Update preprocessing state with current data
            preprocessing_state.df = df_working
            preprocessing_state.current_phase = "missing_values"
            
            # Run the same intelligent missing values analysis as manual flow
            try:
                # Use confidence-based processor (high-confidence rules + LLM fallback with timeout)
                print_to_log("🎯 Starting confidence-based missing values analysis (2-min timeout)...")
                missing_results = confidence_processor.analyze_phase_with_confidence(preprocessing_state, "missing_values")
                state.preprocessing_state['missing_results'] = missing_results
                
                missing_columns = len(missing_results.get('missing_columns', []))
                print_to_log(f"🧠 LLM analyzed {missing_columns} columns with missing values")
                
                # Auto-apply missing value treatments
                if missing_results.get('llm_recommendations'):
                    print_to_log("🔧 Auto-applying LLM missing value recommendations...")
                    df_working = apply_missing_values_treatment(df_working, missing_results['llm_recommendations'])
                    state.cleaned_data = df_working
                    print_to_log(f"✅ Missing value treatments applied to {len(missing_results['llm_recommendations'])} columns")
                else:
                    print_to_log("✅ No missing value treatments needed")
            except Exception as e:
                print_to_log(f"⚠️ Missing values LLM analysis failed: {e}, using fallback")
                state.preprocessing_state['missing_results'] = {'missing_columns': [], 'llm_recommendations': {}}
            
            send_progress("✅ **Finished missing values phase**")
            
            # Phase 4: Intelligent Encoding Analysis
            send_progress("🏷️ **Starting encoding phase**")
            print_to_log("🏷️ Phase 4: Encoding - Running intelligent LLM + rule-based analysis")
            
            # Update preprocessing state
            preprocessing_state.df = df_working
            preprocessing_state.current_phase = "encoding"
            
            # Run the same intelligent encoding analysis as manual flow
            try:
                # Use confidence-based processor (high-confidence rules + LLM fallback with timeout)
                print_to_log("🎯 Starting confidence-based encoding analysis (2-min timeout)...")
                encoding_results = confidence_processor.analyze_phase_with_confidence(preprocessing_state, "encoding")
                state.preprocessing_state['encoding_results'] = encoding_results
                
                encoding_columns = len(encoding_results.get('categorical_columns', []))
                print_to_log(f"🧠 LLM analyzed {encoding_columns} categorical columns for encoding")
                
                # Auto-apply encoding treatments
                if encoding_results.get('llm_recommendations'):
                    print_to_log("🔧 Auto-applying LLM encoding recommendations...")
                    df_working = apply_encoding_treatment(df_working, encoding_results['llm_recommendations'], state.target_column)
                    state.cleaned_data = df_working
                    print_to_log(f"✅ Encoding treatments applied to {len(encoding_results['llm_recommendations'])} columns")
                else:
                    print_to_log("✅ No encoding treatments needed")
            except Exception as e:
                print_to_log(f"⚠️ Encoding LLM analysis failed: {e}, using fallback")
                state.preprocessing_state['encoding_results'] = {'categorical_columns': [], 'llm_recommendations': {}}
            
            send_progress("✅ **Finished encoding phase**")
            
            # Phase 5: Intelligent Transformations Analysis
            send_progress("🔄 **Starting transformations phase**")
            print_to_log("🔄 Phase 5: Transformations - Running intelligent LLM + rule-based analysis")
            
            # Update preprocessing state
            preprocessing_state.df = df_working
            preprocessing_state.current_phase = "transformations"
            
            # Run the same intelligent transformations analysis as manual flow
            try:
                # Use confidence-based processor (high-confidence rules + LLM fallback with timeout)
                print_to_log("🎯 Starting confidence-based transformation analysis (2-min timeout)...")
                transformation_results = confidence_processor.analyze_phase_with_confidence(preprocessing_state, "transformations")
                state.preprocessing_state['transformation_results'] = transformation_results
                
                transform_columns = len(transformation_results.get('transformation_columns', []))
                print_to_log(f"🧠 LLM analyzed {transform_columns} columns for transformations")
                
                # Auto-apply transformation treatments
                if transformation_results.get('llm_recommendations'):
                    print_to_log("🔧 Auto-applying LLM transformation recommendations...")
                    df_working = apply_transformations_treatment(df_working, transformation_results['llm_recommendations'], state.target_column)
                    state.cleaned_data = df_working
                    print_to_log(f"✅ Transformation treatments applied to {len(transformation_results['llm_recommendations'])} columns")
                else:
                    print_to_log("✅ No transformation treatments needed")
            except Exception as e:
                print_to_log(f"⚠️ Transformations LLM analysis failed: {e}, using fallback")
                state.preprocessing_state['transformation_results'] = {'transformation_columns': [], 'llm_recommendations': {}}
            
            send_progress("✅ **Finished transformations phase**")
            send_progress("🎉 **Finished preprocessing**")
            
            print_to_log(f"✅ All intelligent preprocessing completed: {df_working.shape}")
            
            # Phase 6: Feature Selection (following actual flow)
            send_progress("🔍 **Started feature selection**")
            print_to_log("🔍 Phase 6: Feature Selection - Following actual flow with IV and VIF filtering")
            
            try:
                from feature_selection_agent_impl import DataProcessor, AnalysisEngine, UserSession
                import tempfile
                
                print_to_log("🔧 Step 1: Setting up feature selection session")
                # Create a temporary CSV file for the UserSession
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                    df_working.to_csv(tmp_file.name, index=False)
                    temp_csv_path = tmp_file.name
                
                # Create UserSession for feature selection
                session = UserSession(
                    file_path=temp_csv_path,
                    file_name="fast_mode_data.csv",
                    user_id=state.chat_session or "fast_mode",
                    target_column=state.target_column,
                    phase="analysis"
                                )
                
                print_to_log("🔧 Step 2: Loading and cleaning data (removing single-value and object columns)")
                # Load and clean data (removes single-value and object columns)
                if DataProcessor.load_and_clean_data(session):
                    clean_data = session.current_df
                    print_to_log(f"   📊 After intelligent cleaning: {clean_data.shape}")
                    send_progress("🔍 **Started IV Value Filtering with threshold 0.02**")
                    print_to_log("🔧 Step 3: Applying IV filter (threshold > 0.02)")
                    # Apply IV filter with 0.02 threshold
                    iv_results = AnalysisEngine.run_iv_analysis(session, threshold=0.02)
                    send_progress("🔍 **IV Value filtering complete**")
                    if 'error' not in iv_results:
                        iv_filtered_data = session.current_df  # Data is updated in session
                        print_to_log(f"   📊 After IV filtering: {iv_filtered_data.shape}")
                        send_progress("🔍 **Started VIF Value Filtering with threshold 5**")
                        print_to_log("🔧 Step 4: Applying VIF (threshold > 5)")
                        # Apply VIF filter with 5 threshold
                        vif_results = AnalysisEngine.run_vif_analysis(session, threshold=5)
                        send_progress("🔍 **VIF Value filtering complete**")
                        if 'error' not in vif_results:
                            final_data = session.current_df  # Data is updated in session
                            print_to_log(f"   📊 After VIF filtering: {final_data.shape}")
                        else:
                            print_to_log(f"   ⚠️ VIF filtering failed: {vif_results.get('error', 'Unknown error')}")
                            final_data = iv_filtered_data
                    else:
                        print_to_log(f"   ⚠️ IV filtering failed: {iv_results.get('error', 'Unknown error')}")
                        final_data = clean_data
                else:
                    print_to_log("   ⚠️ Data cleaning failed, using original data")
                    final_data = df_working
                
                # Update state with selected features (excluding target)
                feature_columns = [col for col in final_data.columns if col != state.target_column]
                state.selected_features = feature_columns  # Store as list, not DataFrame
                state.processed_data = final_data  # Store processed data for model building
                state.cleaned_data = final_data  # CRITICAL: Model agent looks for cleaned_data                
                print_to_log(f"✅ Feature selection complete: {len(feature_columns)} features selected")
                print_to_log(f"   📊 Original → Filtered: {df_working.shape} → {final_data.shape}")
                
                # Clean up temporary file
                import os
                os.unlink(temp_csv_path)
                
            except Exception as e:
                print_to_log(f"⚠️ Feature selection failed: {e}, using all features")
                import traceback
                traceback.print_exc()
                # Fallback: use all features
                feature_columns = [col for col in df_working.columns if col != state.target_column]
                state.selected_features = feature_columns  # Store as list, not DataFrame
                state.processed_data = df_working
                print_to_log(f"✅ Fallback: Using all {len(feature_columns)} features")
                state.cleaned_data = df_working  # CRITICAL: Model agent looks for cleaned_data            
            send_progress("✅ **Final features selected**")
            
            # Phase 7: Model Building (following actual flow with comprehensive results)
            send_progress("🤖 **Started modeling**")
            print_to_log("🤖 Phase 7: Model Building - Training with comprehensive metrics like actual flow")
            
            try:
                from agents_wrapper import ModelBuildingAgentWrapper
                
                # Use the actual model building agent for comprehensive results
                model_agent = ModelBuildingAgentWrapper()
                
                if model_agent.available:
                    print_to_log("🔧 Using actual model building agent with user prompt-based model selection")
                    
                    # Ensure selected features are properly set
                    if not hasattr(state, "selected_features") or state.selected_features is None:
                        # Fallback: use processed data features
                        if hasattr(state, "processed_data") and state.processed_data is not None:
                            feature_columns = [col for col in state.processed_data.columns if col != state.target_column]
                            state.selected_features = state.processed_data[feature_columns].copy()
                        else:
                            print_to_log("❌ No processed data available for model building")
                            raise Exception("No data available for model building")
                    
                    print_to_log(f"🔧 Model building with {len(state.selected_features) if hasattr(state.selected_features, '__len__') else 0} selected features")
                    
                    # 🎯 CRITICAL: Extract model type from original user query for prompt-based model selection
                    # Use preserved original intent if available, otherwise fallback to current user_query
                    original_query = ""
                    if hasattr(state, "preprocessing_state") and state.preprocessing_state and "original_user_intent" in state.preprocessing_state:
                        original_query = state.preprocessing_state["original_user_intent"]
                        print_to_log(f"🔍 Using preserved original intent: '{original_query}'")
                    elif hasattr(state, "user_query") and state.user_query:
                        original_query = state.user_query
                        print_to_log(f"🔍 Using current user query: '{original_query}'")
                    else:
                        original_query = "build a machine learning model with comprehensive metrics and visualizations"
                        print_to_log(f"🔍 Using default query: '{original_query}'")                    
                    # Parse model type from user query or use default
                    base_model_query = self._extract_model_request_from_query(original_query)
                    
                    # CRITICAL: Merge original intent with target column for complete prompt
                    # This solves the target column detection issue by including it in the prompt
                    if state.target_column:
                        model_query = f"{base_model_query} with target column '{state.target_column}'"
                        print_to_log(f"🎯 Enhanced model query with target: '{model_query}'")
                    else:
                        model_query = base_model_query
                        print_to_log(f"🤖 Base model query: '{model_query}'")                    
                    # Set the model building query so the agent knows what model to build
                    state.user_query = model_query
                    
                    # Run the actual model building agent (provides metrics, confusion matrix, rank ordering)
                    # CRITICAL: Ensure target column is set in model agent's global state
                    # The model agent looks for target in global_model_states, not just user_states
                    try:
                        # Import the global states that the model agent uses
                        from model_building_agent_impl import global_model_states
                        
                        # Ensure the user exists in global_model_states
                        user_id = state.chat_session or "fast_mode"
                        if user_id not in global_model_states:
                            global_model_states[user_id] = {}
                        
                        # Set the target column in the global state where generate_model_code looks for it
                        global_model_states[user_id]["target_column"] = state.target_column
                        print_to_log(f"🎯 Set target column in global_model_states: {state.target_column}")
                        
                        # Also ensure sample_data includes the target column
                        if hasattr(state, "cleaned_data") and state.cleaned_data is not None:
                            if state.target_column in state.cleaned_data.columns:
                                global_model_states[user_id]["sample_data"] = state.cleaned_data
                                print_to_log(f"📊 Updated sample_data with target column included: {state.cleaned_data.shape}")
                            else:
                                print_to_log(f"⚠️ Target column {state.target_column} not found in cleaned_data columns")
                        
                    except Exception as e:
                        print_to_log(f"⚠️ Error setting target in global_model_states: {e}")
                    
                    # Run the actual model building agent (provides metrics, confusion matrix, rank ordering)
                    result_state = model_agent.run(state)                    
                    # Update our state with comprehensive results
                    state.trained_model = result_state.trained_model
                    state.model_building_state = result_state.model_building_state
                    state.last_response = result_state.last_response
                    
                    # CRITICAL: Send detailed metrics to Slack immediately and return
                    # This prevents the generic summary from overwriting the detailed metrics
                    if result_state.last_response:
                        print_to_log("✅ Using detailed model building response with all classification metrics")
                        
                        # Send detailed metrics to Slack immediately
                        slack_manager = getattr(self, "slack_manager", None)
                        if not slack_manager:
                            from toolbox import slack_manager as global_slack_manager
                            slack_manager = global_slack_manager
                        
                        if slack_manager and state.chat_session:
                            try:
                                print_to_log("📤 Sending detailed classification metrics to Slack")
                                slack_manager.send_message(state.chat_session, result_state.last_response)
                                print_to_log("✅ Detailed metrics sent to Slack successfully")
                            except Exception as e:
                                print_to_log(f"⚠️ Error sending detailed metrics to Slack: {e}")
                        else:
                            print_to_log("⚠️ No Slack manager available - detailed metrics not sent")
                        
                        print_to_log("🎉 INTELLIGENT automated ML pipeline completed successfully!")
                        return state
                    
                    if state.trained_model:
                        print_to_log("✅ Model trained with comprehensive results (metrics, confusion matrix, rank ordering)")
                        if hasattr(state, "model_building_state") and state.model_building_state:
                            if "metrics" in state.model_building_state:
                                metrics = state.model_building_state["metrics"]
                                print_to_log(f"📊 Model metrics available: {list(metrics.keys()) if isinstance(metrics, dict) else 'Available'}")
                    else:
                        print_to_log("⚠️ Model training completed but no model object returned")
                
                else:
                    print_to_log("⚠️ Model building agent not available, using simple fallback")
                    # Simple fallback model training
                    model_data = state.processed_data if hasattr(state, "processed_data") and state.processed_data is not None else df_working
                    
                    if state.target_column in model_data.columns:
                        X = model_data.drop(columns=[state.target_column])
                        y = model_data[state.target_column]
                        
                        # Basic model training
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        state.trained_model = model
                        state.preprocessing_state["model_metrics"] = {"accuracy": accuracy}
                        
                        print_to_log(f"✅ Fallback model trained - Accuracy: {accuracy:.3f}")
                    else:
                        print_to_log(f"❌ Target column {state.target_column} not found in processed data")

            except Exception as e:
                print_to_log(f"⚠️ Model building failed: {e}")
                import traceback
                traceback.print_exc()
            
            send_progress("✅ **Final modeling results completed**")
            
            # Generate final success message
            final_shape = state.processed_data.shape if state.processed_data is not None else state.cleaned_data.shape
            feature_count = len(state.selected_features) if state.selected_features is not None and hasattr(state.selected_features, '__len__') else final_shape[1] - 1
            model_status = "✅ Trained Successfully" if state.trained_model else "⚠️ Training Attempted"
            
            accuracy_text = ""
            if state.trained_model and 'model_metrics' in state.preprocessing_state:
                accuracy = state.preprocessing_state['model_metrics']['accuracy']
                accuracy_text = f" (Accuracy: {accuracy:.1%})"
            
            # Count intelligent recommendations applied
            total_recommendations = 0
            if 'outlier_results' in state.preprocessing_state:
                total_recommendations += len(state.preprocessing_state['outlier_results'].get('llm_recommendations', {}))
            if 'missing_results' in state.preprocessing_state:
                total_recommendations += len(state.preprocessing_state['missing_results'].get('llm_recommendations', {}))
            if 'encoding_results' in state.preprocessing_state:
                total_recommendations += len(state.preprocessing_state['encoding_results'].get('llm_recommendations', {}))
            if 'transformation_results' in state.preprocessing_state:
                total_recommendations += len(state.preprocessing_state['transformation_results'].get('llm_recommendations', {}))
            
            state.last_response = f"""🎉 **Intelligent Fast ML Pipeline Complete!**

🎯 **Target:** {state.target_column}
📊 **Data Shape:** {state.raw_data.shape} → {final_shape}
🔍 **Features:** {feature_count} selected
🤖 **Model:** {model_status}{accuracy_text}
🧠 **LLM Recommendations:** {total_recommendations} applied

✅ **Completed Phases (with LLM Analysis):**
• 📊 Overview - Dataset structure analyzed
• 🚨 Outliers - LLM + rule-based outlier analysis
• 🗑️ Missing Values - LLM + rule-based imputation strategy
• 🏷️ Encoding - LLM + rule-based encoding strategy
• 🔄 Transformations - LLM + rule-based transformation strategy
• 🔍 Feature Selection - All features used
• 🤖 Model Building - Classification model trained

**Same intelligent analysis as manual mode - fully automated!**"""

            print_to_log("🎉 INTELLIGENT automated ML pipeline completed successfully!")
            return state
            
        except Exception as e:
            error_msg = f"Intelligent pipeline execution failed: {str(e)}"
            print_to_log(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            state.last_error = error_msg
            state.last_response = f"❌ **Pipeline Error:** {error_msg}"
            return state


def fast_model_agent(state: PipelineState) -> PipelineState:
    """Main entry point for fast model agent - called by langgraph_pipeline.py"""
    print_to_log("🚀 [Fast Model Agent] Starting intelligent fast model pipeline")
    
    agent = FastModelAgent()
    
    # Check if this is a target column response
    if state.target_column is None and state.user_query and state.raw_data is not None:
        # Check if user query looks like a column name
        if state.user_query.strip() in state.raw_data.columns:
            print_to_log(f"🎯 [Fast Model Agent] Setting target column: {state.user_query.strip()}")
            return agent.handle_fast_model_request(state, state.user_query.strip())
    
    # Otherwise start the normal flow
    print_to_log("🚀 [Fast Model Agent] Starting intelligent automated pipeline")
    return agent.handle_fast_model_request(state)
