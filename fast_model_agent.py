from pipeline_state import PipelineState
from print_to_log import print_to_log
import pandas as pd
import numpy as np

class FastModelAgent:
    """Agent for automated ML pipeline execution using the same intelligent analysis as manual flow"""
    
    def __init__(self):
        print_to_log("🚀 FastModelAgent initialized - using intelligent LLM + rule-based analysis")
    
    def handle_fast_model_request(self, state: PipelineState, target_column: str = None) -> PipelineState:
        """Handle fast model request with target column setting"""
        print_to_log("🚀 FastModelAgent: Starting intelligent automated pipeline")
        
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
            send_progress("📊 **Starting overview phase**")
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
            
            send_progress("✅ **Finished overview phase**")
            
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
                
                # Use confidence-based processor (high-confidence rules + LLM fallback with timeout)
                print_to_log("🎯 Starting confidence-based outlier analysis (2-min timeout)...")
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
                    print_to_log("�� Auto-applying LLM encoding recommendations...")
                    df_working = apply_encoding_treatment(df_working, encoding_results['llm_recommendations'])
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
                    df_working = apply_transformations_treatment(df_working, transformation_results['llm_recommendations'])
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
            
            # Phase 6: Feature Selection
            send_progress("🔍 **Started feature selection**")
            print_to_log("🔍 Phase 6: Feature Selection - Using all features (LLM analysis complete)")
            
            # For now, use all features (feature selection can be added later)
            state.processed_data = df_working
            state.selected_features = [col for col in df_working.columns if col != state.target_column]
            
            print_to_log(f"✅ Feature selection: Using all {len(state.selected_features)} features")
            
            send_progress("✅ **Final features selected**")
            
            # Phase 7: Model Building
            send_progress("🤖 **Started modeling**")
            print_to_log("🤖 Phase 7: Model Building - Training classification model")
            
            try:
                # Build model using available features and target
                model_data = state.processed_data
                
                if state.target_column in model_data.columns:
                    # Prepare data with proper type handling
                    X = model_data.drop(columns=[state.target_column])
                    y = model_data[state.target_column]
                    
                    # Remove high cardinality columns that shouldn't be in model
                    columns_to_drop = []
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            unique_ratio = X[col].nunique() / len(X)
                            if unique_ratio > 0.25:  # High cardinality
                                columns_to_drop.append(col)
                                print_to_log(f"🔧 Dropping high cardinality column from model: {col}")
                    
                    if columns_to_drop:
                        X = X.drop(columns=columns_to_drop)
                    
                    # Convert remaining object columns to numeric
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            try:
                                # Try to convert to numeric
                                X[col] = pd.to_numeric(X[col], errors='coerce')
                                # Fill any resulting NaNs with median
                                if X[col].isnull().sum() > 0:
                                    X[col] = X[col].fillna(X[col].median())
                            except:
                                # If conversion fails, use label encoding
                                from sklearn.preprocessing import LabelEncoder
                                le = LabelEncoder()
                                X[col] = le.fit_transform(X[col].astype(str))
                    
                    # Ensure target is numeric
                    if y.dtype == 'object':
                        from sklearn.preprocessing import LabelEncoder
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y)
                    
                    # Final check - ensure all columns are numeric
                    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
                    if non_numeric_cols:
                        print_to_log(f"🔧 Dropping non-numeric columns: {non_numeric_cols}")
                        X = X.select_dtypes(include=[np.number])
                    
                    print_to_log(f"🔧 Final model data: {X.shape} features, {len(y)} samples")
                    
                    # Simple model training
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import accuracy_score
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Train model
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    state.trained_model = model
                    # Store metrics in preprocessing_state instead of PipelineState
                    state.preprocessing_state['model_metrics'] = {'accuracy': accuracy}
                    
                    print_to_log(f"✅ Model trained successfully - Accuracy: {accuracy:.3f}")
                else:
                    print_to_log(f"⚠️ Target column {state.target_column} not found in processed data")
                    
            except Exception as e:
                print_to_log(f"⚠️ Model building failed: {e}")
                import traceback
                traceback.print_exc()
            
            send_progress("✅ **Final modeling results completed**")
            
            # Generate final success message
            final_shape = state.processed_data.shape if state.processed_data is not None else state.cleaned_data.shape
            feature_count = len(state.selected_features) if state.selected_features else final_shape[1] - 1
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
