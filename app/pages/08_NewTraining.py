import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.state import AppState
from core.exports import export_training_data
from core.models import BoundaryModel, train_boundary_model, evaluate_model_performance
from core.io import load_spans, load_labels

st.set_page_config(page_title="Training", page_icon="🚀", layout="wide")

AppState.init()

st.title("🚀 2 Model Training")

st.markdown("""
Train boundary detection models from your labeled documents. The trained model will be available 
for predictions in the **Label** page to speed up your annotation workflow.
""")

# Initialize session state for training
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None

def get_available_datasets():
    """Get list of available training datasets"""
    exports_dir = Path("exports")
    if not exports_dir.exists():
        return []
    
    datasets = []
    for file_path in exports_dir.glob("*.parquet"):
        # Skip train_ and test_ prefixed files
        if file_path.name.startswith('train_') or file_path.name.startswith('test_'):
            continue
        
        datasets.append({
            'name': file_path.stem,
            'path': file_path,
            'size': file_path.stat().st_size / (1024 * 1024),  # MB
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
        })
    
    return datasets

def perform_document_split(df, test_size=0.25, random_state=42):
    """
    Split data at document level for better generalization testing.
    
    Args:
        df: DataFrame with 'doc_id' column
        test_size: Fraction of documents to use for testing (default 0.25)
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, test_df: DataFrames split at document level
    """
    if 'doc_id' not in df.columns:
        st.warning("No 'doc_id' column found. Using random split instead.")
        # Fallback to random split
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['y_boundary'])
    
    # Get unique documents
    unique_docs = df['doc_id'].unique()
    n_docs = len(unique_docs)
    
    st.info(f"Found {n_docs} unique documents in dataset")
    
    if n_docs < 2:
        st.warning("Only 1 document found. Using random split instead of document-level split.")
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['y_boundary'])
    
    # Calculate number of test documents
    n_test_docs = max(1, int(n_docs * test_size))
    n_train_docs = n_docs - n_test_docs
    
    # Randomly select test documents
    np.random.seed(random_state)
    test_docs = np.random.choice(unique_docs, size=n_test_docs, replace=False)
    train_docs = [doc for doc in unique_docs if doc not in test_docs]
    
    # Split data
    train_df = df[df['doc_id'].isin(train_docs)].copy()
    test_df = df[df['doc_id'].isin(test_docs)].copy()
    
    # Display split information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Documents", n_train_docs)
        st.caption(f"Docs: {', '.join(train_docs[:3])}{'...' if len(train_docs) > 3 else ''}")
    with col2:
        st.metric("Test Documents", n_test_docs)
        st.caption(f"Docs: {', '.join(test_docs[:3])}{'...' if len(test_docs) > 3 else ''}")
    with col3:
        train_pct = len(train_df) / len(df) * 100
        st.metric("Train/Test Split", f"{train_pct:.0f}% / {100-train_pct:.0f}%")
    
    return train_df, test_df

def get_efficient_model_params(n_samples, n_features):
    """
    Get efficient model parameters based on dataset size.
    
    Args:
        n_samples: Number of training samples
        n_features: Number of features
    
    Returns:
        dict: Optimized parameters for the model
    """
    # Base parameters
    params = {
        'random_state': 42,
        'calibrate': True
    }
    
    # Adjust based on dataset size
    if n_samples < 1000:
        # Small dataset - prevent overfitting
        params.update({
            'n_estimators': 100,
            'max_depth': 10,
            #'min_samples_split': 20,
            #'min_samples_leaf': 10,
            #'max_features': 'sqrt'
        })
    elif n_samples < 5000:
        # Medium dataset
        params.update({
            'n_estimators': 150,
            'max_depth': 15,
            #'min_samples_split': 10,
            #'min_samples_leaf': 5,
            #'max_features': 'sqrt'
        })
    else:
        # Large dataset
        params.update({
            'n_estimators': 200,
            'max_depth': 20,
            #'min_samples_split': 5,
            #'min_samples_leaf': 2,
            #'max_features': 'sqrt'
        })
    
    # Further optimization for very small datasets
    if n_samples < 500:
        params['n_estimators'] = 50
        params['max_depth'] = 8
    
    return params

def train_model_with_optional_split(data_path: Path, model_name: str, test_size: float = 0.25, use_all_data: bool = False):
    """
    Train model with optional document-level train/test split.
    
    Args:
        data_path: Path to the parquet file
        model_name: Name for the saved model
        test_size: Fraction of documents for testing (default 0.25)
        use_all_data: If True, train on all data without splitting
    
    Returns:
        dict: Training results and metrics
    """
    progress_bar = st.progress(0, text="Loading data...")
    
    try:
        # Step 1: Load data
        progress_bar.progress(10, text="Loading data...")
        df = pd.read_parquet(data_path)
        
        # Verify data
        if 'y_boundary' not in df.columns:
            raise ValueError("Training data missing 'y_boundary' column")
        
        # Step 2: Split or use all data
        if use_all_data:
            progress_bar.progress(20, text="Using all data for training (no split)...")
            train_df = df.copy()
            test_df = None
            
            # Display info
            st.info(f"🎯 Training on **all {len(df):,} examples** from **{len(df['doc_id'].unique()) if 'doc_id' in df.columns else 'N/A'} documents** (no test split)")
        else:
            progress_bar.progress(20, text="Performing document-level split...")
            train_df, test_df = perform_document_split(df, test_size=test_size)
        
        # Get feature count (excluding target and metadata columns)
        feature_cols = [col for col in train_df.columns 
                       if col not in ['y_boundary', 'doc_id', 'span_id', 'page_number']]
        n_features = len(feature_cols)
        
        # Step 3: Get optimized model parameters
        progress_bar.progress(30, text="Optimizing model parameters...")
        model_params = get_efficient_model_params(len(train_df), n_features)
        
        # Display model configuration
        with st.expander("🔧 Model Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Dataset Statistics:**")
                st.write(f"- Training samples: {len(train_df):,}")
                if test_df is not None:
                    st.write(f"- Test samples: {len(test_df):,}")
                else:
                    st.write(f"- Test samples: N/A (using all data)")
                st.write(f"- Features: {n_features}")
                st.write(f"- Positive rate (train): {train_df['y_boundary'].mean():.2%}")
                if test_df is not None:
                    st.write(f"- Positive rate (test): {test_df['y_boundary'].mean():.2%}")
            with col2:
                st.write("**Model Parameters:**")
                for param, value in model_params.items():
                    if param not in ['random_state', 'n_jobs', 'calibrate']:
                        st.write(f"- {param}: {value}")
        
        # Step 4: Initialize model with optimized parameters
        progress_bar.progress(40, text="Initializing model...")
        model = BoundaryModel(**model_params)
        
        # Step 5: Train model
        progress_bar.progress(50, text="Training model...")
        training_metrics = model.train(train_df, test_df)
        
        # Step 6: Evaluate model (only if test_df exists)
        if test_df is not None:
            progress_bar.progress(70, text="Evaluating model performance...")
            test_metrics = evaluate_model_performance(model, test_df)
        else:
            test_metrics = None
        
        # Step 7: Cross-validation for robustness (optional and only if not using all data)
        cv_scores = None
        if not use_all_data and 'doc_id' in df.columns and len(df['doc_id'].unique()) >= 4:
            progress_bar.progress(80, text="Performing cross-validation...")
            cv_scores = perform_cross_validation(df, model_params)
        
        # Step 8: Save model
        progress_bar.progress(90, text="Saving model...")
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{model_name}.pkl"
        model.save(model_path)
        
        # Save metadata
        metadata_path = model_dir / f"{model_name}.json"
        metadata = {
            'model_name': model_name,
            'model_params': model_params,
            'train_metrics': training_metrics.get('train_metrics', {}),
            'test_metrics': test_metrics if test_metrics else {},
            'cv_scores': cv_scores,
            'n_train_samples': len(train_df),
            'n_test_samples': len(test_df) if test_df is not None else 0,
            'n_train_docs': len(train_df['doc_id'].unique()) if 'doc_id' in train_df.columns else 'N/A',
            'n_test_docs': len(test_df['doc_id'].unique()) if test_df is not None and 'doc_id' in test_df.columns else 'N/A',
            'feature_count': n_features,
            'feature_importance': training_metrics.get('feature_importance', [])[:20],  # Top 20
            'trained_on_all_data': use_all_data,
            'created_at': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        progress_bar.progress(100, text="Training complete!")
        
        # Store results
        results = {
            **metadata,
            'model_path': str(model_path)
        }
        
        return results
        
    except Exception as e:
        progress_bar.empty()
        raise e

def perform_cross_validation(df, model_params, n_splits=4):
    """
    Perform document-level cross-validation.
    
    Args:
        df: DataFrame with features and targets
        model_params: Model parameters to use
        n_splits: Number of CV folds
    
    Returns:
        dict: Cross-validation scores
    """
    if 'doc_id' not in df.columns:
        return None
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # Use GroupKFold for document-level splits
    gkf = GroupKFold(n_splits=min(n_splits, len(df['doc_id'].unique())))
    
    auc_scores = []
    acc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['y_boundary'], groups=df['doc_id'])):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        
        # Train model on fold
        model = BoundaryModel(**model_params)
        model.train(train_fold, test_df=None)  # No test set for CV folds
        
        # Evaluate
        val_probs = model.predict_proba(val_fold)
        val_preds = model.predict(val_fold)
        
        # If predict_proba returns 2D array (probabilities for both classes), use only positive class
        if len(val_probs.shape) > 1 and val_probs.shape[1] == 2:
            val_probs = val_probs[:, 1]  # Use probabilities for positive class
        
        auc = roc_auc_score(val_fold['y_boundary'], val_probs)
        acc = accuracy_score(val_fold['y_boundary'], val_preds)
        
        auc_scores.append(auc)
        acc_scores.append(acc)
    
    return {
        'auc_mean': np.mean(auc_scores),
        'auc_std': np.std(auc_scores),
        'acc_mean': np.mean(acc_scores),
        'acc_std': np.std(acc_scores),
        'fold_aucs': auc_scores,
        'fold_accs': acc_scores
    }

# Main UI starts here
st.header("📊 Select Training Data")

datasets = get_available_datasets()

if not datasets:
    st.warning("No training datasets found. Please export data from the Export page first.")
    st.stop()

# Dataset selection
dataset_options = {d['name']: d for d in datasets}
selected_dataset_name = st.selectbox(
    "Select training dataset",
    options=list(dataset_options.keys()),
    format_func=lambda x: f"{x} ({dataset_options[x]['size']:.1f} MB)"
)

selected_dataset = dataset_options[selected_dataset_name]

# Display dataset info
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset", selected_dataset_name)

with col2:
    st.metric("Size", f"{selected_dataset['size']:.1f} MB")

with col3:
    st.metric("Modified", selected_dataset['modified'].strftime('%Y-%m-%d %H:%M'))

# Load and show dataset statistics
if st.button("📊 Analyze Dataset"):
    df = pd.read_parquet(selected_dataset['path'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Examples", f"{len(df):,}")
    
    with col2:
        st.metric("Positive Boundaries", f"{df['y_boundary'].sum():,}")
    
    with col3:
        st.metric("Negative Examples", f"{(len(df) - df['y_boundary'].sum()):,}")
    
    with col4:
        boundary_rate = df['y_boundary'].mean() * 100
        st.metric("Boundary Rate", f"{boundary_rate:.1f}%")
    
    # Show document distribution
    if 'doc_id' in df.columns:
        doc_counts = df['doc_id'].value_counts()
        
        st.write(f"**📚 Documents in dataset:** {len(doc_counts)}")
        
        # Document statistics
        doc_stats = pd.DataFrame({
            'Document': doc_counts.index,
            'Examples': doc_counts.values,
            'Percentage': (doc_counts.values / len(df) * 100).round(1)
        })
        
        # Create visualization
        fig = px.bar(
            doc_stats, 
            x='Document', 
            y='Examples',
            title="Examples per Document",
            text='Examples'
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Detailed Document Distribution"):
            st.dataframe(doc_stats, use_container_width=True)

# Training Configuration
st.header("⚙️ Training Configuration")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    model_name = st.text_input(
        "Model Name",
        value=f"boundary_model_{datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Name for the trained model"
    )

with col2:
    use_all_data = st.checkbox(
        "Train on all data",
        value=False,
        help="Train on entire dataset without train/test split. Use this for final production models after validation."
    )

with col3:
    # Only show test size slider if not using all data
    if not use_all_data:
        test_size = st.slider(
            "Test Document Fraction",
            min_value=0.1,
            max_value=0.4,
            value=0.25,
            step=0.05,
            help="Fraction of documents to use for testing (document-level split)"
        )
    else:
        test_size = 0.25  # Default value, won't be used
        st.info("ℹ️ No test split - using 100% of data for training")

# Training Section
st.header("🚀 Train Model")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    train_button = st.button(
        "🚀 Start Training", 
        type="primary", 
        use_container_width=True,
        help="Train model with selected configuration"
    )

with col2:
    if use_all_data:
        st.warning(f"""
        **⚠️ No Validation:**
        - Training on all data
        - No test metrics
        - Use after validation
        """)
    else:
        st.info(f"""
        **Split Strategy:**
        - Document-level split
        - Test on unseen docs
        - Better generalization
        """)

with col3:
    st.info(f"""
    **Model Type:**
    - Random Forest
    - Auto-optimized params
    - Probability calibration
    """)

if train_button:
    if not model_name.strip():
        st.error("Please enter a model name")
    else:
        with st.spinner("Training model..."):
            try:
                results = train_model_with_optional_split(
                    data_path=selected_dataset['path'],
                    model_name=model_name.strip(),
                    test_size=test_size if not use_all_data else 0.25,
                    use_all_data=use_all_data
                )
                
                st.session_state.training_results = results
                st.success("✅ Model training completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)

# Training Results
if st.session_state.training_results:
    st.header("📈 Training Results")
    
    results = st.session_state.training_results
    
    # Check if trained on all data
    trained_on_all = results.get('trained_on_all_data', False)
    
    if trained_on_all:
        st.warning("🎯 **Model trained on all available data** - No test metrics available")
    
    # Main metrics
    if trained_on_all:
        # Show only training metrics when no split
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_auc = results.get('train_metrics', {}).get('auc_roc', 0)
            st.metric("Training AUC", f"{train_auc:.3f}")
        
        with col2:
            st.metric("Features Used", results.get('feature_count', 'N/A'))
        
        with col3:
            st.metric("Total Samples", f"{results.get('n_train_samples', 0):,}")
    else:
        # Show train and test metrics when split is used
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            train_auc = results.get('train_metrics', {}).get('auc_roc', 0)
            st.metric("Training AUC", f"{train_auc:.3f}")
        
        with col2:
            test_auc = results.get('test_metrics', {}).get('auc_roc', 0)
            delta = test_auc - train_auc if train_auc > 0 else 0
            st.metric("Test AUC", f"{test_auc:.3f}", delta=f"{delta:+.3f}")
        
        with col3:
            st.metric("Features Used", results.get('feature_count', 'N/A'))
        
        with col4:
            st.metric("Test Docs", results.get('n_test_docs', 'N/A'))
    
    # Cross-validation results (if available - only shown when not training on all data)
    if results.get('cv_scores') and not trained_on_all:
        st.subheader("🔄 Cross-Validation Results")
        cv_scores = results['cv_scores']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CV AUC", f"{cv_scores['auc_mean']:.3f} ± {cv_scores['auc_std']:.3f}")
        with col2:
            st.metric("CV Accuracy", f"{cv_scores['acc_mean']:.3f} ± {cv_scores['acc_std']:.3f}")
        
        # Show fold scores
        with st.expander("Fold-wise Scores"):
            fold_df = pd.DataFrame({
                'Fold': range(1, len(cv_scores['fold_aucs']) + 1),
                'AUC': cv_scores['fold_aucs'],
                'Accuracy': cv_scores['fold_accs']
            })
            st.dataframe(fold_df, use_container_width=True)
    
    # Performance interpretation (only if we have test metrics)
    if not trained_on_all:
        test_auc = results.get('test_metrics', {}).get('auc_roc', 0)
        train_auc = results.get('train_metrics', {}).get('auc_roc', 0)
        
        if test_auc > 0:
            if test_auc >= 0.85:
                st.success("🎯 **Excellent Performance!** Model is ready for production use.")
            elif test_auc >= 0.75:
                st.info("✅ **Good Performance!** Model is usable but could benefit from more labeled data.")
            elif test_auc >= 0.65:
                st.warning("⚠️ **Fair Performance.** Consider labeling more documents for better results.")
            else:
                st.error("❌ **Poor Performance.** Need more diverse labeled examples.")
            
            # Check for overfitting
            if train_auc - test_auc > 0.1:
                st.warning("⚠️ **Potential Overfitting Detected!** Large gap between train and test AUC. Consider labeling more diverse documents.")
    else:
        # Show guidance for all-data training
        train_auc = results.get('train_metrics', {}).get('auc_roc', 0)
        if train_auc >= 0.9:
            st.info("📊 High training AUC. Model has learned the training data well. Consider validating on new documents before production use.")
        else:
            st.warning("📊 Moderate training AUC. Model may need more diverse training examples.")
    
    # Feature importance chart
    if results.get('feature_importance'):
        st.subheader("🎯 Top Feature Importance")
        
        features = results['feature_importance'][:15]  # Top 15
        feature_names = [f[0] for f in features]
        importances = [f[1] for f in features]
        
        fig = px.bar(
            x=importances,
            y=feature_names,
            orientation='h',
            title="Top 15 Most Important Features",
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model info
    st.subheader("📋 Model Information")
    with st.expander("View Full Model Details"):
        # Clean up display
        display_info = {
            'model_name': results['model_name'],
            'model_path': results['model_path'],
            'trained_on_all_data': results.get('trained_on_all_data', False),
            'training_samples': results.get('n_train_samples', 'N/A'),
            'test_samples': results.get('n_test_samples', 'N/A') if not trained_on_all else 'N/A (no split)',
            'training_documents': results.get('n_train_docs', 'N/A'),
            'test_documents': results.get('n_test_docs', 'N/A') if not trained_on_all else 'N/A (no split)',
            'created_at': results['created_at']
        }
        st.json(display_info)

# Existing Models Section
st.header("📚 Existing Models")

models_dir = Path("models")
if models_dir.exists():
    model_files = list(models_dir.glob("*.pkl"))
    
    if model_files:
        # Sort by modification time
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for model_file in model_files:
            with st.expander(f"🤖 {model_file.stem}"):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    # Load and display metadata
                    metadata_file = model_file.with_suffix('.json')
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            train_metrics = metadata.get('train_metrics', {})
                            test_metrics = metadata.get('test_metrics', {})
                            trained_on_all = metadata.get('trained_on_all_data', False)
                            
                            # Show training type
                            if trained_on_all:
                                st.write("🎯 **Training Type**: All data (no split)")
                            else:
                                st.write("📊 **Training Type**: Train/test split")
                            
                            # Show metrics
                            st.write(f"**Training AUC**: {train_metrics.get('auc_roc', 'N/A')}")
                            
                            if not trained_on_all and test_metrics:
                                st.write(f"**Test AUC**: {test_metrics.get('auc_roc', 'N/A')}")
                                
                                if metadata.get('cv_scores'):
                                    cv = metadata['cv_scores']
                                    st.write(f"**CV AUC**: {cv.get('auc_mean', 0):.3f} ± {cv.get('auc_std', 0):.3f}")
                            
                            st.write(f"**Features**: {metadata.get('feature_count', 'N/A')}")
                            
                            if trained_on_all:
                                st.write(f"**Training Samples**: {metadata.get('n_train_samples', 'N/A')}")
                                st.write(f"**Training Docs**: {metadata.get('n_train_docs', 'N/A')}")
                            else:
                                st.write(f"**Train/Test Docs**: {metadata.get('n_train_docs', 'N/A')} / {metadata.get('n_test_docs', 'N/A')}")
                            
                        except Exception as e:
                            st.write("Metadata not available")
                    else:
                        st.write("No metadata available")
                
                with col2:
                    # File info
                    file_size = model_file.stat().st_size / (1024 * 1024)  # MB
                    mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                    
                    st.write(f"**Size**: {file_size:.1f} MB")
                    st.write(f"**Modified**: {mod_time.strftime('%Y-%m-%d %H:%M')}")
                
                with col3:
                    # Actions
                    if st.button("🗑️ Delete", key=f"delete_{model_file.stem}"):
                        model_file.unlink()
                        # Also delete metadata if exists
                        if metadata_file.exists():
                            metadata_file.unlink()
                        st.rerun()
    else:
        st.info("No trained models found yet")
else:
    st.info("Models directory not found. It will be created when you train your first model.")

# Next Steps Section
st.header("🎯 Next Steps")

st.markdown("""
### Planned Improvements:

1. **🔧 Hyperparameter Optimization** (Coming Next!)
   - Grid search or Bayesian optimization
   - Find optimal model parameters automatically
   - Compare multiple model configurations

2. **🚀 Alternative Models**
   - LightGBM for faster training
   - XGBoost for potentially better performance
   - Neural networks for complex patterns

3. **📊 Advanced Evaluation**
   - Confusion matrices
   - Precision-Recall curves
   - Error analysis by document type

4. **🎨 Active Learning**
   - Identify most uncertain examples
   - Suggest which spans to label next
   - Maximize model improvement per label

**Current Best Practices:**
- Label at least 3-4 complete documents before training
- Use document-level splits to test generalization
- Monitor the train/test AUC gap to detect overfitting
- Iterate: Label → Train → Predict → Correct → Repeat
""")