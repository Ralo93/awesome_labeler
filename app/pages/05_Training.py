import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.state import AppState
from core.exports import export_training_data
from core.models import BoundaryModel, train_boundary_model, evaluate_model_performance
from core.io import load_spans, load_labels

st.set_page_config(page_title="Training", page_icon="🚀", layout="wide")

AppState.init()

st.title("🚀 Model Training")

st.markdown("""
Train boundary detection models from your labeled documents. The trained model will be available 
for predictions in the **Label** page to speed up your annotation workflow.
""")

# Initialize session state for training
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None

# Training Mode Selection
st.header("🎯 Training Mode")
mode_col1, mode_col2 = st.columns([1, 2])

with mode_col1:
    simple_mode = st.checkbox(
        "Simple Mode",
        value=True,
        help="Train on all data without train/test split"
    )

with mode_col2:
    if simple_mode:
        st.info("""
        **Simple Mode Active** 🚀
        - Train on all available labeled data
        - No train/test split or evaluation metrics
        - Perfect for iterative improvement workflow
        - Focus on quick training and application
        """)
    else:
        st.info("""
        **Advanced Mode Active** 🧪
        - Uses train/test split from export
        - Provides evaluation metrics (AUC, accuracy)
        - Best for final model evaluation
        - Requires pre-split data files
        """)

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
        
        # Check if it has train/test splits
        train_path = exports_dir / f'train_{file_path.name}'
        test_path = exports_dir / f'test_{file_path.name}'
        has_splits = train_path.exists() and test_path.exists()
        
        datasets.append({
            'name': file_path.stem,
            'path': file_path,
            'has_splits': has_splits,
            'size': file_path.stat().st_size / (1024 * 1024),  # MB
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
        })
    
    return datasets

def train_simple_model(data_path: Path, model_name: str):
    """Train model in simple mode (no train/test split)"""
    
    progress_bar = st.progress(0, text="Loading training data...")
    
    try:
        # Step 1: Load data
        progress_bar.progress(20, text="Loading training data...")
        df = pd.read_parquet(data_path)
        
        # Verify data
        if 'y_boundary' not in df.columns:
            raise ValueError("Training data missing 'y_boundary' column")
        
        # Step 2: Initialize model
        progress_bar.progress(40, text="Initializing model...")
        
        model = BoundaryModel(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            calibrate=True
        )
        
        # Step 3: Train model on all data
        progress_bar.progress(60, text="Training model on all data...")
        
        # For simple mode, pass None as test_df
        training_metrics = model.train(df, test_df=None)
        
        # Step 4: Save model
        progress_bar.progress(80, text="Saving model...")
        
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{model_name}.pkl"
        model.save(model_path)
        
        progress_bar.progress(100, text="Training complete!")
        
        # Store results
        results = {
            'model_name': model_name,
            'model_path': str(model_path),
            'training_metrics': training_metrics,
            'n_examples': len(df),
            'n_positive': df['y_boundary'].sum(),
            'n_negative': len(df) - df['y_boundary'].sum(),
            'features_count': training_metrics.get('feature_count', 0),
            'feature_importance': training_metrics.get('feature_importance', []),
            'created_at': datetime.now().isoformat(),
            'mode': 'simple'
        }
        
        return results
        
    except Exception as e:
        progress_bar.empty()
        raise e

def train_advanced_model(data_path: Path, model_name: str):
    """Train model in advanced mode (with train/test split)"""
    
    progress_bar = st.progress(0, text="Loading training data...")
    
    try:
        # Check for train/test files
        exports_dir = data_path.parent
        train_path = exports_dir / f'train_{data_path.name}'
        test_path = exports_dir / f'test_{data_path.name}'
        
        if not train_path.exists() or not test_path.exists():
            raise ValueError("Train/test split files not found. Please export with Advanced Mode enabled.")
        
        # Step 1: Load data
        progress_bar.progress(20, text="Loading train/test data...")
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        # Step 2: Initialize model
        progress_bar.progress(40, text="Initializing model...")
        
        model = BoundaryModel(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            calibrate=True
        )
        
        # Step 3: Train model
        progress_bar.progress(60, text="Training model...")
        
        training_metrics = model.train(train_df, test_df)
        
        # Step 4: Save model
        progress_bar.progress(80, text="Saving model...")
        
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{model_name}.pkl"
        model.save(model_path)
        
        # Step 5: Evaluate model
        progress_bar.progress(90, text="Evaluating model...")
        
        test_metrics = evaluate_model_performance(model, test_df)
        
        progress_bar.progress(100, text="Training complete!")
        
        # Store results
        results = {
            'model_name': model_name,
            'model_path': str(model_path),
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'n_train_examples': len(train_df),
            'n_test_examples': len(test_df),
            'features_count': training_metrics.get('feature_count', 0),
            'train_auc': training_metrics.get('train_metrics', {}).get('auc_roc', 0),
            'test_auc': test_metrics.get('auc_roc', 0),
            'feature_importance': training_metrics.get('feature_importance', []),
            'created_at': datetime.now().isoformat(),
            'mode': 'advanced'
        }
        
        return results
        
    except Exception as e:
        progress_bar.empty()
        raise e

# Dataset Selection
st.header("📊 Select Training Data")

datasets = get_available_datasets()

if not datasets:
    st.warning("No training datasets found. Please export data from the Export page first.")
    st.stop()

# Filter datasets based on mode
if simple_mode:
    # In simple mode, show all datasets
    available_datasets = datasets
else:
    # In advanced mode, only show datasets with splits
    available_datasets = [d for d in datasets if d['has_splits']]
    
    if not available_datasets:
        st.warning("No datasets with train/test splits found. Please export with Advanced Mode or switch to Simple Mode.")
        st.stop()

# Dataset selection
dataset_options = {d['name']: d for d in available_datasets}
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
        st.metric("Total Examples", len(df))
    
    with col2:
        st.metric("Positive Boundaries", df['y_boundary'].sum())
    
    with col3:
        st.metric("Negative Examples", len(df) - df['y_boundary'].sum())
    
    with col4:
        boundary_rate = df['y_boundary'].mean() * 100
        st.metric("Boundary Rate", f"{boundary_rate:.1f}%")
    
    # Show document distribution
    if 'doc_id' in df.columns:
        doc_counts = df['doc_id'].value_counts()
        st.write(f"**Documents in dataset:** {len(doc_counts)}")
        
        with st.expander("Document distribution"):
            st.dataframe(doc_counts.reset_index().rename(columns={'index': 'Document', 'doc_id': 'Examples'}))

# Training Configuration
st.header("⚙️ Training Configuration")

model_name = st.text_input(
    "Model Name",
    value=f"boundary_model_{datetime.now().strftime('%Y%m%d_%H%M')}",
    help="Name for the trained model"
)

# Training Section
st.header("🚀 Train Model")

if st.button("🚀 Start Training", type="primary", use_container_width=True):
    if not model_name.strip():
        st.error("Please enter a model name")
    else:
        with st.spinner("Training model..."):
            try:
                if simple_mode:
                    results = train_simple_model(
                        data_path=selected_dataset['path'],
                        model_name=model_name.strip()
                    )
                else:
                    results = train_advanced_model(
                        data_path=selected_dataset['path'],
                        model_name=model_name.strip()
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
    
    if results.get('mode') == 'simple':
        # Simple mode results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Examples", results['n_examples'])
        
        with col2:
            st.metric("Features", results['features_count'])
        
        with col3:
            boundary_rate = (results['n_positive'] / results['n_examples'] * 100) if results['n_examples'] > 0 else 0
            st.metric("Boundary Rate", f"{boundary_rate:.1f}%")
        
        st.success(f"""
        ✅ Model trained successfully in Simple Mode!
        - No test metrics (trained on all data)
        - Ready to use in Label page for predictions
        - Continue labeling and retrain for better results
        """)
        
    else:
        # Advanced mode results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Training AUC", f"{results['train_auc']:.3f}")
        
        with col2:
            st.metric("Test AUC", f"{results['test_auc']:.3f}")
        
        with col3:
            st.metric("Features", results['features_count'])
        
        with col4:
            st.metric("Training Examples", results['n_train_examples'])
    
    # Feature importance chart (both modes)
    if results.get('feature_importance'):
        st.subheader("🎯 Top Feature Importance")
        
        features = results['feature_importance'][:15]  # Top 15
        feature_names = [f[0] for f in features]
        importances = [f[1] for f in features]
        
        fig = px.bar(
            x=importances,
            y=feature_names,
            orientation='h',
            title="Top 15 Most Important Features"
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model info
    st.subheader("📋 Model Information")
    st.json({
        'model_name': results['model_name'],
        'model_path': results['model_path'],
        'mode': results.get('mode', 'unknown'),
        'created_at': results['created_at']
    })

# Existing Models Section
st.header("📚 Existing Models")

models_dir = Path("models")
if models_dir.exists():
    model_files = list(models_dir.glob("*.pkl"))
    
    if model_files:
        for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
            with st.expander(f"🤖 {model_file.stem}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Model metadata
                    metadata_file = model_file.with_suffix('.json')
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            train_metrics = metadata.get('train_metrics', {})
                            test_metrics = metadata.get('test_metrics', {})
                            
                            st.write(f"**Training AUC**: {train_metrics.get('auc_roc', 'N/A')}")
                            
                            if test_metrics:
                                st.write(f"**Test AUC**: {test_metrics.get('auc_roc', 'N/A')}")
                            else:
                                st.write("**Mode**: Simple (no test metrics)")
                            
                            st.write(f"**Features**: {metadata.get('feature_count', 'N/A')}")
                            st.write(f"**Training Examples**: {train_metrics.get('n_samples', 'N/A')}")
                            
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
                        metadata_file = model_file.with_suffix('.json')
                        if metadata_file.exists():
                            metadata_file.unlink()
                        st.rerun()
                    
                    # Test model loading
                    if st.button("🧪 Test Load", key=f"test_{model_file.stem}"):
                        try:
                            from core.models import BoundaryModel
                            model = BoundaryModel.load(model_file)
                            st.success("✅ Model loads successfully")
                        except Exception as e:
                            st.error(f"❌ Failed to load: {e}")
    else:
        st.info("No trained models found yet")
else:
    st.info("Models directory not found. It will be created when you train your first model.")

# Workflow Tips
st.header("💡 Workflow Tips")

if simple_mode:
    st.markdown("""
    ### 🚀 Simple Mode Workflow:
    
    1. **Start Small**: Label just 1-2 pages (50-100 spans)
    2. **Train Quickly**: Use this page to train your first model
    3. **Apply & Correct**: Use predictions in Label page, fix errors
    4. **Iterate**: Export corrected labels and retrain
    5. **Scale Up**: Gradually label more pages as model improves
    
    **Pro Tips:**
    - Focus on getting boundaries right, not perfect coverage
    - The model learns from patterns, so consistent labeling helps
    - After 3-4 iterations, you'll have a strong model
    - Use "Auto-Label Remaining" in Label page to speed up
    """)
else:
    st.markdown("""
    ### 🧪 Advanced Mode Workflow:
    
    **When to use Advanced Mode:**
    - You have 500+ labeled examples
    - You need to evaluate model performance
    - You're comparing different models
    - You're done with iterative labeling
    
    **Metrics Guide:**
    - **AUC > 0.85**: Excellent performance
    - **AUC 0.75-0.85**: Good performance
    - **AUC 0.65-0.75**: Needs improvement
    - **AUC < 0.65**: Need more/better labels
    
    **If test AUC << train AUC**: Model is overfitting, need more diverse data
    """)