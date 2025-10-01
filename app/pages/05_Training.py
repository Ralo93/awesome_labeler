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

def get_document_stats():
    """Get statistics about available documents"""
    data_dir = Path("data/docs")
    if not data_dir.exists():
        return []
    
    doc_stats = []
    for doc_dir in data_dir.iterdir():
        if not doc_dir.is_dir():
            continue
            
        doc_id = doc_dir.name
        spans = load_spans(doc_id)
        labels = load_labels(doc_id)
        
        if not spans:
            continue
            
        n_spans = len(spans)
        n_labeled = len(labels)
        n_boundaries = sum(1 for l in labels.values() if l.boundary == "new")
        n_units = len(set(l.unit_id for l in labels.values())) if labels else 0
        coverage = (n_labeled / n_spans * 100) if n_spans > 0 else 0
        
        doc_stats.append({
            'doc_id': doc_id,
            'spans': n_spans,
            'labeled': n_labeled,
            'coverage': coverage,
            'boundaries': n_boundaries,
            'units': n_units,
            'ready': coverage > 50  # At least 50% labeled
        })
    
    return doc_stats

def train_new_model(doc_ids, model_name, test_size, random_state):
    """Train a new boundary detection model"""
    
    progress_bar = st.progress(0, text="Preparing training data...")
    
    try:
        # Step 1: Export training data
        progress_bar.progress(20, text="Exporting training data...")
        
        temp_export_path = Path("temp") / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        temp_export_path.parent.mkdir(exist_ok=True)
        
        full_df, train_df, test_df = export_training_data(
            doc_ids=doc_ids,
            output_path=temp_export_path,
            test_size=test_size,
            random_state=random_state
        )
        
        # Step 2: Initialize model
        progress_bar.progress(40, text="Initializing model...")
        
        model = BoundaryModel(
            n_estimators=200,
            max_depth=20,
            random_state=random_state,
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
        
        test_metrics = evaluate_model_performance(model, test_df) if not test_df.empty else {}
        
        progress_bar.progress(100, text="Training complete!")
        
        # Clean up temp files
        if temp_export_path.exists():
            temp_export_path.unlink()
        
        # Store results
        results = {
            'model_name': model_name,
            'model_path': str(model_path),
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'n_documents': len(doc_ids),
            'n_train_examples': len(train_df),
            'n_test_examples': len(test_df),
            'features_count': training_metrics.get('feature_count', 0),
            'train_auc': training_metrics.get('train_metrics', {}).get('auc_roc', 0),
            'test_auc': test_metrics.get('auc_roc', 0) if test_metrics else 0,
            'feature_importance': training_metrics.get('feature_importance', []),
            'created_at': datetime.now().isoformat()
        }
        
        return results
        
    except Exception as e:
        progress_bar.empty()
        raise e

# Document Statistics Section
st.header("📊 Document Statistics")

doc_stats = get_document_stats()

if not doc_stats:
    st.warning("No documents found. Please upload and label documents first.")
    st.stop()

# Create stats DataFrame
stats_df = pd.DataFrame(doc_stats)

# Display summary metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Documents", len(stats_df))

with col2:
    ready_docs = stats_df[stats_df['ready'] == True]
    st.metric("Ready for Training", len(ready_docs))

with col3:
    total_spans = stats_df['spans'].sum()
    st.metric("Total Spans", total_spans)

with col4:
    total_boundaries = stats_df['boundaries'].sum()
    boundary_rate = (total_boundaries / total_spans * 100) if total_spans > 0 else 0
    st.metric("Boundary Rate", f"{boundary_rate:.1f}%")

# Document selection table
st.subheader("📋 Document Selection")

# Add selection column
stats_df['selected'] = True  # Default to all selected

edited_df = st.data_editor(
    stats_df,
    column_config={
        "selected": st.column_config.CheckboxColumn(
            "Select",
            help="Include in training",
            default=True,
        ),
        "doc_id": st.column_config.TextColumn(
            "Document ID",
            width="medium",
        ),
        "spans": st.column_config.NumberColumn(
            "Spans",
            width="small",
        ),
        "labeled": st.column_config.NumberColumn(
            "Labeled",
            width="small",
        ),
        "coverage": st.column_config.NumberColumn(
            "Coverage %",
            width="small",
            format="%.1f%%",
        ),
        "boundaries": st.column_config.NumberColumn(
            "Boundaries",
            width="small",
        ),
        "units": st.column_config.NumberColumn(
            "Units",
            width="small",
        ),
        "ready": st.column_config.CheckboxColumn(
            "Ready",
            help="At least 50% labeled",
            disabled=True,
        ),
    },
    disabled=["doc_id", "spans", "labeled", "coverage", "boundaries", "units", "ready"],
    hide_index=True,
    use_container_width=True
)

# Get selected documents
selected_docs = edited_df[edited_df['selected'] == True]['doc_id'].tolist()

if not selected_docs:
    st.warning("Please select at least one document for training.")
    st.stop()

# Training Configuration
st.header("⚙️ Training Configuration")

col1, col2 = st.columns(2)

with col1:
    model_name = st.text_input(
        "Model Name",
        value=f"boundary_model_{datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Name for the trained model"
    )
    
    test_size = st.slider(
        "Test Split Size",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        help="Fraction of documents to use for testing"
    )

with col2:
    random_state = st.number_input(
        "Random Seed",
        value=42,
        help="Seed for reproducible results"
    )
    
    st.info(f"**Selected documents**: {len(selected_docs)}")
    ready_count = len([d for d in selected_docs if d in edited_df[edited_df['ready'] == True]['doc_id'].tolist()])
    st.info(f"**Ready documents**: {ready_count}/{len(selected_docs)}")

# Training Section
st.header("🚀 Train Model")

# Warning for documents with low coverage
low_coverage_docs = edited_df[
    (edited_df['selected'] == True) & 
    (edited_df['coverage'] < 50)
]['doc_id'].tolist()

if low_coverage_docs:
    st.warning(f"⚠️ Low coverage documents: {', '.join(low_coverage_docs)} (<50% labeled)")

if st.button("🚀 Start Training", type="primary", use_container_width=True):
    if not selected_docs:
        st.error("Please select at least one document")
    elif not model_name.strip():
        st.error("Please enter a model name")
    else:
        with st.spinner("Training model..."):
            try:
                results = train_new_model(
                    doc_ids=selected_docs,
                    model_name=model_name.strip(),
                    test_size=test_size,
                    random_state=random_state
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
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training AUC", f"{results['train_auc']:.3f}")
    
    with col2:
        if results['test_auc'] > 0:
            st.metric("Test AUC", f"{results['test_auc']:.3f}")
        else:
            st.metric("Test AUC", "N/A")
    
    with col3:
        st.metric("Features", results['features_count'])
    
    with col4:
        st.metric("Training Examples", results['n_train_examples'])
    
    # Feature importance chart
    if results['feature_importance']:
        st.subheader("🎯 Top Feature Importance")
        
        # Prepare data for plotting
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
        'documents_used': results['n_documents'],
        'created_at': results['created_at'],
        'training_examples': results['n_train_examples'],
        'test_examples': results['n_test_examples']
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
                            st.write(f"**Test AUC**: {test_metrics.get('auc_roc', 'N/A')}")
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

# Training Tips
st.header("💡 Training Tips")

st.markdown("""
### 🎯 For Best Results:

**Data Quality:**
- Label at least **50% of spans** in each document
- Include **diverse document types** (different layouts, fonts, structures)
- Aim for **10+ documents** minimum for initial training

**Model Performance:**
- **AUC > 0.8**: Good performance, ready for active learning
- **AUC 0.7-0.8**: Decent performance, may need more data
- **AUC < 0.7**: Needs more labeled examples or feature engineering

**Active Learning Workflow:**
1. Label 3-5 documents manually
2. Train initial model 
3. Use model predictions to speed up labeling
4. Retrain with more data periodically

**Feature Importance:**
- High importance features indicate what the model learned
- Font size changes, vertical gaps, and text patterns are typically important
- If importance seems wrong, check your labeling consistency
""")