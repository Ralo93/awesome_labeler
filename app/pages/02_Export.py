import streamlit as st
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.state import AppState
from core.exports import export_training_data
from core.io import load_spans, load_labels

st.set_page_config(page_title="Export", page_icon="📤", layout="wide")

AppState.init()

st.title("📤 Export Training Data")

# Get available documents
data_dir = Path("data/docs")
if data_dir.exists():
    doc_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    doc_ids = [d.name for d in doc_dirs]
else:
    doc_ids = []

if not doc_ids:
    st.warning("No documents found. Please upload and label documents first.")
    st.stop()

# Document selection
st.header("📋 Select Documents")
st.info("Export labeled documents as training data for boundary detection model")

selected_docs = st.multiselect(
    "Select documents to export",
    options=doc_ids,
    default=doc_ids,
    help="Choose which documents to include in training data"
)

if not selected_docs:
    st.warning("Please select at least one document")
    st.stop()

# Export configuration
st.header("⚙️ Export Configuration")

col1, col2 = st.columns(2)

with col1:
    output_filename = st.text_input(
        "Output filename",
        value="training_data",
        help="Name for the exported parquet file"
    )
    
    test_size = st.slider(
        "Test split size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Fraction of documents to use for testing"
    )

with col2:
    random_state = st.number_input(
        "Random seed",
        value=42,
        help="Seed for reproducible train/test splits"
    )

# Document statistics
st.header("📊 Document Statistics")

if st.button("🔍 Analyze Documents"):
    doc_stats = []
    total_spans = 0
    total_boundaries = 0
    
    for doc_id in selected_docs:
        # Fix: Use load_spans directly instead of AppState.get_current_spans()
        spans = load_spans(doc_id)
        labels = load_labels(doc_id)
        
        n_spans = len(spans)
        n_labeled = len(labels)
        n_boundaries = sum(1 for l in labels.values() if l.boundary == "new")
        n_units = len(set(l.unit_id for l in labels.values())) if labels else 0
        
        coverage = (n_labeled / n_spans * 100) if n_spans > 0 else 0
        
        doc_stats.append({
            'Document': doc_id,
            'Spans': n_spans,
            'Labeled': n_labeled,
            'Coverage (%)': f"{coverage:.1f}",
            'Boundaries': n_boundaries,
            'Units': n_units
        })
        
        total_spans += n_spans
        total_boundaries += n_boundaries
    
    df_stats = pd.DataFrame(doc_stats)
    st.dataframe(df_stats, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", len(selected_docs))
    with col2:
        st.metric("Total Spans", total_spans)
    with col3:
        st.metric("Total Boundaries", total_boundaries)
    with col4:
        boundary_rate = (total_boundaries / total_spans * 100) if total_spans > 0 else 0
        st.metric("Boundary Rate", f"{boundary_rate:.1f}%")

# Export section
st.header("🚀 Export Training Data")

export_path = Path("exports") / f"{output_filename}.parquet"

if st.button("📤 Export Training Data", type="primary"):
    if not selected_docs:
        st.error("Please select at least one document")
    else:
        with st.spinner("Exporting training data..."):
            try:
                # Export with automatic train/test split
                full_df, train_df, test_df = export_training_data(
                    doc_ids=selected_docs,
                    output_path=export_path,
                    test_size=test_size,
                    random_state=random_state
                )
                
                st.success("✅ Training data exported successfully!")
                
                # Display export summary
                st.subheader("📋 Export Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Examples", len(full_df))
                    st.metric("Training Examples", len(train_df))
                    st.metric("Test Examples", len(test_df))
                
                with col2:
                    st.metric("Features", len([c for c in full_df.columns if c not in ['doc_id', 'page', 'span_id', 'y_boundary']]))
                    st.metric("Positive Boundaries", full_df['y_boundary'].sum())
                    st.metric("Negative Boundaries", len(full_df) - full_df['y_boundary'].sum())
                
                with col3:
                    boundary_rate = full_df['y_boundary'].mean() * 100
                    st.metric("Boundary Rate", f"{boundary_rate:.1f}%")
                    
                    # Class balance in splits
                    train_pos_rate = train_df['y_boundary'].mean() * 100
                    test_pos_rate = test_df['y_boundary'].mean() * 100
                    st.metric("Train Pos Rate", f"{train_pos_rate:.1f}%")
                    st.metric("Test Pos Rate", f"{test_pos_rate:.1f}%")
                
                # File paths
                st.subheader("📁 Generated Files")
                st.code(f"""
Full dataset: {export_path}
Training set: {export_path.parent / f'train_{export_path.name}'}
Test set: {export_path.parent / f'test_{export_path.name}'}
                """)
                
                # Feature preview
                st.subheader("🔍 Feature Preview")
                feature_cols = [c for c in full_df.columns if c not in ['doc_id', 'page', 'span_id', 'y_boundary']]
                st.write(f"**Feature count:** {len(feature_cols)}")
                
                with st.expander("Show all features"):
                    st.write(feature_cols)
                
                # Sample data preview
                st.subheader("📋 Sample Data")
                sample_df = full_df.head(10)[['doc_id', 'page', 'span_id'] + feature_cols[:5] + ['y_boundary']]
                st.dataframe(sample_df)
                
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
                st.exception(e)

# Existing exports
st.header("📂 Existing Exports")

exports_dir = Path("exports")
if exports_dir.exists():
    export_files = list(exports_dir.glob("*.parquet"))
    if export_files:
        st.write("Previously exported training datasets:")
        
        for file_path in sorted(export_files, reverse=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"📄 **{file_path.name}**")
                # File size and modification time
                size_mb = file_path.stat().st_size / (1024 * 1024)
                mtime = pd.Timestamp.fromtimestamp(file_path.stat().st_mtime)
                st.caption(f"Size: {size_mb:.1f} MB | Modified: {mtime.strftime('%Y-%m-%d %H:%M')}")
            
            with col2:
                if st.button("📊 Info", key=f"info_{file_path.name}"):
                    try:
                        df = pd.read_parquet(file_path)
                        st.write(f"**{file_path.name}** - {len(df)} examples, {len(df.columns)} columns")
                        if 'y_boundary' in df.columns:
                            st.write(f"Boundary rate: {df['y_boundary'].mean()*100:.1f}%")
                    except Exception as e:
                        st.error(f"Error reading {file_path.name}: {e}")
            
            with col3:
                if st.button("🗑️ Delete", key=f"del_{file_path.name}"):
                    file_path.unlink()
                    # Also try to delete train/test splits
                    train_path = file_path.parent / f'train_{file_path.name}'
                    test_path = file_path.parent / f'test_{file_path.name}'
                    if train_path.exists():
                        train_path.unlink()
                    if test_path.exists():
                        test_path.unlink()
                    st.rerun()
    else:
        st.info("No exported files found")
else:
    st.info("No exports directory found")

# Training instructions
st.header("🎯 Next Steps")
st.markdown("""
After exporting training data:

1. **Train the model**: Run the training script
   ```bash
   python train/train_model.py --input exports/{filename}.parquet
   ```

2. **Load trained model**: Use the model in the Label page for predictions

3. **Iterate**: Label more documents → Export → Train → Repeat
""")