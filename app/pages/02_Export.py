import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.exports import export_training_data
from app.state import AppState

st.set_page_config(page_title="Export", page_icon="📤", layout="wide")

AppState.init()

st.title("📤 Export Training Data")

st.markdown("""
Export training data with automatic deduplication. Both export options will:
- Build pairwise features from spans and labels
- Apply hash-based deduplication
- Save as Parquet format for efficient training
""")

# Get all documents
data_dir = Path("data/docs")
doc_ids = []
if data_dir.exists():
    doc_ids = [d.name for d in data_dir.iterdir() if d.is_dir()]

if not doc_ids:
    st.error("No documents found in data/docs/")
    st.stop()

st.info(f"Found {len(doc_ids)} documents")

# Class handling options
st.subheader("Class Imbalance Handling")

class_handling = st.radio(
    "How to handle class imbalance?",
    options=["none", "balance", "weight"],
    format_func=lambda x: {
        "none": "None - Keep original distribution",
        "balance": "Balance - Undersample majority class",
        "weight": "Weight - Calculate scale_pos_weight for training"
    }[x],
    help="Choose how to handle imbalanced boundary classes"
)

# Export buttons
col1, col2 = st.columns(2)

with col1:
    st.subheader("Export RAW")
    st.markdown("Export without rule-based features")
    
    if st.button("📦 Export RAW Training Data", use_container_width=True, type="primary"):
        with st.spinner("Exporting RAW data..."):
            results = export_training_data(
                file_name=st.session_state['current_doc'],
                doc_ids=doc_ids,
                export_raw=True,
                export_with_rules=False,
                class_handling=class_handling if class_handling != "none" else None
            )
            
            if 'raw' in results:
                st.success(f"✅ Exported to: {results['raw']['path']}")
                st.json(results['raw']['metadata'])
            else:
                st.error("Export failed")

with col2:
    st.subheader("Export +RULES")
    st.markdown("Export with rule-based features included")
    
    if st.button("📦 Export +RULES Training Data", use_container_width=True, type="primary"):
        with st.spinner("Exporting data with rules..."):
            results = export_training_data(
                file_name=st.session_state['current_doc'],
                doc_ids=doc_ids,
                export_raw=False,
                export_with_rules=True,
                class_handling=class_handling if class_handling != "none" else None
            )
            
            if 'rules' in results:
                if 'error' in results['rules']:
                    st.warning(results['rules']['error'])
                else:
                    st.success(f"✅ Exported to: {results['rules']['path']}")
                    st.json(results['rules']['metadata'])
            else:
                st.error("Export failed")

# Show existing exports
st.divider()
st.subheader("Existing Exports")

export_dir = Path("exports")
if export_dir.exists():
    exports = list(export_dir.glob("*.parquet"))
    
    if exports:
        for export_file in exports:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.text(export_file.name)
            with col2:
                st.text(f"{export_file.stat().st_size / 1024:.1f} KB")
            with col3:
                if st.button(f"Delete", key=f"del_{export_file.name}"):
                    export_file.unlink()
                    st.rerun()
    else:
        st.info("No exports found")
else:
    st.info("Export directory not found")

# Statistics
st.divider()
st.subheader("Label Statistics")

if doc_ids:
    total_labels = 0
    boundaries_new = 0
    boundaries_continue = 0
    
    for doc_id in doc_ids:
        from core.io import load_labels
        labels = load_labels(doc_id)
        total_labels += len(labels)
        boundaries_new += sum(1 for l in labels if l.boundary == "new")
        boundaries_continue += sum(1 for l in labels if l.boundary == "continue")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Labels", total_labels)
    with col2:
        st.metric("New Boundaries", boundaries_new)
    with col3:
        st.metric("Continue Boundaries", boundaries_continue)
    
    if total_labels > 0:
        st.progress(
            boundaries_new / total_labels,
            text=f"Class Distribution: {boundaries_new/total_labels:.1%} new boundaries"
        )