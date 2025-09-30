import streamlit as st
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import subprocess
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from app.state import AppState
from core.io import load_spans, load_labels, save_labels
from core.models import BoundaryModel

# Page config
st.set_page_config(
    page_title="Semantic Unit Labeler",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize state
AppState.init()

# Title
st.title("📝 Semantic Unit Labeler")

# Sidebar
with st.sidebar:
    st.header("Document & Navigation")
    
    # Document selection
    data_dir = Path("data/docs")
    if data_dir.exists():
        doc_ids = [d.name for d in data_dir.iterdir() if d.is_dir()]
        
        selected_doc = st.selectbox(
            "Select Document",
            options=doc_ids,
            index=0 if doc_ids else None,
            key="doc_selector"
        )
        
        if selected_doc:
            AppState.set_document(selected_doc)
    
    # Page navigation
    if st.session_state.current_doc:
        spans = load_spans(st.session_state.current_doc)
        if spans:
            max_page = max(s.page_number for s in spans)
            
            page = st.slider(
                "Page",
                min_value=1,
                max_value=max_page,
                value=st.session_state.current_page,
                key="page_slider"
            )
            AppState.set_page(page)
    
    # Zoom control
    zoom = st.select_slider(
        "Zoom",
        options=[50, 75, 100, 125, 150, 200],
        value=st.session_state.zoom_level,
        format_func=lambda x: f"{x}%",
        key="zoom_slider"
    )
    st.session_state.zoom_level = zoom
    
    st.divider()
    
    # Model management
    #st.header("Model")
    
    #models_dir = Path("models")
    #models_dir.mkdir(exist_ok=True)
    
    #model_files = list(models_dir.glob("*.pkl"))
    #model_options = ["None"] + [str(f) for f in model_files]
    
    #selected_model = st.selectbox(
    #    "Load Model",
    #    options=model_options,
    #    key="model_selector"
    #)
    
    #if selected_model != "None":
    #    if st.session_state.model_path != selected_model:
    #        st.session_state.model_path = selected_model
    #        st.session_state.model = BoundaryModel(Path(selected_model))
    #        st.success(f"Model loaded: {Path(selected_model).name}")
    
    # Training
    #if st.button("🚀 Train Model", use_container_width=True):
    #    export_path = Path("exports/train_raw.parquet")
    #    
    #    if export_path.exists():
    #        output_name = f"model_{len(model_files)+1:03d}.pkl"
    #        output_path = models_dir / output_name
    #        
    #        with st.spinner("Training model..."):
    #            result = subprocess.run([
    #                "python", "train/train_model.py",
    #                "--data", str(export_path),
    #                "--out", str(output_path)
    #            ], capture_output=True, text=True)
    #            
    #            if result.returncode == 0:
    #                st.success(f"Model trained: {output_name}")
    #                st.rerun()
    #            else:
    #                st.error(f"Training failed: {result.stderr}")
    #    else:
    #        st.error("No training data found. Export data first.")
   # 
    st.divider()
    
    # Display options
    st.header("Display Options")
    
    st.session_state.show_features = st.checkbox(
        "Show normalized features",
        value=st.session_state.show_features
    )
    
    st.session_state.show_rules = st.checkbox(
        "Show rule-based classes",
        value=st.session_state.show_rules
    )
    
    st.session_state.confidence_mode = st.checkbox(
        "Enable confidence scoring",
        value=st.session_state.confidence_mode
    )
    
    # Status
    st.divider()
    
    if st.session_state.dirty:
        st.warning("⚠️ Unsaved changes")
    else:
        st.success("✓ All changes saved")
    
    st.caption(f"Last save: {st.session_state.last_save.strftime('%H:%M:%S')}")

# Main content
if st.session_state.current_doc:
    st.info(f"Document: {st.session_state.current_doc} | Page: {st.session_state.current_page}")
else:
    st.warning("Please select a document from the sidebar")