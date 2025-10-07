import streamlit as st
from pathlib import Path
import sys
import subprocess
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.visualization import export_comparison_visualization
from core.io import load_spans, load_labels, save_labels
from core.schematas import Label
from core.features import compute_sliding_window_features
from app.state import AppState

st.set_page_config(page_title="Visualize", page_icon="📊", layout="wide")

AppState.init()

st.title("📊 Export Visualization")

st.markdown("""
Generate comparison PDFs showing:
- **Left side**: Original spans with automatic classifications
- **Right side**: Semantic units (from manual labels OR model predictions)

You can either visualize existing labels or generate semantic units using a trained boundary model.
""")


# Neural Network Architecture (same as training/labeling)
class BoundaryNN(nn.Module):
    """Feedforward neural network for boundary detection"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.network(x))


class NeuralBoundaryModel:
    """Neural network wrapper compatible with pipeline"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], 
                 dropout_rate=0.3, device='cpu'):
        self.model = BoundaryNN(input_dim, hidden_dims, dropout_rate).to(device)
        self.device = device
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.threshold_ = 0.5
        
    def predict_proba(self, X):
        """Predict probabilities - compatible with both DataFrame and list of dicts"""
        import pandas as pd
        
        if isinstance(X, list):
            X = pd.DataFrame(X)
        
        if hasattr(X, 'columns'):
            for col in self.feature_names_:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names_].fillna(0).values
        
        X = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            probs = self.model(torch.FloatTensor(X).to(self.device))
            probs = probs.squeeze().cpu().numpy()
        
        if probs.ndim == 0:
            probs = np.array([probs])
        
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """Predict binary labels"""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold_).astype(int)
    
    @classmethod
    def load(cls, path):
        """Load saved neural network model"""
        path = Path(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path.with_suffix('.pth'), map_location=device)
        
        model = cls(
            input_dim=checkpoint['model_config']['input_dim'],
            hidden_dims=checkpoint['model_config']['hidden_dims'],
            dropout_rate=checkpoint['model_config'].get('dropout_rate', 0.3),
            device=device
        )
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.threshold_ = checkpoint['threshold']
        model.feature_names_ = checkpoint['feature_names']
        model.scaler = joblib.load(path.with_suffix('.scaler'))
        
        return model


def load_model(model_path: Path):
    """Load a trained boundary model"""
    try:
        if model_path.suffix == '.pth':
            st.info("Loading Neural Network model...")
            model = NeuralBoundaryModel.load(model_path)
            return model, "Neural Network 🧠"
        elif model_path.suffix == '.pkl':
            st.info("Loading Random Forest model...")
            model = joblib.load(model_path)
            return model, "Random Forest 🌲"
        else:
            st.error(f"Unknown model format: {model_path.suffix}")
            return None, None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


def generate_semantic_units_from_model(doc_id: str, model, threshold: float = 0.5):
    """Generate semantic units by applying model predictions"""
    
    # Load spans
    spans = load_spans(doc_id)
    if not spans:
        raise ValueError(f"No spans found for document {doc_id}")
    
    # Generate predictions
    predictions = {}
    feature_dicts = []
    span_keys = []
    
    progress_bar = st.progress(0, text="Extracting features...")
    
    for idx, span in enumerate(spans):
        features = compute_sliding_window_features(spans, idx)
        feature_dicts.append(features)
        span_keys.append((span.page_number, span.span_id))
        
        if idx % 100 == 0:
            progress = min(0.5, idx / len(spans) * 0.5)
            progress_bar.progress(progress, text=f"Extracting features... {idx}/{len(spans)}")
    
    progress_bar.progress(0.5, text="Running model predictions...")
    
    # Predict all at once
    try:
        probs = model.predict_proba(feature_dicts)
        for (page_num, span_id), prob in zip(span_keys, probs[:, 1]):
            predictions[(page_num, span_id)] = prob
    except Exception as e:
        progress_bar.empty()
        raise Exception(f"Prediction error: {e}")
    
    progress_bar.progress(0.75, text="Creating semantic units...")
    
    # Create labels from predictions
    labels = {}
    unit_counter = 0
    current_unit_id = None
    
    for span in spans:
        key = (span.page_number, span.span_id)
        prob = predictions.get(key, 0.0)
        
        # Determine if this starts a new unit
        is_new_unit = prob >= threshold
        if is_new_unit or current_unit_id is None:
            unit_counter += 1
            current_unit_id = f"unit_{unit_counter}"
            boundary = "new"
        else:
            boundary = "continue"
        
        label = Label(
            span_id=span.span_id,
            page_number=span.page_number,
            boundary=boundary,
            unit_id=current_unit_id,
            confidence=float(prob)
        )
        labels[key] = label
    
    progress_bar.progress(1.0, text="Complete!")
    progress_bar.empty()
    
    return labels, predictions


def _normalize_progress(x: float) -> float:
    if x is None:
        return 0.0
    try:
        val = float(x)
    except (TypeError, ValueError):
        return 0.0
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


# Session state for model
if 'viz_model' not in st.session_state:
    st.session_state.viz_model = None
if 'viz_model_name' not in st.session_state:
    st.session_state.viz_model_name = None
if 'viz_model_type' not in st.session_state:
    st.session_state.viz_model_type = None


# Get all documents
data_dir = Path("data/docs")
doc_ids = []
labeled_docs = []

if data_dir.exists():
    for doc_dir in data_dir.iterdir():
        if doc_dir.is_dir():
            doc_id = doc_dir.name
            doc_ids.append(doc_id)
            
            # Check if document has labels
            labels_file = doc_dir / "labels.jsonl"
            if labels_file.exists() and labels_file.stat().st_size > 0:
                labeled_docs.append(doc_id)

if not doc_ids:
    st.error("No documents found. Please upload a PDF first.")
    st.stop()

st.info(f"Found {len(doc_ids)} documents ({len(labeled_docs)} with existing labels)")


# Sidebar - Model selection
with st.sidebar:
    st.header("🤖 Boundary Model")
    
    st.markdown("""
    **Optional:** Load a model to generate semantic units on-the-fly.
    
    If no model is loaded, only documents with existing labels can be visualized.
    """)
    
    model_dir = Path("models")
    if model_dir.exists():
        rf_models = list(model_dir.glob("boundary_model*.pkl"))
        nn_models = list(model_dir.glob("nn_boundary*.pth"))
        all_models = rf_models + nn_models
        
        if all_models:
            model_options = ["None"] + [f.name for f in all_models]
            
            def format_model_name(name):
                if name == "None":
                    return name
                if name.endswith('.pth'):
                    return f"🧠 {name}"
                else:
                    return f"🌲 {name}"
            
            selected_model = st.selectbox(
                "Select Model",
                model_options,
                format_func=format_model_name
            )
            
            if selected_model != "None" and selected_model != st.session_state.viz_model_name:
                if st.button("Load Model"):
                    model_path = model_dir / selected_model
                    model, model_type = load_model(model_path)
                    if model:
                        st.session_state.viz_model = model
                        st.session_state.viz_model_name = selected_model
                        st.session_state.viz_model_type = model_type
                        st.success(f"✅ {model_type} loaded")
                        st.rerun()
        else:
            st.info("No models found in /models")
    
    if st.session_state.viz_model:
        st.success(f"✅ {st.session_state.viz_model_type}")
        st.caption(st.session_state.viz_model_name)
        
        st.divider()
        
        st.subheader("Model Settings")
        boundary_threshold = st.slider(
            "Boundary Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for new unit boundaries"
        )
    else:
        boundary_threshold = 0.5


# Document selection
st.header("📄 Select Document")

# Filter documents based on whether model is loaded
if st.session_state.viz_model:
    available_docs = doc_ids  # All documents available with model
    st.info("✨ Model loaded - can generate semantic units for any document")
else:
    available_docs = labeled_docs  # Only labeled documents without model
    if not available_docs:
        st.warning("No labeled documents found. Please label a document first, or load a model to generate semantic units automatically.")
        st.stop()

selected_doc = st.selectbox(
    "Select Document to Visualize",
    options=available_docs,
    help="Documents available for visualization"
)

if selected_doc:
    # Show document info
    spans = load_spans(selected_doc)
    existing_labels = load_labels(selected_doc)
    
    total_pages = max(s.page_number for s in spans) if spans else 0
    total_spans = len(spans)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pages", total_pages)
    
    with col2:
        st.metric("Total Spans", total_spans)
    
    with col3:
        if existing_labels:
            st.metric("Existing Labels", len(existing_labels))
            unique_units = len(set(l.unit_id for l in existing_labels.values()))
            st.caption(f"{unique_units} semantic units")
        else:
            st.metric("Existing Labels", "None")
    
    with col4:
        if st.session_state.viz_model:
            st.metric("Mode", "Model")
            st.caption("Will generate units")
        else:
            st.metric("Mode", "Labels")
            st.caption("Using existing labels")
    
    st.divider()
    
    # Generation mode selection
    st.subheader("⚙️ Visualization Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.viz_model and existing_labels:
            use_mode = st.radio(
                "Semantic Units Source",
                ["Use Existing Labels", "Generate from Model"],
                help="Choose whether to use your manual labels or generate units with the model"
            )
            use_model = (use_mode == "Generate from Model")
        elif st.session_state.viz_model:
            st.info("📊 Will generate semantic units using model")
            use_model = True
        else:
            st.info("📋 Will use existing labels")
            use_model = False
        
        output_filename = st.text_input(
            "Output Filename",
            value=f"comparison_{selected_doc}.pdf",
            help="Name for the exported PDF file"
        )
    
    with col2:
        st.info("File will be saved to `exports_comparisons/` directory")
        
        if use_model:
            st.warning(f"⚙️ Model threshold: {boundary_threshold:.2f}")
            st.caption("Adjust threshold in sidebar")
    
    # Classification preview
    with st.expander("📖 Classification Logic Preview"):
        st.markdown("""
        **Left side** shows automatic span classification:
        - **Title**: Large font (>16pt), short text
        - **Header**: Medium-large font (>14pt) or near top
        - **Footer**: Near bottom of page
        - **Caption**: Short text (<150 chars)
        - **Enumeration**: Starts with numbers
        - **PageNumber**: Very short numeric text
        - **Url**: Contains http/www
        - **TextItem**: Default for body text
        
        **Right side** shows semantic units with:
        - Red borders showing grouped spans
        - Unit IDs and span counts
        - Dashed lines showing individual span boundaries
        - Text previews for larger units
        """)
    
    # Export button
    st.divider()
    
    if st.button("🎨 Generate Comparison PDF", type="primary", use_container_width=True):
        try:
            # Step 1: Determine which labels to use
            if use_model:
                st.info("🤖 Generating semantic units from model predictions...")
                labels_to_use, predictions = generate_semantic_units_from_model(
                    selected_doc, 
                    st.session_state.viz_model,
                    boundary_threshold
                )
                
                # Show statistics
                num_units = len(set(l.unit_id for l in labels_to_use.values()))
                st.success(f"✅ Generated {num_units} semantic units from {len(labels_to_use)} spans")
                
                # Show confidence distribution
                avg_conf = np.mean([l.confidence for l in labels_to_use.values()])
                st.caption(f"Average boundary confidence: {avg_conf:.3f}")
                
            else:
                st.info("📋 Using existing manual labels...")
                labels_to_use = existing_labels
                if not labels_to_use:
                    st.error("No labels found! Load a model or label the document first.")
                    st.stop()
            
            # Step 2: Temporarily save labels if using model
            if use_model:
                # Save to a temporary location so visualization can access them
                temp_labels_path = data_dir / selected_doc / "labels_temp.jsonl"
                save_labels(selected_doc, labels_to_use, data_dir)
            
            # Step 3: Export the comparison
            st.info("🎨 Generating visualization PDF...")
            result_path = export_comparison_visualization(
                doc_id=selected_doc,
                output_filename=output_filename
            )
            
            st.success(f"✅ Visualization exported to: {result_path}")
            
            # Show file info and actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                file_size = Path(result_path).stat().st_size / 1024
                st.info(f"File size: {file_size:.1f} KB")
            
            with col2:
                if sys.platform == "darwin":
                    if st.button("📂 Open in Preview"):
                        subprocess.run(["open", result_path])
                elif sys.platform == "win32":
                    if st.button("📂 Open PDF"):
                        subprocess.run(["start", result_path], shell=True)
                else:
                    if st.button("📂 Open PDF"):
                        subprocess.run(["xdg-open", result_path])
            
            with col3:
                with open(result_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=f.read(),
                        file_name=output_filename,
                        mime="application/pdf"
                    )
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)


# Show existing exports
st.divider()
st.subheader("📚 Existing Visualizations")

export_dir = Path("exports_comparisons")
if export_dir.exists():
    pdf_files = list(export_dir.glob("comparison_*.pdf"))
    
    if pdf_files:
        for pdf_file in sorted(pdf_files, key=lambda x: x.stat().st_mtime, reverse=True):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.text(pdf_file.name)
            
            with col2:
                st.text(f"{pdf_file.stat().st_size / 1024:.1f} KB")
            
            with col3:
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        "⬇️",
                        data=f.read(),
                        file_name=pdf_file.name,
                        mime="application/pdf",
                        key=f"download_{pdf_file.name}"
                    )
            
            with col4:
                if st.button("🗑️", key=f"delete_{pdf_file.name}"):
                    pdf_file.unlink()
                    st.rerun()
    else:
        st.info("No visualizations found yet")
else:
    st.info("Export directory will be created when you export.")


# Instructions
st.divider()
st.markdown("""
### 📖 How to Use

**Option 1: Visualize Manual Labels**
1. Label your document in the Label page
2. Select the document above
3. Click Generate to create the comparison PDF

**Option 2: Generate Units with Model**
1. Load a trained boundary model from the sidebar
2. Select any document (labeled or unlabeled)
3. Adjust threshold if needed
4. Click Generate to create semantic units and visualize

**The PDF shows:**
- **Left**: Original spans with automatic classification
- **Right**: Semantic units (grouped spans) with boundaries

This helps you:
- Verify your labeling quality
- Compare manual vs. automatic results
- Evaluate model performance visually
- Document your semantic segmentation
""")