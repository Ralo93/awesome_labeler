import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import io
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
import joblib
from app.state import AppState
from app.components.overlay import draw_overlay
from app.components.feature_panel import show_feature_panel
from core.io import save_labels
from core.pdf_processor import render_pdf_page
from core.schematas import Label
from core.features import compute_sliding_window_features

st.set_page_config(page_title="Label", page_icon="🏷️", layout="wide")

AppState.init()

st.title("🏷️ Labeling Interface")


# Neural Network Architecture (same as training)
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
        
    def prepare_data(self, df_or_list):
        """Extract and prepare features from dataframe or list of dicts"""
        import pandas as pd
        
        if isinstance(df_or_list, list):
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(df_or_list)
        else:
            df = df_or_list
        
        metadata_cols = ['doc_id', 'page', 'span_id', 'y_boundary', 'page_number']
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        
        X = df[feature_cols].fillna(0).values
        return X, feature_cols
    
    def predict_proba(self, X):
        """Predict probabilities - compatible with both DataFrame and list of dicts"""
        import pandas as pd
        
        if isinstance(X, list):
            # Handle list of feature dictionaries
            X = pd.DataFrame(X)
        
        if hasattr(X, 'columns'):
            # Ensure all expected features are present
            for col in self.feature_names_:
                if col not in X.columns:
                    X[col] = 0
            # Select features in the correct order
            X = X[self.feature_names_].fillna(0).values
        
        X = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            probs = self.model(torch.FloatTensor(X).to(self.device))
            probs = probs.squeeze().cpu().numpy()
        
        # Handle single sample case
        if probs.ndim == 0:
            probs = np.array([probs])
        
        # Return in sklearn format (n_samples, 2)
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """Predict binary labels"""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold_).astype(int)
    
    @classmethod
    def load(cls, path):
        """Load saved neural network model"""
        path = Path(path)
        
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load PyTorch model
        checkpoint = torch.load(path.with_suffix('.pth'), map_location=device)
        
        # Recreate model
        model = cls(
            input_dim=checkpoint['model_config']['input_dim'],
            hidden_dims=checkpoint['model_config']['hidden_dims'],
            dropout_rate=checkpoint['model_config'].get('dropout_rate', 0.3),
            device=device
        )
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.threshold_ = checkpoint['threshold']
        model.feature_names_ = checkpoint['feature_names']
        
        # Load scaler
        model.scaler = joblib.load(path.with_suffix('.scaler'))
        
        return model
    
    def get_model_type(self):
        """Return model type for display"""
        return "Neural Network"


# Initialize label tracking
if 'unit_counter' not in st.session_state:
    st.session_state.unit_counter = 0

if 'model' not in st.session_state:
    st.session_state.model = None

if 'model_name' not in st.session_state:
    st.session_state.model_name = None

if 'model_type' not in st.session_state:
    st.session_state.model_type = None

# Auto-load document if available but not loaded
if not st.session_state.current_doc:
    data_dir = Path("data/docs")
    if data_dir.exists():
        doc_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if doc_dirs:
            latest_doc = max(doc_dirs, key=lambda d: d.stat().st_mtime)
            AppState.load_document_data(latest_doc.name)


def load_model(model_path: Path):
    """Load a trained boundary model (Random Forest or Neural Network)"""
    try:
        # Detect model type based on file extension
        if model_path.suffix == '.pth':
            # Neural Network model
            st.info("Loading Neural Network model...")
            model = NeuralBoundaryModel.load(model_path)
            st.session_state.model = model
            st.session_state.model_name = model_path.stem
            st.session_state.model_type = "Neural Network 🧠"
            st.success(f"✅ Neural Network loaded: {model_path.stem}")
            
        elif model_path.suffix == '.pkl':
            # Random Forest model
            st.info("Loading Random Forest model...")
            model = joblib.load(model_path)
            st.session_state.model = model
            st.session_state.model_name = model_path.stem
            st.session_state.model_type = "Random Forest 🌲"
            st.success(f"✅ Random Forest loaded: {model_path.stem}")
            
        else:
            st.error(f"Unknown model format: {model_path.suffix}")
            return
            
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        import traceback
        st.error(traceback.format_exc())


def apply_model_predictions(page_only: bool = True, overwrite: bool = False, auto_label: bool = True):
    """Apply model predictions and optionally create semantic units"""
    if not st.session_state.model:
        st.error("No model loaded!")
        return
    
    spans = AppState.get_current_spans()
    if not spans:
        return
    
    # Filter to current page if requested
    if page_only:
        page_spans = [s for s in spans if s.page_number == st.session_state.current_page]
        span_indices = [i for i, s in enumerate(spans) if s.page_number == st.session_state.current_page]
    else:
        page_spans = spans
        span_indices = list(range(len(spans)))
    
    if not page_spans:
        return
    
    # Generate predictions using sliding window features
    predictions = {}
    feature_dicts = []
    keys_to_predict = []
    
    for local_idx, (global_idx, span) in enumerate(zip(span_indices, page_spans)):
        # Skip if already labeled and not overwriting
        key = (span.page_number, span.span_id)
        if key in st.session_state.labels and not overwrite:
            continue
        
        # Extract features using sliding window approach
        features = compute_sliding_window_features(spans, global_idx)
        feature_dicts.append(features)
        keys_to_predict.append((key, span))
    
    if not feature_dicts:
        st.info("No unlabeled spans to predict")
        return
    
    # Predict all at once
    try:
        probs = st.session_state.model.predict_proba(feature_dicts)
        
        # Map predictions back to span keys
        for (key, span), prob in zip(keys_to_predict, probs[:, 1]):
            predictions[key] = prob
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return
    
    # Store predictions
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}
    st.session_state.predictions.update(predictions)
    
    # Auto-create semantic units if enabled
    if auto_label and predictions:
        threshold = st.session_state.get('boundary_threshold', 0.5)
        current_unit_id = f"unit_{st.session_state.unit_counter + 1}"
        labels_to_add = {}
        
        # Sort spans by reading order
        sorted_predictions = sorted(keys_to_predict, key=lambda x: (x[1].page_number, x[1].reading_order))
        
        for (key, span) in sorted_predictions:
            prob = predictions[key]
            
            # Determine if this starts a new unit
            is_new_unit = prob >= threshold
            if is_new_unit:
                st.session_state.unit_counter += 1
                current_unit_id = f"unit_{st.session_state.unit_counter}"
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
            labels_to_add[key] = label
        
        # Add to undo stack
        if labels_to_add:
            AppState.add_to_undo_stack({'labels_added': labels_to_add})
            st.session_state.labels.update(labels_to_add)
            st.success(f"Generated {len(predictions)} predictions and created {st.session_state.unit_counter} semantic units")
    else:
        st.success(f"Generated {len(predictions)} predictions")


def merge_selected_spans():
    """Merge selected spans into a single semantic unit"""
    if not st.session_state.selected_spans:
        st.warning("No spans selected for merging")
        return
    
    spans = AppState.get_current_spans()
    selected_span_objs = [s for s in spans if s.span_id in st.session_state.selected_spans]
    
    if not selected_span_objs:
        return
    
    selected_span_objs.sort(key=lambda s: (s.page_number, s.reading_order))
    
    st.session_state.unit_counter += 1
    unit_id = f"unit_{st.session_state.unit_counter}"
    
    labels_to_add = {}
    for i, span in enumerate(selected_span_objs):
        key = (span.page_number, span.span_id)
        boundary = "new" if i == 0 else "continue"
        
        label = Label(
            span_id=span.span_id,
            page_number=span.page_number,
            boundary=boundary,
            unit_id=unit_id,
            confidence=1.0
        )
        labels_to_add[key] = label
    
    AppState.add_to_undo_stack({
        'labels_added': labels_to_add,
        'selected_spans': st.session_state.selected_spans.copy()
    })
    
    st.session_state.labels.update(labels_to_add)
    st.session_state.selected_spans = []
    save_labels(st.session_state.current_doc, st.session_state.labels)


def split_unit():
    """Split selected spans into separate units"""
    if not st.session_state.selected_spans:
        st.warning("No spans selected for splitting")
        return
    
    spans = AppState.get_current_spans()
    selected_span_objs = [s for s in spans if s.span_id in st.session_state.selected_spans]
    
    labels_to_add = {}
    labels_removed = {}
    
    for span in selected_span_objs:
        key = (span.page_number, span.span_id)
        
        if key in st.session_state.labels:
            labels_removed[key] = st.session_state.labels[key]
        
        st.session_state.unit_counter += 1
        unit_id = f"unit_{st.session_state.unit_counter}"
        
        label = Label(
            span_id=span.span_id,
            page_number=span.page_number,
            boundary="new",
            unit_id=unit_id,
            confidence=1.0
        )
        labels_to_add[key] = label
    
    AppState.add_to_undo_stack({
        'labels_added': labels_to_add,
        'labels_removed': labels_removed
    })
    
    st.session_state.labels.update(labels_to_add)
    st.session_state.selected_spans = []
    save_labels(st.session_state.current_doc, st.session_state.labels)


def clear_page_labels():
    """Clear all labels on current page"""
    page_num = st.session_state.current_page
    
    labels_to_remove = {}
    for key, label in st.session_state.labels.items():
        if key[0] == page_num:
            labels_to_remove[key] = label
    
    if not labels_to_remove:
        st.info("No labels to clear on this page")
        return
    
    AppState.add_to_undo_stack({'labels_removed': labels_to_remove})
    
    for key in labels_to_remove:
        del st.session_state.labels[key]
    
    st.success(f"Cleared {len(labels_to_remove)} labels from page {page_num}")


def undo_action():
    """Undo last action"""
    AppState.undo()


def save_current_labels():
    """Save current labels to disk"""
    if st.session_state.current_doc:
        save_labels(st.session_state.current_doc, st.session_state.labels)


def toggle_span_selection(span_id: str):
    """Toggle span selection"""
    if span_id in st.session_state.selected_spans:
        st.session_state.selected_spans.remove(span_id)
    else:
        st.session_state.selected_spans.append(span_id)
    AppState.mark_dirty()


def next_uncertain():
    """Navigate to next span with uncertain prediction"""
    if not st.session_state.predictions:
        st.warning("No predictions available")
        return
    
    uncertain_spans = []
    for key, prob in st.session_state.predictions.items():
        if 0.3 <= prob <= 0.7:
            uncertain_spans.append((key, abs(prob - 0.5)))
    
    if not uncertain_spans:
        st.info("No uncertain predictions found")
        return
    
    uncertain_spans.sort(key=lambda x: x[1])
    page_num, span_id = uncertain_spans[0][0]
    st.session_state.current_page = page_num
    st.session_state.selected_spans = [span_id]
    st.success(f"Navigated to uncertain span: {span_id}")


def label_digits_as_units():
    """Label spans containing numeric values as individual units"""
    import re
    
    spans = AppState.get_current_spans()
    if not spans:
        return
    
    current_page = st.session_state.current_page
    page_spans = [s for s in spans if s.page_number == current_page]
    
    if not page_spans:
        st.info("No spans on current page!")
        return
    
    numeric_patterns = [
        r'^-?\d+$',
        r'^-?\d+[.,]\d+$',
        r'^-?\d{1,3}([\s,]\d{3})*([.,]\d+)?$',
        r'^-?\d{1,3}([\s.]\d{3})*(,\d+)?$',
        r'^-?\d+([.,]\d+)?%$',
        r'^-?\$?\d+([.,]\d{2})?$',
        r'^-?€?\d+([.,]\d{2})?$',
        r'^-?£?\d+([.,]\d{2})?$',
        r'^\d+[/-]\d+[/-]\d+$',
        r'^\d{1,2}:\d{2}(:\d{2})?$',
        r'^\(\d+\)$',
        r'^\[\d+\]$',
        r'^#\d+$',
        r'^\d+\.$',
        r'^-?\d+([.,]\d+)?\s*(k|K|m|M|b|B)$',
    ]
    
    digit_spans = []
    for span in page_spans:
        key = (span.page_number, span.span_id)
        if key not in st.session_state.labels:
            cleaned_text = span.text.strip()
            if cleaned_text:
                is_numeric = any(re.match(pattern, cleaned_text) for pattern in numeric_patterns)
                
                if not is_numeric:
                    no_space = cleaned_text.replace(' ', '').replace('-', '').replace('–', '')
                    if no_space and re.match(r'^\d+$', no_space) and len(no_space) <= 10:
                        is_numeric = True
                
                if is_numeric:
                    digit_spans.append(span)
    
    if not digit_spans:
        st.info("No unlabeled numeric spans found on this page!")
        return
    
    labels_to_add = {}
    for span in digit_spans:
        key = (span.page_number, span.span_id)
        st.session_state.unit_counter += 1
        unit_id = f"unit_{st.session_state.unit_counter}"
        
        label = Label(
            span_id=span.span_id,
            page_number=span.page_number,
            boundary="new",
            unit_id=unit_id,
            confidence=1.0
        )
        labels_to_add[key] = label
    
    AppState.add_to_undo_stack({'labels_added': labels_to_add})
    st.session_state.labels.update(labels_to_add)
    save_labels(st.session_state.current_doc, st.session_state.labels)
    
    sample_numbers = [s.text.strip() for s in digit_spans[:5]]
    examples = ", ".join(sample_numbers) if sample_numbers else ""
    
    if examples:
        st.success(f"Labeled {len(labels_to_add)} numeric spans on page {current_page}. Examples: {examples}")
    else:
        st.success(f"Labeled {len(labels_to_add)} numeric spans on page {current_page}")


def label_rest_as_single_units():
    """Label all remaining unlabeled spans as individual units"""
    spans = AppState.get_current_spans()
    if not spans:
        return
    
    current_page = st.session_state.current_page
    page_spans = [s for s in spans if s.page_number == current_page]
    
    if not page_spans:
        st.info("No spans on current page!")
        return
    
    unlabeled_spans = []
    for span in page_spans:
        key = (span.page_number, span.span_id)
        if key not in st.session_state.labels:
            unlabeled_spans.append(span)
    
    if not unlabeled_spans:
        st.info("All spans on this page already labeled!")
        return
    
    labels_to_add = {}
    for span in unlabeled_spans:
        key = (span.page_number, span.span_id)
        st.session_state.unit_counter += 1
        unit_id = f"unit_{st.session_state.unit_counter}"
        
        label = Label(
            span_id=span.span_id,
            page_number=span.page_number,
            boundary="new",
            unit_id=unit_id,
            confidence=1.0
        )
        labels_to_add[key] = label
    
    AppState.add_to_undo_stack({'labels_added': labels_to_add})
    st.session_state.labels.update(labels_to_add)
    save_labels(st.session_state.current_doc, st.session_state.labels)


# Sidebar - Model loading
with st.sidebar:
    st.header("📄 Document")
    
    data_dir = Path("data/docs")
    available_docs = []
    if data_dir.exists():
        available_docs = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    if available_docs:
        current_doc_index = 0
        if st.session_state.current_doc and st.session_state.current_doc in available_docs:
            current_doc_index = available_docs.index(st.session_state.current_doc)
        
        selected_doc = st.selectbox(
            "Select Document",
            available_docs,
            index=current_doc_index,
            key="doc_selector"
        )
        
        if selected_doc != st.session_state.current_doc:
            AppState.load_document_data(selected_doc)
            st.rerun()
    else:
        st.info("No documents found. Upload a document first.")
    
    st.header("🤖 Model")
    
    model_dir = Path("models")
    if model_dir.exists():
        # Find both .pkl (Random Forest) and .pth (Neural Network) models
        rf_models = list(model_dir.glob("boundary_model*.pkl"))
        nn_models = list(model_dir.glob("nn_boundary*.pth"))
        
        all_models = rf_models + nn_models
        
        if all_models:
            model_options = ["None"] + [f.name for f in all_models]
            
            # Add type indicator in display
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
            
            if selected_model != "None" and selected_model != st.session_state.model_name:
                if st.button("Load Model"):
                    # Find the full path
                    model_path = model_dir / selected_model
                    load_model(model_path)
                    st.rerun()
        else:
            st.info("No models found in /models")
    
    if st.session_state.model:
        st.success(f"✅ {st.session_state.model_type}")
        st.caption(st.session_state.model_name)
        
        # Prediction controls
        st.subheader("Predictions")
        
        st.session_state.boundary_threshold = st.slider(
            "Boundary Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('boundary_threshold', 0.5),
            step=0.05,
            help="Probability threshold for new unit boundaries"
        )
        
        # Check unlabeled spans
        if st.session_state.current_doc:
            spans = AppState.get_current_spans()
            current_page = st.session_state.current_page
            page_spans = [s for s in spans if s.page_number == current_page]
            unlabeled_count = sum(1 for span in page_spans 
                                if (span.page_number, span.span_id) not in st.session_state.labels)
            
            if unlabeled_count > 0:
                st.info(f"📊 {unlabeled_count} unlabeled spans on this page")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔮 Predict & Label Page"):
                apply_model_predictions(page_only=True, auto_label=True)
                st.rerun()
        
        with col2:
            if st.button("👁️ Preview Only"):
                apply_model_predictions(page_only=True, auto_label=False)
        
        if st.button("🔮 Predict & Label All"):
            apply_model_predictions(page_only=False, auto_label=True)
            st.rerun()
        
        if st.button("🎯 Next Uncertain"):
            next_uncertain()
    
    st.divider()
    
    st.header("Display Options")
    show_unit_numbers = st.checkbox("Show Unit Numbers", value=True)
    show_heatmap = st.checkbox(
        "Show Prediction Heatmap",
        value=bool(st.session_state.get('predictions'))
    )


# Toolbar
st.markdown("### 🎯 Labeling Workflow")
st.info("**Select spans** that belong together → **Merge** them → **Auto-label** remaining spans")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

with c1:
    if st.button("🔗 Merge Selected"):
        merge_selected_spans()

with c2:
    if st.button("✂️ Split Selected"):
        split_unit()

with c3:
    if st.button("🗑️ Clear Page"):
        clear_page_labels()

with c4:
    if st.button("↩️ Undo"):
        undo_action()

with c5:
    if st.button("💾 Save"):
        save_current_labels()

with c6:
    if st.button("🔄 Refresh"):
        AppState.refresh_data()

with c7:
    page_units = set()
    for key, label in st.session_state.labels.items():
        if key[0] == st.session_state.current_page:
            page_units.add(label.unit_id)
    st.metric("Units (Page)", len(page_units))

# Rule-based labeling
st.markdown("#### 🤖 Rule-Based Labeling")
rc1, rc2 = st.columns(2)

with rc1:
    if st.button("🔢 Label Digits"):
        label_digits_as_units()

with rc2:
    if st.button("📝 Label Remaining"):
        label_rest_as_single_units()


# Main content
if st.session_state.current_doc:
    spans = AppState.get_current_spans()
    pdf_path = Path("data/docs") / st.session_state.current_doc / "document.pdf"
    
    if spans and pdf_path.exists():
        max_pages = max(s.page_number for s in spans) if spans else 1
        
        new_page = st.number_input(
            "Page", 
            min_value=1, 
            max_value=max_pages, 
            value=st.session_state.get('current_page', 1),
            key="page_input"
        )
        
        if new_page != st.session_state.get('current_page', 1):
            st.session_state.current_page = new_page
            st.session_state.selected_spans = []
            st.rerun()
        
        page_spans = [s for s in spans if s.page_number == st.session_state.current_page]
        
        left, right = st.columns([3, 2])
        
        with left:
            st.subheader("Document Preview (Overlay)")
            
            zoom = st.session_state.get('zoom_level', 100) / 100.0
            pdf_bytes = render_pdf_page(pdf_path, st.session_state.current_page, zoom)
            base_img = Image.open(io.BytesIO(pdf_bytes))
            
            page_predictions = None
            if st.session_state.get('predictions'):
                page_predictions = {
                    k: v for k, v in st.session_state.predictions.items() 
                    if k[0] == st.session_state.current_page
                }
            
            overlay_img = draw_overlay(
                base_img,
                page_spans,
                st.session_state.labels,
                st.session_state.selected_spans,
                predictions=page_predictions,
                show_heatmap=show_heatmap,
                show_unit_numbers=show_unit_numbers,
                zoom=zoom
            )
            
            st.image(overlay_img, use_container_width=True)
            
            new_zoom = st.slider(
                "Zoom", 
                min_value=50, 
                max_value=200, 
                value=st.session_state.get('zoom_level', 100),
                step=10,
                key="zoom_slider"
            )
            
            if new_zoom != st.session_state.get('zoom_level', 100):
                st.session_state.zoom_level = new_zoom
                st.rerun()

        with right:
            st.subheader("🔍 Span Selection")
            
            if page_spans:
                with st.container(height=600):
                    for i, span in enumerate(page_spans):
                        key = (span.page_number, span.span_id)
                        is_selected = span.span_id in st.session_state.selected_spans
                        is_labeled = key in st.session_state.labels
                        
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            
                            with col1:
                                button_label = "☑️" if is_selected else "⬜"
                                if st.button(
                                    button_label, 
                                    key=f"toggle_{span.page_number}_{span.span_id}_{i}"
                                ):
                                    toggle_span_selection(span.span_id)
                                    st.rerun()
                            
                            with col2:
                                text_preview = span.text[:60] + "..." if len(span.text) > 60 else span.text
                                
                                if is_labeled:
                                    label = st.session_state.labels[key]
                                    boundary_icon = "🆕" if label.boundary == "new" else "➡️"
                                    
                                    unit_display = label.unit_id
                                    if show_unit_numbers:
                                        unit_num = label.unit_id.split('_')[-1] if '_' in label.unit_id else label.unit_id
                                        unit_display = f"#{unit_num}"
                                    
                                    st.markdown(f"{boundary_icon} `{span.span_id}` - {text_preview}")
                                    
                                    if show_unit_numbers:
                                        st.caption(f"Unit: {unit_display} ({label.unit_id})")
                                    else:
                                        st.caption(f"Unit: {label.unit_id}")
                                else:
                                    color = "🔵" if is_selected else "⚪"
                                    st.markdown(f"{color} `{span.span_id}` - {text_preview}")
                                
                                if st.session_state.get('predictions') and key in st.session_state.predictions:
                                    prob = st.session_state.predictions[key]
                                    st.caption(f"🎯 Boundary prob: {prob:.3f}")
                            
                            st.markdown("---")
            else:
                st.info("No spans found on this page")
    
    else:
        st.warning("No document loaded or PDF not found")
        if st.button("🔄 Try to Load Document"):
            AppState.refresh_data()
            st.rerun()
else:
    st.info("Please upload a document first using the Upload page")
    if st.button("🔄 Check for New Documents"):
        data_dir = Path("data/docs")
        if data_dir.exists():
            doc_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
            if doc_dirs:
                latest_doc = max(doc_dirs, key=lambda d: d.stat().st_mtime)
                AppState.load_document_data(latest_doc.name)
                st.rerun()