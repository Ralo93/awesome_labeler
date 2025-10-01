import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import io

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

# Initialize label tracking
if 'unit_counter' not in st.session_state:
    st.session_state.unit_counter = 0

if 'model' not in st.session_state:
    st.session_state.model = None

if 'model_name' not in st.session_state:
    st.session_state.model_name = None

# Auto-load document if available but not loaded
if not st.session_state.current_doc:
    data_dir = Path("data/docs")
    if data_dir.exists():
        doc_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if doc_dirs:
            # Auto-load the most recently created document
            latest_doc = max(doc_dirs, key=lambda d: d.stat().st_mtime)
            AppState.load_document_data(latest_doc.name)

def load_model(model_path: Path):
    """Load a trained boundary model"""
    try:
        
        model = joblib.load(model_path)
        st.session_state.model = model
        st.session_state.model_name = model_path.name
        st.success(f"✅ Model loaded: {model_path.name}")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")

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
    feature_dicts = []  # Collect all features first
    keys_to_predict = []  # Track which spans we're predicting
    
    for local_idx, (global_idx, span) in enumerate(zip(span_indices, page_spans)):
        # Skip if already labeled and not overwriting
        key = (span.page_number, span.span_id)
        if key in st.session_state.labels and not overwrite:
            continue
        
        # Extract features using sliding window approach (pass global index)
        features = compute_sliding_window_features(spans, global_idx)
        feature_dicts.append(features)
        keys_to_predict.append((key, span))  # Store both key and span for later
    
    if not feature_dicts:
        st.info("No unlabeled spans to predict")
        return
    
    # Predict all at once using list of dictionaries
    try:
        # The model's predict_proba method handles list of dictionaries properly
        probs = st.session_state.model.predict_proba(feature_dicts)
        
        # Map predictions back to span keys
        for (key, span), prob in zip(keys_to_predict, probs[:, 1]):
            predictions[key] = prob
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return
    
    # Store predictions in session state
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}
    st.session_state.predictions.update(predictions)
    
    # Auto-create semantic units from predictions if enabled
    if auto_label and predictions:
        threshold = st.session_state.get('boundary_threshold', 0.5)
        current_unit_id = f"unit_{st.session_state.unit_counter + 1}"
        labels_to_add = {}
        
        # Sort spans by reading order for proper unit creation
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
            AppState.add_to_undo_stack({
                'labels_added': labels_to_add
            })
            
            # Update labels
            st.session_state.labels.update(labels_to_add)
            
            # Save labels
            save_labels(st.session_state.current_doc, st.session_state.labels)
            
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
    
    # Sort by reading order
    selected_span_objs.sort(key=lambda s: (s.page_number, s.reading_order))
    
    # Generate new unit ID
    st.session_state.unit_counter += 1
    unit_id = f"unit_{st.session_state.unit_counter}"
    
    # Create labels for all spans
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
    
    # Add to undo stack
    AppState.add_to_undo_stack({
        'labels_added': labels_to_add,
        'selected_spans': st.session_state.selected_spans.copy()
    })
    
    # Update labels
    st.session_state.labels.update(labels_to_add)
    st.session_state.selected_spans = []
    
    # Save labels
    save_labels(st.session_state.current_doc, st.session_state.labels)
    
    st.success(f"Merged {len(selected_span_objs)} spans into {unit_id}")

def auto_label_remaining():
    """Auto-label remaining unlabeled spans on the current page using model predictions"""
    if not st.session_state.model:
        st.error("No model loaded!")
        return
    
    spans = AppState.get_current_spans()
    if not spans:
        return
    
    # Get current page number
    current_page = st.session_state.current_page
    
    # Filter spans to current page only
    page_spans = [s for s in spans if s.page_number == current_page]
    
    if not page_spans:
        st.info("No spans on current page!")
        return
    
    # Find unlabeled spans on current page
    unlabeled_spans = []
    for span in page_spans:
        key = (span.page_number, span.span_id)
        if key not in st.session_state.labels:
            unlabeled_spans.append(span)
    
    if not unlabeled_spans:
        st.info("All spans on this page already labeled!")
        return
    
    # Generate unit IDs for high-confidence boundaries
    threshold = st.session_state.get('boundary_threshold', 0.5)
    current_unit_id = f"unit_{st.session_state.unit_counter + 1}"
    labels_to_add = {}
    
    for span in unlabeled_spans:
        key = (span.page_number, span.span_id)
        
        # Get prediction
        prob = st.session_state.predictions.get(key, 0.0)
        
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
    AppState.add_to_undo_stack({
        'labels_added': labels_to_add
    })
    
    # Update labels
    st.session_state.labels.update(labels_to_add)
    
    # Save labels
    save_labels(st.session_state.current_doc, st.session_state.labels)
    
    st.success(f"Auto-labeled {len(labels_to_add)} spans on page {current_page}")

def clear_page_labels():
    """Clear all labels on current page"""
    page_num = st.session_state.current_page
    
    # Find labels to remove
    labels_to_remove = {}
    for key, label in st.session_state.labels.items():
        if key[0] == page_num:
            labels_to_remove[key] = label
    
    if not labels_to_remove:
        st.info("No labels to clear on this page")
        return
    
    # Add to undo stack
    AppState.add_to_undo_stack({
        'labels_removed': labels_to_remove
    })
    
    # Remove labels
    for key in labels_to_remove:
        del st.session_state.labels[key]
    
    # Save labels
    save_labels(st.session_state.current_doc, st.session_state.labels)
    
    st.success(f"Cleared {len(labels_to_remove)} labels from page {page_num}")

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
        
        # Store old label for undo
        if key in st.session_state.labels:
            labels_removed[key] = st.session_state.labels[key]
        
        # Create new unit
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
    
    # Add to undo stack
    AppState.add_to_undo_stack({
        'labels_added': labels_to_add,
        'labels_removed': labels_removed
    })
    
    # Update labels
    st.session_state.labels.update(labels_to_add)
    st.session_state.selected_spans = []
    
    # Save labels
    save_labels(st.session_state.current_doc, st.session_state.labels)
    
    st.success(f"Split {len(selected_span_objs)} spans into separate units")

def undo_action():
    """Undo last action"""
    AppState.undo()

def save_current_labels():
    """Save current labels to disk"""
    if st.session_state.current_doc:
        save_labels(st.session_state.current_doc, st.session_state.labels)
        st.success("Labels saved!")

def toggle_span_selection(span_id: str):
    """Toggle span selection and force rerun for instant feedback"""
    if span_id in st.session_state.selected_spans:
        st.session_state.selected_spans.remove(span_id)
    else:
        st.session_state.selected_spans.append(span_id)
    
    # Mark state as changed for instant overlay update
    AppState.mark_dirty()

def next_uncertain():
    """Navigate to next span with uncertain prediction"""
    if not st.session_state.predictions:
        st.warning("No predictions available")
        return
    
    # Find uncertain predictions (around 0.5 probability)
    uncertain_spans = []
    for key, prob in st.session_state.predictions.items():
        if 0.3 <= prob <= 0.7:  # Uncertain range
            uncertain_spans.append((key, abs(prob - 0.5)))
    
    if not uncertain_spans:
        st.info("No uncertain predictions found")
        return
    
    # Sort by uncertainty (closest to 0.5)
    uncertain_spans.sort(key=lambda x: x[1])
    
    # Navigate to most uncertain
    page_num, span_id = uncertain_spans[0][0]
    st.session_state.current_page = page_num
    st.session_state.selected_spans = [span_id]
    
    st.success(f"Navigated to uncertain span: {span_id}")

# Model loading sidebar
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
        
        # Load new document if selection changed
        if selected_doc != st.session_state.current_doc:
            AppState.load_document_data(selected_doc)
            st.rerun()  # Force immediate refresh
    else:
        st.info("No documents found. Upload a document first.")
    st.header("🤖 Model")
    
    model_dir = Path("models")
    if model_dir.exists():
        model_files = list(model_dir.glob("boundary_model*.pkl"))
        if model_files:
            model_options = ["None"] + [f.name for f in model_files]
            selected_model = st.selectbox("Select Model", model_options)
            
            if selected_model != "None" and selected_model != st.session_state.model_name:
                if st.button("Load Model"):
                    load_model(model_dir / selected_model)
        else:
            st.info("No models found in /models")
        
    if st.session_state.model:
            st.success(f"✅ {st.session_state.model_name}")
            
            # Prediction controls
            st.subheader("Predictions")
            
            # Boundary threshold - moved up so it's set before predictions
            st.session_state.boundary_threshold = st.slider(
                "Boundary Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('boundary_threshold', 0.5),
                step=0.05,
                help="Probability threshold for new unit boundaries"
            )
            
            # Check if current page has unlabeled spans
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
                if st.button("🔮 Predict & Label Page", help="Generate predictions and create semantic units"):
                    apply_model_predictions(page_only=True, auto_label=True)
                    st.rerun()  # Force refresh to show new units
            
            with col2:
                if st.button("👁️ Preview Only", help="Generate predictions without creating units"):
                    apply_model_predictions(page_only=True, auto_label=False)
            
            if st.button("🔮 Predict & Label All Pages", help="Generate predictions for all pages"):
                apply_model_predictions(page_only=False, auto_label=True)
                st.rerun()
            
            # Only show this if predictions exist but haven't been converted to labels
            if st.session_state.get('predictions'):
                unlabeled_with_predictions = sum(
                    1 for key in st.session_state.predictions 
                    if key not in st.session_state.labels
                )
                if unlabeled_with_predictions > 0:
                    if st.button(f"🏷️ Apply {unlabeled_with_predictions} Predictions", 
                            help="Convert existing predictions to semantic units"):
                        auto_label_remaining()
                        st.rerun()
            
            if st.button("🎯 Next Uncertain"):
                next_uncertain()
    
    st.divider()
    
    # Display options
    st.header("Display Options")
    show_unit_numbers = st.checkbox(
        "Show Unit Numbers", 
        value=True,
        help="Display semantic unit numbers as small blue boxes"
    )
    show_heatmap = st.checkbox(
        "Show Prediction Heatmap",
        value=bool(st.session_state.get('predictions')),
        help="Show model prediction confidence colors"
    )
def label_digits_as_units():
    """Label spans containing numeric values as individual units on current page (rule-based feature)"""
    import re
    
    spans = AppState.get_current_spans()
    if not spans:
        return
    
    # Get current page number
    current_page = st.session_state.current_page
    
    # Filter spans to current page only
    page_spans = [s for s in spans if s.page_number == current_page]
    
    if not page_spans:
        st.info("No spans on current page!")
        return
    
    # Comprehensive numeric patterns
    numeric_patterns = [
        r'^-?\d+$',                                    # Simple integers (123, -456)
        r'^-?\d+[.,]\d+$',                            # Decimals with . or , (4.93, 4,98)
        r'^-?\d{1,3}([\s,]\d{3})*([.,]\d+)?$',       # Thousands with comma (1,234.56)
        r'^-?\d{1,3}([\s.]\d{3})*(,\d+)?$',          # European thousands (1.234,56)
        r'^-?\d+([.,]\d+)?%$',                        # Percentages (45.3%, 45,3%)
        r'^-?\$?\d+([.,]\d{2})?$',                   # Currency amounts ($4.99, 4.99)
        r'^-?€?\d+([.,]\d{2})?$',                    # Euro amounts (€4,99)
        r'^-?£?\d+([.,]\d{2})?$',                    # Pound amounts (£4.99)
        r'^\d+[/-]\d+[/-]\d+$',                      # Dates (12/31/2024, 31-12-2024)
        r'^\d{1,2}:\d{2}(:\d{2})?$',                 # Times (12:45, 12:45:30)
        r'^\(\d+\)$',                                # Numbers in parentheses ((123))
        r'^\[\d+\]$',                                # Numbers in brackets ([123])
        r'^#\d+$',                                   # Issue/ticket numbers (#123)
        r'^\d+\.$',                                  # Numbers ending with period (list items: 1.)
        r'^-?\d+([.,]\d+)?\s*(k|K|m|M|b|B)$',       # Abbreviated numbers (1.5k, 2M)
    ]
    
    # Find unlabeled spans on current page that match numeric patterns
    digit_spans = []
    for span in page_spans:
        key = (span.page_number, span.span_id)
        if key not in st.session_state.labels:
            cleaned_text = span.text.strip()
            if cleaned_text:
                # Check if text matches any numeric pattern
                is_numeric = any(re.match(pattern, cleaned_text) for pattern in numeric_patterns)
                
                # Also check for simple numeric strings that might have spaces
                # (like page numbers "1 2 3" or ranges "1 - 5")
                if not is_numeric:
                    # Remove spaces and check if it's mostly digits
                    no_space = cleaned_text.replace(' ', '').replace('-', '').replace('–', '')
                    if no_space and re.match(r'^\d+$', no_space) and len(no_space) <= 10:
                        is_numeric = True
                
                if is_numeric:
                    digit_spans.append(span)
    
    if not digit_spans:
        st.info("No unlabeled numeric spans found on this page!")
        return
    
    # Create individual units for numeric spans
    labels_to_add = {}
    for span in digit_spans:
        key = (span.page_number, span.span_id)
        
        # Each numeric span gets its own unit
        st.session_state.unit_counter += 1
        unit_id = f"unit_{st.session_state.unit_counter}"
        
        label = Label(
            span_id=span.span_id,
            page_number=span.page_number,
            boundary="new",  # Each is a new unit
            unit_id=unit_id,
            confidence=1.0
        )
        labels_to_add[key] = label
    
    # Add to undo stack following state management pattern
    AppState.add_to_undo_stack({
        'labels_added': labels_to_add
    })
    
    # Update labels
    st.session_state.labels.update(labels_to_add)
    
    # Save labels
    save_labels(st.session_state.current_doc, st.session_state.labels)
    
    # Show what types of numbers were found (for debugging/verification)
    sample_numbers = [s.text.strip() for s in digit_spans[:5]]  # Show first 5 examples
    examples = ", ".join(sample_numbers) if sample_numbers else ""
    
    if examples:
        st.success(f"Labeled {len(labels_to_add)} numeric spans on page {current_page}. Examples: {examples}")
    else:
        st.success(f"Labeled {len(labels_to_add)} numeric spans on page {current_page}")


def label_rest_as_single_units():
    """Label all remaining unlabeled spans as individual units on current page"""
    spans = AppState.get_current_spans()
    if not spans:
        return
    
    # Get current page number
    current_page = st.session_state.current_page
    
    # Filter spans to current page only
    page_spans = [s for s in spans if s.page_number == current_page]
    
    if not page_spans:
        st.info("No spans on current page!")
        return
    
    # Find unlabeled spans on current page
    unlabeled_spans = []
    for span in page_spans:
        key = (span.page_number, span.span_id)
        if key not in st.session_state.labels:
            unlabeled_spans.append(span)
    
    if not unlabeled_spans:
        st.info("All spans on this page already labeled!")
        return
    
    # Create individual units for each unlabeled span
    labels_to_add = {}
    for span in unlabeled_spans:
        key = (span.page_number, span.span_id)
        
        # Each span gets its own unit
        st.session_state.unit_counter += 1
        unit_id = f"unit_{st.session_state.unit_counter}"
        
        label = Label(
            span_id=span.span_id,
            page_number=span.page_number,
            boundary="new",  # Each is a new unit
            unit_id=unit_id,
            confidence=1.0
        )
        labels_to_add[key] = label
    
    # Add to undo stack following state management pattern
    AppState.add_to_undo_stack({
        'labels_added': labels_to_add
    })
    
    # Update labels
    st.session_state.labels.update(labels_to_add)
    
    # Save labels
    save_labels(st.session_state.current_doc, st.session_state.labels)
    
    st.success(f"Labeled {len(labels_to_add)} spans as individual units on page {current_page}")

# Toolbar
st.markdown("### 🎯 Labeling Workflow")
st.info("**Select spans** that belong together → **Merge** them → **Auto-label** remaining spans")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

with c1:
    if st.button("🔗 Merge Selected", help="Merge selected spans into unit"):
        merge_selected_spans()

with c2:
    if st.button("✂️ Split Selected", help="Split each selected span into separate units"):
        split_unit()

with c3:
    if st.button("🗑️ Clear Page", help="Clear all labels on current page"):
        clear_page_labels()

with c4:
    if st.button("↩️ Undo", help="Undo last action"):
        undo_action()

with c5:
    if st.button("💾 Save", help="Save labels to disk"):
        save_current_labels()

with c6:
    if st.button("🔄 Refresh", help="Reload spans and labels"):
        AppState.refresh_data()

with c7:
    st.metric("Units", len(set(l.unit_id for l in st.session_state.labels.values())))

# Add rule-based labeling buttons below
st.markdown("#### 🤖 Rule-Based Labeling")
rc1, rc2 = st.columns(2)

with rc1:
    if st.button("🔢 Label Digits", help="Label spans containing only digits as units"):
        label_digits_as_units()

with rc2:
    if st.button("📝 Label Remaining", help="Label all remaining unlabeled spans as individual units"):
        label_rest_as_single_units()
# Main content
if st.session_state.current_doc:
    spans = AppState.get_current_spans()
    pdf_path = Path("data/docs") / st.session_state.current_doc / "document.pdf"
    
    if spans and pdf_path.exists():
        # Page navigation
        max_pages = max(s.page_number for s in spans) if spans else 1
        
        # Use callback for instant page change
        def on_page_change():
            st.session_state.selected_spans = []  # Clear selection on page change
        
        new_page = st.number_input(
            "Page", 
            min_value=1, 
            max_value=max_pages, 
            value=st.session_state.get('current_page', 1),
            key="page_input"
        )
        
        if new_page != st.session_state.get('current_page', 1):
            st.session_state.current_page = new_page
            st.session_state.selected_spans = []  # Clear selection on page change
            st.rerun()  # Force refresh for instant page change
        
        # Filter spans for current page
        page_spans = [s for s in spans if s.page_number == st.session_state.current_page]
        
        # Main layout
        left, right = st.columns([3, 2])
        
        with left:
            st.subheader("Document Preview (Overlay)")
            
            zoom = st.session_state.get('zoom_level', 100) / 100.0
            pdf_bytes = render_pdf_page(pdf_path, st.session_state.current_page, zoom)
            base_img = Image.open(io.BytesIO(pdf_bytes))
            
            # Get predictions for current page
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
            
            # Zoom control with instant feedback
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
                st.rerun()  # Force refresh for instant zoom
        # Replace the span selection section in 01_Label.py (starting from line ~437)

        with right:
            # Span selection interface with instant feedback
            st.subheader("🔍 Span Selection")
            
            if page_spans:
                # Use enumerate to ensure unique indices
                for i, span in enumerate(page_spans):
                    key = (span.page_number, span.span_id)
                    is_selected = span.span_id in st.session_state.selected_spans
                    is_labeled = key in st.session_state.labels
                    
                    # Use form for instant feedback
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            # Use button with unique key including index
                            button_label = "☑️" if is_selected else "⬜"
                            # Include index i to ensure uniqueness even if span_ids repeat
                            if st.button(
                                button_label, 
                                key=f"toggle_{span.page_number}_{span.span_id}_{i}",  # More unique key
                                help=f"Toggle selection of {span.span_id}"
                            ):
                                toggle_span_selection(span.span_id)
                                st.rerun()  # Force instant overlay refresh
                        
                        with col2:
                            # Show span text preview
                            text_preview = span.text[:60] + "..." if len(span.text) > 60 else span.text
                            
                            # Color coding based on state
                            if is_labeled:
                                label = st.session_state.labels[key]
                                boundary_icon = "🆕" if label.boundary == "new" else "➡️"
                                st.markdown(f"{boundary_icon} `{span.span_id}` - {text_preview}")
                                st.caption(f"Unit: {label.unit_id}")
                            else:
                                color = "🔵" if is_selected else "⚪"
                                st.markdown(f"{color} `{span.span_id}` - {text_preview}")
                            
                            # Show prediction if available
                            if st.session_state.get('predictions') and key in st.session_state.predictions:
                                prob = st.session_state.predictions[key]
                                st.caption(f"🎯 Boundary prob: {prob:.3f}")
            else:
                st.info("No spans found on this page")
            
            # Feature panel
            show_feature_panel(page_spans, st.session_state.labels, st.session_state.selected_spans)

    else:
        st.warning("No document loaded or PDF not found")
        if st.button("🔄 Try to Load Document"):
            AppState.refresh_data()
            st.rerun()
else:
    st.info("Please upload a document first using the Upload page")
    if st.button("🔄 Check for New Documents"):
        # Try to auto-load if documents are available
        data_dir = Path("data/docs")
        if data_dir.exists():
            doc_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
            if doc_dirs:
                latest_doc = max(doc_dirs, key=lambda d: d.stat().st_mtime)
                AppState.load_document_data(latest_doc.name)
                st.rerun()