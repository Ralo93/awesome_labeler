# app/pages/01_Label.py
import streamlit as st
from pathlib import Path
import sys
from PIL import Image
import io
import numpy as np
import pandas as pd
import joblib
import copy

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.state import AppState
from app.components.overlay import draw_overlay
from app.components.feature_panel import show_feature_panel
from core.io import load_spans, load_labels, save_labels
from core.pdf_processor import render_pdf_page
from core.schematas import Label
from core.features import compute_pairwise_features

st.set_page_config(page_title="Label", page_icon="🏷️", layout="wide")
st.markdown(
    """
    <script>
    document.addEventListener("keydown", function(event) {
        // Use lowercase 'm' for merge
        if (event.key === "m" || event.key === "M") {
            // Find the hidden merge button and click it
            var btn = window.parent.document.querySelector('button[data-merge-key]');
            if (btn) { btn.click(); }
        }
    });
    </script>
    """,
    unsafe_allow_html=True
)

AppState.init()

st.title("🏷️ Labeling Interface")

# --- CSS tweaks for smaller span/unit buttons ---


# Initialize label tracking
if 'unit_counter' not in st.session_state:
    st.session_state.unit_counter = 0

if 'model' not in st.session_state:
    st.session_state.model = None

if 'model_name' not in st.session_state:
    st.session_state.model_name = None

def load_model(model_path: Path):
    """Load a trained boundary model"""
    try:
        model_data = joblib.load(model_path)
        
        # Handle different model formats
        if isinstance(model_data, dict):
            # New format with metadata
            st.session_state.model = model_data.get('model')
            st.session_state.model_metadata = model_data.get('metadata', {})
            st.session_state.model_features = model_data.get('feature_columns', [])
        else:
            # Old format - just the model
            st.session_state.model = model_data
            st.session_state.model_metadata = {}
            st.session_state.model_features = []
        
        st.session_state.model_name = model_path.stem.replace('_model', '')
        return True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return False

def compute_features_for_model(prev_span, curr_span):
    """Compute features matching the trained model's expectations"""
    features = compute_pairwise_features(prev_span, curr_span)
    
    # Add missing features that the model expects
    # Linebreak flag - check if there's a significant vertical gap suggesting a line break
    avg_height = (prev_span.height + curr_span.height) / 2 if (prev_span.height + curr_span.height) > 0 else 20
    vertical_gap = curr_span.y_bottom - prev_span.y_bottom
    features['linebreak_flag'] = 1 if vertical_gap > avg_height * 0.5 else 0
    
    # Hyphenation flag - already computed as prev_ends_with_hyphen, but also add the specific flag
    features['hyphenation_flag'] = features.get('prev_ends_with_hyphen', 0)
    
    return features

def apply_model_predictions(page_only: bool = True, overwrite: bool = False):
    """Apply model predictions to create semantic units"""
    if not st.session_state.model:
        st.warning("No model loaded")
        return False
    
    spans = load_spans(st.session_state.current_doc)
    
    if page_only:
        # Filter to current page
        target_spans = [s for s in spans if s.page_number == st.session_state.current_page]
        pages_to_process = [st.session_state.current_page]
    else:
        # Process all pages
        target_spans = spans
        pages_to_process = sorted(set(s.page_number for s in spans))
    
    total_boundaries = 0
    created_units = 0
    skipped_spans = 0
    
    # Store old state for undo
    old_labels = copy.deepcopy(st.session_state.labels)
    
    for page_num in pages_to_process:
        page_spans = sorted([s for s in target_spans if s.page_number == page_num],
                           key=lambda s: s.reading_order)
        
        if len(page_spans) < 2:
            continue
        
        # Compute features for ALL consecutive pairs
        rows = []
        span_pairs = []
        
        for i in range(1, len(page_spans)):
            prev_span = page_spans[i-1]
            curr_span = page_spans[i]
            
            # Check if already labeled (for statistics)
            if not overwrite and (page_num, curr_span.span_id) in st.session_state.labels:
                skipped_spans += 1
                continue
            
            features = compute_features_for_model(prev_span, curr_span)
            rows.append(features)
            span_pairs.append((prev_span, curr_span))
        
        if not rows:
            if skipped_spans > 0:
                st.info(f"All {skipped_spans} spans on page {page_num} are already labeled. Use 'Clear Page' first or enable overwrite.")
            continue
        
        # Make predictions
        df = pd.DataFrame(rows)
        
        # Get probability of boundary
        try:
            if hasattr(st.session_state.model, 'predict_proba'):
                probs = st.session_state.model.predict_proba(df)[:, 1]
            else:
                # Fallback for models without predict_proba
                probs = st.session_state.model.predict(df)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            # Diagnostic info
            st.write("Features in data:", df.columns.tolist())
            if hasattr(st.session_state.model, 'feature_names_in_'):
                st.write("Features expected by model:", st.session_state.model.feature_names_in_.tolist())
            continue
        
        # Apply threshold (default 0.5, can be adjusted)
        threshold = st.session_state.get('model_threshold', 0.5)
        boundaries = probs >= threshold
        
        # Create semantic units based on boundaries
        current_unit_id = None
        
        for idx, (prev_span, curr_span) in enumerate(span_pairs):
            is_boundary = boundaries[idx]
            
            # Check if previous span is labeled
            prev_key = (page_num, prev_span.span_id)
            if prev_key not in st.session_state.labels:
                # Label the previous span if not already done
                if current_unit_id is None:
                    st.session_state.unit_counter += 1
                    current_unit_id = f"SU_{st.session_state.unit_counter:04d}"
                    created_units += 1
                
                st.session_state.labels[prev_key] = Label(
                    doc_id=st.session_state.current_doc,
                    page_number=page_num,
                    span_id=prev_span.span_id,
                    unit_id=current_unit_id,
                    boundary="new" if current_unit_id.endswith(f"{st.session_state.unit_counter:04d}") else "continue",
                    confidence=int(probs[idx] * 5) if idx < len(probs) else 3
                )
            
            # Label current span
            curr_key = (page_num, curr_span.span_id)
            
            if is_boundary:
                # Start new unit
                st.session_state.unit_counter += 1
                current_unit_id = f"SU_{st.session_state.unit_counter:04d}"
                boundary_label = "new"
                created_units += 1
                total_boundaries += 1
            else:
                # Continue current unit
                if current_unit_id is None:
                    st.session_state.unit_counter += 1
                    current_unit_id = f"SU_{st.session_state.unit_counter:04d}"
                    created_units += 1
                boundary_label = "continue"
            
            st.session_state.labels[curr_key] = Label(
                doc_id=st.session_state.current_doc,
                page_number=page_num,
                span_id=curr_span.span_id,
                unit_id=current_unit_id,
                boundary=boundary_label,
                confidence=int((1 - abs(probs[idx] - 0.5) * 2) * 5) if idx < len(probs) else 3
            )
        
        # Label any remaining spans on the page
        for span in page_spans:
            key = (page_num, span.span_id)
            if key not in st.session_state.labels:
                if current_unit_id is None:
                    st.session_state.unit_counter += 1
                    current_unit_id = f"SU_{st.session_state.unit_counter:04d}"
                    created_units += 1
                
                st.session_state.labels[key] = Label(
                    doc_id=st.session_state.current_doc,
                    page_number=page_num,
                    span_id=span.span_id,
                    unit_id=current_unit_id,
                    boundary="continue",
                    confidence=1
                )
    
    # Add to undo stack
    AppState.add_to_undo_stack({
        'type': 'apply_model',
        'old_labels': old_labels,
        'new_labels': copy.deepcopy(st.session_state.labels)
    })
    
    AppState.mark_dirty()
    
    if page_only:
        st.success(f"Applied model to page {st.session_state.current_page}: {created_units} units created, {total_boundaries} boundaries detected")
    else:
        st.success(f"Applied model to document: {created_units} units created across {len(pages_to_process)} pages")
    
    return True
def merge_selected_spans():
    # Allow 1+ selected spans (single = create its own unit)
    if not st.session_state.selected_spans:
        st.warning("Please select at least 1 span to create or merge a unit")
        return
    
    # Save state for undo
    old_labels = copy.deepcopy(st.session_state.labels)
    
    spans = load_spans(st.session_state.current_doc)
    page_spans = sorted(
        [s for s in spans if s.page_number == st.session_state.current_page],
        key=lambda s: (s.line_id if s.line_id is not None else 10**9,
                       s.reading_order if s.reading_order is not None else 10**9)
    )
    
    selected = [s for s in page_spans if s.span_id in st.session_state.selected_spans]
    selected.sort(key=lambda s: s.reading_order)
    
    if selected:
        st.session_state.unit_counter += 1
        unit_id = f"SU_{st.session_state.unit_counter:04d}"
        
        for i, span in enumerate(selected):
            key = (st.session_state.current_page, span.span_id)
            st.session_state.labels[key] = Label(
                doc_id=st.session_state.current_doc,
                page_number=st.session_state.current_page,
                span_id=span.span_id,
                unit_id=unit_id,
                boundary="new" if i == 0 else "continue",
                confidence=st.session_state.get('confidence_level', 3) if st.session_state.confidence_mode else None
            )
        
        # Add undo action
        AppState.add_to_undo_stack({
            'type': 'merge_spans' if len(selected) > 1 else 'create_unit_single',
            'old_labels': old_labels,
            'new_labels': copy.deepcopy(st.session_state.labels)
        })
        
        AppState.mark_dirty()
        st.session_state.selected_spans = []
        st.success(f"Created unit {unit_id} with {len(selected)} span{'s' if len(selected)!=1 else ''}")
        st.rerun()

import re

def is_number_only(text: str) -> bool:
    """Return True if text is ONLY a number (int or decimal), allowing optional sign and whitespace."""
    if text is None:
        return False
    # Accepts: "123", "-42", "+7", "3.14", "2,5" (comma or dot decimal), with surrounding spaces
    return bool(re.match(r'^\s*[+-]?\d+(?:[.,]\d+)?\s*$', text))


def enforce_digit_units():
    """Put every span containing a digit into its own semantic unit on the current page.
       Overwrites labels for those spans and fixes boundaries of affected original units.
    """
    spans = load_spans(st.session_state.current_doc)
    page_spans = sorted(
        [s for s in spans if s.page_number == st.session_state.current_page],
        key=lambda s: (s.line_id if s.line_id is not None else 10**9,
                       s.reading_order if s.reading_order is not None else 10**9)
    )

    # Find spans that contain at least one digit
    # Find spans whose text is ONLY a number
    digit_spans = [s for s in page_spans if is_number_only(getattr(s, "text", ""))]


    if not digit_spans:
        st.info("No digit-containing spans found on this page.")
        return

    # Save state for undo
    old_labels = copy.deepcopy(st.session_state.labels)

    # Map original units on page (before changes) to their spans (ordered)
    original_units = {}
    for s in page_spans:
        k = (st.session_state.current_page, s.span_id)
        if k in st.session_state.labels:
            uid = st.session_state.labels[k].unit_id
        else:
            # Also consider previously saved labels (if any) to track pre-change memberships
            label_pre = next((lbl for lbl in load_labels(st.session_state.current_doc)
                              if lbl.page_number == st.session_state.current_page and lbl.span_id == s.span_id), None)
            uid = label_pre.unit_id if label_pre else None
        if uid:
            original_units.setdefault(uid, []).append(s)

    # Assign each digit span to its own new unit
    for s in digit_spans:
        st.session_state.unit_counter += 1
        new_uid = f"SU_{st.session_state.unit_counter:04d}"
        key = (st.session_state.current_page, s.span_id)
        st.session_state.labels[key] = Label(
            doc_id=st.session_state.current_doc,
            page_number=st.session_state.current_page,
            span_id=s.span_id,
            unit_id=new_uid,
            boundary="new",
            confidence=1 if st.session_state.confidence_mode else None
        )

    # Fix boundary flags for any original units that lost one or more digit spans
    # Ensure the first remaining span in each affected unit is marked "new", others "continue"
    affected_units = set()
    for s in digit_spans:
        # Find which (old) unit this span was in based on old_labels
        k = (st.session_state.current_page, s.span_id)
        if k in old_labels:
            affected_units.add(old_labels[k].unit_id)

    for uid in affected_units:
        if uid not in original_units:
            continue
        remaining = [s for s in original_units[uid] if s not in digit_spans]
        # Keep old unit id on remaining spans; just fix boundary flags
        for i, s in enumerate(remaining):
            k = (s.page_number, s.span_id)
            if k in st.session_state.labels and st.session_state.labels[k].unit_id == uid:
                st.session_state.labels[k].boundary = "new" if i == 0 else "continue"

    # Add undo action
    AppState.add_to_undo_stack({
        'type': 'digits_to_single_units',
        'old_labels': old_labels,
        'new_labels': copy.deepcopy(st.session_state.labels)
    })

    AppState.mark_dirty()
    st.success(f"Assigned {len(digit_spans)} digit span{'s' if len(digit_spans)!=1 else ''} to their own semantic unit{'s' if len(digit_spans)!=1 else ''}.")
    st.rerun()


def auto_label_remaining():
    # Save state for undo
    old_labels = copy.deepcopy(st.session_state.labels)
    
    spans = load_spans(st.session_state.current_doc)
    page_spans = [s for s in spans if s.page_number == st.session_state.current_page]
    
    labeled_count = 0
    for span in page_spans:
        key = (st.session_state.current_page, span.span_id)
        if key in st.session_state.labels:
            continue
        
        st.session_state.unit_counter += 1
        unit_id = f"SU_{st.session_state.unit_counter:04d}"
        
        st.session_state.labels[key] = Label(
            doc_id=st.session_state.current_doc,
            page_number=st.session_state.current_page,
            span_id=span.span_id,
            unit_id=unit_id,
            boundary="new",
            confidence=1 if st.session_state.confidence_mode else None
        )
        labeled_count += 1
    
    if labeled_count > 0:
        # Add undo action
        AppState.add_to_undo_stack({
            'type': 'auto_label',
            'old_labels': old_labels,
            'new_labels': copy.deepcopy(st.session_state.labels)
        })
        
        AppState.mark_dirty()
        st.success(f"Auto-labeled {labeled_count} remaining spans as individual units")
        st.rerun()
    else:
        st.info("All spans are already labeled")

def clear_page_labels():
    # Save state for undo
    old_labels = copy.deepcopy(st.session_state.labels)
    
    keys_to_remove = [k for k in st.session_state.labels if k[0] == st.session_state.current_page]
    
    for k in keys_to_remove:
        del st.session_state.labels[k]
    
    if keys_to_remove:
        # Add undo action
        AppState.add_to_undo_stack({
            'type': 'clear_page',
            'old_labels': old_labels,
            'new_labels': copy.deepcopy(st.session_state.labels)
        })
        
        AppState.mark_dirty()
        st.success(f"Cleared {len(keys_to_remove)} labels")
        st.rerun()

def split_unit():
    if not st.session_state.selected_spans:
        st.warning("Please select a span where you want to split")
        return
    
    # Save state for undo
    old_labels = copy.deepcopy(st.session_state.labels)
    
    spans = load_spans(st.session_state.current_doc)
    page_spans = sorted([s for s in spans if s.page_number == st.session_state.current_page],
                       key=lambda s: s.reading_order)
    
    for span_id in st.session_state.selected_spans:
        key = (st.session_state.current_page, span_id)
        if key not in st.session_state.labels:
            continue
        
        original_unit = st.session_state.labels[key].unit_id
        
        unit_spans = []
        for span in page_spans:
            span_key = (span.page_number, span.span_id)
            if span_key in st.session_state.labels and st.session_state.labels[span_key].unit_id == original_unit:
                unit_spans.append(span)
        
        split_span = next((s for s in unit_spans if s.span_id == span_id), None)
        if not split_span:
            continue
        
        split_index = unit_spans.index(split_span)
        
        if split_index < len(unit_spans) - 1:
            st.session_state.unit_counter += 1
            new_unit_id = f"SU_{st.session_state.unit_counter:04d}"
            
            for i, span in enumerate(unit_spans[split_index:]):
                span_key = (span.page_number, span.span_id)
                st.session_state.labels[span_key].unit_id = new_unit_id
                st.session_state.labels[span_key].boundary = "new" if i == 0 else "continue"
            
            # Add undo action
            AppState.add_to_undo_stack({
                'type': 'split_unit',
                'old_labels': old_labels,
                'new_labels': copy.deepcopy(st.session_state.labels)
            })
            
            st.success(f"Split unit {original_unit} into two units")
            AppState.mark_dirty()
    
    st.session_state.selected_spans = []
    st.rerun()

def undo_action():
    """Custom undo handler for label operations"""
    if st.session_state.undo_stack:
        action = st.session_state.undo_stack.pop()
        
        # Save current state to redo stack
        current_state = {
            'type': action['type'],
            'old_labels': action['new_labels'],
            'new_labels': action['old_labels']
        }
        st.session_state.redo_stack.append(current_state)
        
        # Restore old labels
        st.session_state.labels = copy.deepcopy(action['old_labels'])
        st.success("Undone last action")
        return True
    else:
        st.warning("Nothing to undo")
        return False

def save_current_labels():
    if st.session_state.current_doc and st.session_state.labels:
        save_labels(st.session_state.current_doc, list(st.session_state.labels.values()))
        AppState.mark_clean()
        st.success("Labels saved!")
        return True
    return False

def next_uncertain():
    if st.session_state.predictions:
        # Find the boundary with probability closest to 0.5
        most_uncertain = None
        min_certainty = 1.0
        
        for (span_prev, span_curr), prob in st.session_state.predictions.items():
            certainty = abs(prob - 0.5)
            if certainty < min_certainty:
                min_certainty = certainty
                most_uncertain = (span_prev, span_curr, prob)
        
        if most_uncertain:
            span_prev, span_curr, prob = most_uncertain
            st.session_state.selected_spans = [span_prev, span_curr]
            st.info(f"Most uncertain boundary: {span_prev} & {span_curr} (P={prob:.3f})")
            st.rerun()

# Model loading sidebar
with st.sidebar:
    st.header("🤖 Model Management")
    
    models_dir = Path("/Users/rvonlottne001/repositories/GPT4Gov-Doc_Translation/backend_pdf/training_SU_model/app/models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*_model.pkl"))
        
        if model_files:
            model_names = [f.stem.replace('_model', '') for f in model_files]
            
            selected_model = st.selectbox(
                "Select Model",
                ["None"] + model_names,
                index=0 if st.session_state.model_name is None else 
                      (model_names.index(st.session_state.model_name) + 1 
                       if st.session_state.model_name in model_names else 0)
            )
            
            if selected_model != "None":
                model_path = models_dir / f"{selected_model}_model.pkl"
                if st.button("📥 Load Model", use_container_width=True):
                    if load_model(model_path):
                        st.success(f"Loaded model: {selected_model}")
                        st.rerun()
            
            if st.session_state.model:
                st.success(f"✅ Model loaded: {st.session_state.model_name}")
                
                # Model application controls
                st.divider()
                st.subheader("Apply Model")
                
                st.session_state.model_threshold = st.slider(
                    "Boundary Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Probability threshold for detecting boundaries"
                )
                
                overwrite = st.checkbox(
                    "Overwrite existing labels",
                    value=False,
                    help="Replace existing labels with model predictions"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📄 Apply to Page", use_container_width=True,
                                help="Apply model to current page only"):
                        if apply_model_predictions(page_only=True, overwrite=overwrite):
                            st.rerun()
                
                with col2:
                    if st.button("📚 Apply to Document", use_container_width=True,
                                help="Apply model to entire document"):
                        if apply_model_predictions(page_only=False, overwrite=overwrite):
                            st.rerun()
                
                # Model info
                if st.session_state.get('model_metadata'):
                    st.divider()
                    st.caption("Model Info")
                    meta = st.session_state.model_metadata
                    if 'test_accuracy' in meta:
                        st.metric("Test Accuracy", f"{meta['test_accuracy']:.3f}")
                    if 'feature_count' in meta:
                        st.metric("Features", meta['feature_count'])
        else:
            st.info("No trained models found in app/models/")
    else:
        st.warning("Models directory not found")

# Toolbar
st.markdown("### 🎯 Labeling Workflow")
st.info("**Select spans** that belong together → **Merge** them → **Auto-label** remaining spans")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

with c1:
    if st.button("🔗 **Merge Selected**", 
                 help="Merge selected spans into one unit — or create a unit if only one is selected (M)", 
                 use_container_width=True, 
                 type="primary", 
                 key="merge_button"):
        merge_selected_spans()

    # Add a hidden attribute so our JS knows which button to trigger
    st.markdown(
        """
        <script>
        // Add a marker attribute to the merge button
        const mergeBtn = window.parent.document.querySelector('button[kind][data-testid="stButton"][aria-label="🔗 **Merge Selected**"]');
        if (mergeBtn && !mergeBtn.hasAttribute("data-merge-key")) {
            mergeBtn.setAttribute("data-merge-key", "true");
        }
        </script>
        """,
        unsafe_allow_html=True
    )


with c2:
    if st.button("🤖 Auto-Label Rest", help="Label all remaining spans as individual units", 
                use_container_width=True):
        auto_label_remaining()

with c3:
    if st.button("✂️ Split Unit", help="Split unit at selected span (S)", use_container_width=True):
        split_unit()

with c4:
    if st.button("🗑️ Clear Page", help="Clear all labels on this page", use_container_width=True):
        clear_page_labels()

with c5:
    if st.button("↩️ Undo", help="Undo last action (Ctrl+Z)", use_container_width=True):
        if undo_action():
            st.rerun()

with c6:
    if st.button("💾 Save", help="Save labels", use_container_width=True):
        save_current_labels()

with c7:
    if st.button("🔢 DigitSU", help="Put every digit-containing span on this page into its own semantic unit",
                 use_container_width=True):
        enforce_digit_units()

# Confidence slider
if st.session_state.confidence_mode:
    st.session_state.confidence_level = st.slider(
        "Confidence for manual merges", 1, 5, 3,
        help="Confidence level for manually merged units (auto-labels get confidence=1)"
    )

# Main content
if st.session_state.current_doc:
    spans = load_spans(st.session_state.current_doc)
    page_spans = [s for s in spans if s.page_number == st.session_state.current_page]
    
    # Load existing labels once
    if not st.session_state.labels:
        for label in load_labels(st.session_state.current_doc):
            key = (label.page_number, label.span_id)
            st.session_state.labels[key] = label
            unit_num = int(label.unit_id.split('_')[1]) if '_' in label.unit_id else 0
            st.session_state.unit_counter = max(st.session_state.unit_counter, unit_num)
    
    # --- Page Navigation (Prev / Next / Jump) ---
    total_pages = max((s.page_number for s in spans), default=1)
    
    nav_l, nav_c, nav_r = st.columns([1, 2, 1])
    
    with nav_l:
        disabled_prev = st.session_state.current_page <= 1
        if st.button("⬅️ Previous", use_container_width=True, disabled=disabled_prev):
            AppState.set_page(max(1, st.session_state.current_page - 1))
            st.rerun()
    
    with nav_c:
        new_page = st.number_input(
            "Page", min_value=1, max_value=total_pages,
            value=int(st.session_state.current_page), step=1
        )
        if int(new_page) != st.session_state.current_page:
            AppState.set_page(int(new_page))
            st.rerun()
        st.caption(f"of {total_pages} pages")
    
    with nav_r:
        disabled_next = st.session_state.current_page >= total_pages
        if st.button("Next ➡️", use_container_width=True, disabled=disabled_next):
            AppState.set_page(min(total_pages, st.session_state.current_page + 1))
            st.rerun()
    
    pdf_path = Path("data/docs") / st.session_state.current_doc / "document.pdf"
    
    # Stats
    labeled_spans = sum(1 for s in page_spans if (st.session_state.current_page, s.span_id) in st.session_state.labels)
    total_spans = len(page_spans)
    
    m1, m2, m3 = st.columns([2, 1, 1])
    with m1:
        st.progress(labeled_spans / total_spans if total_spans > 0 else 0,
                   text=f"Page Progress: {labeled_spans}/{total_spans} spans labeled")
    with m2:
        unique_units = len(set(l.unit_id for l in st.session_state.labels.values() 
                             if l.page_number == st.session_state.current_page))
        st.metric("Units on page", unique_units)
    with m3:
        st.metric("Selected", len(st.session_state.selected_spans))
    
    # === Side-by-side: left overlay preview, right span selector ===
    left, right = st.columns([3, 2], gap="large")
    
    with left:
        st.subheader("Document Preview (Overlay)")
        
        if pdf_path.exists() and page_spans:
            zoom = st.session_state.zoom_level / 100.0
            pdf_bytes = render_pdf_page(pdf_path, st.session_state.current_page, zoom)
            base_img = Image.open(io.BytesIO(pdf_bytes))
            
            overlay_img = draw_overlay(
                base_img,
                page_spans,
                st.session_state.labels,
                st.session_state.selected_spans,
                predictions=st.session_state.predictions,
                show_heatmap=bool(st.session_state.predictions),
                zoom=1.0  # already zoomed
            )
            
            st.image(overlay_img, use_container_width=True)
            st.caption("💡 Select spans on the right, then use Merge / Split.")
        elif not pdf_path.exists():
            st.warning("PDF file not found. Please re-upload the document.")
        else:
            st.info("No spans detected on this page.")
    
    with right:
        st.subheader("Span Selector & Units")
        
        if page_spans:
            # Group spans by lines (approximate) for layout
            lines = {}
            for span in page_spans:
                line_key = round(span.y_bottom / 15) * 15
                lines.setdefault(line_key, []).append(span)
            
            for _, line_spans in sorted(lines.items()):
                line_spans.sort(key=lambda s: s.x_center)
                cols = st.columns(len(line_spans)) if line_spans else []
                
                for i, span in enumerate(line_spans):
                    with cols[i]:
                        is_selected = span.span_id in st.session_state.selected_spans
                        key = (st.session_state.current_page, span.span_id)
                        label = st.session_state.labels.get(key)
                        
                        txt = span.text[:15] + "..." if len(span.text) > 15 else span.text
                        if label and label.boundary == "new":
                            txt = f"[{label.unit_id}] {txt}"
                        
                        if st.button(
                            txt,
                            key=f"span_{span.span_id}",
                            type="primary" if is_selected else "secondary",
                            help=f"ID: {span.span_id}\nUnit: {label.unit_id if label else 'None'}\nText: {span.text}",
                            use_container_width=True
                        ):
                            AppState.toggle_span_selection(span.span_id)
                            st.rerun()
            
            # Quick actions for current selection
            if st.session_state.selected_spans:
                st.divider()
                qa1, qa2, qa3 = st.columns(3)
                with qa1:
                    if st.button("Clear Selection", use_container_width=True):
                        st.session_state.selected_spans = []
                        st.rerun()
                with qa2:
                    if st.button("Select All", use_container_width=True):
                        st.session_state.selected_spans = [s.span_id for s in page_spans]
                        st.rerun()
                with qa3:
                    if st.button("Invert Selection", use_container_width=True):
                        current = set(st.session_state.selected_spans)
                        all_spans = set(s.span_id for s in page_spans)
                        st.session_state.selected_spans = list(all_spans - current)
                        st.rerun()
        
        # Feature panel
        if st.session_state.show_features and st.session_state.selected_spans:
            st.divider()
            st.subheader("Feature Details")
            show_feature_panel(
                st.session_state.selected_spans,
                page_spans,
                show_normalized=True,
                show_rules=st.session_state.show_rules
            )
    
    # Inference controls (only show if model loaded but not applied yet)
    if st.session_state.model:
        st.divider()
        st.subheader("🔮 Model Inference")
        
        i1, i2, i3 = st.columns(3)
        with i1:
            if st.button("Run Inference - Page", use_container_width=True):
                with st.spinner("Running inference..."):
                    rows, span_pairs = [], []
                    sorted_spans = sorted(page_spans, key=lambda s: s.reading_order)
                    
                    for i in range(1, len(sorted_spans)):
                        features = compute_features_for_model(sorted_spans[i-1], sorted_spans[i])
                        rows.append(features)
                        span_pairs.append((sorted_spans[i-1].span_id, sorted_spans[i].span_id))
                    
                    if rows:
                        df = pd.DataFrame(rows)
                        try:
                            if hasattr(st.session_state.model, 'predict_proba'):
                                probs = st.session_state.model.predict_proba(df)[:, 1]
                            else:
                                probs = st.session_state.model.predict(df)
                            st.session_state.predictions = dict(zip(span_pairs, probs))
                            st.success(f"Analyzed {len(probs)} potential boundaries")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Inference failed: {str(e)}")
                            st.write("Features in data:", df.columns.tolist())
                            if hasattr(st.session_state.model, 'feature_names_in_'):
                                st.write("Features expected by model:", st.session_state.model.feature_names_in_.tolist())
        
        with i2:
            if st.button("Find Uncertain", use_container_width=True):
                next_uncertain()
        
        with i3:
            if st.button("Clear Predictions", use_container_width=True):
                st.session_state.predictions = {}
                st.rerun()
else:
    st.warning("Please select a document from the main page or upload a new PDF")
    if st.button("📄 Upload New PDF", type="primary"):
        st.switch_page("pages/00_Upload.py")