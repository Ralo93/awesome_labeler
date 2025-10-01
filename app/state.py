import streamlit as st
from typing import Any, List, Dict, Optional, Tuple
from datetime import datetime
import json

from core.schematas import Span, Label
from core.io import save_labels

class AppState:
    """Centralized state management"""
    
    @staticmethod
    def init():
        """Initialize session state"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.current_doc = None
            st.session_state.current_page = 1
            st.session_state.zoom_level = 100
            st.session_state.selected_spans = []
            st.session_state.labels = {}
            st.session_state.spans = []
            st.session_state.model_path = None
            st.session_state.model = None
            st.session_state.predictions = {}
            st.session_state.show_features = False
            st.session_state.show_rules = False
            st.session_state.confidence_mode = False
            st.session_state.undo_stack = []
            st.session_state.redo_stack = []
            st.session_state.dirty = False
            st.session_state.last_save = datetime.now()
            st.session_state.unit_counter = 0
    
    @staticmethod
    def set_document(doc_id: str):
        """Set current document"""
        st.session_state.current_doc = doc_id
        st.session_state.current_page = 1
        st.session_state.selected_spans = []
        st.session_state.predictions = {}
    
    @staticmethod
    def set_page(page: int):
        """Set current page"""
        st.session_state.current_page = page
        st.session_state.selected_spans = []
    
    @staticmethod
    def toggle_span_selection(span_id: str):
        """Toggle span selection"""
        if span_id in st.session_state.selected_spans:
            st.session_state.selected_spans.remove(span_id)
        else:
            st.session_state.selected_spans.append(span_id)
    
    @staticmethod
    def add_to_undo_stack(action: Dict[str, Any]):
        """Add action to undo stack"""
        st.session_state.undo_stack.append(action)
        st.session_state.redo_stack = []  # Clear redo stack on new action
        st.session_state.dirty = True
        
        # Limit undo stack size
        if len(st.session_state.undo_stack) > 100:
            st.session_state.undo_stack.pop(0)
    
    @staticmethod
    def undo():
        """Undo last action"""
        if not st.session_state.undo_stack:
            st.warning("Nothing to undo")
            return
            
        action = st.session_state.undo_stack.pop()
        
        # Handle the action format used in 01_Label.py
        # Actions have 'labels_added', 'labels_removed', and optionally 'selected_spans'
        
        # Restore removed labels
        if 'labels_removed' in action:
            st.session_state.labels.update(action['labels_removed'])
        
        # Remove added labels
        if 'labels_added' in action:
            for key in action['labels_added']:
                if key in st.session_state.labels:
                    del st.session_state.labels[key]
        
        # Restore selection if applicable
        if 'selected_spans' in action:
            st.session_state.selected_spans = action['selected_spans']
        
        # Save the changes
        if st.session_state.current_doc:
            save_labels(st.session_state.current_doc, st.session_state.labels)
        
        # Add to redo stack
        st.session_state.redo_stack.append(action)
        
        st.success("Action undone")
    
    @staticmethod
    def redo():
        """Redo last undone action"""
        if not st.session_state.redo_stack:
            st.warning("Nothing to redo")
            return
            
        action = st.session_state.redo_stack.pop()
        
        # Re-apply the action
        # Actions have 'labels_added', 'labels_removed', and optionally 'selected_spans'
        
        # Re-add labels that were added
        if 'labels_added' in action:
            st.session_state.labels.update(action['labels_added'])
        
        # Re-remove labels that were removed
        if 'labels_removed' in action:
            for key in action['labels_removed']:
                if key in st.session_state.labels:
                    del st.session_state.labels[key]
        
        # Clear selection (as original action would)
        if 'selected_spans' in action:
            st.session_state.selected_spans = []
        
        # Save the changes
        if st.session_state.current_doc:
            save_labels(st.session_state.current_doc, st.session_state.labels)
        
        # Add back to undo stack
        st.session_state.undo_stack.append(action)
        
        st.success("Action redone")
    
    @staticmethod
    def mark_dirty():
        """Mark state as dirty (unsaved changes)"""
        st.session_state.dirty = True
    
    @staticmethod
    def mark_clean():
        """Mark state as clean (saved)"""
        st.session_state.dirty = False
        st.session_state.last_save = datetime.now()

    @staticmethod
    def clear_predictions():
        """Clear model predictions"""
        if 'predictions' in st.session_state:
            st.session_state.predictions = {}
    
    @staticmethod
    def get_page_predictions(page_num: int) -> Dict[Tuple[int, str], float]:
        """Get predictions for specific page"""
        if 'predictions' not in st.session_state:
            return {}
        
        return {
            k: v for k, v in st.session_state.predictions.items() 
            if k[0] == page_num
        }
    
    @staticmethod
    def get_current_spans() -> List[Span]:
        """Get spans for current document"""
        return st.session_state.get('spans', [])
    
    @staticmethod
    def load_document_data(doc_id: str):
        """Load spans and labels for document"""
        from core.io import load_spans, load_labels
        
        # Load spans and labels from disk
        st.session_state.spans = load_spans(doc_id)
        st.session_state.labels = load_labels(doc_id)
        st.session_state.current_doc = doc_id
        st.session_state.current_page = 1
        st.session_state.selected_spans = []
        st.session_state.predictions = {}
        
        # Initialize unit counter based on existing labels
        if st.session_state.labels:
            # Extract the maximum unit number from existing unit IDs
            max_unit_num = 0
            for label in st.session_state.labels.values():
                if label.unit_id.startswith('unit_'):
                    try:
                        unit_num = int(label.unit_id.split('_')[1])
                        max_unit_num = max(max_unit_num, unit_num)
                    except (ValueError, IndexError):
                        pass
            st.session_state.unit_counter = max_unit_num
        else:
            st.session_state.unit_counter = 0
        
        return st.session_state.spans, st.session_state.labels
    
    @staticmethod
    def refresh_data():
        """Refresh spans and labels from disk"""
        if st.session_state.current_doc:
            # Clear predictions when refreshing data
            AppState.clear_predictions()
            # Reload document data
            AppState.load_document_data(st.session_state.current_doc)