import streamlit as st
from typing import Any, List, Dict, Optional
from datetime import datetime
import json

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
        if st.session_state.undo_stack:
            action = st.session_state.undo_stack.pop()
            st.session_state.redo_stack.append(action)
            # Apply reverse action
            AppState._reverse_action(action)
    
    @staticmethod
    def redo():
        """Redo last undone action"""
        if st.session_state.redo_stack:
            action = st.session_state.redo_stack.pop()
            st.session_state.undo_stack.append(action)
            # Apply action
            AppState._apply_action(action)
    
    @staticmethod
    def _reverse_action(action: Dict[str, Any]):
        """Reverse an action"""
        if action['type'] == 'boundary_toggle':
            # Toggle boundary back
            span_id = action['span_id']
            old_value = action['old_value']
            # Restore old value in labels
            key = (st.session_state.current_page, span_id)
            if old_value:
                st.session_state.labels[key] = old_value
            else:
                st.session_state.labels.pop(key, None)
    
    @staticmethod
    def _apply_action(action: Dict[str, Any]):
        """Apply an action"""
        if action['type'] == 'boundary_toggle':
            # Apply new boundary value
            span_id = action['span_id']
            new_value = action['new_value']
            key = (st.session_state.current_page, span_id)
            st.session_state.labels[key] = new_value
    
    @staticmethod
    def mark_dirty():
        """Mark state as dirty (unsaved changes)"""
        st.session_state.dirty = True
    
    @staticmethod
    def mark_clean():
        """Mark state as clean (saved)"""
        st.session_state.dirty = False
        st.session_state.last_save = datetime.now()