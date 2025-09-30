import streamlit as st
from typing import Callable

def render_toolbar(
    on_boundary_toggle: Callable,
    on_merge: Callable,
    on_split: Callable,
    on_undo: Callable,
    on_redo: Callable,
    on_save: Callable,
    on_next_uncertain: Callable,
    confidence_mode: bool = False
):
    """Render labeling toolbar"""
    
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        if st.button("🔀 Boundary (B)", help="Toggle boundary", use_container_width=True):
            on_boundary_toggle()
    
    with col2:
        if st.button("🔗 Merge (M)", help="Merge units", use_container_width=True):
            on_merge()
    
    with col3:
        if st.button("✂️ Split (S)", help="Split unit", use_container_width=True):
            on_split()
    
    with col4:
        if st.button("↩️ Undo", help="Ctrl+Z", use_container_width=True):
            on_undo()
    
    with col5:
        if st.button("↪️ Redo", help="Ctrl+Y", use_container_width=True):
            on_redo()
    
    with col6:
        if st.button("💾 Save", help="Save labels", use_container_width=True):
            on_save()
    
    with col7:
        if st.button("❓ Next Uncertain", help="Jump to next uncertain boundary", use_container_width=True):
            on_next_uncertain()
    
    if confidence_mode:
        confidence = st.slider(
            "Confidence",
            min_value=1,
            max_value=5,
            value=3,
            help="Confidence level for current labeling action"
        )
        return confidence
    
    return None