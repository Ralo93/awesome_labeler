import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Tuple
from core.schematas import Span, Label

def show_feature_panel(
    spans: List[Span],
    labels: Dict[Tuple[int, str], Label], 
    selected_spans: List[str]
):
    """Display feature panel for selected spans"""
    
    if not selected_spans:
        st.info("Select spans to view features")
        return
    
    # Get selected span objects
    selected_objs = [s for s in spans if s.span_id in selected_spans]
    
    if not selected_objs:
        return
    
    st.subheader("📊 Span Features")
    
    # Create feature dataframe following MVP focus
    data = []
    for span in selected_objs:
        span_key = (span.page_number, span.span_id)
        label = labels.get(span_key)
        
        row = {
            'Span ID': span.span_id,
            'Text': span.text[:30] + "..." if len(span.text) > 30 else span.text,
            'Font Size': span.font_size,
            'Bold': '✓' if span.bold else '✗',
            'Italic': '✓' if span.italic else '✗',
            'Line Height': f"{span.line_height:.1f}",
            'Column': span.column,
            'Reading Order': span.reading_order,
            'X Center': f"{span.x_center:.1f}",
            'Y Bottom': f"{span.y_bottom:.1f}",
        }
        
        # Add label info if available (following state management pattern)
        if label:
            row['Boundary'] = label.boundary
            row['Unit ID'] = label.unit_id
            row['Confidence'] = f"{label.confidence:.3f}"
        else:
            row['Boundary'] = 'unlabeled'
            row['Unit ID'] = '-'
            row['Confidence'] = '-'
        
        data.append(row)
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
