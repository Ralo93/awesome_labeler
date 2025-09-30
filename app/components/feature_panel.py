import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
from core.schematas import Span

def show_feature_panel(
    selected_spans: List[str],
    spans: List[Span],
    show_normalized: bool = True,
    show_rules: bool = False,
    rules: Optional[Dict] = None
):
    """Display feature panel for selected spans"""
    
    if not selected_spans:
        st.info("Select spans to view features")
        return
    
    # Get selected span objects
    selected_objs = [s for s in spans if s.span_id in selected_spans]
    
    if not selected_objs:
        return
    
    # Create feature dataframe
    data = []
    for span in selected_objs:
        row = {
            'Span ID': span.span_id,
            'Text': span.text[:30] + "..." if len(span.text) > 30 else span.text,
            'Font Size': span.font_size,
            'Bold': '✓' if span.bold else '✗',
            'Italic': '✓' if span.italic else '✗',
            'Line Height': span.line_height,
            'Column': span.column,
            'Reading Order': span.reading_order
        }
        
        if show_normalized:
            # Add normalized features
            row.update({
                'X Norm': f"{span.x_center / 595.0:.3f}",  # Normalized to page width
                'Y Norm': f"{span.y_bottom / 842.0:.3f}",  # Normalized to page height
            })
        
        if show_rules and rules and span.span_id in rules:
            rule = rules[span.span_id]
            row.update({
                'Header': '✓' if rule.is_header else '✗',
                'Caption': '✓' if rule.is_caption else '✗',
                'Page Num': '✓' if rule.is_page_num else '✗'
            })
        
        data.append(row)
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)