import streamlit as st
import numpy as np
from typing import List, Dict, Tuple
import plotly.graph_objects as go

def create_heatmap_bars(
    predictions: Dict[Tuple[str, str], float],
    spans: List,
    page_height: float = 842.0
) -> go.Figure:
    """Create heatmap visualization for boundary probabilities"""
    
    if not predictions:
        return None
    
    # Prepare data for visualization
    y_positions = []
    probabilities = []
    hover_texts = []
    
    for (span_id_prev, span_id_curr), prob in predictions.items():
        # Find spans
        prev_span = next((s for s in spans if s.span_id == span_id_prev), None)
        curr_span = next((s for s in spans if s.span_id == span_id_curr), None)
        
        if prev_span and curr_span:
            # Position between spans
            y_pos = (prev_span.y_bottom + curr_span.y_bottom) / 2
            y_positions.append(y_pos)
            probabilities.append(prob)
            
            # Create hover text
            hover_text = f"P(boundary): {prob:.3f}<br>"
            hover_text += f"Between: {span_id_prev} → {span_id_curr}"
            hover_texts.append(hover_text)
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=probabilities,
        y=y_positions,
        orientation='h',
        marker=dict(
            color=probabilities,
            colorscale='RdYlGn_r',
            cmin=0,
            cmax=1,
            colorbar=dict(title="P(new)")
        ),
        hovertext=hover_texts,
        hoverinfo='text',
        width=5
    ))
    
    # Update layout
    fig.update_layout(
        title="Boundary Probabilities",
        xaxis_title="Probability",
        yaxis_title="Position",
        height=600,
        showlegend=False,
        yaxis=dict(range=[page_height, 0])  # Invert y-axis
    )
    
    return fig

def get_top_features(
    span_pair: Tuple[str, str],
    feature_data: dict,
    top_n: int = 3
) -> List[Tuple[str, float]]:
    """Get top contributing features for a prediction"""
    
    # This would use feature importance from the model
    # Simplified version for now
    features = []
    
    if span_pair in feature_data:
        feat_dict = feature_data[span_pair]
        # Sort by absolute value
        sorted_feats = sorted(
            feat_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        features = sorted_feats
    
    return features