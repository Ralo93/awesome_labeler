import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Dict
from core.schematas import Span, Label

def get_unit_color(unit_index: int, alpha: int = 80) -> Tuple[int, int, int, int]:
    """Get color for a unit based on index"""
    colors = [
        (255, 107, 107),  # Red
        (107, 255, 161),  # Green  
        (107, 185, 255),  # Blue
        (255, 193, 107),  # Orange
        (196, 107, 255),  # Purple
        (255, 107, 196),  # Pink
        (107, 255, 239),  # Cyan
        (239, 255, 107),  # Yellow
    ]
    base_color = colors[unit_index % len(colors)]
    return (*base_color, alpha)

def draw_overlay(
    image: Image.Image,
    spans: List[Span],
    labels: Dict[Tuple[int, str], Label],
    selected_spans: List[str],
    predictions: Optional[Dict[Tuple[str, str], float]] = None,
    show_heatmap: bool = True,
    zoom: float = 1.0
) -> Image.Image:
    """Draw overlay on document image with units and boundaries"""
    
    # Create overlay with transparency
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # First pass: Group spans by unit
    units = {}
    unit_order = {}
    
    for span in spans:
        key = (span.page_number, span.span_id)
        if key in labels:
            label = labels[key]
            unit_id = label.unit_id
            
            if unit_id not in units:
                units[unit_id] = []
                unit_order[unit_id] = len(unit_order)
            units[unit_id].append(span)
    
    # Draw units (background colors)
    for unit_id, unit_spans in units.items():
        color = get_unit_color(unit_order[unit_id], alpha=40)
        
        for span in unit_spans:
            # Scale bbox if needed
            bbox = [coord * zoom for coord in span.bbox]
            
            # Draw filled rectangle for unit
            draw.rectangle(bbox, fill=color, outline=None)
    
    # # Draw boundaries between units
    # for i, span in enumerate(spans):
    #     key = (span.page, span.span_id)
    #     if key in labels:
    #         label = labels[key]
            
    #         if label.boundary == "new" and i > 0:
    #             # Draw boundary line above this span
    #             bbox = [coord * zoom for coord in span.bbox]
    #             prev_span = spans[i-1] if i > 0 else None
                
    #             if prev_span:
    #                 prev_bbox = [coord * zoom for coord in prev_span.bbox]
    #                 # Draw line between spans
    #                 y_pos = (prev_bbox[3] + bbox[1]) / 2
    #                 draw.line(
    #                     [(0, y_pos), (image.width, y_pos)],
    #                     fill=(255, 0, 0, 100),
    #                     width=2
    #                 )
    
    # Draw predictions/confidence if available
    if show_heatmap and predictions:
        margin = 20 * zoom
        bar_width = 15 * zoom
        
        for (span_id_prev, span_id_curr), prob in predictions.items():
            # Find the spans
            prev_span = next((s for s in spans if s.span_id == span_id_prev), None)
            curr_span = next((s for s in spans if s.span_id == span_id_curr), None)
            
            if prev_span and curr_span:
                prev_bbox = [coord * zoom for coord in prev_span.bbox]
                curr_bbox = [coord * zoom for coord in curr_span.bbox]
                
                # Position bar between spans
                y_center = (prev_bbox[3] + curr_bbox[1]) / 2
                
                # Color based on probability
                if prob > 0.7:
                    color = (255, 0, 0, 200)  # Red for high boundary prob
                elif prob > 0.3:
                    color = (255, 165, 0, 200)  # Orange for medium
                else:
                    color = (0, 255, 0, 200)  # Green for low
                
                # Draw probability bar on the left margin
                bar_height = max(2, int(prob * 20 * zoom))
                draw.rectangle(
                    [margin, y_center - bar_height/2, margin + bar_width, y_center + bar_height/2],
                    fill=color,
                    outline=None
                )
                
                # Add probability text
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", int(10 * zoom))
                except:
                    font = None
                
                draw.text(
                    (margin + bar_width + 5, y_center - 5),
                    f"{prob:.2f}",
                    fill=(100, 100, 100, 255),
                    font=font
                )
    
    # Draw selected spans (on top)
    for span in spans:
        if span.span_id in selected_spans:
            bbox = [coord * zoom for coord in span.bbox]
            # Draw selection border
            draw.rectangle(
                bbox,
                fill=None,
                outline=(0, 0, 255, 255),
                width=max(2, int(2 * zoom))
            )
    
    # Composite overlay onto original image
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    return result

def snap_to_span(
    click_pos: Tuple[float, float],
    spans: List[Span],
    threshold: float = 10.0
) -> Optional[str]:
    """Find span closest to click position"""
    
    x, y = click_pos
    min_dist = float('inf')
    closest_span = None
    
    for span in spans:
        bbox = span.bbox
        # Check if click is inside bbox
        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
            return span.span_id
        
        # Calculate distance to bbox center
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        if dist < min_dist and dist < threshold:
            min_dist = dist
            closest_span = span.span_id
    
    return closest_span