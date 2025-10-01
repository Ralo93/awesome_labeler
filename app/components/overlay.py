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
    predictions: Optional[Dict[Tuple[int, str], float]] = None,
    show_heatmap: bool = True,
    show_unit_numbers: bool = True,
    zoom: float = 1.0
) -> Image.Image:
    """Draw overlay on document image with units and boundaries"""
    
    # Create overlay with transparency
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Load font for unit numbers
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", size=int(14 * zoom))
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", int(14 * zoom))
        except:
            font = ImageFont.load_default()
    
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
    
    # Sort units by reading order of first span in each unit
    sorted_units = []
    for unit_id, unit_spans in units.items():
        # Find first span by reading order
        first_span = min(unit_spans, key=lambda s: s.reading_order or 999999)
        sorted_units.append((first_span.reading_order or 999999, unit_id, unit_spans))
    sorted_units.sort()
    
    # Draw units (background colors)
    for unit_id, unit_spans in units.items():
        color = get_unit_color(unit_order[unit_id], alpha=40)
        
        for span in unit_spans:
            # Scale bbox if needed
            bbox = [coord * zoom for coord in span.bbox]
            
            # Draw filled rectangle for unit
            draw.rectangle(bbox, fill=color, outline=None)
    
    # Draw semantic unit numbers
    if show_unit_numbers and sorted_units:
        for unit_idx, (_, unit_id, unit_spans) in enumerate(sorted_units):
            # Find the topmost, leftmost span in the unit for positioning
            first_span = min(unit_spans, key=lambda s: (s.bbox[1], s.bbox[0]))
            
            bbox = [coord * zoom for coord in first_span.bbox]
            x0, y0 = bbox[0], bbox[1]
            
            # Unit number text
            unit_num = unit_idx + 1
            unit_text = f"#{unit_num}"
            
            # Get text dimensions for box sizing
            text_bbox = draw.textbbox((0, 0), unit_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Box padding
            padding = int(3 * zoom)
            box_width = text_width + 2 * padding
            box_height = text_height + 2 * padding
            
            # Position box at top-left of semantic unit, slightly offset
            box_x = max(0, x0 - int(5 * zoom))
            box_y = max(0, y0 - box_height - int(5 * zoom))
            
            # Draw unit number box background (dark blue)
            draw.rectangle(
                [box_x, box_y, box_x + box_width, box_y + box_height],
                fill=(25, 25, 112, 230),  # Dark blue background
                outline=(255, 255, 255, 255),  # White border
                width=max(1, int(zoom))
            )
            
            # Draw unit number text (white)
            text_x = box_x + padding
            text_y = box_y + padding
            draw.text((text_x, text_y), unit_text, fill=(255, 255, 255, 255), font=font)
    
    # Draw predictions/confidence if available
    # FIXED: Handle predictions with (page_number, span_id) keys
    if show_heatmap and predictions:
        margin = 10 * zoom
        bar_width = 8 * zoom
        
        # Sort spans by reading order for proper visualization
        sorted_spans = sorted(spans, key=lambda s: s.reading_order)
        
        for span in sorted_spans:
            key = (span.page_number, span.span_id)
            
            if key in predictions:
                prob = predictions[key]
                
                # Scale bbox
                bbox = [coord * zoom for coord in span.bbox]
                
                # Position bar at the top of the span (indicating boundary probability)
                y_position = bbox[1]  # Top of span
                
                # Color based on probability
                if prob > 0.7:
                    color = (255, 0, 0, 200)  # Red for high boundary prob
                elif prob > 0.3:
                    color = (255, 165, 0, 200)  # Orange for medium
                else:
                    color = (0, 255, 0, 200)  # Green for low
                
                # Draw probability bar on the left margin
                bar_height = max(3, int(prob * 15 * zoom))
                draw.rectangle(
                    [margin, y_position - bar_height/2, margin + bar_width, y_position + bar_height/2],
                    fill=color,
                    outline=None
                )
                
                # Add probability text (only for high probability boundaries)
                if prob > 0.5:
                    try:
                        prob_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", int(9 * zoom))
                    except:
                        prob_font = font
                    
                    draw.text(
                        (margin + bar_width + 2, y_position - 6),
                        f"{prob:.2f}",
                        fill=(100, 100, 100, 255),
                        font=prob_font
                    )
                
                # Draw a thin line at high probability boundaries
                if prob > st.session_state.get('boundary_threshold', 0.5):
                    # Draw horizontal line above span to indicate predicted boundary
                    line_y = bbox[1] - 2
                    draw.line(
                        [(bbox[0], line_y), (bbox[2], line_y)],
                        fill=(255, 0, 0, 150),
                        width=max(1, int(zoom))
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