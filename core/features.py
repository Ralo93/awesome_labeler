from typing import List, Dict, Tuple
import numpy as np
from .schematas import Span

def compute_sliding_window_features(spans: List[Span], target_index: int, window_size: int = 5) -> Dict[str, float]:
    """
    Compute features for boundary detection using sliding window around target span.
    
    This replaces pairwise features with a richer context window approach.
    """
    if target_index >= len(spans):
        return {}
    
    target_span = spans[target_index]
    features = {}
    half_window = window_size // 2
    
    # Core span features (most predictive for boundaries)
    features.update({
        'font_size': target_span.font_size,
        'is_bold': float(target_span.bold),
        'is_italic': float(target_span.italic),
        'char_count': target_span.char_count,
        'word_count': target_span.word_count,
        'line_height': target_span.line_height,
        'width': target_span.width,
        'height': target_span.height,
        'starts_with_number': float(target_span.starts_with_number),
        'starts_with_bullet': float(target_span.starts_with_bullet),
        'is_all_caps': float(target_span.is_all_caps),
        'ends_with_colon': float(target_span.ends_with_colon),
        'ends_with_period': float(target_span.ends_with_period),
        'is_first_on_page': float(target_span.is_first_on_page),
        'is_last_on_page': float(target_span.is_last_on_page),
        'relative_position': target_span.relative_position,
        'column': float(target_span.column),
    })
    
    # Sequential context features (key for boundary detection)
    features.update({
        'font_size_changed': float(target_span.font_size_changed),
        'style_changed': float(target_span.style_changed),
        'vertical_gap_norm': target_span.vertical_gap / target_span.line_height if target_span.line_height > 0 else 0,
        'reading_order_gap': float(target_span.reading_order_gap),
        'same_page_as_prev': float(target_span.same_page_as_prev),
        'prev_font_size': target_span.prev_font_size,
        'prev_is_bold': float(target_span.prev_is_bold),
        'prev_ends_period': float(target_span.prev_ends_period),
        'next_font_size': target_span.next_font_size,
        'next_starts_bullet': float(target_span.next_starts_bullet),
    })
    
    # Window context features
    for offset in range(-half_window, half_window + 1):
        if offset == 0:
            continue  # Skip current span
            
        neighbor_idx = target_index + offset
        prefix = f"offset_{offset:+d}_"
        
        if 0 <= neighbor_idx < len(spans):
            neighbor = spans[neighbor_idx]
            features.update({
                f"{prefix}font_size": neighbor.font_size,
                f"{prefix}is_bold": float(neighbor.bold),
                f"{prefix}ends_period": float(neighbor.ends_with_period),
                f"{prefix}starts_bullet": float(neighbor.starts_with_bullet),
                f"{prefix}is_all_caps": float(neighbor.is_all_caps),
                f"{prefix}char_count": float(neighbor.char_count),
                f"{prefix}word_count": float(neighbor.word_count),
            })
        else:
            # Padding for out-of-bounds
            features.update({
                f"{prefix}exists": 0.0,
                f"{prefix}font_size": 0.0,
                f"{prefix}is_bold": 0.0,
                f"{prefix}ends_period": 0.0,
                f"{prefix}starts_bullet": 0.0,
                f"{prefix}is_all_caps": 0.0,
                f"{prefix}char_count": 0.0,
                f"{prefix}word_count": 0.0,
            })
    
    return features

def add_reading_order(spans: List[Span]) -> List[Span]:
    """Add reading order to spans if missing (backward compatibility)"""
    # Group by page
    pages = {}
    for span in spans:
        page_num = span.page_number
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(span)
    
    # Sort each page and assign reading order
    updated_spans = []
    for page_num in sorted(pages.keys()):
        page_spans = pages[page_num]
        
        # Sort by y_bottom (top to bottom), then x_center (left to right)
        page_spans.sort(key=lambda s: (s.y_bottom, s.x_center))
        
        # Update reading order
        for idx, span in enumerate(page_spans):
            object.__setattr__(span, 'reading_order', idx + 1)
        
        updated_spans.extend(page_spans)
    
    return updated_spans

def extract_boundary_training_data(spans: List[Span], labels: Dict[Tuple[int, str], 'Label']) -> Tuple[List[Dict], List[int]]:
    """Extract training data for boundary classification model using sliding window features"""
    features = []
    targets = []
    
    for i in range(len(spans)):
        # Use sliding window features for every span
        feature_dict = compute_sliding_window_features(spans, i)
        features.append(feature_dict)
        
        # Determine target (is this span the START of a new semantic unit?)
        key = (spans[i].page_number, spans[i].span_id)
        is_boundary = 1 if key in labels and labels[key].boundary == "new" else 0
        targets.append(is_boundary)
    
    return features, targets