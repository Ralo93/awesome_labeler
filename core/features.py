from typing import List, Dict, Tuple
import numpy as np
from .schematas import Span

def compute_sliding_window_features(spans: List[Span], target_index: int, window_size: int = 5) -> Dict[str, float]:
    """
    Compute features for boundary detection using sliding window around target span.
    Now includes column-aware features for better multi-column support.
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
    
    # NEW: Column-specific features
    features.update({
        'is_first_in_column': float(getattr(target_span, 'is_first_in_column', False)),
        'is_last_in_column': float(getattr(target_span, 'is_last_in_column', False)),
        'num_columns_on_page': float(getattr(target_span, 'num_columns_on_page', 1)),
        'is_multi_column_page': float(getattr(target_span, 'num_columns_on_page', 1) > 1),
        'column_changed': float(getattr(target_span, 'column_changed', False)),
        'column_break': float(getattr(target_span, 'column_break', False)),
        'horizontal_gap': float(getattr(target_span, 'horizontal_gap', 0)),
        'next_column_different': float(getattr(target_span, 'next_column_different', False)),
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
    
    # Column transition indicators
    if target_index > 0:
        prev_span = spans[target_index - 1]
        # Strong boundary signal: column transition AND style change
        column_style_change = (
            float(target_span.column != prev_span.column) * 
            float(abs(target_span.font_size - prev_span.font_size) > 2)
        )
        features['column_style_change'] = column_style_change
        
        # Check if returning to first column (often indicates new section)
        features['returns_to_first_column'] = float(
            prev_span.column > 0 and target_span.column == 0
        )
    else:
        features['column_style_change'] = 0.0
        features['returns_to_first_column'] = 0.0
    
    # Window context features with column awareness
    for offset in range(-half_window, half_window + 1):
        if offset == 0:
            continue  # Skip current span
            
        neighbor_idx = target_index + offset
        prefix = f"offset_{offset:+d}_"
        
        if 0 <= neighbor_idx < len(spans):
            neighbor = spans[neighbor_idx]
            
            # Standard features
            features.update({
                f"{prefix}font_size": neighbor.font_size,
                f"{prefix}is_bold": float(neighbor.bold),
                f"{prefix}ends_period": float(neighbor.ends_with_period),
                f"{prefix}starts_bullet": float(neighbor.starts_with_bullet),
                f"{prefix}is_all_caps": float(neighbor.is_all_caps),
                f"{prefix}char_count": float(neighbor.char_count),
                f"{prefix}word_count": float(neighbor.word_count),
            })
            
            # NEW: Column relationship features
            features[f"{prefix}same_column"] = float(neighbor.column == target_span.column)
            features[f"{prefix}column_diff"] = float(abs(neighbor.column - target_span.column))
            
            # Check if neighbor is in adjacent column (for detecting parallel content)
            features[f"{prefix}is_adjacent_column"] = float(
                abs(neighbor.column - target_span.column) == 1 and
                neighbor.page_number == target_span.page_number
            )
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
                f"{prefix}same_column": 0.0,
                f"{prefix}column_diff": 0.0,
                f"{prefix}is_adjacent_column": 0.0,
            })
    
    # Column-aware reading flow features
    # Look for patterns in same column (more relevant for boundaries)
    same_column_neighbors = []
    for i in range(max(0, target_index - 10), min(len(spans), target_index + 10)):
        if i != target_index and spans[i].page_number == target_span.page_number:
            if spans[i].column == target_span.column:
                same_column_neighbors.append(spans[i])
    
    if same_column_neighbors:
        # Average font size in same column
        avg_font_same_col = np.mean([s.font_size for s in same_column_neighbors])
        features['font_size_vs_column_avg'] = target_span.font_size / avg_font_same_col if avg_font_same_col > 0 else 1.0
        
        # Is this span significantly larger than column average? (heading indicator)
        features['is_larger_than_column'] = float(target_span.font_size > avg_font_same_col * 1.2)
    else:
        features['font_size_vs_column_avg'] = 1.0
        features['is_larger_than_column'] = 0.0
    
    return features

def add_reading_order(spans: List[Span]) -> List[Span]:
    """Add reading order to spans if missing (backward compatibility)"""
    # This is now handled in the pdf_processor with column awareness
    # But keep this for backward compatibility with older data
    
    # Group by page
    pages = {}
    for span in spans:
        page_num = span.page_number
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(span)
    
    # Sort each page respecting columns if present
    updated_spans = []
    for page_num in sorted(pages.keys()):
        page_spans = pages[page_num]
        
        # Check if spans have column information
        has_columns = any(hasattr(s, 'column') and s.column > 0 for s in page_spans)
        
        if has_columns:
            # Group by column first
            columns = {}
            for span in page_spans:
                col = getattr(span, 'column', 0)
                if col not in columns:
                    columns[col] = []
                columns[col].append(span)
            
            # Sort within each column, then concatenate
            reading_order = 1
            for col in sorted(columns.keys()):
                col_spans = sorted(columns[col], key=lambda s: s.y_bottom)
                for span in col_spans:
                    object.__setattr__(span, 'reading_order', reading_order)
                    reading_order += 1
                    updated_spans.append(span)
        else:
            # Simple top-to-bottom, left-to-right
            page_spans.sort(key=lambda s: (s.y_bottom, s.x_center))
            
            for idx, span in enumerate(page_spans):
                object.__setattr__(span, 'reading_order', idx + 1)
            
            updated_spans.extend(page_spans)
    
    return updated_spans

def extract_boundary_training_data(spans: List[Span], labels: Dict[Tuple[int, str], 'Label']) -> Tuple[List[Dict], List[int]]:
    """
    Extract training data for boundary classification model using sliding window features.
    Now includes column-aware features for better multi-column document support.
    """
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

def analyze_column_layout(spans: List[Span]) -> Dict[str, any]:
    """
    Analyze the column layout of a document for debugging/visualization.
    Returns statistics about column usage.
    """
    stats = {
        'total_pages': len(set(s.page_number for s in spans)),
        'pages_with_columns': {},
        'column_transitions': 0,
        'average_spans_per_column': {},
    }
    
    # Analyze each page
    page_groups = {}
    for span in spans:
        if span.page_number not in page_groups:
            page_groups[span.page_number] = []
        page_groups[span.page_number].append(span)
    
    for page_num, page_spans in page_groups.items():
        columns = set(s.column for s in page_spans)
        stats['pages_with_columns'][page_num] = {
            'num_columns': len(columns),
            'column_ids': list(columns),
            'spans_per_column': {}
        }
        
        for col in columns:
            col_spans = [s for s in page_spans if s.column == col]
            stats['pages_with_columns'][page_num]['spans_per_column'][col] = len(col_spans)
    
    # Count column transitions
    for i in range(1, len(spans)):
        if spans[i].page_number == spans[i-1].page_number:
            if spans[i].column != spans[i-1].column:
                stats['column_transitions'] += 1
    
    # Calculate averages
    all_column_counts = []
    for page_data in stats['pages_with_columns'].values():
        all_column_counts.extend(page_data['spans_per_column'].values())
    
    if all_column_counts:
        stats['average_spans_per_column'] = np.mean(all_column_counts)
    
    return stats