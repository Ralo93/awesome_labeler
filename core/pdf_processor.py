import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re
import shutil
import numpy as np
from sklearn.cluster import DBSCAN
from .schematas import Span

def detect_columns(spans: List[dict], page_width: float) -> List[List[dict]]:
    """
    Detect column layout using clustering on x-coordinates.
    Returns list of column groups (each group is a list of spans).
    """
    if not spans:
        return []
    
    # Extract x-coordinates (use left edge of bbox)
    x_coords = np.array([[s['bbox'][0]] for s in spans])
    
    # Use DBSCAN clustering to find column groups
    # eps is the maximum distance between samples in the same cluster
    # We use page_width/10 as a reasonable threshold
    clustering = DBSCAN(eps=page_width/10, min_samples=3)
    labels = clustering.fit_predict(x_coords)
    
    # Group spans by cluster
    columns = {}
    for i, label in enumerate(labels):
        if label == -1:  # Noise/outlier - assign to nearest column
            # Find nearest cluster center
            min_dist = float('inf')
            best_label = 0
            for other_label in set(labels):
                if other_label != -1:
                    cluster_spans = [spans[j] for j, l in enumerate(labels) if l == other_label]
                    cluster_x = np.mean([s['bbox'][0] for s in cluster_spans])
                    dist = abs(spans[i]['bbox'][0] - cluster_x)
                    if dist < min_dist:
                        min_dist = dist
                        best_label = other_label
            label = best_label
        
        if label not in columns:
            columns[label] = []
        columns[label].append(spans[i])
    
    # Sort columns by average x position (left to right)
    column_list = []
    for label, col_spans in columns.items():
        avg_x = np.mean([s['bbox'][0] for s in col_spans])
        column_list.append((avg_x, col_spans))
    column_list.sort(key=lambda x: x[0])
    
    # Return just the span lists
    return [col[1] for col in column_list]

def assign_reading_order_with_columns(spans_on_page: List[dict], page_width: float) -> List[dict]:
    """
    Assign reading order respecting column layout.
    Reads each column completely before moving to the next.
    """
    if not spans_on_page:
        return spans_on_page
    
    # Detect columns
    columns = detect_columns(spans_on_page, page_width)
    
    # If only one column detected, use simple top-to-bottom ordering
    if len(columns) <= 1:
        spans_on_page.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))  # y_top, x_left
        for i, span in enumerate(spans_on_page):
            span['reading_order'] = i + 1
            span['detected_column'] = 0
        return spans_on_page
    
    # Assign reading order within each column, then across columns
    ordered_spans = []
    reading_order = 1
    
    for col_idx, col_spans in enumerate(columns):
        # Sort spans within column by y position
        col_spans.sort(key=lambda s: s['bbox'][1])  # Sort by y_top
        
        for span in col_spans:
            span['reading_order'] = reading_order
            span['detected_column'] = col_idx
            reading_order += 1
            ordered_spans.append(span)
    
    return ordered_spans

def calculate_column_features(span_dict: dict, all_spans: List[dict], span_idx: int) -> dict:
    """
    Calculate additional column-aware features for better boundary detection.
    """
    features = {}
    
    # Current span column
    current_col = span_dict.get('detected_column', 0)
    features['detected_column'] = current_col
    
    # Check if this is the first/last span in its column
    same_col_spans = [s for s in all_spans if s.get('detected_column', 0) == current_col]
    col_position = next((i for i, s in enumerate(same_col_spans) if s is span_dict), 0)
    
    features['is_first_in_column'] = col_position == 0
    features['is_last_in_column'] = col_position == len(same_col_spans) - 1
    features['column_position_ratio'] = col_position / len(same_col_spans) if same_col_spans else 0
    
    # Previous span column features
    if span_idx > 0:
        prev_span = all_spans[span_idx - 1]
        prev_col = prev_span.get('detected_column', 0)
        features['column_changed'] = current_col != prev_col
        features['prev_column'] = prev_col
        
        # Horizontal gap to previous span (important for column transitions)
        if current_col != prev_col:
            # Calculate horizontal distance between columns
            features['horizontal_gap'] = abs(span_dict['bbox'][0] - prev_span['bbox'][2])
        else:
            features['horizontal_gap'] = 0
    else:
        features['column_changed'] = False
        features['prev_column'] = current_col
        features['horizontal_gap'] = 0
    
    # Next span column features
    if span_idx < len(all_spans) - 1:
        next_span = all_spans[span_idx + 1]
        next_col = next_span.get('detected_column', 0)
        features['next_column_changes'] = current_col != next_col
        features['next_column'] = next_col
    else:
        features['next_column_changes'] = False
        features['next_column'] = current_col
    
    # Multi-column indicator
    total_columns = len(set(s.get('detected_column', 0) for s in all_spans))
    features['num_columns_on_page'] = total_columns
    features['is_multi_column'] = total_columns > 1
    
    return features

def extract_spans_from_pdf(pdf_path: Path, doc_id: str) -> List[Span]:
    """Extract spans from PDF with enhanced column-aware features"""
    doc = fitz.open(pdf_path)
    all_spans = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_dict = page.get_text("dict")
        
        # Page dimensions for normalization
        page_width = page.rect.width
        page_height = page.rect.height
        
        spans_on_page = []
        seen_spans = {}  # For deduplication
        
        # First pass: extract all spans as dictionaries
        for block_idx, block in enumerate(page_dict["blocks"]):
            if block["type"] == 0:  # Text block
                for line_idx, line in enumerate(block["lines"]):
                    for span_idx, span in enumerate(line["spans"]):
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        bbox = span["bbox"]
                        bbox_rounded = tuple(round(coord, 2) for coord in bbox)
                        dedup_key = (bbox_rounded, text)
                        
                        if dedup_key in seen_spans:
                            continue
                        
                        font_size = span["size"]
                        font_flags = span["flags"]
                        font_name = span.get("font", "")
                        
                        is_bold = bool(font_flags & 2**4)
                        is_italic = bool(font_flags & 2**1)
                        
                        # Store as dictionary temporarily for column detection
                        span_dict = {
                            'bbox': bbox,
                            'text': text,
                            'font_size': font_size,
                            'font_flags': font_flags,
                            'font_name': font_name,
                            'is_bold': is_bold,
                            'is_italic': is_italic,
                            'block_idx': block_idx,
                            'line_idx': line_idx,
                            'span_idx': span_idx
                        }
                        
                        spans_on_page.append(span_dict)
                        seen_spans[dedup_key] = span_dict
        
        # Detect columns and assign reading order
        spans_on_page = assign_reading_order_with_columns(spans_on_page, page_width)
        
        # Second pass: create Span objects with all features
        span_objects = []
        for idx, span_dict in enumerate(spans_on_page):
            bbox = span_dict['bbox']
            text = span_dict['text']
            
            # Calculate geometry
            x_center = (bbox[0] + bbox[2]) / 2
            y_bottom = bbox[3]
            line_height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # Text analysis
            char_count = len(text)
            word_count = len(text.split())
            is_whitespace = text.isspace()
            starts_with_number = bool(re.match(r'^\d+\.?\s*', text))
            starts_with_bullet = bool(re.match(r'^[\u2022\u2023\u25E6\u2043\u2219•·▪▫▣◦-]', text))
            is_all_caps = text.isupper() and len(text) > 2
            ends_with_colon = text.endswith(':')
            ends_with_period = text.endswith('.')
            
            # Get column features
            col_features = calculate_column_features(span_dict, spans_on_page, idx)
            
            span_obj = Span(
                doc_id=doc_id,
                page_number=page_num + 1,
                span_id=f"p{page_num+1}_s{idx+1}",
                bbox=tuple(bbox),
                text=text,
                font_size=span_dict['font_size'],
                bold=span_dict['is_bold'],
                italic=span_dict['is_italic'],
                line_height=line_height,
                x_center=x_center,
                y_bottom=y_bottom,
                column=col_features['detected_column'],  # Use detected column instead of simple binary
                reading_order=span_dict['reading_order'],
                rotation=0.0,
                language=None,
                is_whitespace=is_whitespace,
                char_count=char_count,
                word_count=word_count,
                font_name=span_dict['font_name'],
                width=width,
                height=height,
                starts_with_number=starts_with_number,
                starts_with_bullet=starts_with_bullet,
                is_all_caps=is_all_caps,
                ends_with_colon=ends_with_colon,
                ends_with_period=ends_with_period,
                is_first_on_page=(idx == 0),
                is_last_on_page=(idx == len(spans_on_page) - 1),
                relative_position=idx / len(spans_on_page) if spans_on_page else 0,
                # Add new column features
                is_first_in_column=col_features['is_first_in_column'],
                is_last_in_column=col_features['is_last_in_column'],
                column_changed=col_features['column_changed'],
                horizontal_gap=col_features['horizontal_gap'],
                num_columns_on_page=col_features['num_columns_on_page'],
            )
            
            span_objects.append(span_obj)
        
        all_spans.extend(span_objects)
    
    # Add cross-page and sequential features
    _add_sequence_features(all_spans)
    
    doc.close()
    return all_spans

def _add_sequence_features(spans: List[Span]) -> None:
    """Add features based on sequence context with column awareness"""
    for i, span in enumerate(spans):
        # Previous span features
        if i > 0:
            prev_span = spans[i-1]
            object.__setattr__(span, 'prev_font_size', prev_span.font_size)
            object.__setattr__(span, 'prev_is_bold', prev_span.bold)
            object.__setattr__(span, 'prev_ends_period', prev_span.ends_with_period)
            object.__setattr__(span, 'font_size_changed', abs(span.font_size - prev_span.font_size) > 0.5)
            object.__setattr__(span, 'style_changed', span.bold != prev_span.bold or span.italic != prev_span.italic)
            
            # Column-aware vertical gap
            if span.page_number == prev_span.page_number:
                if span.column == prev_span.column:
                    # Same column: normal vertical gap
                    object.__setattr__(span, 'vertical_gap', span.bbox[1] - prev_span.bbox[3])
                else:
                    # Different column: this might be a column break
                    object.__setattr__(span, 'vertical_gap', 0.0)  # Reset gap for column change
                    object.__setattr__(span, 'column_break', True)
            else:
                object.__setattr__(span, 'vertical_gap', 0.0)
                object.__setattr__(span, 'column_break', False)
        else:
            object.__setattr__(span, 'prev_font_size', span.font_size)
            object.__setattr__(span, 'prev_is_bold', False)
            object.__setattr__(span, 'prev_ends_period', False)
            object.__setattr__(span, 'font_size_changed', False)
            object.__setattr__(span, 'style_changed', False)
            object.__setattr__(span, 'vertical_gap', 0.0)
            object.__setattr__(span, 'column_break', False)
        
        # Next span features
        if i < len(spans) - 1:
            next_span = spans[i+1]
            object.__setattr__(span, 'next_font_size', next_span.font_size)
            object.__setattr__(span, 'next_starts_bullet', next_span.starts_with_bullet)
            object.__setattr__(span, 'next_column_different', span.column != next_span.column)
        else:
            object.__setattr__(span, 'next_font_size', span.font_size)
            object.__setattr__(span, 'next_starts_bullet', False)
            object.__setattr__(span, 'next_column_different', False)
        
        # Reading order gap (should be 1 in proper column reading)
        if i > 0:
            object.__setattr__(span, 'reading_order_gap', span.reading_order - spans[i-1].reading_order)
            object.__setattr__(span, 'same_page_as_prev', span.page_number == spans[i-1].page_number)
        else:
            object.__setattr__(span, 'reading_order_gap', 1)
            object.__setattr__(span, 'same_page_as_prev', True)

# Keep other functions unchanged
def save_pdf_copy(pdf_path: Path, doc_id: str, data_dir: Path = Path("data")) -> Path:
    """Save a copy of the PDF in the data directory"""
    doc_dir = data_dir / "docs" / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_copy_path = doc_dir / "document.pdf"
    shutil.copy2(pdf_path, pdf_copy_path)
    
    return pdf_copy_path

def render_pdf_page(pdf_path: Path, page_num: int, zoom: float = 1.0) -> bytes:
    """Render a PDF page as PNG image bytes"""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    
    doc.close()
    return img_bytes

def get_pdf_info(pdf_path: Path) -> dict:
    """Get PDF metadata"""
    doc = fitz.open(pdf_path)
    info = {
        "pages": len(doc),
        "metadata": doc.metadata,
        "page_sizes": [(page.rect.width, page.rect.height) for page in doc]
    }
    doc.close()
    return info