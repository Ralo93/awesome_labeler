import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import asdict, is_dataclass
from .schematas import Span, Label, SemanticUnit

def save_jsonl(file_path: Path, data: List[Any]) -> None:
    """Save list of objects to JSONL format"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            if is_dataclass(item):
                json_data = asdict(item)
            elif isinstance(item, dict):
                json_data = item
            else:
                json_data = item.__dict__ if hasattr(item, '__dict__') else str(item)
            
            f.write(json.dumps(json_data, ensure_ascii=False) + '\n')

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file as list of dictionaries"""
    if not file_path.exists():
        return []
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    return data

def load_spans(doc_id: str, data_dir: Path = Path("data")) -> List[Span]:
    """Load spans for a document"""
    spans_path = data_dir / "docs" / doc_id / "spans.jsonl"
    
    if not spans_path.exists():
        return []
    
    spans_data = load_jsonl(spans_path)
    spans = []
    
    for span_dict in spans_data:
        # Handle backward compatibility - convert old format to new
        span = Span(
            doc_id=span_dict.get('doc_id', doc_id),
            page_number=span_dict['page_number'],
            span_id=span_dict['span_id'],
            bbox=tuple(span_dict['bbox']),
            text=span_dict['text'],
            font_size=span_dict.get('font_size', 12.0),
            bold=span_dict.get('bold', False),
            italic=span_dict.get('italic', False),
            line_height=span_dict.get('line_height', 14.0),
            x_center=span_dict.get('x_center', 0.0),
            y_bottom=span_dict.get('y_bottom', 0.0),
            column=span_dict.get('column', 0),
            reading_order=span_dict.get('reading_order', 1),
            rotation=span_dict.get('rotation', 0.0),
            language=span_dict.get('language'),
            is_whitespace=span_dict.get('is_whitespace', False),
            char_count=span_dict.get('char_count', len(span_dict.get('text', ''))),
            word_count=span_dict.get('word_count', len(span_dict.get('text', '').split())),
            
            # Enhanced features with defaults
            font_name=span_dict.get('font_name', ''),
            width=span_dict.get('width', 0.0),
            height=span_dict.get('height', 0.0),
            starts_with_number=span_dict.get('starts_with_number', False),
            starts_with_bullet=span_dict.get('starts_with_bullet', False),
            is_all_caps=span_dict.get('is_all_caps', False),
            ends_with_colon=span_dict.get('ends_with_colon', False),
            ends_with_period=span_dict.get('ends_with_period', False),
            is_first_on_page=span_dict.get('is_first_on_page', False),
            is_last_on_page=span_dict.get('is_last_on_page', False),
            relative_position=span_dict.get('relative_position', 0.0),
            
            # Sequential context features
            prev_font_size=span_dict.get('prev_font_size', 12.0),
            prev_is_bold=span_dict.get('prev_is_bold', False),
            prev_ends_period=span_dict.get('prev_ends_period', False),
            font_size_changed=span_dict.get('font_size_changed', False),
            style_changed=span_dict.get('style_changed', False),
            vertical_gap=span_dict.get('vertical_gap', 0.0),
            next_font_size=span_dict.get('next_font_size', 12.0),
            next_starts_bullet=span_dict.get('next_starts_bullet', False),
            reading_order_gap=span_dict.get('reading_order_gap', 1),
            same_page_as_prev=span_dict.get('same_page_as_prev', True),
        )
        spans.append(span)
    
    return spans

def load_labels(doc_id: str, data_dir: Path = Path("data")) -> Dict[Tuple[int, str], Label]:
    """Load labels for a document"""
    labels_path = data_dir / "docs" / doc_id / "labels.jsonl"
    
    if not labels_path.exists():
        return {}
    
    labels_data = load_jsonl(labels_path)
    labels = {}
    
    for label_dict in labels_data:
        key = (label_dict['page_number'], label_dict['span_id'])
        
        label = Label(
            span_id=label_dict['span_id'],
            page_number=label_dict['page_number'],
            boundary=label_dict.get('boundary', label_dict.get('boundary_type', 'continue')),  # Handle old format
            unit_id=label_dict['unit_id'],
            confidence=label_dict.get('confidence', 1.0),
            timestamp=label_dict.get('timestamp')
        )
        labels[key] = label
    
    return labels

def save_labels(doc_id: str, labels: Dict[Tuple[int, str], Label], data_dir: Path = Path("data")) -> None:
    """Save labels for a document"""
    labels_path = data_dir / "docs" / doc_id / "labels.jsonl"
    
    labels_list = []
    for (page_num, span_id), label in labels.items():
        labels_list.append(label)
    
    save_jsonl(labels_path, labels_list)

def create_semantic_units(spans: List[Span], labels: Dict[Tuple[int, str], Label]) -> List[SemanticUnit]:
    """Create semantic units from spans and labels"""
    units_dict = {}
    
    # Group spans by unit_id
    for span in spans:
        key = (span.page_number, span.span_id)
        
        if key in labels:
            unit_id = labels[key].unit_id
            
            if unit_id not in units_dict:
                units_dict[unit_id] = []
            units_dict[unit_id].append(span)
    
    # Create SemanticUnit objects
    semantic_units = []
    for unit_id, unit_spans in units_dict.items():
        # Sort spans by reading order
        unit_spans.sort(key=lambda s: (s.page_number, s.reading_order))
        
        unit = SemanticUnit(
            unit_id=unit_id,
            spans=unit_spans,
            unit_type=None,  # Could be inferred later
            confidence=1.0
        )
        semantic_units.append(unit)
    
    # Sort units by first span's reading order
    semantic_units.sort(key=lambda u: (u.spans[0].page_number, u.spans[0].reading_order) if u.spans else (0, 0))
    
    return semantic_units

def export_training_data(doc_ids: List[str], output_path: Path, data_dir: Path = Path("data")) -> None:
    """Export training data for multiple documents"""
    from .features import extract_boundary_training_data
    import pandas as pd
    
    all_features = []
    all_targets = []
    all_doc_ids = []
    
    for doc_id in doc_ids:
        spans = load_spans(doc_id, data_dir)
        labels = load_labels(doc_id, data_dir)
        
        if not spans or not labels:
            continue
        
        features, targets = extract_boundary_training_data(spans, labels)
        
        all_features.extend(features)
        all_targets.extend(targets)
        all_doc_ids.extend([doc_id] * len(features))
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    df['y_boundary'] = all_targets
    df['doc_id'] = all_doc_ids
    
    # Save as parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"Exported {len(df)} training examples to {output_path}")
    return df