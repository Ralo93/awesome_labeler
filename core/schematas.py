from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json

@dataclass
class Span:
    """Immutable text span from PDF with enhanced boundary detection features"""
    doc_id: str
    page_number: int
    span_id: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    text: str
    font_size: float
    bold: bool
    italic: bool
    line_height: float
    x_center: float
    y_bottom: float
    column: int
    reading_order: int
    rotation: float = 0.0
    language: Optional[str] = None
    is_whitespace: bool = False
    char_count: int = 0
    word_count: int = 0
    
    # Enhanced features for boundary detection
    font_name: str = ""
    width: float = 0.0
    height: float = 0.0
    
    # Semantic indicators
    starts_with_number: bool = False
    starts_with_bullet: bool = False
    is_all_caps: bool = False
    ends_with_colon: bool = False
    ends_with_period: bool = False
    
    # Positional context
    is_first_on_page: bool = False
    is_last_on_page: bool = False
    relative_position: float = 0.0
    
    # Sequential context (populated by _add_sequence_features)
    prev_font_size: float = 0.0
    prev_is_bold: bool = False
    prev_ends_period: bool = False
    font_size_changed: bool = False
    style_changed: bool = False
    vertical_gap: float = 0.0
    next_font_size: float = 0.0
    next_starts_bullet: bool = False
    reading_order_gap: int = 1
    same_page_as_prev: bool = True

    # Properties for backward compatibility
    @property
    def x_left(self) -> float:
        return self.bbox[0]
    
    @property
    def y_top(self) -> float:
        return self.bbox[1]
    
    @property
    def x_right(self) -> float:
        return self.bbox[2]
    
    @property
    def y_bottom_coord(self) -> float:
        return self.bbox[3]

@dataclass
class Label:
    """User annotation for semantic unit boundary"""
    span_id: str
    page_number: int
    boundary: str  # "new" or "continue"
    unit_id: str
    confidence: float = 1.0
    timestamp: Optional[str] = None
    
    # Backward compatibility
    @property
    def boundary_type(self) -> str:
        return "new_unit" if self.boundary == "new" else "continue_unit"


# for visualization and later integration
@dataclass
class TextSpan:
    """Original text span from document"""
    page_number: int
    span_id: str
    bbox: Tuple[float, float, float, float]
    text: str
    font_size: float
    bold: bool
    italic: bool
    line_height: float
    x_center: float
    y_bottom: float
    column: int
    reading_order: int

@dataclass
class SemanticUnit:
    """Collection of spans forming a logical document element"""
    unit_id: str
    spans: List[Any]  # Can be either Span or TextSpan
    unit_type: Optional[str] = None
    confidence: float = 1.0
    page_number: Optional[int] = None  # Add this back as optional
    
    @property
    def text(self) -> str:
        """Combined text of all spans"""
        return " ".join(span.text for span in self.spans)
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box covering all spans"""
        if not self.spans:
            return (0, 0, 0, 0)
        
        x0 = min(span.bbox[0] for span in self.spans)
        y0 = min(span.bbox[1] for span in self.spans)
        x1 = max(span.bbox[2] for span in self.spans)
        y1 = max(span.bbox[3] for span in self.spans)
        
        return (x0, y0, x1, y1)

@dataclass
class Document:
    """Complete document with spans and labels"""
    doc_id: str
    spans: List[Span]
    labels: Dict[Tuple[int, str], Label]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'spans': [asdict(span) for span in self.spans],
            'labels': {f"{k[0]}_{k[1]}": asdict(v) for k, v in self.labels.items()}
        }