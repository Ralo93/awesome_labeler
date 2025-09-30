# schemas.py
from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple, Dict, List

BBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF coords

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

# for visualization
@dataclass
class SemanticUnit:
    """Grouped semantic unit containing multiple spans"""
    page_number: int
    unit_id: str
    unit_type: str  # Classification like 'Title', 'Header', 'TextItem', etc.
    bbox: Tuple[float, float, float, float]  # Bounding box of entire unit
    text: str  # Combined text
    spans: List[TextSpan]  # Original spans in this unit


@dataclass()
class Span:
    """Immutable span representation.
    bbox = (x0, y0, x1, y1) with PDF-style coordinates (origin bottom-left)."""
    doc_id: str
    page_number: int
    span_id: str
    bbox: BBox
    text: str
    font_size: float
    bold: bool
    italic: bool
    line_height: float
    x_center: float
    y_bottom: float
    column: int
    reading_order: int

    line_id: Optional[int] = None
    prev_id: Optional[str] = None
    next_id: Optional[str] = None

    # Optional but useful extras
    rotation: float = 0.0                 # degrees
    language: Optional[str] = None        # e.g., 'de', 'en'
    is_whitespace: bool = False
    char_count: int = 0
    word_count: int = 0

    @property
    def left(self) -> float: return self.bbox[0]
    @property
    def bottom(self) -> float: return self.bbox[1]
    @property
    def right(self) -> float: return self.bbox[2]
    @property
    def top(self) -> float: return self.bbox[3]
    @property
    def width(self) -> float: return max(0.0, self.right - self.left)
    @property
    def height(self) -> float: return max(0.0, self.top - self.bottom)

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "page_number": self.page_number,
            "span_id": self.span_id,
            "bbox": list(self.bbox),
            "text": self.text,
            "font_size": self.font_size,
            "bold": self.bold,
            "italic": self.italic,
            "line_height": self.line_height,
            "x_center": self.x_center,
            "y_bottom": self.y_bottom,
            "column": self.column,
            "reading_order": self.reading_order,
            "rotation": self.rotation,
            "language": self.language,
            "is_whitespace": self.is_whitespace,
            "char_count": self.char_count,
            "word_count": self.word_count,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        d = dict(data)
        d["bbox"] = tuple(d["bbox"])
        # backfill optional counts if missing
        d.setdefault("char_count", len(d.get("text", "")))
        d.setdefault("word_count", len(d.get("text", "").split()))
        d.setdefault("is_whitespace", d.get("text", "").strip() == "")
        return cls(**d)


@dataclass
class Label:
    """Boundary label for a span (does the current span start a new unit?)."""
    doc_id: str
    page_number: int
    span_id: str
    boundary: Literal["new", "continue"]
    unit_id: Optional[str] = None
    confidence: Optional[float] = None  # 0..1

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "page_number": self.page_number,
            "span_id": self.span_id,
            "unit_id": self.unit_id,
            "boundary": self.boundary,
            "confidence": self.confidence,

        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class Rule:
    """Optional rule-based classification per span."""
    doc_id: str
    page_number: int
    span_id: str

    # coarse classes (one span may satisfy multiple)
    is_header: int
    is_caption: int
    is_page_num: int
    is_textitem: int
    is_toc: int
    is_digit: int
    is_misc: int

    # useful extras
    header_level: Optional[int] = None      # e.g., 1..6 if detected
    is_list_bullet: int = 0
    is_list_number: int = 0
    is_footnote: int = 0
    is_reference: int = 0

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "page_number": self.page_number,
            "span_id": self.span_id,
            "is_header": self.is_header,
            "is_caption": self.is_caption,
            "is_page_num": self.is_page_num,
            "is_textitem": self.is_textitem,
            "is_toc": self.is_toc,
            "is_digit": self.is_digit,
            "is_misc": self.is_misc,
            "header_level": self.header_level,
            "is_list_bullet": self.is_list_bullet,
            "is_list_number": self.is_list_number,
            "is_footnote": self.is_footnote,
            "is_reference": self.is_reference,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)
