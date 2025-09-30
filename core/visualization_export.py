import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
from matplotlib import pyplot as plt
sys.path.append(str(Path(__file__).parent.parent))
from .schematas import Span, Label, SemanticUnit, TextSpan


def convert_span_to_textspan(span: Span) -> TextSpan:
    """Convert internal Span to visualization TextSpan"""
    return TextSpan(
        page_number=span.page_number,
        span_id=span.span_id,
        bbox=span.bbox,
        text=span.text,
        font_size=span.font_size,
        bold=span.bold,
        italic=span.italic,
        line_height=span.line_height,
        x_center=span.x_center,
        y_bottom=span.y_bottom,
        column=span.column,
        reading_order=span.reading_order
    )

def classify_unit_type(unit_spans: List[Span]) -> str:
    """Classify semantic unit type based on heuristics"""
    if not unit_spans:
        return 'Misc'
    
    # Get first span for initial classification
    first_span = unit_spans[0]
    combined_text = ' '.join(s.text for s in unit_spans)
    avg_font_size = sum(s.font_size for s in unit_spans) / len(unit_spans)
    
    # Simple heuristics for classification
    if len(combined_text) < 50 and avg_font_size > 16:
        return 'Title'
    elif len(combined_text) < 100 and avg_font_size > 14:
        return 'Header'
    elif first_span.y_bottom < 100:  # Near top of page
        return 'Header'
    elif first_span.y_bottom > 750:  # Near bottom of page
        return 'Footer'
    elif len(combined_text) < 30 and any(char.isdigit() for char in combined_text[:3]):
        return 'Enumeration'
    elif 'http' in combined_text.lower() or 'www.' in combined_text.lower():
        return 'Url'
    elif len(combined_text) < 5 and combined_text.replace(' ', '').replace('.', '').isdigit():
        return 'PageNumber'
    elif len(combined_text) < 150:
        return 'Caption'
    else:
        return 'TextItem'

def classify_individual_spans(spans: List[Span]) -> Dict[str, str]:
    """Classify individual spans for the left-side visualization"""
    classifications = {}
    
    for span in spans:
        # Simple heuristic classification based on individual span properties
        if span.font_size > 16:
            class_type = 'Title'
        elif span.font_size > 14:
            class_type = 'Header'
        elif span.y_bottom < 100:
            class_type = 'Header'
        elif span.y_bottom > 750:
            class_type = 'Footer'
        elif len(span.text) < 30 and any(char.isdigit() for char in span.text[:min(3, len(span.text))]):
            class_type = 'Enumeration'
        elif 'http' in span.text.lower() or 'www.' in span.text.lower():
            class_type = 'Url'
        elif len(span.text) < 5 and span.text.replace(' ', '').replace('.', '').isdigit():
            class_type = 'PageNumber'
        elif len(span.text) < 50:
            class_type = 'Caption'
        else:
            class_type = 'TextItem'
        
        classifications[span.span_id] = class_type
    
    return classifications

def create_semantic_units_from_labels(
    spans: List[Span],
    labels: List[Label]
) -> Tuple[List[SemanticUnit], List[TextSpan], Dict[str, str]]:
    """Convert labeled spans into SemanticUnit objects for visualization"""
    
    # Create label lookup
    label_map = {}
    for label in labels:
        label_map[(label.page_number, label.span_id)] = label
    
    # Group spans by unit_id
    units_dict = {}
    for span in spans:
        key = (span.page_number, span.span_id)
        if key in label_map:
            label = label_map[key]
            unit_id = label.unit_id
            
            if unit_id not in units_dict:
                units_dict[unit_id] = {
                    'page': span.page_number,
                    'unit_id': unit_id,
                    'spans': []
                }
            units_dict[unit_id]['spans'].append(span)
    
    # Create SemanticUnit objects
    semantic_units = []
    for unit_data in units_dict.values():
        unit_spans = unit_data['spans']
        if not unit_spans:
            continue
        
        # Sort spans by reading order
        unit_spans.sort(key=lambda s: s.reading_order)
        
        # Calculate bounding box for entire unit
        min_x = min(s.bbox[0] for s in unit_spans)
        min_y = min(s.bbox[1] for s in unit_spans)
        max_x = max(s.bbox[2] for s in unit_spans)
        max_y = max(s.bbox[3] for s in unit_spans)
        
        # Combine text
        combined_text = ' '.join(s.text for s in unit_spans)
        
        # Convert spans to TextSpan objects
        text_spans = [convert_span_to_textspan(s) for s in unit_spans]
        
        # Classify unit type
        unit_type = classify_unit_type(unit_spans)
        
        semantic_unit = SemanticUnit(
            page_number=unit_data['page'],
            unit_id=unit_data['unit_id'],
            unit_type=unit_type,
            bbox=(min_x, min_y, max_x, max_y),
            text=combined_text,
            spans=text_spans
        )
        semantic_units.append(semantic_unit)
    
    # Convert all original spans to TextSpan objects
    all_text_spans = [convert_span_to_textspan(s) for s in spans]
    
    # Classify individual spans
    span_classifications = classify_individual_spans(spans)
    
    return semantic_units, all_text_spans, span_classifications

def export_comparison_visualization(
    doc_id: str,
    data_dir: Path = Path("data"),
    output_dir: Path = Path("exports"),
    output_filename: Optional[str] = None
) -> str:
    """Export comparison PDF visualization for a document"""
    
    from core.io import load_spans, load_labels
    
    # Load data
    spans = load_spans(doc_id, data_dir)
    labels = load_labels(doc_id, data_dir)
    
    if not spans:
        raise ValueError(f"No spans found for document {doc_id}")
    
    if not labels:
        raise ValueError(f"No labels found for document {doc_id}. Please label the document first.")
    
    # Convert to visualization format
    semantic_units, text_spans, span_classifications = create_semantic_units_from_labels(spans, labels)
    
    # Determine total pages
    total_pages = max(s.page_number for s in spans)
    
    # Create output path
    if output_filename is None:
        output_filename = f"comparison_{doc_id}.pdf"
    
    output_path = output_dir / output_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison PDF
    result_path = create_comparison_pdf(
        semantic_units=semantic_units,
        original_spans=text_spans,
        span_classifications=span_classifications,
        total_pages=total_pages,
        output_path=str(output_path)
    )
    
    return result_path

# Updated color map to match your class names
SEMANTIC_COLOR_MAP = {
    'Title': '#FF6B6B',        # Red
    'Header': '#4ECDC4',       # Teal
    'SectionHeader': '#45B7D1', # Blue
    'TextItem': '#96CEB4',     # Green
    'Caption': '#FECA57',      # Yellow
    'Footnote': '#FF9FF3',     # Pink
    'Footer': '#8B4513',       # Brown
    'Enumeration': '#FFA07A',  # Light Salmon
    'FormulaItem': '#DDA0DD',  # Plum
    'Url': '#20B2AA',          # Light Sea Green
    'PageNumber': '#D3D3D3',   # Light Gray
    'Digit': '#FFB6C1',        # Light Pink
    'Misc': '#000000',         # Black
    'default': '#E0E0E0'       # Default Gray
}

# Color map for individual span classifications (slightly different shades)
SPAN_COLOR_MAP = {
    'Title': '#FF4444',        # Darker Red
    'Header': '#2EAAA4',       # Darker Teal
    'SectionHeader': '#3597B1', # Darker Blue
    'TextItem': '#76AE94',     # Darker Green
    'Caption': '#DEAD37',      # Darker Yellow
    'Footnote': '#DF7FD3',     # Darker Pink
    'Footer': '#6B3410',       # Darker Brown
    'Enumeration': '#DE806A',  # Darker Salmon
    'FormulaItem': '#BD80BD',  # Darker Plum
    'Url': '#108A8A',          # Darker Sea Green
    'PageNumber': '#B3B3B3',   # Darker Gray
    'Digit': '#DF96A1',        # Darker Light Pink
    'Misc': '#000000',         # Black
    'default': '#C0C0C0'       # Darker Default Gray
}


class LabelPlacer:
    """Helper class to manage non-overlapping label placement with connecting lines."""
    
    def __init__(self, page_width: float, page_height: float):
        self.page_width = page_width
        self.page_height = page_height
        self.placed_labels = []  # List of (x, y, width, height) for placed labels
        
    def get_text_bbox(self, text: str, fontsize: int) -> Tuple[float, float]:
        """Estimate text bounding box dimensions."""
        lines = text.split('\n')
        width = max(len(line) for line in lines) * fontsize * 0.6 + 8  # padding
        height = len(lines) * fontsize * 1.2 + 6  # line spacing + padding
        return width, height
    
    def check_overlap(self, x: float, y: float, width: float, height: float) -> bool:
        """Check if a label at given position would overlap with existing labels."""
        for placed_x, placed_y, placed_w, placed_h in self.placed_labels:
            if (x < placed_x + placed_w and x + width > placed_x and
                y < placed_y + placed_h and y + height > placed_y):
                return True
        return False
    
    def get_unit_center(self, unit_bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Get the center point of a unit."""
        x0, y0, x1, y1 = unit_bbox
        return (x0 + x1) / 2, (y0 + y1) / 2
    
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def find_best_label_position(self, unit_bbox: Tuple[float, float, float, float], 
                                text: str, fontsize: int = 8) -> Tuple[float, float, bool]:
        """Find the best position for the label, prioritizing proximity to unit.
        Returns (x, y, needs_connector_line)."""
        x0, y0, x1, y1 = unit_bbox
        unit_center = self.get_unit_center(unit_bbox)
        label_w, label_h = self.get_text_bbox(text, fontsize)
        
        # Define positions in order of preference (closest first)
        # Start with positions very close to the unit
        close_positions = [
            # Just outside the unit boundaries
            (x0 - label_w - 2, y0),  # Left, aligned with top
            (x1 + 2, y0),  # Right, aligned with top
            (x0, y0 - label_h - 2),  # Above, aligned with left
            (x0, y1 + 2),  # Below, aligned with left
            # Corner positions close to unit
            (x0 - label_w - 2, y0 - label_h - 2),  # Top-left
            (x1 + 2, y0 - label_h - 2),  # Top-right
            (x0 - label_w - 2, y1 + 2),  # Bottom-left
            (x1 + 2, y1 + 2),  # Bottom-right
        ]
        
        # Try close positions first
        best_pos = None
        min_distance = float('inf')
        
        for pos_x, pos_y in close_positions:
            # Check bounds
            if (pos_x >= 0 and pos_y >= 0 and 
                pos_x + label_w <= self.page_width and 
                pos_y + label_h <= self.page_height):
                
                if not self.check_overlap(pos_x, pos_y, label_w, label_h):
                    # Found a close position, use it immediately
                    self.placed_labels.append((pos_x, pos_y, label_w, label_h))
                    return pos_x, pos_y, False  # No connector line needed
        
        # If no close position available, try slightly farther positions
        medium_positions = [
            (x0 - label_w - 10, y0),  # Left with more space
            (x1 + 10, y0),  # Right with more space
            (x0, y0 - label_h - 10),  # Above with more space
            (x0, y1 + 10),  # Below with more space
            # More corner positions
            (x0 - label_w - 10, y0 - label_h - 10),
            (x1 + 10, y0 - label_h - 10),
            (x0 - label_w - 10, y1 + 10),
            (x1 + 10, y1 + 10),
        ]
        
        for pos_x, pos_y in medium_positions:
            if (pos_x >= 0 and pos_y >= 0 and 
                pos_x + label_w <= self.page_width and 
                pos_y + label_h <= self.page_height):
                
                if not self.check_overlap(pos_x, pos_y, label_w, label_h):
                    distance = self.calculate_distance((pos_x + label_w/2, pos_y + label_h/2), unit_center)
                    if distance < min_distance:
                        min_distance = distance
                        best_pos = (pos_x, pos_y)
        
        if best_pos:
            self.placed_labels.append((best_pos[0], best_pos[1], label_w, label_h))
            return best_pos[0], best_pos[1], True  # Needs connector line
        
        # Last resort: try a systematic grid search near the unit
        return self.find_systematic_position(unit_bbox, text, fontsize)
    
    def find_systematic_position(self, unit_bbox: Tuple[float, float, float, float], 
                               text: str, fontsize: int = 8) -> Tuple[float, float, bool]:
        """Systematic search for label position when preferred positions are taken."""
        x0, y0, x1, y1 = unit_bbox
        unit_center = self.get_unit_center(unit_bbox)
        label_w, label_h = self.get_text_bbox(text, fontsize)
        
        # Search in expanding squares around the unit
        for radius in range(15, 100, 15):  # Search in 15-pixel increments
            positions = []
            
            # Generate positions in a circle around the unit
            for angle in range(0, 360, 30):  # Every 30 degrees
                rad = math.radians(angle)
                pos_x = unit_center[0] + radius * math.cos(rad) - label_w/2
                pos_y = unit_center[1] + radius * math.sin(rad) - label_h/2
                positions.append((pos_x, pos_y))
            
            # Sort by distance to unit center
            positions.sort(key=lambda p: self.calculate_distance(
                (p[0] + label_w/2, p[1] + label_h/2), unit_center))
            
            for pos_x, pos_y in positions:
                if (pos_x >= 0 and pos_y >= 0 and 
                    pos_x + label_w <= self.page_width and 
                    pos_y + label_h <= self.page_height):
                    
                    if not self.check_overlap(pos_x, pos_y, label_w, label_h):
                        self.placed_labels.append((pos_x, pos_y, label_w, label_h))
                        return pos_x, pos_y, True  # Needs connector line
        
        # Final fallback: place inside unit
        final_x = x0 + 2
        final_y = y0 + 2
        self.placed_labels.append((final_x, final_y, label_w, label_h))
        return final_x, final_y, False  # Inside unit, no connector needed


def draw_connector_line(ax, unit_bbox: Tuple[float, float, float, float], 
                       label_pos: Tuple[float, float], label_size: Tuple[float, float]):
    """Draw a subtle connector line between unit and label."""
    x0, y0, x1, y1 = unit_bbox
    label_x, label_y = label_pos
    label_w, label_h = label_size
    
    # Find the closest points between unit and label
    unit_center_x, unit_center_y = (x0 + x1) / 2, (y0 + y1) / 2
    label_center_x, label_center_y = label_x + label_w/2, label_y + label_h/2
    
    # Find edge points for cleaner connection
    # Unit edge point
    if label_center_x < x0:  # Label is to the left
        unit_point_x = x0
    elif label_center_x > x1:  # Label is to the right
        unit_point_x = x1
    else:  # Label is above or below
        unit_point_x = unit_center_x
    
    if label_center_y < y0:  # Label is above
        unit_point_y = y0
    elif label_center_y > y1:  # Label is below
        unit_point_y = y1
    else:  # Label is to the side
        unit_point_y = unit_center_y
    
    # Label edge point
    if unit_center_x < label_x:  # Unit is to the left of label
        label_point_x = label_x
    elif unit_center_x > label_x + label_w:  # Unit is to the right of label
        label_point_x = label_x + label_w
    else:  # Unit is above or below label
        label_point_x = label_center_x
    
    if unit_center_y < label_y:  # Unit is above label
        label_point_y = label_y
    elif unit_center_y > label_y + label_h:  # Unit is below label
        label_point_y = label_y + label_h
    else:  # Unit is to the side of label
        label_point_y = label_center_y
    
    # Draw a subtle dashed line
    ax.plot([unit_point_x, label_point_x], [unit_point_y, label_point_y], 
           color='gray', linewidth=1, linestyle='--', alpha=0.6)


def create_comparison_pdf(
    semantic_units: List[SemanticUnit],
    original_spans: List[TextSpan] = None,
    span_classifications: Dict[str, str] = None,  # span_id -> classification
    total_pages: int = None,  
    page_width: float = 595.276,
    page_height: float = 841.890,
    output_path: str = None
) -> str:
    """Create a side-by-side comparison of original spans and semantic units with classifications."""

    assert total_pages is not None, "You must provide total_pages explicitly."
    
    # DEBUG: Print classification info
    if span_classifications:
        print(f"DEBUG: Received {len(span_classifications)} span classifications")
        print(f"DEBUG: Sample classifications: {dict(list(span_classifications.items())[:5])}")
        class_counts = {}
        for classification in span_classifications.values():
            class_counts[classification] = class_counts.get(classification, 0) + 1
        print(f"DEBUG: Classification distribution: {class_counts}")
    else:
        print("DEBUG: No span_classifications provided - all will show as 'Misc'")
    
    # Group semantic units by page
    pages_sem = {}
    for unit in semantic_units:
        pages_sem.setdefault(unit.page_number, []).append(unit)

    # Group original spans by page - FIXED: use attribute access
    pages_span = {}
    if original_spans:
        for span in original_spans:
            pages_span.setdefault(span.page_number, []).append(span)  # FIXED: span.page_number

    all_pages = list(range(1, total_pages + 1))

    with PdfPages(output_path) as pdf:
        for page in all_pages:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 11))
            for ax in (ax1, ax2):
                ax.set_xlim(0, page_width)
                ax.set_ylim(0, page_height)
                ax.invert_yaxis()

            # Left side: Original spans with classifications
            ax1.set_title(f'Original Spans with Classifications - Page {page}')
            spans = pages_span.get(page, [])
            
            if spans:
                span_types_on_page = set()
                # DEBUG: Track span ID matching
                missing_classifications = []
                found_classifications = []
                
                for span in spans:
                    x0, y0, x1, y1 = span.bbox
                    width, height = x1 - x0, y1 - y0
                    
                    # Get classification for this span
                    span_id = span.span_id or f"{span.page_number}_{span.bbox}"
                    classification = span_classifications.get(span_id, 'Misc') if span_classifications else 'Misc'
                    
                    # DEBUG: Track what's happening with classifications
                    if span_classifications:
                        if span_id in span_classifications:
                            found_classifications.append((span_id, classification))
                        else:
                            missing_classifications.append(span_id)
                    
                    span_types_on_page.add(classification)
                    
                    # Use span classification color
                    color = SPAN_COLOR_MAP.get(classification, SPAN_COLOR_MAP['default'])
                    
                    rect = mpatches.Rectangle((x0, y0), width, height,
                                             linewidth=1.5, edgecolor='black',
                                             facecolor=color, alpha=0.7)
                    ax1.add_patch(rect)
                    
                    # Always show classification label (make it more visible)
                    if width > 20 and height > 8:  # Lower threshold
                        # Show classification prominently
                        ax1.text(x0 + 2, y0 + 4, classification, fontsize=7, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
                        
                        # Show text preview below if there's space
                        if height > 20:
                            preview = span.text[:15] + "..." if len(span.text) > 15 else span.text
                            ax1.text(x0 + 2, y0 + height - 8, preview, fontsize=5, 
                                    bbox=dict(boxstyle="round,pad=0.1", facecolor='lightgray', alpha=0.7))
                    elif width > 10 and height > 5:  # Very small spans, just show first letter of classification
                        ax1.text(x0 + 1, y0 + 2, classification[0], fontsize=5, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9))
                
                # DEBUG: Print span ID matching results for this page
                if span_classifications:
                    print(f"DEBUG Page {page}: Found {len(found_classifications)} classifications, missing {len(missing_classifications)}")
                    if missing_classifications:
                        print(f"DEBUG: Sample missing span IDs: {missing_classifications[:3]}")
                    if found_classifications:
                        print(f"DEBUG: Sample found classifications: {found_classifications[:3]}")
                
                # Add legend for span classifications
                if span_types_on_page:
                    handles = [mpatches.Patch(color=SPAN_COLOR_MAP.get(t, SPAN_COLOR_MAP['default']), 
                                            alpha=0.7, label=t) for t in sorted(span_types_on_page)]
                    ax1.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.02, 1), fontsize=8)
            else:
                ax1.text(50, 100, "No original spans", fontsize=12, color='gray')

            # Right side: Semantic units (grouped spans) with smart label placement
            ax2.set_title(f'Semantic Units (Grouped Results) - Page {page}')
            units = pages_sem.get(page, [])
            
            if units:
                # Initialize label placer for this page
                label_placer = LabelPlacer(page_width, page_height)
                
                unit_types_on_page = set()
                
                # First pass: draw all the units and span boundaries
                for unit in units:
                    unit_types_on_page.add(unit.unit_type)
                    x0, y0, x1, y1 = unit.bbox
                    width, height = x1 - x0, y1 - y0
                    
                    color = SEMANTIC_COLOR_MAP.get(unit.unit_type, SEMANTIC_COLOR_MAP['default'])
                    
                    # Draw semantic unit boundary (thicker border to show grouping)
                    rect = mpatches.Rectangle((x0, y0), width, height,
                                            linewidth=3, edgecolor='darkred',
                                            facecolor=color, alpha=0.3)
                    ax2.add_patch(rect)
                    
                    # Show individual span boundaries within the unit (subtle dashed lines)
                    for span in unit.spans:
                        sx0, sy0, sx1, sy1 = span.bbox
                        span_rect = mpatches.Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0,
                                                    linewidth=0.5, edgecolor='gray',
                                                    facecolor='none', alpha=0.5, linestyle='--')
                        ax2.add_patch(span_rect)
                
                # Second pass: place labels with smart positioning
                for unit in units:
                    x0, y0, x1, y1 = unit.bbox
                    width, height = x1 - x0, y1 - y0
                    
                    # Create concise label text
                    label_text = f"{unit.unit_type}\n{unit.unit_id}\n({len(unit.spans)})"
                    
                    # Find optimal position for the label
                    label_x, label_y, needs_connector = label_placer.find_best_label_position(
                        unit.bbox, label_text, fontsize=8)
                    
                    # Draw connector line if label is placed far from unit
                    if needs_connector:
                        label_w, label_h = label_placer.get_text_bbox(label_text, 8)
                        draw_connector_line(ax2, unit.bbox, (label_x, label_y), (label_w, label_h))
                    
                    # Place the label
                    ax2.text(label_x, label_y, label_text,
                             fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='black'))
                    
                    # Show text preview if unit is large enough (keep this inside the unit)
                    if width > 150 and height > 40:
                        preview = unit.text[:40] + "..." if len(unit.text) > 40 else unit.text
                        wrapped = textwrap.fill(preview, width=int(width / 8))
                        ax2.text(x0 + width / 2, y0 + height - 20, wrapped,
                                fontsize=6, ha='center', va='center',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.8))
                
                # Add legend for semantic unit types only
                if unit_types_on_page:
                    handles = [mpatches.Patch(color=SEMANTIC_COLOR_MAP.get(t, SEMANTIC_COLOR_MAP['default']), 
                                            alpha=0.7, label=t) for t in sorted(unit_types_on_page)]
                    ax2.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.02, 1), fontsize=8)
            else:
                ax2.text(50, 100, "No semantic units", fontsize=12, color='gray')

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close(fig)

    print(f"✅ Comparison saved to: {output_path}")
    return output_path


def create_comparison_figure(
    page_num: int,
    semantic_units: List[SemanticUnit],
    original_spans: List[TextSpan],
    span_classifications: Optional[Dict[str, str]] = None,
    page_width: float = 595.276,
    page_height: float = 841.890,
) -> plt.Figure:
    """
    Create a matplotlib figure comparing original spans vs semantic units for one page.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 11))

    for ax in (ax1, ax2):
        ax.set_xlim(0, page_width)
        ax.set_ylim(0, page_height)
        ax.invert_yaxis()

    # ----------- LEFT: Original spans and their classifications
    ax1.set_title(f'Original Spans with Classifications - Page {page_num + 1}')
    if original_spans:
        span_types_on_page = set()
        for span in original_spans:
            x0, y0, x1, y1 = span.bbox
            width, height = x1 - x0, y1 - y0

            # Determine classification for this span
            span_id = getattr(span, "span_id", None) or f"{span.page_number}_{span.bbox}"
            classification = span_classifications.get(span_id, 'Misc') if span_classifications else 'Misc'
            span_types_on_page.add(classification)
            color = SPAN_COLOR_MAP.get(classification, SPAN_COLOR_MAP['default'])

            rect = mpatches.Rectangle((x0, y0), width, height, linewidth=1.5, edgecolor='black',
                                     facecolor=color, alpha=0.7)
            ax1.add_patch(rect)

            # Classification label and preview
            if width > 20 and height > 8:
                ax1.text(x0 + 2, y0 + 4, classification, fontsize=7, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
                if height > 20:
                    preview = span.text[:15] + "..." if len(span.text) > 15 else span.text
                    ax1.text(x0 + 2, y0 + height - 8, preview, fontsize=5, 
                             bbox=dict(boxstyle="round,pad=0.1", facecolor='lightgray', alpha=0.7))
            elif width > 10 and height > 5:
                ax1.text(x0 + 1, y0 + 2, classification[0], fontsize=5, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9))

        # Legend for span types
        if span_types_on_page:
            handles = [mpatches.Patch(color=SPAN_COLOR_MAP.get(t, SPAN_COLOR_MAP['default']),
                                      alpha=0.7, label=t) for t in sorted(span_types_on_page)]
            ax1.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.02, 1), fontsize=8)
    else:
        ax1.text(50, 100, "No original spans", fontsize=12, color='gray')

    # ----------- RIGHT: Semantic Units with smart label placement
    ax2.set_title(f'Semantic Units (Grouped Results) - Page {page_num + 1}')
    if semantic_units:
        # Initialize label placer for this page
        label_placer = LabelPlacer(page_width, page_height)
        
        unit_types_on_page = set()
        
        # First pass: draw units and boundaries
        for unit in semantic_units:
            unit_types_on_page.add(unit.unit_type)
            x0, y0, x1, y1 = unit.bbox
            color = SEMANTIC_COLOR_MAP.get(unit.unit_type, SEMANTIC_COLOR_MAP['default'])
            
            # Semantic unit boundary
            rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=3, edgecolor='darkred',
                                    facecolor=color, alpha=0.3)
            ax2.add_patch(rect)
            
            # Show individual span boundaries within the unit (dashed lines)
            for span in getattr(unit, 'spans', []):
                sx0, sy0, sx1, sy1 = span.bbox
                span_rect = mpatches.Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0,
                                              linewidth=0.5, edgecolor='gray',
                                              facecolor='none', alpha=0.5, linestyle='--')
                ax2.add_patch(span_rect)
        
        # Second pass: place labels
        for unit in semantic_units:
            x0, y0, x1, y1 = unit.bbox
            width, height = x1 - x0, y1 - y0
            
            # Create label with smart positioning
            label_text = f"{unit.unit_id}\n{len(getattr(unit, 'spans', []))}"
            label_x, label_y, needs_connector = label_placer.find_best_label_position(
                unit.bbox, label_text, fontsize=8)
            
            # Draw connector line if needed
            if needs_connector:
                label_w, label_h = label_placer.get_text_bbox(label_text, 8)
                draw_connector_line(ax2, unit.bbox, (label_x, label_y), (label_w, label_h))
            
            ax2.text(label_x, label_y, label_text, fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='black'))
            
            # Text preview if large enough (inside unit)
            if width > 150 and height > 40:
                preview = unit.text[:40] + "..." if len(unit.text) > 40 else unit.text
                wrapped = textwrap.fill(preview, width=int(width / 8))
                ax2.text(x0 + width / 2, y0 + height - 20, wrapped,
                        fontsize=6, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.8))
        
        # Legend for unit types
        if unit_types_on_page:
            handles = [mpatches.Patch(color=SEMANTIC_COLOR_MAP.get(t, SEMANTIC_COLOR_MAP['default']),
                                      alpha=0.7, label=t) for t in sorted(unit_types_on_page)]
            ax2.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.02, 1), fontsize=8)
    else:
        ax2.text(50, 100, "No semantic units", fontsize=12, color='gray')

    plt.tight_layout()
    return fig