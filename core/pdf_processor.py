import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re
import shutil
from .schematas import Span

def extract_spans_from_pdf(pdf_path: Path, doc_id: str) -> List[Span]:
    """Extract spans from PDF with enhanced features for boundary detection"""
    doc = fitz.open(pdf_path)
    all_spans = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_dict = page.get_text("dict")
        
        # Page dimensions for normalization
        page_width = page.rect.width
        page_height = page.rect.height
        
        spans_on_page = []
        seen_spans = {}  # For deduplication: key = (bbox, text), value = span
        
        # Extract blocks
        for block_idx, block in enumerate(page_dict["blocks"]):
            if block["type"] == 0:  # Text block
                for line_idx, line in enumerate(block["lines"]):
                    for span_idx, span in enumerate(line["spans"]):
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        # Extract span properties
                        bbox = span["bbox"]  # (x0, y0, x1, y1)
                        
                        # Create deduplication key
                        # Round bbox coordinates to avoid floating point precision issues
                        bbox_rounded = tuple(round(coord, 2) for coord in bbox)
                        dedup_key = (bbox_rounded, text)
                        
                        # Skip if we've seen this exact span before (same position and text)
                        if dedup_key in seen_spans:
                            continue
                        
                        font_size = span["size"]
                        font_flags = span["flags"]
                        font_name = span.get("font", "")
                        
                        # Determine style flags
                        is_bold = bool(font_flags & 2**4)  # Bold flag
                        is_italic = bool(font_flags & 2**1)  # Italic flag
                        
                        # Calculate basic geometry
                        x_center = (bbox[0] + bbox[2]) / 2
                        y_bottom = bbox[3]
                        line_height = bbox[3] - bbox[1]
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        
                        # Column detection (simple heuristic)
                        column = 0 if x_center < page_width / 2 else 1
                        
                        # Enhanced text analysis
                        char_count = len(text)
                        word_count = len(text.split())
                        
                        # Key semantic features
                        is_whitespace = text.isspace()
                        starts_with_number = bool(re.match(r'^\d+\.?\s*', text))
                        starts_with_bullet = bool(re.match(r'^[\u2022\u2023\u25E6\u2043\u2219•·▪▫‣⁃-]', text))
                        is_all_caps = text.isupper() and len(text) > 2
                        ends_with_colon = text.endswith(':')
                        ends_with_period = text.endswith('.')
                        
                        # Create span ID - will be updated after sorting
                        span_id = f"p{page_num+1}_s{len(spans_on_page)+1}_temp"
                        
                        span_obj = Span(
                            doc_id=doc_id,
                            page_number=page_num + 1,
                            span_id=span_id,
                            bbox=tuple(bbox),
                            text=text,
                            font_size=font_size,
                            bold=is_bold,
                            italic=is_italic,
                            line_height=line_height,
                            x_center=x_center,
                            y_bottom=y_bottom,
                            column=column,
                            reading_order=0,  # Will be set after sorting
                            rotation=0.0,
                            language=None,
                            is_whitespace=is_whitespace,
                            char_count=char_count,
                            word_count=word_count,
                            font_name=font_name,
                            width=width,
                            height=height,
                            starts_with_number=starts_with_number,
                            starts_with_bullet=starts_with_bullet,
                            is_all_caps=is_all_caps,
                            ends_with_colon=ends_with_colon,
                            ends_with_period=ends_with_period,
                        )
                        
                        spans_on_page.append(span_obj)
                        seen_spans[dedup_key] = span_obj
        
        # Sort spans by reading order (top-to-bottom, left-to-right)
        spans_on_page.sort(key=lambda s: (s.y_bottom, s.x_center))
        
        # Update span IDs and reading order after sorting
        for idx, span in enumerate(spans_on_page):
            # Update span_id to final value
            object.__setattr__(span, 'span_id', f"p{page_num+1}_s{idx+1}")
            object.__setattr__(span, 'reading_order', idx + 1)
            
            # Add positional context
            object.__setattr__(span, 'is_first_on_page', idx == 0)
            object.__setattr__(span, 'is_last_on_page', idx == len(spans_on_page) - 1)
            object.__setattr__(span, 'relative_position', idx / len(spans_on_page) if spans_on_page else 0)
        
        all_spans.extend(spans_on_page)
    
    # Add cross-page context
    _add_sequence_features(all_spans)
    
    # Final deduplication check across all pages (in case of repeated headers/footers)
    all_spans = _deduplicate_cross_page(all_spans)
    
    doc.close()
    return all_spans

def _deduplicate_cross_page(spans: List[Span]) -> List[Span]:
    """
    Remove duplicate spans that appear across multiple pages 
    (like repeated headers/footers) keeping only the first occurrence
    """
    seen_content: Dict[str, List[Span]] = {}
    
    # Group spans by text content
    for span in spans:
        text_key = span.text.strip()
        if text_key not in seen_content:
            seen_content[text_key] = []
        seen_content[text_key].append(span)
    
    # Find potential duplicates (same text appearing on multiple pages)
    duplicates_to_remove = set()
    for text_key, span_list in seen_content.items():
        if len(span_list) > 1:
            # Check if these are likely headers/footers (similar position on different pages)
            pages = {}
            for span in span_list:
                if span.page_number not in pages:
                    pages[span.page_number] = []
                pages[span.page_number].append(span)
            
            # If text appears once per page in similar position, likely header/footer
            if len(pages) > 1 and all(len(spans) == 1 for spans in pages.values()):
                # Check if positions are similar (within 10 pixels)
                positions = [(s.bbox[0], s.bbox[1]) for spans in pages.values() for s in spans]
                x_positions = [p[0] for p in positions]
                y_positions = [p[1] for p in positions]
                
                x_variance = max(x_positions) - min(x_positions)
                y_variance = max(y_positions) - min(y_positions)
                
                # If positions are very similar across pages, it's likely a header/footer
                if x_variance < 10 and y_variance < 10:
                    # Keep only the first occurrence
                    sorted_spans = sorted(span_list, key=lambda s: (s.page_number, s.reading_order))
                    for span in sorted_spans[1:]:
                        duplicates_to_remove.add(span.span_id)
    
    # Filter out duplicates
    filtered_spans = [s for s in spans if s.span_id not in duplicates_to_remove]
    
    # Re-index span IDs if we removed any
    if len(filtered_spans) < len(spans):
        print(f"Removed {len(spans) - len(filtered_spans)} duplicate spans (likely headers/footers)")
        
        # Group by page and re-index
        page_spans = {}
        for span in filtered_spans:
            if span.page_number not in page_spans:
                page_spans[span.page_number] = []
            page_spans[span.page_number].append(span)
        
        # Re-index spans
        final_spans = []
        for page_num in sorted(page_spans.keys()):
            page_span_list = sorted(page_spans[page_num], key=lambda s: s.reading_order)
            for idx, span in enumerate(page_span_list):
                object.__setattr__(span, 'span_id', f"p{page_num}_s{idx+1}")
                object.__setattr__(span, 'reading_order', idx + 1)
            final_spans.extend(page_span_list)
        
        return final_spans
    
    return filtered_spans

def _add_sequence_features(spans: List[Span]) -> None:
    """Add features based on sequence context (sliding window approach)"""
    for i, span in enumerate(spans):
        # Previous span features
        if i > 0:
            prev_span = spans[i-1]
            object.__setattr__(span, 'prev_font_size', prev_span.font_size)
            object.__setattr__(span, 'prev_is_bold', prev_span.bold)
            object.__setattr__(span, 'prev_ends_period', prev_span.ends_with_period)
            object.__setattr__(span, 'font_size_changed', abs(span.font_size - prev_span.font_size) > 0.5)
            object.__setattr__(span, 'style_changed', span.bold != prev_span.bold or span.italic != prev_span.italic)
            
            # Calculate vertical gap only if on same page
            if span.page_number == prev_span.page_number:
                object.__setattr__(span, 'vertical_gap', span.bbox[1] - prev_span.bbox[3])
            else:
                object.__setattr__(span, 'vertical_gap', 0.0)
        else:
            object.__setattr__(span, 'prev_font_size', span.font_size)
            object.__setattr__(span, 'prev_is_bold', False)
            object.__setattr__(span, 'prev_ends_period', False)
            object.__setattr__(span, 'font_size_changed', False)
            object.__setattr__(span, 'style_changed', False)
            object.__setattr__(span, 'vertical_gap', 0.0)
        
        # Next span features (lookahead)
        if i < len(spans) - 1:
            next_span = spans[i+1]
            object.__setattr__(span, 'next_font_size', next_span.font_size)
            object.__setattr__(span, 'next_starts_bullet', next_span.starts_with_bullet)
        else:
            object.__setattr__(span, 'next_font_size', span.font_size)
            object.__setattr__(span, 'next_starts_bullet', False)
        
        # Reading order gap (critical for boundary detection)
        if i > 0:
            object.__setattr__(span, 'reading_order_gap', span.reading_order - spans[i-1].reading_order)
            object.__setattr__(span, 'same_page_as_prev', span.page_number == spans[i-1].page_number)
        else:
            object.__setattr__(span, 'reading_order_gap', 1)
            object.__setattr__(span, 'same_page_as_prev', True)

def save_pdf_copy(pdf_path: Path, doc_id: str, data_dir: Path = Path("data")) -> Path:
    """Save a copy of the PDF in the data directory"""
    doc_dir = data_dir / "docs" / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_copy_path = doc_dir / "document.pdf"
    
    # Copy PDF file
    shutil.copy2(pdf_path, pdf_copy_path)
    
    return pdf_copy_path

def render_pdf_page(pdf_path: Path, page_num: int, zoom: float = 1.0) -> bytes:
    """Render a PDF page as PNG image bytes"""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # 0-indexed
    
    # Create transformation matrix for zoom
    mat = fitz.Matrix(zoom, zoom)
    
    # Render page to pixmap
    pix = page.get_pixmap(matrix=mat)
    
    # Convert to PNG bytes
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