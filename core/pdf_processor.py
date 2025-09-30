import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Optional
import json
from .schematas import Span
import hashlib

def extract_spans_from_pdf(pdf_path: Path, doc_id: str) -> List[Span]:
    """Extract spans from PDF with position and style information"""
    doc = fitz.open(pdf_path)
    all_spans = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_dict = page.get_text("dict")
        
        # Page dimensions
        page_width = page.rect.width
        page_height = page.rect.height
        
        spans_on_page = []
        
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
                        font_size = span["size"]
                        font_flags = span["flags"]
                        
                        # Determine style flags
                        is_bold = bool(font_flags & 2**20)  # Bold flag
                        is_italic = bool(font_flags & 2**1)  # Italic flag
                        
                        # Calculate derived features
                        x_center = (bbox[0] + bbox[2]) / 2
                        y_bottom = bbox[3]
                        line_height = bbox[3] - bbox[1]
                        
                        # Determine column (simple heuristic)
                        column = 0 if x_center < page_width / 2 else 1
                        
                        # Create span ID
                        span_id = f"p{page_num+1}_s{len(spans_on_page)+1}"
                        
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
                            reading_order=len(spans_on_page) + 1
                        )
                        
                        spans_on_page.append(span_obj)
        
        # Sort spans by reading order (top-to-bottom, left-to-right)
        spans_on_page.sort(key=lambda s: (s.column, s.y_bottom, s.x_center))
        
        # Update reading order
        for idx, span in enumerate(spans_on_page):
            span.reading_order = idx + 1
        
        all_spans.extend(spans_on_page)
    
    doc.close()
    return all_spans

def save_pdf_copy(pdf_path: Path, doc_id: str, data_dir: Path = Path("data")) -> Path:
    """Save a copy of the PDF in the data directory"""
    doc_dir = data_dir / "docs" / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_copy_path = doc_dir / "document.pdf"
    
    # Copy PDF file
    import shutil
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