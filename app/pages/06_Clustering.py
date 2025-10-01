import io
import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2
import json
import torch
from dataclasses import dataclass, field

# Page configuration
st.set_page_config(page_title="MinerU2 Layout Detection", page_icon="🔬", layout="wide")

# Initialize session state
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'mineru_model' not in st.session_state:
    st.session_state.mineru_model = None

# Category types from MinerU
CATEGORY_TYPES = {
    'title': {'color': '#FF6B6B', 'label': 'Title'},
    'text': {'color': '#4ECDC4', 'label': 'Text'},
    'plain_text': {'color': '#4ECDC4', 'label': 'Plain Text'},
    'figure': {'color': '#95E77E', 'label': 'Figure'},
    'figure_caption': {'color': '#7FD157', 'label': 'Figure Caption'},
    'table': {'color': '#FFE66D', 'label': 'Table'},
    'table_caption': {'color': '#F4D03F', 'label': 'Table Caption'},
    'table_footnote': {'color': '#E8C547', 'label': 'Table Footnote'},
    'formula': {'color': '#DDA0DD', 'label': 'Formula'},
    'isolate_formula': {'color': '#BA55D3', 'label': 'Block Formula'},
    'formula_caption': {'color': '#9370DB', 'label': 'Formula Caption'},
    'header': {'color': '#FFA07A', 'label': 'Header'},
    'footer': {'color': '#FFB6C1', 'label': 'Footer'},
    'page_number': {'color': '#D3D3D3', 'label': 'Page Number'},
    'list': {'color': '#87CEEB', 'label': 'List'},
    'enumeration': {'color': '#6495ED', 'label': 'Enumeration'},
    'abandon': {'color': '#808080', 'label': 'Abandoned'},
    'embedding': {'color': '#FF69B4', 'label': 'Inline Formula'},
}

@dataclass
class ContentBlock:
    """Represents a detected content block from MinerU"""
    type: str
    bbox: List[float]  # [xmin, ymin, xmax, ymax] normalized to [0, 1] or image dimensions
    content: Optional[str] = None
    page_idx: int = 0
    confidence: float = 1.0
    angle: Optional[int] = None

@st.cache_resource
def load_mineru_model():
    """Load MinerU2 model"""
    try:
        from mineru_vl_utils import MinerUClient
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        
        with st.spinner("Loading MinerU2.5-2509-1.2B model..."):
            # Load model and processor
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "opendatalab/MinerU2.5-2509-1.2B",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            
            processor = AutoProcessor.from_pretrained(
                "opendatalab/MinerU2.5-2509-1.2B",
                use_fast=True
            )
            
            # Create MinerU client
            client = MinerUClient(
                backend="transformers",
                model=model,
                processor=processor
            )
            
            return client
    except ImportError:
        st.error("""
        MinerU Vision-Language utilities not installed. Please install:
        ```bash
        pip install mineru-vl-utils
        pip install transformers>=4.56.0
        pip install torch torchvision
        ```
        """)
        return None
    except Exception as e:
        st.error(f"Failed to load MinerU model: {str(e)}")
        return None

def render_pdf_page(pdf_path: Path, page_num: int) -> Image.Image:
    """Render a PDF page as an image"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        doc.close()
        return img
    except ImportError:
        st.error("PyMuPDF not installed. Please install: pip install pymupdf")
        return None

def detect_layout_mineru(image: Image.Image, client) -> List[ContentBlock]:
    """Perform layout detection using MinerU2"""
    if client is None:
        return []
    
    try:
        # Two-step extraction: layout detection + content recognition
        extracted_blocks = client.two_step_extract(image)
        
        # Convert to ContentBlock objects
        blocks = []
        for block in extracted_blocks:
            content_block = ContentBlock(
                type=block.get('type', 'text'),
                bbox=block.get('bbox', [0, 0, 1, 1]),
                content=block.get('content', ''),
                confidence=block.get('score', 1.0),
                angle=block.get('angle', None)
            )
            blocks.append(content_block)
        
        return blocks
        
    except Exception as e:
        st.error(f"Layout detection failed: {str(e)}")
        return []

def detect_layout_fallback(image: Image.Image) -> List[ContentBlock]:
    """Fallback layout detection using simple CV methods"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blocks = []
    img_h, img_w = gray.shape[:2]
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter small contours
        if w < 20 or h < 10:
            continue
        
        # Normalize bbox to [0, 1]
        bbox = [x/img_w, y/img_h, (x+w)/img_w, (y+h)/img_h]
        
        # Simple heuristic classification based on size and position
        if h > img_h * 0.15:
            block_type = 'text'
        elif y < img_h * 0.1:
            block_type = 'header'
        elif y > img_h * 0.9:
            block_type = 'footer'
        elif w > img_w * 0.7 and h < img_h * 0.05:
            block_type = 'title'
        else:
            block_type = 'plain_text'
        
        blocks.append(ContentBlock(
            type=block_type,
            bbox=bbox,
            content=f"{block_type.replace('_', ' ').title()} Block",
            confidence=0.8
        ))
    
    return blocks

def visualize_layout(image: Image.Image, blocks: List[ContentBlock], show_labels: bool = True) -> plt.Figure:
    """Visualize detected layout blocks with bounding boxes"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    
    # Display image
    ax.imshow(image)
    ax.axis('off')
    
    img_w, img_h = image.size
    
    # Draw bounding boxes
    for idx, block in enumerate(blocks):
        # Get block type info
        block_info = CATEGORY_TYPES.get(block.type, {'color': '#808080', 'label': 'Unknown'})
        color = block_info['color']
        label = block_info['label']
        
        # Convert normalized bbox to pixel coordinates
        if all(0 <= v <= 1 for v in block.bbox):
            # Normalized coordinates
            x1 = block.bbox[0] * img_w
            y1 = block.bbox[1] * img_h
            x2 = block.bbox[2] * img_w
            y2 = block.bbox[3] * img_h
        else:
            # Already in pixel coordinates
            x1, y1, x2, y2 = block.bbox
        
        width = x2 - x1
        height = y2 - y1
        
        # Draw rectangle
        rect = Rectangle(
            (x1, y1), width, height,
            linewidth=2, 
            edgecolor=color,
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add label
        if show_labels:
            # Position label
            label_y = y1 - 5 if y1 > 20 else y1 + height + 15
            
            # Add background for better readability
            ax.text(
                x1, label_y,
                f"{idx}: {label}",
                color='white',
                fontsize=8,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7)
            )
    
    plt.title(f"Layout Detection - {len(blocks)} blocks detected", fontsize=14, pad=20)
    plt.tight_layout()
    
    return fig

def create_legend() -> plt.Figure:
    """Create a legend for block types"""
    fig, ax = plt.subplots(figsize=(3, 6))
    ax.axis('off')
    
    # Create legend elements
    legend_elements = []
    for block_type, info in CATEGORY_TYPES.items():
        legend_elements.append(
            patches.Patch(color=info['color'], label=info['label'])
        )
    
    ax.legend(
        handles=legend_elements,
        loc='center',
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Block Types",
        title_fontsize=12,
        fontsize=10
    )
    
    return fig

# Sidebar
with st.sidebar:
    st.header("📄 Document Selection")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=['pdf'],
        help="Select a PDF file to analyze"
    )
    
    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{uploaded_file.name}")
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.current_doc = temp_path
        st.success(f"Loaded: {uploaded_file.name}")
    
    st.divider()
    
    # Model settings
    st.header("⚙️ Model Settings")
    
    use_mineru = st.checkbox(
        "Use MinerU2 Model",
        value=True,
        help="Use MinerU2.5-2509-1.2B for layout detection"
    )
    
    if use_mineru:
        if st.button("🔧 Load MinerU2 Model", type="primary", use_container_width=True):
            with st.spinner("Loading model..."):
                st.session_state.mineru_model = load_mineru_model()
                if st.session_state.mineru_model:
                    st.success("Model loaded successfully!")
    else:
        st.info("Using fallback CV-based detection")
    
    st.divider()
    
    # Visualization settings
    st.header("🎨 Visualization")
    
    show_labels = st.checkbox(
        "Show Labels",
        value=True,
        help="Display block type labels on detection"
    )
    
    show_legend = st.checkbox(
        "Show Legend",
        value=True,
        help="Display color legend for block types"
    )
    
    st.divider()
    
    # Detection stats
    if st.session_state.detection_results:
        st.header("📊 Detection Stats")
        
        # Count blocks by type
        type_counts = {}
        for block in st.session_state.detection_results:
            type_counts[block.type] = type_counts.get(block.type, 0) + 1
        
        for block_type, count in sorted(type_counts.items()):
            info = CATEGORY_TYPES.get(block_type, {'label': block_type})
            st.metric(info['label'], count)

# Main content
st.title("🔬 MinerU2 Layout Detection")
st.markdown("Advanced document layout analysis using MinerU2 Vision-Language Model")

if st.session_state.current_doc:
    pdf_path = Path(st.session_state.current_doc)
    
    if pdf_path.exists():
        # Get number of pages
        try:
            import fitz
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()
        except:
            num_pages = 1
        
        # Page navigation
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            page_num = st.slider(
                "Select Page",
                min_value=1,
                max_value=num_pages,
                value=st.session_state.current_page,
                key="page_slider"
            )
            st.session_state.current_page = page_num
        
        # Render current page
        page_image = render_pdf_page(pdf_path, page_num - 1)
        
        if page_image:
            # Run detection button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Run Layout Detection", type="primary", use_container_width=True):
                    with st.spinner("Detecting layout..."):
                        if use_mineru and st.session_state.mineru_model:
                            # Use MinerU2 model
                            blocks = detect_layout_mineru(page_image, st.session_state.mineru_model)
                            method = "MinerU2.5-2509-1.2B"
                        else:
                            # Use fallback method
                            blocks = detect_layout_fallback(page_image)
                            method = "CV-based Fallback"
                        
                        st.session_state.detection_results = blocks
                        st.success(f"Detected {len(blocks)} blocks using {method}")
            
            # Display results
            if st.session_state.detection_results:
                # Create visualization
                col1, col2 = st.columns([3, 1] if show_legend else [1, 0])
                
                with col1:
                    st.subheader("Detection Results")
                    fig = visualize_layout(
                        page_image,
                        st.session_state.detection_results,
                        show_labels=show_labels
                    )
                    st.pyplot(fig)
                
                if show_legend:
                    with col2:
                        st.subheader("Legend")
                        legend_fig = create_legend()
                        st.pyplot(legend_fig)
                
                # Show block details
                with st.expander("📋 Block Details", expanded=False):
                    for idx, block in enumerate(st.session_state.detection_results):
                        block_info = CATEGORY_TYPES.get(block.type, {'label': 'Unknown'})
                        st.markdown(f"### Block {idx}: {block_info['label']}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Type:** `{block.type}`")
                        with col2:
                            st.write(f"**Confidence:** {block.confidence:.2f}")
                        with col3:
                            st.write(f"**Angle:** {block.angle if block.angle else 'None'}")
                        st.write(f"**BBox:** [{block.bbox[0]:.3f}, {block.bbox[1]:.3f}, {block.bbox[2]:.3f}, {block.bbox[3]:.3f}]")
                        if block.content:
                            st.write(f"**Content:** {block.content[:200]}{'...' if len(block.content) > 200 else ''}")
                        st.divider()
                
                # Export results
                if st.button("💾 Export Results as JSON"):
                    results = []
                    for block in st.session_state.detection_results:
                        results.append({
                            'type': block.type,
                            'bbox': block.bbox,
                            'content': block.content,
                            'confidence': block.confidence,
                            'angle': block.angle,
                            'page_idx': page_num - 1
                        })
                    
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"layout_detection_page_{page_num}.json",
                        mime="application/json"
                    )
            else:
                # Show original image
                st.subheader("Original Document")
                st.image(page_image, use_column_width=True)
                st.info("Click 'Run Layout Detection' to analyze the document layout")
    else:
        st.error("PDF file not found")
else:
    st.info("""
    ### Getting Started
    1. Upload a PDF document using the sidebar
    2. Load the MinerU2 model (optional, for better accuracy)
    3. Navigate to the desired page
    4. Click "Run Layout Detection" to analyze the layout
    
    ### Features
    - **MinerU2 Integration**: State-of-the-art Vision-Language model for accurate layout detection
    - **Multi-category Detection**: Identifies text, tables, figures, captions, formulas, headers, footers, and more
    - **Visual Feedback**: Color-coded bounding boxes for different content types
    - **Fallback Mode**: CV-based detection when MinerU2 is not available
    - **Export Support**: Save detection results as JSON for further processing
    """)