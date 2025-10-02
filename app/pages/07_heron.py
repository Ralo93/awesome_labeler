import io
import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2
import json
import torch
from dataclasses import dataclass, field
import time
from functools import wraps
from datetime import datetime

from core.helpers import measure_processing_time

# Page configuration
st.set_page_config(page_title="Document Layout Detection with docling-layout", page_icon="🔬", layout="wide")

# Initialize session state
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'layout_model' not in st.session_state:
    st.session_state.layout_model = None
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Category types for layout detection
CATEGORY_TYPES = {
    'title': {'color': '#FF6B6B', 'label': 'Title'},
    'text': {'color': '#4ECDC4', 'label': 'Text'},
    'plain_text': {'color': '#4ECDC4', 'label': 'Plain Text'},
    'paragraph': {'color': '#4ECDC4', 'label': 'Paragraph'},
    'figure': {'color': '#95E77E', 'label': 'Figure'},
    'figure_caption': {'color': '#7FD157', 'label': 'Figure Caption'},
    'table': {'color': '#FFE66D', 'label': 'Table'},
    'table_caption': {'color': '#F4D03F', 'label': 'Table Caption'},
    'table_footnote': {'color': '#E8C547', 'label': 'Table Footnote'},
    'formula': {'color': '#DDA0DD', 'label': 'Formula'},
    'equation': {'color': '#DDA0DD', 'label': 'Equation'},
    'isolate_formula': {'color': '#BA55D3', 'label': 'Block Formula'},
    'formula_caption': {'color': '#9370DB', 'label': 'Formula Caption'},
    'header': {'color': '#FFA07A', 'label': 'Header'},
    'footer': {'color': '#FFB6C1', 'label': 'Footer'},
    'page_number': {'color': '#D3D3D3', 'label': 'Page Number'},
    'list': {'color': '#87CEEB', 'label': 'List'},
    'list_item': {'color': '#87CEEB', 'label': 'List Item'},
    'enumeration': {'color': '#6495ED', 'label': 'Enumeration'},
    'reference': {'color': '#FF69B4', 'label': 'Reference'},
    'section_header': {'color': '#FFA07A', 'label': 'Section Header'},
    'code': {'color': '#98FB98', 'label': 'Code Block'},
    'abandon': {'color': '#808080', 'label': 'Abandoned'},
}

@dataclass
class ContentBlock:
    """Represents a detected content block"""
    type: str
    bbox: List[float]  # [xmin, ymin, xmax, ymax] 
    content: Optional[str] = None
    page_idx: int = 0
    confidence: float = 1.0
    angle: Optional[int] = None
    metadata: Optional[Dict] = field(default_factory=dict)

@st.cache_resource
def load_huggingpanda_model():
    """Load HuggingPanda/docling-layout model"""
    try:
        from transformers import AutoModel, AutoProcessor, AutoModelForObjectDetection
        
        with st.spinner("Loading HuggingPanda/docling-layout model..."):
            model_name = "HuggingPanda/docling-layout"
            
            try:
                # Try loading as object detection model first
                model = AutoModelForObjectDetection.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            except:
                # Fallback to general AutoModel
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.session_state["docling_device"] = device
            model = model.to(device)
            model.eval()
            
            # Create wrapper for consistent interface
            class HuggingPandaWrapper:
                def __init__(self, model, processor):
                    self.model = model
                    self.processor = processor
                    self.device = device
                    self.label_map = {
                        0: 'title', 1: 'text', 2: 'figure', 3: 'table',
                        4: 'list', 5: 'equation', 6: 'section_header',
                        7: 'footer', 8: 'header', 9: 'reference',
                        10: 'figure_caption', 11: 'table_caption',
                        12: 'code', 13: 'paragraph', 14: 'formula',
                        15: 'list_item', 16: 'page_number'
                    }
                
                @measure_processing_time(model_name="HuggingPanda/docling-layout")
                def detect(self, image: Image.Image, page_num: int = 1) -> List[ContentBlock]:
                    # Preprocess image
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Process outputs
                    blocks = []
                    
                    # Handle object detection outputs
                    if hasattr(outputs, 'logits'):
                        # Get predictions
                        logits = outputs.logits
                        boxes = outputs.pred_boxes if hasattr(outputs, 'pred_boxes') else None
                        
                        if boxes is not None:
                            # Post-process predictions
                            target_sizes = torch.tensor([image.size[::-1]])
                            results = self.processor.post_process_object_detection(
                                outputs, 
                                target_sizes=target_sizes,
                                threshold=0.5
                            )[0]
                            
                            for score, label, box in zip(
                                results["scores"], 
                                results["labels"], 
                                results["boxes"]
                            ):
                                box = box.cpu().numpy()
                                block_type = self.label_map.get(label.item(), 'text')
                                
                                blocks.append(ContentBlock(
                                    type=block_type,
                                    bbox=box.tolist(),
                                    confidence=score.item(),
                                    page_idx=page_num - 1,
                                    content=f"{block_type.replace('_', ' ').title()} Block"
                                ))
                    
                    return blocks
            
            return HuggingPandaWrapper(model, processor)
            
    except ImportError as e:
        st.error(f"""
        Required libraries not installed. Please install:
        ```bash
        pip install transformers>=4.40.0
        pip install torch torchvision
        pip install accelerate
        ```
        Error: {str(e)}
        """)
        return None
    except Exception as e:
        st.error(f"Failed to load HuggingPanda/docling-layout model: {str(e)}")
        return None

def render_pdf_page(pdf_path: Path, page_num: int) -> Optional[Image.Image]:
    """
    Render a PDF page as an image
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-based index)
        
    Returns:
        PIL Image or None if error
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        
        # Validate page number
        if page_num < 0 or page_num >= len(doc):
            st.error(f"Page {page_num} out of range. PDF has {len(doc)} pages.")
            doc.close()
            return None
        
        # Use array indexing (0-based)
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        doc.close()
        return img
    except ImportError:
        st.error("PyMuPDF not installed. Please install: pip install pymupdf")
        return None
    except Exception as e:
        st.error(f"Error rendering page: {str(e)}")
        return None

st.warning(f"Using device: {st.session_state.get('docling_device', 'unknown')}")

@measure_processing_time(model_name="CV-Fallback")
def detect_layout_fallback(image: Image.Image, page_num: int = 1) -> List[ContentBlock]:
    """Fallback layout detection using simple CV methods"""
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
        
        # Store actual pixel coordinates
        bbox = [x, y, x+w, y+h]
        
        # Simple heuristic classification
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
            confidence=0.8,
            page_idx=page_num - 1
        ))
    
    return blocks

def visualize_layout(image: Image.Image, blocks: List[ContentBlock], show_labels: bool = True) -> plt.Figure:
    """Visualize detected layout blocks with bounding boxes"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    
    # Display image
    ax.imshow(image)
    ax.axis('off')
    
    # Draw bounding boxes
    for idx, block in enumerate(blocks):
        # Get block type info
        block_info = CATEGORY_TYPES.get(block.type, {'color': '#808080', 'label': 'Unknown'})
        color = block_info['color']
        label = block_info['label']
        
        # Get box coordinates
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
    
    # Model selection
    use_huggingpanda = st.checkbox(
        "Use HuggingPanda/docling-layout",
        value=True,
        help="Use HuggingPanda docling-layout model for advanced detection"
    )
    
    # Model status display
    st.markdown("### 🤖 Active Detection Method")
    
    if use_huggingpanda:
        if st.session_state.layout_model is not None:
            st.success("✅ **HuggingPanda Model Ready**")
            st.caption("Advanced neural network detection")
            st.session_state.model_loaded = True
        else:
            st.warning("⚠️ **Model Not Loaded**")
            st.caption("Click 'Load Model' to enable")
            st.session_state.model_loaded = False
            
            if st.button("🔧 Load Model", type="primary", use_container_width=True):
                with st.spinner("Loading model..."):
                    st.session_state.layout_model = load_huggingpanda_model()
                    if st.session_state.layout_model:
                        st.success("Model loaded successfully!")
                        st.session_state.model_loaded = True
                        st.rerun()
    else:
        st.info("🔄 **CV-Based Fallback Active**")
        st.caption("Simple computer vision detection")
        st.session_state.model_loaded = False
    
    st.divider()
    
    # Detection preview
    st.markdown("### 📋 Detection Preview")
    if use_huggingpanda and st.session_state.model_loaded:
        st.info("**Will use:** HuggingPanda/docling-layout")
    else:
        st.info("**Will use:** CV-based Fallback")
    
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
    
    # Processing metrics
    if st.session_state.processing_times:
        st.header("⏱️ Performance Metrics")
        
        latest = st.session_state.processing_times[-1]
        st.metric("Last Processing Time", f"{latest.processing_time:.3f}s")
        st.metric("Blocks Detected", latest.num_blocks_detected)
        
        if len(st.session_state.processing_times) > 1:
            avg_time = np.mean([m.processing_time for m in st.session_state.processing_times])
            st.metric("Avg Processing Time", f"{avg_time:.3f}s")
    
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
st.title("🔬 Document Layout Detection")
st.markdown("Advanced document layout analysis using HuggingPanda/docling-layout model")

# Model status banner
if use_huggingpanda:
    if st.session_state.model_loaded:
        st.success("🟢 **HuggingPanda/docling-layout model is active and ready**")
    else:
        st.warning("🟡 **HuggingPanda model selected but not loaded** - Will use CV fallback until model is loaded")
else:
    st.info("🔵 **CV-based fallback detection is active**")

if st.session_state.current_doc:
    pdf_path = Path(st.session_state.current_doc)
    
    if pdf_path.exists():
        # Get number of pages
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
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
        
        # Render current page (convert 1-based to 0-based for render function)
        page_image = render_pdf_page(pdf_path, page_num - 1)
        
        if page_image:
            # Detection method indicator
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Show what will happen when button is clicked
                if use_huggingpanda and st.session_state.model_loaded:
                    detection_method = "HuggingPanda/docling-layout"
                    button_color = "primary"
                    button_text = "🚀 Run HuggingPanda Detection"
                else:
                    detection_method = "CV-based Fallback"
                    button_color = "secondary"
                    button_text = "🔄 Run CV Fallback Detection"
                
                st.info(f"**Ready to detect using:** {detection_method}")
                
                if st.button(button_text, type=button_color, use_container_width=True):
                    with st.spinner(f"Detecting layout using {detection_method}..."):
                        if use_huggingpanda and st.session_state.model_loaded:
                            # Use HuggingPanda model
                            blocks = st.session_state.layout_model.detect(page_image, page_num)
                        else:
                            # Use fallback method
                            blocks = detect_layout_fallback(page_image, page_num)
                        
                        st.session_state.detection_results = blocks
                        st.success(f"✅ Detected {len(blocks)} blocks using {detection_method}")
            
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
                        st.write(f"**BBox:** [{block.bbox[0]:.1f}, {block.bbox[1]:.1f}, {block.bbox[2]:.1f}, {block.bbox[3]:.1f}]")
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
                            'metadata': block.metadata,
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
                st.image(page_image, use_container_width=True)
                st.info("Select detection method and click the button above to analyze")
    else:
        st.error("PDF file not found")
else:
    st.info("""
    ### Getting Started
    1. Upload a PDF document using the sidebar
    2. Choose detection method (HuggingPanda or CV fallback)
    3. Load the model if using HuggingPanda
    4. Navigate to the desired page
    5. Click the detection button to analyze the layout
    
    ### Detection Methods
    - **HuggingPanda/docling-layout**: Advanced neural network model for accurate detection
    - **CV-based Fallback**: Simple computer vision method (always available, no model loading required)
    
    ### Features
    - **Multi-category Detection**: Identifies text, tables, figures, equations, code blocks, and more
    - **Performance Tracking**: Built-in timing measurements for optimization
    - **Visual Feedback**: Color-coded bounding boxes for different content types
    - **Export Support**: Save detection results as JSON for further processing
    """)