import io
import streamlit as st
import sys
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2
import json
import torch
from dataclasses import dataclass, field

# Page configuration
st.set_page_config(page_title="Docling Heron Layout Detection", page_icon="🔬", layout="wide")

# Initialize session state
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'docling_model' not in st.session_state:
    st.session_state.docling_model = None
if 'docling_processor' not in st.session_state:
    st.session_state.docling_processor = None

# Category types with ID mappings for Docling/LayoutLMv3 style models
CATEGORY_ID_MAP = {
    0: 'text',
    1: 'title',
    2: 'list',
    3: 'table',
    4: 'figure',
    5: 'figure_caption',
    6: 'table_caption',
    7: 'section_header',
    8: 'footer',
    9: 'header',
    10: 'reference',
    11: 'equation',
    12: 'abstract',
    13: 'code',
    14: 'paragraph',
    15: 'page_number',
}

CATEGORY_TYPES = {
    'title': {'color': '#FF6B6B', 'label': 'Title'},
    'text': {'color': '#4ECDC4', 'label': 'Text'},
    'paragraph': {'color': '#4ECDC4', 'label': 'Paragraph'},
    'abstract': {'color': '#45B7D1', 'label': 'Abstract'},
    'figure': {'color': '#95E77E', 'label': 'Figure'},
    'figure_caption': {'color': '#7FD157', 'label': 'Figure Caption'},
    'table': {'color': '#FFE66D', 'label': 'Table'},
    'table_caption': {'color': '#F4D03F', 'label': 'Table Caption'},
    'equation': {'color': '#DDA0DD', 'label': 'Equation'},
    'formula': {'color': '#DDA0DD', 'label': 'Formula'},
    'header': {'color': '#FFA07A', 'label': 'Header'},
    'footer': {'color': '#FFB6C1', 'label': 'Footer'},
    'page_number': {'color': '#D3D3D3', 'label': 'Page Number'},
    'list': {'color': '#87CEEB', 'label': 'List'},
    'reference': {'color': '#FF69B4', 'label': 'Reference'},
    'section_header': {'color': '#FFA07A', 'label': 'Section Header'},
    'code': {'color': '#98FB98', 'label': 'Code Block'},
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
def load_docling_model():
    """Load a document layout detection model"""
    try:
        # Try multiple approaches to get a working model
        
        # Approach 1: Try LayoutLMv3 which is proven for layout detection
        try:
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
            from transformers import AutoModelForObjectDetection, AutoProcessor
            
            with st.spinner("Loading LayoutLMv3 for document layout detection..."):
                # Use Microsoft's LayoutLMv3 for layout detection
                model = AutoModelForObjectDetection.from_pretrained(
                    "microsoft/layoutlmv3-base",
                    trust_remote_code=True
                )
                processor = AutoProcessor.from_pretrained(
                    "microsoft/layoutlmv3-base",
                    trust_remote_code=True,
                    apply_ocr=False
                )
                return model, processor, "layoutlmv3"
        except:
            pass
        
        # Approach 2: Try DETR-based layout model
        try:
            from transformers import DetrForObjectDetection, DetrImageProcessor
            
            with st.spinner("Loading DETR-based layout detection model..."):
                # Use a DETR model fine-tuned for document layout
                model = DetrForObjectDetection.from_pretrained(
                    "facebook/detr-resnet-50",
                    num_labels=len(CATEGORY_ID_MAP),
                    ignore_mismatched_sizes=True
                )
                processor = DetrImageProcessor.from_pretrained(
                    "facebook/detr-resnet-50"
                )
                return model, processor, "detr"
        except:
            pass
        
        # Approach 3: Try YOLOs for document layout
        try:
            from ultralytics import YOLO
            
            with st.spinner("Loading YOLO-based layout detection..."):
                # Use a YOLO model (you might need a document-specific one)
                model = YOLO('yolov8n.pt')  # You can use a custom trained model here
                return model, None, "yolo"
        except:
            pass
        
        # Approach 4: Fallback to Detectron2
        try:
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            
            with st.spinner("Loading Detectron2 layout model..."):
                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CATEGORY_ID_MAP)
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
                cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                predictor = DefaultPredictor(cfg)
                return predictor, None, "detectron2"
        except:
            pass
        
        st.error("""
        No suitable layout detection model could be loaded. Please install one of:
        ```bash
        # Option 1: LayoutLMv3
        pip install transformers>=4.40.0 torch torchvision
        
        # Option 2: YOLO
        pip install ultralytics
        
        # Option 3: Detectron2
        pip install detectron2
        ```
        """)
        return None, None, None
        
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None, None

def render_pdf_page(pdf_path: Path, page_num: int) -> Image.Image:
    """Render a PDF page as an image"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        
        # Validate page number
        if doc.page_count == 0:
            st.error("PDF file appears to be empty")
            doc.close()
            return None
            
        if page_num < 0 or page_num >= doc.page_count:
            st.error(f"Page {page_num + 1} does not exist. Document has {doc.page_count} pages.")
            doc.close()
            return None
        
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
    except Exception as e:
        st.error(f"Error rendering PDF page: {str(e)}")
        return None

def detect_layout_model(image: Image.Image, model, processor, model_type: str) -> List[ContentBlock]:
    """Perform layout detection using loaded model"""
    if model is None:
        return []
    
    blocks = []
    img_w, img_h = image.size
    
    try:
        if model_type == "layoutlmv3":
            # Process image
            inputs = processor(images=image, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Post-process predictions
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = box.tolist()
                label_id = label.item()
                block_type = CATEGORY_ID_MAP.get(label_id, 'text')
                
                blocks.append(ContentBlock(
                    type=block_type,
                    bbox=[box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h],
                    confidence=score.item(),
                    content=f"{block_type.replace('_', ' ').title()} Region"
                ))
        
        elif model_type == "detr":
            # Process image
            inputs = processor(images=image, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = box.tolist()
                label_id = label.item()
                block_type = CATEGORY_ID_MAP.get(label_id, 'text')
                
                blocks.append(ContentBlock(
                    type=block_type,
                    bbox=[box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h],
                    confidence=score.item(),
                    content=f"{block_type.replace('_', ' ').title()} Region"
                ))
        
        elif model_type == "yolo":
            # Run YOLO inference
            results = model(image)
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        
                        block_type = CATEGORY_ID_MAP.get(cls, 'text')
                        
                        blocks.append(ContentBlock(
                            type=block_type,
                            bbox=[xyxy[0]/img_w, xyxy[1]/img_h, xyxy[2]/img_w, xyxy[3]/img_h],
                            confidence=conf,
                            content=f"{block_type.replace('_', ' ').title()} Region"
                        ))
        
        elif model_type == "detectron2":
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Run inference
            outputs = model(img_array)
            instances = outputs["instances"].to("cpu")
            
            for i in range(len(instances)):
                box = instances.pred_boxes[i].tensor.numpy()[0]
                score = instances.scores[i].item()
                label = instances.pred_classes[i].item()
                
                block_type = CATEGORY_ID_MAP.get(label, 'text')
                
                blocks.append(ContentBlock(
                    type=block_type,
                    bbox=[box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h],
                    confidence=score,
                    content=f"{block_type.replace('_', ' ').title()} Region"
                ))
        
        # If no blocks detected, use fallback
        if len(blocks) == 0:
            st.warning("Model didn't detect any blocks. Using fallback detection...")
            return detect_layout_fallback(image)
        
        return blocks
        
    except Exception as e:
        st.warning(f"Model inference failed: {str(e)}. Using fallback detection...")
        return detect_layout_fallback(image)

def detect_layout_fallback(image: Image.Image) -> List[ContentBlock]:
    """Enhanced fallback layout detection using CV methods"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply adaptive threshold for better results
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to merge text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blocks = []
    img_h, img_w = gray.shape[:2]
    
    # Sort contours by position (top to bottom, left to right)
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter small contours
        if w < 30 or h < 15:
            continue
        
        # Calculate features for classification
        aspect_ratio = w / h
        area_ratio = (w * h) / (img_w * img_h)
        position_y = y / img_h
        position_x = x / img_w
        
        # Enhanced heuristic classification
        if position_y < 0.1 and w > img_w * 0.3:
            block_type = 'header'
        elif position_y > 0.9:
            block_type = 'footer'
        elif aspect_ratio > 3 and position_y < 0.2:
            block_type = 'title'
        elif aspect_ratio > 5 and w > img_w * 0.6:
            block_type = 'section_header'
        elif area_ratio > 0.3:
            block_type = 'figure'  # Large blocks might be figures
        elif aspect_ratio > 1.5 and aspect_ratio < 4 and area_ratio > 0.05:
            block_type = 'table'  # Medium-sized rectangular blocks
        elif w < img_w * 0.1 and position_x < 0.2:
            block_type = 'list'  # Small blocks on the left
        elif h > img_h * 0.05 and w > img_w * 0.4:
            block_type = 'paragraph'
        else:
            block_type = 'text'
        
        # Normalize bbox
        bbox = [x/img_w, y/img_h, (x+w)/img_w, (y+h)/img_h]
        
        blocks.append(ContentBlock(
            type=block_type,
            bbox=bbox,
            content=f"{block_type.replace('_', ' ').title()} Block",
            confidence=0.7  # Lower confidence for fallback
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
        if all(0 <= v <= 1.1 for v in block.bbox):  # Allow slight overflow
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
            
            # Add confidence to label if not 1.0
            if block.confidence < 0.99:
                label_text = f"{idx}: {label} ({block.confidence:.2f})"
            else:
                label_text = f"{idx}: {label}"
            
            # Add background for better readability
            ax.text(
                x1, label_y,
                label_text,
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
        "Upload PDF or Image",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        help="Select a document to analyze"
    )
    
    if uploaded_file:
        # Create temp directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "docling_uploads"
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.current_doc = temp_path
        st.success(f"Loaded: {uploaded_file.name}")
    
    st.divider()
    
    # Model settings
    st.header("⚙️ Model Settings")
    
    use_model = st.checkbox(
        "Use ML Model",
        value=True,
        help="Use machine learning model for layout detection"
    )
    
    if use_model:
        if st.button("🔧 Load Detection Model", type="primary", use_container_width=True):
            with st.spinner("Loading model..."):
                model, processor, model_type = load_docling_model()
                st.session_state.docling_model = model
                st.session_state.docling_processor = processor
                st.session_state.model_type = model_type
                if model:
                    st.success(f"Model loaded successfully! Type: {model_type}")
    else:
        st.info("Using CV-based detection only")
    
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
    
    show_confidence = st.checkbox(
        "Show Confidence Scores",
        value=True,
        help="Display confidence scores in labels"
    )
    
    st.divider()
    
    # Detection stats
    if st.session_state.detection_results:
        st.header("📊 Detection Stats")
        
        # Count blocks by type
        type_counts = {}
        avg_confidence = {}
        for block in st.session_state.detection_results:
            type_counts[block.type] = type_counts.get(block.type, 0) + 1
            if block.type not in avg_confidence:
                avg_confidence[block.type] = []
            avg_confidence[block.type].append(block.confidence)
        
        for block_type, count in sorted(type_counts.items()):
            info = CATEGORY_TYPES.get(block_type, {'label': block_type})
            avg_conf = sum(avg_confidence[block_type]) / len(avg_confidence[block_type])
            st.metric(info['label'], count, f"Conf: {avg_conf:.2f}")

# Main content
st.title("🔬 Document Layout Detection")
st.markdown("Advanced document layout analysis using state-of-the-art models")

if st.session_state.current_doc:
    doc_path = Path(st.session_state.current_doc)
    
    if doc_path.exists():
        # Check if it's a PDF or image
        is_pdf = doc_path.suffix.lower() == '.pdf'
        
        if is_pdf:
            # Get number of pages
            try:
                import fitz
                doc = fitz.open(doc_path)
                num_pages = doc.page_count
                doc.close()
                
                # Check if PDF is empty
                if num_pages == 0:
                    st.error("The uploaded PDF file is empty or corrupted.")
                    st.stop()
            except Exception as e:
                st.error(f"Error reading PDF file: {str(e)}")
                st.stop()
            
            # Page navigation
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                # Ensure current page is within valid range
                if st.session_state.current_page > num_pages:
                    st.session_state.current_page = 1
                
                page_num = st.slider(
                    "Select Page",
                    min_value=1,
                    max_value=num_pages,
                    value=st.session_state.current_page,
                    key="page_slider"
                )
                st.session_state.current_page = page_num
            
            # Render current page
            page_image = render_pdf_page(doc_path, page_num - 1)
        else:
            # Load image directly
            page_image = Image.open(doc_path)
            page_num = 1
        
        if page_image:
            # Run detection button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Run Layout Detection", type="primary", use_container_width=True):
                    with st.spinner("Detecting layout..."):
                        if use_model and st.session_state.docling_model:
                            # Use ML model
                            blocks = detect_layout_model(
                                page_image, 
                                st.session_state.docling_model, 
                                st.session_state.docling_processor,
                                st.session_state.get('model_type', 'unknown')
                            )
                            method = st.session_state.get('model_type', 'ML Model')
                        else:
                            # Use fallback method
                            blocks = detect_layout_fallback(page_image)
                            method = "CV-based Detection"
                        
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
                        if block.metadata:
                            st.write(f"**Metadata:** {block.metadata}")
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
                st.info("Click 'Run Layout Detection' to analyze the document layout")
    else:
        st.error("Document file not found")
else:
    st.info("""
    ### Getting Started
    1. Upload a PDF or image document using the sidebar
    2. Load the detection model (optional, for better accuracy)
    3. Navigate to the desired page (for PDFs)
    4. Click "Run Layout Detection" to analyze the layout
    
    ### Features
    - **Multiple Model Support**: Supports various layout detection models including LayoutLMv3, DETR, YOLO, and Detectron2
    - **Enhanced CV Fallback**: Improved computer vision-based detection when ML models aren't available
    - **Multi-category Detection**: Identifies text, tables, figures, equations, code blocks, headers, and more
    - **Confidence Scores**: Shows detection confidence for each block
    - **Visual Feedback**: Color-coded bounding boxes for different content types
    - **Export Support**: Save detection results as JSON for further processing
    
    ### Supported Models
    - **LayoutLMv3**: Microsoft's layout-aware language model
    - **DETR**: Facebook's Detection Transformer
    - **YOLO**: Real-time object detection
    - **Detectron2**: Facebook's detection platform
    - **CV Fallback**: Always available OpenCV-based detection
    """)