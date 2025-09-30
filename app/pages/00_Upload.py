import streamlit as st
from pathlib import Path
import sys
import hashlib
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.pdf_processor import extract_spans_from_pdf, save_pdf_copy, get_pdf_info
from core.io import save_jsonl
from app.state import AppState

st.set_page_config(page_title="Upload PDF", page_icon="📄", layout="wide")

AppState.init()

st.title("📄 Upload & Process PDF")

st.markdown("""
Upload a PDF document to extract text spans with position and style information.
The document will be processed and made available for labeling.
""")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=['pdf'],
    help="Select a PDF document to process for semantic unit labeling"
)

if uploaded_file is not None:
    # Display file info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
    with col3:
        st.metric("Type", uploaded_file.type)
    
    # Generate document ID
    file_hash = hashlib.md5(uploaded_file.read()).hexdigest()[:8]
    uploaded_file.seek(0)  # Reset file pointer
    
    doc_id = f"{Path(uploaded_file.name).stem}_{file_hash}"
    
    st.info(f"Document ID: `{doc_id}`")
    
    # Check if already processed
    data_dir = Path("data/docs") / doc_id
    if data_dir.exists() and (data_dir / "spans.jsonl").exists():
        st.warning("⚠️ This document has already been processed!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Go to Labeling", type="primary", use_container_width=True):
                st.switch_page("pages/01_Label.py")
        with col2:
            if st.button("🔄 Reprocess Document", type="secondary", use_container_width=True):
                # Clear existing data
                import shutil
                shutil.rmtree(data_dir)
                st.rerun()
    else:
        # Process button
        if st.button("🚀 Process PDF", type="primary", use_container_width=True):
            with st.spinner("Processing PDF..."):
                # Save uploaded file temporarily
                temp_path = Path(f"/tmp/{uploaded_file.name}")
                temp_path.write_bytes(uploaded_file.read())
                
                # Get PDF info
                pdf_info = get_pdf_info(temp_path)
                st.success(f"PDF has {pdf_info['pages']} pages")
                
                # Extract spans
                progress_bar = st.progress(0, text="Extracting spans...")
                spans = extract_spans_from_pdf(temp_path, doc_id)
                progress_bar.progress(50, text=f"Extracted {len(spans)} spans")
                
                # Save PDF copy
                pdf_path = save_pdf_copy(temp_path, doc_id)
                progress_bar.progress(75, text="Saving PDF copy...")
                
                # Save spans
                spans_path = Path("data/docs") / doc_id / "spans.jsonl"
                save_jsonl(spans_path, spans)
                progress_bar.progress(100, text="Complete!")
                
                # Clean up temp file
                temp_path.unlink()
                
                # Show statistics
                st.success("✅ PDF processed successfully!")
                
                # Statistics
                pages = max(s.page_number for s in spans) if spans else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Pages", pages)
                with col2:
                    st.metric("Total Spans", len(spans))
                with col3:
                    st.metric("Avg Spans/Page", f"{len(spans)/pages:.1f}" if pages > 0 else 0)
                
                # Sample of extracted spans
                st.subheader("Sample Extracted Spans")
                
                sample_spans = spans[:10] if len(spans) > 10 else spans
                for span in sample_spans:
                    with st.expander(f"{span.span_id}: {span.text[:50]}..."):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Page:** {span.page_number}")
                            st.write(f"**Font Size:** {span.font_size:.1f}")
                            st.write(f"**Bold:** {'Yes' if span.bold else 'No'}")
                            st.write(f"**Italic:** {'Yes' if span.italic else 'No'}")
                        with col2:
                            st.write(f"**Position:** ({span.x_center:.1f}, {span.y_bottom:.1f})")
                            st.write(f"**Column:** {span.column}")
                            st.write(f"**Reading Order:** {span.reading_order}")
                            st.write(f"**BBox:** {span.bbox}")
                
                # Navigation button
                if st.button("📝 Start Labeling", type="primary", use_container_width=True):
                    AppState.set_document(doc_id)
                    st.switch_page("pages/01_Label.py")

# Show existing documents
st.divider()
st.subheader("📚 Existing Documents")

data_dir = Path("data/docs")
if data_dir.exists():
    doc_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if doc_dirs:
        for doc_dir in doc_dirs:
            doc_id = doc_dir.name
            spans_file = doc_dir / "spans.jsonl"
            labels_file = doc_dir / "labels.jsonl"
            pdf_file = doc_dir / "document.pdf"
            
            if spans_file.exists():
                # Count spans and labels
                with open(spans_file) as f:
                    span_count = sum(1 for _ in f)
                
                label_count = 0
                if labels_file.exists():
                    with open(labels_file) as f:
                        label_count = sum(1 for _ in f)
                
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                
                with col1:
                    st.write(f"📁 **{doc_id}**")
                with col2:
                    st.caption(f"{span_count} spans")
                with col3:
                    st.caption(f"{label_count} labels")
                with col4:
                    if st.button("Open", key=f"open_{doc_id}"):
                        AppState.set_document(doc_id)
                        st.switch_page("pages/01_Label.py")
                with col5:
                    if st.button("Delete", key=f"del_{doc_id}"):
                        import shutil
                        shutil.rmtree(doc_dir)
                        st.rerun()
    else:
        st.info("No documents found. Upload a PDF to get started!")
else:
    st.info("Data directory not found. Upload a PDF to create it!")