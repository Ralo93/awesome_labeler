import streamlit as st
from pathlib import Path
import sys
import hashlib
from datetime import datetime
import shutil

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.pdf_processor import extract_spans_from_pdf, save_pdf_copy, get_pdf_info
from core.io import save_jsonl, load_labels
from app.state import AppState

st.set_page_config(page_title="Upload PDF", page_icon="📄", layout="wide")

AppState.init()

st.title("📄 Upload & Process PDF")

st.markdown("""
Upload a PDF document to extract text spans with position and style information.
The document will be processed and made available for labeling.
""")

def reprocess_document(doc_id: str) -> bool:
    """
    Reprocess a document's spans while preserving labels.
    Used when feature extraction is updated.
    """
    try:
        doc_dir = Path("data/docs") / doc_id
        pdf_path = doc_dir / "document.pdf"
        
        if not pdf_path.exists():
            st.error(f"PDF not found for {doc_id}")
            return False
        
        # Load existing labels (to preserve them)
        labels = load_labels(doc_id)
        label_count = len(labels)
        
        # Backup old spans.jsonl just in case
        old_spans_path = doc_dir / "spans.jsonl"
        backup_path = doc_dir / "spans_backup.jsonl"
        if old_spans_path.exists():
            shutil.copy2(old_spans_path, backup_path)
        
        # Re-extract spans with new features
        with st.spinner(f"Re-extracting spans for {doc_id}..."):
            spans = extract_spans_from_pdf(pdf_path, doc_id)
            
            if not spans:
                st.error("Failed to extract spans")
                return False
            
            # Save new spans
            spans_path = doc_dir / "spans.jsonl"
            save_jsonl(spans_path, spans)
            
            # Analyze column detection
            pages_with_columns = {}
            for span in spans:
                if span.page_number not in pages_with_columns:
                    pages_with_columns[span.page_number] = set()
                pages_with_columns[span.page_number].add(span.column)
            
            multi_column_pages = sum(1 for cols in pages_with_columns.values() if len(cols) > 1)
            
            st.success(f"""
            ✅ Reprocessed successfully!
            - **Spans extracted:** {len(spans)}
            - **Labels preserved:** {label_count}
            - **Multi-column pages detected:** {multi_column_pages}/{len(pages_with_columns)}
            """)
            
            # Show column statistics
            if multi_column_pages > 0:
                st.info(f"📊 Column detection found {multi_column_pages} pages with multiple columns")
            
            return True
            
    except Exception as e:
        st.error(f"Error reprocessing document: {e}")
        return False

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
                
                # Statistics including column detection
                pages = max(s.page_number for s in spans) if spans else 0
                
                # Check for multi-column pages
                pages_with_columns = {}
                for span in spans:
                    if span.page_number not in pages_with_columns:
                        pages_with_columns[span.page_number] = set()
                    pages_with_columns[span.page_number].add(span.column)
                
                multi_column_pages = sum(1 for cols in pages_with_columns.values() if len(cols) > 1)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pages", pages)
                with col2:
                    st.metric("Total Spans", len(spans))
                with col3:
                    st.metric("Avg Spans/Page", f"{len(spans)/pages:.1f}" if pages > 0 else 0)
                with col4:
                    st.metric("Multi-Column Pages", multi_column_pages)
                
                # Sample of extracted spans
                st.subheader("Sample Extracted Spans")
                
                sample_spans = spans[:10] if len(spans) > 10 else spans
                for span in sample_spans:
                    with st.expander(f"{span.span_id}: {span.text[:50]}..."):
                        col1, col2, col3 = st.columns(3)
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
                        with col3:
                            # Show new column features
                            st.write(f"**First in Column:** {getattr(span, 'is_first_in_column', False)}")
                            st.write(f"**Last in Column:** {getattr(span, 'is_last_in_column', False)}")
                            st.write(f"**Column Changed:** {getattr(span, 'column_changed', False)}")
                            st.write(f"**Columns on Page:** {getattr(span, 'num_columns_on_page', 1)}")
                
                # Navigation button
                if st.button("📝 Start Labeling", type="primary", use_container_width=True):
                    AppState.set_document(doc_id)
                    st.switch_page("pages/01_Label.py")

# Show existing documents with reprocess option
st.divider()
st.subheader("📚 Existing Documents")

# Add batch reprocess option
if st.button("🔄 Reprocess All Documents", help="Re-extract spans for all documents with updated features"):
    data_dir = Path("data/docs")
    if data_dir.exists():
        doc_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        success_count = 0
        for doc_dir in doc_dirs:
            if reprocess_document(doc_dir.name):
                success_count += 1
        st.success(f"Reprocessed {success_count}/{len(doc_dirs)} documents")
        st.rerun()

data_dir = Path("data/docs")
if data_dir.exists():
    doc_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if doc_dirs:
        for doc_dir in sorted(doc_dirs, key=lambda d: d.stat().st_mtime, reverse=True):
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
                
                # Check if this has old-style spans (needs reprocessing)
                needs_reprocess = False
                try:
                    import json
                    with open(spans_file) as f:
                        first_line = f.readline()
                        if first_line:
                            first_span = json.loads(first_line)
                            # Check for new column features
                            needs_reprocess = 'is_first_in_column' not in first_span
                except:
                    pass
                
                col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])
                
                with col1:
                    status_icon = "🔴" if needs_reprocess else "🟢"
                    st.write(f"{status_icon} **{doc_id}**")
                    if needs_reprocess:
                        st.caption("⚠️ Needs reprocessing for column features")
                with col2:
                    st.caption(f"{span_count} spans")
                with col3:
                    st.caption(f"{label_count} labels")
                with col4:
                    if st.button("Open", key=f"open_{doc_id}"):
                        AppState.set_document(doc_id)
                        st.switch_page("pages/01_Label.py")
                with col5:
                    if st.button("🔄", key=f"reprocess_{doc_id}", 
                                help="Reprocess to extract new features"):
                        if reprocess_document(doc_id):
                            st.rerun()
                with col6:
                    if st.button("🗑️", key=f"del_{doc_id}"):
                        import shutil
                        shutil.rmtree(doc_dir)
                        st.rerun()
    else:
        st.info("No documents found. Upload a PDF to get started!")
else:
    st.info("Data directory not found. Upload a PDF to create it!")