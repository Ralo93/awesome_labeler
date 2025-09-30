import streamlit as st
from pathlib import Path
import sys
import subprocess

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.visualization_export import export_comparison_visualization
from app.state import AppState

st.set_page_config(page_title="Visualize", page_icon="📊", layout="wide")

AppState.init()

st.title("📊 Export Visualization")

st.markdown("""
Export a comparison PDF showing:
- **Left side**: Original spans with automatic classifications
- **Right side**: Your labeled semantic units (grouped spans)

This helps you compare your manual labeling results with the automatic classification.
""")


def _normalize_progress(x: float) -> float:
    # Convert percents like 43.8 to 0.438
    if x is None:
        return 0.0
    try:
        val = float(x)
    except (TypeError, ValueError):
        return 0.0
    # If it looks like a percent, convert to ratio
    if val > 1.0:
        val = val / 100.0
    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, val))

# Get all documents
data_dir = Path("data/docs")
doc_ids = []
labeled_docs = []

if data_dir.exists():
    for doc_dir in data_dir.iterdir():
        if doc_dir.is_dir():
            doc_id = doc_dir.name
            doc_ids.append(doc_id)
            
            # Check if document has labels
            labels_file = doc_dir / "labels.jsonl"
            if labels_file.exists() and labels_file.stat().st_size > 0:
                labeled_docs.append(doc_id)

if not doc_ids:
    st.error("No documents found. Please upload a PDF first.")
    st.stop()

if not labeled_docs:
    st.warning("No labeled documents found. Please label a document first.")
    st.stop()

st.info(f"Found {len(labeled_docs)} labeled documents out of {len(doc_ids)} total documents")

# Document selection
selected_doc = st.selectbox(
    "Select Document to Visualize",
    options=labeled_docs,
    help="Only documents with labels can be visualized"
)

if selected_doc:
    # Show document info
    from core.io import load_spans, load_labels
    
    spans = load_spans(selected_doc)
    labels = load_labels(selected_doc)
    
    # Calculate statistics
    total_pages = max(s.page_number for s in spans) if spans else 0
    total_spans = len(spans)
    labeled_spans = len(labels)
    
    # Count unique units
    unique_units = len(set(l.unit_id for l in labels)) if labels else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pages", total_pages)
    
    with col2:
        st.metric("Total Spans", total_spans)
    
    with col3:
        st.metric("Labeled Spans", labeled_spans)
    
    with col4:
        st.metric("Semantic Units", unique_units)
    
    # Coverage indicator
    if total_spans > 0:
        coverage = labeled_spans / total_spans
        coverage_ratio = _normalize_progress(coverage)
        st.progress(coverage_ratio, text=f"Coverage: {coverage_ratio:.1%} of spans labeled")
    
    st.divider()
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        output_filename = st.text_input(
            "Output Filename",
            value=f"comparison_{selected_doc}.pdf",
            help="Name for the exported PDF file"
        )
    
    with col2:
        st.info("File will be saved to `exports/` directory")
    
    # Classification preview
    with st.expander("Classification Logic Preview"):
        st.markdown("""
        The visualization will automatically classify spans based on:
        - **Title**: Large font (>16pt), short text
        - **Header**: Medium-large font (>14pt) or near top
        - **Footer**: Near bottom of page
        - **Caption**: Short text (<150 chars)
        - **Enumeration**: Starts with numbers
        - **PageNumber**: Very short numeric text
        - **Url**: Contains http/www
        - **TextItem**: Default for body text
        
        Your semantic units will be shown with:
        - Red borders showing grouped spans
        - Unit IDs and span counts
        - Dashed lines showing original span boundaries
        """)
    
    # Export button
    if st.button("🎨 Generate Comparison PDF", type="primary", use_container_width=True):
        try:                
                # Export the comparison
                result_path = export_comparison_visualization(
                    doc_id=selected_doc,
                    output_filename=output_filename
                )
                
                st.success(f"✅ Visualization exported to: {result_path}")
                
                # Offer to open the file
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"File size: {Path(result_path).stat().st_size / 1024:.1f} KB")
                
                with col2:
                    if sys.platform == "darwin":  # macOS
                        if st.button("📂 Open in Preview"):
                            subprocess.run(["open", result_path])
                    elif sys.platform == "win32":  # Windows
                        if st.button("📂 Open PDF"):
                            subprocess.run(["start", result_path], shell=True)
                    else:  # Linux
                        if st.button("📂 Open PDF"):
                            subprocess.run(["xdg-open", result_path])
                
                with col3:
                    # Download button (if running in web environment)
                    with open(result_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=f.read(),
                            file_name=output_filename,
                            mime="application/pdf"
                        )
        
        except FileNotFoundError as e:
            st.error(f"File not found: {e}")
            st.info("Make sure visualization.py is in your project directory")
        
        except ValueError as e:
            st.error(str(e))
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)

# Show existing exports
st.divider()
st.subheader("📚 Existing Visualizations")

export_dir = Path("exports")
if export_dir.exists():
    pdf_files = list(export_dir.glob("comparison_*.pdf"))
    
    if pdf_files:
        for pdf_file in sorted(pdf_files, key=lambda x: x.stat().st_mtime, reverse=True):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.text(pdf_file.name)
            
            with col2:
                st.text(f"{pdf_file.stat().st_size / 1024:.1f} KB")
            
            with col3:
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        "⬇️",
                        data=f.read(),
                        file_name=pdf_file.name,
                        mime="application/pdf",
                        key=f"download_{pdf_file.name}"
                    )
            
            with col4:
                if st.button("🗑️", key=f"delete_{pdf_file.name}"):
                    pdf_file.unlink()
                    st.rerun()
    else:
        st.info("No visualizations found yet")
else:
    st.info("Export directory not found. It will be created when you export.")

# Instructions
st.divider()
st.markdown("""
### 📖 How to Use

1. **Label your document** in the Label page (01_Label)
2. **Select the document** from the dropdown above
3. **Click Generate** to create the comparison PDF
4. **View the PDF** to see:
   - Original spans with automatic classification (left)
   - Your semantic units with grouped spans (right)

The visualization helps you:
- Verify your labeling quality
- Compare with automatic classification
- Identify potential improvements
- Document your labeling decisions
""")