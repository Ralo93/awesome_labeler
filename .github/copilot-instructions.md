# Copilot Instructions - Semantic Unit Labeler

## Project Overview
This is a **Streamlit-based GUI** for labeling semantic boundaries in PDF documents with active learning. The app helps users annotate text spans to create semantic units (like headers, paragraphs, captions) by marking boundaries between consecutive spans.

**⚠️ DEVELOPMENT STATUS**: This repository is **actively under development**. The README and some documentation may be outdated. The current focus is on **stripping down to an MVP** with emphasis on clarity, simplicity, and fast labeling workflows to efficiently process ~20 PDFs for initial model training.

## Architecture Pattern

### Multi-Page Streamlit App
- **Entry point**: `streamlit run app/main.py` 
- **Pages structure**: `app/pages/0X_Name.py` (00_Upload, 01_Label, 02_Export, etc.)
- **Centralized state**: `app/state.py` uses `st.session_state` for document, selections, labels, and undo/redo
- **Components**: Reusable UI in `app/components/` (overlay, heatmap, feature_panel, toolbar)

### Data Flow Architecture
```
PDF → Text Spans (immutable) → Labels (boundaries) → Semantic Units → Training Data → ML Model → Predictions
```

## Key Data Structures

### Core Schemas (`core/schematas.py`)
- **`Span`**: Immutable text segment from PDF with bbox, font properties, reading_order
- **`Label`**: User annotation marking boundary type ("new_unit" vs "continue") 
- **`SemanticUnit`**: Grouped spans forming logical document elements

### File Organization Pattern
```
data/docs/<DOC_ID>/
├── spans.jsonl          # Immutable extracted text spans  
├── labels.jsonl         # User boundary annotations
└── document.pdf         # Original PDF

models/
├── boundary_model.pkl   # Trained RandomForest classifier
├── boundary_model_metadata.json
└── boundary_model_*.png # Visualizations

exports/
└── train_raw_*.parquet  # Feature-engineered training data
```

## Critical Development Patterns

### State Management Pattern
- **Never directly mutate `st.session_state`** - use `AppState` class methods
- **Always call `AppState.init()`** in pages before accessing state
- **Undo/redo**: Use `AppState.add_to_undo_stack()` for reversible actions

### Feature Engineering Pipeline
- **Pairwise features**: All ML features computed between consecutive span pairs
- **Normalized coordinates**: Always normalize bbox/geometry features by page dimensions
- **Reading order**: Critical for sequence - use `add_reading_order()` if missing
- **Rule-based features**: Integration with hand-crafted rules (headers, bullets, etc.)

### Model Training Workflow
1. Export labeled data to `exports/train_raw_*.parquet` via Export page
2. Run `python train/train_model.py` (expects concatenated parquet input)
3. Model artifacts saved to `models/boundary_model.*`
4. Load model in app for inference predictions

## Development Guidelines

### MVP Focus & Priorities
- **Speed over features**: Prioritize fast, intuitive labeling experience
- **Simplicity first**: Remove complexity that doesn't directly support core labeling workflow
- **Performance critical**: Need to efficiently process ~20 PDFs for initial training dataset
- **UI clarity**: Every interaction should be obvious and minimize cognitive load

### Adding New Features
- **Geometry features**: Add to `compute_pairwise_features()` in `core/features.py`
- **UI components**: Create in `app/components/` with clear interface
- **New pages**: Follow `0X_Name.py` pattern in `app/pages/`
- **Before adding**: Ask if this feature is essential for MVP labeling workflow

### PDF Coordinate System
- **PDF coords**: Origin bottom-left, y increases upward
- **Streamlit display**: Convert to top-left origin for overlay rendering
- **Bboxes**: Always `(x0, y0, x1, y1)` format in PDF coordinates

### Model Integration
- **Calibrated probabilities**: Use `CalibratedClassifierCV` wrapper for uncertainty
- **Feature columns**: Must match training exactly - stored in `model.feature_columns`
- **Threshold tuning**: Adjust `model.threshold_` for precision/recall balance

## Key Commands

```bash
# Start application
streamlit run app/main.py

# Train new model (after exporting training data)
python train/train_model.py

# Install with poetry
poetry install && poetry run streamlit run app/main.py
```

## Common Gotchas

- **Feature column mismatch**: Ensure inference features match `model.feature_columns` order
- **Reading order gaps**: Use `reading_order_gap` feature, not absolute positions
- **State persistence**: Labels auto-save on changes, but check `st.session_state.dirty`
- **PDF rendering**: Use `core/pdf_processor.py` for consistent page rendering
- **Coordinate transforms**: Be careful with PDF vs display coordinate systems

## Debugging Patterns
- **Feature inspection**: Enable "Show Features" toggle in Label page sidebar
- **Model diagnostics**: Check `models/boundary_model_metadata.json` for training stats
- **State issues**: Use browser dev tools to inspect `st.session_state` in console

After providing any code, please include exactly which file needs a change.