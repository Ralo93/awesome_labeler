# Semantic Unit Labeler

A Streamlit-based GUI for labeling semantic units in documents with active learning support.

## Features

- **Document Display**: Side-by-side view of original PDF and overlay annotations
- **Interactive Labeling**: Click/drag to select spans, keyboard shortcuts for actions
- **Boundary Detection**: Toggle boundaries between semantic units
- **Heatmap Visualization**: Probability-based visual feedback for boundaries
- **Feature Transparency**: View normalized features and rule-based classes
- **Active Learning**: Train models and run inference for assisted labeling
- **Export Pipeline**: Automatic deduplication and class balancing options

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with poetry
poetry install
```

## Usage

1. **Start the application**:
```bash
streamlit run app/main.py
```

2. **Load documents**: Place documents in `data/docs/<DOC_ID>/` with:
   - `spans.jsonl`: Document spans
   - `labels.jsonl`: Labels (created by the app)

3. **Label boundaries**:
   - Select spans by clicking
   - Press B to toggle boundary
   - Use confidence slider if enabled

4. **Train model**:
   - Export training data (RAW or +RULES)
   - Click "Train Model" in sidebar
   - Model will be saved to `models/`

5. **Run inference**:
   - Load a trained model
   - Click inference buttons for predictions
   - Correct predictions by clicking

## Keyboard Shortcuts

- **B**: Toggle boundary
- **M**: Merge units
- **S**: Split unit
- **Ctrl+Z**: Undo
- **Ctrl+Y**: Redo
- **←/→**: Navigate pages
- **Enter**: Next uncertain boundary

## Project Structure

```
project/
├── app/              # Streamlit application
├── core/             # Business logic
├── data/             # Document data
├── models/           # Trained models
├── exports/          # Training exports
└── train/            # Training scripts
```

## Data Formats

### Spans (immutable)
```json
{
  "doc_id": "D1",
  "page": 3,
  "span_id": "p3_s27",
  "bbox": [x0, y0, x1, y1],
  "text": "...",
  "font_size": 10.5,
  "bold": false,
  "italic": false,
  "line_height": 12.8,
  "x_center": 241.3,
  "y_bottom": 682.1,
  "column": 0,
  "reading_order": 27
}
```

### Labels
```json
{
  "doc_id": "D1",
  "page": 3,
  "span_id": "p3_s27",
  "unit_id": "SU_12",
  "boundary": "continue",
  "confidence": 4,
  "ts": "2025-09-28T12:34:56Z"
}
```

## License

MIT


I tried mineru2, docling-layout-heron-101, docling-models, docling-layout