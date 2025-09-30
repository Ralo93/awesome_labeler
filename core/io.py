import json
import pandas as pd
from pathlib import Path
from typing import List, Optional
from .schematas import Span, Label, Rule

# core/io.py
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional
from .schematas import Span, Label, Rule

def load_jsonl(filepath: Path, schema_class):
    """Load JSONL file and parse with schema"""
    items = []
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    items.append(schema_class.from_dict(data))
    return items

def save_jsonl(filepath: Path, items: List):
    """Save items to JSONL file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + '\n')

def load_spans(doc_id: str, data_dir: Path = Path("data")) -> List[Span]:
    """Load spans for a document; backfill reading order if missing."""
    filepath = data_dir / "docs" / doc_id / "spans.jsonl"
    spans: List[Span] = load_jsonl(filepath, Span)

    if not spans:
        return spans

    # If any span lacks reading_order, compute it once for the page groups
    if any(s.reading_order is None for s in spans):
        try:
            from .features import add_reading_order  # ensure you placed the helper here
        except Exception:
            # If the helper isn't available, just return as-is
            return spans

        # Convert to DataFrame for group-wise computation
        df = pd.DataFrame([s.to_dict() for s in spans])

        # Map your coordinate names to the helper's expected names
        # (helper expects: 'top','left','height'; we have y_top/x_left/height)
        if "top" not in df.columns and "y_top" in df.columns:
            df["top"] = df["y_top"]
        if "left" not in df.columns and "x_left" in df.columns:
            df["left"] = df["x_left"]
        # segment_id alias for span_id
        if "segment_id" not in df.columns and "span_id" in df.columns:
            df["segment_id"] = df["span_id"]
        # ensure required columns exist
        for col in ("top", "left"):
            if col not in df.columns:
                df[col] = 0.0
        if "height" not in df.columns:
            df["height"] = pd.NA
        if "text" not in df.columns:
            df["text"] = ""

        # Compute reading order within (doc_id,page)
        df_ro = add_reading_order(
            df,
            group_cols=("doc_id", "page"),
            y_col="top",
            x_col="left",
            h_col="height",
            id_col="segment_id",
            text_col="text",
        )

        # Bring results back into Span objects
        take = df_ro.set_index("span_id")[["reading_order", "line_id", "prev_id", "next_id"]].to_dict("index")
        for s in spans:
            if s.span_id in take:
                rec = take[s.span_id]
                s.reading_order = int(rec["reading_order"]) if pd.notna(rec["reading_order"]) else None
                s.line_id = int(rec["line_id"]) if pd.notna(rec["line_id"]) else None
                s.prev_id = rec["prev_id"] if pd.notna(rec["prev_id"]) else None
                s.next_id = rec["next_id"] if pd.notna(rec["next_id"]) else None

    return spans


def load_labels(doc_id: str, data_dir: Path = Path("data")) -> List[Label]:
    """Load labels for a document"""
    filepath = data_dir / "docs" / doc_id / "labels.jsonl"
    return load_jsonl(filepath, Label)

def save_labels(doc_id: str, labels: List[Label], data_dir: Path = Path("data")):
    """Save labels for a document"""
    filepath = data_dir / "docs" / doc_id / "labels.jsonl"
    save_jsonl(filepath, labels)

def load_rules(rules_dir: Path = Path("rules")) -> List[Rule]:
    """Load optional rules"""
    filepath = rules_dir / "rules.jsonl"
    return load_jsonl(filepath, Rule) if filepath.exists() else []

def load_parquet(filepath: Path) -> pd.DataFrame:
    """Load parquet file"""
    return pd.read_parquet(filepath) if filepath.exists() else pd.DataFrame()

def save_parquet(df: pd.DataFrame, filepath: Path, metadata: dict = None):
    """Save dataframe to parquet with optional metadata"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, engine='pyarrow', compression='snappy', index=False)