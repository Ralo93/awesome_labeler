import pandas as pd
import numpy as np
from typing import Tuple, Optional

# ---- Helpers ---------------------------------------------------------------

NUM_DEFAULTS = {
    "font_size_diff_norm": 0.0,
    "y_gap_norm": 0.0,
    "x_offset_norm": 0.0,
    "vertical_overlap": 0.0,
}
FLAG_DEFAULTS = {
    "same_bold": 0,
    "same_italic": 0,
    "linebreak_flag": 0,
    "hyphenation_flag": 0,
    "same_column": 1,  # assume same column unless proven otherwise
}

SORT_DEFAULTS = {
    "reading_order_prev": -1,
    "reading_order_curr": -1,
}

def _num(row: pd.Series, col: str, nd=2, default: float = 0.0):
    val = row.get(col, default)
    try:
        val = float(val) if pd.notna(val) else default
    except Exception:
        val = default
    return round(val, nd)

def _flag(row: pd.Series, col: str, default: int = 0):
    val = row.get(col, default)
    if pd.isna(val):
        return int(default)
    if isinstance(val, (bool, np.bool_)):
        return int(val)
    try:
        return int(val)
    except Exception:
        return int(default)

# ---- Hash key --------------------------------------------------------------

def compute_hash_key(row: pd.Series) -> Tuple:
    """Compute interpretable hash key for deduplication (defensive)."""
    return (
        _num(row, 'font_size_diff_norm', nd=1, default=NUM_DEFAULTS['font_size_diff_norm']),
        _num(row, 'y_gap_norm', nd=2, default=NUM_DEFAULTS['y_gap_norm']),
        _num(row, 'x_offset_norm', nd=2, default=NUM_DEFAULTS['x_offset_norm']),
        int((_num(row, 'vertical_overlap', nd=3, default=NUM_DEFAULTS['vertical_overlap'])) > 0.5),
        _flag(row, 'same_bold', FLAG_DEFAULTS['same_bold']),
        _flag(row, 'same_italic', FLAG_DEFAULTS['same_italic']),
        _flag(row, 'linebreak_flag', FLAG_DEFAULTS['linebreak_flag']),
        _flag(row, 'hyphenation_flag', FLAG_DEFAULTS['hyphenation_flag']),
        _flag(row, 'same_column', FLAG_DEFAULTS['same_column']),
    )

# ---- Main dedup ------------------------------------------------------------

def deduplicate_training_data(
    df: pd.DataFrame,
    scope_column: bool = True,
    class_handling: Optional[str] = None  # None, "balance", "weight"
) -> Tuple[pd.DataFrame, dict]:
    """Deduplicate training data with optional class balancing/weighting (robust)."""

    # Ensure required columns exist with defaults
    for col, default in NUM_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    for col, default in FLAG_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    for col, default in SORT_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    # Hash key
    df = df.copy()
    df['hash_key'] = df.apply(compute_hash_key, axis=1)

    # Scope
    if scope_column:
        df['scope_id'] = df.apply(lambda x: (x.get('doc_id'), x.get('page'), x.get('column', 0)), axis=1)
    else:
        df['scope_id'] = df.apply(lambda x: (x.get('doc_id'), x.get('page')), axis=1)

    # Sort for deterministic keep-first (columns are guaranteed to exist now)
    df = df.sort_values(['doc_id', 'page', 'reading_order_prev', 'reading_order_curr'], kind='mergesort')

    # Deduplicate
    df_dedup = df.drop_duplicates(subset=['scope_id', 'hash_key'], keep='first')

    metadata = {
        'original_size': int(len(df)),
        'dedup_size': int(len(df_dedup)),
        'reduction_rate': float(1 - (len(df_dedup) / max(len(df), 1))),
    }

    # Optional class handling
    if class_handling in {"balance", "weight"} and 'y_boundary' not in df_dedup.columns:
        # If labels are missing, skip gracefully
        metadata['note'] = "y_boundary not found; skipped class handling."
        df_out = df_dedup.drop(columns=['hash_key', 'scope_id'])
        return df_out, metadata

    if class_handling == "balance":
        class_counts = df_dedup['y_boundary'].value_counts(dropna=False)
        # Need both 0 and 1 present to balance
        if not {0, 1}.issubset(class_counts.index):
            metadata['note'] = "Not enough classes to balance; returning deduplicated data."
            df_out = df_dedup.drop(columns=['hash_key', 'scope_id'])
            return df_out, metadata

        minority_count = int(class_counts.min())
        # Guard against zero or tiny minority
        if minority_count == 0:
            metadata['note'] = "Minority class is empty; cannot balance."
            df_out = df_dedup.drop(columns=['hash_key', 'scope_id'])
            return df_out, metadata

        df_balanced = pd.concat([
            df_dedup[df_dedup['y_boundary'] == 0].sample(n=minority_count, random_state=42),
            df_dedup[df_dedup['y_boundary'] == 1].sample(n=minority_count, random_state=42),
        ], ignore_index=True).sort_index(kind='mergesort')

        metadata['balanced_size'] = int(len(df_balanced))
        metadata['class_counts'] = df_balanced['y_boundary'].value_counts().to_dict()

        df_balanced = df_balanced.drop(columns=['hash_key', 'scope_id'])
        return df_balanced, metadata

    elif class_handling == "weight":
        class_counts = df_dedup['y_boundary'].value_counts()
        neg = int(class_counts.get(0, 0))
        pos = int(class_counts.get(1, 0))
        scale_pos_weight = (neg / pos) if pos > 0 else np.inf

        metadata['scale_pos_weight'] = float(scale_pos_weight)
        metadata['class_counts'] = {int(k): int(v) for k, v in class_counts.to_dict().items()}

    # Cleanup
    df_dedup = df_dedup.drop(columns=['hash_key', 'scope_id'])
    return df_dedup, metadata
