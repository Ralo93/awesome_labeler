# features.py
import math
from typing import List, Tuple, Optional, Dict
import pandas as pd
from .schematas import Span, Label, Rule
from typing import Iterable, Tuple
import numpy as np
import pandas as pd

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to [0, 1] with clipping."""
    if max_val == min_val:
        return 0.0
    return _clip01((value - min_val) / (max_val - min_val))

def _page_metrics(prev_span: Span, curr_span: Span,
                  page_height: Optional[float], page_width: Optional[float]) -> Tuple[float, float, float]:
    """Return (width, height, diagonal) with sensible defaults."""
    w = page_width if page_width and page_width > 0 else max(prev_span.right, curr_span.right)
    h = page_height if page_height and page_height > 0 else max(prev_span.top, curr_span.top)
    d = math.hypot(w, h) if w and h else 1.0
    return w, h, d

def compute_pairwise_features(
    prev_span: Span,
    curr_span: Span,
    page_height: Optional[float] = 842.0,  # A4 default
    page_width: Optional[float] = 595.0
) -> Dict[str, float]:
    """Compute pairwise features between consecutive spans on the same page."""
    assert prev_span.page_number == curr_span.page_number, "pair must be on same page"

    w, h, diag = _page_metrics(prev_span, curr_span, page_height, page_width)

    # Basic geometry
    dx = curr_span.x_center - prev_span.x_center
    dy = curr_span.y_bottom - prev_span.y_bottom

    # Normalized gaps
    x_offset_norm = normalize_value(abs(dx), 0.0, w)
    y_gap_norm = normalize_value(abs(dy), 0.0, h)
    center_dist = math.hypot(dx, dy)
    center_dist_norm = normalize_value(center_dist, 0.0, diag)

    # Vertical overlap (based on bbox)
    overlap_h = max(0.0, min(prev_span.top, curr_span.top) - max(prev_span.bottom, curr_span.bottom))
    avg_h = (prev_span.height + curr_span.height) / 2.0 if (prev_span.height + curr_span.height) > 0 else 0.0
    vertical_overlap = (overlap_h / avg_h) if avg_h > 0 else 0.0

    # Indentation & left-edge deltas
    indent_diff = curr_span.left - prev_span.left
    indent_diff_norm = normalize_value(abs(indent_diff), 0.0, w)

    right_edge_diff = curr_span.right - prev_span.right
    right_edge_diff_norm = normalize_value(abs(right_edge_diff), 0.0, w)

    # Typography
    font_size_diff = curr_span.font_size - prev_span.font_size
    font_size_diff_norm = normalize_value(abs(font_size_diff), 0.0, 20.0)

    line_height_diff = curr_span.line_height - prev_span.line_height
    line_height_diff_norm = normalize_value(abs(line_height_diff), 0.0, 20.0)

    same_font_size = int(abs(font_size_diff) < 0.1)
    same_line_height = int(abs(line_height_diff) < 0.1)

    # Binary style/layout
    same_column = int(prev_span.column == curr_span.column)
    same_bold = int(prev_span.bold == curr_span.bold)
    same_italic = int(prev_span.italic == curr_span.italic)

    # Reading order gap (>=0)
    reading_order_gap = max(0, curr_span.reading_order - prev_span.reading_order - 1)

    # Text cues
    prev_text = (prev_span.text or "").rstrip()
    curr_text = curr_span.text or ""
    prev_ends_with_newline = int(prev_span.text.endswith("\n")) if prev_span.text else 0
    prev_ends_with_hyphen = int(prev_text.endswith("-"))
    prev_ends_with_colon = int(prev_text.endswith(":"))
    prev_ends_with_period = int(prev_text.endswith("."))
    prev_ends_with_comma = int(prev_text.endswith(","))

    curr_starts_lower = int(curr_text[:1].islower()) if curr_text else 0
    curr_starts_upper = int(curr_text[:1].isupper()) if curr_text else 0

    return {
        "x_offset_norm": x_offset_norm,
        "y_gap_norm": y_gap_norm,
        "center_dist_norm": center_dist_norm,
        "vertical_overlap": vertical_overlap,
        "indent_diff_norm": indent_diff_norm,
        "right_edge_diff_norm": right_edge_diff_norm,
        "font_size_diff_norm": font_size_diff_norm,
        "line_height_diff_norm": line_height_diff_norm,
        "same_font_size": same_font_size,
        "same_line_height": same_line_height,
        "same_column": same_column,
        "same_bold": same_bold,
        "same_italic": same_italic,
        "reading_order_gap": reading_order_gap,
        "prev_ends_with_newline": prev_ends_with_newline,
        "prev_ends_with_hyphen": prev_ends_with_hyphen,
        "prev_ends_with_colon": prev_ends_with_colon,
        "prev_ends_with_period": prev_ends_with_period,
        "prev_ends_with_comma": prev_ends_with_comma,
        "curr_starts_lower": curr_starts_lower,
        "curr_starts_upper": curr_starts_upper,
        # raw helpers (optional to keep)
        "prev_height": prev_span.height,
        "curr_height": curr_span.height,
        "prev_width": prev_span.width,
        "curr_width": curr_span.width,
    }

def _rule_class_id(r: Rule) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
    """Compact rule signature for equality checks."""
    return (
        r.is_header, r.is_caption, r.is_page_num, r.is_textitem, r.is_toc,
        r.is_digit, r.is_misc, r.is_list_bullet, r.is_list_number, r.is_footnote
    )

def build_training_data(
    spans: List[Span],
    labels: List[Label],
    rules: Optional[List[Rule]] = None,
    page_sizes: Optional[Dict[Tuple[str, int], Tuple[float, float]]] = None,  # {(doc_id, page): (W, H)}
) -> pd.DataFrame:
    """Build training data from spans and labels.
    page_sizes allows per-page (width, height) overrides."""
    label_map = {(l.page_number, l.span_id): l for l in labels}

    rule_map = {}
    if rules:
        rule_map = {(r.page_number, r.span_id): r for r in rules}

    # sort by (doc_id, page, reading_order) for multi-doc robustness
    sorted_spans = sorted(spans, key=lambda s: (s.doc_id, s.page_number, s.reading_order))

    rows = []
    for i in range(1, len(sorted_spans)):
        prev_span = sorted_spans[i - 1]
        curr_span = sorted_spans[i]

        # Only consider consecutive spans on the same doc & page
        if prev_span.doc_id != curr_span.doc_id or prev_span.page_number != curr_span.page_number:
            continue

        # Need a label for the current span
        lbl = label_map.get((curr_span.page_number, curr_span.span_id))
        if not lbl:
            continue

        # Per-page size (if provided)
        W = H = None
        if page_sizes:
            W, H = page_sizes.get((curr_span.doc_id, curr_span.page_number), (None, None))

        # Features
        feats = compute_pairwise_features(prev_span, curr_span, page_height=H, page_width=W)

        row = {
            "doc_id": curr_span.doc_id,
            "page": curr_span.page_number,
            "span_id_prev": prev_span.span_id,
            "span_id_curr": curr_span.span_id,
            "reading_order_prev": prev_span.reading_order,
            "reading_order_curr": curr_span.reading_order,
            "y_boundary": 1 if lbl.boundary == "new" else 0,
            **feats,
        }

        # Add rule features if available for BOTH spans
        if rule_map:
            pr = rule_map.get((prev_span.page_number, prev_span.span_id))
            cr = rule_map.get((curr_span.page_number, curr_span.span_id))
            if pr and cr:
                row.update({
                    "prev_is_header": pr.is_header,
                    "curr_is_header": cr.is_header,
                    "prev_is_caption": pr.is_caption,
                    "curr_is_caption": cr.is_caption,
                    "prev_is_page_num": pr.is_page_num,
                    "curr_is_page_num": cr.is_page_num,
                    "prev_is_list_bullet": pr.is_list_bullet,
                    "curr_is_list_bullet": cr.is_list_bullet,
                    "prev_is_list_number": pr.is_list_number,
                    "curr_is_list_number": cr.is_list_number,
                    "prev_is_footnote": pr.is_footnote,
                    "curr_is_footnote": cr.is_footnote,
                    "prev_header_level": pr.header_level if pr.header_level is not None else -1,
                    "curr_header_level": cr.header_level if cr.header_level is not None else -1,
                    "same_rule_class": int(_rule_class_id(pr) == _rule_class_id(cr)),
                })

        rows.append(row)

    df = pd.DataFrame(rows)
    # (Optional) ensure dtypes are consistent
    if not df.empty:
        bool_like = [
            "same_font_size", "same_line_height", "same_column",
            "same_bold", "same_italic", "prev_ends_with_newline",
            "prev_ends_with_hyphen", "prev_ends_with_colon",
            "prev_ends_with_period", "prev_ends_with_comma",
            "curr_starts_lower", "curr_starts_upper",
        ]
        for c in set(bool_like).intersection(df.columns):
            df[c] = df[c].astype("int8")
    return df


# --- features.py ---


def add_reading_order(
    df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("doc_id", "page_num"),
    y_col: str = "top",
    x_col: str = "left",
    h_col: str = "height",
    y_tolerance_ratio: float = 0.4,
    id_col: str = "segment_id",
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Compute a reading order within each (doc_id, page_num) by grouping rows into
    'lines' with a vertical tolerance, then sorting left-to-right within lines,
    and finally top-to-bottom across lines.

    Adds columns:
      - reading_order (int, 0-based within group)
      - line_id (int, 0-based within group)
      - prev_id / next_id (neighbor segment_ids within the group)
      - prev_text (shifted text within group)

    Requirements: y_col, x_col exist (page-relative coords). If h_col isn't present,
    a fixed tolerance is used.
    """
    req = set(group_cols + (y_col, x_col))
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for reading order: {missing}")

    out = df.copy()

    # Line grouping: rows are on the same 'line' if their 'top' is within a tolerance.
    def _assign_lines(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values([y_col, x_col]).reset_index(drop=True)
        if h_col in g.columns:
            # Median height -> tolerance
            med_h = float(np.nanmedian(g[h_col])) if g[h_col].notna().any() else 0.0
        else:
            med_h = 0.0

        # Tolerance: fraction of median height; fallback to a tiny constant
        tol = med_h * y_tolerance_ratio if med_h > 0 else 5.0

        line_ids = []
        current_line = 0
        last_top = None
        for _, row in g.iterrows():
            t = float(row[y_col])
            if last_top is None:
                line_ids.append(current_line)
                last_top = t
                continue
            if abs(t - last_top) <= tol:
                line_ids.append(current_line)
            else:
                current_line += 1
                line_ids.append(current_line)
                last_top = t
        g["line_id"] = line_ids

        # Sort by (line_id asc, x asc) for LTR reading
        g = g.sort_values(["line_id", x_col], kind="mergesort").reset_index(drop=True)

        # Stable final order index
        g["reading_order"] = np.arange(len(g), dtype=int)

        # Neighbor helpers
        if id_col in g.columns:
            g["prev_id"] = g[id_col].shift(1)
            g["next_id"] = g[id_col].shift(-1)
        else:
            # create a temporary id from index if missing
            tmp_ids = g.index.astype(str)
            g["prev_id"] = tmp_ids.shift(1)
            g["next_id"] = tmp_ids.shift(-1)

        if text_col in g.columns:
            g["prev_text"] = g[text_col].shift(1).fillna("")
        else:
            g["prev_text"] = ""

        return g

    out = (
        out.groupby(list(group_cols), group_keys=False, dropna=False)
           .apply(_assign_lines)
           .reset_index(drop=True)
    )
    return out
