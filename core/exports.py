from pathlib import Path
import pandas as pd
from typing import List, Optional
from .io import load_spans, load_labels, load_rules, save_parquet
from .features import build_training_data, add_reading_order
from .dedup import deduplicate_training_data

def export_training_data(
    file_name: str,
    doc_ids: List[str],
    export_raw: bool = True,
    export_with_rules: bool = True,
    class_handling: Optional[str] = None,
    data_dir: Path = Path("data"),
    rules_dir: Path = Path("rules"),
    export_dir: Path = Path("exports"),
) -> dict:
    """Export deduplicated training data (with reading order)."""

    all_spans = []
    all_labels = []

    # Load all data
    for doc_id in doc_ids:
        spans = load_spans(doc_id, data_dir)
        labels = load_labels(doc_id, data_dir)
        all_spans.extend(spans)
        all_labels.extend(labels)

    export_results = {}
    export_dir.mkdir(parents=True, exist_ok=True)

    # ---------- RAW ----------
    if export_raw:
        df_raw = build_training_data(all_spans, all_labels, rules=None)

        # Deduplicate first
        df_raw_dedup, metadata_raw = deduplicate_training_data(
            df_raw,
            class_handling=class_handling
        )

        # Add reading order AFTER dedup
        #df_raw_export = _attach_reading_order(df_raw_dedup)

        raw_path = export_dir / f"train_raw_{file_name}.parquet"
        save_parquet(df_raw_dedup, raw_path, metadata_raw)

        export_results["raw"] = {
            "path": str(raw_path),
            "shape": df_raw_dedup.shape,
            "metadata": metadata_raw,
        }

    # ---------- WITH RULES ----------
    if export_with_rules:
        df_rules = build_training_data(all_spans, all_labels, rules=rules_dir)

        df_rules_dedup, metadata_rules = deduplicate_training_data(
            df_rules,
            class_handling=class_handling
        )

        #df_rules_export = _attach_reading_order(df_rules_dedup)

        rules_path = export_dir / f"train_rules_{file_name}.parquet"
        save_parquet(df_rules_dedup, rules_path, metadata_rules)

        export_results["with_rules"] = {
            "path": str(rules_path),
            "shape": df_rules_dedup.shape,
            "metadata": metadata_rules,
        }

    return export_results

def _ensure_segment_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a stable per-page segment_id if missing.
    Assumes doc_id + page_num define a page. Uses row order as a fallback.
    """
    if "segment_id" not in df.columns:
        df = df.copy()
        df["segment_id"] = (
            df.sort_values(["doc_id", "page", "top", "left"])
              .groupby(["doc_id", "page"], dropna=False)
              .cumcount()
              .astype(str)
        )
    return df

def _attach_reading_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute reading order and neighbor helpers. Adds defaults if geometry/text missing.
    """
    df = _ensure_segment_id(df)

    # Guard: if geometry missing, add minimal columns so add_reading_order can run
    for col in ("top", "left"):
        if col not in df.columns:
            df[col] = 0.0
    if "height" not in df.columns:
        df["height"] = pd.NA
    if "text" not in df.columns:
        df["text"] = ""

    df_ro = add_reading_order(
        df,
        group_cols=("doc_id", "page_num"),
        y_col="top",
        x_col="left",
        h_col="height",
        id_col="segment_id",
        text_col="text",
    )

    # Make sure expected columns exist (paranoia)
    for c in ["reading_order", "line_id", "prev_id", "next_id", "prev_text"]:
        if c not in df_ro.columns:
            df_ro[c] = pd.NA

    # Cast ints where possible (helps downstream)
    for c in ["reading_order", "line_id"]:
        if c in df_ro.columns:
            try:
                df_ro[c] = df_ro[c].astype("Int64")
            except Exception:
                pass

    return df_ro