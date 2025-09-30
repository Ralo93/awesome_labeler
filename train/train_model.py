#!/usr/bin/env python
"""
Train a boundary classification model for semantic unit creation.

- Training data is ALWAYS expected at: exports/train_raw_*.parquet
- All parquet files in exports/ matching train_raw_*.parquet are concatenated.
- Output artifacts are ALWAYS saved under: models/
- The parquet files contain pre-computed features.

Artifacts written:
- models/boundary_model.pkl                (sklearn Pipeline)
- models/boundary_model_metadata.json      (training metadata)
- models/boundary_feature_importance.csv   (sorted importances)
- models/boundary_confusion_matrix.png     (confusion matrix)
- models/boundary_top_features.png         (top-20 feature importances)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# -----------------------------
# Constants
# -----------------------------

# Columns to always drop (metadata columns)
DROP_ALWAYS = {
    "doc_id", "page", 
    "span_id_prev", "span_id_curr",
    "reading_order_prev", "reading_order_curr",
}

# The target column
TARGET_COLUMN = "y_boundary"

# Feature columns expected in the parquet files (for validation)
EXPECTED_FEATURE_TYPES = {
    "numeric": [
        "x_offset_norm", "y_gap_norm", "center_dist_norm",
        "vertical_overlap", "indent_diff_norm", "right_edge_diff_norm",
        "font_size_diff_norm", "line_height_diff_norm",
        "prev_height", "curr_height", "prev_width", "curr_width"
    ],
    "boolean": [
        "same_font_size", "same_line_height", "same_column",
        "same_bold", "same_italic",
        "prev_ends_with_newline", "prev_ends_with_hyphen",
        "prev_ends_with_colon", "prev_ends_with_period", "prev_ends_with_comma",
        "curr_starts_lower", "curr_starts_upper"
    ],
    "integer": [
        "reading_order_gap", "linebreak_flag", "hyphenation_flag"
    ]
}

# -----------------------------
# Feature extraction
# -----------------------------

def extract_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract features from the pre-computed parquet data.
    
    Returns:
        - DataFrame with features
        - List of feature column names
    """
    df = df.copy()
    
    # Check if target column exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data. Available columns: {list(df.columns)}")
    
    # Identify feature columns (everything except DROP_ALWAYS and target)
    feature_cols = []
    for col in df.columns:
        if col in DROP_ALWAYS or col == TARGET_COLUMN:
            continue
        feature_cols.append(col)
    
    # Validate and clean numeric features
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Replace inf/-inf with NaN, then fill with 0
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        elif df[col].dtype == 'bool' or df[col].dtype == 'int8':
            # Boolean columns - ensure they're numeric (0/1)
            df[col] = df[col].astype(int).fillna(0)
        else:
            # Try to convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    print(f"Extracted {len(feature_cols)} features: {feature_cols[:10]}..." if len(feature_cols) > 10 else feature_cols)
    
    return df, feature_cols


# -----------------------------
# Model training
# -----------------------------

def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                X_test: pd.DataFrame, y_test: pd.Series,
                feature_cols: List[str]) -> Tuple[Pipeline, dict]:
    """
    Train the boundary classification model.
    
    Returns:
        - Trained sklearn Pipeline
        - Dictionary of metrics
    """
    # Create pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=20,  # Limit depth to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',  # Use sqrt of features for each split
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    
    print("Training Random Forest model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    y_pred = pipeline.predict(X_test)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Get classification report
    report = classification_report(y_test, y_pred, 
                                 target_names=["continue", "new"],
                                 output_dict=True)
    
    metrics = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "classification_report": report
    }
    
    return pipeline, metrics


# -----------------------------
# Visualization
# -----------------------------

def create_visualizations(pipeline: Pipeline, X_test: pd.DataFrame, 
                         y_test: pd.Series, feature_cols: List[str],
                         models_dir: Path) -> dict:
    """
    Create and save visualization artifacts.
    
    Returns:
        Dictionary of paths to saved visualizations
    """
    artifacts = {}
    
    # 1. Confusion Matrix
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                 display_labels=["continue", "new"])
    disp.plot(ax=ax, values_format='d')
    ax.set_title('Boundary Classifier - Confusion Matrix')
    fig_cm.tight_layout()
    
    cm_path = models_dir / "boundary_model_confusion_matrix.png"
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)
    artifacts["confusion_matrix"] = str(cm_path)
    
    # 2. Feature Importances
    importances = pipeline.named_steps["clf"].feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_cols, 
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    fi_path = models_dir / "boundary_model_feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    artifacts["feature_importance_csv"] = str(fi_path)
    
    # 3. Top Features Plot
    topn = fi_df.head(20)
    fig_imp, ax = plt.subplots(figsize=(10, 8))
    ax.barh(topn["feature"][::-1], topn["importance"][::-1])
    ax.set_title("Top-20 Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig_imp.tight_layout()
    
    imp_path = models_dir / "boundary_model_top_features.png"
    fig_imp.savefig(imp_path, dpi=150)
    plt.close(fig_imp)
    artifacts["top_features_plot"] = str(imp_path)
    
    return artifacts


# -----------------------------
# Main
# -----------------------------

def main():
    # Setup paths
    input_dir = Path("../app/exports")
    models_dir = Path("../app/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all training parquet files
    parquet_files = list(input_dir.glob("train_raw_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No training parquet files found in {input_dir}")
    
    print(f"Found {len(parquet_files)} training files:")
    for pf in parquet_files:
        print(f"  - {pf.name}")
    
    # Concatenate all parquet files
    dfs = []
    for p in parquet_files:
        df_temp = pd.read_parquet(p)
        print(f"  Loaded {len(df_temp):,} rows from {p.name}")
        dfs.append(df_temp)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(df):,} training examples")
    
    # Check target column and values
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found. Available columns: {list(df.columns)}")
    
    # Extract target variable
    y = df[TARGET_COLUMN].astype(int)
    unique_vals = y.unique()
    value_counts = y.value_counts()
    
    print(f"\nTarget column '{TARGET_COLUMN}' statistics:")
    print(f"  Unique values: {sorted(unique_vals)}")
    print(f"  Distribution:")
    print(f"    0 (continue): {value_counts.get(0, 0):,} ({value_counts.get(0, 0)/len(y)*100:.1f}%)")
    print(f"    1 (new): {value_counts.get(1, 0):,} ({value_counts.get(1, 0)/len(y)*100:.1f}%)")
    
    # Extract features
    print("\nExtracting features...")
    df_feat, feature_cols = extract_features(df)
    X = df_feat[feature_cols]
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data
    print("\nSplitting train/test (80/20)...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        stratified = True
    except ValueError as e:
        print(f"  Warning: Stratified split failed ({e}). Using regular split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        stratified = False
    
    print(f"  Train set: {len(X_train):,} examples")
    print(f"  Test set: {len(X_test):,} examples")
    print(f"  Stratified: {stratified}")
    
    # Train model
    pipeline, metrics = train_model(X_train, y_train, X_test, y_test, feature_cols)
    
    # Print classification report
    print("\nClassification Report:")
    report = metrics["classification_report"]
    print(f"              precision  recall  f1-score  support")
    print(f"continue      {report['continue']['precision']:.2f}      {report['continue']['recall']:.2f}   {report['continue']['f1-score']:.2f}     {report['continue']['support']}")
    print(f"new           {report['new']['precision']:.2f}      {report['new']['recall']:.2f}   {report['new']['f1-score']:.2f}     {report['new']['support']}")
    print(f"\nAccuracy: {metrics['test_accuracy']:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    viz_artifacts = create_visualizations(pipeline, X_test, y_test, 
                                         feature_cols, models_dir)
    
    # Save model
    model_path = models_dir / "boundary2_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\n✅ Saved model: {model_path}")
    
    # Save metadata
    metadata = {
        "model_file": str(model_path),
        "created_at": datetime.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "sklearn_version": joblib.__version__,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_accuracy": metrics["train_accuracy"],
        "test_accuracy": metrics["test_accuracy"],
        "label_mapping": {0: "continue", 1: "new"},
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
        "target_column": TARGET_COLUMN,
        "stratified_split": stratified,
        "classification_report": metrics["classification_report"],
        "artifacts": viz_artifacts,
        "training_files": [str(p) for p in parquet_files],
        "total_training_examples": int(len(df))
    }
    
    meta_path = models_dir / "boundary_model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved metadata: {meta_path}")
    print(f"✅ Saved feature importances: {viz_artifacts['feature_importance_csv']}")
    print(f"✅ Saved confusion matrix: {viz_artifacts['confusion_matrix']}")
    print(f"✅ Saved top features plot: {viz_artifacts['top_features_plot']}")
    
    # Print top 10 most important features
    fi_df = pd.read_csv(viz_artifacts['feature_importance_csv'])
    print("\nTop 10 Most Important Features:")
    for idx, row in fi_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['feature']:30s} {row['importance']:.4f}")


if __name__ == "__main__":
    main()