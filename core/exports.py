from pathlib import Path
import pandas as pd
from typing import List, Tuple
from .io import load_spans, load_labels
from .features import extract_boundary_training_data

def export_training_data(
    doc_ids: List[str], 
    output_path: Path, 
    test_size: float = 0.2,
    random_state: int = 42,
    data_dir: Path = Path("data")
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Export training data for multiple documents using sliding window features"""
    
    all_features = []
    all_targets = []
    all_doc_ids = []
    all_span_ids = []
    all_pages = []
    
    for doc_id in doc_ids:
        spans = load_spans(doc_id, data_dir)
        labels = load_labels(doc_id, data_dir)
        
        if not spans:
            print(f"Warning: No spans found for {doc_id}")
            continue
        if not labels:
            print(f"Warning: No labels found for {doc_id}")
            continue
        
        # Extract sliding window features (one per span)
        features, targets = extract_boundary_training_data(spans, labels)
        
        all_features.extend(features)
        all_targets.extend(targets)
        all_doc_ids.extend([doc_id] * len(features))
        all_span_ids.extend([span.span_id for span in spans])
        all_pages.extend([span.page_number for span in spans])
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    df['y_boundary'] = all_targets
    df['doc_id'] = all_doc_ids
    df['span_id'] = all_span_ids
    df['page'] = all_pages
    
    # Reorder columns for clarity
    metadata_cols = ['doc_id', 'page', 'span_id']
    feature_cols = [col for col in df.columns if col not in metadata_cols + ['y_boundary']]
    df = df[metadata_cols + feature_cols + ['y_boundary']]
    
    # Create train/test split
    train_df, test_df = create_train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Save full dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    # Save train and test sets
    train_path = output_path.parent / f'train_{output_path.name}'
    test_path = output_path.parent / f'test_{output_path.name}'
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"✅ Exported {len(df)} training examples from {len(doc_ids)} documents")
    print(f"   - Full dataset: {output_path}")
    print(f"   - Training set: {train_path} ({len(train_df)} examples)")
    print(f"   - Test set: {test_path} ({len(test_df)} examples)")
    print(f"   - Positive boundaries: {df['y_boundary'].sum()}")
    print(f"   - Negative boundaries: {len(df) - df['y_boundary'].sum()}")
    print(f"   - Features: {len(feature_cols)}")
    
    return df, train_df, test_df

def create_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by document ID to avoid data leakage.
    
    Since each span appears only once, we can safely split by document.
    """
    import numpy as np
    
    # Get unique doc_ids and split them
    doc_ids = df['doc_id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(doc_ids)
    
    n_test = int(len(doc_ids) * test_size)
    test_doc_ids = set(doc_ids[:n_test])
    
    train_df = df[~df['doc_id'].isin(test_doc_ids)].copy()
    test_df = df[df['doc_id'].isin(test_doc_ids)].copy()
    
    print(f"📊 Train/Test Split:")
    print(f"   - Train docs: {len(doc_ids) - n_test}, examples: {len(train_df)}")
    print(f"   - Test docs: {n_test}, examples: {len(test_df)}")
    
    return train_df, test_df