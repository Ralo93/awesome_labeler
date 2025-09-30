import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.stats import entropy

def compute_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    """Compute uncertainty (entropy) for each prediction"""
    # Convert single probability to binary probabilities
    probs = np.column_stack([1 - probabilities, probabilities])
    
    # Compute entropy
    uncertainties = entropy(probs, axis=1, base=2)
    return uncertainties

def find_uncertain_boundaries(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    top_n: int = 10
) -> List[Tuple[int, str, str, float]]:
    """Find most uncertain boundaries"""
    
    uncertainties = compute_uncertainty(probabilities)
    
    # Add uncertainties to dataframe
    df_with_uncertainty = df.copy()
    df_with_uncertainty['uncertainty'] = uncertainties
    
    # Sort by uncertainty
    df_sorted = df_with_uncertainty.nlargest(top_n, 'uncertainty')
    
    # Return list of (page, span_id_prev, span_id_curr, uncertainty)
    results = []
    for _, row in df_sorted.iterrows():
        results.append((
            row['page'],
            row['span_id_prev'],
            row['span_id_curr'],
            row['uncertainty']
        ))
    
    return results

def suggest_next_page(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    current_page: int
) -> Tuple[int, float]:
    """Suggest next page with highest average uncertainty"""
    
    uncertainties = compute_uncertainty(probabilities)
    df_with_uncertainty = df.copy()
    df_with_uncertainty['uncertainty'] = uncertainties
    
    # Group by page and compute mean uncertainty
    page_uncertainty = df_with_uncertainty.groupby('page')['uncertainty'].mean()
    
    # Filter pages after current
    future_pages = page_uncertainty[page_uncertainty.index > current_page]
    
    if len(future_pages) > 0:
        next_page = future_pages.idxmax()
        return int(next_page), float(future_pages.max())
    
    return current_page, 0.0