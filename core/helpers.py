from dataclasses import dataclass, asdict
import streamlit as st
from PIL import Image
from typing import Callable, Tuple, Optional, Dict, Any
import time
from functools import wraps
from datetime import datetime
import json
import csv
from pathlib import Path
import os

@dataclass
class ProcessingMetrics:
    """Stores processing metrics for a single detection"""
    model_name: str
    page_number: int
    processing_time: float
    num_blocks_detected: int
    timestamp: datetime
    image_size: Tuple[int, int]
    document_name: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime to ISO format string
        data['timestamp'] = self.timestamp.isoformat()
        # Convert tuple to list for JSON compatibility
        data['image_size'] = list(self.image_size)
        return data
    
    def to_csv_row(self) -> Dict[str, Any]:
        """Convert metrics to flat dictionary for CSV export"""
        return {
            'model_name': self.model_name,
            'page_number': self.page_number,
            'processing_time': self.processing_time,
            'num_blocks_detected': self.num_blocks_detected,
            'timestamp': self.timestamp.isoformat(),
            'image_width': self.image_size[0],
            'image_height': self.image_size[1],
            'document_name': self.document_name or '',
            'pixels_processed': self.image_size[0] * self.image_size[1],
            'blocks_per_second': self.num_blocks_detected / self.processing_time if self.processing_time > 0 else 0,
            'ms_per_block': (self.processing_time * 1000) / self.num_blocks_detected if self.num_blocks_detected > 0 else 0
        }


class MetricsPersistence:
    """Handles saving and loading of processing metrics"""
    
    def __init__(self, base_dir: str = "./metrics"):
        """
        Initialize metrics persistence
        
        Args:
            base_dir: Base directory for storing metrics files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.json_dir = self.base_dir / "json"
        self.csv_dir = self.base_dir / "csv"
        self.json_dir.mkdir(exist_ok=True)
        self.csv_dir.mkdir(exist_ok=True)
        
        # File paths
        self.json_file = self.json_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
        self.csv_file = self.csv_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.csv"
        self.summary_file = self.base_dir / "metrics_summary.json"
        
    def save_metrics(self, metrics: ProcessingMetrics) -> bool:
        """
        Save metrics to both JSON and CSV files
        
        Args:
            metrics: ProcessingMetrics object to save
            
        Returns:
            True if successful, False otherwise
        """
        success = True
        
        # Save to JSON
        try:
            self._save_to_json(metrics)
        except Exception as e:
            print(f"Error saving to JSON: {e}")
            success = False
        
        # Save to CSV
        try:
            self._save_to_csv(metrics)
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            success = False
        
        # Update summary
        try:
            self._update_summary(metrics)
        except Exception as e:
            print(f"Error updating summary: {e}")
            success = False
        
        return success
    
    def _save_to_json(self, metrics: ProcessingMetrics):
        """Save metrics to JSON file (append mode)"""
        # Load existing data or create new list
        if self.json_file.exists():
            with open(self.json_file, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        # Append new metrics
        data.append(metrics.to_dict())
        
        # Save back to file
        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_to_csv(self, metrics: ProcessingMetrics):
        """Save metrics to CSV file (append mode)"""
        csv_row = metrics.to_csv_row()
        
        # Check if file exists to determine if we need headers
        file_exists = self.csv_file.exists()
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(csv_row)
    
    def _update_summary(self, metrics: ProcessingMetrics):
        """Update summary statistics"""
        # Load existing summary or create new
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {
                'total_pages_processed': 0,
                'total_blocks_detected': 0,
                'total_processing_time': 0,
                'models_used': {},
                'documents_processed': set(),
                'first_run': datetime.now().isoformat(),
                'last_run': None,
                'fastest_processing': None,
                'slowest_processing': None
            }
        
        # Update summary
        summary['total_pages_processed'] += 1
        summary['total_blocks_detected'] += metrics.num_blocks_detected
        summary['total_processing_time'] += metrics.processing_time
        summary['last_run'] = metrics.timestamp.isoformat()
        
        # Update model-specific stats
        if metrics.model_name not in summary['models_used']:
            summary['models_used'][metrics.model_name] = {
                'runs': 0,
                'total_time': 0,
                'total_blocks': 0,
                'avg_time': 0,
                'avg_blocks': 0
            }
        
        model_stats = summary['models_used'][metrics.model_name]
        model_stats['runs'] += 1
        model_stats['total_time'] += metrics.processing_time
        model_stats['total_blocks'] += metrics.num_blocks_detected
        model_stats['avg_time'] = model_stats['total_time'] / model_stats['runs']
        model_stats['avg_blocks'] = model_stats['total_blocks'] / model_stats['runs']
        
        # Update fastest/slowest
        if summary['fastest_processing'] is None or metrics.processing_time < summary['fastest_processing']['time']:
            summary['fastest_processing'] = {
                'time': metrics.processing_time,
                'model': metrics.model_name,
                'blocks': metrics.num_blocks_detected,
                'timestamp': metrics.timestamp.isoformat()
            }
        
        if summary['slowest_processing'] is None or metrics.processing_time > summary['slowest_processing']['time']:
            summary['slowest_processing'] = {
                'time': metrics.processing_time,
                'model': metrics.model_name,
                'blocks': metrics.num_blocks_detected,
                'timestamp': metrics.timestamp.isoformat()
            }
        
        # Handle documents set (convert to list for JSON)
        if metrics.document_name:
            if isinstance(summary['documents_processed'], list):
                if metrics.document_name not in summary['documents_processed']:
                    summary['documents_processed'].append(metrics.document_name)
            else:
                summary['documents_processed'] = list(summary['documents_processed'])
                summary['documents_processed'].append(metrics.document_name)
        
        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load_metrics(self, date: Optional[str] = None) -> list:
        """
        Load metrics from JSON file
        
        Args:
            date: Date string in YYYYMMDD format, defaults to today
            
        Returns:
            List of metrics dictionaries
        """
        if date is None:
            json_file = self.json_file
        else:
            json_file = self.json_dir / f"metrics_{date}.json"
        
        if json_file.exists():
            with open(json_file, 'r') as f:
                return json.load(f)
        return []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                return json.load(f)
        return {}


def measure_processing_time(
    model_name: str = "Unknown Model",
    save_to_file: bool = True,
    metrics_dir: str = "./metrics",
    capture_streamlit_state: bool = True
):
    """
    Decorator to measure processing time of layout detection functions.
    
    Args:
        model_name: Name of the model being used for tracking
        save_to_file: Whether to save metrics to file
        metrics_dir: Directory to save metrics files
        capture_streamlit_state: Whether to capture metrics in Streamlit session state
        
    Returns:
        Decorated function that measures and stores processing time
    """
    # Initialize persistence if saving to file
    persistence = MetricsPersistence(metrics_dir) if save_to_file else None
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract image and document info from args if available
            image = None
            page_num = kwargs.get('page_num', 1)
            document_name = kwargs.get('document_name', None)
            
            # Try to find image in args
            for arg in args:
                if isinstance(arg, Image.Image):
                    image = arg
                    break
            
            # Get document name from Streamlit session state if available
            if document_name is None and capture_streamlit_state:
                try:
                    if 'current_doc' in st.session_state and st.session_state.current_doc:
                        document_name = Path(st.session_state.current_doc).name
                except:
                    pass
            
            # Start timing
            start_time = time.perf_counter()
            
            # Run the actual function
            try:
                result = func(*args, **kwargs)
                success = True
                error_msg = None
            except Exception as e:
                result = []
                success = False
                error_msg = str(e)
            
            # End timing
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            # Create metrics
            if image:
                metrics = ProcessingMetrics(
                    model_name=model_name,
                    page_number=page_num,
                    processing_time=processing_time,
                    num_blocks_detected=len(result) if isinstance(result, list) else 0,
                    timestamp=datetime.now(),
                    image_size=image.size,
                    document_name=document_name,
                    additional_info={
                        'success': success,
                        'error': error_msg,
                        'function_name': func.__name__,
                        'pixels_processed': image.size[0] * image.size[1]
                    }
                )
                
                # Save to file if enabled
                if save_to_file and persistence:
                    try:
                        persistence.save_metrics(metrics)
                    except Exception as e:
                        print(f"Failed to save metrics: {e}")
                
                # Store in Streamlit session state if available
                if capture_streamlit_state:
                    try:
                        if 'processing_times' not in st.session_state:
                            st.session_state.processing_times = []
                        st.session_state.processing_times.append(metrics)
                        
                        # Display in sidebar if available
                        with st.sidebar:
                            st.success(f"⏱️ Processing time: {processing_time:.3f}s")
                            st.caption(f"Detected {metrics.num_blocks_detected} blocks")
                            if save_to_file:
                                st.caption("✅ Metrics saved to file")
                    except:
                        pass
            
            return result
        return wrapper
    return decorator


# Example usage function
def load_metrics_for_analysis(metrics_dir: str = "./metrics") -> Dict[str, Any]:
    """
    Load all metrics for analysis
    
    Args:
        metrics_dir: Directory containing metrics files
        
    Returns:
        Dictionary with metrics data and summary
    """
    persistence = MetricsPersistence(metrics_dir)
    
    return {
        'today_metrics': persistence.load_metrics(),
        'summary': persistence.get_summary(),
        'csv_files': list(persistence.csv_dir.glob("*.csv")),
        'json_files': list(persistence.json_dir.glob("*.json"))
    }


# Streamlit UI component for displaying metrics
def display_metrics_dashboard(metrics_dir: str = "./metrics"):
    """Display a dashboard of processing metrics in Streamlit"""
    st.header("📊 Processing Metrics Dashboard")
    
    data = load_metrics_for_analysis(metrics_dir)
    
    if data['summary']:
        summary = data['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pages", summary.get('total_pages_processed', 0))
        
        with col2:
            st.metric("Total Blocks", summary.get('total_blocks_detected', 0))
        
        with col3:
            avg_time = summary['total_processing_time'] / summary['total_pages_processed'] if summary.get('total_pages_processed', 0) > 0 else 0
            st.metric("Avg Time/Page", f"{avg_time:.3f}s")
        
        with col4:
            st.metric("Models Used", len(summary.get('models_used', {})))
        
        # Model comparison
        if summary.get('models_used'):
            st.subheader("Model Performance Comparison")
            
            model_data = []
            for model_name, stats in summary['models_used'].items():
                model_data.append({
                    'Model': model_name,
                    'Runs': stats['runs'],
                    'Avg Time (s)': f"{stats['avg_time']:.3f}",
                    'Avg Blocks': f"{stats['avg_blocks']:.1f}",
                    'Total Time (s)': f"{stats['total_time']:.2f}"
                })
            
            st.dataframe(model_data)
        
        # Best/Worst performance
        col1, col2 = st.columns(2)
        
        with col1:
            if summary.get('fastest_processing'):
                st.success(f"⚡ Fastest: {summary['fastest_processing']['time']:.3f}s ({summary['fastest_processing']['model']})")
        
        with col2:
            if summary.get('slowest_processing'):
                st.warning(f"🐌 Slowest: {summary['slowest_processing']['time']:.3f}s ({summary['slowest_processing']['model']})")
    else:
        st.info("No metrics data available yet. Process some documents to see metrics.")