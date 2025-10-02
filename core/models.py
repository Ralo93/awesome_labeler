from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json

class BoundaryModel:
    """
    Boundary detection model using sliding window features.
    
    Replaces the old pairwise approach with richer contextual features.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 random_state: int = 42,
                 calibrate: bool = True):
        
        self.base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'  # Handle imbalanced boundaries
        )
        self.calibrate = calibrate
        self.model = None
        self.feature_names_ = None
        self.feature_names_in_ = None  # Added for sklearn compatibility
        self.threshold_ = 0.5
        self.metadata_ = {}
        
    def train(self, 
              train_df: pd.DataFrame, 
              test_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train boundary detection model using sliding window features.
        
        Args:
            train_df: Training data with sliding window features
            test_df: Optional test data for evaluation
            
        Returns:
            Training metrics and feature importance
        """
        
        # Separate features from metadata and target
        metadata_cols = ['doc_id', 'page', 'span_id', 'y_boundary']
        feature_cols = [c for c in train_df.columns if c not in metadata_cols]
        
        X_train = train_df[feature_cols].fillna(0)  # Handle any missing values
        y_train = train_df['y_boundary']
        
        print(f"🏋️  Training on {len(X_train)} examples with {len(feature_cols)} features")
        print(f"   📊 Class distribution: {y_train.value_counts().to_dict()}")
        
        # Store feature names for consistency
        self.feature_names_ = feature_cols
        self.feature_names_in_ = np.array(feature_cols)  # Store as numpy array for sklearn compatibility
        
        # Train model (with optional calibration)
        if self.calibrate:
            self.model = CalibratedClassifierCV(self.base_model, cv=3)
        else:
            self.model = self.base_model
            
        self.model.fit(X_train, y_train)
        
        # For calibrated models, extract base estimator feature importance
        if self.calibrate and hasattr(self.model, 'calibrated_classifiers_'):
            # Store feature names in base estimator for consistency
            for clf in self.model.calibrated_classifiers_:
                if hasattr(clf, 'estimator'):
                    clf.estimator.feature_names_in_ = self.feature_names_in_
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_prob = self.model.predict_proba(X_train)[:, 1]
        
        train_metrics = {
            'accuracy': (train_pred == y_train).mean(),
            'auc_roc': roc_auc_score(y_train, train_prob),
            'n_samples': len(y_train),
            'n_features': len(feature_cols),
            'pos_rate': y_train.mean()
        }
        
        # Evaluate on test set if provided
        test_metrics = {}
        if test_df is not None and len(test_df) > 0:
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df['y_boundary']
            
            test_pred = self.model.predict(X_test)
            test_prob = self.model.predict_proba(X_test)[:, 1]
            
            test_metrics = {
                'accuracy': (test_pred == y_test).mean(),
                'auc_roc': roc_auc_score(y_test, test_prob),
                'n_samples': len(y_test),
                'pos_rate': y_test.mean()
            }
            
            print(f"🧪 Test AUC: {test_metrics['auc_roc']:.3f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif self.calibrate and hasattr(self.model, 'calibrated_classifiers_'):
            # Get average importance from calibrated classifiers
            importances_list = []
            for clf in self.model.calibrated_classifiers_:
                # Use 'estimator' instead of 'base_estimator'
                if hasattr(clf, 'estimator') and hasattr(clf.estimator, 'feature_importances_'):
                    importances_list.append(clf.estimator.feature_importances_)
            if importances_list:
                importances = np.mean(importances_list, axis=0)
            else:
                importances = np.zeros(len(feature_cols))
        else:
            importances = np.zeros(len(feature_cols))
        
        feature_importance = list(zip(feature_cols, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Store metadata
        self.metadata_ = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance[:20],  # Top 20 features
            'model_type': 'RandomForest + Calibration' if self.calibrate else 'RandomForest',
            'feature_count': len(feature_cols)
        }
        
        print(f"✅ Training complete. Train AUC: {train_metrics['auc_roc']:.3f}")
        
        return self.metadata_
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict boundary probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure feature order matches training
        if hasattr(X, 'columns'):  # DataFrame
            X = X[self.feature_names_].fillna(0)
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            # List of dictionaries
            df = pd.DataFrame(X)
            # Add missing columns with default value 0
            for col in self.feature_names_:
                if col not in df.columns:
                    df[col] = 0
            X = df[self.feature_names_].fillna(0)
        
        return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict boundaries using threshold"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold_).astype(int)
    
    def save(self, model_path: Path) -> None:
        """Save model and metadata"""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self, model_path)
        
        # Save metadata separately for easy inspection
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata_, f, indent=2, default=str)
        
        print(f"💾 Model saved: {model_path}")
        print(f"📋 Metadata saved: {metadata_path}")
    
    @classmethod
    def load(cls, model_path: Path) -> 'BoundaryModel':
        """Load trained model"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        
        # Load metadata if available
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model.metadata_ = json.load(f)
        
        print(f"📂 Model loaded: {model_path}")
        return model
    
    def get_feature_importance(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """Get top feature importances"""
        if not self.metadata_ or 'feature_importance' not in self.metadata_:
            return []
        
        return self.metadata_['feature_importance'][:top_k]
    
    def tune_threshold(self, 
                      X_val: pd.DataFrame, 
                      y_val: pd.Series,
                      metric: str = 'f1') -> float:
        """
        Tune decision threshold on validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels  
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold
        """
        from sklearn.metrics import precision_recall_curve, f1_score
        
        # Get prediction probabilities
        proba = self.predict_proba(X_val[self.feature_names_])[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        scores = []
        
        for thresh in thresholds:
            y_pred = (proba >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif metric == 'precision':
                from sklearn.metrics import precision_score
                score = precision_score(y_val, y_pred, zero_division=0)
            elif metric == 'recall':
                from sklearn.metrics import recall_score
                score = recall_score(y_val, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            scores.append(score)
        
        # Find optimal threshold
        best_idx = np.argmax(scores)
        optimal_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        self.threshold_ = optimal_threshold
        
        print(f"🎯 Optimal threshold: {optimal_threshold:.3f} ({metric}: {best_score:.3f})")
        
        return optimal_threshold

def train_boundary_model(train_path: Path, 
                        test_path: Optional[Path] = None,
                        model_dir: Path = Path("models")) -> BoundaryModel:
    """
    Train a boundary detection model from exported data.
    
    Args:
        train_path: Path to training parquet file
        test_path: Optional path to test parquet file
        model_dir: Directory to save model
        
    Returns:
        Trained model
    """
    
    # Load data
    print(f"📂 Loading training data from {train_path}")
    train_df = pd.read_parquet(train_path)
    
    test_df = None
    if test_path and test_path.exists():
        print(f"📂 Loading test data from {test_path}")
        test_df = pd.read_parquet(test_path)
    
    # Initialize and train model
    model = BoundaryModel(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        calibrate=True
    )
    
    # Train model
    metrics = model.train(train_df, test_df)
    
    # Save model
    model_path = model_dir / "boundary_model.pkl"
    model.save(model_path)
    
    return model

def evaluate_model_performance(model: BoundaryModel, 
                             test_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate model performance on test set"""
    
    if test_df.empty:
        return {}
    
    # Prepare data
    metadata_cols = ['doc_id', 'page', 'span_id', 'y_boundary']
    feature_cols = [c for c in test_df.columns if c not in metadata_cols]
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['y_boundary']
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
    
    metrics = {
        'accuracy': (y_pred == y_test).mean(),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'auc_pr': average_precision_score(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'n_samples': len(y_test),
        'threshold': model.threshold_
    }
    
    return metrics