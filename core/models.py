# boundary_model.py
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.inspection import permutation_importance

MODEL_VERSION = "1.2.0"


DEFAULT_FEATURES = [
    # geometry & spacing
    "x_offset_norm", "y_gap_norm", "center_dist_norm", "vertical_overlap",
    "indent_diff_norm", "right_edge_diff_norm",
    # typography
    "font_size_diff_norm", "line_height_diff_norm",
    "same_font_size", "same_line_height",
    # style/layout
    "same_column", "same_bold", "same_italic",
    # ordering
    "reading_order_gap",
    # text cues
    "prev_ends_with_newline", "prev_ends_with_hyphen", "prev_ends_with_colon",
    "prev_ends_with_period", "prev_ends_with_comma",
    "curr_starts_lower", "curr_starts_upper",
    # (optionally present raw helpers)
    "prev_height", "curr_height", "prev_width", "curr_width",
    # rule features (if present)
    "prev_is_header", "curr_is_header",
    "prev_is_caption", "curr_is_caption",
    "prev_is_page_num", "curr_is_page_num",
    "prev_is_list_bullet", "curr_is_list_bullet",
    "prev_is_list_number", "curr_is_list_number",
    "prev_is_footnote", "curr_is_footnote",
    "prev_header_level", "curr_header_level",
    "same_rule_class",
]

IDENT_COLS = {
    "doc_id", "page", "span_id_prev", "span_id_curr",
    "reading_order_prev", "reading_order_curr"
}

class BoundaryModel:
    """Boundary predictor with calibrated probabilities and robust feature handling."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model = None                     # may be RF or CalibratedClassifierCV
        self.base_model = None                # uncalibrated RF (for importances)
        self.feature_columns: List[str] = []  # in-order training features
        self.threshold_: float = 0.5          # decision threshold on P(boundary=new)
        self.metadata: Dict = {}              # saved extra info

        if model_path and Path(model_path).exists():
            self.load(model_path)

    # --------- persistence ---------
    def load(self, model_path: Path):
        data = joblib.load(model_path)
        self.model = data["model"]
        self.base_model = data.get("base_model")
        self.feature_columns = data["feature_columns"]
        self.threshold_ = data.get("threshold_", 0.5)
        self.metadata = data.get("metadata", {})

    def save(self, model_path: Path):
        payload = {
            "model": self.model,
            "base_model": self.base_model,
            "feature_columns": self.feature_columns,
            "threshold_": self.threshold_,
            "metadata": {
                **self.metadata,
                "version": MODEL_VERSION,
                "saved_at": int(time.time()),
            },
        }
        joblib.dump(payload, model_path)

    # --------- training ---------
    def _resolve_features(self, df: pd.DataFrame, feature_columns: Optional[List[str]]) -> List[str]:
        if feature_columns is None:
            # prefer DEFAULT_FEATURES but only those present; if empty, fallback to all numeric except identifiers/target
            cols = [c for c in DEFAULT_FEATURES if c in df.columns]
            if not cols:
                cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in IDENT_COLS and c != "y_boundary"]
            return cols
        # keep order given by user but only those present
        return [c for c in feature_columns if c in df.columns]

    def _prepare_matrix(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> np.ndarray:
        # add any missing expected columns as zeros (robust to schema drift)
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = 0.0
        # ensure order
        X = df[feature_columns].astype(float).values
        return X

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        *,
        calibrate: bool = True,
        calibration_cv: int = 3,
        class_weight: str = "balanced_subsample",
        rf_params: Optional[Dict] = None,
        set_threshold_by: Optional[str] = None,  # "f1", "best_f1", "target_recall:0.90", "target_precision:0.90"
    ):
        """
        Train a RandomForest on features -> y_boundary, optionally calibrate probabilities and set decision threshold.
        """
        if "y_boundary" not in df.columns:
            raise ValueError("Training DataFrame must contain 'y_boundary' column.")

        self.feature_columns = self._resolve_features(df, feature_columns)
        if not self.feature_columns:
            raise ValueError("No usable feature columns found.")

        X = self._prepare_matrix(df.copy(), self.feature_columns)
        y = df["y_boundary"].astype(int).values

        params = {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": class_weight,
        }
        if rf_params:
            params.update(rf_params)

        rf = RandomForestClassifier(**params)
        rf.fit(X, y)
        self.base_model = rf

        if calibrate:
            # Calibrate on the training data via CV; for production you might want a held-out set
            cal = CalibratedClassifierCV(rf, method="isotonic", cv=calibration_cv)
            cal.fit(X, y)
            self.model = cal
        else:
            self.model = rf

        # optional threshold selection
        if set_threshold_by:
            probs = self.predict_proba(df)
            self.threshold_ = self._choose_threshold(y, probs, set_threshold_by)

        # store helpful metadata
        self.metadata = {
            "rf_params": params,
            "calibrated": calibrate,
            "calibration_cv": calibration_cv if calibrate else None,
            "feature_count": len(self.feature_columns),
            "features": list(self.feature_columns),
        }

    # --------- inference ---------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded or trained.")
        X = self._prepare_matrix(df.copy(), self.feature_columns)
        # proba for class 1 (boundary=new)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        thr = self.threshold_ if threshold is None else float(threshold)
        p = self.predict_proba(df)
        return (p >= thr).astype(int)

    # --------- utilities ---------
    def _choose_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, strategy: str) -> float:
        """
        strategy:
          - "f1" or "best_f1": threshold maximizing F1
          - "target_recall:0.90": smallest threshold with recall >= 0.90
          - "target_precision:0.90": smallest threshold with precision >= 0.90
        """
        strategy = strategy.lower()
        prec, rec, thr = precision_recall_curve(y_true, y_proba)
        thr = np.append(thr, 1.0)  # align lengths with (prec, rec)

        if strategy in {"f1", "best_f1"}:
            f1s = 2 * (prec * rec) / np.clip(prec + rec, 1e-9, None)
            best_idx = int(np.nanargmax(f1s))
            return float(thr[best_idx])

        if strategy.startswith("target_recall:"):
            target = float(strategy.split(":")[1])
            # choose lowest threshold that achieves target recall
            idx = np.where(rec >= target)[0]
            return float(thr[idx[0]]) if len(idx) else 1.0

        if strategy.startswith("target_precision:"):
            target = float(strategy.split(":")[1])
            idx = np.where(prec >= target)[0]
            return float(thr[idx[0]]) if len(idx) else 1.0

        # fallback
        return 0.5

    def get_feature_importance(self, df_sample: Optional[pd.DataFrame] = None, n_repeats: int = 5, random_state: int = 42) -> pd.DataFrame:
        """
        If a base RF exists, return its impurity importances.
        If not (or additionally), and df_sample is provided, return permutation importances (more comparable across models).
        """
        rows = []

        # Tree-based (fast)
        if self.base_model is not None and hasattr(self.base_model, "feature_importances_"):
            rows.extend([
                {"feature": f, "importance": float(w), "kind": "rf_impurity"}
                for f, w in zip(self.feature_columns, self.base_model.feature_importances_)
            ])

        # Permutation (optional, slower)
        if df_sample is not None:
            if "y_boundary" not in df_sample.columns:
                raise ValueError("df_sample must include 'y_boundary' for permutation importance.")
            X = self._prepare_matrix(df_sample.copy(), self.feature_columns)
            y = df_sample["y_boundary"].astype(int).values
            # sklearn permutation importance expects the original feature matrix; we wrap predict_proba
            def _predict_proba(X_arr):
                # create a temporary DF to respect column order on perturbation
                tmp = pd.DataFrame(X_arr, columns=self.feature_columns)
                return self.model.predict_proba(tmp.values)[:, 1]

            # Workaround: sklearn's permutation_importance works with estimators; we provide a lambda
            # Simplify by building a shallow wrapper
            class _Wrapper:
                def __init__(self, predict_fn): self.predict_fn = predict_fn
                def predict(self, X_): 
                    # convert probabilities to class for importance; or pass proba into scorer (default uses score method)
                    # We'll instead use estimator with score method:
                    from sklearn.metrics import roc_auc_score
                    proba = self.predict_fn(X_)
                    # Return a pseudo "score" per sample isn't supported; permutation_importance calls score(X, y)
                    # So we implement score:
                    self._y = None
                    return proba
                def score(self, X_, y_):
                    from sklearn.metrics import roc_auc_score
                    proba = self.predict_fn(X_)
                    return roc_auc_score(y_, proba)

            wrapper = _Wrapper(_predict_proba)
            perm = permutation_importance(wrapper, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
            rows.extend([
                {"feature": f, "importance": float(imp), "kind": "permutation_auc"}
                for f, imp in zip(self.feature_columns, perm.importances_mean)
            ])

        if not rows:
            return pd.DataFrame(columns=["feature", "importance", "kind"]).sort_values("importance", ascending=False)

        df_imp = pd.DataFrame(rows)
        return df_imp.sort_values(["kind", "importance"], ascending=[True, False]).reset_index(drop=True)

    # convenience for evaluation on a labeled frame
    def evaluate_on(self, df: pd.DataFrame, threshold: Optional[float] = None) -> Dict[str, float]:
        """Return simple metrics on a labeled DataFrame."""
        if "y_boundary" not in df.columns:
            raise ValueError("Evaluation DataFrame must contain 'y_boundary'.")
        y_true = df["y_boundary"].astype(int).values
        y_proba = self.predict_proba(df)
        thr = self.threshold_ if threshold is None else float(threshold)
        y_pred = (y_proba >= thr).astype(int)

        # basic metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        out = {
            "threshold": thr,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
        }
        return out
