"""XGBoost wrapper for per-pixel flood classification.

Same interface as RF model. GPU-accelerated if available.
"""

import numpy as np
from pathlib import Path
from xgboost import XGBClassifier


class XGBFloodModel:
    """XGBoost classifier for pixel-level flood prediction."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        tree_method: str = "auto",
        random_state: int = 42,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method=tree_method,
            random_state=random_state,
            eval_metric="logloss",
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBFloodModel":
        """Train on flattened pixel features.

        Args:
            X: (N_pixels, N_features)
            y: (N_pixels,) binary labels
            X_val, y_val: optional validation set for early stopping
        """
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = 50

        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels. Returns (N_pixels,)."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities. Returns (N_pixels, 2)."""
        return self.model.predict_proba(X)

    def predict_tile(self, geo_features: np.ndarray) -> np.ndarray:
        """Predict on a full tile.

        Args:
            geo_features: (N_features, H, W)
        Returns:
            probs: (H, W) — flood probability
        """
        C, H, W = geo_features.shape
        X = geo_features.reshape(C, -1).T  # (H*W, C)
        probs = self.predict_proba(X)[:, 1]
        return probs.reshape(H, W)

    def feature_importance(self) -> np.ndarray:
        """Return feature importance scores."""
        return self.model.feature_importances_

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> "XGBFloodModel":
        self.model.load_model(path)
        return self
