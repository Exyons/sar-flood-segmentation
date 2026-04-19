"""Random Forest wrapper for per-pixel flood classification.

Flattens geospatial features to (N_pixels, N_features), trains RF classifier.
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


class RFFloodModel:
    """Scikit-learn Random Forest for pixel-level flood prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_leaf: int = 10,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFFloodModel":
        """Train on flattened pixel features.

        Args:
            X: (N_pixels, N_features)
            y: (N_pixels,) binary labels
        """
        self.model.fit(X, y)
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
        probs = self.predict_proba(X)[:, 1]  # flood class prob
        return probs.reshape(H, W)

    def oob_score(self) -> float | None:
        """Return OOB score if available."""
        if hasattr(self.model, "oob_score_"):
            return self.model.oob_score_
        return None

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str) -> "RFFloodModel":
        self.model = joblib.load(path)
        return self
