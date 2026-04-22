"""Prediction fusion: combine SegFormer + RF/XGBoost outputs.

Methods:
    - weighted_average: Weighted blend of softmax probs + ML probs
    - stacking: Logistic regression on [vit_prob, rf_prob, xgb_prob]
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def weighted_average_fusion(
    vit_probs: np.ndarray,
    ml_probs: np.ndarray,
    vit_weight: float = 0.6,
    ml_weight: float = 0.4,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse via weighted average of flood probabilities.

    Args:
        vit_probs: (H, W) — SegFormer flood probability
        ml_probs: (H, W) — RF or XGBoost flood probability
        vit_weight: Weight for ViT predictions
        ml_weight: Weight for ML predictions
        threshold: Classification threshold

    Returns:
        fused_probs: (H, W) fused probabilities
        fused_pred: (H, W) binary predictions
    """
    total = vit_weight + ml_weight
    fused = (vit_weight * vit_probs + ml_weight * ml_probs) / total
    pred = (fused >= threshold).astype(np.int64)
    return fused, pred


class StackingFusion:
    """Learn fusion weights via stacking (logistic regression meta-learner)."""

    def __init__(self):
        self.meta_model = LogisticRegression(max_iter=1000)

    def fit(
        self,
        vit_probs: np.ndarray,
        rf_probs: np.ndarray,
        xgb_probs: np.ndarray,
        labels: np.ndarray,
    ) -> "StackingFusion":
        """Train meta-learner on validation predictions.

        All inputs: (N_pixels,) flattened
        """
        X = np.stack([vit_probs, rf_probs, xgb_probs], axis=1)
        self.meta_model.fit(X, labels)
        return self

    def predict(
        self,
        vit_probs: np.ndarray,
        rf_probs: np.ndarray,
        xgb_probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict using meta-learner.

        All inputs: (N_pixels,) or (H, W) — will be flattened/reshaped.

        Returns:
            fused_probs: same shape as input
            fused_pred: same shape as input
        """
        orig_shape = vit_probs.shape
        X = np.stack([
            vit_probs.flatten(),
            rf_probs.flatten(),
            xgb_probs.flatten(),
        ], axis=1)

        probs = self.meta_model.predict_proba(X)[:, 1]
        pred = self.meta_model.predict(X)

        return probs.reshape(orig_shape), pred.reshape(orig_shape)

    def weights(self) -> dict:
        """Return learned fusion weights."""
        coefs = self.meta_model.coef_[0]
        return {
            "vit_weight": coefs[0],
            "rf_weight": coefs[1],
            "xgb_weight": coefs[2],
            "intercept": self.meta_model.intercept_[0],
        }
