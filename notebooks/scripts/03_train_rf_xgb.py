# %% [markdown]
# # 03 — RF & XGBoost Training on Geospatial Features
#
# Train pixel-level classifiers using DEM, slope, TWI, NDVI,
# distance-to-water, and land cover features.

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from models.rf_model import RFFloodModel
from models.xgb_model import XGBFloodModel
from train_rf_xgb import load_pixel_data, get_tile_names

with open("../configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

# %% [markdown]
# ## Load Pixel Data

# %%
train_tiles, test_tiles = get_tile_names(cfg)
print(f"Train: {len(train_tiles)} tiles, Test: {len(test_tiles)} tiles")

sen1_dir = Path(cfg["paths"]["sen1floods11_dir"])
geo_dir = cfg["paths"]["geo_features_dir"]
mask_dir = str(sen1_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand")
max_pixels = cfg["geo_features"]["pixel_subsample"]

X_train, y_train = load_pixel_data(geo_dir, mask_dir, train_tiles, max_pixels)
X_test, y_test = load_pixel_data(geo_dir, mask_dir, test_tiles, max_pixels // 4)

# %% [markdown]
# ## Train Random Forest

# %%
rf_cfg = cfg["rf"]
rf = RFFloodModel(
    n_estimators=rf_cfg["n_estimators"],
    max_depth=rf_cfg["max_depth"],
    min_samples_leaf=rf_cfg["min_samples_leaf"],
    n_jobs=rf_cfg["n_jobs"],
    random_state=rf_cfg["random_state"],
)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, rf_pred, target_names=["no-flood", "flood"]))

# %%
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, rf_pred)
ConfusionMatrixDisplay(cm, display_labels=["no-flood", "flood"]).plot(ax=ax, cmap="Blues")
ax.set_title("RF Confusion Matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Train XGBoost

# %%
xgb_cfg = cfg["xgb"]
xgb = XGBFloodModel(
    n_estimators=xgb_cfg["n_estimators"],
    max_depth=xgb_cfg["max_depth"],
    learning_rate=xgb_cfg["learning_rate"],
    subsample=xgb_cfg["subsample"],
    colsample_bytree=xgb_cfg["colsample_bytree"],
    tree_method=xgb_cfg["tree_method"],
    random_state=xgb_cfg["random_state"],
)

val_size = min(100_000, len(X_train) // 5)
xgb.fit(X_train[val_size:], y_train[val_size:],
        X_val=X_train[:val_size], y_val=y_train[:val_size])

xgb_pred = xgb.predict(X_test)
print("XGBoost Results:")
print(classification_report(y_test, xgb_pred, target_names=["no-flood", "flood"]))

# %%
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, xgb_pred)
ConfusionMatrixDisplay(cm, display_labels=["no-flood", "flood"]).plot(ax=ax, cmap="Oranges")
ax.set_title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Feature Importance (XGBoost)

# %%
feature_names = cfg["geo_features"]["features"]
importances = xgb.feature_importance()

sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 4))
plt.bar(range(len(feature_names)),
        importances[sorted_idx],
        color="coral")
plt.xticks(range(len(feature_names)),
           [feature_names[i] for i in sorted_idx],
           rotation=45, ha="right")
plt.ylabel("Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Save Models

# %%
ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
ckpt_dir.mkdir(parents=True, exist_ok=True)

rf.save(str(ckpt_dir / "rf_model.joblib"))
xgb.save(str(ckpt_dir / "xgb_model.json"))
print(f"Models saved to {ckpt_dir}")
