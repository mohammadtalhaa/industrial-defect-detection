# %% [markdown]
# # 03 — Training (EfficientNet-B0)
# Requires GPU runtime in Colab.

# %%
import sys
from pathlib import Path
sys.path.append("/content/drive/MyDrive/industrial-defect-detection")

# %%
!cd /content/drive/MyDrive/industrial-defect-detection && python -m src.train --config configs/config.yaml

# %% [markdown]
# Checkpoints land in `models/best_model.pt`. Artifacts (history, split CSVs) in `artifacts/`.