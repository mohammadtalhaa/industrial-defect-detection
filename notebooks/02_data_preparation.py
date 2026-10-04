# %% [markdown]
# # 02 — Split manifests (train/val from official train; official test untouched)

# %%
import sys
from pathlib import Path
sys.path.append("/content/drive/MyDrive/industrial-defect-detection")

from src.config import load_config
from src.data_loader import discover_samples, split_train_val, samples_to_dataframe

cfg = load_config("/content/drive/MyDrive/industrial-defect-detection/configs/config.yaml")
cfg.ensure_dirs()

train_pool = discover_samples(cfg._root / cfg.paths["data_raw"] / "train")
test_pool  = discover_samples(cfg._root / cfg.paths["data_raw"] / "test")

tr, va = split_train_val(train_pool, cfg.data["val_split"], cfg.seed)
samples_to_dataframe(tr).to_csv(cfg.artifacts_dir / "split_train.csv", index=False)
samples_to_dataframe(va).to_csv(cfg.artifacts_dir / "split_val.csv", index=False)
samples_to_dataframe(test_pool).to_csv(cfg.artifacts_dir / "split_test.csv", index=False)

print("Train:", len(tr), "Val:", len(va), "Test:", len(test_pool))