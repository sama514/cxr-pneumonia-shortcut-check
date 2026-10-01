"""Create a tiny fake dataset with the Kaggle naming style, for testing the pipeline without downloading anything.
python scripts/make_fake_data.py fake_data"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "fake_data")
rng = np.random.default_rng(0)
for label, n_pat in [("NORMAL", 40), ("PNEUMONIA", 60)]:
    d = root / "chest_xray" / "train" / label
    d.mkdir(parents=True, exist_ok=True)
    for pid in range(n_pat):
        for k in range(1, 4):
            size = (300, 260) if label == "NORMAL" else (360, 300)  # planted shortcut
            base = 90 if label == "NORMAL" else 120
            arr = np.clip(rng.normal(base, 40, size[::-1]), 0, 255).astype("uint8")
            name = f"IM-{pid:04d}-{k:04d}.jpeg" if label == "NORMAL" else f"person{pid}_bacteria_{k}.jpeg"
            Image.fromarray(arr).save(d / name)
print("done", root)
