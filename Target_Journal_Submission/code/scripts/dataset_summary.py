"""Download every dataset once and write data/dataset_summary.csv (sizes, classes, imbalance)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.data import ucr
from src.training import runner
rows = []
for d in runner.all_datasets():
    try:
        Xtr, ytr, Xte, yte = ucr.load(d)
        cnt = np.bincount(ytr)
        rows.append(dict(dataset=d, n_train=len(ytr), n_test=len(yte), length=Xtr.shape[-1], n_classes=len(cnt),
                         min_class_train=int(cnt.min()), imbalance_ratio=float(cnt.max() / max(cnt.min(), 1)),
                         has_nan=bool(np.isnan(Xtr).any() or np.isnan(Xte).any())))
    except Exception as e:
        print("FAILED", d, repr(e)[:100], flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(ucr.ROOT, "data", "dataset_summary.csv"), index=False)
print(len(df), df.n_train.describe().to_dict(), "nan:", df.has_nan.sum())
