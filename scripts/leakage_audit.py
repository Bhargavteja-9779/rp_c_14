"""Leakage audit: exact and near-duplicate series within train and across the train/test boundary
(default archive split).  Writes tables/table_leakage_audit.csv."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.data import ucr
from src.training import runner

rows = []
for d in runner.all_datasets():
    Xtr, ytr, Xte, yte = ucr.load(d)
    A, B = Xtr.reshape(len(Xtr), -1).astype(np.float64), Xte.reshape(len(Xte), -1).astype(np.float64)
    ha = {a.tobytes(): i for i, a in enumerate(np.round(A, 6))}
    cross = [(ha[b.tobytes()], j) for j, b in enumerate(np.round(B, 6)) if b.tobytes() in ha]
    conflict = sum(int(ytr[i] != yte[j]) for i, j in cross)
    within = len(A) - len({a.tobytes() for a in np.round(A, 6)})
    # near duplicates: correlation > 0.9999 across the boundary (subsampled test set for large data)
    rng = np.random.RandomState(0); jb = rng.choice(len(B), min(len(B), 500), replace=False)
    za = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-12); zb = (B[jb] - B[jb].mean(1, keepdims=True)) / (B[jb].std(1, keepdims=True) + 1e-12)
    near = float(((zb @ za.T) / A.shape[1] > 0.9999).any(1).mean())
    rows.append(dict(dataset=d, n_train=len(A), n_test=len(B), exact_dup_within_train=within, exact_dup_train_test=len(cross),
                     exact_dup_label_conflict=conflict, frac_test_with_near_dup_in_train=near))
df = pd.DataFrame(rows); os.makedirs(os.path.join(ucr.ROOT, "tables"), exist_ok=True)
df.to_csv(os.path.join(ucr.ROOT, "tables", "table_leakage_audit.csv"), index=False)
print("datasets with exact train/test duplicates:", int((df.exact_dup_train_test > 0).sum()), "| total dup test series:", int(df.exact_dup_train_test.sum()),
      "| datasets with >5% near-dup test series:", int((df.frac_test_with_near_dup_in_train > 0.05).sum()))
print(df[df.exact_dup_train_test > 0][["dataset", "n_test", "exact_dup_train_test", "exact_dup_label_conflict"]].to_string(index=False))
