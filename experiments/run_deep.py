"""Deep baselines trained on noisy labels (CPU): FCN (Wang et al., 2017) with
cross-entropy, generalised cross-entropy (Zhang & Sabuncu, 2018) and Co-teaching
(Han et al., 2018; given the TRUE noise rate, i.e. an oracle advantage).
Same splits and noise realisations as the main experiment.
usage: python experiments/run_deep.py --datasets a,b --seeds 0,1,2 --workers 4"""
import argparse, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

EPOCHS, BATCH, LR = 100, 32, 1e-3


def fcn(C, in_ch=1):
    import torch.nn as nn
    def blk(i, o, k): return [nn.Conv1d(i, o, k, padding="same"), nn.BatchNorm1d(o), nn.ReLU()]
    return nn.Sequential(*blk(in_ch, 128, 8), *blk(128, 256, 5), *blk(256, 128, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, C))


def train(X, y, C, method, seed, rate, device):
    import torch, torch.nn.functional as Fn
    torch.manual_seed(seed); np.random.seed(seed)
    X = torch.tensor(X, dtype=torch.float32, device=device); y = torch.tensor(y, device=device)
    nets = [fcn(C, X.shape[1]).to(device) for _ in range(2 if method == "coteach" else 1)]
    opts = [torch.optim.Adam(n.parameters(), lr=LR) for n in nets]
    n = len(y); bs = min(BATCH, max(8, n // 4))
    for ep in range(EPOCHS):
        for n_ in nets: n_.train()
        perm = torch.randperm(n, device=device)
        keep_frac = 1 - rate * min(1.0, ep / 10.0)  # Co-teaching schedule (T_k = 10)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2: continue
            xb, yb = X[idx], y[idx]
            if method == "ce":
                loss = [Fn.cross_entropy(nets[0](xb), yb)]
            elif method == "gce":
                p = Fn.softmax(nets[0](xb), 1).gather(1, yb[:, None]).clamp_min(1e-7)
                loss = [((1 - p ** 0.7) / 0.7).mean()]
            else:
                o1, o2 = nets[0](xb), nets[1](xb)
                l1, l2 = Fn.cross_entropy(o1, yb, reduction="none"), Fn.cross_entropy(o2, yb, reduction="none")
                k = max(1, int(keep_frac * len(idx)))
                i1, i2 = torch.argsort(l1.detach())[:k], torch.argsort(l2.detach())[:k]
                loss = [l1[i2].mean(), l2[i1].mean()]
            for o, l in zip(opts, loss):
                o.zero_grad(); l.backward(); o.step()
    return nets[0].eval()


def znorm(X):
    return (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-8)


def one(args):
    d, seeds, conds, methods = args
    import torch; torch.set_num_threads(1)
    from src.training import runner
    from src.data import ucr
    from src.models import noise as nz
    from src.evaluation import metrics
    out_dir = os.path.join(ucr.ROOT, "results", "raw", "deep"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{d}.csv")
    if os.path.exists(out): return d
    rows = []
    for seed in seeds:
        Xtr, ytr, Xte, yte = ucr.resample(*ucr.load(d), seed)
        Xtr, Xte = znorm(Xtr), znorm(Xte); C = int(max(ytr.max(), yte.max()) + 1)
        for kind, rate in conds:
            yn = nz.apply_noise(kind, ytr, C, rate, runner.noise_rng(d, seed, kind, rate))
            for m in methods:
                t0 = time.time()
                net = train(Xtr, yn, C, m, seed, float(np.mean(yn != ytr)), "cpu")
                with torch.no_grad():
                    pred = torch.cat([net(torch.tensor(Xte[i:i + 256])) for i in range(0, len(Xte), 256)]).argmax(1).numpy()
                row = dict(dataset=d, transform="raw", seed=seed, noise=kind, rate=rate, method="fcn_" + m, n_train=len(ytr),
                           n_test=len(yte), n_classes=C, true_noise=float(np.mean(yn != ytr)), time_s=time.time() - t0)
                row.update(metrics.classification(yte, pred)); rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--conds", default="sym:0,sym:0.2,sym:0.4,asym:0.2,asym:0.4")
    ap.add_argument("--methods", default="ce,gce,coteach")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    conds = [(c.split(":")[0], float(c.split(":")[1])) for c in a.conds.split(",")]
    jobs = [(d, [int(s) for s in a.seeds.split(",")], conds, a.methods.split(",")) for d in a.datasets.split(",")]
    from multiprocessing import Pool
    with Pool(a.workers) as p:
        for d in p.imap_unordered(one, jobs): print(d, flush=True)
