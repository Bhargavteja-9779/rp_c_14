import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'.')
from aeon.datasets.tsc_datasets import univariate_equal_length as UEL
from src.preprocessing.features import get_features
from src.models.ridge_loo import GramRidge, one_hot_pm
from src.models import cleaning as cl, noise
names=sorted(UEL); rng=np.random.RandomState(2026); dev=sorted(rng.choice(names,12,replace=False))
conds=[("sym",0.0),("sym",0.2),("sym",0.4),("asym",0.3),("asym",0.4)]
WIDE=np.logspace(-3,9,25)
variants=eval(open(sys.argv[1]).read())
tot={c:{v:[] for v in variants} for c in conds}; alph={c:{v:[] for v in variants} for c in conds}
for d in dev:
    f=get_features(d,0); C=int(max(f["ytr"].max(),f["yte"].max())+1)
    gs={}
    for c in conds:
        acc={v:[] for v in variants}
        for s in range(3):
            yn=noise.apply_noise(c[0],f["ytr"],C,c[1],np.random.RandomState(100+s))
            for v,(fn,kw,grid) in variants.items():
                if grid not in gs:
                    gs[grid]=GramRidge(alphas=WIDE if grid=="wide" else np.logspace(-3,3,10)).fit(f["Ftr"]); gs[grid].Kte=gs[grid].test_kernel(f["Fte"])
                g=gs[grid]; r=fn(g,yn,C,**kw); F,a=cl.final_predict(g,f["Fte"],g.Kte,r,C)
                acc[v].append((F.argmax(1)==f["yte"]).mean()); alph[c][v].append(np.log10(a))
        for v in variants: tot[c][v].append(np.mean(acc[v]))
print("variants",list(variants))
for c in conds: print(c," ".join(f"{np.mean(tot[c][v]):.4f}" for v in variants), "| log10a:"," ".join(f"{np.mean(alph[c][v]):.1f}" for v in variants))
big=[i for i,d in enumerate(dev) if d not in("DiatomSizeReduction","Rock","ShapeletSim","ToeSegmentation1","GunPoint")]
print("n>=100 only")
for c in conds: print(c," ".join(f"{np.mean(np.array(tot[c][v])[big]):.4f}" for v in variants))
