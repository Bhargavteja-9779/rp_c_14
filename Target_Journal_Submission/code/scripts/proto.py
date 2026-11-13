import sys, time, numpy as np
sys.path.insert(0,'.')
from aeon.datasets.tsc_datasets import univariate_equal_length as UEL
from src.preprocessing.features import get_features
from src.models.ridge_loo import GramRidge, one_hot_pm
from src.models import cleaning as cl, noise
names=sorted(UEL); rng=np.random.RandomState(2026); dev=sorted(rng.choice(names,12,replace=False))
print(len(names),dev)
conds=[("sym",0.0),("sym",0.2),("sym",0.4),("asym",0.3)]
variants={"ridge":(cl.none,{}),"em_full":(cl.loo_em,{}),"em_sym":(cl.loo_em,dict(transition="sym")),"em_hard":(cl.loo_em,dict(soft_targets=False)),
 "em_ins":(cl.loo_em,dict(in_sample=True)),"em_noT":(cl.loo_em,dict(use_noise_model=False)),"CL":(cl.confident_learning,{}),"kf_rel":(cl.kfold_relabel,{})}
conds=[("sym",0.0),("sym",0.2),("sym",0.4),("asym",0.3),("asym",0.4)]
tot={c:{v:[] for v in variants} for c in conds}
for d in dev:
    t=time.time(); f=get_features(d,0); C=int(max(f["ytr"].max(),f["yte"].max())+1)
    g=GramRidge().fit(f["Ftr"]); Kte=g.test_kernel(f["Fte"]); t1=time.time()-t
    line=f"{d[:22]:22s} n={len(f['ytr']):5d} C={C:2d} t={t1:5.1f}s |"
    for kind,rate in conds:
        yn=noise.apply_noise(kind,f["ytr"],C,rate,np.random.RandomState(1))
        for v,(fn,kw) in variants.items():
            r=fn(g,yn,C,**kw); F,a=cl.final_predict(g,f["Fte"],Kte,r,C)
            acc=(F.argmax(1)==f["yte"]).mean(); tot[(kind,rate)][v].append(acc)
        line+=f" {kind}{rate}: "+" ".join(f"{tot[(kind,rate)][v][-1]:.3f}" for v in variants)+" |"
print("variants",list(variants))
for c in conds: print(c," ".join(f"{np.mean(tot[c][v]):.4f}" for v in variants))
