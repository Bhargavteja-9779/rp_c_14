"""Run the main noise experiment on a list of datasets (resumable).
usage: python experiments/run_main.py [--transform minirocket] [--datasets a,b] [--workers 4]
                                      [--min-n 0 --max-n 1e9] [--quick]"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _one(args):
    d, transform, quick, threads, methods, seeds, sub = args
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    import threadpoolctl
    threadpoolctl.threadpool_limits(threads)
    from src.training import runner
    cfg = runner.load_config()
    kw = {}
    if quick:
        kw = dict(seeds=cfg["quick"]["seeds"], out_dir=os.path.join(runner.ucr.ROOT, "results", "quick", transform))
    if methods:
        allm = dict(cfg["methods"]); allm.update(cfg.get("sensitivity_methods", {})); allm.update(cfg.get("methods_v2", {})); allm.update(cfg.get("methods_v3", {}))
        kw["methods"] = {m: allm[m] for m in methods}
    if seeds:
        kw["seeds"] = seeds
    if sub:
        kw["out_dir"] = os.path.join(runner.ucr.ROOT, "results", "raw", sub)
    t = time.time()
    try:
        runner.run_dataset(d, cfg, transform=transform, n_jobs=threads, **kw)
        return d, time.time() - t, ""
    except Exception as e:
        return d, time.time() - t, repr(e)[:300]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", default="minirocket")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--min-n", type=float, default=0)
    ap.add_argument("--max-n", type=float, default=1e9)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--sub", default="", help="results/raw/<sub> output folder")
    a = ap.parse_args()
    from src.training import runner
    cfg = runner.load_config()
    names = a.datasets.split(",") if a.datasets else (cfg["quick"]["datasets"] if a.quick else runner.all_datasets())
    if a.min_n > 0 or a.max_n < 1e9:
        import pandas as pd
        sizes = pd.read_csv(os.path.join(runner.ucr.ROOT, "data", "dataset_summary.csv")).set_index("dataset")["n_train"]
        names = [d for d in names if a.min_n <= sizes[d] < a.max_n]
    methods = a.methods.split(",") if a.methods else None
    seeds = [int(x) for x in a.seeds.split(',')] if a.seeds else None
    jobs = [(d, a.transform, a.quick, a.threads, methods, seeds, a.sub) for d in names]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    try:  # a worker killed by the OS (e.g. out of memory) raises instead of hanging; the run is resumable
        with ProcessPoolExecutor(a.workers) as ex:
            for fut in as_completed([ex.submit(_one, j) for j in jobs]):
                d, t, err = fut.result()
                print(f"{d:35s} {t:8.1f}s {err}", flush=True)
    except BrokenProcessPool:
        print("WORKER_KILLED: rerun the same command to resume", flush=True)
        sys.exit(3)
