"""Generalisation to multivariate series: UEA archive (equal-length problems with n_train <= 3000 that the
transform can process on this machine), MiniRocket features, seeds 0-2."""
import argparse, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
NOISE = {"sym": [0.0, 0.2, 0.4], "asym": [0.2, 0.4], "boundary": [0.2]}
SKIP = {"InsectWingbeat", "FaceDetection", "PenDigits", "EigenWorms", "MotorImagery", "DuckDuckGeese", "PEMS-SF",
        "SpokenArabicDigits", "Phoneme", "LSST"}  # too large for a 15 GB / 4-core machine, or n_train > 3000


def one(d):
    os.environ["OMP_NUM_THREADS"] = "1"
    import threadpoolctl; threadpoolctl.threadpool_limits(1)
    from src.training import runner
    cfg = runner.load_config()
    allm = {**cfg["methods"], **cfg["methods_v2"]}
    m = {k: allm[k] for k in ("ridge_std", "ridge_wide", "loo_em_bic", "loo_em", "em_sym_T", "conf_learning", "enn")}
    try:
        runner.run_dataset(d, cfg, methods=m, seeds=[0, 1, 2], noise=NOISE, n_jobs=1,
                           out_dir=os.path.join(runner.ucr.ROOT, "results", "raw", "uea"))
        return d, "ok"
    except Exception as e:
        return d, "FAILED " + repr(e)[:200]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    from aeon.datasets.tsc_datasets import multivariate_equal_length
    names = sorted(str(d) for d in multivariate_equal_length if d not in SKIP)
    from multiprocessing import Pool
    with Pool(a.workers) as p:
        for r in p.imap_unordered(one, names): print(*r, flush=True)
