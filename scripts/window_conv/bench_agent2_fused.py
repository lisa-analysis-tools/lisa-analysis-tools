"""Agent 2 micro-benchmark: fused_log_like (instrument bases) vs legacy compute_log_like.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python bench_agent2_fused.py --config gal0_spl0_sym --walkers 8 32
"""
import argparse
import copy
import time

import numpy as np

import _use_worktree  # noqa: F401
from bench_2yr import PeakTracker, bench_case, make_bench_inputs
from fixtures import CONFIGS, make_backend
from lisatools import convolution_helpers as sw


def bench_fused(sm, inp, repeats, warmup, cp, tracker):
    from cupy.cuda import runtime

    dev = cp.cuda.Device()
    pool = tracker.pool
    pool.free_all_blocks()
    dev.synchronize()
    base = pool.used_bytes()
    free0, _ = runtime.memGetInfo()
    tracker.reset()
    times = []
    for i in range(warmup + repeats):
        dev.synchronize()
        t0 = time.perf_counter()
        co = sw.basis_coeffs(sm, inp["Soms_in_all"], inp["Sa_in_all"])
        ll = sw.fused_log_like(sm, inp["data_in_all"], inp["data_index_all"], co)
        dev.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    free1, _ = runtime.memGetInfo()
    return dict(t_median_s=float(np.median(times)), t_min_s=float(np.min(times)),
                peak_over_baseline_GB=(tracker.peak - base) / 1e9,
                device_free_drop_GB=(free0 - free1) / 1e9, ll=cp.asnumpy(ll))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="gal0_spl0_sym")
    ap.add_argument("--T-days", type=float, default=730.0)
    ap.add_argument("--walkers", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--legacy", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    import cupy as cp

    cfg = copy.deepcopy(CONFIGS[a.config])
    cfg.update(T_days=a.T_days)
    tracker = PeakTracker(cp)
    cp.cuda.set_allocator(tracker)
    sm = make_backend(cfg, "cuda12x")
    print(f"{a.config} T={a.T_days} d nf={sm.num_freqs}")
    used0 = tracker.pool.used_bytes()
    tracker.reset()
    t0 = time.perf_counter()
    B = sw.get_instrument_bases(sm)
    cp.cuda.Device().synchronize()
    print(f"  bases: build {time.perf_counter() - t0:.2f} s, resident {sum(b.nbytes for b in B) / 1e9:.3f} GB, "
          f"build peak +{(tracker.peak - used0) / 1e9:.3f} GB")
    for nw in a.walkers:
        inp = make_bench_inputs(sm, nw)
        cp.cuda.Device().synchronize()
        r = bench_fused(sm, inp, a.repeats, 1, cp, tracker)
        print(f"  walkers={nw} fused: t_med={r['t_median_s']:.4f}s t_min={r['t_min_s']:.4f}s "
              f"peak +{r['peak_over_baseline_GB']:.4f} GB dev_free_drop={r['device_free_drop_GB']:.3f} GB")
        for m in a.legacy:
            try:
                q = bench_case(sm, inp, bool(m), a.repeats, 1, cp, tracker)
                ll = cp.asnumpy(sm.compute_log_like(**inp, convolve_window=bool(m)))
                extra = f" |dll fused-legacy|={np.abs(ll - r['ll']).max():.3e} (ll~{ll[0]:.4e})" if m else ""
                print(f"  walkers={nw} legacy conv={bool(m)}: t_med={q['t_median_s']:.4f}s t_min={q['t_min_s']:.4f}s "
                      f"peak +{q['peak_over_baseline_GB']:.3f} GB dev_free_drop={q['device_free_drop_GB']:.3f} GB{extra}")
            except cp.cuda.memory.OutOfMemoryError as e:
                print(f"  walkers={nw} legacy conv={bool(m)}: OOM ({str(e)[:100]})")
                cp.get_default_memory_pool().free_all_blocks()
        del inp
        tracker.pool.free_all_blocks()


if __name__ == "__main__":
    main()
