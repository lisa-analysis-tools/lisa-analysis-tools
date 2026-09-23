"""Maximum-likelihood (Soms_d, Sa_a) from a noise-only store, vs the brick truth.

Companion to ``run_noise_6mo_1gpu.py`` / the laptop coarse runner.

    python scripts/noise/psd_ml_point.py <store_dir>

TRUTH is resolved by ``psd_truth_levels``: PSD_TRUTH="soms,saa" if set, else
the brick's own tabulated estimates refitted with the UNEQUAL-arm model at its
/ltts, else mojito-light's round injection (1.5e-11, 3e-15). Do NOT paste a
literal back in here -- the old 1.496182e-11 / 2.982412e-15 was the EQUAL-arm
fit, 0.26% / 0.59% low, and it made an accurate fit look biased (2026-09-23).
"""
import os, sys, glob, h5py, numpy as np

from lisatools.globalfit.stock.erebor.noise import psd_truth_levels

TRUTH = np.array(psd_truth_levels(
    mojito_data_path=os.environ.get("MOJITO_DATA_PATH")))
store = sys.argv[1] if len(sys.argv) > 1 else None
path = sorted(glob.glob(f"{store}/*_testing.h5"))[0]

with h5py.File(path, "r") as f:
    c = f["global_fit/sub_backend/psd/chain"][:]     # (it, temp, walker, leaf, 2)
    ll = f["global_fit/sub_backend/psd/log_like"][:]  # (it, temp, walker)
    bet = f["global_fit/sub_backend/psd/betas"][:]

# live iterations only: an unwritten row is all-zero
live = ~np.all(ch_flat := c.reshape(c.shape[0], -1) == 0, axis=1)
n = int(live.sum())
c, ll = c[:n], ll[:n]
cold = c[:, 0, :, 0, :]          # (n, walker, 2) -- beta = 1
llc = ll[:, 0, :]                # (n, walker)

it, w = np.unravel_index(np.nanargmax(llc), llc.shape)
ml = cold[it, w]
print(f"store            : {path.split('/')[-2]}")
print(f"stored iterations: {n}   (cold beta = {bet[0,0]:.3g})")
print(f"max cold lnL     : {llc[it, w]:.6g}   at iteration {it}, walker {w}")
print()
print(f"{'':<10} {'ML point':>14} {'truth':>14} {'ratio':>9} {'error':>9}")
for i, name in enumerate(("Soms_d", "Sa_a")):
    r = ml[i] / TRUTH[i]
    print(f"{name:<10} {ml[i]:14.6e} {TRUTH[i]:14.6e} {r:9.4f} {100*(r-1):+8.2f}%")

# spread of the cold ensemble at the last iteration, as a sanity band
last = cold[n - 1]
print()
print("cold ensemble, last stored iteration (8 walkers):")
for i, name in enumerate(("Soms_d", "Sa_a")):
    v = last[:, i]
    print(f"  {name:<8} median {np.median(v):.6e}  min {v.min():.6e}  "
          f"max {v.max():.6e}   median/truth {np.median(v)/TRUTH[i]:.4f}")
print()
print("cold lnL per iteration (is it still climbing?):")
for i in range(max(0, n - 12), n):
    print(f"  it {i:3d}: max {np.nanmax(llc[i]):.8g}   median {np.nanmedian(llc[i]):.8g}")
