#!/usr/bin/env python
"""Per-channel-pair measured/modelled noise ratio -- is the Soms bias per-channel?

WHY THIS AND NOT THREE SINGLE-CHANNEL FITS. A literal "fit X alone" needs a
1x1 covariance path that does not exist: ``_mat3x3_det_inv``
(sensitivity.py:82) indexes ``M[1,1]``/``M[2,2]`` directly and the whole PSD
move is written against a 3x3 stack. Zeroing the off-diagonals instead would
mis-specify the likelihood, because X and Y genuinely share links and so are
correlated -- the recovered levels would absorb the neglected correlation.

This measures the thing those fits were meant to reveal, directly and with no
sampler at all: at a FIXED (Soms_d, Sa_a), form

    R_ij(f) = < d_i(f,t) conj(d_j(f,t)) >_t  /  C_ij(f)

per channel pair. R == 1 everywhere means the model describes the data. The
diagnostic is WHICH pairs depart and by how much:

  * all six flat at the same R != 1  -> a LEVEL error; one global pair of
    parameters can absorb it, so per-link fitting buys nothing.
  * XX/YY/ZZ differ from each other  -> per-channel structure; the per-link
    model (6 OMS + 6 TM, PD by construction) is worth building.
  * R varies with f within a pair    -> a SHAPE error (window/edge treatment,
    or the layer_calibrated correction), which no level re-parameterisation
    fixes.

Runs anywhere the fit builds -- GPU if one is present, CPU otherwise. The
cost is dominated by the one-time 5.9 GB L1 read and WDM transform.

    python scripts/noise/per_channel_noise_ratio.py [--soms S --saa A] [--png out.png]

Defaults to the NOISE brick's own fitted levels (the run's truth line); pass
the fitted ML point to ask "where is the FIT still wrong".
"""

import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("--soms", type=float, default=None, help="Soms_d [m]; default = brick value")
p.add_argument("--saa", type=float, default=None, help="Sa_a [m/s^2]; default = brick value")
p.add_argument("--png", default="per_channel_noise_ratio.png")
p.add_argument("--nbins", type=int, default=24, help="log-f bins for the table")
args = p.parse_args()

E = os.environ.setdefault
E("MOJITO_DATA_PATH", "/shared/data/mojito_cache")
E("DATA_PROCESSOR", "mojito")
E("NF", "1440")
E("NT", "4320")
E("TOBS_TARGET", "15552000")
E("MIN_FREQ", "4e-4")
E("MAX_FREQ", "2.5e-2")
E("EDGE_CROP_WAVELETS", "20")
E("UNEQUAL_ARM", "1")
E("UNEQUAL_ARM_STRIDE", "200")
E("WDM_PSD_METHOD", "layer_calibrated")
E("COARSE_Q", "1")            # exact-fine: no surrogate between us and the answer
E("COARSE_GPU_MODE", "off")
E("COARSE_USE_WS", "1")
E("COARSE_FIDUCIAL", "injection")
E("NWALKERS", "1")            # nothing is sampled; one walker is enough
E("NUM_ITERATIONS", "1")
E("FILE_STORE_DIR", "./gf_noise_ratio_probe/")
E("BASE_FILE_NAME", "ratio_probe")
E("HDF5_USE_FILE_LOCKING", "FALSE")
E("PROGRESS", "0")

import logging  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

import numpy as np  # noqa: E402
from lisatools.globalfit.stock import erebor  # noqa: E402


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


fit = erebor.noise_mojito(nwalkers=1)
g = fit.general
Nf, Nt, wav_dur, Tobs = fit.wdm_grid
assert g.tdi_chan == "XYZ" and g.nchannels == 3, (g.tdi_chan, g.nchannels)
assert (Nf, Nt) == (1440, 4320), (Nf, Nt)
print(f"grid Nf={Nf} Nt={Nt} Tobs={Tobs/86400:.1f} d  chan={g.tdi_chan} (TDI-2)")

fit.build()
truth = np.asarray(g.psd_injection, dtype=float)
soms = args.soms if args.soms is not None else float(truth[0])
saa = args.saa if args.saa is not None else float(truth[1])
print(f"brick truth : Soms_d={truth[0]:.6e}  Sa_a={truth[1]:.6e}")
print(f"evaluating C at: Soms_d={soms:.6e}  Sa_a={saa:.6e}\n")

# One generator step is the documented way to get the live model object.
model = None
for model, _state in fit.sample():
    break
acs = model.analysis_container_arr

# ---- the modelled covariance at (soms, saa) --------------------------------
# LISAModel stores the SQUARED levels and the covariance is linear in them:
#   C = Soms_d**2 * B_oms + Sa_a**2 * B_acc       (psdmove.py:1713-1719)
sens = acs.sens_mat if not hasattr(acs, "__len__") else acs[0].sens_mat
comp = None
for attr in ("components", "_components"):
    for c in getattr(sens, attr, []) or []:
        if hasattr(c, "_bases") and hasattr(getattr(c, "model", None), "Soms_d"):
            comp = c
            break
    if comp is not None:
        break
if comp is None:
    raise SystemExit(
        "could not locate the instrument-noise component on the sensitivity "
        f"matrix (type {type(sens).__name__}); inspect it and widen the search."
    )
comp.model.Soms_d, comp.model.Sa_a = soms, saa

# The domain settings the BASES were built against. general.domain_settings is
# None after build (the live object moved onto the sensitivity matrix), and
# _bases() dereferences settings.xp, so take the one the component itself
# cached rather than the settings-tree copy.
dset = None
for owner, attr in ((comp, "basis_settings"), (comp, "_basis_settings"),
                    (sens, "basis_settings"), (sens, "_basis_settings"),
                    (g, "domain_settings")):
    cand = getattr(owner, attr, None)
    if cand is not None and getattr(cand, "xp", None) is not None:
        dset, where = cand, f"{type(owner).__name__}.{attr}"
        break
if dset is None:
    raise SystemExit("no domain settings with an .xp found on the component, "
                     "the sensitivity matrix, or the general block.")
print(f"domain settings from {where}: {type(dset).__name__}")
C = _host(comp.base_covariance(dset))                       # (3,3,Nf_act[,Nt])
print("C shape:", C.shape)

# ---- the measured cross-power ----------------------------------------------
d = _host(acs.data_res_arr[:] if hasattr(acs, "data_res_arr")
          else acs[0].data_res_arr[:])                       # (3, Nf_act, Nt)
print("data shape:", d.shape)
if d.ndim != 3:
    raise SystemExit(f"expected (3, Nf, Nt) data; got {d.shape}")

P = np.einsum("ift,jft->ijf", d, d.conj()) / d.shape[-1]     # < d_i d_j* >_t
if C.ndim == 4:
    C = C.mean(axis=-1)

# ---- report -----------------------------------------------------------------
# WDM layer centres: layer m sits at m * df_layer, df_layer = 1/(2 Nf dt),
# and the active rows map to m = ind_min_f .. ind_max_f (sensitivity.py:1444).
df_layer = 1.0 / (2.0 * Nf * g.dt)
m0 = getattr(dset, "ind_min_f", None)
if m0 is None:
    m0 = int(round(float(os.environ["MIN_FREQ"]) / df_layer))
    print(f"(ind_min_f absent; inferring first layer {m0} from MIN_FREQ)")
f = (int(m0) + np.arange(P.shape[-1])) * df_layer
print(f"frequency axis: layers {int(m0)}..{int(m0)+P.shape[-1]-1} -> "
      f"{f[0]*1e3:.3f}-{f[-1]*1e3:.3f} mHz  (df_layer={df_layer*1e3:.4f} mHz)")

names = ["XX", "YY", "ZZ", "XY", "YZ", "ZX"]
idx = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
band = (f > 0) & np.all(np.isfinite(P.reshape(-1, P.shape[-1])), axis=0) \
       & np.all(np.isfinite(C.reshape(-1, C.shape[-1])), axis=0)

# ABSOLUTE SCALE IS NOT THE POINT and is not even well defined here: <d d*>
# and C differ by the WDM transform's own normalisation. The 2-parameter fit
# already measures the overall level. What this asks is whether the pairs
# depart from each other -- so everything is normalised to the mean of the
# three AUTO-spectra, which makes R dimensionless and scale-free.
# USE THE MEAN, NOT THE MEDIAN. <d_i d_j*> is the unbiased estimator of the
# covariance element; the MEDIAN of the per-pixel ratio is biased by the
# sampling distribution, and that bias DIFFERS between autos (|d|^2, chi^2-
# like, median/mean ~ 0.45) and crosses (a product of correlated Gaussians,
# which is signed). Comparing auto medians to cross medians is therefore not
# apples-to-apples -- it manufactures an apparent cross-spectrum deficit.
# Ratio of means, per pair, aggregated over all (f,t) pixels.
raw = {}
for nm, (i, j) in zip(names, idx):
    num, den = P[i, j][band], C[i, j][band]
    ok = np.abs(den) > 0
    ff = f[band][ok]
    r_mean = float(np.real(num[ok].sum()) / np.real(den[ok].sum()))
    # per-frequency ratio of means over the time axis, for the trend plot
    raw[nm] = (ff, np.real(num[ok]) / np.real(den[ok]), r_mean)
norm = np.mean([raw[n][2] for n in ("XX", "YY", "ZZ")])
ratios = {nm: (ff, rr / norm) for nm, (ff, rr, _) in raw.items()}
means = {nm: raw[nm][2] / norm for nm in names}

print(f"\n{'pair':>5} {'R (mean)':>10} {'R (median)':>12}   "
      "(normalised to mean of the three autos)")
for nm in names:
    print(f"{nm:>5} {means[nm]:>10.4f} {np.median(ratios[nm][1]):>12.4f}")

auto = [means[n] for n in ("XX", "YY", "ZZ")]
print(f"\nauto-spectrum spread  max/min = {max(auto)/min(auto):.4f}")
print("  (≈1.00 -> no per-channel structure; a level or shape error, and a")
print("   per-link model will NOT fix it. >1.01 -> per-link is worth building.)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK2 = "#fcfcfb", "#52514e"
    COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    for c, nm in zip(COLS, names):
        ff, rr = ratios[nm]
        bins = np.logspace(np.log10(max(ff.min(), 1e-5)), np.log10(ff.max()),
                           args.nbins + 1)
        who = np.digitize(ff, bins) - 1
        med = np.array([np.median(rr[who == k]) if np.any(who == k) else np.nan
                        for k in range(args.nbins)])
        ax.plot(np.sqrt(bins[1:] * bins[:-1]) * 1e3, med, lw=2.0, color=c, label=nm)
    ax.axhline(1.0, color=INK2, lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("frequency [mHz]")
    ax.set_ylabel("measured / modelled")
    ax.set_title("Per-channel-pair noise ratio", fontweight="bold", loc="left")
    ax.legend(ncol=6, frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.png, dpi=160, facecolor=SURF)
    print(f"\nwrote {args.png}")
except Exception as exc:  # plotting must never lose the table above
    print(f"\n(plot skipped: {exc!r})")
