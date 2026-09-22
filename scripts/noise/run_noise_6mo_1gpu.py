#!/usr/bin/env python
"""6-month PSD-only fit on mojito's L1 instrument noise -- 1 GPU, EXACT-FINE.

The GPU twin of the laptop run of 2026-09-21. Everything about the model and
the grid is IDENTICAL to that run; the ONE difference is the likelihood:

    laptop   COARSE_Q=8  coarse CPU backend-replacement path  ~31 s/iteration
    here     COARSE_Q=1  the BASE PSD likelihood, exact-fine

so the pair is a clean A/B on the coarse surrogate. That matters because the
laptop run recovered (Soms_d, Sa_a) to +0.23% / +0.37% of the NOISE brick's
own values -- but the posterior is tight enough that those offsets are +4.5
sigma and +1.8 sigma. The coarse statistic is the prime suspect for a bias
that size, and it is the only thing this run changes.

Single process on ONE GPU: intended for an interactive ``salloc`` window, so
there is no sbatch header, no mpiexec and no saver rank.

    salloc ... --gres=gpu:1
    cd /shared/home/mlkatz1/lisa-analysis-tools
    python scripts/noise/run_noise_6mo_1gpu.py

Every knob below is overridable from the environment, e.g. a shorter run::

    NUM_ITERATIONS=60 python scripts/noise/run_noise_6mo_1gpu.py

There is no galfor branch and no GW sources: mojito's INSTRUMENT stream
carries no galaxy, so a foreground branch would rail at its prior edge.
"""

import os
import sys

E = os.environ.setdefault

# ---- data ------------------------------------------------------------------
E("MOJITO_DATA_PATH", "/shared/data/mojito_cache")
E("DATA_PROCESSOR", "mojito")

# ---- 180-day grid ----------------------------------------------------------
# NF/NT OVERRIDE TOBS_TARGET. ``EreborFit.wdm_grid`` returns nf*nt*dt whenever
# BOTH are set and only falls back to derive_wdm_grid(tobs_target, ...) when
# one is None -- and noise_mojito pins nf/nt. Exporting TOBS_TARGET alone
# leaves the variant's 90-day grid in place and the run silently analyses 3
# months. 1440 * 4320 * 2.5 s = 1.5552e7 s = 180 d, the same Nf/Nt the 6mo
# production arm runs.
E("NF", "1440")
E("NT", "4320")
E("TOBS_TARGET", "15552000")
E("MIN_FREQ", "4e-4")
E("MAX_FREQ", "2.5e-2")
# 20, not the GB script's 60: that larger crop exists for the sig-het taper
# (22+8 > 20 would trip the build guard), and there is no GB branch here.
E("EDGE_CROP_WAVELETS", "20")

# ---- noise model (v8 production settings) ----------------------------------
E("UNEQUAL_ARM", "1")
E("UNEQUAL_ARM_STRIDE", "200")
E("WDM_PSD_METHOD", "layer_calibrated")

# ---- THE BASE PSD LIKELIHOOD (the point of this run) -----------------------
# coarse_Q=1 => nothing coarse to score, the exact-fine path. Note the
# validator's noise-only rule (stock/erebor/noise.py): coarse_gpu_mode must be
# "off" (the GPU sidecar is all-source-only) and coarse_Q > 1 is rejected
# while gpus is set. Q=1 + GPUs is the combination that is allowed.
E("COARSE_Q", "1")
E("COARSE_GPU_MODE", "off")
E("COARSE_USE_WS", "1")
E("COARSE_FIDUCIAL", "injection")

# ---- one GPU ---------------------------------------------------------------
E("USE_GPU", "1")
E("GPUS", "0")

# ---- sampler ---------------------------------------------------------------
# The laptop run was converged by iteration 25 (cold lnL within 1.0 of final
# by 15) and the remaining 375 iterations changed nothing. 200 is already
# generous for a 2-parameter branch; drop it if the GPU cost surprises.
E("NWALKERS", "8")
E("NUM_ITERATIONS", "200")
E("FILE_STORE_DIR", "/shared/data/global_fit_output/gf_noise_6mo_1gpu_exact/")
E("BASE_FILE_NAME", "gf_noise_6mo")
E("VERBOSE", "1")
E("PROGRESS", "0")
E("HDF5_USE_FILE_LOCKING", "FALSE")

import logging  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from lisatools.globalfit.stock import erebor  # noqa: E402
from lisatools import has_backend  # noqa: E402


def cuda_backend_live() -> bool:
    """Is ANY CUDA backend actually loadable?

    ``has_backend`` takes only the VERSIONED names -- the "cuda"/"gpu"
    aliases that ``get_backend`` understands raise ValueError here -- and
    which of 11x/12x/13x a node has is not ours to assume. Try each and
    swallow the name error.
    """
    for name in ("cuda13x", "cuda12x", "cuda11x"):
        try:
            if has_backend(name):
                return True
        except Exception:
            continue
    return False
from lisatools.sensitivity import tdi_generation_from_channel  # noqa: E402

fit = erebor.noise_mojito(nwalkers=int(os.environ["NWALKERS"]))
g = fit.general

# GATE THE CONFIG BEFORE PAYING FOR THE DATA LOAD (the 5.9 GB L1 read). Each
# of these has a silent-wrong-answer failure mode rather than a crash: the
# grid one analyses 3 months and reports 6, and a TDI-generation mismatch
# fits the wrong transfer function to the right data.
Nf, Nt, wav_dur, Tobs = fit.wdm_grid
checks = {
    "tdi_chan": (g.tdi_chan, "XYZ"),
    "nchannels": (g.nchannels, 3),
    "Nf": (Nf, 1440),
    "Nt": (Nt, 4320),
    "Tobs [s]": (Tobs, 15552000.0),
    "source_types": (tuple(g.source_types), ("NOISE",)),
    "unequal_arm": (bool(g.unequal_arm), True),
    "wdm_psd_method": (g.wdm_psd_method, "layer_calibrated"),
    "coarse_gpu_mode": (g.coarse_gpu_mode, "off"),
    "coarse_Q": (int(g.coarse_Q), 1),          # <-- EXACT-FINE
    "window_tukey_alpha": (float(g.window_tukey_alpha), 0.0),
    "edge_crop_wavelets": (int(g.edge_crop_wavelets), 20),
    "use_gpu": (bool(g.use_gpu), True),
    "gpus": (list(g.gpus or []), [0]),
    # INTENT IS NOT AVAILABILITY. use_gpu/gpus are the REQUESTED settings:
    # they read True/[0] even on a node with no cupy, where the run would
    # silently fall back to numpy and take days instead of minutes. Assert
    # the backend is actually live.
    "cuda backend live": (cuda_backend_live(), True),
}

print("=" * 68)
print("6-MONTH PSD-ONLY FIT, EXACT-FINE, 1 GPU -- CONFIG GATE")
print("=" * 68)
bad = []
for name, (got, want) in checks.items():
    ok = got == want
    if not ok:
        bad.append(f"{name}: got {got!r}, want {want!r}")
    print(f"  {'OK ' if ok else 'BAD'}  {name:<20} = {got!r}")
print(f"  ---  TDI generation      = "
      f"{tdi_generation_from_channel(g.tdi_chan)} (from tdi_chan)")
print(f"  ---  branches            = {list(fit._branch_names)}")
print(f"  ---  wavelet duration    = {wav_dur:.1f} s ({wav_dur / 3600:.2f} h)")
print(f"  ---  Tobs                = {Tobs / 86400:.1f} d")
print(f"  ---  store               = {g.file_store_dir}")
if bad:
    raise SystemExit("CONFIG GATE FAILED:\n  " + "\n  ".join(bad))
print("gate passed; building (reads the 5.9 GB L1 noise brick)\n", flush=True)

fit.build()
print("\nbuild done; sampling\n", flush=True)
for _model, _state in fit.sample():
    pass
print("\ndone.", flush=True)
sys.exit(0)
