"""Agent 3: verify the 2-yr production default (galaxy_fmax=None -> M < N short circle WITH the
constant G_hi term) and a short-support plan against the legacy galaxy-only convolution.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/window_conv/check_agent3_2yr.py
"""
import _use_worktree  # noqa: F401
import copy
import numpy as np
import cupy as cp
from fixtures import CONFIGS, FIDUCIAL, make_backend
from compare import band_reldiff
from lisatools import sensitivity_window as sw

cfg = copy.deepcopy(CONFIGS["gal1_spl0_sym"]); cfg["T_days"] = 730
sm = make_backend(cfg, "cuda12x")
rng = np.random.default_rng(0)
n = 4
p = [cp.asarray(FIDUCIAL[k] * rng.uniform(0.9, 1.1, n)) for k in ("Amp", "alpha", "f_1", "kn", "f_2")]
z = cp.zeros(n)
leg = sm._compute_matrix_elements(sm.f_arr, z, z, *p, convolve_window=True)
by = dict(zip(("00", "11", "22", "01", "02", "12"), leg))
leg = [cp.asnumpy(by[k].real) for k in sw.GAL_RATIO_ORDER]
del by
cp.get_default_memory_pool().free_all_blocks()
for fmax in (None, 0.03):
    for direct in (True, False):
        st = sw.get_galaxy_setup(sm, galaxy_fmax=fmax)
        st["direct"] = direct
        gc, R = sw.galaxy_convolved_rows(sm, *p, setup=st)
        gc = cp.asnumpy(gc[:, 0]); R = cp.asnumpy(R)[0]
        full = np.zeros((n, sm.num_freqs)); full[:, : gc.shape[1]] = gc
        r = max(band_reldiff(R[k] * full, leg[k])[0] for k in range(6))
        pl = st["plan"]
        print(f"fmax={fmax} direct={direct}: M={pl['M']} N={st['N']} const_term={pl['full']} n_gal={pl['n_gal']} "
              f"-> galaxy band_reldiff vs legacy {r:.3e}")
        st["direct"] = True
