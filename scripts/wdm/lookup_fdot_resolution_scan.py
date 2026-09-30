"""Off-node fdot interpolation error of the n_ref lookup vs the fdot-axis step (A4 sizing).

Linear chirp, TD->WDM truth, lookup via get_wdm_coeffs (quarter_turn rule). fdot values
sit mid-way between table nodes (worst case for linear interpolation). Grid Nf=64,
Nt=128, dt=56.25 (layer_dt 3600 s); f step 0.005 df.
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import tests.test_wdm_lookup_basis_cycle as T  # noqa: E402  (reuses its truth / error helpers)
from lisatools.domains import WDMLookupTable  # noqa: E402

C = T.BasisCycleChirpTest
C.setUpClass()
df, ldt, wdm = C.wdm.layer_df, C.wdm.layer_dt, C.wdm
norm_f, m_diffs, m_ref = WDMLookupTable.apply_eps_frequency(0.005, wdm, m_ref=20, num_layers_diff=2)
for eps_fd in [float(x) for x in os.environ.get("EPS_FDS", "0.05,0.02,0.01").split(",")]:
    t0 = time.perf_counter()
    C.table = WDMLookupTable(wdm, 1, m_ref=m_ref, norm_freq_single_layer=norm_f, m_diffs=m_diffs,
                             fdot_vals=WDMLookupTable.apply_eps_fdot(eps_fd, wdm, fdot_max_factor=1.0),
                             store_path=os.path.join(tempfile.mkdtemp(), "t.h5"), batch_size_gen=64,
                             build_kind="n_ref_complex", time_layers=int(os.environ.get("TIME_LAYERS", "64")))
    tb = time.perf_counter() - t0
    errs = [C()._rel_err(f0 * df, (k + 0.5) * eps_fd * df / ldt, 0.4) for f0, k in ((18.2, 4), (15.3, 6), (20.3, 1))]
    print(f"eps_fdot={eps_fd:.3f}: fdot rows={len(C.table.fdot_vals)} build {tb:.0f}s  mid-node rel L2 err "
          + " ".join(f"{e:.2e}" for e in errs), flush=True)
