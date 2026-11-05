"""Validate :class:`SOBBHTaylorT3TDIWaveform` (TDPyResponseWaveformBase path).

Checks:
  1. legacy PN outputs unchanged by the ``_pn_intrinsics`` refactor
     (optional; needs a ``legacy_ref.npz`` saved from the pre-refactor tree);
  2. masked/vectorized core == scalar core on the pre-merger samples;
  3. single-source XYZ vs the legacy ``ResponseWrapper`` + ``SOBBHWaveform``
     path (same orbits, ``reference_time == waveform_t0``, trimmed data start);
  4. batched ``compute_tdi_channels`` == loop of single-source calls,
     including a source that merges mid-window.

Usage::

    python scripts/sobbh/sobbh_tdpyresponse_check.py [--backend cpu|gpu] [--legacy-ref path.npz]
"""

import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="cpu")
ap.add_argument("--legacy-ref", default=None)
ap.add_argument("--src", default=None, help="import lisatools from this src dir (bypass editable finder)")
args = ap.parse_args()

if args.src is not None:
    sys.meta_path[:] = [f for f in sys.meta_path if "ScikitBuild" not in type(f).__name__]
    sys.path.insert(0, args.src)

import lisatools  # noqa: E402
from lisatools.detector import EqualArmlengthOrbits  # noqa: E402
from lisatools.domains import TDSettings  # noqa: E402
from lisatools.response.tdiconfig import TDIConfig  # noqa: E402
from lisatools.sources.sobbh import SOBBHTaylorT3TDIWaveform  # noqa: E402
from lisatools.sources.sobbh.response import get_sobbh_response_wrapper  # noqa: E402
from lisatools.sources.sobbh.waveform import (  # noqa: E402
    _pn_amp_phase_core,
    _pn_amp_phase_masked,
    _pn_intrinsics,
    waveform_generate_amp_phase,
    waveform_generate_h_plus_cross,
)

print("lisatools from", lisatools.__file__)
BACKEND = args.backend
ok = True


def to_np(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


def report(name, val, tol):
    global ok
    passed = val <= tol
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {val:.3e} (tol {tol:.0e})")


# ------------------------------------------------------------- 1. legacy
if args.legacy_ref is not None:
    print("1. legacy bit-identity")
    ref = np.load(args.legacy_ref)
    t = np.arange(0, 3.0e7, 5.0)
    srcs = [(30.0, 25.0, 0.3, -0.2, 1e8, 0.4, 0.018, 1.1), (60.0, 40.0, 0.0, 0.5, 3e8, 1.2, 0.02, 0.3)]
    for k, (m1, m2, s1, s2, D, inc, fl, ph) in enumerate(srcs):
        hp, hx, ts = waveform_generate_h_plus_cross(m1, m2, D, inc, fl, s1, s2, t, reference_phase=ph)
        A, P, _ = waveform_generate_amp_phase(m1, m2, D, fl, s1, s2, t, reference_phase=ph)
        same = all(
            np.array_equal(np.asarray(a), ref[f"{n}{k}"])
            for a, n in ((hp, "hp"), (hx, "hx"), (ts, "t"), (A, "A"), (P, "P"))
        )
        ok &= same
        print(f"  [{'PASS' if same else 'FAIL'}] source {k} exactly equal")

# --------------------------------------------------------- 2. masked core
print("2. masked core vs scalar core")
DT = 5.0


def f_low_for_tc(m1, m2, s1, s2, tc_target):
    lo, hi = 1e-3, 0.5
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        tc = float(_pn_intrinsics(m1, m2, mid, s1, s2)[-1])
        lo, hi = (mid, hi) if tc > tc_target else (lo, mid)
    return np.sqrt(lo * hi)


WAVEFORM_T0 = 2.0e5  # reference epoch (f_low defined here)
DATA_T0 = WAVEFORM_T0 + 1.0e5  # trimmed data start, mojito-like
N = 2**19  # ~30 days
# (m1, m2, s1, s2, dist[Gpc], inc, f_low, phi0, psi, ra, dec)
srcs = np.array(
    [
        [30.0, 25.0, 0.3, -0.2, 0.1, 0.4, 0.018, 1.1, 0.3, 1.0, 0.2],
        [60.0, 40.0, 0.0, 0.5, 0.3, 1.2, 0.02, 0.3, 1.9, 4.0, -0.7],
        [35.0, 30.0, 0.1, 0.1, 0.2, 2.0, 0.0, 2.2, 0.8, 2.5, 0.9],
    ]
)
# third source merges ~15 days after the data start (tc placed half a sample
# off the grid so ULP-level tc differences between paths can't flip a sample)
srcs[2, 6] = f_low_for_tc(*srcs[2, :4], DATA_T0 - WAVEFORM_T0 + 15 * 86400.0 + 0.5 * DT)
print(f"  merging source f_low = {srcs[2, 6]:.6f} Hz")

t_pn = np.arange(-2000, N) * DT + (DATA_T0 - WAVEFORM_T0)
for row in srcs:
    m1, m2, s1, s2, dist, inc, fl, ph = row[:8]
    A_s, P_s, ts = _pn_amp_phase_core(m1, m2, dist * 1e9, fl, s1, s2, t_pn, reference_phase=ph)
    A_m, P_m, mask = _pn_amp_phase_masked(m1, m2, dist * 1e9, fl, s1, s2, t_pn, reference_phase=ph)
    A_m, P_m, mask = np.asarray(A_m[0]), np.asarray(P_m[0]), np.asarray(mask[0])
    n = len(ts)
    assert mask.sum() == n and np.all(A_m[n:] == 0.0)
    # FP summation order differs between the vectorized and scalar paths:
    # compare relative to the amplitude / accumulated-phase magnitude. The
    # last samples before tc suffer catastrophic cancellation in (tc - t)
    # (rel ~1e-9 at the final sample), so they are excluded from the check.
    P_s, A_s = np.asarray(P_s), np.asarray(A_s)
    m = slice(0, n - 100) if n < len(t_pn) else slice(0, n)
    report(
        f"src m1={m1:.0f} ({n}/{len(t_pn)} pre-merger) rel |dA|, rel |dPhi|",
        max(
            np.max(np.abs(A_m[m] - A_s[m])) / np.max(np.abs(A_s)),
            np.max(np.abs(P_m[m] - P_s[m])) / np.max(np.abs(P_s)),
        ),
        1e-12,
    )

# ------------------------------------------------------- 3. vs legacy path
print(f"3. single-source XYZ vs legacy ResponseWrapper path [{BACKEND}]")
ORDER = 25
orbits_new = EqualArmlengthOrbits(force_backend=BACKEND)
orbits_old = EqualArmlengthOrbits(force_backend=BACKEND)
grid = TDSettings(N=N, dt=DT, t0=DATA_T0, force_backend=BACKEND)
gen = SOBBHTaylorT3TDIWaveform(
    waveform_t0=WAVEFORM_T0,
    data_td_settings=grid,
    tdi_generation="2nd generation",
    tdi_channels="XYZ",
    sampling_frequency=1.0 / DT,
    orbits=orbits_new,
    order=ORDER,
    buffer_time=1.0e4,
    force_backend=BACKEND,
)
legacy = get_sobbh_response_wrapper(
    Tobs=N * DT,
    dt=DT,
    t_start=DATA_T0,
    tdi_config=TDIConfig("2nd generation", force_backend=BACKEND),
    tdi_chan="XYZ",
    order=ORDER,
    t_buffer=1.0e4,
    orbits=orbits_old,
    force_backend=BACKEND,
    reference_time=WAVEFORM_T0,
)


def legacy_xyz(row):
    m1, m2, s1, s2, dist, inc, fl, ph, psi, ra, dec = row
    out = legacy(m1, m2, s1, s2, dist, inc, fl, ra, dec, psi, ph)
    return np.array([to_np(o) for o in out])


def place_on_data_grid(times, chans):
    """Put (times, chans) onto the N-sample data grid starting at DATA_T0."""
    times, chans = to_np(times), to_np(chans)
    idx = np.rint((times - DATA_T0) / DT).astype(int)
    assert np.allclose(times, DATA_T0 + idx * DT, atol=1e-6), "off data grid"
    out = np.zeros((chans.shape[0], N))
    keep = (idx >= 0) & (idx < N)
    out[:, idx[keep]] = chans[:, keep]
    return out


def mismatch(a, b):
    num = np.sum(a * b)
    return 1.0 - num / np.sqrt(np.sum(a * a) * np.sum(b * b))


EDGE = int(3.0e4 / DT)  # skip response edges of both paths
singles = []
for k, row in enumerate(srcs):
    t_new, c_new = gen.compute_tdi_channels(*row)
    new = place_on_data_grid(t_new, c_new)
    singles.append(new)
    old = legacy_xyz(row)
    sl = slice(EDGE, N - EDGE)
    mm = mismatch(new[:, sl], old[:, sl])
    rel = np.max(np.abs(new[:, sl] - old[:, sl])) / np.max(np.abs(old[:, sl]))
    report(f"src {k}: mismatch", abs(mm), 1e-8)
    print(f"        max rel diff = {rel:.3e}")

# ------------------------------------------------------- 4. batch == loop
print("4. batched compute_tdi_channels vs single-source loop")
cols = [srcs[:, i].copy() for i in range(srcs.shape[1])]
if BACKEND != "cpu":
    import cupy as cp

    cols = [cp.asarray(c) for c in cols]
t_b, c_b = gen.compute_tdi_channels(*cols)
for k in range(len(srcs)):
    batched = place_on_data_grid(t_b[k], c_b[k])
    rel = np.max(np.abs(batched - singles[k])) / np.max(np.abs(singles[k]))
    report(f"src {k}: max rel diff", rel, 1e-10)

# end-to-end domain call (FD via base __call__) just has to run & be finite
fd = gen(*srcs[0])
fd_arr = to_np(fd[0] if isinstance(fd, tuple) else fd)
finite = bool(np.all(np.isfinite(fd_arr)))
ok &= finite
print(f"  [{'PASS' if finite else 'FAIL'}] base __call__ (FD) finite, shape {fd_arr.shape}")

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
