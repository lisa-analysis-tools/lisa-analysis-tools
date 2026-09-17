"""Parallel F-stat epoch fit: goldens, split helpers and the fan-out gate.

CPU-only, one python process, tiny synthetic fixtures. ``call_fstat`` is the
analytic fake from ``tests/test_fstat_gridfit.py`` (no kernel, no GPU, no
sampler), so every number here is a deterministic function of the node grid
-- which is exactly what makes bit-identity a meaningful assertion.

The goldens in ``tests/data`` were captured from the SERIAL stage B before
the parallel split existed. They are the single-process regression gate:
``n_compute == 1`` must stay byte-identical forever.
"""

import contextlib
import os
import shutil
import tempfile
import unittest

import numpy as np

from lisatools.sampling import fstat_gridfit as G

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GOLDEN_SINGLE = os.path.join(DATA_DIR, "fstat_stage_b_golden_single.npz")
GOLDEN_GROUPED = os.path.join(DATA_DIR, "fstat_stage_b_golden_grouped.npz")

#: 6 sub-bands spanning a wide enough f0 range that the per-box Mc
#: requirement (~f0^(11/3)) crosses a ladder level -- that is what makes the
#: GROUPED golden actually carry more than one group.
BAND_EDGES = np.linspace(6.0e-3, 18.0e-3, 7)
TOBS = 7.776e6  # 90 d, the value the other fstat tests use


def _fake_call_fstat(counter=None, raise_after=None):
    """``params -> (N (n,4), M_upper (n,10))`` with an f0-dependent bump.

    ``M_upper`` is the upper triangle of the identity, so ``compute_fstat``
    reduces to ``0.5 * sum(N**2)`` and F is a clean analytic function of f0.
    Copied from tests/test_fstat_gridfit.py so this module never imports it
    (that module pins FSTAT_FDOT_AXIS process-wide in setUpModule).
    """
    state = {"rows": 0}

    def call(params):
        p = np.asarray(params.get() if hasattr(params, "get") else params)
        n = p.shape[0]
        if raise_after is not None and state["rows"] + n > raise_after:
            raise RuntimeError("simulated death mid-sweep")
        state["rows"] += n
        if counter is not None:
            counter["calls"] += 1
            counter["rows"] += n
        f0_mHz = p[:, 1] * 1e3
        amp = np.zeros(n)
        for c, a in ((6.5, 40.0), (9.0, 25.0), (12.0, 60.0), (16.0, 35.0)):
            amp += a * np.exp(-0.5 * ((f0_mHz - c) / 2e-3) ** 2)
        N = np.zeros((n, 4))
        N[:, 0] = np.sqrt(2.0 * np.maximum(amp, 0.0))
        M = np.zeros((n, 10))
        M[:, 0] = M[:, 4] = M[:, 7] = M[:, 9] = 1.0
        return N, M

    return call


def make_peaks():
    """A deterministic ``(K, 4)`` peak table spanning every interior band.

    Columns are ``select_comb_peaks``' contract: ``(f0_mHz, F, node_idx,
    band_idx)``. Built by hand rather than run through the comb so the box
    set -- and therefore the group structure -- is fixed by this file and
    cannot drift with a comb knob.
    """
    edges_mHz = BAND_EDGES * 1e3
    rows = []
    for bi in range(1, len(edges_mHz) - 2):  # interior bands only
        lo, hi = edges_mHz[bi], edges_mHz[bi + 1]
        for j, frac in enumerate((0.25, 0.5, 0.75)):
            f0 = lo + frac * (hi - lo)
            rows.append((f0, 10.0 + 3.0 * bi + j, 100 * bi + j, bi))
    return np.asarray(rows, dtype=float)


@contextlib.contextmanager
def stage_b_env(**overrides):
    """Pin every knob stage B reads, so a golden is reproducible."""
    env = {
        "FSTAT_BATCH": "512",
        "FSTAT_CKPT_SECS": "0",          # checkpoint every chunk
        "FSTAT_N_ALPHA": "2",
        "FSTAT_N_SINDELTA": "2",
        "FSTAT_PEAK_HALF_MHZ": "0.02",
        "FSTAT_FDOT_AXIS": "0",          # Mc basis: the documented escape
        "FSTAT_MC_GROUPING": "1",
        "FSTAT_PEAK_WEIGHTING": "fstat",
        "FSTAT_GRID_MEM_MB": "",
    }
    env.update({k: str(v) for k, v in overrides.items()})
    old = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v == "":
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_golden(tmpdir, *, grouped, sweep_runner=None):
    """Run the serial stage B into ``tmpdir``; return the stacked npz path.

    ``grouped=False`` pins ``FSTAT_N_MC`` so every box lands on one ladder
    level (one group, the legacy npz keys); ``grouped=True`` lets the auto
    criterion size each box, which the wide band grid splits into several.
    """
    cache_path = os.path.join(tmpdir, G.GRID_BASENAME)
    extra = {} if grouped else {"FSTAT_N_MC": "3"}
    kwargs = {} if sweep_runner is None else {"sweep_runner": sweep_runner}
    with stage_b_env(**extra):
        G.run_stacked_stage_b(
            _fake_call_fstat(), make_peaks(), xp=np, Tobs=TOBS,
            band_edges_hz=BAND_EDGES, mc_lims=[0.01, 1.0],
            cache_path=cache_path, fingerprint_extra="|epoch=0|gbfree=1",
            epoch=0, **kwargs)
    return cache_path.replace(".npz", "_peaks_stacked.npz")


def write_goldens(out_dir=DATA_DIR):
    """Capture both goldens from whatever stage B currently does."""
    os.makedirs(out_dir, exist_ok=True)
    for grouped, dest in ((False, GOLDEN_SINGLE), (True, GOLDEN_GROUPED)):
        d = tempfile.mkdtemp()
        try:
            shutil.copyfile(run_golden(d, grouped=grouped), dest)
            print(f"wrote {dest}")
        finally:
            shutil.rmtree(d, ignore_errors=True)


def assert_npz_identical(tc, path_a, path_b):
    """Every key present in both, byte-for-byte equal (dtype + shape too)."""
    a = np.load(path_a, allow_pickle=False)
    b = np.load(path_b, allow_pickle=False)
    tc.assertEqual(sorted(a.files), sorted(b.files),
                   f"key sets differ: {sorted(a.files)} vs {sorted(b.files)}")
    for key in sorted(a.files):
        xa, xb = np.asarray(a[key]), np.asarray(b[key])
        tc.assertEqual(xa.dtype, xb.dtype, f"{key}: dtype")
        tc.assertEqual(xa.shape, xb.shape, f"{key}: shape")
        tc.assertEqual(xa.tobytes(), xb.tobytes(), f"{key}: bytes differ")


class GoldenSerialStageBTest(unittest.TestCase):
    """The serial stage B must never move. This is the n_compute==1 gate."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_single_group_matches_the_golden(self):
        got = run_golden(self.d, grouped=False)
        assert_npz_identical(self, got, GOLDEN_SINGLE)

    def test_grouped_matches_the_golden(self):
        got = run_golden(self.d, grouped=True)
        assert_npz_identical(self, got, GOLDEN_GROUPED)

    def test_the_goldens_cover_both_npz_formats(self):
        single = np.load(GOLDEN_SINGLE, allow_pickle=False)
        grouped = np.load(GOLDEN_GROUPED, allow_pickle=False)
        self.assertIn("logp_grids", single.files,
                      "the single-group golden must use the LEGACY keys")
        self.assertNotIn("logp_grids", grouped.files)
        self.assertIn("group_sizes", grouped.files)
        self.assertGreaterEqual(
            len(np.asarray(grouped["group_sizes"])), 2,
            "the grouped golden must actually carry >1 Mc group; widen "
            "BAND_EDGES until the ladder splits")


if __name__ == "__main__":
    unittest.main()
