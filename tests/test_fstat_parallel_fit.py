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
    """``params -> (N (n,4), M_upper (n,10))`` with structure on EVERY axis.

    ``M_upper`` is the upper triangle of the identity, so ``compute_fstat``
    reduces to ``0.5 * sum(N**2)`` and F is a clean analytic function of the
    row. Deliberately depends on f0 (column 1), fdot (column 2 -- which
    tracks the Mc axis, since the Mc-basis sweep sets
    ``fdot = fdot_gr(f0, Mc)``), alpha (column 7) and the sky angle beta
    (column 8, ``arcsin(sin_delta)``): a golden built from a fixture that
    is flat along an axis cannot gate a later split that mis-orders rows
    along that axis -- a row-ordering bug there would reproduce the golden
    byte-identically. The envelope is BROAD (sigma ~ mHz, not micro-Hz) and
    spans the whole band so every hand-built peak box in ``make_peaks``
    lands on real structure, unlike the narrow-comb-scan bumps this
    fixture originally inherited from tests/test_fstat_gridfit.py.

    The alpha term uses ``cos(alpha / 2)``, not ``cos(alpha)``: stage B's
    ``alpha_ax = linspace(0, 2*pi, n_alpha)`` always samples a FULL period,
    so any function with period ``2*pi`` (or an integer fraction of it)
    gives the SAME value at the first and last alpha node -- exactly the
    degenerate-axis trap this fixture exists to catch. Halving the
    argument (period ``4*pi``) makes the two endpoints of a one-period
    grid genuinely different.
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
        fdot = p[:, 2]
        alpha = p[:, 7]
        beta = p[:, 8]  # arcsin(sin_delta)
        amp = (30.0
               + 20.0 * np.exp(-0.5 * ((f0_mHz - 9.0) / 1.5) ** 2)
               + 12.0 * np.exp(-0.5 * ((f0_mHz - 15.0) / 1.2) ** 2))
        amp = amp * (1.0 + 0.30 * np.cos(0.5 * alpha)
                          + 0.20 * np.sin(beta)
                          + 0.40 * np.tanh(fdot / 1e-16))
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
        # Every other knob run_stacked_stage_b's Mc-basis path reads, forced
        # unset so a golden can never silently pick up whatever happens to
        # be in the calling shell's environment.
        "FSTAT_PEAKS_TO_FIT": "",
        "FSTAT_MC_MIN": "",
        "FSTAT_MC_ETA": "",
        "FSTAT_N_F0": "",
        "FSTAT_N_PER_AXIS": "",
        "FSTAT_N_MC": "",
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
    """Every key present in both, byte-for-byte equal (dtype + shape too).

    A raw ``assertEqual(bytes, bytes)`` prints unittest's untruncated
    ``safe_repr`` of both blobs on failure -- a multi-megabyte message,
    twice over, for a single ``logp_grids_g*`` mismatch. Compare the bytes
    for the pass/fail decision but report a short locator on failure
    instead of the blobs themselves.
    """
    with np.load(path_a, allow_pickle=False) as a, np.load(path_b, allow_pickle=False) as b:
        tc.assertEqual(sorted(a.files), sorted(b.files),
                       f"key sets differ: {sorted(a.files)} vs {sorted(b.files)}")
        for key in sorted(a.files):
            xa, xb = np.asarray(a[key]), np.asarray(b[key])
            tc.assertEqual(xa.dtype, xb.dtype, f"{key}: dtype")
            tc.assertEqual(xa.shape, xb.shape, f"{key}: shape")
            if xa.tobytes() != xb.tobytes():
                # Locate the mismatch by BYTES (safe for NaN, where element
                # ``!=`` is true even between bit-identical values), then
                # map back to an element index for a readable message --
                # never the raw blobs, which unittest would print in full.
                itemsize = max(xa.dtype.itemsize, 1)
                va = np.frombuffer(xa.tobytes(), dtype=np.uint8).reshape(-1, itemsize)
                vb = np.frombuffer(xb.tobytes(), dtype=np.uint8).reshape(-1, itemsize)
                bad = np.flatnonzero(np.any(va != vb, axis=1))
                fa, fb = xa.reshape(-1), xb.reshape(-1)
                i = int(bad[0])
                tc.fail(
                    f"{key}: bytes differ in {bad.size} of {fa.size} "
                    f"elements; first at flat index {i}: "
                    f"{fa[i]!r} != {fb[i]!r}")


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
        with np.load(GOLDEN_SINGLE, allow_pickle=False) as single, \
                np.load(GOLDEN_GROUPED, allow_pickle=False) as grouped:
            self.assertIn("logp_grids", single.files,
                          "the single-group golden must use the LEGACY keys")
            self.assertNotIn("logp_grids", grouped.files)
            self.assertIn("group_sizes", grouped.files)
            self.assertGreaterEqual(
                len(np.asarray(grouped["group_sizes"])), 2,
                "the grouped golden must actually carry >1 Mc group; widen "
                "BAND_EDGES until the ladder splits")

    def test_the_goldens_are_non_degenerate(self):
        """A constant grid would pass a row-ordering bug byte-identically.

        The bit-identity gate is only as strong as the variation in the
        fixture: every peak box must carry structure, and every grid axis
        must matter, or a split that mis-orders rows along a flat axis
        reproduces the golden exactly.
        """
        for path in (GOLDEN_SINGLE, GOLDEN_GROUPED):
            with np.load(path, allow_pickle=False) as d:
                grids = [k for k in d.files if k.startswith("logp_grids")]
                self.assertTrue(grids, f"{path}: no grid keys")
                for key in grids:
                    g = np.asarray(d[key])
                    per_box = g.reshape(g.shape[0], -1)
                    self.assertTrue(
                        np.all(per_box.std(axis=1) > 0.0),
                        f"{path}:{key}: {int((per_box.std(axis=1) == 0).sum())} "
                        f"of {g.shape[0]} boxes are constant")
                    # A one-box group (the grouped golden's g3, group_sizes
                    # [3, 3, 5, 1]) has nothing to differ FROM -- it is a
                    # real, valuable edge case for a later per-group split
                    # to cover, not a degenerate fixture, so only require
                    # cross-box variation where there is more than one box.
                    if per_box.shape[0] > 1:
                        self.assertGreater(
                            len(np.unique(per_box.mean(axis=1))), 1,
                            f"{path}:{key}: every box has the same mean")
                    for ax in range(1, g.ndim):
                        moved = np.moveaxis(g, ax, 0)
                        self.assertFalse(
                            np.array_equal(moved[0], moved[-1]),
                            f"{path}:{key}: axis {ax} is degenerate -- a row "
                            f"mis-ordered along it would pass the gate")


if __name__ == "__main__":
    unittest.main()
