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
import warnings
from unittest import mock

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


class SweepRunnerInjectionTest(unittest.TestCase):
    """``sweep_runner`` sees one spec per group and may return any grid."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_default_runner_is_the_serial_path(self):
        seen = []

        def runner(spec, call_fstat, *, xp):
            seen.append(spec)
            return G.run_stage_b_group(spec, call_fstat, xp=xp)

        got = run_golden(self.d, grouped=True, sweep_runner=runner)
        assert_npz_identical(self, got, GOLDEN_GROUPED)
        self.assertGreaterEqual(len(seen), 2, "one spec per Mc group")
        for gi, spec in enumerate(seen):
            self.assertEqual(spec.gi, gi)
            self.assertEqual(spec.n_boxes, spec.b - spec.a)
            self.assertEqual(spec.node_shape[0], spec.n_boxes)
            self.assertEqual(len(spec.f0_los), spec.n_boxes)
            self.assertEqual(len(spec.f0_dxs), spec.n_boxes)

    def test_sub_range_slices_boxes_and_renames_the_checkpoint(self):
        seen = []

        def runner(spec, call_fstat, *, xp):
            seen.append(spec)
            return G.run_stage_b_group(spec, call_fstat, xp=xp)

        run_golden(self.d, grouped=True, sweep_runner=runner)
        spec = seen[0]
        if spec.n_boxes < 2:
            self.skipTest("first group has a single box")
        mid = spec.a + spec.n_boxes // 2
        left = spec.sub_range(spec.a, mid, ckpt_name="stageb_g0_r0")
        right = spec.sub_range(mid, spec.b, ckpt_name="stageb_g0_r1")
        self.assertEqual(left.n_boxes + right.n_boxes, spec.n_boxes)
        self.assertEqual(left.node_shape, (left.n_boxes,) + tuple(spec.node_shape[1:]))
        np.testing.assert_array_equal(
            np.concatenate([left.f0_los, right.f0_los]), spec.f0_los)
        np.testing.assert_array_equal(
            np.concatenate([left.f0_dxs, right.f0_dxs]), spec.f0_dxs)
        self.assertEqual(left.ckpt_name, "stageb_g0_r0")
        # the axes and the basis are group-wide, never sliced
        np.testing.assert_array_equal(left.mc_ax, spec.mc_ax)
        np.testing.assert_array_equal(right.alpha_ax, spec.alpha_ax)
        self.assertEqual(left.c_t, spec.c_t)
        self.assertEqual(right.fdot_axis, spec.fdot_axis)

    def test_split_sweeps_reassemble_bit_identically(self):
        """Two half-range sweeps concatenated == the whole-group sweep.

        Self-guarding: the runner falls back to the whole-group sweep for
        any group with ``n_boxes < 2`` (``group_sizes`` is ``[3, 3, 5, 1]``
        today, so the last group takes that fallback). If the fixture's
        group structure ever became all-singleton, this test would keep
        passing while no longer exercising a split at all -- so count the
        groups that actually took the split path and require at least one.
        """
        split_count = [0]

        def runner(spec, call_fstat, *, xp):
            if spec.n_boxes < 2:
                return G.run_stage_b_group(spec, call_fstat, xp=xp)
            split_count[0] += 1
            mid = spec.a + spec.n_boxes // 2
            parts = [
                G.run_stage_b_group(
                    spec.sub_range(lo, hi, ckpt_name=f"{spec.ckpt_name}_r{i}"),
                    call_fstat, xp=xp)
                for i, (lo, hi) in enumerate(
                    ((spec.a, mid), (mid, spec.b)))
            ]
            return np.concatenate(parts, axis=0)

        got = run_golden(self.d, grouped=True, sweep_runner=runner)
        assert_npz_identical(self, got, GOLDEN_GROUPED)
        self.assertGreaterEqual(
            split_count[0], 1,
            "no group took the split path -- this test would pass "
            "vacuously if every group were a singleton")


class SplitBoxRangeTest(unittest.TestCase):
    def test_contiguous_covering_and_ordered(self):
        for total, n in ((12, 3), (13, 3), (1, 1), (5, 5), (7, 4)):
            parts = G.split_box_range(10, 10 + total, n)
            self.assertEqual(len(parts), n)
            self.assertEqual(parts[0][0], 10)
            self.assertEqual(parts[-1][1], 10 + total)
            for (a0, b0), (a1, _b1) in zip(parts, parts[1:]):
                self.assertLessEqual(a0, b0)
                self.assertEqual(b0, a1, "ranges must be contiguous")
            self.assertEqual(sum(b - a for a, b in parts), total)

    def test_sizes_differ_by_at_most_one(self):
        parts = G.split_box_range(0, 13, 4)
        widths = sorted(b - a for a, b in parts)
        self.assertEqual(widths, [3, 3, 3, 4])
        self.assertEqual(parts, [(0, 4), (4, 7), (7, 10), (10, 13)])

    def test_more_ranks_than_boxes_gives_empty_tail_ranges(self):
        parts = G.split_box_range(0, 2, 4)
        self.assertEqual(parts, [(0, 1), (1, 2), (2, 2), (2, 2)])

    def test_is_a_pure_function_of_its_arguments(self):
        self.assertEqual(G.split_box_range(3, 29, 5), G.split_box_range(3, 29, 5))

    def test_zero_parts_raises(self):
        with self.assertRaises(ValueError):
            G.split_box_range(0, 10, 0)


class StageBPartIOTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_round_trip_with_checksum(self):
        rng = np.random.default_rng(7)
        grid = rng.normal(size=(3, 4, 2, 2, 2))
        path, n_rows, sha = G.save_stage_b_part(self.d, 1, 2, grid)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(n_rows, 3)
        back = G.load_stage_b_part(self.d, 1, 2)
        self.assertEqual(back.dtype, np.float64)
        self.assertEqual(back.tobytes(), np.ascontiguousarray(grid).tobytes())
        _p2, _n2, sha2 = G.save_stage_b_part(self.d, 1, 2, grid)
        self.assertEqual(sha, sha2)

    def test_assemble_concatenates_in_rank_order_and_checks_shape(self):
        rng = np.random.default_rng(11)
        whole = rng.normal(size=(5, 4, 2, 2, 2))
        for r, (a, b) in enumerate(G.split_box_range(0, 5, 3)):
            G.save_stage_b_part(self.d, 0, r, whole[a:b])
        got = G.assemble_stage_b_group(self.d, 0, 3, whole.shape, xp=np)
        self.assertEqual(np.asarray(got).tobytes(),
                         np.ascontiguousarray(whole).tobytes())
        with self.assertRaises(RuntimeError):
            G.assemble_stage_b_group(self.d, 0, 3, (6, 4, 2, 2, 2), xp=np)

    def test_assemble_rejects_a_corrupt_partial(self):
        rng = np.random.default_rng(13)
        whole = rng.normal(size=(4, 2, 2, 2, 2))
        shas = {}
        for r, (a, b) in enumerate(G.split_box_range(0, 4, 2)):
            _p, _n, shas[r] = G.save_stage_b_part(self.d, 0, r, whole[a:b])
        G.save_stage_b_part(self.d, 0, 1, whole[2:4] + 1.0)  # tamper
        with self.assertRaises(RuntimeError):
            G.assemble_stage_b_group(self.d, 0, 2, whole.shape, xp=np, sha1s=shas)

    def test_assemble_with_zero_width_parts_matches_the_whole(self):
        """More ranks than boxes (split_box_range's own empty-tail case)."""
        rng = np.random.default_rng(17)
        whole = rng.normal(size=(2, 3, 2, 2, 2))
        for r, (a, b) in enumerate(G.split_box_range(0, 2, 4)):  # 2 boxes, 4 ranks
            G.save_stage_b_part(self.d, 0, r, whole[a:b])
        got = G.assemble_stage_b_group(self.d, 0, 4, whole.shape, xp=np)
        self.assertEqual(np.asarray(got).tobytes(),
                         np.ascontiguousarray(whole).tobytes())

    def test_assemble_raises_on_a_rank_missing_from_sha1s(self):
        """sha1s given but incomplete must raise, not silently go unverified.

        A rank absent from the dict (a lost or malformed MPI reply) is
        exactly the failure the sha1 check exists to catch -- skipping
        verification for that one rank would defeat the whole point.
        """
        rng = np.random.default_rng(19)
        whole = rng.normal(size=(4, 2, 2, 2, 2))
        shas = {}
        for r, (a, b) in enumerate(G.split_box_range(0, 4, 2)):
            _p, _n, shas[r] = G.save_stage_b_part(self.d, 0, r, whole[a:b])
        del shas[1]  # rank 1's reply was lost
        with self.assertRaises(RuntimeError) as ctx:
            G.assemble_stage_b_group(self.d, 0, 2, whole.shape, xp=np, sha1s=shas)
        self.assertIn("rank(s) [1]", str(ctx.exception))

    def test_clear_removes_only_this_group(self):
        g = np.zeros((1, 2, 2, 2, 2))
        G.save_stage_b_part(self.d, 0, 0, g)
        G.save_stage_b_part(self.d, 1, 0, g)
        G.clear_stage_b_parts(self.d, 0, 1)
        self.assertFalse(os.path.exists(G.stage_b_part_path(self.d, 0, 0)))
        self.assertTrue(os.path.exists(G.stage_b_part_path(self.d, 1, 0)))

    def test_part_names_are_cleared_by_the_existing_stageb_prefix(self):
        """ckpt_clear(parts, "stageb") must reach the per-rank checkpoints."""
        self.assertTrue(
            os.path.basename(G.stage_b_part_path(self.d, 0, 3)).startswith("stageb"))


class _FakeDeviceArray:
    """Duck-types just enough cupy to prove the validation never converts.

    cupy raises on implicit numpy conversion; the module's own ``_to_host``
    (``x.get() if hasattr(x, "get") else np.asarray(x)``) exists for
    exactly that reason. A validation that reads shapes through
    ``np.asarray`` therefore breaks every GPU production run at the cache
    write, which a CPU-only suite cannot see -- hence this stand-in.
    """

    def __init__(self, arr):
        self._arr = np.asarray(arr)
        self.shape = self._arr.shape
        self.dtype = self._arr.dtype
        self.ndim = self._arr.ndim

    def __array__(self, *args, **kwargs):
        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed. "
            "Please use .get() to construct a NumPy array explicitly.")

    def get(self):
        return self._arr


class WriteStackedNpzValidationTest(unittest.TestCase):
    """(A) write_stacked_npz must validate shapes before touching disk.

    The parallel fit feeds this function rank-ASSEMBLED concatenations, so
    a short or long partial must raise loudly rather than write a cache
    whose box axis silently disagrees with ``f0_los[a:b]``.
    """

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _common_kwargs(self, n_total):
        peaks = np.zeros((n_total, 4))
        peaks[:, 0] = np.linspace(8.0, 9.0, n_total)  # peak_f0_mHz
        peaks[:, 1] = 1.0                             # peak_F
        return dict(
            f0_los=np.linspace(8.0, 9.0, n_total),
            f0_dxs=np.full(n_total, 1e-6),
            alpha_ax=np.linspace(0.0, 2 * np.pi, 2),
            sd_ax=np.linspace(-1.0, 1.0, 2),
            grid_basis="Mc", grid_c_t=0.0, peaks=peaks,
            band_idx=np.zeros(n_total, dtype=int),
            band_edges_mHz=np.array([7.0, 10.0]),
            band_edges_hz=np.array([7.0e-3, 10.0e-3]),
        )

    def test_short_group_grid_raises_and_names_the_group(self):
        """group_sizes says [2, 3] (5 boxes); group 1's grid has only 2."""
        grids_g = [np.zeros((2, 2, 2, 2, 2)), np.zeros((2, 2, 2, 2, 2))]
        mc_ax_g = [np.linspace(0.1, 1.0, 2), np.linspace(0.1, 1.0, 2)]
        with self.assertRaises(ValueError) as ctx:
            G.write_stacked_npz(
                os.path.join(self.d, "out.npz"), grids_g=grids_g,
                mc_ax_g=mc_ax_g, group_sizes=[2, 3],
                **self._common_kwargs(5))
        msg = str(ctx.exception)
        # Distinctive substrings, not bare digits: a lone "2" or "3" would
        # match almost any message and not actually test that the TWO
        # disagreeing numbers are both named.
        self.assertIn("group 1 grid has 2 box(es)", msg)
        self.assertIn("group_sizes[1] says 3", msg)
        self.assertFalse(os.path.exists(os.path.join(self.d, "out.npz")),
                          "a rejected write must not touch disk")

    def test_total_box_count_mismatch_raises(self):
        """Both groups match group_sizes, but f0_los is one box short."""
        grids_g = [np.zeros((2, 2, 2, 2, 2)), np.zeros((3, 2, 2, 2, 2))]
        mc_ax_g = [np.linspace(0.1, 1.0, 2), np.linspace(0.1, 1.0, 2)]
        with self.assertRaises(ValueError) as ctx:
            G.write_stacked_npz(
                os.path.join(self.d, "out.npz"), grids_g=grids_g,
                mc_ax_g=mc_ax_g, group_sizes=[2, 3],
                **self._common_kwargs(4))  # f0_los/f0_dxs sized for 4, not 5
        msg = str(ctx.exception)
        self.assertIn("5 box(es) across all groups", msg)
        self.assertIn("f0_los has 4 entries", msg)

    def test_valid_shapes_write_cleanly(self):
        """The happy path must still write -- validation is not overzealous."""
        grids_g = [np.zeros((2, 2, 2, 2, 2)), np.zeros((3, 2, 2, 2, 2))]
        mc_ax_g = [np.linspace(0.1, 1.0, 2), np.linspace(0.1, 1.0, 2)]
        out = os.path.join(self.d, "out.npz")
        G.write_stacked_npz(
            out, grids_g=grids_g, mc_ax_g=mc_ax_g, group_sizes=[2, 3],
            **self._common_kwargs(5))
        self.assertTrue(os.path.exists(out))

    def test_device_array_shapes_validate_and_write_without_conversion(self):
        """A cupy-like grid must validate and write via ``.get()``, never
        ``np.asarray`` -- ``np.asarray(cupy_array)`` raises for real cupy."""
        host = [np.zeros((2, 2, 2, 2, 2)), np.ones((3, 2, 2, 2, 2))]
        grids_g = [_FakeDeviceArray(h) for h in host]
        mc_ax_g = [np.linspace(0.1, 1.0, 2), np.linspace(0.1, 1.0, 2)]
        out = os.path.join(self.d, "out.npz")
        G.write_stacked_npz(
            out, grids_g=grids_g, mc_ax_g=mc_ax_g, group_sizes=[2, 3],
            **self._common_kwargs(5))
        with np.load(out, allow_pickle=False) as d:
            np.testing.assert_array_equal(d["logp_grids_g0"], host[0])
            np.testing.assert_array_equal(d["logp_grids_g1"], host[1])

    def test_device_array_short_grid_still_raises_without_conversion(self):
        """The mismatch path must also avoid ``np.asarray`` on the grid."""
        grids_g = [_FakeDeviceArray(np.zeros((2, 2, 2, 2, 2))),
                   _FakeDeviceArray(np.zeros((2, 2, 2, 2, 2)))]  # short: want 3
        mc_ax_g = [np.linspace(0.1, 1.0, 2), np.linspace(0.1, 1.0, 2)]
        with self.assertRaises(ValueError) as ctx:
            G.write_stacked_npz(
                os.path.join(self.d, "out.npz"), grids_g=grids_g,
                mc_ax_g=mc_ax_g, group_sizes=[2, 3],
                **self._common_kwargs(5))
        self.assertIn("group 1 grid has 2 box(es)", str(ctx.exception))


def _build_fake_layout(nwalkers, n_compute):
    """One head + ``n_compute`` compute ranks (a saver fills out the world)
    over ``FakeWorld`` -- the same in-process, no-MPI stand-in ``ranks.py``
    and ``fanout.py`` are unit-tested against elsewhere. Module-scope so both
    :class:`OwnerOfTest` and :class:`GlobalReferenceTest` share it.
    """
    from lisatools.globalfit.communication import ranks as R
    from lisatools.globalfit.communication.fakecomm import FakeWorld

    world = FakeWorld(n_compute + 1)
    # ``gpu_pool`` is ``[0]`` -- ONE device -- while these fixtures ask for up
    # to ``n_compute`` compute ranks, which ``build_layout`` warns about
    # ("size-N launch on a per-node GPU pool [0] that supports only 1 compute
    # rank"). That over-subscription is deliberate and irrelevant here (no
    # device is ever touched), so swallow it rather than let every layout
    # fixture smear the warning across the test log.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = world.run(lambda r, comm: R.build_layout(
            comm, nwalkers, list(range(n_compute))))
    return out[0]


class OwnerOfTest(unittest.TestCase):
    def _layout(self, nwalkers, n_compute):
        return _build_fake_layout(nwalkers, n_compute)

    def test_maps_every_global_walker_to_its_rank_and_local_row(self):
        layout = self._layout(8, 2)
        block = layout.block
        for w in range(8):
            rank, local = layout.owner_of(w)
            w0, w1 = layout.block_of(rank)
            self.assertTrue(w0 <= w < w1)
            self.assertEqual(local, w - w0)
            self.assertTrue(0 <= local < block)

    def test_single_compute_rank_is_the_identity(self):
        layout = self._layout(6, 1)
        for w in range(6):
            self.assertEqual(layout.owner_of(w), (layout.head_rank, w))

    def test_out_of_range_raises(self):
        layout = self._layout(4, 2)
        with self.assertRaises(ValueError):
            layout.owner_of(4)
        with self.assertRaises(ValueError):
            layout.owner_of(-1)


class _StubFanout:
    """Head-side fake -- just enough of ``WalkerFanout`` for
    ``_fstat_global_reference``: a ``layout``, an ``is_head`` flag, and
    ``gather_likelihood`` returning a fixed per-walker vector (or raising,
    to exercise the never-silent fallback).
    """

    def __init__(self, layout, lls, is_head=True, raises=None):
        self.layout = layout
        self.is_head = is_head
        self._lls = lls
        self._raises = raises

    def gather_likelihood(self, acs):
        if self._raises is not None:
            raise self._raises
        return np.asarray(self._lls, dtype=float)


class _DummyModel:
    """Stand-in ``model``: ``_StubFanout.gather_likelihood`` ignores ``acs``
    entirely, but ``_fstat_global_reference`` still reads the attribute off
    ``model`` before handing it to the fan-out."""

    analysis_container_arr = None


class GlobalReferenceTest(unittest.TestCase):
    """``GBSpecialBase._fstat_global_reference`` -- the helper this task
    exists to deliver -- as a pure head-side function: a bare instance via
    ``__new__`` (skips the heavy GPU/eryn ``__init__``) paired with
    :class:`_StubFanout`, covering all four of its branches.
    """

    def _move(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move.name = "gb_test"
        return move

    def test_multi_rank_global_argmax_is_not_the_heads_local_one(self):
        """The max sits in the SECOND block (not the head's own, [0, 4)):
        the defect this task fixes -- an argmax over the head's local block
        only -- would pick an index below 4 here and fail every assertion
        below."""
        layout = _build_fake_layout(8, 2)
        lls = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 50.0, 8.0]  # global max at 6
        move = self._move()
        move.fanout = _StubFanout(layout, lls)
        w_global, owner_rank, local_index, lls_out = move._fstat_global_reference(
            _DummyModel())
        self.assertEqual(w_global, 6)
        self.assertEqual(owner_rank, layout.compute_ranks[1])
        w0, _w1 = layout.block_of(owner_rank)
        self.assertEqual(local_index, 6 - w0)
        np.testing.assert_array_equal(lls_out, lls)

    def test_single_compute_rank_matches_todays_local_pick(self):
        """The hard spec constraint: with ONE compute rank this must return
        exactly what ``_fstat_reference_walker`` picks today, same local
        index -- the path every current production run takes."""
        layout = _build_fake_layout(6, 1)
        lls = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]  # max at 5
        move = self._move()
        move.fanout = _StubFanout(layout, lls)
        w_global, owner_rank, local_index, lls_out = move._fstat_global_reference(
            _DummyModel())
        self.assertEqual((w_global, owner_rank, local_index), (5, layout.head_rank, 5))
        np.testing.assert_array_equal(lls_out, lls)

    def test_no_fanout_delegates_to_the_local_reference_walker(self):
        """``_propose_legacy`` / single-process ``fit.sample()``: no
        ``fanout`` attribute at all -- must fall back to
        ``_fstat_reference_walker`` and report the all-NaN length-1 ``lls``
        the brief specifies for the fallback path."""
        move = self._move()
        move._fstat_reference_walker = lambda model: 3  # instance-only stub
        w_global, owner_rank, local_index, lls = move._fstat_global_reference(
            _DummyModel())
        self.assertEqual((w_global, owner_rank, local_index), (3, 0, 3))
        self.assertEqual(lls.size, 1)
        self.assertTrue(np.isnan(lls).all())

    def test_ranking_exception_falls_back_to_walker_0_with_a_warning(self):
        """A broken collective must not fail silently (logs a WARNING) and
        must not raise -- and must name walker 0's actual owner
        (``layout.owner_of(0)``), not just ``layout.head_rank``."""
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        layout = _build_fake_layout(8, 2)
        move = self._move()
        move.fanout = _StubFanout(layout, lls=None, raises=RuntimeError("boom"))
        with self.assertLogs(gbs.logger, level="WARNING") as captured:
            w_global, owner_rank, local_index, lls = move._fstat_global_reference(
                _DummyModel())
        self.assertEqual(w_global, 0)
        self.assertEqual((owner_rank, local_index), layout.owner_of(0))
        self.assertEqual(lls.size, 1)
        self.assertTrue(np.isnan(lls).all())
        self.assertTrue(any("could not rank walkers" in line for line in captured.output))


class FakeCommBcastTest(unittest.TestCase):
    """``FakeComm.Bcast`` -- the uppercase, IN-PLACE buffer broadcast.

    The F-stat reference row pair is tens of MB, which mpi4py moves through
    the uppercase buffer form (no pickling) into a receive buffer the caller
    allocated itself. The laptop gate runs the production op body over
    :class:`FakeWorld`, so the fake needs that entry point with the same
    in-place semantics -- a ``bcast``-shaped stand-in that RETURNED the
    payload would let a body that forgot to use the return value pass here
    and lose every row on the cluster.
    """

    def test_buffer_broadcast_fills_every_rank_in_place(self):
        from lisatools.globalfit.communication.fakecomm import FakeWorld

        world = FakeWorld(3)

        def body(rank, comm):
            buf = np.zeros(4, dtype=np.float64)
            if rank == 1:
                buf[:] = [1.5, 2.5, 3.5, 4.5]
            comm.Bcast(buf, root=1)
            return buf.copy()

        out = world.run(body)
        for rank in range(3):
            np.testing.assert_array_equal(out[rank], [1.5, 2.5, 3.5, 4.5])

    def test_dtype_and_shape_are_preserved(self):
        from lisatools.globalfit.communication.fakecomm import FakeWorld

        world = FakeWorld(2)

        def body(rank, comm):
            buf = np.zeros((2, 3), dtype=np.float64)
            if rank == 0:
                buf[:] = np.arange(6, dtype=np.float64).reshape(2, 3)
            comm.Bcast(buf, root=0)
            return buf.copy()

        out = world.run(body)
        np.testing.assert_array_equal(
            out[1], np.arange(6, dtype=np.float64).reshape(2, 3))

    def test_complex_rows_survive_the_broadcast(self):
        """The residual row is COMPLEX in the FD basis.

        ``AnalysisContainerArray.data_dtype`` is ``float`` only under
        ``WDMSettings``; in FD it is ``complex``, and ``noise_dtype`` is
        complex whenever the sensitivity matrix is. A transport (or a
        sender-side cast) that assumed float64 would discard the imaginary
        part of every sample, so the fake has to move complex buffers
        faithfully.
        """
        from lisatools.globalfit.communication.fakecomm import FakeWorld

        world = FakeWorld(3)
        want = np.array([1 + 2j, -3 + 0.5j, 0 - 7j], dtype=np.complex128)

        def body(rank, comm):
            buf = np.zeros(3, dtype=np.complex128)
            if rank == 2:
                buf[:] = want
            comm.Bcast(buf, root=2)
            return buf.copy()

        out = world.run(body)
        for rank in range(3):
            self.assertEqual(out[rank].dtype, np.complex128)
            np.testing.assert_array_equal(out[rank], want)

    def test_a_strided_receive_buffer_is_refused(self):
        """A non-contiguous buffer must RAISE, not silently drop the payload.

        ``arr.reshape(-1)`` is a VIEW only for a C-contiguous array; on a
        strided one it COPIES, so the in-place write would land in a
        temporary and the rank would keep its old values with no error at
        all. Everything the production body broadcasts is a fresh
        ``np.empty``, but a silent wrong answer is the wrong failure mode to
        leave lying in a test harness.
        """
        from lisatools.globalfit.communication.fakecomm import FakeWorld

        world = FakeWorld(2)

        def body(rank, comm):
            buf = np.zeros((2, 4), dtype=np.float64)[:, ::2]  # strided view
            if rank == 0:
                buf[:] = 1.0
            comm.Bcast(buf, root=0)
            return buf.copy()

        with self.assertRaises(RuntimeError) as ctx:
            world.run(body)
        self.assertIn("C-contiguous", str(ctx.exception))


class RefRowOpTest(unittest.TestCase):
    """``gb_fstat_ref_row`` replicates the owner's rows to every rank."""

    def test_op_is_registered(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        self.assertIn("gb_fstat_ref_row", gbs.GB_OPS)
        self.assertIn("gb_fstat_stage_b", gbs.GB_OPS)

    def test_unknown_op_still_raises(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move.name = "gb_test"
        with self.assertRaises(ValueError):
            move.gf_serve("not_an_op", None, {}, None)

    def test_payload_is_identical_for_every_rank(self):
        """The op is SYMMETRIC: every rank must compute the same ``root``.

        The base payload is one dict shared by every rank, so none of the
        fields ``root = layout.fanout_rank(payload["owner_rank"])`` depends
        on can vary by rank -- that would have them broadcasting against
        different roots and deadlock the run. (The owner alone gets a
        ``gb_branch`` slice added on top; see
        ``test_only_the_owner_payload_carries_the_branch_slice``.)
        """
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        p = gbs.GBSpecialBase._fstat_ref_row_payload(6, 3, 2, True)
        self.assertEqual(
            p, {"walker_ref": 6, "owner_rank": 3, "local_index": 2,
                "gb_free": True})
        self.assertEqual(
            gbs.GBSpecialBase._fstat_ref_row_payload(6, 3, 2, None)["gb_free"],
            False)

    def test_branch_slice_round_trips_what_bandsorter_needs(self):
        """The shipped slice must satisfy ``BandSorter``'s actual contract.

        ``BandSorter.__init__``'s non-copy path reads exactly
        ``gb_branch.shape`` / ``.inds`` / ``.coords`` (gbbands.py :5564-5578,
        :5627) and stores the object as ``gb_branch_orig``, which nothing
        reads back -- so a bare ``Branch(coords, inds=inds)`` is enough. It
        must ALSO not look like a BandSorter to the copy-constructor branch,
        whose test is ``hasattr(gb_branch, "num_sources")``.

        The slice is the owner's BLOCK and TEMP 0 only: ``walker_inds`` is
        handed straight to ``fill_template`` as the ACA row index, so the
        walker axis has to BE the rank's block for ``walker=local_index`` to
        address the branch column and the ACA row identically.
        """
        from eryn.state import Branch
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move.branch_name = "gb"
        ntemps, nw, nleaves, ndim = 3, 8, 4, 9
        rng = np.random.default_rng(23)
        coords = rng.normal(size=(ntemps, nw, nleaves, ndim))
        inds = rng.random((ntemps, nw, nleaves)) > 0.5
        blob = move._fstat_ref_branch_slice(
            {"gb": Branch(coords, inds=inds)}, 4, 8)

        self.assertEqual(blob["coords"].shape, (1, 4, nleaves, ndim))
        self.assertEqual(blob["inds"].shape, (1, 4, nleaves))
        np.testing.assert_array_equal(blob["coords"], coords[0:1, 4:8])
        np.testing.assert_array_equal(blob["inds"], inds[0:1, 4:8])
        self.assertTrue(blob["coords"].flags["C_CONTIGUOUS"])
        self.assertEqual(blob["inds"].dtype, np.bool_)

        back = move._fstat_ref_branch_from_payload(blob)
        branch = back["gb"]
        self.assertEqual(branch.shape, (1, 4, nleaves, ndim))
        np.testing.assert_array_equal(branch.coords, coords[0:1, 4:8])
        np.testing.assert_array_equal(branch.inds, inds[0:1, 4:8])
        self.assertFalse(hasattr(branch, "num_sources"),
                         "a Branch must not take BandSorter's copy path")
        # temp 0 of the slice is temp 0 of the original, for the block's rows
        np.testing.assert_array_equal(branch.coords[0, 2], coords[0, 6])

    def test_branch_slice_is_none_without_a_branch(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move.branch_name = "gb"
        self.assertIsNone(move._fstat_ref_branch_slice(None, 0, 4))
        self.assertIsNone(move._fstat_ref_branch_slice({}, 0, 4))
        self.assertIsNone(move._fstat_ref_branch_from_payload(None))

    def test_fanout_reads_replies_as_the_bare_result_dicts(self):
        """``_fanout_cmd`` yields ``{rank: result}``, not ``{rank: {"result"}}``.

        ``WalkerFanout.run`` unwraps the reply envelope itself before
        ``merge`` (``fanout.py`` :172 single / :262 multi), so indexing
        ``r["result"]`` here would ``KeyError`` on the FIRST multi-rank
        refit -- a crash no single-rank test can reach. This pins the shape
        and the owner-reply pick together.
        """
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move.name = "gb_test"
        move.branch_name = "gb"
        move.fanout = _StubFanout(_build_fake_layout(8, 2), lls=None)
        move.fanout.single = False
        seen = {}

        def fake_cmd(op, per_rank_payload, model):
            seen["op"] = op
            seen["payloads"] = {r: per_rank_payload(r, 0, 4) for r in (0, 1)}
            return ({
                0: {"rank": 0, "is_owner": False, "gb_free_opened": False,
                    "n_live": -1, "data_bytes": 40, "psd_bytes": 20},
                1: {"rank": 1, "is_owner": True, "gb_free_opened": True,
                    "n_live": 7, "data_bytes": 4_000_000,
                    "psd_bytes": 2_000_000},
            }, ("seq", 1))

        move._fanout_cmd = fake_cmd
        with self.assertLogs(gbs.logger, level="INFO") as captured:
            replies = move._fstat_ref_row_fanout(None, None, 6, 1, 2)

        self.assertEqual(seen["op"], "gb_fstat_ref_row")
        self.assertEqual(set(replies), {0, 1})
        line = "\n".join(captured.output)
        # the OWNER's numbers, not rank 0's: n_live 7 and 4.0 MB
        self.assertIn("7 cold GB signal(s)", line)
        self.assertIn("4.0 MB residual", line)
        self.assertIn("2.0 MB invC", line)
        self.assertIn("GB-free window OPEN", line)
        # branches=None -> nothing was requested -> no false alarm
        self.assertFalse(any("did NOT open" in x for x in captured.output))

    def test_a_requested_but_unopened_gb_free_window_warns_on_the_head(self):
        """The safety net: the head must SAY SO when the owner reports the
        window never opened. Its own INFO line is the only thing an operator
        reads, and ``n_live == 0`` is indistinguishable from a correctly
        disabled window."""
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move.name = "gb_test"
        move.branch_name = "gb"
        move.fanout = _StubFanout(_build_fake_layout(8, 2), lls=None)
        move.fanout.single = False
        move._fanout_cmd = lambda op, prp, model: ({
            1: {"rank": 1, "is_owner": True, "gb_free_opened": False,
                "n_live": 0, "data_bytes": 8, "psd_bytes": 8},
        }, None)
        move._fstat_ref_branch_slice = lambda branches, a, b: {"coords": 1}

        with mock.patch.dict(os.environ, {"GB_FSTAT_GB_FREE": "1"}), \
                self.assertLogs(gbs.logger, level="WARNING") as captured:
            move._fstat_ref_row_fanout(None, {"gb": object()}, 6, 1, 2)
        self.assertTrue(any("did NOT open" in x for x in captured.output))

    def test_only_the_owner_payload_carries_the_branch_slice(self):
        """Per-rank payload, but identical in every field the COLLECTIVE uses.

        Every rank derives ``root`` from ``owner_rank``, so those fields must
        match everywhere or the ranks broadcast against different roots and
        hang. The branch slice is megabytes and only the owner reads it.
        """
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move.name = "gb_test"
        move.branch_name = "gb"
        move.fanout = _StubFanout(_build_fake_layout(8, 2), lls=None)
        move.fanout.single = False
        move._fstat_ref_branch_slice = lambda branches, a, b: {"block": (a, b)}
        grabbed = {}

        def fake_cmd(op, per_rank_payload, model):
            grabbed.update({r: per_rank_payload(r, 0, 4)
                            for r in move.fanout.layout.compute_ranks})
            return ({1: {"rank": 1, "is_owner": True, "gb_free_opened": True,
                         "n_live": 3, "data_bytes": 8, "psd_bytes": 8}}, None)

        move._fanout_cmd = fake_cmd
        with mock.patch.dict(os.environ, {"GB_FSTAT_GB_FREE": "1"}):
            move._fstat_ref_row_fanout(None, {"gb": object()}, 6, 1, 2)

        owner = grabbed.pop(1)
        self.assertIn("gb_branch", owner)
        self.assertEqual(owner["gb_branch"], {"block": (4, 8)})  # owner's block
        for rank, p in grabbed.items():
            self.assertNotIn("gb_branch", p, f"rank {rank} got the slice")
            for key in ("walker_ref", "owner_rank", "local_index", "gb_free"):
                self.assertEqual(p[key], owner[key], f"rank {rank}: {key}")

    def test_wire_dtype_never_crosses_the_real_complex_boundary(self):
        """``_bcast_dtype`` normalizes WIDTH, never realness.

        The FD residual row is complex; casting it to float64 to broadcast
        would silently drop the imaginary part of every sample.
        """
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        self.assertEqual(gbs._bcast_dtype(np.zeros(2, dtype=np.float32)),
                         np.float64)
        self.assertEqual(gbs._bcast_dtype(np.zeros(2, dtype=np.float64)),
                         np.float64)
        self.assertEqual(gbs._bcast_dtype(np.zeros(2, dtype=np.complex64)),
                         np.complex128)
        self.assertEqual(gbs._bcast_dtype(np.zeros(2, dtype=np.complex128)),
                         np.complex128)


class _FakeAcs:
    """Minimal single-shard ACA stand-in (no devices, nothing to slice)."""

    gpus = None

    def __init__(self):
        self.linear_data_arr = [object()]
        self.linear_psd_arr = [object()]


class RefRowStatusWordTest(unittest.TestCase):
    """The owner's failure reaches every rank as an ERROR, never as a hang.

    ``Important 1``'s whole mechanism, over a real ``FakeWorld``: the owner
    catches its own failure, broadcasts a status-0 header, and every rank
    raises off that header one collective in. A regression would show up
    here as a ``TimeoutError`` from ``FakeWorld.run`` -- i.e. exactly the
    permanent, silent cluster hang, made observable.
    """

    def _move(self, fr, comm, layout, owner_index, *, bind_raises=False,
              window_raises=False):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move.name = "gb_test"
        # ``xp`` and ``backend`` are read-only properties deriving from
        # ``_backend_name`` (the deepcopy-safety rule: never stash an array
        # module, or a backend object, on an instance).
        move._backend_name = "lisatools_cpu"
        move.branch_name = "gb"
        acs = _FakeAcs()

        def bind(model):
            if bind_raises and fr == owner_index:
                raise RuntimeError("device pin blew up")
            return acs

        @contextlib.contextmanager
        def window(model, branches, walker_ref):
            if window_raises and fr == owner_index:
                raise RuntimeError("BandSorter build blew up")
            move._gb_free_opened = True
            move._gb_free_n_live = 2
            yield

        move._bind_rank_acs = bind
        move._gb_free_residual = window
        move.fanout = _StubFanout(layout, lls=None)
        # The fan-out communicator's rank order IS ``compute_ranks`` order,
        # so FakeWorld rank ``fr`` stands in for ``compute_ranks[fr]``.
        move.fanout.rank = layout.compute_ranks[fr]
        move.fanout.comm = comm
        move.fanout.single = False
        return move

    def _run(self, **kw):
        from lisatools.globalfit.communication.fakecomm import FakeWorld

        layout = _build_fake_layout(8, 2)
        owner_index = 1                      # the SECOND compute rank owns it
        owner_rank = layout.compute_ranks[owner_index]
        world = FakeWorld(2)

        def body(fr, comm):
            move = self._move(fr, comm, layout, owner_index, **kw)
            payload = move._fstat_ref_row_payload(6, owner_rank, 2, True)
            try:
                move._gb_serve_fstat_ref_row(payload, {}, None)
            except BaseException as exc:     # noqa: BLE001 - reported, not swallowed
                return ("raised", str(exc), move._fstat_ref_holder)
            return ("ok", None, move._fstat_ref_holder)

        # The bodies catch their own exception and report it, so ``run``
        # returns one row per rank instead of re-raising the lowest rank's --
        # the point of the test is that EVERY rank got out.
        return world.run(body), owner_rank

    def _assert_every_rank_aborted(self, out, owner_rank, needle):
        self.assertEqual(sorted(out), [0, 1])
        for fr, (status, message, holder) in out.items():
            self.assertEqual(status, "raised", f"rank {fr} did not raise")
            self.assertIn(needle, message, f"rank {fr}: {message}")
            self.assertIn(f"rank {owner_rank}", message)
            self.assertIsNone(holder, f"rank {fr} kept a holder anyway")

    def test_a_failed_snapshot_on_the_owner_aborts_every_rank(self):
        out, owner_rank = self._run(window_raises=True)
        self._assert_every_rank_aborted(
            out, owner_rank, "gb_fstat_ref_row aborted")

    def test_a_failed_bind_on_the_owner_aborts_every_rank(self):
        """``_bind_rank_acs`` used to sit OUTSIDE the guard, so an owner that
        failed there exited the body without broadcasting and parked every
        other rank in ``Bcast(header)`` forever. It is a failure the ROOT can
        signal, and now does."""
        out, owner_rank = self._run(bind_raises=True)
        self._assert_every_rank_aborted(
            out, owner_rank, "gb_fstat_ref_row aborted")

    def test_a_zero_length_snapshot_is_refused_on_every_rank(self):
        from lisatools.globalfit.moves import gbbands

        empty = (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64))
        with mock.patch.object(gbbands, "snapshot_ref_rows",
                               lambda *a, **k: empty):
            out, owner_rank = self._run()
        self._assert_every_rank_aborted(out, owner_rank, "EMPTY")

    def test_a_good_snapshot_reaches_every_rank(self):
        """The success path of the same fixture: proves the aborts above are
        the STATUS WORD firing, not the harness failing to work at all."""
        from lisatools.globalfit.moves import gbbands

        data = np.arange(4, dtype=np.float64) + 0.5
        psd = np.arange(6, dtype=np.float64) + 1.5
        with mock.patch.object(gbbands, "snapshot_ref_rows",
                               lambda *a, **k: (data, psd)):
            out, _owner_rank = self._run()
        for fr, (status, message, holder) in out.items():
            self.assertEqual(status, "ok", f"rank {fr}: {message}")
            np.testing.assert_array_equal(holder.linear_data_arr[0], data)
            np.testing.assert_array_equal(holder.linear_psd_arr[0], psd)

    def test_the_owner_says_so_when_a_requested_window_had_no_branch(self):
        """``gb_free`` is not decoration: it is what lets the OWNER's own log
        distinguish "the head disabled the window" from "the branch never
        reached me"."""
        from lisatools.globalfit.moves import gbbands
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        data = np.zeros(4) + 1.0
        with mock.patch.object(gbbands, "snapshot_ref_rows",
                               lambda *a, **k: (data, data)), \
                self.assertLogs(gbs.logger, level="WARNING") as captured:
            self._run()
        self.assertTrue(any("carries no GB branch" in line
                            for line in captured.output), captured.output)


class StageBPayloadTest(unittest.TestCase):
    """The shipped payload carries host arrays and reconstructs the spec."""

    def _spec(self):
        return G.StageBGroupSpec(
            gi=2, n_groups=3, a=10, b=18,
            f0_los=np.linspace(6.0, 6.7, 8),
            f0_dxs=np.full(8, 1e-3),
            mc_ax=np.linspace(0.01, 1.0, 3),
            alpha_ax=np.linspace(0.0, 2 * np.pi, 2),
            sd_ax=np.linspace(-1.0, 1.0, 2),
            node_shape=(8, 5, 3, 2, 2),
            ckpt_name="stageb_g2", parts_dir="/tmp/parts",
            fingerprint_extra="|epoch=3|gbfree=1",
            fdot_axis=False, c_t=0.0)

    def test_payload_round_trips_to_an_equivalent_sub_spec(self):
        import pickle

        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        spec = self._spec()
        payload = move._fstat_stage_b_payload(spec, 1, 13, 16)
        payload = pickle.loads(pickle.dumps(payload))   # the wire does this
        got = gbs.GBSpecialBase._fstat_stage_b_spec(payload)
        self.assertEqual(got.gi, 2)
        self.assertEqual((got.a, got.b), (13, 16))
        self.assertEqual(got.node_shape, (3, 5, 3, 2, 2))
        self.assertEqual(got.ckpt_name, "stageb_g2_r1")
        np.testing.assert_array_equal(got.f0_los, spec.f0_los[3:6])
        np.testing.assert_array_equal(got.f0_dxs, spec.f0_dxs[3:6])
        np.testing.assert_array_equal(got.mc_ax, spec.mc_ax)
        np.testing.assert_array_equal(got.alpha_ax, spec.alpha_ax)
        np.testing.assert_array_equal(got.sd_ax, spec.sd_ax)
        self.assertEqual(got.fingerprint_extra, spec.fingerprint_extra)
        self.assertEqual(got.parts_dir, spec.parts_dir)

    def test_payload_holds_no_device_arrays(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        payload = move._fstat_stage_b_payload(self._spec(), 0, 10, 13)
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                self.assertIs(type(value), np.ndarray, key)

    def test_empty_range_is_a_legal_payload(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        payload = move._fstat_stage_b_payload(self._spec(), 3, 18, 18)
        got = gbs.GBSpecialBase._fstat_stage_b_spec(payload)
        self.assertEqual(got.n_boxes, 0)
        self.assertEqual(got.node_shape[0], 0)

    def test_fstat_call_accepts_a_holder_override(self):
        """The override is KEYWORD-ONLY with a ``None`` default.

        ``_fstat_call`` lives on :class:`GBSpecialRJFStatGridMove`, not on
        ``GBSpecialBase`` (the brief's ``GBSpecialBase._fstat_call`` is a
        mis-attribution carried through Tasks 6-9); ``_fstat_NM`` genuinely
        is on the base. Positional would be worse than useless here: every
        existing call site passes ``(model, walker_ref)`` and a third
        positional slot would silently accept a stray argument as the
        holder.
        """
        import inspect

        from lisatools.globalfit.moves import gbspecialstretch as gbs

        sig = inspect.signature(gbs.GBSpecialRJFStatGridMove._fstat_call)
        self.assertIn("holder", sig.parameters)
        self.assertEqual(sig.parameters["holder"].kind,
                         inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(sig.parameters["holder"].default)
        sig_nm = inspect.signature(gbs.GBSpecialBase._fstat_NM)
        self.assertIn("holder", sig_nm.parameters)
        self.assertEqual(sig_nm.parameters["holder"].kind,
                         inspect.Parameter.KEYWORD_ONLY)


class HolderCallTest(unittest.TestCase):
    """``_fstat_holder_call``: built once per fit, against the shipped row."""

    def _move(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialRJFStatGridMove.__new__(gbs.GBSpecialRJFStatGridMove)
        move.name = "gb_test"
        return move

    def test_it_is_built_once_and_cached(self):
        """The sig-het scorer is STATEFUL -- rebuilding the closure per group
        would throw away the bucketed reference blocks the f0-sorted box
        order exists to keep."""
        move = self._move()
        move._fstat_ref_holder = object()
        calls = []

        def fake_fstat_call(model, walker_ref, *, holder=None):
            calls.append((model, walker_ref, holder))
            return lambda params: params

        move._fstat_call = fake_fstat_call
        first = move._fstat_holder_call("model")
        second = move._fstat_holder_call("model")
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 0)                 # the holder's row 0
        self.assertIs(calls[0][2], move._fstat_ref_holder)

    def test_no_holder_refuses_rather_than_scoring_row_zero(self):
        """Without the guard ``holder=None`` falls back to the LIVE ACA at
        ``walker_ref=0`` -- a different walker's residual, silently."""
        move = self._move()
        move._fstat_ref_holder = None
        move._fstat_call = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not build a scorer without a holder"))
        with self.assertRaises(RuntimeError) as ctx:
            move._fstat_holder_call("model")
        self.assertIn("gb_fstat_ref_row", str(ctx.exception))


class _StopHere(Exception):
    """Cut a runner short once the payloads have been inspected."""


_KEEP = object()


class StageBRunnerTest(unittest.TestCase):
    """The head-side ``sweep_runner``: split, fan out, assemble, clear."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _spec(self, n_boxes=7, parts_dir=_KEEP):
        return G.StageBGroupSpec(
            gi=0, n_groups=1, a=0, b=n_boxes,
            f0_los=np.linspace(6.0, 6.7, n_boxes),
            f0_dxs=np.full(n_boxes, 1e-3),
            mc_ax=np.linspace(0.01, 1.0, 2),
            alpha_ax=np.linspace(0.0, 2 * np.pi, 2),
            sd_ax=np.linspace(-1.0, 1.0, 2),
            node_shape=(n_boxes, 3, 2, 2, 2),
            ckpt_name="stageb_g0",
            parts_dir=self.tmp if parts_dir is _KEEP else parts_dir,
            fingerprint_extra="|epoch=0", fdot_axis=False, c_t=0.0)

    def _move(self, n_compute=2):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialRJFStatGridMove.__new__(gbs.GBSpecialRJFStatGridMove)
        move.name = "gb_test"
        layout = _build_fake_layout(4 * n_compute, n_compute)
        move.fanout = _StubFanout(layout, lls=None)
        move.fanout.single = (n_compute == 1)
        return move, layout

    def _serve(self, move, layout, spec, per_rank_payload, grid_of):
        """Every compute rank runs the op body's I/O half for real."""
        replies = {}
        for rank in layout.compute_ranks:
            payload = per_rank_payload(rank, 0, 0)
            sub = move._fstat_stage_b_spec(payload)
            ri = int(payload["rank_index"])
            _p, n_rows, sha = G.save_stage_b_part(
                sub.parts_dir, sub.gi, ri, grid_of(sub))
            replies[rank] = {"gi": int(sub.gi), "a": int(sub.a),
                             "b": int(sub.b), "rank_index": ri,
                             "n_rows": int(n_rows), "sha1": sha,
                             "wall_s": 1.0 + ri}
        return replies

    def test_split_partials_assemble_into_the_whole_group_grid(self):
        """The contract the bit-identity gate rests on: rank-ordered
        concatenation of contiguous box ranges IS the whole-group sweep
        (box is the slowest axis)."""
        move, layout = self._move(n_compute=2)
        spec = self._spec()
        whole = np.arange(
            int(np.prod(spec.node_shape)), dtype=float).reshape(spec.node_shape)
        grabbed = {}

        def fake_cmd(op, per_rank_payload, model):
            self.assertEqual(op, "gb_fstat_stage_b")
            grabbed["op"] = op
            replies = self._serve(
                move, layout, spec, per_rank_payload,
                lambda sub: whole[sub.a - spec.a:sub.b - spec.a])
            # ``_fanout_cmd`` returns the BARE result dicts (fanout.py:172 /
            # :262 unwrap the envelope), not ``{rank: {"result": ...}}``.
            return replies, None

        move._fanout_cmd = fake_cmd
        runner = move._fstat_stage_b_runner("model")
        with self.assertLogs("lisatools.globalfit.moves.gbspecialstretch",
                             level="INFO"):
            grid = runner(spec, None, xp=np)
        np.testing.assert_array_equal(grid, whole)
        self.assertEqual(grabbed["op"], "gb_fstat_stage_b")
        # the partials are cleared once the group is assembled
        self.assertEqual(
            [f for f in os.listdir(self.tmp) if f.endswith(".npy")], [])

    def test_every_rank_gets_its_own_contiguous_range(self):
        move, layout = self._move(n_compute=3)
        spec = self._spec(n_boxes=7)
        seen = {}

        def fake_cmd(op, per_rank_payload, model):
            for rank in layout.compute_ranks:
                p = per_rank_payload(rank, 0, 0)
                seen[int(p["rank_index"])] = (p["a"], p["b"])
            raise _StopHere()

        move._fanout_cmd = fake_cmd
        runner = move._fstat_stage_b_runner("model")
        with self.assertRaises(_StopHere):
            runner(spec, None, xp=np)
        self.assertEqual(seen, {0: (0, 3), 1: (3, 5), 2: (5, 7)})
        self.assertEqual(
            sorted(seen), list(range(layout.n_compute)))

    def test_a_reply_that_misreports_its_range_is_refused(self):
        """``assemble_stage_b_group`` checks only the TOTAL shape, so two
        ranks whose partials swapped places would assemble silently into a
        grid whose box axis disagrees with ``f0_los[a:b]``."""
        move, layout = self._move(n_compute=2)
        spec = self._spec()
        whole = np.zeros(spec.node_shape)

        def fake_cmd(op, per_rank_payload, model):
            replies = self._serve(
                move, layout, spec, per_rank_payload,
                lambda sub: whole[sub.a - spec.a:sub.b - spec.a])
            first = layout.compute_ranks[0]
            replies[first] = dict(replies[first], a=99, b=101)
            return replies, None

        move._fanout_cmd = fake_cmd
        runner = move._fstat_stage_b_runner("model")
        with self.assertRaises(RuntimeError) as ctx:
            runner(spec, None, xp=np)
        self.assertIn("box range", str(ctx.exception))

    def test_a_reply_without_a_rank_index_is_refused(self):
        move, layout = self._move(n_compute=2)
        spec = self._spec()
        whole = np.zeros(spec.node_shape)

        def fake_cmd(op, per_rank_payload, model):
            replies = self._serve(
                move, layout, spec, per_rank_payload,
                lambda sub: whole[sub.a - spec.a:sub.b - spec.a])
            replies[layout.compute_ranks[1]] = {}
            return replies, None

        move._fanout_cmd = fake_cmd
        runner = move._fstat_stage_b_runner("model")
        with self.assertRaises(RuntimeError) as ctx:
            runner(spec, None, xp=np)
        self.assertIn("rank_index", str(ctx.exception))

    def test_a_group_with_no_shared_parts_dir_is_refused(self):
        """Each rank writes its slice to the SHARED ``_parts`` dir; without
        one there is nowhere for the head to read them back from."""
        move, _layout = self._move(n_compute=2)
        spec = self._spec(parts_dir=None)
        move._fanout_cmd = lambda *a, **k: self.fail("must not fan out")
        runner = move._fstat_stage_b_runner("model")
        with self.assertRaises(RuntimeError) as ctx:
            runner(spec, None, xp=np)
        self.assertIn("parts_dir", str(ctx.exception))


class FStatOpGuardTest(unittest.TestCase):
    """A GB move that cannot run an F-stat fit must say so, not
    ``AttributeError`` halfway through a served command."""

    def test_stage_b_on_a_non_grid_move_names_the_class(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move.name = "gb_test"
        with self.assertRaises(ValueError) as ctx:
            move.gf_serve("gb_fstat_stage_b", {}, {}, None)
        self.assertIn("GBSpecialRJFStatGridMove", str(ctx.exception))

    def test_ref_row_on_a_non_grid_move_names_the_class(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move.name = "gb_test"
        with self.assertRaises(ValueError) as ctx:
            move.gf_serve("gb_fstat_ref_row", {}, {}, None)
        self.assertIn("GBSpecialRJFStatGridMove", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
