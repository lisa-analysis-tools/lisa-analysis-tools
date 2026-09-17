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
import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
import threading
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


def run_golden(tmpdir, *, grouped, sweep_runner=None, call_fstat=None):
    """Run the serial stage B into ``tmpdir``; return the stacked npz path.

    ``grouped=False`` pins ``FSTAT_N_MC`` so every box lands on one ladder
    level (one group, the legacy npz keys); ``grouped=True`` lets the auto
    criterion size each box, which the wide band grid splits into several.

    ``call_fstat`` defaults to a fresh :func:`_fake_call_fstat` -- the
    goldens' own scorer, and the only thing the serial path ever passes.
    The parallel gate hands in the head's HOLDER-SCORED call instead,
    because :meth:`GBSpecialBase._fstat_stage_b_runner`'s guard refuses any
    scorer that is not the rank's cached ``_fstat_ref_call``; substituting a
    twin here would make the gate pass through a hole that guard exists to
    close.
    """
    cache_path = os.path.join(tmpdir, G.GRID_BASENAME)
    extra = {} if grouped else {"FSTAT_N_MC": "3"}
    kwargs = {} if sweep_runner is None else {"sweep_runner": sweep_runner}
    with stage_b_env(**extra):
        G.run_stacked_stage_b(
            _fake_call_fstat() if call_fstat is None else call_fstat,
            make_peaks(), xp=np, Tobs=TOBS,
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
        ``merge`` (``fanout.py`` :223-225 single / :308 multi; the envelope
        is built by ``_reply``, :71), so indexing ``r["result"]`` here would
        ``KeyError`` on the FIRST multi-rank refit -- a crash no single-rank
        test can reach. This pins the shape and the owner-reply pick
        together.
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
        # A status-word regression is a HANG, and the default 60 s timeout
        # would make every such regression cost a minute per test. Five
        # seconds is ~500x the healthy run.
        world = FakeWorld(2, timeout=5.0)

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
        """Every array-shaped field must be host numpy.

        NOT ``isinstance(value, np.ndarray)`` as the filter: a
        ``cupy.ndarray`` is not an instance of ``np.ndarray``, so filtering
        on it would SKIP exactly the type this test exists to catch. Check
        everything that is not a plain scalar instead.
        """
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        payload = move._fstat_stage_b_payload(self._spec(), 0, 10, 13)
        for key, value in payload.items():
            if isinstance(value, (int, float, bool, str, tuple, type(None))):
                continue
            self.assertEqual(type(value).__module__.split(".")[0], "numpy",
                             f"{key} is a {type(value)!r}")

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


class ServedStageBBodyTest(unittest.TestCase):
    """``_gb_serve_fstat_stage_b`` itself -- the PRODUCER's contract.

    The head-side runner's checks are tested against hand-built replies; this
    drives the real body, with a real ``run_stage_b_group`` sweep over a tiny
    spec, and pins the reply dict the runner indexes by name. A typo in
    ``"rank_index"`` or ``"sha1"``, or a regression that drops the
    missing-reference-row guard (scoring the live ACA at row 0 -- the silent
    wrong-walker failure), would otherwise first appear on the cluster.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _move(self, *, holder=_KEEP, scorer=_KEEP):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialRJFStatGridMove.__new__(gbs.GBSpecialRJFStatGridMove)
        move.name = "gb_test"
        move._backend_name = "lisatools_cpu"
        move.mempool = gbs._NoOpMempool()
        move._bind_rank_acs = lambda model: None
        if holder is not _KEEP:
            move._fstat_ref_holder = holder
        if scorer is not _KEEP:
            move._fstat_holder_call = lambda model: scorer
        return move

    def _payload(self, move, rank_index=1, a=0, b=2):
        spec = G.StageBGroupSpec(
            gi=3, n_groups=4, a=0, b=2,
            f0_los=np.array([8.5, 8.6]), f0_dxs=np.full(2, 5e-4),
            mc_ax=np.linspace(0.1, 0.9, 2),
            alpha_ax=np.linspace(0.0, 2 * np.pi, 2),
            sd_ax=np.linspace(-1.0, 1.0, 2),
            node_shape=(2, 3, 2, 2, 2), ckpt_name="stageb_g3",
            parts_dir=self.tmp, fingerprint_extra="|epoch=0",
            fdot_axis=False, c_t=0.0)
        return move._fstat_stage_b_payload(spec, rank_index, a, b)

    def test_the_reply_describes_the_partial_it_wrote(self):
        move = self._move(holder=object(), scorer=_fake_call_fstat())
        payload = self._payload(move, rank_index=1, a=1, b=2)
        with stage_b_env():
            reply = move._gb_serve_fstat_stage_b(payload, {}, None)

        self.assertEqual(
            set(reply),
            {"gi", "a", "b", "rank_index", "n_rows", "sha1", "wall_s", "rank"})
        self.assertEqual((reply["gi"], reply["a"], reply["b"]), (3, 1, 2))
        self.assertEqual(reply["rank_index"], 1)
        self.assertEqual(reply["n_rows"], 1)
        self.assertIsNone(reply["rank"])          # no fan-out: not "rank 0"
        self.assertGreaterEqual(reply["wall_s"], 0.0)

        # the reply's sha1 must describe the file the head will read back
        arr = G.load_stage_b_part(self.tmp, 3, 1)
        self.assertEqual(arr.shape, (1, 3, 2, 2, 2))
        self.assertEqual(arr.dtype, np.float64)
        self.assertEqual(
            hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16],
            reply["sha1"])

    def test_the_grid_it_writes_is_the_serial_sweep_of_that_range(self):
        move = self._move(holder=object(), scorer=_fake_call_fstat())
        payload = self._payload(move, rank_index=0, a=0, b=2)
        with stage_b_env():
            move._gb_serve_fstat_stage_b(payload, {}, None)
            spec = move._fstat_stage_b_spec(payload)
            want = G.run_stage_b_group(
                dataclasses.replace(spec, ckpt_name=None, parts_dir=None),
                _fake_call_fstat(), xp=np)
        np.testing.assert_array_equal(G.load_stage_b_part(self.tmp, 3, 0), want)

    def test_without_a_reference_row_it_refuses(self):
        """No holder means ``holder=None`` reaches ``_fstat_call``, which
        scores the LIVE ACA at row 0 -- a different walker, silently."""
        move = self._move(holder=None)          # and NO stubbed scorer
        payload = self._payload(move)
        with self.assertRaises(RuntimeError) as ctx:
            move._gb_serve_fstat_stage_b(payload, {}, None)
        self.assertIn("gb_fstat_ref_row", str(ctx.exception))
        self.assertFalse(os.listdir(self.tmp))  # nothing was written

    def test_an_empty_range_writes_an_empty_partial(self):
        """More ranks than boxes: the tail ranks still have to produce the
        file the head assembles from."""
        move = self._move(holder=object(), scorer=_fake_call_fstat())
        payload = self._payload(move, rank_index=2, a=2, b=2)
        with stage_b_env():
            reply = move._gb_serve_fstat_stage_b(payload, {}, None)
        self.assertEqual(reply["n_rows"], 0)
        self.assertEqual(G.load_stage_b_part(self.tmp, 3, 2).shape,
                         (0, 3, 2, 2, 2))


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
        # LIVE, not decoration: the runner refuses ``single`` outright, so
        # that the byte-identity-gated serial path cannot be routed through
        # per-rank partials by a wiring mistake.
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
            # ``_fanout_cmd`` returns the BARE result dicts (fanout.py:223-225
            # single / :308 multi unwrap the envelope ``_reply`` built at
            # :71), not ``{rank: {"result": ...}}``.
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

    def test_a_reply_that_misreports_its_row_count_is_refused(self):
        """A SHORT partial is caught downstream too, but only as an
        aggregate shape error naming no rank -- after the whole stage-B wall
        has been paid. The per-rank check fires while the rank is still in
        hand, and nothing else in this suite would go red if it vanished."""
        move, layout = self._move(n_compute=2)
        spec = self._spec()
        whole = np.zeros(spec.node_shape)

        def fake_cmd(op, per_rank_payload, model):
            replies = self._serve(
                move, layout, spec, per_rank_payload,
                lambda sub: whole[sub.a - spec.a:sub.b - spec.a])
            first = layout.compute_ranks[0]
            replies[first] = dict(replies[first], n_rows=0)
            return replies, None

        move._fanout_cmd = fake_cmd
        runner = move._fstat_stage_b_runner("model")
        with self.assertRaises(RuntimeError) as ctx:
            runner(spec, None, xp=np)
        msg = str(ctx.exception)
        self.assertIn("box row(s)", msg)
        self.assertIn("rank index 0", msg)

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

    def test_one_compute_rank_refuses_to_build_a_runner_at_all(self):
        """``n_compute == 1`` is the spec's byte-identity gate.

        ``WalkerFanout.run``'s single branch calls only ``local_body``, so the
        runner would "work" -- and silently rename the group's checkpoint from
        the legacy ``stageb`` to ``stageb_r0`` (killing an in-flight fit's
        resume) while round-tripping the gated serial sweep through disk. The
        caller must pass no ``sweep_runner`` there, and this refuses rather
        than trust a docstring.
        """
        move, _layout = self._move(n_compute=1)
        with self.assertRaises(RuntimeError) as ctx:
            move._fstat_stage_b_runner("model")
        self.assertIn("sweep_runner", str(ctx.exception))

    def test_orphan_partials_of_a_wider_fit_are_swept_tmp_files_too(self):
        """``clear_stage_b_parts`` only unlinks ``r < n_parts``, so a fit that
        died at a LARGER ``n_compute`` leaves hundreds of MB per orphan --
        including a ``.npy.tmp`` from a rank that died mid-``np.save``, which
        the plain ``*.npy`` pattern did not match. Another group's partials
        must survive: ``g1`` must not sweep ``g11``."""
        move, layout = self._move(n_compute=2)
        spec = self._spec()
        whole = np.zeros(spec.node_shape)
        planted = ("stageb_g0_r7.npy", "stageb_g0_r9.npy.tmp")
        kept = ("stageb_g01_r0.npy", "stageb_g1_r0.npy",
                "stageb_g0.progress.npz")
        for name in planted + kept:
            with open(os.path.join(self.tmp, name), "wb") as f:
                f.write(b"x")

        def fake_cmd(op, per_rank_payload, model):
            return self._serve(
                move, layout, spec, per_rank_payload,
                lambda sub: whole[sub.a - spec.a:sub.b - spec.a]), None

        move._fanout_cmd = fake_cmd
        runner = move._fstat_stage_b_runner("model")
        with self.assertLogs("lisatools.globalfit.moves.gbspecialstretch",
                             level="INFO"):
            runner(spec, None, xp=np)
        left = set(os.listdir(self.tmp))
        self.assertEqual(left & set(planted), set())
        self.assertEqual(left & set(kept), set(kept))

    def test_an_unreadable_partial_names_the_shared_filesystem(self):
        """Not only ``FileNotFoundError``: a flaky or node-local shared mount
        surfaces just as often as ``ESTALE`` or a ``PermissionError``, and
        those used to die bare inside ``np.load`` -- after the whole stage-B
        wall -- saying nothing about why."""
        move, layout = self._move(n_compute=2)
        spec = self._spec()
        whole = np.zeros(spec.node_shape)

        def fake_cmd(op, per_rank_payload, model):
            return self._serve(
                move, layout, spec, per_rank_payload,
                lambda sub: whole[sub.a - spec.a:sub.b - spec.a]), None

        def boom(*a, **k):
            raise PermissionError(13, "Permission denied", "stageb_g0_r1.npy")

        move._fanout_cmd = fake_cmd
        with mock.patch.object(G, "assemble_stage_b_group", boom):
            runner = move._fstat_stage_b_runner("model")
            with self.assertRaises(RuntimeError) as ctx:
                runner(spec, None, xp=np)
        msg = str(ctx.exception)
        self.assertIn("SHARED by every compute rank", msg)
        self.assertIn("PermissionError", msg)
        self.assertIn("stageb_g0_r1.npy", msg)

    def test_a_fanout_with_no_single_flag_is_refused_not_crashed(self):
        """Reading ``fanout.single`` bare answers a fan-out-like object that
        does not carry it with an ``AttributeError`` instead of the refusal.
        Unknown shape -> keep the byte-identity gate CLOSED."""
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialRJFStatGridMove.__new__(gbs.GBSpecialRJFStatGridMove)
        move.name = "gb_test"
        move.fanout = object()          # no ``single``
        with self.assertRaises(RuntimeError) as ctx:
            move._fstat_stage_b_runner("model")
        self.assertIn("sweep_runner", str(ctx.exception))

    def test_no_fanout_refuses_to_build_a_runner_at_all(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialRJFStatGridMove.__new__(gbs.GBSpecialRJFStatGridMove)
        move.name = "gb_test"
        with self.assertRaises(RuntimeError) as ctx:
            move._fstat_stage_b_runner("model")
        self.assertIn("sweep_runner", str(ctx.exception))

    def test_a_scorer_that_is_not_the_cached_holder_call_is_refused(self):
        """Stage A, stage B and the centre table must score against ONE
        residual snapshot (spec decision 6). The ranks build their own scorer,
        so nothing else ties the head's to theirs."""
        move, _layout = self._move(n_compute=2)
        move._fstat_ref_call = object()          # the head built its scorer
        move._fanout_cmd = lambda *a, **k: self.fail("must not fan out")
        runner = move._fstat_stage_b_runner("model")
        with self.assertRaises(RuntimeError) as ctx:
            runner(self._spec(), lambda p: p, xp=np)   # a DIFFERENT scorer
        self.assertIn("_fstat_holder_call", str(ctx.exception))


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


class DoneManifestTest(unittest.TestCase):
    """DONE.json records the GLOBAL reference walker and the rank count."""

    def test_manifest_keys(self):
        """``_run_fstat_fit`` lives on :class:`GBSpecialRJFStatGridMove`, not
        on ``GBSpecialBase`` (the brief's attribution is the same one Tasks
        6-7 corrected for ``_fstat_call``)."""
        import inspect

        from lisatools.globalfit.moves import gbspecialstretch as gbs

        src = inspect.getsource(gbs.GBSpecialRJFStatGridMove._run_fstat_fit)
        for key in ("walker_ref", "n_compute", "n_peaks", "wall_seconds",
                    "num_proposals", "clock", "epoch"):
            self.assertIn(key, src, f"DONE.json must record {key!r}")

    def test_run_fstat_grid_fit_forwards_a_sweep_runner(self):
        import inspect

        sig = inspect.signature(G.run_fstat_grid_fit)
        self.assertIn("sweep_runner", sig.parameters)
        self.assertIsNone(sig.parameters["sweep_runner"].default)


class _CountingMempool:
    """``_NoOpMempool`` that says how many times it was asked to free."""

    def __init__(self):
        self.frees = 0

    def free_all_blocks(self):
        self.frees += 1


class RunFstatFitWiringTest(unittest.TestCase):
    """``_run_fstat_fit``: global reference, holder scoring, split stage B.

    The real method, on a ``__new__`` skeleton, with ``run_fstat_grid_fit``
    patched to capture what it was handed. What is being pinned is the
    WIRING -- which walker the manifest names, which scorer the sweep gets,
    whether the split runner is installed, and that the head no longer opens
    a GB-free window of its own (it belongs to the owning rank now, inside
    ``gb_fstat_ref_row``).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _move(self, n_compute=2, *, w_global=6):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialRJFStatGridMove.__new__(gbs.GBSpecialRJFStatGridMove)
        move.name = "gb_test"
        move._backend_name = "lisatools_cpu"
        move.branch_name = "gb"
        move.band_edges = BAND_EDGES
        move.df = 1.0 / TOBS
        move.fstat_fit_kwargs = {"mc_lims": [0.02, 0.8]}
        move.num_proposals = 3
        move.mempool = _CountingMempool()
        move._epoch_dir = lambda k: self.tmp
        move._fstat_clock = lambda: 11

        layout = _build_fake_layout(4 * n_compute, n_compute)
        move.fanout = _StubFanout(layout, lls=None)
        move.fanout.single = (n_compute == 1)
        self.owner_rank = layout.compute_ranks[-1]
        self.lls = np.arange(4 * n_compute, dtype=float)
        move._fstat_global_reference = lambda model: (
            w_global, self.owner_rank, 2, self.lls)

        self.seen = {"ref_row": [], "release": [], "runner": 0, "window": 0}
        self.scorer = lambda params: params
        self.runner = lambda spec, call, *, xp: None

        def ref_row(model, branches, w, owner, local):
            self.seen["ref_row"].append((w, owner, local, branches))

        def release(model):
            # the ranks' rows go only once this epoch's artifacts are on disk
            self.seen["release"].append(
                os.path.exists(os.path.join(self.tmp, "DONE.json")))

        def stage_b_runner(model):
            self.seen["runner"] += 1
            # R2's ORDERING: in production ``_fstat_holder_call`` is what
            # populates ``_fstat_ref_call``, and the stage-B runner's
            # scorer-identity guard is vacuous until it has. Record whether
            # the scorer was already cached when the runner was built.
            self.seen["scorer_cached_at_runner_build"] = (
                move._fstat_ref_call is not None)
            return self.runner

        @contextlib.contextmanager
        def window(model, branches, walker_ref):
            self.seen["window"] += 1
            yield

        def holder_call(model):
            # what the real one does, and the only thing that does it
            move._fstat_ref_call = self.scorer
            return self.scorer

        move._fstat_ref_row_fanout = ref_row
        move._fstat_release_fanout = release
        move._fstat_stage_b_runner = stage_b_runner
        move._fstat_holder_call = holder_call
        move._gb_free_residual = window
        move._fstat_ref_call = None
        return move

    @contextlib.contextmanager
    def _patched_fit(self, n_peaks=17):
        captured = {}

        def fake_fit(call_fstat, **kw):
            captured["call_fstat"] = call_fstat
            captured.update(kw)
            return "STACKED", n_peaks

        with mock.patch.object(G, "run_fstat_grid_fit", fake_fit):
            yield captured

    def _manifest(self):
        import json

        with open(os.path.join(self.tmp, "DONE.json")) as f:
            return json.load(f)

    def test_the_sweep_is_scored_through_the_replicated_row(self):
        move = self._move(n_compute=2)
        with self._patched_fit() as captured:
            stacked, n_peaks = move._run_fstat_fit("model", 4, branches={"gb": 1})
        self.assertEqual((stacked, n_peaks), ("STACKED", 17))
        self.assertIs(captured["call_fstat"], self.scorer)
        # the row is replicated BEFORE anything scores through it
        self.assertEqual(self.seen["ref_row"],
                         [(6, self.owner_rank, 2, {"gb": 1})])
        # ... and the head no longer opens a GB-free window of its own: it
        # belongs to the OWNING rank, inside gb_fstat_ref_row
        self.assertEqual(self.seen["window"], 0)
        self.assertEqual(captured["cache_dir"], self.tmp)
        self.assertEqual(captured["epoch"], 4)
        self.assertIn("epoch=4", captured["fingerprint_extra"])
        self.assertIn("gbfree=1", captured["fingerprint_extra"])
        self.assertAlmostEqual(captured["Tobs"], TOBS)
        self.assertEqual(captured["mc_lims"], [0.02, 0.8])

    def test_several_compute_ranks_install_the_split_stage_b_runner(self):
        move = self._move(n_compute=3)
        with self._patched_fit() as captured:
            move._run_fstat_fit("model", 0)
        self.assertIs(captured["sweep_runner"], self.runner)
        self.assertEqual(self.seen["runner"], 1)
        self.assertEqual(self._manifest()["n_compute"], 3)
        # R2's ORDERING, asserted rather than only documented: the runner is
        # built AFTER the holder-scored scorer is cached, so its "is this my
        # cached call_fstat?" guard compares two live objects instead of
        # passing vacuously on ``None is not None``.
        self.assertTrue(self.seen["scorer_cached_at_runner_build"],
                        "the split runner was built before the scorer -- "
                        "its identity guard would pass vacuously")

    def test_one_compute_rank_runs_the_serial_path_with_no_runner(self):
        """The byte-identity gate: at one compute rank ``sweep_runner`` must
        stay ``None`` and the split runner must never even be BUILT (it
        raises there, by design -- Task 7's I-1)."""
        move = self._move(n_compute=1)
        with self._patched_fit() as captured:
            move._run_fstat_fit("model", 0)
        self.assertIsNone(captured["sweep_runner"])
        self.assertEqual(self.seen["runner"], 0)
        self.assertEqual(self._manifest()["n_compute"], 1)

    def test_no_fanout_at_all_is_one_compute_rank(self):
        move = self._move(n_compute=1)
        move.fanout = None
        with self._patched_fit() as captured:
            move._run_fstat_fit("model", 0)
        self.assertIsNone(captured["sweep_runner"])
        self.assertEqual(self._manifest()["n_compute"], 1)

    def test_the_epoch_line_keeps_its_lnl_with_no_fanout(self):
        """``_fstat_global_reference`` has no gather to do there, so it
        reports an all-NaN placeholder ``lls`` -- but this process's ACA
        holds EVERY walker, so the line must not lose the lnL it has carried
        since 2026-08-24.

        BOTH walkers, and walker 0 is the point: a ``lls.size > w_global``
        test taken first is TRUE exactly at ``w_global == 0`` (the
        placeholder has length 1), so the fallback silently never fired for
        the one index an ordinary local argmax hits most often.
        """
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        class _Acs:
            @staticmethod
            def likelihood():
                return np.array([-10.0, -4.0, -7.0])

        class _Model:
            analysis_container_arr = _Acs()

        for w_global, want in ((0, "lnL=-10.000"), (1, "lnL=-4.000")):
            with self.subTest(w_global=w_global):
                move = self._move(n_compute=1, w_global=w_global)
                move.fanout = None
                move._fstat_global_reference = lambda model, w=w_global: (
                    w, 0, w, np.full(1, np.nan))
                with self._patched_fit(), \
                        self.assertLogs(gbs.logger, "INFO") as cap:
                    move._run_fstat_fit(_Model(), 0)
                line = "\n".join(cap.output)
                self.assertIn(want, line)
                self.assertIn("spread=6.000", line)

    def test_the_manifest_names_the_global_walker_and_the_rank_count(self):
        move = self._move(n_compute=2, w_global=7)
        with self._patched_fit(n_peaks=5) as _captured:
            move._run_fstat_fit("model", 2)
        got = self._manifest()
        self.assertEqual(got["walker_ref"], 7)       # GLOBAL, not a local row
        self.assertEqual(got["n_compute"], 2)
        self.assertEqual(got["epoch"], 2)
        self.assertEqual(got["n_peaks"], 5)
        self.assertEqual(got["num_proposals"], 3)
        self.assertEqual(got["clock"], 11)
        self.assertGreaterEqual(got["wall_seconds"], 0.0)

    def test_the_ranks_are_released_only_after_the_manifest_is_written(self):
        """Spec decision 2 + the release ruling: a worker's holder (and the
        ~GB sig-het scorer cached beside it) is dropped as soon as the last
        group is assembled and the epoch's artifacts are on disk -- not at
        the next fit, ~50 iterations later."""
        move = self._move(n_compute=2)
        with self._patched_fit():
            move._run_fstat_fit("model", 0)
        self.assertEqual(self.seen["release"], [True])

    def test_the_epoch_line_names_the_global_walker_and_its_owner(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = self._move(n_compute=2)
        with self._patched_fit(), self.assertLogs(gbs.logger, "INFO") as cap:
            move._run_fstat_fit("model", 0)
        line = "\n".join(cap.output)
        self.assertIn("walker_ref=6", line)
        self.assertIn(f"rank {self.owner_rank}", line)
        self.assertIn("n_compute=2", line)

    def test_a_release_failure_cannot_destroy_a_finished_fit(self):
        """C-1: ``gb_fstat_release`` is the cheapest command in the protocol
        and it was allowed to discard a 1 h 45 min section that had already
        succeeded -- the raise propagates out of ``_run_fstat_fit`` AFTER
        DONE.json is durable, so ``setup()`` never reaches ``_install``, the
        ``_fstat_last_fit_hit`` belt or ``_install_ctr_table``, and the run
        dies. A release must never be the thing that fails."""
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = self._move(n_compute=2)

        def boom(model):
            self.seen["release"].append("raised")
            raise RuntimeError("remote worker error during gb_fstat_release")

        move._fstat_release_fanout = boom
        with self._patched_fit(n_peaks=17) as _captured, \
                self.assertLogs(gbs.logger, "WARNING") as cap:
            stacked, n_peaks = move._run_fstat_fit("model", 0)
        self.assertEqual((stacked, n_peaks), ("STACKED", 17))
        self.assertEqual(self.seen["release"], ["raised"])
        self.assertEqual(self._manifest()["n_peaks"], 17)
        self.assertIn("gb_fstat_release failed", "\n".join(cap.output))

    def test_a_complete_stacked_npz_costs_no_broadcast(self):
        """Task 8 concern 6.2. ``run_fstat_grid_fit`` short-circuits on a
        finished stage-B npz -- but the ~72 MB per-rank row broadcast and the
        sig-het scorer build had already been paid by then, for a row nothing
        would score through, and a ``gb_fstat_release`` was then issued for
        it. The completeness test needs no reference row, so it runs first.

        Reachable whenever ``_fstat_fit_decision`` says "fit" over a complete
        npz: a ``DONE.json`` lost beside one, or an offline grid dropped into
        the epoch dir. The REAL ``run_fstat_grid_fit`` runs here, on a real
        golden npz, so the ``call_fstat`` this path hands it is exercised.
        """
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = self._move(n_compute=2)
        move._fstat_holder_call = lambda model: self.fail(
            "a complete epoch must not build an F-stat scorer")
        shutil.copyfile(GOLDEN_SINGLE, G.stacked_grid_path(self.tmp))
        # the golden was fitted in the Mc basis; stacked_from_cache refuses a
        # cache whose basis disagrees with the flag
        with stage_b_env(), self.assertLogs(gbs.logger, "INFO") as cap:
            stacked, n_peaks = move._run_fstat_fit("model", 4,
                                                   branches={"gb": 1})
        self.assertEqual(self.seen["ref_row"], [],
                         "no gb_fstat_ref_row may be issued")
        self.assertEqual(self.seen["release"], [],
                         "nothing was replicated, so nothing to release")
        self.assertEqual(self.seen["runner"], 0)
        self.assertIsNotNone(stacked)
        with np.load(GOLDEN_SINGLE, allow_pickle=False) as d:
            self.assertEqual(n_peaks, int(len(d["peak_f0_mHz"])))
        # the manifest is still written -- that is what stops the next
        # window deciding "fit" all over again
        self.assertEqual(self._manifest()["n_peaks"], n_peaks)
        self.assertEqual(self._manifest()["walker_ref"], 6)
        self.assertIn("already fitted", "\n".join(cap.output))

    def test_the_completeness_test_names_the_file_the_fit_loads(self):
        """One implementation, so the move's pre-check can never disagree
        with the orchestrator's own short circuit."""
        self.assertFalse(G.stage_b_complete(self.tmp))
        path = G.stacked_grid_path(self.tmp)
        self.assertEqual(
            path,
            os.path.join(self.tmp, G.GRID_BASENAME).replace(
                ".npz", "_peaks_stacked.npz"))
        with open(path, "wb") as f:
            f.write(b"")
        self.assertTrue(G.stage_b_complete(self.tmp))


class ReleaseBodyTest(unittest.TestCase):
    """``gb_fstat_release``: the seventh op, and what serving it does."""

    def _move(self, cls_name="GBSpecialRJFStatGridMove", *, held=True):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        cls = getattr(gbs, cls_name)
        move = cls.__new__(cls)
        move.name = "gb_test"
        move.mempool = _CountingMempool()
        if held:
            move._fstat_ref_holder = object()
            move._fstat_ref_call = object()
            move._fstat_ref_walker = 6
        return move

    def test_the_op_follows_the_session_commands_in_gb_ops(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        self.assertEqual(
            gbs.GB_OPS[:4],
            ("gb_run_proposal", "gb_run_tempering", "gb_finish", "gb_sync"))
        self.assertEqual(gbs.GB_OPS[4:],
                         ("gb_fstat_ref_row", "gb_fstat_stage_b",
                          "gb_fstat_release"))

    def test_serving_it_drops_the_row_and_the_cached_scorer(self):
        """BOTH, not just the row: Task 7 caches the built sig-het scorer on
        ``_fstat_ref_call``, whose bucketed reference blocks are the ~GB half
        of what a worker is holding."""
        move = self._move()
        reply = move.gf_serve("gb_fstat_release", None, {}, None)
        self.assertIsNone(move._fstat_ref_holder)
        self.assertIsNone(move._fstat_ref_call)
        self.assertIsNone(move._fstat_ref_walker)
        self.assertEqual(move.mempool.frees, 1)
        self.assertTrue(reply["released"])

    def test_holding_nothing_is_a_no_op_that_still_frees_the_pool(self):
        move = self._move(held=False)
        reply = move.gf_serve("gb_fstat_release", None, {}, None)
        self.assertFalse(reply["released"])
        self.assertIsNone(move._fstat_ref_holder)
        self.assertEqual(move.mempool.frees, 1)

    def test_it_serves_on_a_move_with_no_fstat_surface(self):
        """Deliberately NOT gated by ``_FSTAT_OP_REQUIRES`` (unlike the other
        two F-stat ops): the body touches nothing that belongs to the grid
        move, and a release that REFUSED would leave that rank holding the
        row and the scorer with nothing left to drop them."""
        move = self._move("GBSpecialBase")
        reply = move.gf_serve("gb_fstat_release", None, {}, None)
        self.assertTrue(reply["released"])
        self.assertIsNone(move._fstat_ref_holder)

    def test_it_needs_no_payload_and_no_session(self):
        move = self._move()
        move.gf_serve("gb_fstat_release", None, {}, None)   # payload None
        self.assertIsNone(move._fstat_ref_holder)


class ReleaseFanoutTest(unittest.TestCase):
    """2-rank ``FakeWorld``: every WORKER drops its row, the head keeps its own.

    A real :class:`WalkerFanout` over the fake communicator with a real
    :class:`ComputeService` on the worker, so the command genuinely crosses
    the wire and is served by ``gf_serve``. The head's own holder must
    SURVIVE -- the centre table has not run yet, and it must score against
    the same row the grids were fitted with (spec decision 6).
    """

    def _move(self, *, mempool=None):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialRJFStatGridMove.__new__(gbs.GBSpecialRJFStatGridMove)
        move.name = "gb_test"
        move.gf_move_name = "gb_test"
        move.mempool = _CountingMempool() if mempool is None else mempool
        move._fstat_ref_holder = object()
        move._fstat_ref_call = object()
        move._fstat_ref_walker = 6
        return move

    def _run(self, n_compute=2):
        from lisatools.globalfit.communication import ranks as R
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.fanout import (
            ComputeService,
            WalkerFanout,
        )

        # a hang would be the regression here too, so keep the timeout short
        world = FakeWorld(n_compute + 1, timeout=5.0)

        def body(rank, comm):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                layout = R.build_layout(comm, 4 * n_compute,
                                        list(range(n_compute)))
            fcomm = layout.make_fanout_comm(comm)
            role = layout.role_of(rank)
            if role == R.RankRole.SAVER:
                return ("saver", None, None)
            move = self._move()
            if role == R.RankRole.HEAD:
                move.fanout = WalkerFanout(fcomm, layout, rank, model=None)
                try:
                    replies = move._fstat_release_fanout(None)
                finally:
                    move.fanout.stop()
                # the head's pool free is included: surviving its OWN
                # free_all_blocks is exactly what the save/restore claims
                return ("head", (move._fstat_ref_holder, move._fstat_ref_call),
                        (sorted(replies), move.mempool.frees))
            ComputeService(fcomm, layout, rank,
                           registry={"gb_test": move}, model=None).serve()
            return ("worker", (move._fstat_ref_holder, move._fstat_ref_call),
                    move.mempool.frees)

        return world.run(body)

    def test_workers_drop_both_halves_and_the_head_keeps_its_own(self):
        out = self._run(n_compute=2)
        roles = {row[0] for row in out.values()}
        self.assertEqual(roles, {"head", "worker", "saver"})
        for rank, (role, held, extra) in out.items():
            if role == "worker":
                self.assertEqual(held, (None, None), f"rank {rank} kept a row")
                self.assertEqual(extra, 1, "the worker never freed its pool")
            elif role == "head":
                # the centre table still has to score against this row
                self.assertIsNotNone(held[0], "the head dropped its own row")
                self.assertIsNotNone(held[1], "the head dropped its scorer")
                ranks_replied, head_frees = extra
                self.assertEqual(len(ranks_replied), 2)  # one per compute rank
                # the head served the command like every other rank -- pool
                # sweep included -- and its row survived that too, because
                # the ``saved`` tuple holds those arrays alive
                self.assertEqual(head_frees, 1,
                                 "the head skipped the uniform body")

    def test_a_worker_that_cannot_release_does_not_abort_the_epoch(self):
        """C-1, over the real wire: a worker whose served body raises turns
        into an exception on the HEAD, out of ``_fstat_release_fanout``. That
        is the failure ``_run_fstat_fit`` now swallows (see
        ``RunFstatFitWiringTest::test_a_release_failure_cannot_destroy_a_
        finished_fit``) -- and the head's own row must still be RESTORED by
        the driver's ``finally``, or the centre table would have nothing to
        score against even on the path that survives."""
        from lisatools.globalfit.communication import ranks as R
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.fanout import (
            ComputeService,
            WalkerFanout,
        )

        class _AngryMempool(_CountingMempool):
            def free_all_blocks(self):
                super().free_all_blocks()
                raise RuntimeError("device pool is wedged")

        world = FakeWorld(3, timeout=5.0)

        def body(rank, comm):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                layout = R.build_layout(comm, 8, [0, 1])
            fcomm = layout.make_fanout_comm(comm)
            role = layout.role_of(rank)
            if role == R.RankRole.SAVER:
                return ("saver", None)
            if role == R.RankRole.HEAD:
                move = self._move()
                move.fanout = WalkerFanout(fcomm, layout, rank, model=None)
                try:
                    move._fstat_release_fanout(None)
                    raised = None
                except Exception as exc:                      # noqa: BLE001
                    raised = repr(exc)
                finally:
                    move.fanout.stop()
                return ("head", (raised, move._fstat_ref_holder is not None,
                                 move._fstat_ref_call is not None))
            move = self._move(mempool=_AngryMempool())
            ComputeService(fcomm, layout, rank,
                           registry={"gb_test": move}, model=None).serve()
            return ("worker", None)

        out = world.run(body)
        head = next(v for v in out.values() if v[0] == "head")
        raised, kept_row, kept_scorer = head[1]
        self.assertIsNotNone(raised, "a worker failure must reach the head")
        self.assertTrue(kept_row, "the head lost its row on the failure path")
        self.assertTrue(kept_scorer, "the head lost its scorer")

    def test_one_compute_rank_issues_nothing(self):
        """Nothing to release remotely, and the golden-gated single-rank path
        must not grow a command it never had. Both shapes: no fan-out object
        at all (``_propose_legacy`` / ``fit.sample()``) and a real one
        reporting a single compute rank."""
        layout = _build_fake_layout(4, 1)
        for fanout in (None, _StubFanout(layout, lls=None)):
            if fanout is not None:
                fanout.single = True
            move = self._move()
            move.fanout = fanout
            move._fanout_cmd = lambda *a, **k: self.fail("must not fan out")
            self.assertEqual(move._fstat_release_fanout(None), {})
            self.assertIsNotNone(move._fstat_ref_holder)


class CentreTableReferenceTest(unittest.TestCase):
    """The centre sweep must reuse the fit's replicated row, not re-derive one."""

    def test_no_local_reference_walker_derivation_left(self):
        """``_install_ctr_table`` lives on :class:`GBSpecialRJFStatGridMove`,
        not on ``GBSpecialBase`` (the brief's attribution is the one Tasks
        6-8 corrected for ``_fstat_call`` / ``_run_fstat_fit``)."""
        import inspect

        from lisatools.globalfit.moves import gbspecialstretch as gbs

        src = inspect.getsource(gbs.GBSpecialRJFStatGridMove._install_ctr_table)
        self.assertNotIn(
            "self._fstat_reference_walker(model)", src,
            "the centre table must not re-derive a LOCAL reference walker; "
            "it scores through the fit's replicated reference row")
        self.assertIn("_fstat_ref_holder", src)
        self.assertIn("_fstat_holder_call", src)

    def test_the_holder_call_is_cached_so_grids_and_centres_share_it(self):
        """Spec verification 1(d): SAME scorer object, not a rebuilt twin.

        The fit and the centre sweep must score against one reference row
        and one sig-het reference-block stash. ``_fstat_holder_call``
        caching it is what makes that literal -- a rebuilt closure would
        silently re-snapshot and re-bucket.
        """
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move._fstat_ref_holder = object()
        move._fstat_ref_call = None
        sentinel = object()
        calls = []

        def fake_fstat_call(model, walker_ref, *, holder=None):
            calls.append((walker_ref, holder))
            return sentinel

        move._fstat_call = fake_fstat_call
        first = move._fstat_holder_call(None)
        second = move._fstat_holder_call(None)
        self.assertIs(first, sentinel)
        self.assertIs(second, first, "the scorer must be built once per fit")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 0, "the holder is scored at row 0")
        self.assertIs(calls[0][1], move._fstat_ref_holder)

        move._fstat_release_ref_row()
        self.assertIsNone(move._fstat_ref_call)
        self.assertIsNone(move._fstat_ref_holder)


class CentreTableScoringTest(unittest.TestCase):
    """The REAL ``_install_ctr_table``, driven on a ``__new__`` skeleton.

    What is pinned is which residual the centre sweep sees: the fit's live
    replicated row when there is one, and otherwise one replicated here
    through the SAME global-reference + ``gb_fstat_ref_row`` path -- never a
    local argmax against the live ACA (spec decision 6).
    """

    def setUp(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        self.gbs = gbs
        self.tmp = tempfile.mkdtemp()
        self.events = []
        self.scorer = lambda params: params
        self.move = self._move()

    def tearDown(self):
        # the table registry is process-global; a leftover entry would make
        # the next test short-circuit before it reached the sweep
        self.gbs._FSTAT_CTR_TABLE_REGISTRY.pop(self.tmp, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _move(self, *, holder=True):
        gbs = self.gbs
        move = gbs.GBSpecialRJFStatGridMove.__new__(gbs.GBSpecialRJFStatGridMove)
        move.name = "gb_test"
        move._backend_name = "lisatools_cpu"
        move.branch_name = "gb"
        move.fstat_fit_kwargs = {"mc_lims": [0.02, 0.8]}
        move._epoch_dir = lambda k: self.tmp
        move._fstat_ref_holder = object() if holder else None
        move._fstat_ref_call = None
        move._fstat_ref_walker = 6 if holder else None
        self.owner_rank = 3

        events = self.events

        def holder_call(model):
            events.append("holder_call")
            return self.scorer

        def ref_row(model, branches, w, owner, local):
            events.append(("ref_row", w, owner, local, branches))
            move._fstat_ref_holder = object()

        def release_fanout(model):
            events.append("release_fanout")

        def global_reference(model):
            events.append("global_reference")
            return 6, self.owner_rank, 2, np.arange(8, dtype=float)

        def local_walker(model):                       # the banned route
            events.append("LOCAL_ARGMAX")
            return 0

        @contextlib.contextmanager
        def window(model, branches, walker_ref):
            events.append("gb_free_window")
            yield

        move._fstat_holder_call = holder_call
        move._fstat_ref_row_fanout = ref_row
        move._fstat_release_fanout = release_fanout
        move._fstat_global_reference = global_reference
        move._fstat_reference_walker = local_walker
        move._gb_free_residual = window
        return move

    @contextlib.contextmanager
    def _patched_build(self, raises=None):
        captured = {}

        def fake_build(call_fstat, **kw):
            self.events.append("sweep" if call_fstat is not None else "load")
            captured["call_fstat"] = call_fstat
            captured.update(kw)
            if raises is not None:
                raise raises
            return None

        with mock.patch.object(G, "build_fstat_center_table", fake_build):
            yield captured

    def test_a_live_holder_is_what_the_sweep_scores_through(self):
        with self._patched_build() as captured:
            self.move._install_ctr_table(4, model="model", branches={"gb": 1})
        self.assertIs(captured["call_fstat"], self.scorer)
        self.assertEqual(self.events, ["holder_call", "sweep"])
        # nothing re-derived, nothing re-broadcast, no second GB-free window
        self.assertNotIn("LOCAL_ARGMAX", self.events)
        # ... and the head's row survives: setup()'s finally owns it
        self.assertIsNotNone(self.move._fstat_ref_holder)
        self.assertEqual(captured["cache_dir"], self.tmp)
        self.assertEqual(captured["mc_lims"], [0.02, 0.8])

    def test_without_a_holder_it_replicates_the_global_row_itself(self):
        """The load-only path (an offline grid dropped in, or a centre table
        missing from a complete epoch) must take the SAME global reference
        and the SAME broadcast the fit does -- a local argmax there would
        score the centres against another walker's residual, and its index
        is not even a valid ACA row off its owner."""
        move = self._move(holder=False)
        with self._patched_build() as captured:
            move._install_ctr_table(4, model="model", branches={"gb": 1})
        self.assertIs(captured["call_fstat"], self.scorer)
        self.assertEqual(
            self.events,
            ["global_reference", ("ref_row", 6, self.owner_rank, 2, {"gb": 1}),
             "holder_call", "sweep", "release_fanout"])
        # both halves put back: the workers' by the fan-out release, the
        # head's here, because this path -- not setup()'s fit -- took it
        self.assertIsNone(move._fstat_ref_holder)
        self.assertIsNone(move._fstat_ref_call)

    def test_a_failed_sweep_still_drops_the_row_it_took(self):
        """... and issues NO fan-out command while the exception unwinds:
        that would replace the real error with a fan-out failure (the same
        reason ``_run_fstat_fit``'s release is not in a ``finally``)."""
        move = self._move(holder=False)
        boom = RuntimeError("sweep died")
        with self._patched_build(raises=boom):
            with self.assertRaises(RuntimeError) as ctx:
                move._install_ctr_table(4, model="model", branches={"gb": 1})
        self.assertIs(ctx.exception, boom)
        self.assertIsNone(move._fstat_ref_holder)
        self.assertNotIn("release_fanout", self.events)

    def test_a_table_already_on_disk_takes_neither_a_row_nor_a_scorer(self):
        """The checkpoint-load branch: no sweep, so no reference row and --
        crucially under FSTAT_USE_SIGHET -- no scorer build."""
        with open(os.path.join(self.tmp, G.CENTER_TABLE_BASENAME), "wb") as f:
            f.write(b"")
        with self._patched_build() as captured:
            self.move._install_ctr_table(4, model="model", branches={"gb": 1})
        self.assertIsNone(captured["call_fstat"])
        self.assertEqual(self.events, ["load"])

    def test_the_rank_side_load_never_replicates_a_row(self):
        """``_setup_from_directive`` calls this with ``model=None``; a rank
        that fanned out from there would deadlock -- the fan-out is head-only
        and every other rank is parked in ``ComputeService.serve``."""
        move = self._move(holder=False)
        with self._patched_build() as captured:
            move._install_ctr_table(4, model=None)
        self.assertIsNone(captured["call_fstat"])
        self.assertEqual(self.events, ["load"])


# =========================================================================
# THE ACCEPTANCE GATE (spec "Verification" 1(a)-(d)).
#
# A real WalkerFanout / ComputeService over a real FakeWorld drives the
# PRODUCTION served bodies and the production head-side drivers on the tiny
# analytic fixture. Nothing about the wire, the dispatcher, the guard, the
# split, the assembly or the release is stubbed; what IS stubbed is only
# what needs a GPU (the ACA bind, the sig-het scorer build, the BandSorter
# window and the memory pool).
# =========================================================================


def _row_keyed_call_fstat(ref_row=None):
    """The analytic fixture scorer, KEYED BY THE REFERENCE ROW it scores.

    The production ``_fstat_call`` builds its scorer AGAINST the replicated
    reference row, so a rank that scored its OWN live residual instead would
    produce different numbers on every node. The analytic fixture ignores
    its data entirely -- which would let exactly that bug reproduce a golden
    byte for byte -- so the stand-in folds ONE number off the row into the
    amplitude. That is what turns "every rank scored the owner's row" from a
    wiring assertion into a BYTE-LEVEL one.

    An all-zero (or absent) row is the EXACT identity: ``1.0 + 0.01 * 0.0``
    is ``1.0`` and ``x * 1.0`` is exact for every finite float, which is
    what lets the stage-B gate keep scoring the stored goldens' own
    function through this wrapper.
    """
    base = _fake_call_fstat()
    row = np.asarray([] if ref_row is None else ref_row).reshape(-1)
    scale = 1.0 + 0.01 * float(row[0]) if row.size else 1.0

    def call(params):
        N, M = base(params)
        return N * scale, M

    return call


def _kill_stage_b_once(rank_suffix="_r1", gi=None):
    """Patch ``run_stage_b_group`` to kill ONE rank's sweep, exactly once.

    Deterministic by NAME (``ckpt_name`` ends ``_r<index>``) rather than by
    call count: the FakeWorld ranks are threads and a counter would pick a
    different victim from run to run. Returns ``(patch, fired event)``.
    """
    from lisatools.sampling import fstat_gridfit as GG

    real = GG.run_stage_b_group
    fired = threading.Event()

    def flaky(spec, call_fstat, *, xp):
        if ((gi is None or int(spec.gi) == int(gi))
                and str(spec.ckpt_name).endswith(rank_suffix)
                and not fired.is_set()):
            fired.set()
            raise RuntimeError("simulated rank death mid-sweep")
        return real(spec, call_fstat, xp=xp)

    return mock.patch.object(GG, "run_stage_b_group", flaky), fired


class _StubRefRow:
    """Stands in for ``gbbands.FStatRefRowHolder`` where no row is shipped.

    ``_fstat_holder_call`` tests a holder for ``None`` and hands it to
    ``_fstat_call``; nothing else on the stage-B path reads one. The
    all-zero row makes :func:`_row_keyed_call_fstat` the exact identity, so
    the stage-B gate still scores the stored goldens' own function.
    """

    def __init__(self, n=4):
        self.linear_data_arr = [np.zeros(int(n))]
        self.linear_psd_arr = [np.zeros(int(n))]


class _StageBRankStub:
    """A stand-in move exposing exactly the F-stat stage-B rank surface.

    Real ``GBSpecialBase`` needs a GPU, a built ``BandSorter`` and gigabytes
    of buffers, which the laptop budget forbids -- but the code under test
    (``_fstat_stage_b_payload`` / ``_fstat_stage_b_spec`` /
    ``_gb_serve_fstat_stage_b`` / ``_fstat_stage_b_runner``, reached through
    the real ``gf_serve`` and its ``_require_fstat_grid_move`` guard)
    touches only the fan-out, the spec slicing, the scorer chokepoint and
    ``run_stage_b_group``. Every method below is BORROWED from production,
    so what runs is the dispatcher and the bodies the cluster runs; only
    ``_bind_rank_acs``, ``_fstat_call`` and the memory pool are stubs.

    THE GUARD IS NOT WEAKENED. ``gf_serve`` refuses ``gb_fstat_stage_b`` on a
    move whose ``_fstat_call`` is ``None`` (``_FSTAT_OP_REQUIRES``); this
    stub HAS one, and it is holder-scored exactly as the real one is --
    it raises without a holder, and the scorer it returns depends on the
    row, so the guard is satisfied honestly rather than defeated.
    """

    from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase as _B

    # ``_fstat_stage_b_payload`` / ``_fstat_stage_b_spec`` /
    # ``_gb_session_token`` are staticmethods, so reading them off the class
    # yields plain functions -- re-wrap.
    _fstat_stage_b_payload = staticmethod(_B._fstat_stage_b_payload)
    _fstat_stage_b_spec = staticmethod(_B._fstat_stage_b_spec)
    _gb_session_token = staticmethod(_B._gb_session_token)
    _gb_serve_fstat_stage_b = _B._gb_serve_fstat_stage_b
    _gb_serve_fstat_release = _B._gb_serve_fstat_release
    _fstat_stage_b_runner = _B._fstat_stage_b_runner
    _fstat_holder_call = _B._fstat_holder_call
    _fstat_release_ref_row = _B._fstat_release_ref_row
    _require_fstat_grid_move = _B._require_fstat_grid_move
    _fanout_cmd = _B._fanout_cmd
    _rank_tag = _B._rank_tag
    gf_move_name = "gb_pe"
    name = "gb_stub"

    def __init__(self, fanout, *, rank=None):
        self.fanout = fanout
        self.gf_rank = rank
        self.mempool = _CountingMempool()
        self.ops_served = []
        self.nwalkers = None
        self.ntemps = None
        self._prop_timer = None
        self._fstat_ref_holder = _StubRefRow()
        self._fstat_ref_call = None
        self._fstat_ref_walker = 0

    @property
    def xp(self):
        return np

    def _bind_rank_acs(self, model):
        return None

    def _fstat_call(self, model, walker_ref, *, holder=None):
        """The one stub the guard tests for -- holder-scored, like the real one."""
        if holder is None:
            raise AssertionError(
                "a stage-B scorer must be built from the replicated row")
        return _row_keyed_call_fstat(holder.linear_data_arr[0])

    def gf_serve(self, op, payload, clock, model):
        """Record the op, then run the PRODUCTION dispatcher unchanged."""
        self.ops_served.append(op)
        return self._B.gf_serve(self, op, payload, clock, model)


class _RefRowRankStub:
    """A stand-in move for ``gb_fstat_ref_row``: fake ACA, fake GB-free window.

    Borrows the production ``_gb_serve_fstat_ref_row`` (and the real
    ``gf_serve`` that dispatches to it, guard included), so the collective,
    the header exchange and the holder construction under test are the real
    ones. The GB-free window is faked as "+1.0 on this walker's residual
    row", which makes two things checkable at once: the SHIPPED row carries
    the window's effect, and the owner's LIVE residual is back to its
    original value afterwards.
    """

    from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase as _B

    _gb_serve_fstat_ref_row = _B._gb_serve_fstat_ref_row
    _fstat_ref_row_payload = staticmethod(_B._fstat_ref_row_payload)
    _fstat_ref_shard = staticmethod(_B._fstat_ref_shard)
    _fstat_ref_branch_from_payload = _B._fstat_ref_branch_from_payload
    _require_fstat_grid_move = _B._require_fstat_grid_move
    gf_serve = _B.gf_serve
    _fstat_ref_holder = None
    _fstat_ref_call = None
    _fstat_ref_walker = None
    _fstat_ref_branches = None
    branch_name = "gb"
    name = "gb_stub"

    def __init__(self, fanout, acs):
        self.fanout = fanout
        self._acs = acs

    @property
    def xp(self):
        return np

    def _bind_rank_acs(self, model):
        return self._acs

    @contextlib.contextmanager
    def _gb_free_residual(self, model, branches, walker_ref):
        # RESTORE BY ASSIGNMENT, not by ``-= 1.0``: ``(x + 1) - 1`` is not
        # ``x`` when the two straddle a binade, and the point of the
        # ``residual_restored`` assertion is that the body left the window,
        # not that floating-point addition is associative.
        rows = np.asarray(self._acs.linear_data_arr[0]).reshape(
            self._acs.acs_total_entries, -1)
        saved = np.array(rows[int(walker_ref)], copy=True)
        rows[int(walker_ref)] += 1.0
        self._gb_free_n_live = 7
        self._gb_free_opened = True
        try:
            yield
        finally:
            rows[int(walker_ref)] = saved


class RefRowReplicationTest(unittest.TestCase):
    """Spec verification 1(a): every rank gets the OWNER's windowed row."""

    def _run(self, n_compute, owner_rank_index, local_index):
        from lisatools.globalfit.communication import ranks as R
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.fanout import (
            ComputeService,
            WalkerFanout,
        )
        from tests.test_fstat_ref_row_holder import _FakeParent

        world = FakeWorld(n_compute + 1, timeout=20.0)
        B = 2

        def body(rank, comm):
            with warnings.catch_warnings():
                # n_compute == 1 is a `-n 2` launch on a one-device pool:
                # the size-2 saver fallback warns, and that is not this
                # test's subject.
                warnings.simplefilter("ignore", UserWarning)
                layout = R.build_layout(comm, B * n_compute,
                                        list(range(n_compute)))
            # ``make_fanout_comm`` is a COLLECTIVE ``Split``: every rank of
            # the world must enter it, the saver included, or the compute
            # ranks block in it forever.
            fcomm = layout.make_fanout_comm(comm)
            if rank not in layout.compute_ranks:
                return None
            acs = _FakeParent(B, 6, 9)
            # make every rank's buffers DIFFERENT so a missing broadcast
            # cannot pass by coincidence
            acs.linear_data_arr[0] += 100.0 * layout.fanout_rank(rank)
            acs.linear_psd_arr[0] += 100.0 * layout.fanout_rank(rank)
            before = np.array(acs.linear_data_arr[0], copy=True)
            fanout = WalkerFanout(fcomm, layout, rank, model=None)
            stub = _RefRowRankStub(fanout, acs)
            owner_rank = layout.compute_ranks[owner_rank_index]
            w_global = layout.block_of(owner_rank)[0] + local_index
            if rank == layout.head_rank:
                # ``gb_free`` is what the HEAD asked for, and this bare
                # ``fanout.run`` ships no branch -- so ``False`` is the
                # honest value and the owner's "asked for a window, got no
                # branch" WARNING is not provoked.
                try:
                    fanout.run(
                        "gb_fstat_ref_row", move="gb_pe",
                        per_rank_payload=lambda r, w0, w1:
                            stub._fstat_ref_row_payload(
                                w_global, owner_rank, local_index, False),
                        local_body=lambda p, _m: stub.gf_serve(
                            "gb_fstat_ref_row", p, fanout.clock, None),
                        merge=lambda r: r)
                finally:
                    # WITHOUT the finally a head-side failure leaves every
                    # worker parked in ``serve()`` and the whole world dies
                    # on the timeout instead of reporting the real error.
                    fanout.stop()
            else:
                service = ComputeService(fcomm, layout, rank,
                                         registry={"gb_pe": stub}, model=None)
                service.serve()
            after = np.asarray(acs.linear_data_arr[0])
            return {
                "data_row": np.asarray(
                    stub._fstat_ref_holder.linear_data_arr[0]).copy(),
                "psd_row": np.asarray(
                    stub._fstat_ref_holder.linear_psd_arr[0]).copy(),
                "walker_ref": stub._fstat_ref_walker,
                "residual_restored": np.array_equal(before, after),
                "expected": (
                    np.asarray(before).reshape(B, -1)[local_index] + 1.0
                    if rank == owner_rank else None),
                "expected_psd": (
                    np.asarray(acs.linear_psd_arr[0]).reshape(B, -1)[local_index]
                    if rank == owner_rank else None),
            }

        return world.run(body)

    def test_every_rank_holds_the_owners_windowed_row(self):
        for n_compute, owner_idx, local in ((2, 1, 0), (2, 0, 1), (3, 2, 1)):
            with self.subTest(n_compute=n_compute, owner=owner_idx):
                out = {r: v for r, v in self._run(n_compute, owner_idx, local).items()
                       if v is not None}
                self.assertEqual(len(out), n_compute)
                expected = next(v["expected"] for v in out.values()
                                if v["expected"] is not None)
                expected_psd = next(v["expected_psd"] for v in out.values()
                                    if v["expected_psd"] is not None)
                for rank, v in out.items():
                    np.testing.assert_array_equal(
                        v["data_row"], expected,
                        f"rank {rank} did not receive the owner's row")
                    np.testing.assert_array_equal(
                        v["psd_row"], expected_psd,
                        f"rank {rank} did not receive the owner's invC row")
                    self.assertTrue(v["residual_restored"],
                                    f"rank {rank}'s live residual was left mutated")
                self.assertEqual(
                    len({v["walker_ref"] for v in out.values()}), 1,
                    "every rank must record the same GLOBAL reference walker")

    def test_single_compute_rank_needs_no_collective(self):
        out = {r: v for r, v in self._run(1, 0, 1).items() if v is not None}
        self.assertEqual(len(out), 1)
        v = next(iter(out.values()))
        np.testing.assert_array_equal(v["data_row"], v["expected"])
        self.assertTrue(v["residual_restored"])


class ParallelStageBGateTest(unittest.TestCase):
    """THE acceptance gate: 2 (and 3) ranks == 1 rank, byte for byte."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _fanout_run(self, n_compute, tmpdir, *, grouped=True):
        """Run the grouped stage B over ``n_compute`` FakeWorld ranks."""
        from lisatools.globalfit.communication import ranks as R
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.fanout import (
            ComputeService,
            WalkerFanout,
        )

        world = FakeWorld(n_compute + 1, timeout=300.0)

        def body(rank, comm):
            layout = R.build_layout(comm, 2 * n_compute, list(range(n_compute)))
            # a COLLECTIVE ``Split``: every rank enters it, saver included
            fcomm = layout.make_fanout_comm(comm)
            if rank not in layout.compute_ranks:
                return None
            fanout = WalkerFanout(fcomm, layout, rank, model=None)
            stub = _StageBRankStub(fanout, rank=rank)
            if rank != layout.head_rank:
                ComputeService(fcomm, layout, rank,
                               registry={None: stub, "gb_pe": stub},
                               model=None).serve()
                return None
            # The scorer the runner's guard compares against must BE the
            # rank's cached holder-scored call: built here through the
            # production chokepoint, exactly as ``_run_fstat_fit`` builds it
            # before entering ``run_fstat_grid_fit``.
            call_fstat = stub._fstat_holder_call(None)
            try:
                return run_golden(tmpdir, grouped=grouped,
                                  call_fstat=call_fstat,
                                  sweep_runner=stub._fstat_stage_b_runner(None))
            finally:
                fanout.stop()

        return next(v for v in world.run(body).values() if v is not None)

    def test_two_ranks_are_bit_identical_to_the_serial_fit(self):
        got = self._fanout_run(2, self.d, grouped=True)
        assert_npz_identical(self, got, GOLDEN_GROUPED)

    def test_three_ranks_are_bit_identical_too(self):
        got = self._fanout_run(3, self.d, grouped=True)
        assert_npz_identical(self, got, GOLDEN_GROUPED)

    def test_single_group_split_is_bit_identical(self):
        got = self._fanout_run(2, self.d, grouped=False)
        assert_npz_identical(self, got, GOLDEN_SINGLE)

    def test_partials_are_deleted_after_assembly(self):
        self._fanout_run(2, self.d, grouped=True)
        parts = os.path.join(self.d, "fstat_grid_parts")
        leftovers = [f for f in os.listdir(parts) if f.endswith(".npy")] \
            if os.path.isdir(parts) else []
        self.assertEqual(leftovers, [], f"stage-B partials left behind: {leftovers}")

    def _flaky_group(self, gi, rank_suffix):
        return _kill_stage_b_once(rank_suffix, gi=gi)

    def test_resume_with_the_same_rank_count_reuses_the_checkpoints(self):
        """A rank that dies mid-sweep resumes from its own progress file.

        Group 0 finishes on BOTH ranks before rank 1 dies in group 1, so the
        rerun has real completed work to reuse -- ``ckpt_clear`` only runs
        at the end of a whole successful stage B, which is exactly what did
        not happen here.
        """
        patch, fired = self._flaky_group(1, "_r1")
        with patch:
            with self.assertRaises(Exception):
                self._fanout_run(2, self.d, grouped=True)
        self.assertTrue(fired.is_set(), "the simulated death never fired")
        # the progress files from the surviving sweeps must still be there
        parts = os.path.join(self.d, "fstat_grid_parts")
        progress = [f for f in os.listdir(parts) if f.endswith(".progress.npz")]
        self.assertTrue(progress, "per-rank checkpoints must survive a death")
        self.assertTrue(any("_r" in f for f in progress),
                        f"checkpoints must be per rank, got {progress}")
        with self.assertLogs(G.logger, level="INFO") as captured:
            got = self._fanout_run(2, self.d, grouped=True)
        self.assertTrue(
            any("[ckpt] resuming" in m for m in captured.output),
            "the same rank count must RESUME the per-rank checkpoints, not "
            "silently redo them")
        assert_npz_identical(self, got, GOLDEN_GROUPED)

    def test_a_different_rank_count_restarts_the_stale_checkpoints(self):
        """Spec verification 1(b), second half: a changed ``n_compute``.

        The per-rank checkpoint FINGERPRINT hashes the SLICED inputs and
        ``node_shape``, so a different split makes every surviving progress
        file invalid -- and the fit must say so and recompute, never stitch
        a 2-way slice's rows into a 3-way one.
        """
        patch, fired = self._flaky_group(1, "_r1")
        with patch:
            with self.assertRaises(Exception):
                self._fanout_run(2, self.d, grouped=True)
        self.assertTrue(fired.is_set())
        with self.assertLogs(G.logger, level="INFO") as captured:
            got = self._fanout_run(3, self.d, grouped=True)
        self.assertTrue(
            any("restarting this sweep" in m for m in captured.output),
            "a 2-rank checkpoint must be refused by a 3-rank sweep")
        assert_npz_identical(self, got, GOLDEN_GROUPED)

    def test_a_different_rank_count_on_a_clean_tree_is_identical_too(self):
        self._fanout_run(2, self.d, grouped=True)
        os.remove(os.path.join(self.d, "fstat_grid_peaks_stacked.npz"))
        got = self._fanout_run(3, self.d, grouped=True)
        assert_npz_identical(self, got, GOLDEN_GROUPED)

    def test_the_workers_go_through_the_real_dispatcher(self):
        """Not a stub ``gf_serve``: the command is dispatched by the
        production one, past ``_require_fstat_grid_move``, on every rank."""
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        self.assertIn("gb_fstat_stage_b", gbs._FSTAT_OP_REQUIRES)
        stub = _StageBRankStub(None)
        stub._fstat_call = None            # a GB move that cannot fit a grid
        with self.assertRaises(ValueError) as ctx:
            stub.gf_serve("gb_fstat_stage_b", {}, {}, None)
        self.assertIn("_fstat_call", str(ctx.exception))


# -------------------------------------------------------------------------
# The WHOLE epoch fit over a FakeWorld: ``_run_fstat_fit`` end to end.
#
# The stage-B gate above pins the SPLIT. This arm pins the FIT: the global
# reference, the replicated row (including a WORKER-owned reference walker
# and the block-sliced branch that reaches its GB-free window), the real
# ``run_fstat_grid_fit`` on top of the split, the manifest, and the release.
# -------------------------------------------------------------------------

#: 6 sub-bands spanning BOTH of the fixture's amplitude bumps (9.0 and
#: 15.0 mHz), so the comb finds more than one peak box and
#: ``split_box_range`` has something to split -- while a 0.02 mHz comb over
#: 10 mHz keeps stage A (head-only, real) at milliseconds.
EPOCH_BAND_EDGES = np.linspace(7.0e-3, 17.0e-3, 7)


@contextlib.contextmanager
def epoch_fit_env(**overrides):
    """Pin every knob the WHOLE fit reads (stage A included).

    Set by the test, OUTSIDE the ``FakeWorld``: ``os.environ`` is process
    global and every rank is a thread of this process, so a rank that read
    it while the head was still setting it would sweep a different grid.
    """
    env = {
        "FSTAT_BATCH": "512",
        "FSTAT_CKPT_SECS": "0",
        "FSTAT_F0_SPACING_MHZ": "0.02",
        "FSTAT_COMB_NSKY": "2",
        "FSTAT_PEAK_MIN_F": "10",
        "FSTAT_PEAKS_PER_BAND": "2",
        "FSTAT_N_MC": "2",
        "FSTAT_N_ALPHA": "2",
        "FSTAT_N_SINDELTA": "2",
        "FSTAT_PEAK_HALF_MHZ": "0.05",
        "FSTAT_FDOT_AXIS": "0",
        "FSTAT_MC_GROUPING": "1",
        "FSTAT_PEAK_WEIGHTING": "fstat",
        "FSTAT_GRID_MEM_MB": "",
        "FSTAT_PEAKS_TO_FIT": "",
        "FSTAT_MC_MIN": "",
        "FSTAT_MC_ETA": "",
        "FSTAT_N_F0": "",
        "FSTAT_N_PER_AXIS": "",
        "GB_FSTAT_GB_FREE": "1",
        "GB_FSTAT_CTR_MODE": "epoch",
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


def _epoch_global_state(nwalkers, drow=6, prow=9, seed=17):
    """``(residual rows, invC rows, lnL)`` for the WHOLE ensemble.

    ``lls`` ascends, so the global argmax is the LAST walker -- under a real
    fan-out that walker is owned by a WORKER, which is the case the design
    exists for (the head ships the owner a block-sliced branch and the
    ``Bcast`` root is not the head).
    """
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(int(nwalkers), int(drow))),
            rng.normal(size=(int(nwalkers), int(prow))),
            np.arange(int(nwalkers), dtype=float))


def _epoch_branches(nwalkers, seed=5):
    from eryn.state import Branch

    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(2, int(nwalkers), 2, 9))
    inds = rng.random((2, int(nwalkers), 2)) > 0.3
    return {"gb": Branch(coords, inds=inds)}


class _EpochAcs:
    """Flat per-walker residual + inverse-PSD buffers; this rank's block."""

    gpus = None
    device = None
    nchannels = 3
    shape_sens = (3, 3)
    psd_row_index = None

    def __init__(self, rows_d, rows_p, lls, offset=0.0):
        self.acs_total_entries = int(np.shape(rows_d)[0])
        self.linear_data_arr = [np.ascontiguousarray(
            np.asarray(rows_d, dtype=float).reshape(-1) + float(offset))]
        self.linear_psd_arr = [np.ascontiguousarray(
            np.asarray(rows_p, dtype=float).reshape(-1) + float(offset))]
        self._lls = np.asarray(lls, dtype=float)

    @property
    def xp(self):
        return np

    def likelihood(self, complex=False):
        return self._lls


class _EpochModel:
    def __init__(self, acs):
        self.analysis_container_arr = acs


class _EpochFitRankStub:
    """A stand-in move exposing the WHOLE epoch-fit surface, head and rank.

    Everything on the class body is BORROWED from production: the head's
    ``_run_fstat_fit`` and ``_install_ctr_table``, the global-reference
    gather, the reference-row driver and its served body, the stage-B runner
    and its served body, the release driver and its served body,
    ``gf_serve`` with its ``_require_fstat_grid_move`` guard, and
    ``_fanout_cmd``. ``run_fstat_grid_fit`` underneath is the real one too,
    comb scan and all.

    The stubs are exactly the GPU-bound halves: ``_bind_rank_acs`` (a flat
    numpy stand-in ACA), ``_gb_free_residual`` (a +1.0 window that restores
    by assignment and RECORDS the branch column it was given), and
    ``_fstat_call`` -- the analytic fixture scorer KEYED BY THE ROW it is
    handed (:func:`_row_keyed_call_fstat`). That keying is what makes "every
    rank scored the OWNER's replicated row" a byte-level assertion: a rank
    that scored its own residual instead produces a different grid, and the
    bit-identity gate fails.
    """

    from lisatools.globalfit.moves.gbspecialstretch import (
        GBSpecialBase as _B,
        GBSpecialRJFStatGridMove as _G,
    )

    _run_fstat_fit = _G._run_fstat_fit
    _install_ctr_table = _G._install_ctr_table
    _fstat_ctr_mode = staticmethod(_G._fstat_ctr_mode)
    _fstat_ctr_smear = _G._fstat_ctr_smear
    _CTR_TABLE_DEVICE_FIELDS = _G._CTR_TABLE_DEVICE_FIELDS
    _fstat_global_reference = _B._fstat_global_reference
    _fstat_ref_row_payload = staticmethod(_B._fstat_ref_row_payload)
    _fstat_ref_shard = staticmethod(_B._fstat_ref_shard)
    _fstat_ref_branch_slice = _B._fstat_ref_branch_slice
    _fstat_ref_branch_from_payload = _B._fstat_ref_branch_from_payload
    _gb_serve_fstat_ref_row = _B._gb_serve_fstat_ref_row
    _fstat_ref_row_fanout = _B._fstat_ref_row_fanout
    _warn_if_gb_free_missed = _B._warn_if_gb_free_missed
    _fstat_holder_call = _B._fstat_holder_call
    _fstat_stage_b_payload = staticmethod(_B._fstat_stage_b_payload)
    _fstat_stage_b_spec = staticmethod(_B._fstat_stage_b_spec)
    _gb_serve_fstat_stage_b = _B._gb_serve_fstat_stage_b
    _fstat_stage_b_runner = _B._fstat_stage_b_runner
    _gb_serve_fstat_release = _B._gb_serve_fstat_release
    _fstat_release_ref_row = _B._fstat_release_ref_row
    _fstat_release_fanout = _B._fstat_release_fanout
    _require_fstat_grid_move = _B._require_fstat_grid_move
    _fanout_cmd = _B._fanout_cmd
    _gb_session_token = staticmethod(_B._gb_session_token)
    _rank_tag = _B._rank_tag
    fanout_active = _B.fanout_active
    name = "gb_stub"
    gf_move_name = "gb_pe"
    branch_name = "gb"
    num_proposals = 0

    def __init__(self, fanout, acs, cache_root, *, rank=None):
        self.fanout = fanout
        self._acs = acs
        self.cache_root = cache_root
        self.gf_rank = rank
        self.mempool = _CountingMempool()
        self.ops_served = []
        self.window_log = []
        self.held_after_ref_row = None
        self.nwalkers = None
        self.ntemps = None
        self._prop_timer = None
        self._fstat_ctr_table = None
        self._fstat_ref_holder = None
        self._fstat_ref_call = None
        self._fstat_ref_walker = None
        self._fstat_ref_branches = None
        self.band_edges = EPOCH_BAND_EDGES
        self.df = 1.0 / TOBS
        self.fstat_fit_kwargs = {"mc_lims": [0.01, 1.0]}

    @property
    def xp(self):
        return np

    def _epoch_dir(self, k):
        return os.path.join(self.cache_root, f"epoch_{int(k):04d}")

    def _fstat_clock(self):
        return 0

    def _bind_rank_acs(self, model):
        return self._acs

    def _fstat_reference_walker(self, model):
        return int(np.argmax(np.asarray(self._acs.likelihood())))

    def _fstat_call(self, model, walker_ref, *, holder=None):
        if holder is None:
            raise AssertionError(
                "the epoch scorer must be built from the replicated row")
        return _row_keyed_call_fstat(holder.linear_data_arr[0])

    @contextlib.contextmanager
    def _gb_free_residual(self, model, branches, walker_ref):
        branch = None if not branches else branches.get(self.branch_name)
        self.window_log.append({
            "walker": int(walker_ref),
            "coords": (None if branch is None else np.array(
                np.asarray(branch.coords)[0, int(walker_ref)], copy=True)),
        })
        rows = np.asarray(self._acs.linear_data_arr[0]).reshape(
            self._acs.acs_total_entries, -1)
        saved = np.array(rows[int(walker_ref)], copy=True)
        rows[int(walker_ref)] += 1.0
        self._gb_free_n_live = 3
        self._gb_free_opened = True
        try:
            yield
        finally:
            rows[int(walker_ref)] = saved

    def gf_serve(self, op, payload, clock, model):
        """Record the op, then run the PRODUCTION dispatcher unchanged."""
        self.ops_served.append(op)
        out = self._B.gf_serve(self, op, payload, clock, model)
        if op == "gb_fstat_ref_row":
            self.held_after_ref_row = self._fstat_ref_holder is not None
        return out


def run_epoch_fit_world(n_compute, nwalkers, cache_root, *, epoch=0,
                        after_fit=None):
    """One whole ``_run_fstat_fit`` over ``n_compute`` FakeWorld ranks.

    Returns ``{world rank: dict}``. ``after_fit(move, model, branches)`` runs
    on the HEAD after the fit and before the fan-out is stopped, so a follow
    on head-side command (the centre table's fallback) still has its ranks.
    """
    from lisatools.globalfit.communication import ranks as R
    from lisatools.globalfit.communication.fakecomm import FakeWorld
    from lisatools.globalfit.communication.fanout import (
        LIKELIHOOD_OP,
        ComputeService,
        WalkerFanout,
    )

    world = FakeWorld(n_compute + 1 if n_compute > 1 else 1, timeout=300.0)
    rows_d, rows_p, lls = _epoch_global_state(nwalkers)
    branches = _epoch_branches(nwalkers)

    def body(rank, comm):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            layout = R.build_layout(comm, nwalkers, list(range(n_compute)))
        fcomm = layout.make_fanout_comm(comm)
        if rank not in layout.compute_ranks:
            return None
        w0, w1 = layout.block_of(rank)
        ri = layout.fanout_rank(rank)
        # REPLICA MODE: every rank holds the same single walker, so a rank
        # that scored its OWN residual instead of the broadcast one would be
        # invisible. Offsetting the replicas' buffers is what makes the
        # bit-identity gate detect it (production replicas do hold equal
        # residuals; this is the deliberate negative control).
        acs = _EpochAcs(rows_d[w0:w1], rows_p[w0:w1], lls[w0:w1],
                        offset=(100.0 * ri if layout.replica_mode else 0.0))
        fanout = WalkerFanout(fcomm, layout, rank, model=None)
        move = _EpochFitRankStub(fanout, acs, cache_root, rank=rank)
        model = _EpochModel(acs)
        facts = {
            "replica_mode": bool(layout.replica_mode),
            "n_replicas": int(layout.n_replicas),
            "n_compute": int(layout.n_compute),
            "block": layout.block_of(rank),
        }
        if rank != layout.head_rank:
            ComputeService(
                fcomm, layout, rank, registry={None: move, "gb_pe": move},
                model=model,
                builtins={LIKELIHOOD_OP: (
                    lambda p, c, m, _a=acs: np.asarray(
                        _a.likelihood(complex=False)))},
            ).serve()
            return dict(facts, role="worker", ops=list(move.ops_served),
                        held_after_ref_row=move.held_after_ref_row,
                        holder=move._fstat_ref_holder,
                        call=move._fstat_ref_call,
                        walker=move._fstat_ref_walker,
                        window=list(move.window_log))
        try:
            _stacked, n_peaks = move._run_fstat_fit(model, epoch, branches)
            held = move._fstat_ref_holder is not None
            extra = (None if after_fit is None
                     else after_fit(move, model, branches))
        finally:
            fanout.stop()
        # setup()'s own ``finally``, which is what finally drops the head's
        cache_dir = move._epoch_dir(epoch)
        move._fstat_release_ref_row()
        with open(os.path.join(cache_dir, "DONE.json")) as f:
            manifest = json.load(f)
        return dict(facts, role="head", n_peaks=int(n_peaks),
                    path=G.stacked_grid_path(cache_dir), manifest=manifest,
                    owner_of_argmax=layout.owner_of(int(np.argmax(lls))),
                    head_rank=int(layout.head_rank),
                    holder_after_fit=held,
                    holder_after_release=move._fstat_ref_holder,
                    ops=list(move.ops_served), extra=extra,
                    window=list(move.window_log))

    return world.run(body)


class EpochFitGateTest(unittest.TestCase):
    """The whole fit: N compute ranks == 1, byte for byte, plus the manifest."""

    def setUp(self):
        self.a = tempfile.mkdtemp()
        self.b = tempfile.mkdtemp()

    def tearDown(self):
        for d in (self.a, self.b):
            shutil.rmtree(d, ignore_errors=True)

    def test_two_walker_blocks_match_the_one_rank_fit(self):
        """Spec verification 1(a) + (c) at the level of the WHOLE fit.

        The global argmax is the LAST walker, which a 2-rank layout puts on
        the WORKER -- so this is the path where the head ships a block-sliced
        branch to a rank that has none, the owner opens the GB-free window
        there, and the ``Bcast`` root is not the head. The 1-rank arm reaches
        the same walker through its own ACA with no wire at all.
        """
        with epoch_fit_env():
            serial = run_epoch_fit_world(1, 4, self.a)
            parallel = run_epoch_fit_world(2, 4, self.b)
        s_head = next(v for v in serial.values() if v and v["role"] == "head")
        p_head = next(v for v in parallel.values() if v and v["role"] == "head")
        self.assertGreaterEqual(
            s_head["n_peaks"], 2,
            "the fixture must produce more than one peak box, or "
            "split_box_range has nothing to split and the gate is vacuous")
        assert_npz_identical(self, p_head["path"], s_head["path"])
        self.assertEqual(s_head["manifest"]["walker_ref"], 3)
        self.assertEqual(p_head["manifest"]["walker_ref"], 3)
        self.assertEqual(s_head["manifest"]["n_compute"], 1)
        self.assertEqual(p_head["manifest"]["n_compute"], 2)
        # the reference walker's owner is a WORKER, and the window opened
        # there on the shipped slice -- at the OWNER's LOCAL row
        self.assertNotEqual(p_head["owner_of_argmax"][0], p_head["head_rank"])
        worker = next(v for v in parallel.values()
                      if v and v["role"] == "worker" and v["window"])
        self.assertEqual(len(worker["window"]), 1)
        self.assertEqual(worker["window"][0]["walker"],
                         p_head["owner_of_argmax"][1])
        # ... and it is the SAME branch column the serial fit's window saw
        np.testing.assert_array_equal(worker["window"][0]["coords"],
                                      s_head["window"][0]["coords"])
        # the head opened no window of its own under the fan-out
        self.assertEqual(p_head["window"], [])

    def test_the_workers_are_released_and_the_head_keeps_its_row(self):
        with epoch_fit_env():
            out = run_epoch_fit_world(2, 4, self.b)
        head = next(v for v in out.values() if v and v["role"] == "head")
        workers = [v for v in out.values() if v and v["role"] == "worker"]
        self.assertTrue(workers)
        for w in workers:
            self.assertEqual(w["ops"][0], "gb_fstat_ref_row")
            self.assertEqual(w["ops"][-1], "gb_fstat_release")
            self.assertTrue(w["held_after_ref_row"],
                            "the worker never built a holder")
            self.assertIsNone(w["holder"], "a worker kept its reference row")
            self.assertIsNone(w["call"], "a worker kept its sig-het scorer")
        # the head's own survives the fit (the centre table still needs it)
        # and goes only at the setup()-level release
        self.assertTrue(head["holder_after_fit"])
        self.assertIsNone(head["holder_after_release"])

    @staticmethod
    def _walker_zero_reference():
        """Pretend the global argmax moved to walker 0 across the restart."""
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        real_ref = gbs.GBSpecialBase._fstat_global_reference

        def moved(self_, model):
            _w, _o, _l, lls = real_ref(self_, model)
            fan = getattr(self_, "fanout", None)
            owner, local = (0, 0) if fan is None else fan.layout.owner_of(0)
            return 0, int(owner), int(local), lls

        return mock.patch.object(
            _EpochFitRankStub, "_fstat_global_reference", moved)

    def _die_mid_fit(self, cache_root):
        """Leave a half-finished epoch behind: surviving per-rank checkpoints."""
        patch, fired = _kill_stage_b_once("_r1")
        with patch:
            with self.assertRaises(Exception):
                run_epoch_fit_world(2, 4, cache_root)
        self.assertTrue(fired.is_set(), "the simulated death never fired")
        parts = os.path.join(cache_root, "epoch_0000", "fstat_grid_parts")
        progress = [f for f in os.listdir(parts)
                    if f.endswith(".progress.npz")]
        self.assertTrue(progress, "no per-rank checkpoint survived the death")
        return progress

    def test_the_same_reference_walker_resumes_its_checkpoints(self):
        with epoch_fit_env():
            self._die_mid_fit(self.a)
            with self.assertLogs(G.logger, level="INFO") as captured:
                out = run_epoch_fit_world(2, 4, self.a)
        head = next(v for v in out.values() if v and v["role"] == "head")
        self.assertEqual(head["manifest"]["walker_ref"], 3)
        self.assertTrue(
            any("[ckpt] resuming" in m for m in captured.output),
            "an unchanged reference must RESUME the surviving checkpoints")
        self.assertFalse(
            any("restarting this sweep" in m for m in captured.output),
            "an unchanged reference must not invalidate its own cache")

    def test_a_different_reference_walker_restarts_the_sweep(self):
        """Task 9 concern 5: ``walker_ref`` is part of the cache fingerprint.

        A restart of the SAME epoch whose global argmax has MOVED must not
        reuse the earlier walker's stage-B checkpoints -- those rows were
        scored against a different residual, and stitching them into the new
        walker's grid is silent. Under the walker-block layout the global
        argmax is exactly the volatile quantity, which is why this is
        enforced and not merely recorded in ``DONE.json``.
        """
        with epoch_fit_env():
            self._die_mid_fit(self.b)
            with self._walker_zero_reference():
                with self.assertLogs(G.logger, level="INFO") as captured:
                    out = run_epoch_fit_world(2, 4, self.b)
        head = next(v for v in out.values() if v and v["role"] == "head")
        self.assertEqual(head["manifest"]["walker_ref"], 0)
        self.assertTrue(
            any("restarting this sweep" in m for m in captured.output),
            "a CHANGED reference walker must invalidate the epoch's "
            "checkpoints -- otherwise the new walker's grid is stitched "
            "out of the old walker's rows")
        self.assertFalse(
            any("[ckpt] resuming" in m for m in captured.output),
            "nothing of the old walker's sweep may be reused")

    def test_the_fingerprint_names_the_reference_walker(self):
        import inspect

        from lisatools.globalfit.moves import gbspecialstretch as gbs

        src = inspect.getsource(gbs.GBSpecialRJFStatGridMove._run_fstat_fit)
        self.assertIn("wref={w_global}", src)


class ReplicaModeEpochFitTest(unittest.TestCase):
    """One-walker replica mode (dev's ``GF_ONE_WALKER_REPLICAS``).

    ``nwalkers == 1`` on several compute ranks: every rank's block is
    ``(0, 1)``, ``owner_of(0)`` is the HEAD at local row 0, and the replicas
    split stage B between them. The replicas' residual buffers are
    deliberately made to DIFFER here, so a rank that scored its own row
    instead of the broadcast one breaks the bit-identity gate.
    """

    def setUp(self):
        self.a = tempfile.mkdtemp()
        self.b = tempfile.mkdtemp()

    def tearDown(self):
        for d in (self.a, self.b):
            shutil.rmtree(d, ignore_errors=True)

    def test_two_replicas_are_bit_identical_to_the_one_rank_fit(self):
        with epoch_fit_env():
            serial = run_epoch_fit_world(1, 1, self.a)
            replicas = run_epoch_fit_world(2, 1, self.b)
        s_head = next(v for v in serial.values() if v and v["role"] == "head")
        r_head = next(v for v in replicas.values() if v and v["role"] == "head")

        self.assertFalse(s_head["replica_mode"], "one rank is not replicated")
        self.assertTrue(r_head["replica_mode"])
        self.assertEqual(r_head["n_replicas"], 2)
        for v in replicas.values():
            if v:
                self.assertEqual(v["block"], (0, 1))
        # the owner of the single walker is the HEAD, at local row 0
        self.assertEqual(r_head["owner_of_argmax"],
                         (r_head["head_rank"], 0))

        self.assertGreater(s_head["n_peaks"], 0)
        assert_npz_identical(self, r_head["path"], s_head["path"])
        self.assertEqual(r_head["manifest"]["walker_ref"], 0)
        self.assertEqual(r_head["manifest"]["n_compute"], 2)
        self.assertEqual(s_head["manifest"]["n_compute"], 1)

        workers = [v for v in replicas.values() if v and v["role"] == "worker"]
        self.assertEqual(len(workers), 1)
        for w in workers:
            self.assertTrue(w["held_after_ref_row"])
            self.assertIsNone(w["holder"], "a replica kept its row")
            self.assertIsNone(w["call"], "a replica kept its scorer")
            self.assertEqual(w["window"], [],
                             "only the OWNER opens the GB-free window")
        self.assertTrue(r_head["holder_after_fit"])
        self.assertIsNone(r_head["holder_after_release"])


class CentreTableFallbackOverTheWireTest(unittest.TestCase):
    """Task 9 concern 1, over a REAL fan-out.

    ``_install_ctr_table``'s fallback -- a load/reuse decision meeting a
    MISSING centre table -- replicates the reference row itself. Unit tests
    pin the call and its arguments; this drives it over two FakeWorld ranks,
    which is the shape that would HANG (a ``gb_fstat_ref_row`` from
    ``setup()`` on a rank that is not parked in ``serve()``) if the wiring
    were wrong.
    """

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        gbs._FSTAT_CTR_TABLE_REGISTRY.pop(
            os.path.join(self.d, "epoch_0000"), None)
        shutil.rmtree(self.d, ignore_errors=True)

    def test_the_fallback_scores_through_a_row_it_replicates_and_releases(self):
        captured = {}
        current = {}

        def fake_build(call_fstat, **kw):
            """Record WHAT the centre sweep was handed, and at which walker.

            Object identity against ``_fstat_ref_call`` is the assertion
            that matters (spec decision 6: literally the same scorer), and
            it has to be taken HERE -- by the time ``_install_ctr_table``
            returns, the fallback has already dropped both halves.
            """
            move = current["move"]
            captured["is_ref_call"] = call_fstat is move._fstat_ref_call
            captured["walker"] = move._fstat_ref_walker
            captured["holder"] = move._fstat_ref_holder is not None
            captured["cache_dir"] = kw.get("cache_dir")
            return {name: np.zeros(2)
                    for name in _EpochFitRankStub._CTR_TABLE_DEVICE_FIELDS}

        def after_fit(move, model, branches):
            """The SECOND fit call: npz complete, centre table still absent."""
            current["move"] = move
            move._fstat_release_ref_row()      # setup()'s own finally fired
            move._run_fstat_fit(model, 0, branches)
            holder_after_second_fit = move._fstat_ref_holder
            before = len(move.ops_served)
            with mock.patch.object(G, "build_fstat_center_table", fake_build):
                move._install_ctr_table(0, model=model, branches=branches)
            return {
                "holder_after_second_fit": holder_after_second_fit,
                "ctr_table": move._fstat_ctr_table is not None,
                "holder_after": move._fstat_ref_holder,
                "new_ops": move.ops_served[before:],
                "captured": dict(captured),
            }

        with epoch_fit_env():
            out = run_epoch_fit_world(2, 4, self.d, after_fit=after_fit)
        head = next(v for v in out.values() if v and v["role"] == "head")
        extra = head["extra"]
        # the second fit found a complete npz: no row, no scorer, no release
        self.assertIsNone(extra["holder_after_second_fit"],
                          "a complete stage-B npz must cost no broadcast")
        cap = extra["captured"]
        self.assertTrue(cap, "the centres were never swept")
        self.assertTrue(cap["holder"], "the fallback replicated no row")
        self.assertTrue(cap["is_ref_call"],
                        "the centres must be scored through THE holder call")
        self.assertEqual(cap["walker"], 3,
                         "the centres must score the fit's GLOBAL reference")
        self.assertEqual(cap["cache_dir"], os.path.join(self.d, "epoch_0000"))
        self.assertTrue(extra["ctr_table"])
        # the row it took is put back on this rank ...
        self.assertIsNone(extra["holder_after"])
        # ... and the workers were told, over the wire, to drop theirs
        self.assertEqual(extra["new_ops"],
                         ["gb_fstat_ref_row", "gb_fstat_release"])
        for v in out.values():
            if v and v["role"] == "worker":
                self.assertIsNone(v["holder"], "a worker kept the centre row")
                self.assertIsNone(v["call"])
                self.assertEqual(v["ops"][-2:],
                                 ["gb_fstat_ref_row", "gb_fstat_release"])


if __name__ == "__main__":
    unittest.main()
