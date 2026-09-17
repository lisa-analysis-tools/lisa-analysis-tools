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
        self.assertIn("group 1", msg)
        self.assertIn("2", msg)
        self.assertIn("3", msg)
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
        self.assertIn("5", msg)
        self.assertIn("4", msg)

    def test_valid_shapes_write_cleanly(self):
        """The happy path must still write -- validation is not overzealous."""
        grids_g = [np.zeros((2, 2, 2, 2, 2)), np.zeros((3, 2, 2, 2, 2))]
        mc_ax_g = [np.linspace(0.1, 1.0, 2), np.linspace(0.1, 1.0, 2)]
        out = os.path.join(self.d, "out.npz")
        G.write_stacked_npz(
            out, grids_g=grids_g, mc_ax_g=mc_ax_g, group_sizes=[2, 3],
            **self._common_kwargs(5))
        self.assertTrue(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
