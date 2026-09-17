"""The GB per-propose rank session and its three ``gf_serve`` commands.

Skeleton-level: a bare ``GBSpecialBase`` carrying only the attributes the
rank plumbing reads, with the three device-binding calls
(``pin_main_device`` / ``_configure_domain`` / ``_bind_parent_acs``)
patched out. Nothing here needs a GPU, a build or a ``BandSorter`` -- the
neutral path is exactly the "head decided this block has nothing to do"
branch, which must still reply with correctly shaped arrays so the
command count stays symmetric across ranks.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves import gbspecialstretch as gbs
from lisatools.globalfit.moves.globalfitmove import GlobalFitMove
from lisatools.globalfit.state import GBState, GFState

NTEMPS, B, NLEAVES, NDIM = 2, 3, 2, 8
BAND_EDGES = np.linspace(1e-3, 2e-3, 5)
NUM_BANDS = len(BAND_EDGES) - 1


class _FakeFanout:
    def __init__(self, single, is_head):
        self.single = single
        self.is_head = is_head


class _Curr:
    def __init__(self, fanout, rank):
        self.fanout = fanout
        self.rank = rank


class _FakeACS:
    gpus = None


class _FakeModel:
    def __init__(self):
        self.analysis_container_arr = _FakeACS()


def make_move(cls=gbs.GBSpecialBase):
    """A skeleton GB move with only the attributes the rank plumbing reads."""
    move = cls.__new__(cls)
    move.branch_name = "gb"
    move.name = "gb_test"
    # ``xp`` / ``backend`` are derived properties (deepcopy-safety rule)
    move._backend_name = "lisatools_cpu"
    move.band_edges = BAND_EDGES
    move.num_bands = NUM_BANDS
    move.nwalkers = 4
    move.ntemps = 2
    move.time = 3
    move.num_proposals = 7
    move.is_rj_prop = False
    move.use_prior_removal = False
    move.rj_replace = False
    move._reseed_firing = False
    move.temper_vertical = False
    move._cap_leaf_cap = None
    move._band_leaf_cap = None
    move._rj_band_shutoff = None
    move._gb_session = None
    move.reset_non_gb_linear_data_arr = None
    move.fanout = None
    move.gf_rank = None
    move.eigen_store_path = "x.h5"
    move.mempool = gbs._NoOpMempool()
    # the three heavy binding calls are no-ops on a skeleton
    move._configure_domain = lambda acs: None
    move._bind_parent_acs = lambda acs: None
    return move


def make_grid_move(epoch_complete=True, root="/nowhere"):
    """An F-stat grid move whose install side effects are recorded, not run.

    ``_setup_from_directive`` probes the epoch directory before installing
    anything: ``_epoch_missing_for_ranks`` (``DONE.json`` AND the stage-B
    npz, the zero-peak manifest aside) and, when the head asks for the
    center table, ``fstat_centers.npz``. ``epoch_complete`` stubs the first
    (``True``/``False``); the second is a REAL ``os.path.exists`` against
    ``root``, so a test that wants it present points ``root`` at a tmpdir
    and writes the file.

    ``epoch_complete=None`` leaves BOTH completeness predicates as the real
    implementations, for the tests that exercise them against a directory.
    """
    move = make_move(gbs.GBSpecialRJFStatGridMove)
    move.installs, move.ctr_installs = [], []
    move._install = lambda k, **kw: move.installs.append((k, kw))
    move._install_ctr_table = lambda k, model=None, branches=None: (
        move.ctr_installs.append((k, model)))
    move._epoch_dir = lambda k: os.path.join(root, f"epoch_{int(k):04d}")
    move._epoch_fit_clock = lambda k: 0
    if epoch_complete is not None:
        move._epoch_complete = lambda d: bool(epoch_complete)
        move._epoch_missing_for_ranks = (
            lambda d: None if epoch_complete else "DONE.json")
    return move


def make_epoch_dir(tmpdir, k=3, ctr_npz=True, done=True, peaks_npz=False,
                   n_peaks=None):
    """An epoch dir under ``tmpdir``; returns its ROOT.

    Defaults to the shape the stubbed tests want (``DONE.json`` + the center
    table). ``done`` / ``peaks_npz`` / ``n_peaks`` build the real
    combinations ``_epoch_missing_for_ranks`` has to separate.
    """
    from lisatools.sampling.fstat_gridfit import (
        CENTER_TABLE_BASENAME,
        GRID_BASENAME,
    )

    d = os.path.join(tmpdir, f"epoch_{int(k):04d}")
    os.makedirs(d, exist_ok=True)
    if done:
        body = '{"clock": 0}' if n_peaks is None else (
            '{"clock": 0, "n_peaks": %d}' % int(n_peaks))
        with open(os.path.join(d, "DONE.json"), "w") as f:
            f.write(body)
    if peaks_npz:
        np.savez(os.path.join(
            d, GRID_BASENAME.replace(".npz", "_peaks_stacked.npz")),
            logp_grids=np.zeros(1))
    if ctr_npz:
        np.savez(os.path.join(d, CENTER_TABLE_BASENAME), f0_mHz=np.zeros(1))
    return tmpdir


def make_slice(nwalkers=B):
    """What ``slice_state(state, w0, w1, sub_states=['gb'])`` hands a rank."""
    rng = np.random.default_rng(11)
    coords = {"gb": rng.standard_normal((NTEMPS, nwalkers, NLEAVES, NDIM))}
    inds = {"gb": np.ones((NTEMPS, nwalkers, NLEAVES), dtype=bool)}
    part = GFState(
        coords,
        inds=inds,
        log_like=rng.standard_normal((NTEMPS, nwalkers)),
        log_prior=rng.standard_normal((NTEMPS, nwalkers)),
        sub_state_bases={"gb": GBState},
    )
    part.sub_states["gb"].pull_from_main(part, "gb")
    return part


def neutral_payload(**over):
    payload = {
        "ntemps": NTEMPS,
        "nwalkers": B,
        "neutral": True,
        "state": make_slice(),
        "clock_vals": {
            "time": 11,
            "num_proposals": 5,
            "reseed_firing": True,
            "temper_vertical": True,
            "branch_propose_count": 9,
        },
        "tables": {"cap_leaf_cap": None, "band_leaf_cap": None, "rj_band_shutoff": None},
        "rank_seed": 4242,
        "directive": {"epoch": None, "ctr_table": False},
        "band_temps": np.ones((NUM_BANDS, NTEMPS)),
        "keep_all_inds": True,
        "scan_schedule": None,
        "engine_ntemps": NTEMPS,
    }
    payload.update(over)
    return payload


class RankBlockTest(unittest.TestCase):
    def setUp(self):
        self.move = make_move()
        self.model = _FakeModel()
        self._census = dict(gbs.GBSpecialBase._branch_propose_counts)
        self.addCleanup(self._restore_census)
        patcher = mock.patch.object(gbs, "pin_main_device", lambda xp, gpus: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _restore_census(self):
        gbs.GBSpecialBase._branch_propose_counts.clear()
        gbs.GBSpecialBase._branch_propose_counts.update(self._census)

    def test_enter_sets_the_block_and_exit_restores_it(self):
        payload = neutral_payload()
        saved = self.move._enter_rank_block(payload, {"seq": 3, "call_index": 1}, self.model)
        self.assertEqual(self.move.nwalkers, B)
        self.assertEqual(self.move.ntemps, NTEMPS)
        self.assertEqual(self.move.time, 11)
        self.assertEqual(self.move.num_proposals, 5)
        self.assertTrue(self.move._reseed_firing)
        self.assertTrue(self.move.temper_vertical)
        self.assertEqual(self.move._rank_rng_seed, 4242)
        self.assertIsNone(self.move._temper_rng)
        self.assertEqual(gbs.GBSpecialBase._branch_propose_counts["gb"], 9)

        self.move._exit_rank_block(saved)
        self.assertEqual(self.move.nwalkers, 4)
        self.assertEqual(self.move.ntemps, 2)
        self.assertEqual(self.move.time, 3)
        self.assertEqual(self.move.num_proposals, 7)
        self.assertFalse(self.move._reseed_firing)
        self.assertFalse(self.move.temper_vertical)
        self.assertIsNone(self.move._rank_rng_seed)

    def test_enter_takes_the_block_width_from_the_slice(self):
        payload = neutral_payload()
        payload.pop("nwalkers")
        saved = self.move._enter_rank_block(payload, {"seq": 1, "call_index": 1}, self.model)
        self.assertEqual(self.move.nwalkers, B)
        self.move._exit_rank_block(saved)

    def test_reseed_firing_defaults_to_false_not_to_the_current_value(self):
        # the head ships it every propose; a partial clock_vals must read as
        # "not firing", never as "keep whatever this move last held"
        self.move._reseed_firing = True
        payload = neutral_payload(clock_vals={"time": 4, "num_proposals": 2})
        saved = self.move._enter_rank_block(payload, {"seq": 1, "call_index": 1}, self.model)
        self.assertFalse(self.move._reseed_firing)
        self.move._exit_rank_block(saved)
        self.assertTrue(self.move._reseed_firing)

    def test_exit_restores_the_shipped_tables(self):
        cap = np.arange(NUM_BANDS)
        shut = np.zeros(NUM_BANDS, dtype=bool)
        payload = neutral_payload(
            tables={"cap_leaf_cap": cap, "band_leaf_cap": cap, "rj_band_shutoff": shut}
        )
        saved = self.move._enter_rank_block(payload, {"seq": 1, "call_index": 1}, self.model)
        self.assertIs(self.move._cap_leaf_cap, cap)
        self.assertIs(self.move._rj_band_shutoff, shut)
        self.move._exit_rank_block(saved)
        self.assertIsNone(self.move._cap_leaf_cap)
        self.assertIsNone(self.move._band_leaf_cap)
        self.assertIsNone(self.move._rj_band_shutoff)


class TemperRngSeedTest(unittest.TestCase):
    """The seeded vertical-swap Generator survives ONE propose's commands.

    The head ships one ``rank_seed`` per propose and the rank serves three
    commands with it; dropping the Generator on every command replayed an
    identical stream three times over.
    """

    def setUp(self):
        self.move = make_move()
        self.model = _FakeModel()
        patcher = mock.patch.object(gbs, "pin_main_device", lambda xp, gpus: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _enter(self, seed):
        return self.move._enter_rank_block(
            neutral_payload(rank_seed=seed), {"seq": 1, "call_index": 1}, self.model
        )

    def test_two_consecutive_enters_with_the_same_seed_keep_the_generator(self):
        self._enter(4242)
        self.assertIsNone(self.move._temper_rng)
        rng = self.move._temper_rng = np.random.default_rng(4242)
        self._enter(4242)
        self.assertIs(self.move._temper_rng, rng)

    def test_the_generator_survives_the_exit_restore_between_commands(self):
        # the REAL command sequence: every command is enter / body / exit,
        # and ``_rank_rng_seed`` IS restored on exit -- so the comparison
        # cannot be against it
        saved = self._enter(4242)
        rng = self.move._temper_rng = np.random.default_rng(4242)
        self.move._exit_rank_block(saved)
        self.assertIsNone(self.move._rank_rng_seed)  # restore contract intact
        saved = self._enter(4242)                    # gb_run_tempering
        self.assertIs(self.move._temper_rng, rng)
        self.move._exit_rank_block(saved)
        saved = self._enter(4242)                    # gb_finish
        self.assertIs(self.move._temper_rng, rng)
        self.move._exit_rank_block(saved)

    def test_a_different_seed_drops_the_generator(self):
        saved = self._enter(4242)
        self.move._temper_rng = np.random.default_rng(4242)
        self.move._exit_rank_block(saved)
        self._enter(4243)                            # the NEXT propose
        self.assertIsNone(self.move._temper_rng)

    def test_a_none_seed_keeps_whatever_generator_there_is(self):
        rng = self.move._temper_rng = np.random.default_rng()
        self._enter(None)
        self.assertIs(self.move._temper_rng, rng)

    def test_one_compute_rank_stamps_none_and_falls_to_the_legacy_branch(self):
        # the head ships ``rank_seed=None`` at ONE compute rank (fix round 5)
        # so the rank body takes the SAME derivation ``_propose_legacy``
        # takes: ``gf_temper_seed_base``, not a derived per-rank seed
        self.move.gf_temper_seed_base = 99
        saved = self._enter(None)
        self.assertIsNone(self.move._rank_rng_seed)
        rank_side = gbs.GBSpecialBase._make_temper_rng(self.move)
        n_prop = int(self.move.num_proposals)   # the head's, not the move's
        self.move._exit_rank_block(saved)
        self.assertIsNone(self.move._rank_rng_seed)
        # bit-identical to what ``_propose_legacy`` builds at the same propose
        legacy_side = np.random.default_rng(np.random.SeedSequence([99, n_prop]))
        np.testing.assert_array_equal(rank_side.random(5), legacy_side.random(5))


class MakeTemperRngTest(unittest.TestCase):
    """``_make_temper_rng``: ONE derivation shared by both propose bodies.

    Rank seed wins; else the recipe-stamped ``gf_temper_seed_base`` keyed by
    ``num_proposals``; else entropy. Before fix round 5 the orchestrator
    always took branch 1 at one rank while ``_propose_legacy`` always took
    branch 3, so the two bodies drew different vertical-swap streams.
    """

    @staticmethod
    def _move(base=None, rank_seed=None, num_proposals=0):
        move = make_move()
        move.gf_temper_seed_base = base
        move._rank_rng_seed = rank_seed
        move.num_proposals = num_proposals
        return move

    def test_the_same_base_and_propose_count_give_the_same_stream(self):
        a = gbs.GBSpecialBase._make_temper_rng(self._move(base=7, num_proposals=4))
        b = gbs.GBSpecialBase._make_temper_rng(self._move(base=7, num_proposals=4))
        np.testing.assert_array_equal(a.random(5), b.random(5))

    def test_a_different_propose_count_gives_a_different_stream(self):
        a = gbs.GBSpecialBase._make_temper_rng(self._move(base=7, num_proposals=4))
        b = gbs.GBSpecialBase._make_temper_rng(self._move(base=7, num_proposals=5))
        self.assertNotEqual(a.random(), b.random())

    def test_a_different_base_gives_a_different_stream(self):
        a = gbs.GBSpecialBase._make_temper_rng(self._move(base=7, num_proposals=4))
        b = gbs.GBSpecialBase._make_temper_rng(self._move(base=8, num_proposals=4))
        self.assertNotEqual(a.random(), b.random())

    def test_both_seeds_none_is_entropy(self):
        a = gbs.GBSpecialBase._make_temper_rng(self._move())
        b = gbs.GBSpecialBase._make_temper_rng(self._move())
        self.assertNotEqual(a.random(), b.random())

    def test_the_rank_seed_wins_over_the_base(self):
        a = gbs.GBSpecialBase._make_temper_rng(
            self._move(base=7, rank_seed=123, num_proposals=4))
        # same rank seed, DIFFERENT base and propose count: still one stream
        b = gbs.GBSpecialBase._make_temper_rng(
            self._move(base=8, rank_seed=123, num_proposals=9))
        np.testing.assert_array_equal(a.random(5), b.random(5))
        c = gbs.GBSpecialBase._make_temper_rng(self._move(rank_seed=123))
        self.assertEqual(
            np.random.default_rng(123).random(), c.random())

    def test_the_class_default_is_none_so_an_unstamped_move_is_entropy(self):
        """The class default is a DELIBERATE fail-open, not an oversight.

        A move nobody stamped (a hand-built move, a test skeleton) must keep
        today's behaviour rather than silently share one fixed stream.
        Production is covered -- every GB/VGB move construction in ``src/``
        goes through ``build_gb_moves`` / ``build_vgb_moves``, which stamp
        this on the moves they return -- so a move built OUTSIDE those two
        builders (a hand-rolled script, a legacy settings-file recipe, a
        test skeleton) is the only one that keeps this ``None`` default and
        silently reverts to OS entropy rather than raising. Accepted ruling,
        review N-4.
        """
        self.assertIsNone(gbs.GBSpecialBase.gf_temper_seed_base)


class _FakeLayout:
    """Just what ``install_walker_fanout``'s block check reads."""

    def __init__(self, block, nwalkers=8, n_compute=2):
        self._block = block
        self.nwalkers = nwalkers
        self.n_compute = n_compute

    def block_of(self, rank):
        return self._block


class _FakeFanoutWithLayout(_FakeFanout):
    def __init__(self, layout, single=False, is_head=False):
        super().__init__(single, is_head)
        self.layout = layout


def make_vgb_move(kind):
    move = make_move(gbs.VGBSpecialStretchMove)
    move._inmodel_kind = lambda: kind
    return move


class VGBWalkerBlockCheckTest(unittest.TestCase):
    """The VGB red-blue stretch needs an EVEN block >= 2 -- refuse at install.

    Without this the only guard is the ``assert`` inside
    ``VGBSpecialStretchMove.get_proposal``, which fires at the FIRST propose
    (after the whole build) instead of at launch.
    """

    def _install(self, kind, block):
        move = make_vgb_move(kind)
        fanout = _FakeFanoutWithLayout(_FakeLayout(block))
        move.install_walker_fanout(_Curr(fanout, 1))
        return move

    def test_an_odd_block_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._install("stretch", (0, 3))
        msg = str(ctx.exception)
        self.assertIn("NWALKERS", msg)
        self.assertIn("3 walker(s)", msg)

    def test_a_one_walker_block_is_refused(self):
        with self.assertRaisesRegex(ValueError, "EVEN and >= 2"):
            self._install("stretch", (2, 3))

    def test_an_even_block_installs(self):
        move = self._install("stretch", (0, 4))
        self.assertTrue(move.fanout_active)

    def test_the_observable_and_eigen_kinds_do_not_need_an_even_block(self):
        for kind in ("observable", "eigen"):
            move = self._install(kind, (0, 3))
            self.assertTrue(move.fanout_active)

    def test_a_gb_move_never_needs_an_even_block(self):
        move = make_move()
        move.install_walker_fanout(
            _Curr(_FakeFanoutWithLayout(_FakeLayout((0, 3))), 1))
        self.assertTrue(move.fanout_active)

    def test_single_rank_skips_the_check_entirely(self):
        move = make_vgb_move("stretch")
        fanout = _FakeFanoutWithLayout(_FakeLayout((0, 3)), single=True,
                                       is_head=True)
        move.install_walker_fanout(_Curr(fanout, 0))
        self.assertFalse(move.fanout_active)


class InstallWalkerFanoutTest(unittest.TestCase):
    def test_single_rank_is_inactive_and_keeps_the_sidecar(self):
        move = make_move()
        move.install_walker_fanout(_Curr(_FakeFanout(single=True, is_head=True), 0))
        self.assertFalse(move.fanout_active)
        self.assertEqual(move.eigen_store_path, "x.h5")
        self.assertEqual(move.gf_rank, 0)

    def test_no_fanout_at_all_is_inactive(self):
        move = make_move()
        move.install_walker_fanout(object())
        self.assertIsNone(move.fanout)
        self.assertFalse(move.fanout_active)

    def test_worker_rank_clears_the_eigen_sidecar(self):
        move = make_move()
        move.install_walker_fanout(_Curr(_FakeFanout(single=False, is_head=False), 2))
        self.assertTrue(move.fanout_active)
        self.assertIsNone(move.eigen_store_path)
        self.assertEqual(move.gf_rank, 2)

    def test_head_keeps_the_eigen_sidecar(self):
        move = make_move()
        move.install_walker_fanout(_Curr(_FakeFanout(single=False, is_head=True), 0))
        self.assertTrue(move.fanout_active)
        self.assertEqual(move.eigen_store_path, "x.h5")


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.move = make_move()
        self.model = _FakeModel()

    def test_ops_constant(self):
        """The op names are a wire contract: the head issues exactly these and
        a rank refuses anything else. The first three are the per-propose
        session; the last two are the F-stat epoch fit, issued from the head's
        ``setup()`` with no session open (parallel-fit spec 2026-09-16).
        """
        self.assertEqual(
            gbs.GB_OPS,
            ("gb_run_proposal", "gb_run_tempering", "gb_finish", "gb_sync",
             "gb_fstat_ref_row", "gb_fstat_stage_b"),
        )

    def test_gb_moves_are_no_longer_fanout_unready(self):
        # run.py::_fanout_unready_moves keys off exactly this identity
        self.assertIsNot(gbs.GBSpecialBase.gf_serve, GlobalFitMove.gf_serve)

    def test_unknown_op_raises(self):
        with self.assertRaises(ValueError):
            self.move.gf_serve("nope", {}, {"seq": 1, "call_index": 1}, self.model)

    def test_tempering_without_a_session_is_stale(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.move.gf_serve(
                "gb_run_tempering", {"session": 99}, {"seq": 1, "call_index": 1}, self.model
            )
        self.assertIn("stale", str(ctx.exception))

    def test_finish_without_a_session_is_stale(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.move.gf_serve(
                "gb_finish", {"session": 99}, {"seq": 1, "call_index": 1}, self.model
            )
        self.assertIn("stale", str(ctx.exception))

    def test_setup_from_directive_is_a_no_op_on_the_base_class(self):
        # the base class has no ``_install``: the directive must not create
        # one, nor adopt an epoch, nor raise
        self.move._setup_from_directive({"epoch": None, "ctr_table": False})
        self.move._setup_from_directive({"epoch": 3, "ctr_table": True})
        self.move._setup_from_directive(None)
        self.assertFalse(hasattr(self.move, "_install"))
        self.assertFalse(hasattr(self.move, "_install_ctr_table"))
        self.assertIsNone(getattr(self.move, "_fstat_epoch", None))


class SetupFromDirectiveTest(unittest.TestCase):
    """The rank stand-in for ``setup()`` on a real F-stat grid move.

    ``_install`` / ``_install_ctr_table`` / ``_epoch_dir`` are recording
    stubs and the process-global epoch registry is patched, so nothing here
    reads a fit directory, an npz or a ``DONE.json``.
    """

    def setUp(self):
        self.move = make_grid_move()
        self.model = _FakeModel()
        patcher = mock.patch.object(gbs, "pin_main_device", lambda xp, gpus: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        reg = mock.patch.dict(gbs._FSTAT_GRID_REGISTRY, {}, clear=True)
        reg.start()
        self.addCleanup(reg.stop)

    def test_a_missing_epoch_is_installed_without_the_shutoff_sync(self):
        self.move._setup_from_directive({"epoch": 3, "ctr_table": False})
        self.assertEqual(self.move.installs, [(3, {"sync_shutoff": False})])
        self.assertEqual(self.move.ctr_installs, [])

    def test_the_ctr_table_installs_only_when_the_head_asks(self):
        tmp = tempfile.mkdtemp(prefix="gb_epoch_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        move = make_grid_move(root=make_epoch_dir(tmp, 3, ctr_npz=True))
        move._setup_from_directive({"epoch": 3, "ctr_table": True})
        self.assertEqual(move.installs, [(3, {"sync_shutoff": False})])
        # model=None -> the npz-only branch, never an F-stat sweep
        self.assertEqual(move.ctr_installs, [(3, None)])

    # ---- completeness: a rank NEVER falls back silently (review C1) ----
    def test_an_incomplete_epoch_raises_instead_of_installing(self):
        move = make_grid_move(epoch_complete=False)
        move.gf_rank = 1
        with self.assertRaises(RuntimeError) as ctx:
            move._setup_from_directive({"epoch": 3, "ctr_table": False})
        msg = str(ctx.exception)
        self.assertIn("incomplete", msg)
        self.assertIn("rank 1", msg)
        self.assertIn("epoch_0003", msg)
        # nothing installed, nothing memoised: a fallback grid for this epoch
        # would have been cached and used for the whole epoch
        self.assertEqual(move.installs, [])
        self.assertEqual(move.ctr_installs, [])

    # ---- and the check is the STRICT one (npz first, manifest last) ----
    def _real_move(self, **kw):
        tmp = tempfile.mkdtemp(prefix="gb_epoch_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        root = make_epoch_dir(tmp, 3, ctr_npz=False, **kw)
        return make_grid_move(epoch_complete=None, root=root)

    def test_the_mid_write_epoch_is_refused_by_name(self):
        # the head writes the stage-B npz FIRST and DONE.json LAST, so this
        # is exactly the state a rank must refuse -- and the one
        # ``_epoch_complete``'s OR accepts
        move = self._real_move(done=False, peaks_npz=True)
        move.gf_rank = 1
        with self.assertRaises(RuntimeError) as ctx:
            move._setup_from_directive({"epoch": 3, "ctr_table": False})
        msg = str(ctx.exception)
        self.assertIn("DONE.json", msg)
        self.assertIn("incomplete", msg)
        self.assertEqual(move.installs, [])
        # the permissive predicate would have let this through
        self.assertTrue(gbs.GBSpecialRJFStatGridMove._epoch_complete(
            move._epoch_dir(3)))

    def test_both_artifacts_present_installs(self):
        move = self._real_move(peaks_npz=True, n_peaks=17)
        move._setup_from_directive({"epoch": 3, "ctr_table": False})
        self.assertEqual(move.installs, [(3, {"sync_shutoff": False})])

    def test_a_zero_peak_epoch_needs_no_stage_b_npz(self):
        # legitimate: the head writes no *_peaks_stacked.npz when the fit
        # found nothing, records it in the manifest, and falls back to the
        # prior for births -- the rank must do the same, not die
        move = self._real_move(peaks_npz=False, n_peaks=0)
        move._setup_from_directive({"epoch": 3, "ctr_table": False})
        self.assertEqual(move.installs, [(3, {"sync_shutoff": False})])

    def test_a_manifest_claiming_peaks_still_needs_the_npz(self):
        move = self._real_move(peaks_npz=False, n_peaks=17)
        with self.assertRaises(RuntimeError) as ctx:
            move._setup_from_directive({"epoch": 3, "ctr_table": False})
        self.assertIn("_peaks_stacked.npz", str(ctx.exception))
        self.assertEqual(move.installs, [])

    def test_an_incomplete_epoch_raises_even_from_the_registry(self):
        # the process-global memo must not short-circuit the check either
        move = make_grid_move(epoch_complete=False)
        gbs._FSTAT_GRID_REGISTRY[move._epoch_dir(3)] = ("container", 3, 17)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            move._setup_from_directive({"epoch": 3, "ctr_table": False})

    def test_a_requested_ctr_table_with_no_npz_raises(self):
        tmp = tempfile.mkdtemp(prefix="gb_epoch_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        move = make_grid_move(root=make_epoch_dir(tmp, 3, ctr_npz=False))
        move.gf_rank = 2
        with self.assertRaises(RuntimeError) as ctx:
            move._setup_from_directive({"epoch": 3, "ctr_table": True})
        msg = str(ctx.exception)
        self.assertIn("incomplete", msg)
        self.assertIn("fstat_centers.npz", msg)
        self.assertEqual(move.installs, [])
        self.assertEqual(move.ctr_installs, [])

    def test_no_ctr_table_requested_does_not_need_the_npz(self):
        tmp = tempfile.mkdtemp(prefix="gb_epoch_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        move = make_grid_move(root=make_epoch_dir(tmp, 3, ctr_npz=False))
        move._setup_from_directive({"epoch": 3, "ctr_table": False})
        self.assertEqual(move.installs, [(3, {"sync_shutoff": False})])

    def test_an_epoch_already_in_the_registry_is_adopted_not_installed(self):
        gbs._FSTAT_GRID_REGISTRY[self.move._epoch_dir(3)] = ("container", 3, 17)
        self.move._setup_from_directive({"epoch": 3, "ctr_table": False})
        self.assertEqual(self.move.installs, [])
        self.assertEqual(self.move.rj_proposal_distribution, "container")
        self.assertEqual(self.move._fstat_epoch, 3)

    def test_no_epoch_installs_nothing(self):
        self.move._setup_from_directive({"epoch": None, "ctr_table": True})
        self.move._setup_from_directive(None)
        self.move._setup_from_directive({})
        self.assertEqual(self.move.installs, [])
        self.assertEqual(self.move.ctr_installs, [])

    def test_the_shipped_shutoff_valve_is_reapplied_after_the_install(self):
        # _install's own _band_shutoff_epoch_sync would revive every band the
        # head has shut off; gb_run_proposal re-applies the shipped table
        # AFTER the install, which is what this pins
        shut = np.zeros(NUM_BANDS, dtype=bool)
        self.move._install = lambda k, **kw: setattr(
            self.move, "_rj_band_shutoff", np.ones(NUM_BANDS, dtype=bool))
        seen = {}
        real_timer = self.move._gb_new_timer

        def _spy_timer(model):
            # runs immediately after the re-apply, and before _exit_rank_block
            # puts the move's own table back
            seen["shutoff"] = self.move._rj_band_shutoff
            return real_timer(model)

        self.move._gb_new_timer = _spy_timer
        payload = neutral_payload(
            directive={"epoch": 3, "ctr_table": False},
            tables={"cap_leaf_cap": None, "band_leaf_cap": None,
                    "rj_band_shutoff": shut},
        )
        self.move.gf_serve(
            "gb_run_proposal", payload, {"seq": 2, "call_index": 1}, self.model
        )
        self.move._gb_session = None
        self.assertIs(seen["shutoff"], shut)


class FlushEpochArtifactsTest(unittest.TestCase):
    """The head half of the completeness contract (review C1).

    The memo is judged COMPLETE by the same rule a rank applies to the
    epoch dir -- ``_epoch_missing_for_ranks``, which accepts a zero-peak
    ``DONE.json`` with no stage-B npz -- plus the centre table when this
    move has a live one (NEW-E4, tightened against the zero-peak case by
    round 6 / review N-1), while the sync loop still fsyncs every artifact
    that is THERE. So these fixtures write ``peaks_npz=True`` wherever the
    epoch is meant to read as complete via the npz, and ``n_peaks=0``
    wherever it is meant to read as complete WITHOUT one.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gb_flush_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_every_present_artifact_is_fsynced(self):
        move = make_grid_move(
            root=make_epoch_dir(self.tmp, 3, ctr_npz=True, peaks_npz=True))
        synced = []
        with mock.patch.object(gbs.os, "fsync", synced.append):
            move._flush_epoch_artifacts(3)
        # DONE.json + the stage-B npz + fstat_centers.npz -- the centre table
        # is fsynced because it EXISTS, whether or not this move expects it
        self.assertEqual(len(synced), 3)

    def test_the_second_flush_of_the_same_epoch_is_a_no_op(self):
        # the caller sits in the head's propose (every iteration) while an
        # epoch's bytes only change when setup() writes a new one
        move = make_grid_move(
            root=make_epoch_dir(self.tmp, 3, ctr_npz=True, peaks_npz=True))
        synced = []
        with mock.patch.object(gbs.os, "fsync", synced.append):
            with self.assertLogs(gbs.logger, level="INFO") as log:
                move._flush_epoch_artifacts(3)
                move._flush_epoch_artifacts(3)
                move._flush_epoch_artifacts(3)
        self.assertEqual(len(synced), 3)  # ONE pass over the three artifacts
        self.assertEqual(
            len([line for line in log.output if "head flushed epoch" in line]), 1)

    def test_a_new_epoch_flushes_again(self):
        move = make_grid_move(
            root=make_epoch_dir(self.tmp, 3, ctr_npz=True, peaks_npz=True))
        make_epoch_dir(self.tmp, 4, ctr_npz=True, peaks_npz=True)
        synced = []
        with mock.patch.object(gbs.os, "fsync", synced.append):
            move._flush_epoch_artifacts(3)
            move._flush_epoch_artifacts(4)
        self.assertEqual(len(synced), 6)

    def test_a_missing_epoch_dir_is_not_fatal(self):
        move = make_grid_move(root=os.path.join(self.tmp, "absent"))
        with self.assertLogs(gbs.logger, level="INFO") as log:
            move._flush_epoch_artifacts(3)  # nothing to flush, no raise
        self.assertTrue(any("no artifacts" in line for line in log.output))

    def test_a_flush_that_found_nothing_does_not_suppress_the_real_one(self):
        # the memo is recorded AFTER a pass that found the WHOLE epoch:
        # a first call racing ahead of the setup() that writes it must not
        # permanently suppress that epoch's real flush on this move
        root = os.path.join(self.tmp, "late")
        move = make_grid_move(root=root)
        synced = []
        with mock.patch.object(gbs.os, "fsync", synced.append):
            move._flush_epoch_artifacts(3)  # epoch dir not written yet
            self.assertEqual(len(synced), 0)
            make_epoch_dir(root, 3, ctr_npz=True, peaks_npz=True)
            move._flush_epoch_artifacts(3)  # now it is there
        self.assertEqual(len(synced), 3)

    def test_a_partial_flush_leaves_the_memo_unset_and_retries(self):
        # NEW-E4: a pass that found the MANIFEST but not the artifacts still
        # being written must not memoise the epoch -- those two would then
        # never be fsynced on this move instance, and the ranks would open an
        # unflushed npz with only their own completeness check as a backstop
        root = os.path.join(self.tmp, "partial")
        move = make_grid_move(root=root)
        make_epoch_dir(root, 3, ctr_npz=False, peaks_npz=False)  # DONE.json only
        synced = []
        with mock.patch.object(gbs.os, "fsync", synced.append):
            move._flush_epoch_artifacts(3)
            self.assertEqual(len(synced), 1)
            self.assertIsNone(getattr(move, "_epoch_flushed", None),
                              "a PARTIAL pass must not set the memo")
            make_epoch_dir(root, 3, ctr_npz=False, peaks_npz=True)
            move._flush_epoch_artifacts(3)          # retried, not suppressed
            self.assertEqual(len(synced), 3)        # DONE.json again + the npz
            move._flush_epoch_artifacts(3)          # NOW it is memoised
            self.assertEqual(len(synced), 3)
        self.assertEqual(move._epoch_flushed[0], 3)

    def test_a_zero_peak_epoch_sets_the_memo_without_the_stacked_npz(self):
        # review N-1: the legitimate zero-peak F-stat epoch never writes
        # fstat_grid_peaks_stacked.npz (``_epoch_missing_for_ranks`` and
        # ``_epoch_complete`` both carve this out); the memo must still be
        # set on it, or the head re-fsyncs DONE.json and re-logs
        # [FSTAT_EPOCH] on every propose of the whole run
        root = os.path.join(self.tmp, "zero_peak")
        move = make_grid_move(root=make_epoch_dir(
            root, 3, ctr_npz=False, peaks_npz=False, n_peaks=0))
        synced = []
        with mock.patch.object(gbs.os, "fsync", synced.append):
            with self.assertLogs(gbs.logger, level="INFO") as log:
                move._flush_epoch_artifacts(3)   # DONE.json only -> complete
                move._flush_epoch_artifacts(3)   # memoised -> no-op
        self.assertEqual(len(synced), 1)         # DONE.json is the only file
        self.assertEqual(
            len([line for line in log.output if "head flushed epoch" in line]), 1)
        self.assertEqual(move._epoch_flushed[0], 3)

    def test_the_centre_table_is_expected_only_when_the_move_has_one(self):
        # the same head-side answer the rank directive's ``ctr_table`` flag
        # carries: with a live table a missing fstat_centers.npz is an
        # INCOMPLETE epoch, so the memo stays unset and the next pass retries
        root = os.path.join(self.tmp, "ctr")
        move = make_grid_move(root=root)
        make_epoch_dir(root, 3, ctr_npz=False, peaks_npz=True)
        move._fstat_ctr_table = {"f0_mHz": np.zeros(1)}
        patcher = mock.patch.dict(os.environ, {"GB_FSTAT_CTR_MODE": "epoch"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertIsNotNone(move._fstat_ctr_table_active())
        synced = []
        with mock.patch.object(gbs.os, "fsync", synced.append):
            move._flush_epoch_artifacts(3)
            self.assertEqual(len(synced), 2)
            self.assertIsNone(getattr(move, "_epoch_flushed", None))
            move._fstat_ctr_table = None            # no table -> not expected
            move._flush_epoch_artifacts(3)
        self.assertEqual(move._epoch_flushed[0], 3)

    def test_no_epoch_and_no_epoch_dir_are_no_ops(self):
        move = make_grid_move(root=self.tmp)
        base = make_move()  # no ``_epoch_dir`` on the base class
        with self.assertNoLogs(gbs.logger, level="INFO"):
            move._flush_epoch_artifacts(None)
            base._flush_epoch_artifacts(3)

    def test_an_unreadable_artifact_warns_rather_than_raises(self):
        move = make_grid_move(root=make_epoch_dir(self.tmp, 3, ctr_npz=True))

        def _boom(fd):
            raise OSError("no fsync here")

        with mock.patch.object(gbs.os, "fsync", _boom):
            with self.assertLogs(gbs.logger, level="WARNING") as log:
                move._flush_epoch_artifacts(3)
        self.assertTrue(any("fsync" in line for line in log.output))


class BlockACAWidthTest(unittest.TestCase):
    """The ACA-width rule guard of the orchestrated GB propose.

    Its body runs in the gated two-rank smoke, but the RAISE branch and its
    message were untested (re-review NEW-3): an ensemble-width ACA on a rank
    silently scores the wrong walkers, so the message has to name both
    widths.
    """

    def _move(self, entries, block=(0, 2)):
        move = make_move()
        fanout = _FakeFanoutWithLayout(_FakeLayout(block), is_head=True)
        fanout.rank = 0
        move.fanout = fanout
        acs = _FakeACS()
        if entries is not None:
            acs.acs_total_entries = entries
        return move, acs

    def test_an_ensemble_width_aca_raises_naming_both_widths(self):
        move, acs = self._move(4)  # 4 = the ensemble, 2 = this block
        with self.assertRaises(RuntimeError) as ctx:
            move._check_block_aca_width(acs)
        msg = str(ctx.exception)
        self.assertIn("gb_test", msg)
        self.assertIn("4 walker rows", msg)
        self.assertIn("[0, 2) (2 walkers)", msg)

    def test_the_block_width_passes(self):
        move, acs = self._move(2)
        self.assertIsNone(move._check_block_aca_width(acs))

    def test_an_aca_that_reports_no_rows_is_not_checked(self):
        move, acs = self._move(None)
        self.assertIsNone(move._check_block_aca_width(acs))


class NeutralBlockTest(unittest.TestCase):
    """A block the HEAD marked neutral still answers all three commands."""

    def setUp(self):
        self.move = make_move()
        self.model = _FakeModel()
        self._census = dict(gbs.GBSpecialBase._branch_propose_counts)
        self.addCleanup(self._restore_census)
        patcher = mock.patch.object(gbs, "pin_main_device", lambda xp, gpus: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.clock = {"seq": 7, "call_index": 2}
        self.token = (7, 2)

    def _restore_census(self):
        gbs.GBSpecialBase._branch_propose_counts.clear()
        gbs.GBSpecialBase._branch_propose_counts.update(self._census)

    def test_the_three_commands_round_trip(self):
        rep = self.move.gf_serve(
            "gb_run_proposal", neutral_payload(), self.clock, self.model
        )
        sess = self.move._gb_session
        self.assertIsNotNone(sess)
        self.assertEqual(sess.token, self.token)
        self.assertIsNone(sess.band_sorter)
        # the head's N is restored the moment the command returns
        self.assertEqual(self.move.nwalkers, 4)

        self.assertEqual(np.shape(rep["log_like_cold"]), (B,))
        self.assertEqual(np.shape(rep["prop_counts"]), (2, NTEMPS, NUM_BANDS))
        self.assertEqual(np.shape(rep["acc_counts"]), (2, NTEMPS, NUM_BANDS))
        self.assertEqual(np.shape(rep["cold_counts"]), (2, 2, B, NUM_BANDS))
        self.assertEqual(np.shape(rep["start_diffs"]), (B,))
        self.assertEqual(list(rep["alive_per_temp"]), [0] * NTEMPS)
        self.assertEqual(rep["n_alive"], 0)
        self.assertEqual(rep["drift"], 0.0)
        self.assertIsNone(rep["rj_at_cap"])
        self.assertIn("timing", rep)
        self.assertTrue(np.all(np.asarray(rep["prop_counts"]) == 0))

        rep2 = self.move.gf_serve(
            "gb_run_tempering",
            neutral_payload(session=self.token, tmp_start=0),
            {"seq": 7, "call_index": 3},
            self.model,
        )
        self.assertEqual(np.shape(rep2["log_like_cold"]), (B,))
        self.assertEqual(np.shape(rep2["band_swaps_accepted"]), (NUM_BANDS, NTEMPS - 1))
        self.assertEqual(np.shape(rep2["band_swaps_proposed"]), (NUM_BANDS, NTEMPS - 1))
        self.assertEqual(np.shape(rep2["ll_change_sum_temp_cold"]), (B,))
        self.assertEqual(rep2["drift"], 0.0)
        self.assertIsNone(rep2["census"])
        self.assertIsNotNone(self.move._gb_session)

        rep3 = self.move.gf_serve(
            "gb_finish",
            neutral_payload(session=self.token, want_cap_stats=False),
            {"seq": 7, "call_index": 4},
            self.model,
        )
        # a neutral block ran nothing, so it ships NO branch block: the head
        # keeps the coords/inds it sliced for those walkers
        self.assertIsNone(rep3["block_coords"])
        self.assertIsNone(rep3["block_inds"])
        self.assertEqual(np.shape(rep3["d_h"]), (B, NLEAVES))
        self.assertEqual(np.shape(rep3["h_h"]), (B, NLEAVES))
        self.assertEqual(np.shape(rep3["band_counts"]), (NTEMPS, B, NUM_BANDS))
        self.assertEqual(np.shape(rep3["log_like_final"]), (B,))
        self.assertIsNone(rep3["cap_stats"])
        self.assertEqual(rep3["fstat_ctr_fallback_rows"], 0)
        for key in ("band_dof", "rj_split", "replace_census", "timing"):
            self.assertIn(key, rep3)
        # the session is gone and the head's block width is back
        self.assertIsNone(self.move._gb_session)
        self.assertEqual(self.move.nwalkers, 4)

    def test_the_neutral_path_requires_a_state_slice(self):
        payload = neutral_payload()
        payload["state"] = None
        with self.assertRaises(ValueError) as ctx:
            self.move.gf_serve("gb_run_proposal", payload, self.clock, self.model)
        self.assertIn("no state slice", str(ctx.exception))

    def test_a_neutral_block_still_ships_its_cap_rows(self):
        rows = np.arange(B * NUM_BANDS, dtype=float).reshape(B, NUM_BANDS)
        self.move._cap_stats_local = lambda model, st: {
            "band_lls": rows, "lls": rows, "dof": 7.0,
            "band_dof": np.full(NUM_BANDS, 5.0), "is_cells": False,
        }
        self.move.gf_serve(
            "gb_run_proposal", neutral_payload(), self.clock, self.model
        )
        rep = self.move.gf_serve(
            "gb_finish",
            neutral_payload(session=self.token, want_cap_stats=True),
            {"seq": 7, "call_index": 4},
            self.model,
        )
        stats = rep["cap_stats"]
        self.assertIsNotNone(stats)
        self.assertEqual(np.shape(stats["band_lls"]), (B, NUM_BANDS))
        self.assertEqual(np.shape(stats["lls"]), (B, NUM_BANDS))
        self.assertFalse(stats["is_cells"])

    def test_a_neutral_block_skips_the_cap_work_when_not_asked(self):
        self.move._cap_stats_local = lambda model, st: self.fail(
            "cap statistics must not be computed when the head did not ask")
        self.move.gf_serve(
            "gb_run_proposal", neutral_payload(), self.clock, self.model
        )
        rep = self.move.gf_serve(
            "gb_finish",
            neutral_payload(session=self.token, want_cap_stats=False),
            {"seq": 7, "call_index": 4},
            self.model,
        )
        self.assertIsNone(rep["cap_stats"])
        self.assertIsNone(rep["band_dof"])

    def test_the_session_snapshot_is_rebound_on_the_later_commands(self):
        self.move.gf_serve(
            "gb_run_proposal", neutral_payload(), self.clock, self.model
        )
        snap = object()
        self.move._gb_session.snapshot = snap
        self.move.reset_non_gb_linear_data_arr = "stale"
        self.move.gf_serve(
            "gb_run_tempering",
            neutral_payload(session=self.token, tmp_start=0),
            {"seq": 7, "call_index": 3},
            self.model,
        )
        self.assertIs(self.move.reset_non_gb_linear_data_arr, snap)
        self.move.reset_non_gb_linear_data_arr = "stale"
        self.move.gf_serve(
            "gb_finish",
            neutral_payload(session=self.token, want_cap_stats=False),
            {"seq": 7, "call_index": 4},
            self.model,
        )
        self.assertIs(self.move.reset_non_gb_linear_data_arr, snap)

    def test_the_finish_reply_carries_the_stashed_censuses(self):
        self.move.gf_serve(
            "gb_run_proposal", neutral_payload(), self.clock, self.model
        )
        # opening the session clears any previous propose's stashes
        self.assertIsNone(self.move._rj_split_last)
        self.assertIsNone(self.move._replace_split_last)
        self.move._rj_split_last = {"births": 12, "birth_acc": 3}
        self.move._replace_split_last = {"proposals": 5, "acc": 1}
        rep = self.move.gf_serve(
            "gb_finish",
            neutral_payload(session=self.token, want_cap_stats=False),
            {"seq": 7, "call_index": 4},
            self.model,
        )
        self.assertEqual(rep["rj_split"], {"births": 12, "birth_acc": 3})
        self.assertEqual(rep["replace_census"], {"proposals": 5, "acc": 1})

    def test_a_wrong_session_token_is_stale(self):
        self.move.gf_serve("gb_run_proposal", neutral_payload(), self.clock, self.model)
        with self.assertRaises(RuntimeError) as ctx:
            self.move.gf_serve(
                "gb_run_tempering",
                neutral_payload(session=(1, 1)),
                {"seq": 8, "call_index": 1},
                self.model,
            )
        self.assertIn("stale", str(ctx.exception))
        self.move._gb_session = None


class KeepAllIndsDefaultTest(unittest.TestCase):
    """The safety-net default is the LEGACY expression, not ``True``."""

    def setUp(self):
        self.model = _FakeModel()
        patcher = mock.patch.object(gbs, "pin_main_device", lambda xp, gpus: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _session_keep_all(self, **attrs):
        move = make_move()
        for k, v in attrs.items():
            setattr(move, k, v)
        payload = neutral_payload()
        payload.pop("keep_all_inds")        # the head omitted the key
        move.gf_serve("gb_run_proposal", payload, {"seq": 1, "call_index": 1}, self.model)
        keep = move._gb_session.keep_all_inds
        move._gb_session = None
        return keep

    def test_rj_replace_defaults_to_alive_only(self):
        self.assertFalse(self._session_keep_all(rj_replace=True))

    def test_prior_removal_defaults_to_alive_only(self):
        self.assertFalse(self._session_keep_all(use_prior_removal=True))

    def test_a_plain_move_defaults_to_keeping_every_slot(self):
        self.assertTrue(self._session_keep_all())

    def test_an_explicit_key_still_wins(self):
        move = make_move()
        move.rj_replace = True
        move.gf_serve(
            "gb_run_proposal",
            neutral_payload(keep_all_inds=True),
            {"seq": 1, "call_index": 1},
            self.model,
        )
        self.assertTrue(move._gb_session.keep_all_inds)
        move._gb_session = None


class CensusStashTest(unittest.TestCase):
    """``_replace_split`` is cleared by its own report site before gb_finish."""

    def _move(self, fanout_single):
        move = make_move()
        move.fanout = _FakeFanout(single=fanout_single, is_head=True)
        move.gf_rank = 0
        move._replace_split = dict(
            proposals=10, proposals_cold=4, acc=3, acc_cold=1, snr=2,
            nonfinite=0, dll_cold_sum=1.5, dll_cold_max=0.9)
        return move

    def test_the_report_stashes_a_copy_under_fanout(self):
        move = self._move(fanout_single=False)
        live = move._replace_split
        move._replace_census_report()
        self.assertIsNone(move._replace_split)      # cleared, as always
        self.assertEqual(move._replace_split_last["proposals"], 10)
        self.assertIsNot(move._replace_split_last, live)   # a COPY

    def test_single_process_output_is_untouched(self):
        move = self._move(fanout_single=True)
        move._replace_census_report()
        self.assertIsNone(move._replace_split)
        self.assertFalse(hasattr(move, "_replace_split_last"))

    def test_no_census_stashes_nothing(self):
        move = self._move(fanout_single=False)
        move._replace_split = None
        move._replace_census_report()
        self.assertFalse(hasattr(move, "_replace_split_last"))


class LegacySetupGuardTest(unittest.TestCase):
    """``GBSpecialRJSerialSearchMCMC.setup`` refuses several compute ranks.

    This is the FD dev-search path's scalar-walker ``ParaEnsembleSampler``
    (Task 5, site map section 7) -- not ported to the walker-block fan-out,
    so it must raise rather than silently run wrong under
    ``fanout_active``. A skeleton built with ``__new__`` is enough: the
    guard is the first statement in ``setup``, before anything else on the
    move is touched.
    """

    def _move(self):
        return gbs.GBSpecialRJSerialSearchMCMC.__new__(gbs.GBSpecialRJSerialSearchMCMC)

    def test_fanout_active_setup_raises(self):
        move = self._move()
        move.fanout = _FakeFanout(single=False, is_head=True)
        with self.assertRaises(NotImplementedError) as ctx:
            move.setup(None, None)
        msg = str(ctx.exception)
        self.assertIn("GBSpecialRJSerialSearchMCMC", msg)
        self.assertIn("several compute ranks", msg)

    def test_no_fanout_proceeds_past_the_guard(self):
        # fanout=None -> fanout_active is False (single-process default) ->
        # the guard does not fire and execution reaches the method body.
        # Prove it by making the first call the body makes raise a sentinel
        # distinct from NotImplementedError, and asserting THAT sentinel
        # (not the guard) is what comes out.
        move = self._move()
        move.fanout = None
        move.search_kwargs = {
            "nwalkers": 4,
            "ntemps": 2,
            "shutoff_band_iteration": 1,
            "shutoff_frequency_threshold": 0.0,
            "burn_1": 1,
            "nsteps_1": 1,
            "snr_threshold": 8.0,
            "burn_2": 1,
            "nsteps_2": 1,
        }
        sentinel = RuntimeError("reached the guarded body")
        model = mock.Mock()
        model.analysis_container_arr.likelihood.side_effect = sentinel

        with self.assertRaises(RuntimeError) as ctx:
            move.setup(model, None)
        self.assertIs(ctx.exception, sentinel)


class GbHostTest(unittest.TestCase):
    def test_namedtuples_survive_the_host_coercion(self):
        import collections

        Pair = collections.namedtuple("Pair", "lo hi")
        out = gbs._gb_host(Pair(np.arange(3), 2.0))
        self.assertIsInstance(out, Pair)
        np.testing.assert_array_equal(out.lo, np.arange(3))
        self.assertEqual(out.hi, 2.0)

    def test_plain_sequences_keep_their_type(self):
        self.assertIsInstance(gbs._gb_host([1, "a", None]), list)
        self.assertIsInstance(gbs._gb_host((1, "a", None)), tuple)
        self.assertEqual(gbs._gb_host({"a": [1, 2]})["a"], [1, 2])


if __name__ == "__main__":
    unittest.main()
