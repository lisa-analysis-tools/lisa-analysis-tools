"""Zero-peak F-stat epochs: bookkeeping only (user ruling 2026-10-10).

"When the F-stat literally has zero peaks, skip the F-stat RJ proposal and
its in-model partner." 9mo job 753, gb_search_1: epochs 10-14 found 0 peaks,
``rj_fstat_search`` fell back to the PRIOR for births and spent ~10 min a
cycle proposing births that accepted nothing, and ``in_model_fstat`` then
polished nothing new.

Pinned here, on the move side (the leg runner's half -- skipping the
in-model partner -- is in ``tests/test_search_legs.py``):

* ADOPTION. An epoch with LITERALLY zero peaks (the fit's count, else the
  epoch's DONE.json) leaves ``rj_proposal_distribution`` None -- never the
  prior -- and arms ``_fstat_zero_peak_epoch`` on the F-stat BIRTH moves
  (rj_fstat_search and its PE twin rj_fstat_pe, which share the code path).
  rj_replace keeps the historical prior fallback, and every adopter sharing
  a fit dir (head install, head cross-move reuse, rank reuse) applies its OWN
  rule. An epoch with peaks clears the skip with no residue.
* THE REFIT CLOCK keeps running on a zero-peak epoch and the cadence opens
  the next epoch -- a refit that finds peaks clears the skip in the same
  ``setup()``.
* THE PROPOSE does its bookkeeping (``setup()``, the band valve tick, the
  designated updater's caps / search stage / per-walker valve JUDGE) and
  nothing else: no births, deaths, in-model repeats, tempering or fan-out
  command, and arms ``gf_skip_partner_once`` for the leg runner.

Skeleton-level: the orchestrator harness of ``test_gb_orchestrator_merge``
for the propose (both bodies), a bare ``GBSpecialRJFStatGridMove`` for the
adoption; no GB kernel, no sampler, CPU only.
"""

import copy
import inspect
import json
import os
import pickle
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves import gbspecialstretch as gbs
from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialBase,
    GBSpecialRJFStatGridMove,
)
import tests.test_gb_orchestrator_merge as OM
from tests.test_gf_substate_roundtrip import NTEMPS, NUM_BANDS, NWALKERS

LOGGER = "lisatools.globalfit.moves.gbspecialstretch"
BUILD = "lisatools.sampling.fstat_gridfit.build_gb_birth_distribution"
ZERO_EPOCH = 12


class _Container:
    """A picklable stand-in for an epoch's birth container."""

    def __init__(self, tag):
        self.tag = tag

    def __eq__(self, other):
        return isinstance(other, _Container) and other.tag == self.tag


def _grid_move(root, *, name="rj_fstat_search", rj_replace=False):
    """A bare F-stat grid move: only what the adoption / decision path reads."""
    m = GBSpecialRJFStatGridMove.__new__(GBSpecialRJFStatGridMove)
    m.name = name
    m.branch_name = "gb"
    m._backend_name = "lisatools_cpu"
    m.fstat_fit_dir = root
    m.fstat_search_residual = True
    m.fstat_fit_kwargs = {}
    m.fstat_refit_every = 3
    m.df = 1.0 / 7.9e6
    m.band_edges = np.linspace(1e-3, 2e-3, 6)
    m.is_rj_prop = True
    m.rj_replace = rj_replace
    m.rj_removal_only = False
    m.priors = {"gb": "PRIOR"}
    m.gpu_priors = {"gb": "GPU_PRIOR"}
    m.rj_proposal_distribution = None
    m._fstat_epoch = None
    m._fstat_last_fit_hit = -1
    m._fstat_zero_peak_epoch = None
    m.gf_iteration = None
    m.gf_rank = None
    m.num_proposals = 0
    # heavy neighbours the adoption path calls but does not depend on
    m._band_shutoff_epoch_sync = lambda: 0
    m._install_ctr_table = lambda k, model=None, branches=None: None
    return m


def _manifest(move, k, n_peaks, clock=0):
    d = move._epoch_dir(k)
    os.makedirs(d, exist_ok=True)
    man = {"epoch": k, "clock": clock}
    if n_peaks is not None:
        man["n_peaks"] = n_peaks
    with open(os.path.join(d, "DONE.json"), "w") as f:
        json.dump(man, f)


class _IsolatedFstatState(unittest.TestCase):
    """Fresh fit root, empty registry, and the class-level refit clock saved
    and restored around every test."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        p = mock.patch.dict(gbs._FSTAT_GRID_REGISTRY, {}, clear=True)
        p.start()
        self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for k in ("GB_FSTAT_FIT_PER_MOVE", "GB_FSTAT_FORCE_REFIT"):
            os.environ.pop(k, None)
        saved = [(d, dict(d)) for d in (
            GBSpecialBase._branch_propose_counts,
            GBSpecialBase._branch_iteration_counts,
            GBSpecialBase._branch_iteration_seen,
            GBSpecialRJFStatGridMove._fstat_clock_written)]
        seeded = set(GBSpecialRJFStatGridMove._fstat_clock_seeded)
        for d, _ in saved:
            d.clear()
        GBSpecialRJFStatGridMove._fstat_clock_seeded.clear()

        def _restore():
            for d, v in saved:
                d.clear()
                d.update(v)
            GBSpecialRJFStatGridMove._fstat_clock_seeded.clear()
            GBSpecialRJFStatGridMove._fstat_clock_seeded.update(seeded)
        self.addCleanup(_restore)


# ======================================================================
# 1. adoption: zero peaks arm the skip, never the prior
# ======================================================================

class ZeroPeakAdoptionTest(_IsolatedFstatState):

    def test_a_zero_peak_epoch_arms_the_skip_and_not_the_prior(self):
        m = _grid_move(self.root)
        with mock.patch(BUILD, return_value=None), \
                self.assertLogs(LOGGER, "WARNING") as logs:
            m._install(10, stacked=None, n_peaks=0)
        self.assertIsNone(m.rj_proposal_distribution)
        self.assertEqual(m._fstat_zero_peak_epoch, 10)
        self.assertEqual(m._fstat_epoch, 10)
        # the registry carries "no distribution", not this move's answer
        self.assertEqual(gbs._FSTAT_GRID_REGISTRY[m._epoch_dir(10)],
                         (None, 10, 0))
        self.assertIn("NOT falling back to the prior", "\n".join(logs.output))

    def test_an_epoch_with_peaks_restores_normal_behaviour_with_no_residue(self):
        m = _grid_move(self.root)
        with mock.patch(BUILD, return_value=None):
            m._install(10, stacked=None, n_peaks=0)
        self.assertEqual(m._fstat_zero_peak_epoch, 10)
        with mock.patch(BUILD, return_value=_Container("e11")):
            m._install(11, stacked="STACKED", n_peaks=4)
        self.assertEqual(m.rj_proposal_distribution, {"gb": _Container("e11")})
        self.assertIsNone(m._fstat_zero_peak_epoch)
        self.assertEqual(gbs._FSTAT_GRID_REGISTRY[m._epoch_dir(11)],
                         ({"gb": _Container("e11")}, 11, 4))
        # deepcopy / pickle safety: nothing new on the move but plain values
        del m._band_shutoff_epoch_sync, m._install_ctr_table
        m.gf_skip_partner_once = True
        back = pickle.loads(pickle.dumps(copy.deepcopy(m)))
        self.assertIsNone(back._fstat_zero_peak_epoch)
        self.assertTrue(back.gf_skip_partner_once)
        self.assertEqual(back.rj_proposal_distribution, {"gb": _Container("e11")})

    def test_the_load_path_takes_the_peak_count_from_the_manifest(self):
        m = _grid_move(self.root)
        _manifest(m, 3, n_peaks=0)
        with mock.patch(BUILD, return_value=None):
            m._install(3)                       # a resume: no count passed
        self.assertIsNone(m.rj_proposal_distribution)
        self.assertEqual(m._fstat_zero_peak_epoch, 3)
        # NOT literally zero peaks -> the historical prior fallback, unchanged
        for k, n in ((4, 7), (5, None)):
            other = _grid_move(self.root)
            if n is not None:
                _manifest(other, k, n_peaks=n)
            with mock.patch(BUILD, return_value=None):
                other._install(k)
            self.assertEqual(other.rj_proposal_distribution, {"gb": "PRIOR"})
            self.assertIsNone(other._fstat_zero_peak_epoch)

    def test_moves_sharing_the_fit_dir_each_apply_their_own_rule(self):
        """rj_replace keeps the prior; the birth move reusing the SAME
        registry entry skips -- in either install order."""
        rep = _grid_move(self.root, name="rj_replace", rj_replace=True)
        birth = _grid_move(self.root)
        _manifest(rep, 6, n_peaks=0)
        with mock.patch(BUILD, return_value=None):
            rep._install(6)
        self.assertEqual(rep.rj_proposal_distribution, {"gb": "PRIOR"})
        self.assertIsNone(rep._fstat_zero_peak_epoch)
        birth._setup_epoch(None, None, "load", 6)       # cross-move reuse
        self.assertIsNone(birth.rj_proposal_distribution)
        self.assertEqual(birth._fstat_zero_peak_epoch, 6)
        # ... and the other way round
        birth2 = _grid_move(self.root)
        rep2 = _grid_move(self.root, name="rj_replace", rj_replace=True)
        with mock.patch(BUILD, return_value=None):
            birth2._install(7, stacked=None, n_peaks=0)
        rep2._setup_epoch(None, None, "load", 7)
        self.assertEqual(birth2._fstat_zero_peak_epoch, 7)
        self.assertEqual(rep2.rj_proposal_distribution, {"gb": "PRIOR"})
        self.assertIsNone(rep2._fstat_zero_peak_epoch)

    def test_a_rank_reusing_the_registry_applies_the_same_rule(self):
        rep = _grid_move(self.root, name="rj_replace", rj_replace=True)
        _manifest(rep, 8, n_peaks=0)
        with mock.patch(BUILD, return_value=None):
            rep._install(8)
        rank_birth = _grid_move(self.root)
        rank_birth.gf_rank = 1
        rank_birth._setup_from_directive({"epoch": 8, "ctr_table": False})
        self.assertIsNone(rank_birth.rj_proposal_distribution)
        self.assertEqual(rank_birth._fstat_zero_peak_epoch, 8)


# ======================================================================
# 2. the refit clock keeps running; the cadence opens the next epoch
# ======================================================================

class ZeroPeakRefitTest(_IsolatedFstatState):

    def _zero_installed(self, every):
        m = _grid_move(self.root)
        m.fstat_refit_every = every
        _manifest(m, 2, n_peaks=0, clock=0)
        with mock.patch(BUILD, return_value=None):
            m._install(2)
        self.assertEqual(m._fstat_zero_peak_epoch, 2)
        self.assertEqual(m._fstat_last_fit_hit, 0)
        return m

    def test_the_clock_advances_and_the_cadence_refits(self):
        m = self._zero_installed(every=3)
        for it, want in ((0, ("skip", 2)), (1, ("skip", 2)), (2, ("fit", 3))):
            m.gf_iteration = it
            self.assertEqual(m._fstat_fit_decision(), want, f"iteration {it}")
        # one tick per iteration, exactly as under an installed grid
        self.assertEqual(GBSpecialBase._branch_iteration_counts["gb"], 3)

    def test_a_due_refit_that_finds_peaks_clears_the_skip_in_the_same_setup(self):
        m = self._zero_installed(every=1)
        fits = []

        def _fit(model, k, branches=None):
            fits.append(k)
            return "STACKED", 4

        m._run_fstat_fit = _fit
        m._free_inmodel_batch_pools = lambda model, why: None
        m._fstat_release_ref_row = lambda: None
        m.gf_iteration = 0
        with mock.patch(BUILD, return_value=_Container("e3")):
            m.setup(None, None)
        self.assertEqual(fits, [3])
        self.assertIsNone(m._fstat_zero_peak_epoch)
        self.assertEqual(m.rj_proposal_distribution, {"gb": _Container("e3")})
        self.assertEqual(m._fstat_epoch, 3)


# ======================================================================
# 3. the propose: bookkeeping only, in both bodies
# ======================================================================

class ZeroPeakProposeTest(unittest.TestCase):

    def setUp(self):
        if gbs.cp is not np:  # pragma: no cover - GPU box
            self.skipTest("the orchestrator harness runs on the host array module")
        self.state = OM.make_state(np.random.default_rng(11))
        census = dict(GBSpecialBase._branch_propose_counts)

        def _restore():
            GBSpecialBase._branch_propose_counts.clear()
            GBSpecialBase._branch_propose_counts.update(census)
        self.addCleanup(_restore)
        for p in (mock.patch.object(gbs, "pin_main_device", lambda xp, gpus: None),
                  # the proposal body must never be reached
                  mock.patch.object(gbs, "BandSorter", self._no_sorter)):
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def _no_sorter(*a, **k):
        raise AssertionError("the proposal body built a BandSorter")

    def _run(self, *, size, zero_peaks=True, stale_flag=False, stage_kind=None):
        real = OM.make_move

        def wrapped(*a, **k):
            move = real(*a, **k)
            move.setup_calls, move.judge_calls = [], []
            move.stage_calls, move.body_calls = [], []

            def setup(model, branches, _m=move):
                # what GBSpecialRJFStatGridMove.setup leaves behind after
                # adopting a zero-peak epoch
                _m.setup_calls.append(int(_m.num_proposals))
                if zero_peaks:
                    _m.rj_proposal_distribution = None
                    _m._fstat_zero_peak_epoch = ZERO_EPOCH

            move.setup = setup
            move._update_search_band_shutoff = (
                lambda model, st, counts, _m=move:
                _m.judge_calls.append((st, np.array(counts))))
            move._update_search_stages = (
                lambda st, counts, _m=move: _m.stage_calls.append(np.array(counts)))
            move.run_proposal = (
                lambda *a, _m=move, **k: _m.body_calls.append(a))
            if stale_flag:
                move.gf_skip_partner_once = True
            if stage_kind is not None:
                move.gf_stage_kind = stage_kind
            return move

        with mock.patch.object(OM, "make_move", wrapped):
            return OM.run_propose(self.state, size=size)

    def _common_bookkeeping(self, head, new, acc):
        self.assertIs(new, self.state)                  # nothing to write back
        self.assertEqual(np.shape(acc), (NTEMPS, NWALKERS))
        self.assertFalse(np.any(acc))
        # setup() ran (refit clock + a due fit live there), then the counter
        self.assertEqual(head.setup_calls, [7])
        self.assertEqual(head.num_proposals, 8)
        # the per-band empty-band valve tick, on this propose's occupancy
        self.assertEqual(len(head.shutoff_calls), 1)
        np.testing.assert_array_equal(
            head.shutoff_calls[0], head._band_occupancy_cold_max(self.state))
        # the designated updater's block: search stage + the valve JUDGE
        counts = head._band_counts_from_state(self.state)
        self.assertEqual(len(head.stage_calls), 1)
        np.testing.assert_array_equal(head.stage_calls[0], counts)
        self.assertEqual(len(head.judge_calls), 1)
        self.assertIs(head.judge_calls[0][0], self.state)
        np.testing.assert_array_equal(head.judge_calls[0][1], counts)
        # no proposal body at all
        self.assertEqual(head.body_calls, [])
        # the leg runner is asked to skip the in-model partner
        self.assertTrue(head.gf_skip_partner_once)

    def test_single_process_does_the_bookkeeping_and_no_proposal(self):
        with self.assertLogs(LOGGER, "INFO") as logs:
            (new, acc), moves = self._run(size=1)
        head = moves[0]
        self._common_bookkeeping(head, new, acc)
        # caps computed locally (the whole ensemble is here)
        self.assertEqual(head.cap_calls, [None])
        self.assertIsNone(getattr(head, "_stage_band_lls", None))
        self.assertIn(
            f"[GB_FSTAT {OM.MOVE_NAME}] epoch {ZERO_EPOCH} has 0 peaks: F-stat "
            "RJ births/deaths and the in_model_fstat partner skipped this cycle "
            "(bookkeeping only)", "\n".join(logs.output))

    def test_fanout_sends_no_command_and_tells_the_judge_it_observed_nothing(self):
        with self.assertLogs(LOGGER, "INFO") as logs:
            (new, acc), moves = self._run(size=3)
        head = moves[0]
        self._common_bookkeeping(head, new, acc)
        for rank, move in moves.items():
            self.assertEqual(move.served, [], f"rank {rank} was sent a command")
        # no N-walker statistic without the ranks: caps not advanced (warned)
        self.assertEqual(head.cap_calls, [])
        self.assertIn("leaf caps are NOT advanced", "\n".join(logs.output))
        # ... and the judge reads an all-NaN stash for THIS propose instead of
        # a head-local (one-block) statistic
        self.assertEqual(head._stage_band_lls_stamp, head.num_proposals)
        lls = head._shutoff_band_lls(None, self.state)
        self.assertEqual(lls.shape, (NWALKERS, NUM_BANDS))
        self.assertTrue(np.all(np.isnan(lls)))

    def test_the_pe_twin_logs_that_it_has_no_partner(self):
        with self.assertLogs(LOGGER, "INFO") as logs:
            (new, acc), moves = self._run(size=1, stage_kind="pe")
        self._common_bookkeeping(moves[0], new, acc)
        self.assertIn("a PE stage has no in-model partner to skip",
                      "\n".join(logs.output))

    def test_without_a_zero_peak_epoch_nothing_changes_and_a_stale_flag_clears(self):
        (_new, _acc), moves = self._run(size=3, zero_peaks=False, stale_flag=True)
        for rank in (0, 1):
            self.assertEqual(moves[rank].served,
                             ["gb_run_proposal", "gb_run_tempering", "gb_finish"])
        self.assertFalse(moves[0].gf_skip_partner_once)

    def test_band_counts_use_the_sorter_convention(self):
        move = OM.make_move(0, (0, NWALKERS))
        work = move._work_branch(self.state)
        work.inds[:] = False
        # temp 0 walker 1: an exact band edge goes UP (side="right"), and the
        # two out-of-grid leaves are not counted
        for leaf, f0_mHz in enumerate((2.0, 2.5, 0.5, 7.0)):
            work.coords[0, 1, leaf, 1] = f0_mHz
            work.inds[0, 1, leaf] = True
        work.coords[2, 3, 0, 1] = 6.999
        work.inds[2, 3, 0] = True
        got = move._band_counts_from_state(self.state)
        want = np.zeros((NTEMPS, NWALKERS, NUM_BANDS), dtype=int)
        want[0, 1, 1] = 2
        want[2, 3, 5] = 1
        np.testing.assert_array_equal(got, want)


class ZeroPeakWiringTest(unittest.TestCase):
    """Both bodies check the skip AFTER setup() (the refit clock and a due
    fit) and BEFORE the no-distribution return (which would silence the
    valve judge)."""

    def _order(self, fn):
        src = inspect.getsource(fn)
        needles = ("self.setup(model, state.branches)",
                   "self._propose_zero_peak_skip(",
                   "if self.is_rj_prop and self.rj_proposal_distribution is None:")
        pos = [src.index(n) for n in needles]
        self.assertEqual(pos, sorted(pos))

    def test_the_legacy_body(self):
        self._order(GBSpecialBase._propose_legacy)

    def test_the_orchestrator_head(self):
        self._order(GBSpecialBase._propose_orchestrated)


if __name__ == "__main__":
    unittest.main()
