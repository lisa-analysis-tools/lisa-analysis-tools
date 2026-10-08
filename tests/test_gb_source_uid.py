"""Per-source IDs for GB_CHOL_CACHE: continuity through every path that moves
or relabels GB leaves, and the cache keyed by ``(walker, ID)``.

Every living GB source carries an int64 ID (``-1`` = none). It lives on the
GB sub-state (``GBState.source_uid``, next to the per-leaf ``d_h``/``h_h``),
rides each propose's ``BandSorter`` as a per-ROW column (``source_uid``,
gathered like ``coords``), and goes back to the sub-state in the write-back
repack. The paths covered here:

* sorter build (alive-only and ``keep_all_inds``; subsets);
* the first use after load (every alive leaf gets an ID, exactly once);
* RJ birth (new ID) and death (ID retired), replace (new ID);
* the vertical swap (a relabel of the row: the ID follows the source);
* the write-back repack (IDs land where their sources land);
* GFState copy, walker slice/merge (multi-rank fan-out), the head's block
  merge, the one-walker replica merge, the cold->hot reseed;
* NOT the HDF5 store (the ID is per-process state; the cache is empty after
  a relaunch anyway).

No GB kernel and no GPU: a real CPU ``BandSorter`` over a small GB sub-state,
and a stub info-matrix engine whose factor for a source carries the source's
MARKER (sampling column 3, unique per source) so a test can tell whose factor
a row received.
"""

import copy
import inspect
import os
import pickle
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.communication.walkerslice import slice_state
from lisatools.globalfit.moves import gb_chol_cache as CC
from lisatools.globalfit.moves import gbspecialstretch as gbs
from lisatools.globalfit.moves.gbbands import BandSorter, pack_special_index
from lisatools.globalfit.state import GBState, GFState

NT, NW, NL, ND = 3, 2, 8, 4
BAND_EDGES = np.linspace(1e-3, 7e-3, 7)          # Hz, 6 bands
NUM_BANDS = len(BAND_EDGES) - 1
N_VALS = np.full(NUM_BANDS, 128)
SIG = np.array([0.01, 1e-6, 1e-3, 0.1])
MARK = 3                                          # the per-source marker column
#: the cell (walker 0, band 0) is occupied on rungs 0 AND 1 (leaf 0 each), so
#: a vertical swap there moves a real source in each direction
SWAP_W, SWAP_B = 0, 0

_ENV_ON = {"GB_CHOL_CACHE": "1"}


def _env(**kv):
    return mock.patch.dict(os.environ, kv)


def make_state(seed=0):
    """A GB-only GFState with a tempered GB sub-state (no IDs yet)."""
    rng = np.random.default_rng(seed)
    coords = np.zeros((NT, NW, NL, ND))
    coords[..., 0] = rng.uniform(0.5, 1.5, (NT, NW, NL))
    coords[..., 1] = rng.uniform(1.2, 6.8, (NT, NW, NL))   # f0 [mHz], in-grid
    coords[..., 2] = rng.uniform(0.1, 0.5, (NT, NW, NL))
    coords[..., MARK] = 1.0 + np.arange(NT * NW * NL).reshape(NT, NW, NL)
    inds = rng.random((NT, NW, NL)) < 0.6
    coords[0:2, SWAP_W, 0, 1] = [1.5, 1.7]       # band 0 on rungs 0 and 1
    inds[0:2, SWAP_W, 0] = True
    state = GFState(
        {"gb": coords.copy()},
        inds={"gb": inds.copy()},
        log_like=np.zeros((NT, NW)),
        log_prior=np.zeros((NT, NW)),
        betas=np.linspace(1.0, 0.1, NT),
        sub_state_bases={"gb": GBState},
    )
    sub = state.sub_states["gb"]
    sub.initialize_band_information(NW, NT, BAND_EDGES, np.ones((NUM_BANDS, NT)))
    sub.pull_from_main(state, "gb")
    return state


class _RJProp:
    """The two methods ``BandSorter`` calls on an RJ proposal; markers of the
    pre-drawn dead-row fill are unique (>= 1e5) so they never alias a source."""

    def __init__(self):
        self.next = 100000.0

    def rvs(self, size):
        out = np.zeros((size, ND))
        out[:, 0], out[:, 1], out[:, 2] = 1.0, 3.3, 0.3
        out[:, MARK] = self.next + np.arange(size)
        self.next += size
        return out

    @staticmethod
    def logpdf(x):
        return np.zeros(len(x))


_FROM_STATE = object()


def make_sorter(state, keep_all_inds=True, uid=_FROM_STATE):
    sub = state.sub_states["gb"]
    return BandSorter(
        sub.branch, BAND_EDGES, N_VALS, force_backend="cpu",
        rj_prop=_RJProp() if keep_all_inds else None,
        keep_all_inds=keep_all_inds,
        source_uid=sub.ensure_source_uid() if uid is _FROM_STATE else uid,
    )


def make_move():
    """A skeleton GB move carrying what the ID helpers and the repack read."""
    m = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
    m.branch_name = "gb"
    m.name = "gb_uid_test"
    m._backend_name = "lisatools_cpu"
    m.preserve_leaf_identity = False
    m.nwalkers, m.ntemps = NW, NT
    m._sorter_dh = None
    return m


def alive_uid_by_marker(state):
    """``{marker: uid}`` over the sub-state's alive leaves (and the dead-slot
    uids, which must all be -1)."""
    sub = state.sub_states["gb"]
    alive = sub.inds
    marks = sub.coords[..., MARK][alive]
    uids = sub.source_uid[alive]
    return dict(zip(marks.tolist(), uids.tolist())), sub.source_uid[~alive]


def sorter_uid_by_marker(sorter):
    a = sorter.inds
    return dict(zip(sorter.coords[a, MARK].tolist(), sorter.source_uid[a].tolist()))


def swap_rungs_0_1(sorter):
    """The vertical swap of cell (rung 0, SWAP_W, SWAP_B) with rung 1, through
    the sorter's real label exchange."""
    sa = np.array([pack_special_index(0, SWAP_W, SWAP_B, NW)])
    sb = np.array([pack_special_index(1, SWAP_W, SWAP_B, NW)])
    sorter.exchange_cell_labels_batch(
        sa, np.array([0]), np.array([SWAP_W]), sb, np.array([1]), np.array([SWAP_W]))


def cell_rows(sorter, t):
    """Alive rows of the swap cell on rung ``t``."""
    return np.nonzero(sorter.inds & (sorter.temp_inds == t)
                      & (sorter.walker_inds == SWAP_W)
                      & (sorter.band_inds == SWAP_B))[0]


# ----------------------------------------------------------------------
# the state side
# ----------------------------------------------------------------------
class GBStateSourceUidTest(unittest.TestCase):
    def setUp(self):
        self.state = make_state()
        self.sub = self.state.sub_states["gb"]
        self.sub.source_uid[:] = np.arange(self.sub.inds.size).reshape(self.sub.inds.shape)

    def test_a_fresh_tempered_block_has_no_ids(self):
        sub = make_state().sub_states["gb"]
        self.assertEqual(sub.source_uid.dtype, np.int64)
        self.assertEqual(sub.source_uid.shape, (NT, NW, NL))
        self.assertTrue((sub.source_uid == -1).all())

    def test_ensure_allocates_on_an_object_that_predates_the_ids(self):
        del self.sub.source_uid                     # an older pickle
        uid = self.sub.ensure_source_uid()
        self.assertIs(uid, self.sub.source_uid)
        self.assertTrue((uid == -1).all())
        self.assertEqual(uid.shape, (NT, NW, NL))

    def test_gfstate_copy_carries_the_ids_by_value(self):
        new = GFState(self.state, copy=True)
        np.testing.assert_array_equal(
            new.sub_states["gb"].source_uid, self.sub.source_uid)
        new.sub_states["gb"].source_uid[:] = -7
        self.assertFalse((self.sub.source_uid == -7).any())

    def test_walker_slice_and_merge_carry_the_ids(self):
        part = self.sub.slice_walkers(1, 2)
        np.testing.assert_array_equal(part.source_uid, self.sub.source_uid[:, 1:2])
        part.source_uid[:] = 4242
        self.sub.merge_walkers(part, 1, 2)
        self.assertTrue((self.sub.source_uid[:, 1] == 4242).all())
        self.assertFalse((self.sub.source_uid[:, 0] == 4242).any())

    def test_the_fanout_slice_ships_the_ids(self):
        part = slice_state(self.state, 1, 2, sub_states=["gb"])
        np.testing.assert_array_equal(
            part.sub_states["gb"].source_uid, self.sub.source_uid[:, 1:2])

    def test_reseed_gives_the_hot_rung_new_sources(self):
        before = self.sub.source_uid.copy()
        self.sub.reseed_cold_into_hottest(perm=np.array([1, 0]))
        self.assertTrue((self.sub.source_uid[NT - 1] == -1).all())
        np.testing.assert_array_equal(self.sub.source_uid[:NT - 1], before[:NT - 1])

    def test_the_ids_are_not_part_of_the_hdf5_schema(self):
        self.assertNotIn("source_uid", self.sub.storage_arrays())
        self.assertNotIn("source_uid", self.sub.static_arrays())

    def test_a_pickled_state_keeps_its_ids(self):
        """The mid-iteration checkpoint pickles the state: the IDs come back."""
        back = pickle.loads(pickle.dumps(copy.deepcopy(self.state)))
        np.testing.assert_array_equal(
            back.sub_states["gb"].source_uid, self.sub.source_uid)


# ----------------------------------------------------------------------
# the sorter side
# ----------------------------------------------------------------------
class SorterSourceUidTest(unittest.TestCase):
    def setUp(self):
        self.state = make_state()
        self.sub = self.state.sub_states["gb"]
        self.sub.source_uid[:] = np.where(
            self.sub.inds, 700 + np.arange(self.sub.inds.size).reshape(self.sub.inds.shape), -1)
        self.move = make_move()

    def test_keep_all_inds_sorter_gathers_ids_like_coords(self):
        uid0 = self.sub.source_uid.copy()
        s = make_sorter(self.state, keep_all_inds=True)
        np.testing.assert_array_equal(s.source_uid, np.where(
            self.sub.inds.reshape(-1), uid0.reshape(-1), -1))
        s.source_uid[:] = 1                      # a copy, never a view
        np.testing.assert_array_equal(self.sub.source_uid, uid0)

    def test_alive_only_sorter_gathers_ids_like_coords(self):
        s = make_sorter(self.state, keep_all_inds=False)
        np.testing.assert_array_equal(s.source_uid, self.sub.source_uid[self.sub.inds])

    def test_a_dead_row_never_carries_an_id(self):
        self.sub.source_uid[~self.sub.inds] = 99999          # stale garbage
        s = make_sorter(self.state, keep_all_inds=True)
        self.assertTrue((s.source_uid[~s.inds] == -1).all())

    def test_a_subset_carries_its_rows_ids(self):
        s = make_sorter(self.state, keep_all_inds=True)
        sub = s.get_subset(temp=1)
        np.testing.assert_array_equal(
            sub.source_uid, s.source_uid[s.temp_inds == 1])

    def test_no_ids_given_means_none(self):
        s = make_sorter(self.state, uid=None)
        self.assertIsNone(s.source_uid)
        self.assertEqual(self.move._issue_missing_uids(s), 0)

    def test_first_use_after_load_issues_every_alive_leaf_an_id_exactly_once(self):
        self.sub.source_uid[:] = -1                           # e.g. from_stored
        s = make_sorter(self.state)
        n_alive = int(s.inds.sum())
        self.assertEqual(self.move._issue_missing_uids(s), n_alive)
        got = s.source_uid[s.inds]
        self.assertTrue((got >= 0).all())
        self.assertEqual(len(np.unique(got)), n_alive)
        self.assertTrue((s.source_uid[~s.inds] == -1).all())
        self.assertEqual(self.move._issue_missing_uids(s), 0)  # exactly once
        # ...and it survives the repack: the next propose issues nothing
        self.move._write_back_state(self.state, s)
        self.assertEqual(self.move._issue_missing_uids(make_sorter(self.state)), 0)

    def test_rj_birth_gets_a_new_id_and_death_retires_its_id(self):
        s = make_sorter(self.state)
        before = set(s.source_uid[s.inds].tolist())
        birth = int(np.nonzero(~s.inds)[0][0])
        death = int(np.nonzero(s.inds)[0][0])
        acc_ids = np.array([birth, death])
        s.inds[acc_ids] = ~s.inds[acc_ids]                    # the RJ accept flip
        self.move._rj_flip_uids(s, acc_ids)
        self.assertEqual(s.source_uid[death], -1)
        self.assertGreaterEqual(s.source_uid[birth], 0)
        self.assertNotIn(int(s.source_uid[birth]), before)

    def test_replace_gives_the_replaced_row_a_new_id(self):
        s = make_sorter(self.state)
        rows = np.nonzero(s.inds)[0][:2]
        others = np.nonzero(s.inds)[0][2:]
        keep = s.source_uid[others].copy()
        old = set(s.source_uid[s.inds].tolist())
        self.move._replace_uids(s, rows)
        self.assertTrue((s.source_uid[rows] >= 0).all())
        self.assertFalse(set(s.source_uid[rows].tolist()) & old)
        np.testing.assert_array_equal(s.source_uid[others], keep)

    def test_the_gate(self):
        """IDs ride only with GB_CHOL_CACHE=1, and GB_CHOL_CACHE_UID=0 rolls
        back to coordinate matching for every source."""
        with _env(GB_CHOL_CACHE="0"):
            self.assertIsNone(self.move._state_source_uid(self.state))
        with _env(GB_CHOL_CACHE="1"):
            self.assertIs(self.move._state_source_uid(self.state), self.sub.source_uid)
            with _env(GB_CHOL_CACHE_UID="0"):
                self.assertIsNone(self.move._state_source_uid(self.state))


# ----------------------------------------------------------------------
# continuity: birth, death, replace, vertical swap, repack
# ----------------------------------------------------------------------
class SourceUidContinuityTest(unittest.TestCase):
    def setUp(self):
        self.state = make_state(seed=3)
        self.move = make_move()
        self.sorter = make_sorter(self.state)
        self.move._issue_missing_uids(self.sorter)

    def test_the_swap_relabels_the_row_and_the_id_stays_with_the_source(self):
        s = self.sorter
        rows0, rows1 = cell_rows(s, 0), cell_rows(s, 1)
        self.assertTrue(rows0.size and rows1.size)
        uid_before = s.source_uid.copy()
        coords_before = s.coords.copy()
        swap_rungs_0_1(s)
        np.testing.assert_array_equal(s.temp_inds[rows0], 1)
        np.testing.assert_array_equal(s.temp_inds[rows1], 0)
        np.testing.assert_array_equal(s.source_uid, uid_before)
        np.testing.assert_array_equal(s.coords, coords_before)

    def test_the_deferred_swap_window_keeps_the_ids_too(self):
        s = self.sorter
        uid_before = s.source_uid.copy()
        rows0 = cell_rows(s, 0)
        with _env(GB_CELL_LABEL_DEFERRED="1"):
            self.assertTrue(s.begin_cell_label_window(s.special_band_inds))
            swap_rungs_0_1(s)
            s.flush_cell_labels(close=True)
        np.testing.assert_array_equal(s.temp_inds[rows0], 1)
        np.testing.assert_array_equal(s.source_uid, uid_before)

    def test_repack_puts_every_id_where_its_source_lands(self):
        """No swap: the dense per-(rung, walker) f0 repack moves leaves; each
        ID must land with its source, dead slots carry none."""
        want = sorter_uid_by_marker(self.sorter)
        self.move._write_back_state(self.state, self.sorter)
        got, dead = alive_uid_by_marker(self.state)
        self.assertEqual(got, want)
        self.assertTrue((dead == -1).all())

    def test_ids_follow_sources_through_birth_death_replace_swap_and_repack(self):
        s = self.sorter
        mv = self.move
        # RJ: one birth (a dead row of the swap walker) and one death
        birth = int(np.nonzero(~s.inds & (s.walker_inds == 1))[0][0])
        death = int(np.nonzero(s.inds & (s.walker_inds == 1))[0][-1])
        acc = np.array([birth, death])
        s.inds[acc] = ~s.inds[acc]
        s.coords[birth, MARK] = -5.0
        mv._rj_flip_uids(s, acc)
        # replace: an alive row outside the swap cell gets a new source
        rep = int([r for r in np.nonzero(s.inds & (s.walker_inds == 1))[0]
                   if r != birth][0])
        s.coords[rep, MARK] = -6.0
        mv._replace_uids(s, np.array([rep]))
        # in-model: every alive source moves (coords change, leaf stays)
        s.coords[s.inds, 1] += 1e-4
        # the vertical swap
        moved_up = cell_rows(s, 0)
        swap_rungs_0_1(s)
        want = sorter_uid_by_marker(s)
        # read BEFORE the repack: on CPU a keep_all_inds sorter's coords are a
        # view of the sub-state array the repack rewrites
        up = [(float(s.coords[r, MARK]), int(s.source_uid[r])) for r in moved_up]
        self.assertTrue(up)
        self.assertNotIn(death, np.nonzero(s.inds)[0])
        mv._write_back_state(self.state, s)
        got, dead = alive_uid_by_marker(self.state)
        self.assertEqual(got, want)
        self.assertTrue((dead == -1).all())
        self.assertEqual(len(set(got.values())), len(got))         # unique
        # the swapped source now sits on rung 1 carrying its own ID
        sub = self.state.sub_states["gb"]
        for m, u in up:
            # alive slots only: a vacated slot keeps its stale coords
            t, w, l = (int(v[0]) for v in
                       np.nonzero((sub.coords[..., MARK] == m) & sub.inds))
            self.assertEqual((t, w), (1, SWAP_W))
            self.assertEqual(int(sub.source_uid[t, w, l]), u)

    def test_preserve_leaf_identity_repack_keeps_the_ids_at_their_leaves(self):
        self.move.preserve_leaf_identity = True
        want = sorter_uid_by_marker(self.sorter)
        self.move._write_back_state(self.state, self.sorter)
        got, dead = alive_uid_by_marker(self.state)
        self.assertEqual(got, want)
        self.assertTrue((dead == -1).all())

    def test_a_sorter_without_ids_leaves_the_state_without_ids(self):
        """The rollback path (GB_CHOL_CACHE_UID=0): nothing stale survives."""
        self.move._write_back_state(self.state, self.sorter)     # IDs written
        self.assertTrue((self.state.sub_states["gb"].source_uid >= 0).any())
        s = make_sorter(self.state, uid=None)
        self.move._write_back_state(self.state, s)
        self.assertTrue((self.state.sub_states["gb"].source_uid == -1).all())


# ----------------------------------------------------------------------
# multi-rank: the head's block merge and the one-walker replica merge
# ----------------------------------------------------------------------
class ReplicaMergeSourceUidTest(unittest.TestCase):
    def _reply(self, f0s, bands, uids, nl=6):
        c = np.full((1, 1, nl, 2), -99.0)
        i = np.zeros((1, 1, nl), bool)
        b = np.full((1, 1, nl), -1, np.int64)
        u = np.full((1, 1, nl), -1, np.int64)
        for k, (f0, band, uid) in enumerate(zip(f0s, bands, uids)):
            c[0, 0, k] = [float(band), f0]
            i[0, 0, k] = True
            b[0, 0, k] = band
            u[0, 0, k] = uid
        return {"block_coords": c, "block_inds": i, "block_band_inds": b,
                "block_source_uid": u, "d_h": np.zeros((1, nl)),
                "h_h": np.zeros((1, nl))}

    def test_ids_follow_their_sources_through_the_merge_by_source(self):
        # rank 0 owns bands [0, 2): a birth (uid 9) in band 1; rank 1 owns
        # [2, 4): a death in band 3 -- each rank's leaves are shifted
        replies = {
            0: self._reply([1.5, 3.0, 3.5, 5.5, 7.5], [0, 1, 1, 2, 3], [1, 9, 2, 3, 4]),
            1: self._reply([1.5, 3.5, 5.5], [0, 1, 2], [1, 2, 3]),
        }
        work_c = np.zeros((1, 1, 6, 2))
        work_i = np.zeros((1, 1, 6), bool)
        uid = np.full((1, 1, 6), 77, np.int64)
        gbs.merge_owned_sources(work_c, work_i, replies, (0, 1),
                                {0: (0, 2), 1: (2, 4)}, 1, source_uid=uid)
        np.testing.assert_array_equal(work_c[0, 0, :4, 1], [1.5, 3.0, 3.5, 5.5])
        np.testing.assert_array_equal(uid[0, 0], [1, 9, 2, 3, -1, -1])

    def test_preserve_leaf_identity_merge_keeps_ids_at_their_slots(self):
        replies = {
            0: self._reply([1.5, 3.5], [0, 1], [5, 6]),
            1: self._reply([1.5, 3.5, 5.5], [0, 1, 2], [5, 6, 7]),
        }
        uid = np.full((1, 1, 6), 77, np.int64)
        gbs.merge_owned_sources(np.zeros((1, 1, 6, 2)), np.zeros((1, 1, 6), bool),
                                replies, (0, 1), {0: (0, 2), 1: (2, 4)}, 1,
                                source_uid=uid, preserve_leaf_identity=True)
        np.testing.assert_array_equal(uid[0, 0], [5, 6, 7, -1, -1, -1])


class WiringTest(unittest.TestCase):
    """Source guards for the call sites the skeleton tests cannot drive (the
    accept sites live inside the kernel-driven RJ / replace steps)."""

    def _src(self, fn):
        return inspect.getsource(fn)

    def test_rj_accept_flip_updates_the_ids(self):
        src = self._src(gbs.GBSpecialBase._run_rj_step)
        i = src.index("band_sorter.inds[acc_ids] = ~band_sorter.inds[acc_ids]")
        self.assertIn("self._rj_flip_uids(band_sorter, acc_ids)", src[i:i + 400])

    def test_replace_accept_issues_new_ids(self):
        src = self._src(gbs.GBSpecialBase._run_replace_step)
        i = src.index("band_sorter.coords[acc_ids] = wrapped_new")
        self.assertIn("self._replace_uids(band_sorter, acc_ids)", src[i:i + 400])

    def test_both_propose_bodies_build_the_sorter_with_ids_before_proposing(self):
        for fn, state in ((gbs.GBSpecialBase._propose_legacy, "new_state"),
                          (gbs.GBSpecialBase._gb_serve_run_proposal, "new_part")):
            src = self._src(fn)
            i = src.index(f"source_uid=self._state_source_uid({state})")
            j = src.index("self._issue_missing_uids(band_sorter)", i)
            k = src.index("self.run_proposal(", j)
            self.assertLess(i, j)
            self.assertLess(j, k)

    def test_the_rank_finish_ships_the_block_ids(self):
        src = self._src(gbs.GBSpecialBase._gb_serve_finish)
        self.assertIn('"block_source_uid": block_source_uid', src)


# ----------------------------------------------------------------------
# end to end: the cache serves each source its own factor after all of it
# ----------------------------------------------------------------------
class _StubEngine:
    """Info-matrix stand-in: the factor of a source is diag(SIG) with the
    source's MARKER at [MARK, MARK]."""

    name = "gb_uid_stub"
    xp = np
    _fdot_col = None
    _obs_gamma_z = None

    def __init__(self):
        self.calls = []

    @staticmethod
    def _obs_eigen_mode():
        return "off"

    def _compute_proposal_cholesky(self, model, bs, ids, slots=None, buffer_obj=None):
        ids = np.asarray(ids)
        self.calls.append(ids.copy())
        self._proposal_param_scales = np.ones(ND)
        B = np.zeros((len(ids), ND, ND))
        B[:, np.arange(ND), np.arange(ND)] = SIG
        B[:, MARK, MARK] = bs.coords[ids, MARK]
        return B


class CacheAfterContinuityTest(unittest.TestCase):
    def test_each_source_keeps_its_factor_through_a_whole_propose(self):
        state = make_state(seed=11)
        mv = make_move()
        with _env(**_ENV_ON):
            s = make_sorter(state, uid=mv._state_source_uid(state))
        mv._issue_missing_uids(s)
        eng = _StubEngine()
        cache = CC._CholCache(("gb", "off", False))
        cache.due(0)
        cache.refresh(eng, None, s)
        eng.calls.clear()
        # the propose: birth, death, replace, every source pushed 50 tolerances
        # by in-model steps (no retrack), the vertical swap, the repack
        birth = int(np.nonzero(~s.inds)[0][0])
        death = int(np.nonzero(s.inds & (s.temp_inds == 2))[0][0])
        acc = np.array([birth, death])
        s.inds[acc] = ~s.inds[acc]
        s.coords[birth, MARK] = -5.0
        mv._rj_flip_uids(s, acc)
        rep = int(np.nonzero(s.inds & (s.temp_inds == 2))[0][0])
        s.coords[rep, MARK] = -6.0
        mv._replace_uids(s, np.array([rep]))
        s.coords[s.inds, 0] += 50 * cache.tol * SIG[0]
        swap_rungs_0_1(s)
        mv._write_back_state(state, s)
        # the next propose's sorter: only the birth and the replacement miss
        with _env(**_ENV_ON):
            s2 = make_sorter(state, uid=mv._state_source_uid(state))
        self.assertEqual(mv._issue_missing_uids(s2), 0)
        ids = np.nonzero(s2.inds)[0]
        chol = cache.take(eng, None, s2, ids, ids, None)
        np.testing.assert_array_equal(chol[:, MARK, MARK], s2.coords[ids, MARK])
        missed = s2.coords[np.concatenate(eng.calls), MARK] if eng.calls else []
        self.assertEqual(sorted(np.asarray(missed).tolist()), [-6.0, -5.0])


if __name__ == "__main__":
    unittest.main()
