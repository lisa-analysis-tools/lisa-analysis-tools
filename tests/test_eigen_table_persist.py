"""Tests for the eigen-table sidecar (persistence across process restarts).

The per-leaf ``(axes, sigmas)`` tables that
``ResidualAddOneRemoveOneMove.refresh_inner_move_tables`` feeds to the
eryn :class:`~eryn.moves.EigenAxisMove` inner moves cost minutes per leaf
to build from an information matrix, and used to live ONLY on the move
object — so every process restart (spot-partition preemption) repaid the
whole first-visit build. ``eigen_table_persist`` writes the DERIVED
tables to a sidecar pickle next to the run's store so a resume reloads
them.

Guards are deliberately narrow: only what makes shapes/physics wrong
(ndim, ladder, walker count, scope, data identity). A stale expansion
point is fine — the table was already frozen between refreshes, so a
frozen table is correct MH whatever point it came from.
"""

import os
import pickle
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from eryn.moves import EigenAxisMove, StretchMove
from eryn.prior import ProbDistContainer, uniform_dist

from lisatools.globalfit.moves import eigen_refresh, eigen_table_persist


def _walker_max_entry(ndim=3, ntemps=2, nwalkers=4, visits=1, seed=11):
    """A ``walker_max`` entry: ONE ``(ndim, ndim)`` + ``(ndim,)`` table."""
    rng = np.random.default_rng(seed)
    axes = np.linalg.qr(rng.standard_normal((ndim, ndim)))[0]
    sigmas = rng.uniform(0.1, 1.0, ndim)
    return eigen_table_persist.make_entry(
        axes, sigmas, visits, scope="walker_max", ndim=ndim,
        ntemps=ntemps, nwalkers=nwalkers, x0=rng.standard_normal(ndim),
    )


def _per_walker_entry(ndim=3, ntemps=2, nwalkers=4, visits=7, seed=12):
    """A ``per_walker`` stash: ``(ntemps, nwalkers, ndim, ndim)``."""
    rng = np.random.default_rng(seed)
    axes = rng.standard_normal((ntemps, nwalkers, ndim, ndim))
    sigmas = rng.uniform(0.1, 1.0, (ntemps, nwalkers, ndim))
    return eigen_table_persist.make_entry(
        axes, sigmas, visits, scope="per_walker", ndim=ndim,
        ntemps=ntemps, nwalkers=nwalkers,
        x0=rng.standard_normal((ntemps, nwalkers, ndim)),
    )


class SidecarPathTest(unittest.TestCase):
    def test_path_sits_next_to_the_store_h5(self):
        self.assertEqual(
            eigen_table_persist.sidecar_path("/runs/gf_prod_6mo.h5"),
            "/runs/gf_prod_6mo_eigen_tables.pkl",
        )

    def test_path_without_the_h5_suffix_still_appends(self):
        self.assertEqual(
            eigen_table_persist.sidecar_path("/runs/store"),
            "/runs/store_eigen_tables.pkl",
        )

    def test_no_store_path_means_no_sidecar(self):
        self.assertIsNone(eigen_table_persist.sidecar_path(None))
        self.assertIsNone(eigen_table_persist.sidecar_path(""))


class RoundTripTest(unittest.TestCase):
    """Test 1: save -> reload gives back the exact arrays and counters."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "store_eigen_tables.pkl")

    def test_walker_max_round_trip(self):
        entry = _walker_max_entry(visits=3)
        eigen_table_persist.save_entry(self.path, "mbh", 2, entry)

        got = eigen_table_persist.load_entry(
            self.path, "mbh", 2, scope="walker_max", ndim=3,
            ntemps=2, nwalkers=4,
        )
        self.assertIsNotNone(got)
        axes, sigmas, visits = got
        np.testing.assert_array_equal(axes, entry["axes"])
        np.testing.assert_array_equal(sigmas, entry["sigmas"])
        self.assertEqual(visits, 3)

    def test_per_walker_stash_round_trip(self):
        entry = _per_walker_entry(visits=7)
        eigen_table_persist.save_entry(self.path, "sobbh", 0, entry)

        got = eigen_table_persist.load_entry(
            self.path, "sobbh", 0, scope="per_walker", ndim=3,
            ntemps=2, nwalkers=4,
        )
        self.assertIsNotNone(got)
        axes, sigmas, visits = got
        self.assertEqual(axes.shape, (2, 4, 3, 3))
        self.assertEqual(sigmas.shape, (2, 4, 3))
        np.testing.assert_array_equal(axes, entry["axes"])
        np.testing.assert_array_equal(sigmas, entry["sigmas"])
        self.assertEqual(visits, 7)

    def test_many_leaves_and_branches_coexist_in_one_file(self):
        eigen_table_persist.save_entry(
            self.path, "mbh", 0, _walker_max_entry(seed=1)
        )
        eigen_table_persist.save_entry(
            self.path, "mbh", 3, _walker_max_entry(seed=2)
        )
        eigen_table_persist.save_entry(
            self.path, "emri", 0, _walker_max_entry(seed=3)
        )
        payload = eigen_table_persist.load_sidecar(self.path)
        self.assertEqual(len(payload["entries"]), 3)
        # a later save of the same (branch, leaf) replaces only that entry
        eigen_table_persist.save_entry(
            self.path, "mbh", 0, _walker_max_entry(seed=9)
        )
        payload = eigen_table_persist.load_sidecar(self.path)
        self.assertEqual(len(payload["entries"]), 3)

    def test_missing_file_loads_as_empty_without_warning(self):
        payload = eigen_table_persist.load_sidecar(
            os.path.join(self._tmp.name, "not_there.pkl")
        )
        self.assertEqual(payload["entries"], {})


class AtomicWriteTest(unittest.TestCase):
    """Test 2: a failed write leaves the previous sidecar intact, no litter."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "store_eigen_tables.pkl")

    def test_failed_replace_keeps_the_old_file_and_removes_the_tmp(self):
        good = _walker_max_entry(seed=4, visits=1)
        eigen_table_persist.save_entry(self.path, "mbh", 0, good)
        before = open(self.path, "rb").read()

        with mock.patch.object(
            eigen_table_persist.os, "replace",
            side_effect=OSError("disk full"),
        ):
            with self.assertLogs(eigen_refresh.logger, level="WARNING"):
                ok = eigen_table_persist.save_entry(
                    self.path, "mbh", 0, _walker_max_entry(seed=5, visits=2)
                )
        self.assertFalse(ok)
        self.assertEqual(open(self.path, "rb").read(), before)
        litter = [
            f for f in os.listdir(self._tmp.name)
            if f != os.path.basename(self.path)
        ]
        self.assertEqual(litter, [])

    def test_the_old_entry_is_still_loadable_after_a_failed_write(self):
        eigen_table_persist.save_entry(
            self.path, "mbh", 0, _walker_max_entry(seed=4, visits=1)
        )
        with mock.patch.object(
            eigen_table_persist.os, "replace", side_effect=OSError("nope")
        ):
            with self.assertLogs(eigen_refresh.logger, level="WARNING"):
                eigen_table_persist.save_entry(
                    self.path, "mbh", 0, _walker_max_entry(seed=5, visits=2)
                )
        got = eigen_table_persist.load_entry(
            self.path, "mbh", 0, scope="walker_max", ndim=3,
            ntemps=2, nwalkers=4,
        )
        self.assertIsNotNone(got)
        self.assertEqual(got[2], 1)


class GuardTest(unittest.TestCase):
    """Test 3: only shape/physics mismatches reject an entry."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "store_eigen_tables.pkl")
        eigen_table_persist.save_entry(
            self.path, "sobbh", 0, _per_walker_entry(ndim=3, ntemps=2,
                                                     nwalkers=4, visits=5)
        )

    def _load(self, **kwargs):
        base = dict(scope="per_walker", ndim=3, ntemps=2, nwalkers=4)
        base.update(kwargs)
        return eigen_table_persist.load_entry(self.path, "sobbh", 0, **base)

    def test_matching_guards_adopt(self):
        got = self._load()
        self.assertIsNotNone(got)
        self.assertEqual(got[2], 5)

    def test_wrong_ndim_rejects_with_a_warning(self):
        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            self.assertIsNone(self._load(ndim=4))

    def test_wrong_ntemps_rejects_with_a_warning(self):
        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            self.assertIsNone(self._load(ntemps=5))

    def test_wrong_nwalkers_rejects_with_a_warning(self):
        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            self.assertIsNone(self._load(nwalkers=10))

    def test_wrong_scope_rejects_with_a_warning(self):
        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            self.assertIsNone(self._load(scope="walker_max"))

    def test_stash_leading_dims_must_match_the_current_ladder(self):
        # metadata agrees with the run but the stashed array does not —
        # a partially written / mangled entry. The seam slices this table
        # with the (ntemps, nwalkers) proposal mask, so row-for-row
        # agreement is the guard that matters.
        entry = _per_walker_entry(ntemps=3, nwalkers=4, visits=2)
        entry["ntemps"] = 2
        eigen_table_persist.save_entry(self.path, "sobbh", 1, entry)
        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            self.assertIsNone(
                eigen_table_persist.load_entry(
                    self.path, "sobbh", 1, scope="per_walker", ndim=3,
                    ntemps=2, nwalkers=4,
                )
            )

    def test_sigmas_inconsistent_with_axes_rejects(self):
        entry = _walker_max_entry(visits=2)
        entry["sigmas"] = np.full(5, 0.1)
        eigen_table_persist.save_entry(self.path, "mbh", 7, entry)
        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            self.assertIsNone(
                eigen_table_persist.load_entry(
                    self.path, "mbh", 7, scope="walker_max", ndim=3,
                    ntemps=2, nwalkers=4,
                )
            )

    def test_a_shared_fallback_table_under_per_walker_scope_is_adopted(self):
        # the identity fallback is 2-dim even under per_walker scope (a
        # custom-builder failure returns _fallback_table); EigenAxisMove
        # broadcasts it, so adopting it is correct, not a guard failure
        entry = _per_walker_entry(visits=2)
        entry["axes"] = np.eye(3)
        entry["sigmas"] = np.full(3, 0.1)
        eigen_table_persist.save_entry(self.path, "sobbh", 2, entry)
        got = eigen_table_persist.load_entry(
            self.path, "sobbh", 2, scope="per_walker", ndim=3,
            ntemps=2, nwalkers=4,
        )
        self.assertIsNotNone(got)

    def test_data_identity_mismatch_rejects_but_a_missing_one_does_not(self):
        entry = _walker_max_entry(visits=1)
        entry["data_identity"] = 1.0 / (0.5 * 365.25 * 24 * 3600.0)
        eigen_table_persist.save_entry(self.path, "mbh", 0, entry)
        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            self.assertIsNone(
                eigen_table_persist.load_entry(
                    self.path, "mbh", 0, scope="walker_max", ndim=3,
                    ntemps=2, nwalkers=4, data_identity=1.0 / 3600.0,
                )
            )
        # unknown on either side -> the guard abstains (the sidecar path is
        # already run-scoped)
        self.assertIsNotNone(
            eigen_table_persist.load_entry(
                self.path, "mbh", 0, scope="walker_max", ndim=3,
                ntemps=2, nwalkers=4, data_identity=None,
            )
        )

    def test_absent_entry_returns_none_without_warning(self):
        self.assertIsNone(
            eigen_table_persist.load_entry(
                self.path, "sobbh", 41, scope="per_walker", ndim=3,
                ntemps=2, nwalkers=4,
            )
        )


class CorruptSidecarTest(unittest.TestCase):
    """Test 4: garbage bytes warn once and behave like an empty cache."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _corrupt(self, name):
        path = os.path.join(self._tmp.name, name)
        with open(path, "wb") as fp:
            fp.write(b"\x00\x01not a pickle at all\xff")
        return path

    def test_garbage_bytes_warn_and_load_empty(self):
        path = self._corrupt("garbage_eigen_tables.pkl")
        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            payload = eigen_table_persist.load_sidecar(path)
        self.assertEqual(payload["entries"], {})

    def test_garbage_bytes_do_not_raise_on_load_entry(self):
        path = self._corrupt("garbage2_eigen_tables.pkl")
        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            self.assertIsNone(
                eigen_table_persist.load_entry(
                    path, "mbh", 0, scope="walker_max", ndim=3,
                    ntemps=2, nwalkers=4,
                )
            )

    def test_a_corrupt_sidecar_is_overwritten_by_the_next_save(self):
        path = self._corrupt("garbage3_eigen_tables.pkl")
        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            eigen_table_persist.save_entry(
                path, "mbh", 0, _walker_max_entry(visits=1)
            )
        got = eigen_table_persist.load_entry(
            path, "mbh", 0, scope="walker_max", ndim=3, ntemps=2, nwalkers=4,
        )
        self.assertIsNotNone(got)

    def test_a_pickle_of_the_wrong_shape_is_ignored(self):
        path = os.path.join(self._tmp.name, "wrong_eigen_tables.pkl")
        with open(path, "wb") as fp:
            pickle.dump(["not", "a", "payload"], fp)
        with self.assertLogs(eigen_refresh.logger, level="WARNING"):
            payload = eigen_table_persist.load_sidecar(path)
        self.assertEqual(payload["entries"], {})


def _stub_move(inner_moves, store_path, ndim=3, refresh=10, scope=None,
               ntemps=2, nwalkers=4):
    """The ``test_eigen_refresh`` stub pattern + a store path to persist to.

    Built with ``__new__`` so nothing heavy runs; every attribute the
    refresh hook touches is filled by hand.
    """
    from lisatools.globalfit.moves.addremovemove import (
        ResidualAddOneRemoveOneMove,
    )

    move = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    move.branch_name = "sobbh"
    move.ndim = ndim
    move.ntemps = ntemps
    move.nwalkers = nwalkers
    move.moves = inner_moves
    move.priors = {
        "sobbh": ProbDistContainer(
            {i: uniform_dist(-5.0, 5.0) for i in range(ndim)}
        )
    }
    move.eigen_refresh_every = refresh
    move.eigen_eps_rel = None
    move.eigen_table_scope = scope
    move.eigen_store_path = store_path
    move._to_phys = lambda x: x
    inv = np.linalg.inv(np.eye(ndim))
    move.compute_like = lambda x, data_index=None: -0.5 * np.einsum(
        "ni,ij,nj->n", np.atleast_2d(x), inv, np.atleast_2d(x)
    )
    return move


def _work(ndim=3, nl=2, ntemps=2, nwalkers=4, seed=71):
    return types.SimpleNamespace(
        coords=np.random.default_rng(seed).standard_normal(
            (ntemps, nwalkers, nl, ndim)
        )
    )


class HookPersistenceTest(unittest.TestCase):
    """Tests 5-8: what the refresh hook does with/without a sidecar."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = os.path.join(self._tmp.name, "gf_prod_6mo_main.h5")
        self.sidecar = eigen_table_persist.sidecar_path(self.store)

    # -- test 5: a valid sidecar skips the build ---------------------
    def test_valid_sidecar_skips_the_builder_and_feeds_the_move(self):
        entry = _walker_max_entry(visits=4, seed=21)
        eigen_table_persist.save_entry(self.sidecar, "sobbh", 0, entry)

        inner = EigenAxisMove()
        move = _stub_move([inner], self.store, refresh=10)
        with mock.patch.object(
            eigen_refresh, "eigen_table_from_ll"
        ) as builder:
            move.refresh_inner_move_tables(0, _work())
        builder.assert_not_called()

        axes, sigmas = inner._tables["sobbh"]
        np.testing.assert_array_equal(axes, entry["axes"])
        np.testing.assert_array_equal(sigmas, entry["sigmas"])
        # the counter continues from the stored value (4 -> 5), so the
        # cadence clock is not reset by the restart
        self.assertEqual(move._eigen_visit_count[0], 5)

    def test_the_sidecar_is_read_once_per_leaf_not_every_visit(self):
        eigen_table_persist.save_entry(
            self.sidecar, "sobbh", 0, _walker_max_entry(visits=1, seed=22)
        )
        inner = EigenAxisMove()
        move = _stub_move([inner], self.store, refresh=10)
        work = _work()
        with mock.patch.object(
            eigen_table_persist, "load_entry",
            wraps=eigen_table_persist.load_entry,
        ) as spy:
            for _ in range(4):
                move.refresh_inner_move_tables(0, work)
        self.assertEqual(spy.call_count, 1)

    def test_a_stored_counter_on_the_cadence_rebuilds_immediately(self):
        # proves the counter is really adopted: 10 % 10 == 0 -> refresh
        eigen_table_persist.save_entry(
            self.sidecar, "sobbh", 0, _walker_max_entry(visits=10, seed=23)
        )
        inner = EigenAxisMove()
        move = _stub_move([inner], self.store, refresh=10)
        with mock.patch.object(
            eigen_refresh, "eigen_table_from_ll",
            wraps=eigen_refresh.eigen_table_from_ll,
        ) as builder:
            move.refresh_inner_move_tables(0, _work())
        self.assertEqual(builder.call_count, 1)

    def test_per_walker_stash_is_adopted_and_sliced_by_the_seam(self):
        entry = _per_walker_entry(visits=3, seed=24)
        eigen_table_persist.save_entry(self.sidecar, "sobbh", 1, entry)

        inner = EigenAxisMove()
        move = _stub_move([inner], self.store, refresh=10,
                          scope="per_walker")
        with mock.patch.object(
            eigen_refresh, "eigen_tables_from_ll_batch"
        ) as builder:
            move.refresh_inner_move_tables(1, _work())
        builder.assert_not_called()
        # a 4-dim stash is NOT installed at refresh time
        self.assertNotIn("sobbh", inner._tables)
        np.testing.assert_array_equal(
            move._eigen_tables[1][0], entry["axes"]
        )
        # ... the split seam installs the sliced 5-dim view
        mask = np.zeros((move.ntemps, move.nwalkers), dtype=bool)
        mask[:, ::2] = True
        move._install_eigen_split_table(inner, 1, mask)
        axes, sigmas = inner._tables["sobbh"]
        self.assertEqual(axes.shape, (move.ntemps, 2, 1, 3, 3))
        self.assertEqual(sigmas.shape, (move.ntemps, 2, 1, 3))

    def test_guard_mismatch_in_the_sidecar_triggers_a_rebuild(self):
        # stored for a 5-temperature ladder; this run has 2
        eigen_table_persist.save_entry(
            self.sidecar, "sobbh", 0,
            _walker_max_entry(visits=2, ntemps=5, seed=25),
        )
        inner = EigenAxisMove()
        move = _stub_move([inner], self.store, refresh=10)
        with mock.patch.object(
            eigen_refresh, "eigen_table_from_ll",
            wraps=eigen_refresh.eigen_table_from_ll,
        ) as builder:
            with self.assertLogs(eigen_refresh.logger, level="WARNING"):
                move.refresh_inner_move_tables(0, _work())
        self.assertEqual(builder.call_count, 1)
        self.assertEqual(move._eigen_visit_count[0], 1)

    # -- test 6: no sidecar -> build, then write it ------------------
    def test_without_a_sidecar_the_builder_runs_and_the_file_is_written(self):
        inner = EigenAxisMove()
        move = _stub_move([inner], self.store, refresh=10)
        self.assertFalse(os.path.exists(self.sidecar))

        with mock.patch.object(
            eigen_refresh, "eigen_table_from_ll",
            wraps=eigen_refresh.eigen_table_from_ll,
        ) as builder:
            move.refresh_inner_move_tables(0, _work())
        self.assertEqual(builder.call_count, 1)

        self.assertTrue(os.path.exists(self.sidecar))
        got = eigen_table_persist.load_entry(
            self.sidecar, "sobbh", 0, scope="walker_max", ndim=3,
            ntemps=2, nwalkers=4,
        )
        self.assertIsNotNone(got)
        axes, sigmas, visits = got
        np.testing.assert_array_equal(axes, move._eigen_tables[0][0])
        np.testing.assert_array_equal(sigmas, move._eigen_tables[0][1])
        self.assertEqual(visits, 1)

    def test_the_written_entry_carries_the_guards_and_expansion_point(self):
        move = _stub_move([EigenAxisMove()], self.store, refresh=10)
        move.refresh_inner_move_tables(2, _work(nl=3))
        payload = eigen_table_persist.load_sidecar(self.sidecar)
        entry = payload["entries"]["sobbh:2"]
        self.assertEqual(entry["ndim"], 3)
        self.assertEqual(entry["ntemps"], 2)
        self.assertEqual(entry["nwalkers"], 4)
        self.assertEqual(entry["scope"], "walker_max")
        self.assertIsNotNone(entry["x0"])
        self.assertEqual(np.asarray(entry["x0"]).shape, (3,))

    def test_a_per_walker_build_is_persisted_in_the_stashed_shape(self):
        move = _stub_move([EigenAxisMove()], self.store, refresh=10,
                          scope="per_walker")
        move.refresh_inner_move_tables(0, _work())
        got = eigen_table_persist.load_entry(
            self.sidecar, "sobbh", 0, scope="per_walker", ndim=3,
            ntemps=2, nwalkers=4,
        )
        self.assertIsNotNone(got)
        self.assertEqual(got[0].shape, (2, 4, 3, 3))
        self.assertEqual(got[1].shape, (2, 4, 3))

    def test_no_store_path_means_no_sidecar_and_no_crash(self):
        inner = EigenAxisMove()
        move = _stub_move([inner], None, refresh=10)
        move.refresh_inner_move_tables(0, _work())
        self.assertIn("sobbh", inner._tables)
        self.assertEqual(os.listdir(self._tmp.name), [])

    def test_a_failing_sidecar_write_does_not_break_the_refresh(self):
        inner = EigenAxisMove()
        move = _stub_move([inner], self.store, refresh=10)
        with mock.patch.object(
            eigen_table_persist, "save_entry",
            side_effect=OSError("read-only filesystem"),
        ):
            move.refresh_inner_move_tables(0, _work())
        self.assertIn("sobbh", inner._tables)

    # -- test 7: the env escape restores today's behavior ------------
    def test_persist_off_neither_reads_nor_writes_the_sidecar(self):
        eigen_table_persist.save_entry(
            self.sidecar, "sobbh", 0, _walker_max_entry(visits=4, seed=26)
        )
        before = open(self.sidecar, "rb").read()

        inner = EigenAxisMove()
        move = _stub_move([inner], self.store, refresh=10)
        with mock.patch.dict(os.environ, {"EIGEN_TABLES_PERSIST": "0"}):
            with mock.patch.object(
                eigen_refresh, "eigen_table_from_ll",
                wraps=eigen_refresh.eigen_table_from_ll,
            ) as builder:
                move.refresh_inner_move_tables(0, _work())
        # the build ran (the sidecar was ignored) ...
        self.assertEqual(builder.call_count, 1)
        # ... the counter started from scratch, as it does today ...
        self.assertEqual(move._eigen_visit_count[0], 1)
        # ... and nothing was written
        self.assertEqual(open(self.sidecar, "rb").read(), before)

    def test_persist_off_writes_no_file_at_all(self):
        move = _stub_move([EigenAxisMove()], self.store, refresh=10)
        with mock.patch.dict(os.environ, {"EIGEN_TABLES_PERSIST": "0"}):
            move.refresh_inner_move_tables(0, _work())
        self.assertFalse(os.path.exists(self.sidecar))
        self.assertEqual(os.listdir(self._tmp.name), [])

    # -- test 8: write-through on a cadence refresh ------------------
    def test_a_cadence_rebuild_updates_the_sidecar_entry(self):
        inner = EigenAxisMove()
        move = _stub_move([inner], self.store, refresh=3)
        work = _work()

        tables = [
            (np.eye(3), np.full(3, 0.1)),
            (np.eye(3)[:, ::-1].copy(), np.full(3, 0.2)),
        ]
        with mock.patch.object(
            eigen_refresh, "eigen_table_from_ll", side_effect=tables
        ):
            for _ in range(4):  # visits 0..3 at cadence 3 -> builds 0 and 3
                move.refresh_inner_move_tables(0, work)

        got = eigen_table_persist.load_entry(
            self.sidecar, "sobbh", 0, scope="walker_max", ndim=3,
            ntemps=2, nwalkers=4,
        )
        self.assertIsNotNone(got)
        axes, sigmas, visits = got
        np.testing.assert_array_equal(axes, tables[1][0])
        np.testing.assert_array_equal(sigmas, tables[1][1])
        self.assertEqual(visits, 4)

    def test_only_builds_write_not_every_visit(self):
        move = _stub_move([EigenAxisMove()], self.store, refresh=10)
        work = _work()
        move.refresh_inner_move_tables(0, work)
        with mock.patch.object(
            eigen_table_persist, "save_entry",
            wraps=eigen_table_persist.save_entry,
        ) as spy:
            for _ in range(5):
                move.refresh_inner_move_tables(0, work)
        spy.assert_not_called()

    def test_each_leaf_gets_its_own_entry(self):
        move = _stub_move([EigenAxisMove()], self.store, refresh=10)
        work = _work(nl=3)
        move.refresh_inner_move_tables(0, work)
        move.refresh_inner_move_tables(2, work)
        payload = eigen_table_persist.load_sidecar(self.sidecar)
        self.assertEqual(
            sorted(payload["entries"]), ["sobbh:0", "sobbh:2"]
        )

    def test_the_stamped_store_path_wins_over_the_midit_singleton(self):
        from lisatools.globalfit.moves import addremovemove

        move = _stub_move([EigenAxisMove()], self.store)
        with mock.patch.object(
            addremovemove.midit_checkpoint, "main_store_path",
            return_value="/elsewhere/other_main.h5",
        ):
            self.assertEqual(move._eigen_sidecar_path(), self.sidecar)

    def test_an_unstamped_move_falls_back_to_the_armed_midit_store(self):
        from lisatools.globalfit.moves import addremovemove

        move = _stub_move([EigenAxisMove()], None)
        with mock.patch.object(
            addremovemove.midit_checkpoint, "main_store_path",
            return_value=self.store,
        ):
            self.assertEqual(move._eigen_sidecar_path(), self.sidecar)
            move.refresh_inner_move_tables(0, _work())
        self.assertTrue(os.path.exists(self.sidecar))

    def test_an_unarmed_unstamped_move_has_no_sidecar(self):
        from lisatools.globalfit.moves import addremovemove

        move = _stub_move([EigenAxisMove()], None)
        with mock.patch.object(
            addremovemove.midit_checkpoint, "main_store_path",
            return_value=None,
        ):
            self.assertIsNone(move._eigen_sidecar_path())

    def test_data_identity_is_none_when_the_data_is_unreachable(self):
        move = _stub_move([EigenAxisMove()], self.store)
        self.assertIsNone(move._eigen_data_identity())

    def test_a_stretch_only_stack_touches_nothing(self):
        move = _stub_move([StretchMove()], self.store, refresh=10)
        move.refresh_inner_move_tables(0, _work())
        self.assertFalse(os.path.exists(self.sidecar))


class PersistEnabledTest(unittest.TestCase):
    def test_default_is_on(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EIGEN_TABLES_PERSIST", None)
            self.assertTrue(eigen_table_persist.persist_enabled())

    def test_zero_turns_it_off(self):
        with mock.patch.dict(os.environ, {"EIGEN_TABLES_PERSIST": "0"}):
            self.assertFalse(eigen_table_persist.persist_enabled())

    def test_read_at_use_not_at_import(self):
        with mock.patch.dict(os.environ, {"EIGEN_TABLES_PERSIST": "0"}):
            self.assertFalse(eigen_table_persist.persist_enabled())
        with mock.patch.dict(os.environ, {"EIGEN_TABLES_PERSIST": "1"}):
            self.assertTrue(eigen_table_persist.persist_enabled())


if __name__ == "__main__":
    unittest.main()
