"""Source warm start: MBH / EMRI / SOBBH leaves from a previous run's cold chain.

User request 2026-10-08 (9mo / 1yr production): start every source leaf whose
seed-run (6mo) optimal SNR is over 10 at that run's final cold-chain
positions; every other leaf starts exactly as before (``*_START_FACTOR``).

Every test builds a SMALL synthetic seed store through the real
``GFHDFBackend`` + per-leaf-ladder sub-backends, so the on-disk schema
(``global_fit/sub_backend/<branch>/{chain, inds, h_h}``, the main group's
``iteration`` attr, the ``noise_model_identity`` group) is the one the
production saver writes. No data, no GPU, no waveform.
"""

from __future__ import annotations

import copy
import inspect
import logging
import os
import pickle
import shutil
import tempfile
import types
import unittest
from unittest import mock

import h5py
import numpy as np

from lisatools.globalfit.hdfbackend import (
    EMRIHDFBackend,
    GFHDFBackend,
    MBHHDFBackend,
    SOBBHHDFBackend,
)
from lisatools.globalfit.state import EMRIState, GFState, MBHState, SOBBHState
from lisatools.globalfit.warmstart import sources as sws
from lisatools.globalfit.warmstart.sources import (
    SourceWarmStartError,
    parse_id_list,
    place_warm_leaves,
    read_seed_leaves,
    warm_start_branch,
)

BACKENDS = {"mbh": MBHHDFBackend, "emri": EMRIHDFBackend, "sobbh": SOBBHHDFBackend}
STATES = {"mbh": MBHState, "emri": EMRIState, "sobbh": SOBBHState}
COUNT_ATTR = {"mbh": "num_mbhs", "emri": "num_emris", "sobbh": "num_sobbhs"}

NT_SEED = 8
NW_SEED = 2
NDIM = 3
SEED_IDS = (2, 5, 16, 18)          # the 6mo MBHB_IDS
NEW_IDS = (2, 5, 7, 12, 16, 18)    # the 9mo MBHB_IDS (7, 12 merge after 180 d)
# seed SNR per stored leaf (ids 2, 5, 16, 18): warm / BELOW / warm / warm
SEED_SNR = np.array([10.5, 9.5, 400.0, 30.0])
DATA_T0 = 1.5e9


def seed_value(row, t, w, leaf, d):
    """Distinctive, decodable coordinate so a mis-indexed copy is visible."""
    return 1.0 + 1e4 * row + 1e3 * t + 1e2 * w + 10.0 * leaf + d


def seed_block(row, nt, nw, nl, nd):
    t = np.arange(nt)[:, None, None, None]
    w = np.arange(nw)[None, :, None, None]
    lf = np.arange(nl)[None, None, :, None]
    d = np.arange(nd)[None, None, None, :]
    return seed_value(row, t, w, lf, d) * np.ones((nt, nw, nl, nd))


def hh_for(snr, nw):
    """(nw, nl) <h|h> whose per-leaf MEDIAN over walkers gives ``snr``."""
    snr = np.asarray(snr, dtype=float)
    offs = np.linspace(-0.1, 0.1, nw)[:, None] if nw > 1 else np.zeros((1, 1))
    return (snr[None, :] + offs) ** 2


def write_seed_store(path, spec, nrows=2, data_t0=DATA_T0):
    """Write ``nrows`` saved rows through the production backend classes.

    ``spec``: ``{branch: dict(nleaves, ntemps, h_h=(nw, nl) LAST-row record)}``.
    Earlier rows carry ``h_h = 1`` so a reader of the wrong row is caught.
    """
    names = list(spec)
    backend = GFHDFBackend(
        path,
        sub_backend={b: BACKENDS[b] for b in names},
        sub_state_bases={b: STATES[b] for b in names},
    )
    backend.reset(
        NW_SEED,
        {b: NDIM for b in names},
        nleaves_max={b: spec[b]["nleaves"] for b in names},
        ntemps=1,
        branch_names=names,
        nbranches=len(names),
        rj=False,
        moves=None,
        sub_reset_kwargs={
            b: {
                "nleaves_max": spec[b]["nleaves"],
                "ndim": NDIM,
                "ntemps": spec[b]["ntemps"],
                COUNT_ATTR[b]: spec[b]["nleaves"],
            }
            for b in names
        },
    )
    backend.grow(nrows, None)
    for row in range(nrows):
        full = {
            b: seed_block(row, spec[b]["ntemps"], NW_SEED, spec[b]["nleaves"], NDIM)
            for b in names
        }
        state = GFState(
            {b: full[b][:1].copy() for b in names},
            inds={b: np.ones((1, NW_SEED, spec[b]["nleaves"]), dtype=bool) for b in names},
            log_like=np.zeros((1, NW_SEED)),
            log_prior=np.zeros((1, NW_SEED)),
            betas=np.ones(1),
            random_state=np.random.get_state(),
            sub_state_bases={b: STATES[b] for b in names},
        )
        for b in names:
            nt, nl = spec[b]["ntemps"], spec[b]["nleaves"]
            sub = state.sub_states[b]
            sub.betas_all = np.tile(1.0 / 1.2 ** np.arange(nt), (nl, 1))
            sub.initialize_tempered(nt, NW_SEED, nl, NDIM, coords=full[b])
            last = row == nrows - 1
            sub.h_h[:] = spec[b]["h_h"] if last else 1.0
            sub.d_h[:] = sub.h_h
        backend.save_step(state, np.ones((1, NW_SEED)))
    if data_t0 is not None:
        backend.write_noise_model_identity({"data_t0": float(data_t0)})
    return path


def wide_prior(ndim=NDIM, names=("a", "b", "t_plunge")):
    from eryn.prior import ProbDistContainer, uniform_dist

    return ProbDistContainer({names[i]: uniform_dist(-1e7, 1e7) for i in range(ndim)})


def truth_block(nt, nw, nl, nd=NDIM):
    """What today's START_FACTOR=0 path hands the hook: the injection, every rung."""
    inj = -(1.0 + np.arange(nl)[:, None] + 0.1 * np.arange(nd)[None, :])
    return np.broadcast_to(inj, (nt, nw, nl, nd)).copy()


class _StoreCase(unittest.TestCase):
    """A temp dir with a 4-leaf MBH seed store (ids 2, 5, 16, 18; 8 rungs, 2 walkers)."""

    spec_ntemps = NT_SEED

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = os.path.join(self.tmp.name, "seed_6mo.h5")
        write_seed_store(
            self.store,
            {"mbh": dict(nleaves=4, ntemps=self.spec_ntemps, h_h=hh_for(SEED_SNR, NW_SEED))},
        )

    def tearDown(self):
        self.tmp.cleanup()

    def place(self, coords=None, *, new_ids=NEW_IDS, seed_ids=SEED_IDS, ntemps_branch=NT_SEED,
              snr_min=10.0, snr_unknown="refuse", prior="wide", nw=NW_SEED, store=None):
        if coords is None:
            coords = truth_block(ntemps_branch, nw, len(new_ids))
        return warm_start_branch(
            "mbh",
            coords,
            store=store or self.store,
            seed_ids=seed_ids,
            new_ids=new_ids,
            snr_min=snr_min,
            snr_unknown=snr_unknown,
            ntemps_branch=ntemps_branch,
            prior=wide_prior() if prior == "wide" else prior,
            data_t0=DATA_T0,
        )


# ---------------------------------------------------------------------------
# reading the seed store
# ---------------------------------------------------------------------------
class SeedStoreReadTest(_StoreCase):
    def test_reads_the_last_written_row(self):
        seed = read_seed_leaves(self.store, "mbh", SEED_IDS)
        self.assertEqual(seed.row, 1)
        self.assertEqual(seed.ids, SEED_IDS)
        self.assertEqual(seed.coords.shape, (NT_SEED, NW_SEED, 4, NDIM))
        np.testing.assert_array_equal(seed.coords, seed_block(1, NT_SEED, NW_SEED, 4, NDIM))
        np.testing.assert_allclose(seed.snr(), SEED_SNR)
        self.assertEqual(seed.data_t0, DATA_T0)

    def test_on_disk_schema_is_what_the_reader_assumes(self):
        """Pin the facts the reader relies on: d_h/h_h carry NO rung axis."""
        with h5py.File(self.store, "r") as f:
            g = f["global_fit"]
            sub = g["sub_backend"]["mbh"]
            self.assertEqual(int(g.attrs["iteration"]), 2)
            self.assertEqual(sub["chain"].shape[1:], (NT_SEED, NW_SEED, 4, NDIM))
            self.assertEqual(sub["inds"].shape[1:], (NT_SEED, NW_SEED, 4))
            self.assertEqual(sub["h_h"].shape[1:], (NW_SEED, 4))
            self.assertEqual(sub["d_h"].shape[1:], (NW_SEED, 4))
            self.assertEqual(sub["betas_all"].shape[1:], (4, NT_SEED))

    def test_ids_are_sorted_to_the_stored_leaf_order(self):
        """The stored leaf order is the SORTED id list: a reordered knob maps the same."""
        a = read_seed_leaves(self.store, "mbh", (18, 2, 16, 5))
        self.assertEqual(a.ids, (2, 5, 16, 18))
        out_sorted, _ = self.place(seed_ids=(2, 5, 16, 18))
        out_shuffled, _ = self.place(seed_ids=(18, 2, 16, 5))
        np.testing.assert_array_equal(out_sorted, out_shuffled)

    def test_id_count_must_match_the_stored_leaves(self):
        with self.assertRaisesRegex(SourceWarmStartError, "4 leaves"):
            read_seed_leaves(self.store, "mbh", (2, 5, 16))

    def test_absent_branch_returns_none(self):
        self.assertIsNone(read_seed_leaves(self.store, "sobbh", (0, 1)))

    def test_unwritten_trailing_row_falls_back_to_the_previous_row(self):
        """The saver advances ``iteration`` BEFORE the sub-backend rows land:
        a reader in between sees a zero-filled row (inds all False)."""
        with h5py.File(self.store, "a") as f:
            g = f["global_fit"]
            for key in g["sub_backend"]["mbh"]:
                ds = g["sub_backend"]["mbh"][key]
                ds.resize(3, axis=0)
            g.attrs["iteration"] = 3
        seed = read_seed_leaves(self.store, "mbh", SEED_IDS)
        self.assertEqual(seed.row, 1)
        np.testing.assert_allclose(seed.snr(), SEED_SNR)

    def test_unreadable_primary_falls_back_to_the_running_backup(self):
        backup = self.store[:-3] + "_running_backup_copy.h5"
        shutil.copyfile(self.store, backup)
        with open(self.store, "r+b") as fh:          # tear the primary
            fh.truncate(2048)
        seed = read_seed_leaves(self.store, "mbh", SEED_IDS)
        self.assertEqual(seed.store, backup)
        np.testing.assert_allclose(seed.snr(), SEED_SNR)

    def test_unreadable_primary_without_backup_refuses(self):
        with open(self.store, "r+b") as fh:
            fh.truncate(2048)
        with self.assertRaisesRegex(SourceWarmStartError, "unreadable"):
            read_seed_leaves(self.store, "mbh", SEED_IDS)

    def test_dry_run_cli_prints_the_decision_per_id(self):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = sws.main(["--store", self.store, "--branch", "mbh", "--ids", "18,16,5,2"])
        text = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("row 1, 8 rungs x 2 walkers", text)
        self.assertRegex(text, r"id +5: SNR median +9\.50 .*-> injection")
        self.assertRegex(text, r"id +16: SNR median +400\.00 .*-> warm")

    def test_snr_is_unknown_when_any_walker_record_is_nan(self):
        with h5py.File(self.store, "a") as f:
            hh = f["global_fit/sub_backend/mbh/h_h"]
            row = hh[1]
            row[0, 3] = np.nan                       # walker 0, id 18
            hh[1] = row
        snr = read_seed_leaves(self.store, "mbh", SEED_IDS).snr()
        np.testing.assert_allclose(snr[:3], SEED_SNR[:3])
        self.assertTrue(np.isnan(snr[3]))


# ---------------------------------------------------------------------------
# mapping, threshold, rungs, walkers
# ---------------------------------------------------------------------------
class MappingAndThresholdTest(_StoreCase):
    def test_maps_by_catalogue_id_never_by_leaf_index(self):
        truth = truth_block(NT_SEED, NW_SEED, len(NEW_IDS))
        out, _ = self.place(truth.copy())
        seed = seed_block(1, NT_SEED, NW_SEED, 4, NDIM)
        # new leaf 0 = id 2 = seed leaf 0 (SNR 10.5): warm
        np.testing.assert_array_equal(out[:, :, 0], seed[:, :, 0])
        # new leaf 4 = id 16 = seed leaf 2; new leaf 5 = id 18 = seed leaf 3
        np.testing.assert_array_equal(out[:, :, 4], seed[:, :, 2])
        np.testing.assert_array_equal(out[:, :, 5], seed[:, :, 3])
        # ids 7 and 12 are only in the new run: untouched (the injection start)
        np.testing.assert_array_equal(out[:, :, 2], truth[:, :, 2])
        np.testing.assert_array_equal(out[:, :, 3], truth[:, :, 3])

    def test_threshold_is_strict_and_per_source(self):
        truth = truth_block(NT_SEED, NW_SEED, len(NEW_IDS))
        out, line = self.place(truth.copy())
        # id 5: seed SNR 9.5 <= 10 -> the injection start, byte-identical
        np.testing.assert_array_equal(out[:, :, 1], truth[:, :, 1])
        self.assertIn("5 at the injection (seed SNR 9.5 <= 10)", line)
        # raising the bar above 10.5 sends id 2 back to the injection start too
        out2, _ = self.place(truth.copy(), snr_min=10.5)
        np.testing.assert_array_equal(out2[:, :, 0], truth[:, :, 0])

    def test_one_summary_line(self):
        _, line = self.place()
        self.assertTrue(line.startswith("[SOURCE-WARM] mbh: ids 2,16,18 from the seed cold chain"), line)
        self.assertIn("SNR 10.5/400.0/30.0", line)
        self.assertIn("7,12 at the injection (not in the seed store)", line)
        self.assertIn("seed_6mo.h5 row 1", line)
        self.assertNotIn("\n", line)

    def test_nan_snr_refuses_and_names_branch_id_and_knob(self):
        with h5py.File(self.store, "a") as f:
            hh = f["global_fit/sub_backend/mbh/h_h"]
            row = hh[1]
            row[:, 3] = np.nan                       # id 18
            hh[1] = row
        with self.assertRaises(SourceWarmStartError) as ctx:
            self.place()
        msg = str(ctx.exception)
        for needle in ("mbh", "18", "SOURCE_WARM_START_SNR_UNKNOWN", "MBH_RECORD_DH"):
            self.assertIn(needle, msg)
        truth = truth_block(NT_SEED, NW_SEED, len(NEW_IDS))
        out_t, line_t = self.place(truth.copy(), snr_unknown="truth")
        np.testing.assert_array_equal(out_t[:, :, 5], truth[:, :, 5])
        self.assertIn("18 at the injection (seed SNR unknown", line_t)
        out_w, _ = self.place(truth.copy(), snr_unknown="warm")
        np.testing.assert_array_equal(out_w[:, :, 5], seed_block(1, NT_SEED, NW_SEED, 4, NDIM)[:, :, 3])

    def test_nan_snr_on_a_leaf_this_run_does_not_arm_is_ignored(self):
        """Only leaves this run arms AND the store holds are decided."""
        with h5py.File(self.store, "a") as f:
            hh = f["global_fit/sub_backend/mbh/h_h"]
            row = hh[1]
            row[:, 1] = np.nan                       # id 5
            row[:, 3] = np.nan                       # id 18
            hh[1] = row
        _, line = self.place(new_ids=(7, 12, 16), coords=truth_block(NT_SEED, NW_SEED, 3))
        self.assertIn("ids 16 from the seed cold chain", line)

    def test_unknown_snr_choice_is_validated(self):
        with self.assertRaisesRegex(ValueError, "SOURCE_WARM_START_SNR_UNKNOWN"):
            self.place(snr_unknown="maybe")


class RungAndWalkerTest(_StoreCase):
    def test_full_ladder_copied_when_rung_counts_match(self):
        out, line = self.place(ntemps_branch=NT_SEED)
        seed = seed_block(1, NT_SEED, NW_SEED, 4, NDIM)
        for t in range(NT_SEED):
            np.testing.assert_array_equal(out[t, :, 4], seed[t, :, 2])
        self.assertIn("ladder copied 8/8 rungs", line)

    def test_cold_on_every_rung_when_rung_counts_differ(self):
        out, line = self.place(ntemps_branch=4)
        seed = seed_block(1, NT_SEED, NW_SEED, 4, NDIM)
        for t in range(4):
            np.testing.assert_array_equal(out[t, :, 4], seed[0, :, 2])
        self.assertIn("cold positions on every rung (seed 8 rungs, this run 4)", line)

    def test_draw_rungs_beyond_the_branch_ladder_get_the_cold_point(self):
        coords = truth_block(10, NW_SEED, len(NEW_IDS))      # nt_draw 10 > branch 8
        out, _ = self.place(coords, ntemps_branch=NT_SEED)
        seed = seed_block(1, NT_SEED, NW_SEED, 4, NDIM)
        np.testing.assert_array_equal(out[:8, :, 4], seed[:, :, 2])
        np.testing.assert_array_equal(out[8:, :, 4], np.broadcast_to(seed[0, :, 2], (2, NW_SEED, NDIM)))

    def test_walker_w_maps_to_walker_w(self):
        out, line = self.place()
        seed = seed_block(1, NT_SEED, NW_SEED, 4, NDIM)
        np.testing.assert_array_equal(out[0, 1, 5], seed[0, 1, 3])
        self.assertIn("walker w->w", line)

    def test_more_walkers_cycle_through_the_seed_walkers_with_a_warning(self):
        with self.assertLogs("lisatools.globalfit.warmstart.sources", logging.WARNING) as logs:
            out, line = self.place(nw=4)
        seed = seed_block(1, NT_SEED, NW_SEED, 4, NDIM)
        for w in range(4):
            np.testing.assert_array_equal(out[:, w, 4], seed[:, w % 2, 2])
        self.assertIn("cycling", " ".join(logs.output))
        self.assertIn("walker w->w%2", line)

    def test_no_random_numbers_are_consumed(self):
        state = np.random.get_state()
        self.place()
        after = np.random.get_state()
        self.assertTrue(np.array_equal(state[1], after[1]) and state[2] == after[2])


# ---------------------------------------------------------------------------
# refusals at build
# ---------------------------------------------------------------------------
class RefusalTest(_StoreCase):
    def _move_seed_leaf_out_of_box(self, leaf, value=5e7):
        """Put stored leaf ``leaf``'s t_plunge column outside the wide prior."""
        with h5py.File(self.store, "a") as f:
            ch = f["global_fit/sub_backend/mbh/chain"]
            row = ch[1]
            row[:, :, leaf, 2] = value
            ch[1] = row

    def test_a_warm_start_outside_this_runs_prior_refuses(self):
        self._move_seed_leaf_out_of_box(2)           # id 16 (SNR 400: warm)
        with self.assertRaises(SourceWarmStartError) as ctx:
            self.place()
        msg = str(ctx.exception)
        for needle in ("mbh", "id 16", "t_plunge", "prior", "5e+07"):
            self.assertIn(needle, msg)

    def test_a_below_threshold_leaf_outside_the_prior_is_not_checked(self):
        """Paired control: the same out-of-box seed point on a leaf that is
        NOT warm-started (id 5, SNR 9.5) passes -- and refuses once warm."""
        self._move_seed_leaf_out_of_box(1)           # id 5 (SNR 9.5: not warm)
        truth = truth_block(NT_SEED, NW_SEED, len(NEW_IDS))
        out, _ = self.place(truth.copy())
        np.testing.assert_array_equal(out[:, :, 1], truth[:, :, 1])
        with self.assertRaisesRegex(SourceWarmStartError, "id 5"):
            self.place(truth.copy(), snr_min=9.0)

    def test_a_different_data_start_refuses(self):
        with self.assertRaisesRegex(SourceWarmStartError, "data start"):
            warm_start_branch(
                "mbh", truth_block(NT_SEED, NW_SEED, len(NEW_IDS)), store=self.store,
                seed_ids=SEED_IDS, new_ids=NEW_IDS, snr_min=10.0,
                ntemps_branch=NT_SEED, prior=wide_prior(), data_t0=DATA_T0 + 86400.0,
            )

    def test_missing_seed_id_list_refuses_and_names_the_knob(self):
        with self.assertRaisesRegex(SourceWarmStartError, "MBH_SOURCE_WARM_START_IDS"):
            self.place(seed_ids=None)

    def test_none_disables_the_branch(self):
        truth = truth_block(NT_SEED, NW_SEED, len(NEW_IDS))
        out, line = self.place(truth.copy(), seed_ids=())
        np.testing.assert_array_equal(out, truth)
        self.assertIn("disabled", line)

    def test_unknown_new_run_ids_refuse(self):
        with self.assertRaisesRegex(SourceWarmStartError, "injection_ids"):
            self.place(new_ids=None, coords=truth_block(NT_SEED, NW_SEED, 6))

    def test_branch_absent_from_the_store_starts_at_the_injection(self):
        truth = truth_block(NT_SEED, NW_SEED, 3)
        out, line = warm_start_branch(
            "sobbh", truth.copy(), store=self.store, seed_ids=(0, 1, 2), new_ids=(0, 1, 2),
            snr_min=10.0, ntemps_branch=NT_SEED, prior=wide_prior(), data_t0=DATA_T0,
        )
        np.testing.assert_array_equal(out, truth)
        self.assertIn("not in the seed store", line)

    def test_missing_store_file_refuses(self):
        with self.assertRaisesRegex(SourceWarmStartError, "does not exist"):
            self.place(store=os.path.join(self.tmp.name, "nope.h5"))


# ---------------------------------------------------------------------------
# knobs on the source settings blocks
# ---------------------------------------------------------------------------
class SettingsKnobTest(unittest.TestCase):
    ENV = (
        "SOURCE_WARM_START_STORE", "SOURCE_WARM_START_SNR_MIN", "SOURCE_WARM_START_SNR_UNKNOWN",
        "MBH_SOURCE_WARM_START_IDS", "EMRI_SOURCE_WARM_START_IDS", "SOBBH_SOURCE_WARM_START_IDS",
    )

    def _clean(self):
        return mock.patch.dict(os.environ, {k: "" for k in self.ENV})

    def test_parse_id_list(self):
        self.assertEqual(parse_id_list("2,5, 16,18"), (2, 5, 16, 18))
        self.assertEqual(parse_id_list("none"), ())
        self.assertEqual(parse_id_list("off"), ())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_id_list("2,2")

    def test_defaults_are_off(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        with self._clean():
            for cls in (sr.SourceMBHSettings, sr.SourceEMRISettings, sr.SourceSOBBHSettings):
                s = cls()
                self.assertIsNone(s.source_warm_start_store, cls.__name__)
                self.assertEqual(s.source_warm_start_snr_min, 10.0)
                self.assertEqual(s.source_warm_start_snr_unknown, "refuse")
                self.assertIsNone(s.source_warm_start_ids)
                self.assertIsNone(s.injection_ids)

    def test_env_knobs_reach_the_attributes(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        env = {
            "SOURCE_WARM_START_STORE": "/x/seed.h5",
            "SOURCE_WARM_START_SNR_MIN": "12.5",
            "SOURCE_WARM_START_SNR_UNKNOWN": "truth",
            "MBH_SOURCE_WARM_START_IDS": "2,5,16,18",
            "EMRI_SOURCE_WARM_START_IDS": "0,1,2,3,4,5,6,7",
            "SOBBH_SOURCE_WARM_START_IDS": "none",
        }
        with self._clean(), mock.patch.dict(os.environ, env):
            m, e, s = sr.SourceMBHSettings(), sr.SourceEMRISettings(), sr.SourceSOBBHSettings()
        for blk in (m, e, s):
            self.assertEqual(blk.source_warm_start_store, "/x/seed.h5")
            self.assertEqual(blk.source_warm_start_snr_min, 12.5)
            self.assertEqual(blk.source_warm_start_snr_unknown, "truth")
        self.assertEqual(m.source_warm_start_ids, (2, 5, 16, 18))
        self.assertEqual(e.source_warm_start_ids, tuple(range(8)))
        self.assertEqual(s.source_warm_start_ids, ())
        # settings-tree rule: survives deepcopy + pickle
        back = pickle.loads(pickle.dumps(copy.deepcopy(m)))
        self.assertEqual(back.source_warm_start_ids, (2, 5, 16, 18))


# ---------------------------------------------------------------------------
# this run's leaf ids, recorded by prepare_*_branch
# ---------------------------------------------------------------------------
class InjectionIdsTest(unittest.TestCase):
    DAY = 86400.0
    DATA_T0 = 1000.0 * 86400.0
    WF_T0 = 990.0 * 86400.0
    TOBS = 120.0 * 86400.0

    def test_mbh_ids_follow_the_merger_window_filter(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        with mock.patch.dict(os.environ, {"MBH_MERGER_TIME_BUFFER": ""}):
            mbh = sr.SourceMBHSettings()
        mbh.initialize_kwargs = {}
        mbh.inner_moves = []
        t_rel0 = self.DATA_T0 - self.WF_T0
        # catalogue ids 9, 2, 16, 4: id 4 merges 30 d past the end -> dropped
        days = {9: 3.0, 2: 50.0, 16: 110.0, 4: 150.0}
        rows = {i: np.r_[np.zeros(10), t_rel0 + d * self.DAY] for i, d in days.items()}
        gsetup = types.SimpleNamespace(data_t0=self.DATA_T0, Tobs=self.TOBS)
        gs = types.SimpleNamespace(
            n_injections={"MBHB": len(rows)}, data_mode="mojito", mbh_waveform_t0=self.WF_T0,
        )
        with mock.patch.object(sr, "source_catalogue", return_value=rows), \
                mock.patch.object(sr, "mbh_catalogue_to_sampling_basis", side_effect=lambda r: r):
            out = sr.prepare_mbh_branch(mbh, gsetup, gs)
        self.assertEqual(out.injection_ids, (2, 9, 16))
        # the ids name the injection rows, in leaf order
        np.testing.assert_allclose(
            (np.asarray(out.injection)[:, -1] - t_rel0) / self.DAY, [50.0, 3.0, 110.0]
        )

    def test_emri_ids_are_the_sorted_catalogue_keys(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        emri = sr.SourceEMRISettings()
        emri.initialize_kwargs = {}
        emri.inner_moves = []
        rows = {k: np.full(14, float(k)) for k in (4, 0, 7)}
        gs = types.SimpleNamespace(n_injections={"EMRI": 3}, data_mode="mojito")
        tc = types.SimpleNamespace(both_inverse_transforms=lambda x: x)
        with mock.patch.object(sr, "source_catalogue", return_value=rows), \
                mock.patch.object(sr, "emri_catalogue_to_waveform_basis", side_effect=lambda r: r), \
                mock.patch.object(sr, "make_emri_transform_container", return_value=tc):
            out = sr.prepare_emri_branch(emri, types.SimpleNamespace(force_backend="cpu"), gs)
        self.assertEqual(out.injection_ids, (0, 4, 7))
        np.testing.assert_allclose(np.asarray(out.injection)[:, 0], [0.0, 4.0, 7.0])

    def test_sobbh_ids_are_the_sorted_catalogue_keys(self):
        from lisatools.globalfit.stock.erebor import injections as inj
        from lisatools.globalfit.stock.erebor import source_runtime as sr
        from lisatools.globalfit.stock.erebor import transforms as trf

        sobbh = sr.SourceSOBBHSettings()
        sobbh.initialize_kwargs = {}
        sobbh.inner_moves = []
        sobbh.chirp_fdot_max = 0.0
        rows = {k: np.full(11, float(k)) for k in (5, 1)}
        gs = types.SimpleNamespace(n_injections={"SOBHB": 2}, data_mode="mojito")
        tc = types.SimpleNamespace(both_inverse_transforms=lambda x: x)
        with mock.patch.object(sr, "source_catalogue", return_value=rows), \
                mock.patch.object(inj, "sobbh_catalogue_to_waveform_basis", side_effect=lambda r: r), \
                mock.patch.object(trf, "make_sobbh_transform_container", return_value=tc):
            out = sr.prepare_sobbh_branch(sobbh, types.SimpleNamespace(force_backend="cpu"), gs)
        self.assertEqual(out.injection_ids, (1, 5))
        np.testing.assert_allclose(np.asarray(out.injection)[:, 0], [1.0, 5.0])


# ---------------------------------------------------------------------------
# run.py glue: off switch + the fresh-start hook
# ---------------------------------------------------------------------------
class RunGlueTest(_StoreCase):
    def _fit(self, **info):
        from lisatools.globalfit.run import GlobalFit

        obj = GlobalFit.__new__(GlobalFit)
        self.lines = []
        obj.logger = types.SimpleNamespace(
            info=lambda msg, *a: self.lines.append(msg % a if a else msg),
            warning=lambda *a, **k: None,
        )
        obj.curr = types.SimpleNamespace(
            source_info={"mbh": types.SimpleNamespace(**info)},
            general_info=types.SimpleNamespace(data_t0=DATA_T0),
        )
        obj._branch_ntemps = lambda name: NT_SEED
        return obj

    def test_off_switch_returns_the_same_array_untouched(self):
        obj = self._fit(source_warm_start_store=None)
        coords = truth_block(NT_SEED, NW_SEED, len(NEW_IDS))
        before = coords.copy()
        with mock.patch.object(sws, "read_seed_leaves") as reader:
            out = obj._source_warm_start("mbh", coords, wide_prior())
        self.assertIs(out, coords)
        np.testing.assert_array_equal(coords, before)
        reader.assert_not_called()
        self.assertEqual(self.lines, [])

    def test_armed_hook_places_warm_leaves_and_logs_one_line(self):
        obj = self._fit(
            source_warm_start_store=self.store, source_warm_start_ids=SEED_IDS,
            injection_ids=NEW_IDS, source_warm_start_snr_min=10.0,
            source_warm_start_snr_unknown="refuse",
        )
        out = obj._source_warm_start("mbh", truth_block(NT_SEED, NW_SEED, 6), wide_prior())
        np.testing.assert_array_equal(out[:, :, 4], seed_block(1, NT_SEED, NW_SEED, 4, NDIM)[:, :, 2])
        self.assertEqual(len(self.lines), 1)
        self.assertTrue(self.lines[0].startswith("[SOURCE-WARM] mbh:"))

    def test_load_info_calls_the_hook_before_the_cold_chain_slice(self):
        from lisatools.globalfit.run import GlobalFit

        src = inspect.getsource(GlobalFit.load_info)
        hook = src.find("self._source_warm_start(")
        cut = src.find("coords_full, inds_full = coords, inds")
        self.assertGreater(hook, 0, "load_info no longer calls _source_warm_start")
        self.assertLess(hook, cut, "the warm start must run before the per-branch slicing")
        self.assertIn('("mbh", "emri", "sobbh")', src[src.rfind("for", 0, hook):hook])


if __name__ == "__main__":
    unittest.main()
