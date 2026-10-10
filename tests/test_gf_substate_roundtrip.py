"""Round-trip tests for the global-fit sub-state / sub-backend storage layer.

Phase-2 form (cold-chain storage rework): every sub-state owns the module's
full tempered ensemble (``chain``/``inds`` + per-branch log_like/log_prior +
delta counters) alongside its module extras (GB band_info, per-leaf
``betas_all``). These tests pin the on-disk schema (dataset names, shapes,
attrs), the save/load round-trip, the delta-counter semantics, the GFState
copy path, and the cold-row sync/check primitives.
"""

import os
import tempfile
import unittest

import h5py
import numpy as np

from lisatools.globalfit.hdfbackend import (
    EMRIHDFBackend,
    GBHDFBackend,
    GFHDFBackend,
    MBHHDFBackend,
    ModuleSubBackend,
    SOBBHHDFBackend,
)
from lisatools.globalfit.recipe import release_band_shutoff_window
from lisatools.globalfit.state import (
    CAP_CELL_PER_WALKER_FIELDS,
    SEARCH_SHUTOFF_FIELDS,
    SEARCH_SHUTOFF_WINDOW_FIELDS,
    SEARCH_STAGE_FIELDS,
    EMRIState,
    GBState,
    GFState,
    MBHState,
    ModuleSubState,
    SOBBHState,
    ensure_search_shutoff_fields,
)

NTEMPS = 3
NWALKERS = 4
NUM_BANDS = 6
BAND_EDGES = np.linspace(1e-3, 7e-3, NUM_BANDS + 1)

BRANCH_SHAPES = {
    # branch: (nleaves_max, ndim)
    "gb": (5, 8),
    "mbh": (2, 11),
    "emri": (2, 12),
    "sobbh": (2, 11),
    "psd": (1, 4),
}

SUB_BACKENDS = {
    "gb": GBHDFBackend,
    "mbh": MBHHDFBackend,
    "emri": EMRIHDFBackend,
    "sobbh": SOBBHHDFBackend,
    "psd": ModuleSubBackend,
}

SUB_STATE_BASES = {
    "gb": GBState,
    "mbh": MBHState,
    "emri": EMRIState,
    "sobbh": SOBBHState,
    "psd": ModuleSubState,
}


def _tempered_schema(branch):
    """The standard tempered datasets for one branch (shapes after step axis)."""
    nleaves, ndim = BRANCH_SHAPES[branch]
    out = {
        "chain": (NTEMPS, NWALKERS, nleaves, ndim),
        "inds": (NTEMPS, NWALKERS, nleaves),
        # per-leaf cold-chain inner products (NaN = dead / not recorded)
        "d_h": (NWALKERS, nleaves),
        "h_h": (NWALKERS, nleaves),
    }
    if branch == "gb":
        # band_info carries the GB tempering record; no per-branch ll/counters
        return out
    if branch in ("mbh", "emri", "sobbh"):
        ll_shape = (nleaves, NTEMPS, NWALKERS)
        counter_shape = (nleaves, NTEMPS)
        swaps_shape = (nleaves, NTEMPS - 1)
    else:
        ll_shape = (NTEMPS, NWALKERS)
        counter_shape = (NTEMPS,)
        swaps_shape = (NTEMPS - 1,)
        # base branches carry the flat module ladder (GB uses band_temps,
        # per-leaf branches use betas_all)
        out["betas"] = (NTEMPS,)
    out.update(
        {
            "log_like": ll_shape,
            "log_prior": ll_shape,
            "in_model_proposed": counter_shape,
            "in_model_accepted": counter_shape,
            "rj_proposed": counter_shape,
            "rj_accepted": counter_shape,
            "swaps_proposed": swaps_shape,
            "swaps_accepted": swaps_shape,
        }
    )
    return out


# The exact per-branch datasets the sub-backends put on disk (shapes after
# the leading step axis).
EXPECTED_SUB_SCHEMA = {
    "gb": {
        "band_edges": (NUM_BANDS + 1,),
        # leaf-cap CELL grid (2026-08-15). At the default cap_divisor of 1
        # the cap grid IS the band grid, so cap_edges == band_edges and NO
        # cap_cell_* arrays are allocated (that short circuit is what keeps
        # divisor 1 bit-identical and pre-cap-grid stores resumable).
        "cap_edges": (NUM_BANDS + 1,),
        "band_temps": (NUM_BANDS, NTEMPS),
        "band_swaps_proposed": (NUM_BANDS, NTEMPS - 1),
        "band_swaps_accepted": (NUM_BANDS, NTEMPS - 1),
        "band_num_proposed": (NUM_BANDS, NTEMPS),
        "band_num_accepted": (NUM_BANDS, NTEMPS),
        "band_num_proposed_rj": (NUM_BANDS, NTEMPS),
        "band_num_accepted_rj": (NUM_BANDS, NTEMPS),
        "band_num_binaries": (NTEMPS, NWALKERS, NUM_BANDS),
        "band_leaf_cap": (NUM_BANDS,),
        "band_cap_iters": (NUM_BANDS,),
        "band_best_ll": (NUM_BANDS,),
        # per-cold-walker per-band ll, stored every step (leaf-cap audit)
        "band_cold_ll": (NWALKERS, NUM_BANDS),
        # RJ band-shutoff valve state (2026-08-29). Persisted so the
        # valve's clock counts GB proposes across the WHOLE run rather
        # than within one process -- gf_prod_3mo_v7 took 26 launches with
        # 2-8 iteration segments against a 5-tick clock, so in-memory
        # counters were wiped before they could fire. Rides this channel
        # with the band_leaf_cap family; no schema layer to update beyond
        # this pin. The two (1,) arrays are the revival counters.
        "band_occ_streak": (NUM_BANDS,),
        "band_occ_last": (NUM_BANDS,),
        "band_rj_shutoff": (NUM_BANDS,),
        "band_shutoff_since_revive": (1,),
        "band_shutoff_epoch": (1,),
        **_tempered_schema("gb"),
    },
    "mbh": {
        "betas_all": (BRANCH_SHAPES["mbh"][0], NTEMPS),
        **_tempered_schema("mbh"),
    },
    "emri": {
        "betas_all": (BRANCH_SHAPES["emri"][0], NTEMPS),
        **_tempered_schema("emri"),
    },
    "sobbh": {
        "betas_all": (BRANCH_SHAPES["sobbh"][0], NTEMPS),
        **_tempered_schema("sobbh"),
    },
    "psd": _tempered_schema("psd"),
}

# band_edges / cap_edges are written once with the data (no step axis);
# everything else is a growable per-iteration dataset.
STATIC_DATASETS = {"gb": {"band_edges", "cap_edges"}}


def make_state(rng):
    """Build a fully-populated GFState with all sub-states initialized."""
    coords = {
        name: rng.standard_normal((NTEMPS, NWALKERS, nleaves, ndim))
        for name, (nleaves, ndim) in BRANCH_SHAPES.items()
    }
    inds = {
        name: np.ones((NTEMPS, NWALKERS, nleaves), dtype=bool)
        for name, (nleaves, _) in BRANCH_SHAPES.items()
    }
    state = GFState(
        coords,
        inds=inds,
        log_like=rng.standard_normal((NTEMPS, NWALKERS)),
        log_prior=rng.standard_normal((NTEMPS, NWALKERS)),
        betas=np.linspace(1.0, 0.1, NTEMPS),
        random_state=np.random.get_state(),
        sub_state_bases=SUB_STATE_BASES,
    )

    band_temps = np.tile(np.linspace(1.0, 0.1, NTEMPS), (NUM_BANDS, 1))
    state.sub_states["gb"].initialize_band_information(
        NWALKERS, NTEMPS, BAND_EDGES, band_temps
    )
    for name in ("mbh", "emri", "sobbh"):
        nleaves = BRANCH_SHAPES[name][0]
        state.sub_states[name].betas_all = np.tile(
            np.linspace(1.0, 0.05, NTEMPS), (nleaves, 1)
        )
    # mirror the tempered ensembles into every sub-state
    for name, sub in state.sub_states.items():
        sub.pull_from_main(state, name)
    return state


class GFSubStateRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.fp = os.path.join(self.tmpdir.name, "roundtrip_test.h5")
        self.rng = np.random.default_rng(1234)

        self.backend = GFHDFBackend(
            self.fp,
            sub_backend=dict(SUB_BACKENDS),
            sub_state_bases=dict(SUB_STATE_BASES),
        )
        ndims = {name: shape[1] for name, shape in BRANCH_SHAPES.items()}
        nleaves_max = {name: shape[0] for name, shape in BRANCH_SHAPES.items()}
        sub_reset_kwargs = {
            name: dict(nleaves_max=shape[0], ndim=shape[1])
            for name, shape in BRANCH_SHAPES.items()
        }
        sub_reset_kwargs["gb"].update(num_bands=NUM_BANDS, band_edges=BAND_EDGES)
        sub_reset_kwargs["mbh"].update(num_mbhs=BRANCH_SHAPES["mbh"][0])
        sub_reset_kwargs["emri"].update(num_emris=BRANCH_SHAPES["emri"][0])
        sub_reset_kwargs["sobbh"].update(num_sobbhs=BRANCH_SHAPES["sobbh"][0])
        self.backend.reset(
            NWALKERS,
            ndims,
            nleaves_max=nleaves_max,
            ntemps=NTEMPS,
            branch_names=list(BRANCH_SHAPES.keys()),
            nbranches=len(BRANCH_SHAPES),
            rj=False,
            moves=None,
            sub_reset_kwargs=sub_reset_kwargs,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _save_one(self, state):
        accepted = np.ones((NTEMPS, NWALKERS))
        self.backend.save_step(state, accepted)

    def test_on_disk_schema(self):
        """Dataset names/shapes under sub_backend/<branch> match the pinned schema."""
        import h5py

        self.backend.grow(1, None)
        with h5py.File(self.fp, "r") as f:
            sub = f["global_fit"]["sub_backend"]
            self.assertEqual(set(sub.keys()), set(EXPECTED_SUB_SCHEMA.keys()))
            for branch, datasets in EXPECTED_SUB_SCHEMA.items():
                grp = sub[branch]
                self.assertEqual(
                    set(grp.keys()), set(datasets.keys()), f"branch {branch}"
                )
                for dset_name, bare_shape in datasets.items():
                    on_disk = grp[dset_name].shape
                    if dset_name in STATIC_DATASETS.get(branch, ()):
                        self.assertEqual(on_disk, bare_shape, f"{branch}/{dset_name}")
                    else:
                        # growable: leading step axis then the bare shape
                        self.assertEqual(
                            on_disk, (1,) + bare_shape, f"{branch}/{dset_name}"
                        )
                # inds native bool, chain float
                self.assertEqual(grp["inds"].dtype, np.dtype(bool), branch)
                self.assertEqual(grp["chain"].dtype, np.dtype(float), branch)
                # tempered geometry attrs
                nleaves, ndim = BRANCH_SHAPES[branch]
                self.assertEqual(grp.attrs["ntemps"], NTEMPS)
                self.assertEqual(grp.attrs["nwalkers"], NWALKERS)
                self.assertEqual(grp.attrs["nleaves_max"], nleaves)
                self.assertEqual(grp.attrs["ndim"], ndim)
            # legacy count attrs
            self.assertEqual(sub["gb"].attrs["num_bands"], NUM_BANDS)
            self.assertEqual(sub["mbh"].attrs["num_mbhs"], BRANCH_SHAPES["mbh"][0])
            self.assertEqual(sub["emri"].attrs["num_emris"], BRANCH_SHAPES["emri"][0])
            self.assertEqual(
                sub["sobbh"].attrs["num_sobbhs"], BRANCH_SHAPES["sobbh"][0]
            )

    def test_save_and_reload_roundtrip(self):
        """Two saved iterations read back with get_a_sample identical to what went in."""
        self.backend.grow(2, None)

        state0 = make_state(self.rng)
        state0.sub_states["gb"].band_info["band_num_accepted"][:] = 7
        state0.sub_states["psd"].in_model_accepted[:] = 3
        self._save_one(state0)
        # delta semantics: counters zeroed on the live state after the save
        self.assertTrue(
            np.all(state0.sub_states["gb"].band_info["band_num_accepted"] == 0)
        )
        self.assertTrue(np.all(state0.sub_states["psd"].in_model_accepted == 0))

        state1 = make_state(self.rng)
        state1.sub_states["mbh"].betas_all *= 0.5
        self._save_one(state1)

        for it, state_in in ((0, state0), (1, state1)):
            state_out = self.backend.get_a_sample(it)
            for name, (nleaves, ndim) in BRANCH_SHAPES.items():
                np.testing.assert_allclose(
                    state_out.branches[name].coords,
                    state_in.branches[name].coords,
                    err_msg=f"coords mismatch branch {name} it {it}",
                )
                # sub-state tempered ensemble round-trips too
                np.testing.assert_allclose(
                    state_out.sub_states[name].coords,
                    state_in.sub_states[name].coords,
                    err_msg=f"sub-state coords mismatch branch {name} it {it}",
                )
                np.testing.assert_array_equal(
                    state_out.sub_states[name].inds,
                    state_in.sub_states[name].inds,
                    err_msg=f"sub-state inds mismatch branch {name} it {it}",
                )
            for name in ("mbh", "emri", "sobbh"):
                np.testing.assert_allclose(
                    state_out.sub_states[name].betas_all,
                    state_in.sub_states[name].betas_all,
                    err_msg=f"betas_all mismatch branch {name} it {it}",
                )
            # GBHDFBackend keeps the leading step axis on the band_info
            # arrays (stripped later by initialize_band_information's
            # rank-based logic) -- pin that behavior here.
            np.testing.assert_allclose(
                state_out.sub_states["gb"].band_info["band_temps"][0],
                state_in.sub_states["gb"].band_info["band_temps"],
                err_msg=f"band_temps mismatch it {it}",
            )

    def test_reset_kwargs_roundtrip(self):
        """Sub-backend reset_kwargs read back from disk (incl. the EMRI/SOBBH attrs fix).

        Queried per sub-backend rather than through GFHDFBackend.reset_kwargs:
        eryn's HDFBackend.reset_kwargs references an undefined ``self.moves``
        (latent upstream bug), so the merged property cannot be used here.
        """
        sub = self.backend.sub_backend
        gb_kwargs = sub["gb"].reset_kwargs
        self.assertEqual(gb_kwargs["num_bands"], NUM_BANDS)
        np.testing.assert_allclose(gb_kwargs["band_edges"], BAND_EDGES)
        self.assertEqual(sub["mbh"].reset_kwargs["num_mbhs"], BRANCH_SHAPES["mbh"][0])
        self.assertEqual(
            sub["emri"].reset_kwargs["num_emris"], BRANCH_SHAPES["emri"][0]
        )
        self.assertEqual(
            sub["sobbh"].reset_kwargs["num_sobbhs"], BRANCH_SHAPES["sobbh"][0]
        )
        for name, (nleaves, ndim) in BRANCH_SHAPES.items():
            kwargs = sub[name].reset_kwargs
            self.assertEqual(kwargs["ntemps"], NTEMPS, name)
            self.assertEqual(kwargs["nwalkers"], NWALKERS, name)
            self.assertEqual(kwargs["nleaves_max"], nleaves, name)
            self.assertEqual(kwargs["ndim"], ndim, name)

    def test_gfstate_copy_preserves_substates(self):
        """GFState(state, copy=True) deep-copies sub-states incl. tempered blocks."""
        state = make_state(self.rng)
        copied = GFState(state, copy=True)

        self.assertEqual(copied.sub_states["mbh"].num_mbhs, BRANCH_SHAPES["mbh"][0])
        self.assertEqual(copied.sub_states["emri"].num_emris, BRANCH_SHAPES["emri"][0])
        self.assertEqual(
            copied.sub_states["sobbh"].num_sobbhs, BRANCH_SHAPES["sobbh"][0]
        )

        # deep copy: mutating the copy must not touch the original
        copied.sub_states["mbh"].betas_all[:] = -1.0
        self.assertFalse(np.any(state.sub_states["mbh"].betas_all == -1.0))
        copied.sub_states["gb"].band_info["band_temps"][:] = -1.0
        self.assertFalse(
            np.any(state.sub_states["gb"].band_info["band_temps"] == -1.0)
        )
        for name in BRANCH_SHAPES:
            self.assertTrue(copied.sub_states[name].tempered_initialized, name)
            copied.sub_states[name].coords[:] = -99.0
            self.assertFalse(
                np.any(state.sub_states[name].coords == -99.0), name
            )

    def test_cold_row_check_and_sync(self):
        """check_cold_row trips on divergence; sync_cold_row repairs it."""
        state = make_state(self.rng)
        sub = state.sub_states["gb"]

        sub.check_cold_row(state, "gb")  # consistent after pull_from_main

        # a move that updates the sub-state without the main state
        sub.coords[0, 0, 0, 0] += 1.0
        with self.assertRaises(ValueError):
            sub.check_cold_row(state, "gb")

        sub.sync_cold_row(state, "gb")
        sub.check_cold_row(state, "gb")

        # inds divergence trips too
        sub.inds[0, 0, 0] = False
        with self.assertRaises(ValueError):
            sub.check_cold_row(state, "gb")
        sub.sync_cold_row(state, "gb")
        sub.check_cold_row(state, "gb")


# ---------------------------------------------------------------------------
# The GB per-walker families must reach the store (2026-10-10).
#
# 9mo production store, jobs 751/752/753: the per-(walker, band) RJ shutoff
# valve (GB_SEARCH_BAND_SHUTOFF_PER_WALKER=1) was never written to disk. The
# GB sub-backend built its datasets from GBState.make_template with every
# per-walker flag at its default False, and save_step SKIPPED each live array
# that had no dataset. So every resume found no stored valve, restarted it
# "fresh" with the unset step stamp, and released it at step entry: job 751
# was down to 232 active (walker, band) pairs, and after the resume job 752
# reported 1942. The search-stage record and the per-walker cap family were
# dropped the same way.
# ---------------------------------------------------------------------------

#: every per-walker array the three GB flags allocate (``band_stage`` is the
#: stage record's 1-D mirror and rides with it; the 1-D ``cap_cell_*`` twins
#: ride with the per-walker cap family, which disables the divisor-1 short
#: circuit)
PER_WALKER_FAMILY = (
    tuple(SEARCH_SHUTOFF_FIELDS)
    + tuple(SEARCH_SHUTOFF_WINDOW_FIELDS)
    + tuple(SEARCH_STAGE_FIELDS)
    + ("band_stage", "band_best_ll_w")
    + tuple(name for name, _ in CAP_CELL_PER_WALKER_FIELDS)
)

#: the recipe step serial the test valve is earned in
VALVE_SERIAL = 2

_PER_WALKER_FLAGS = (
    "leaf_cap_per_walker",
    "search_stage_per_walker",
    "search_shutoff_per_walker",
)


def make_gb_state(rng, per_walker):
    """A GB-only GFState whose band info carries (or not) the per-walker families."""
    nleaves, ndim = BRANCH_SHAPES["gb"]
    state = GFState(
        {"gb": rng.standard_normal((NTEMPS, NWALKERS, nleaves, ndim))},
        inds={"gb": np.ones((NTEMPS, NWALKERS, nleaves), dtype=bool)},
        log_like=rng.standard_normal((NTEMPS, NWALKERS)),
        log_prior=rng.standard_normal((NTEMPS, NWALKERS)),
        betas=np.linspace(1.0, 0.1, NTEMPS),
        random_state=np.random.get_state(),
        sub_state_bases={"gb": GBState},
    )
    state.sub_states["gb"].initialize_band_information(
        NWALKERS,
        NTEMPS,
        BAND_EDGES,
        np.tile(np.linspace(1.0, 0.1, NTEMPS), (NUM_BANDS, 1)),
        **{flag: per_walker for flag in _PER_WALKER_FLAGS},
    )
    state.sub_states["gb"].pull_from_main(state, "gb")
    return state


def earn_valve(state):
    """Give the live record non-trivial values; return a copy of what was set.

    Some shut pairs (a whole walker, plus two singles), the CURRENT step's
    stamp, mid-window streaks, a finite all-time cold-lnL max, one FINE
    stage, and an armed per-walker cap.
    """
    bi = state.sub_states["gb"].band_info
    bi["band_rj_shutoff_w"][0, 1] = True
    bi["band_rj_shutoff_w"][2, 4] = True
    bi["band_rj_shutoff_w"][3, :] = True
    bi["band_shutoff_w_step"][:] = VALVE_SERIAL
    bi["band_shutoff_streak_w"][:] = 2
    bi["band_shutoff_best_w"][:] = -5.0
    bi["band_cold_logl_max_w"][:] = (
        np.arange(NWALKERS * NUM_BANDS, dtype=float).reshape(NWALKERS, NUM_BANDS)
        - 100.0
    )
    bi["band_stage_w"][1, 2] = 1
    bi["cap_cell_leaf_cap_w"][:] = 4
    return {name: np.array(bi[name], copy=True) for name in PER_WALKER_FAMILY}


class PerWalkerFamilyStoreTest(unittest.TestCase):
    """save_step persists the per-walker families; a resume restores them."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.fp = os.path.join(self.tmpdir.name, "per_walker_test.h5")
        self.rng = np.random.default_rng(20261010)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _backend(self, gb_reset_kwargs):
        nleaves, ndim = BRANCH_SHAPES["gb"]
        backend = GFHDFBackend(
            self.fp,
            sub_backend={"gb": GBHDFBackend},
            sub_state_bases={"gb": GBState},
        )
        backend.reset(
            NWALKERS,
            {"gb": ndim},
            nleaves_max={"gb": nleaves},
            ntemps=NTEMPS,
            branch_names=["gb"],
            nbranches=1,
            rj=False,
            moves=None,
            sub_reset_kwargs={"gb": dict(gb_reset_kwargs)},
        )
        return backend

    def _legacy_backend(self):
        """The production store's layout: the default GB template, no flags."""
        nleaves, ndim = BRANCH_SHAPES["gb"]
        backend = self._backend(
            dict(
                nleaves_max=nleaves,
                ndim=ndim,
                num_bands=NUM_BANDS,
                band_edges=BAND_EDGES,
            )
        )
        with h5py.File(self.fp, "r") as f:
            grp = f["global_fit"]["sub_backend"]["gb"]
            missing = [name for name in PER_WALKER_FAMILY if name not in grp]
        # precondition: this IS the layout the 9mo store was written with
        self.assertEqual(missing, list(PER_WALKER_FAMILY))
        return backend

    def _save(self, backend, state):
        backend.grow(1, None)
        backend.save_step(state, np.ones((NTEMPS, NWALKERS)))

    def test_save_creates_the_missing_datasets_on_a_legacy_store(self):
        backend = self._legacy_backend()
        # two rows written while the run had the flags OFF
        for _ in range(2):
            self._save(backend, make_gb_state(self.rng, per_walker=False))

        state = make_gb_state(self.rng, per_walker=True)
        saved = earn_valve(state)
        with self.assertLogs("lisatools.globalfit.hdfbackend", level="INFO") as logs:
            self._save(backend, state)

        with h5py.File(self.fp, "r") as f:
            grp = f["global_fit"]["sub_backend"]["gb"]
            nrows = grp["band_num_binaries"].shape[0]
            self.assertEqual(nrows, 3)
            for name in PER_WALKER_FAMILY:
                self.assertIn(name, grp, f"{name} never reached the store")
                self.assertEqual(grp[name].shape, (nrows,) + saved[name].shape, name)
                self.assertIsNone(grp[name].maxshape[0], name)
                np.testing.assert_array_equal(grp[name][2], saved[name], err_msg=name)
            # The rows written before the dataset existed read back as the
            # FRESH state, never as a fabricated zero: a 0 all-time max would
            # out-rank every negative band lnL and shut any walker that ever
            # loaded such a row (rescale_store_walkers --mode lag reads them).
            self.assertTrue(np.all(np.isneginf(grp["band_cold_logl_max_w"][:2])))
            self.assertTrue(np.all(np.isneginf(grp["band_shutoff_best_w"][:2])))
            np.testing.assert_array_equal(grp["band_shutoff_w_step"][:2], -1)
            self.assertFalse(np.any(grp["band_rj_shutoff_w"][:2]))
            np.testing.assert_array_equal(grp["band_shutoff_streak_w"][:2], 0)
            np.testing.assert_array_equal(grp["band_stage_occ_last_w"][:2], -1)
            np.testing.assert_array_equal(grp["cap_cell_leaf_cap_w"][:2], -1)
            # dtypes follow reset's rule: in-memory dtype, except the
            # legacy_dtype_names (the cap family) at the backend float dtype
            self.assertEqual(grp["band_rj_shutoff_w"].dtype, np.dtype(bool))
            self.assertEqual(grp["band_shutoff_w_step"].dtype, np.dtype(np.int64))
            self.assertEqual(grp["band_stage_w"].dtype, np.dtype(np.int8))
            self.assertEqual(grp["cap_cell_leaf_cap_w"].dtype, np.dtype(float))

        # one INFO line per created dataset, naming branch and key
        for name in PER_WALKER_FAMILY:
            hits = [m for m in logs.output if f"sub_backend/gb/{name} " in m]
            self.assertEqual(len(hits), 1, f"{name}: {hits}")
            self.assertIn("2 earlier row", hits[0])

        # the band-info reader returns them
        bi = backend.sub_backend["gb"].get_band_info()
        for name in SEARCH_SHUTOFF_FIELDS + SEARCH_SHUTOFF_WINDOW_FIELDS:
            np.testing.assert_array_equal(bi[name][-1], saved[name], err_msg=name)

        # a following grow + save works and the new datasets grow with the rest
        state.sub_states["gb"].band_info["band_rj_shutoff_w"][1, 0] = True
        with self.assertLogs("lisatools.globalfit.hdfbackend", level="INFO") as logs2:
            self._save(backend, state)
        self.assertFalse(any("[STORE] created" in m for m in logs2.output))
        with h5py.File(self.fp, "r") as f:
            grp = f["global_fit"]["sub_backend"]["gb"]
            for name in PER_WALKER_FAMILY:
                self.assertEqual(grp[name].shape[0], 4, name)
            self.assertTrue(grp["band_rj_shutoff_w"][3][1, 0])
            self.assertFalse(grp["band_rj_shutoff_w"][2][1, 0])

    def _assert_resume_keeps_the_valve(self, backend):
        state = make_gb_state(self.rng, per_walker=True)
        saved = earn_valve(state)
        self._save(backend, state)

        # (1) the stored record validates as RESTORED, not "fresh"
        bi = backend.sub_backend["gb"].get_band_info()
        last = {
            name: np.asarray(bi[name])[-1]
            for name in SEARCH_SHUTOFF_FIELDS + SEARCH_SHUTOFF_WINDOW_FIELDS
        }
        last["nwalkers"] = NWALKERS
        self.assertEqual(
            ensure_search_shutoff_fields(last, NUM_BANDS, per_walker=True), "restored"
        )

        # (2) the real resume path: the last row -> GBState band info with
        # the flags ON (as the run passes them) -> the step-entry release
        # under the SAME recipe step serial must release nothing
        resumed = backend.get_a_sample(backend.iteration - 1)
        sub = resumed.sub_states["gb"]
        sub.initialize_band_information(
            NWALKERS,
            NTEMPS,
            BAND_EDGES,
            np.zeros((NUM_BANDS, NTEMPS)),
            **{flag: True for flag in _PER_WALKER_FLAGS},
        )
        for name in PER_WALKER_FAMILY:
            np.testing.assert_array_equal(sub.band_info[name], saved[name], err_msg=name)
        self.assertEqual(release_band_shutoff_window(resumed, VALVE_SERIAL), (0, False))
        self.assertEqual(
            int(np.count_nonzero(sub.band_info["band_rj_shutoff_w"])),
            int(np.count_nonzero(saved["band_rj_shutoff_w"])),
        )

    def test_resume_of_a_legacy_store_keeps_the_valve(self):
        """The production failure: same step, resumed -> the valve stays shut."""
        backend = self._legacy_backend()
        self._save(backend, make_gb_state(self.rng, per_walker=False))
        self._assert_resume_keeps_the_valve(backend)

    def test_fresh_store_gets_the_datasets_at_reset(self):
        """A store created from the LIVE state's reset_kwargs (the run.py
        route) has the per-walker datasets before the first save, so no save
        ever has to create one."""
        live = make_gb_state(self.rng, per_walker=True)
        backend = self._backend(live.sub_states["gb"].reset_kwargs)
        with h5py.File(self.fp, "r") as f:
            grp = f["global_fit"]["sub_backend"]["gb"]
            for name in PER_WALKER_FAMILY:
                self.assertIn(name, grp, name)
        with self.assertLogs("lisatools.globalfit.hdfbackend", level="INFO") as logs:
            self._assert_resume_keeps_the_valve(backend)
        self.assertFalse(any("[STORE] created" in m for m in logs.output))

    def test_flags_off_store_is_unchanged(self):
        """With the flags off the live reset_kwargs build the old layout."""
        live = make_gb_state(self.rng, per_walker=False)
        self._backend(live.sub_states["gb"].reset_kwargs)
        with h5py.File(self.fp, "r") as f:
            grp = f["global_fit"]["sub_backend"]["gb"]
            self.assertEqual(set(grp.keys()), set(EXPECTED_SUB_SCHEMA["gb"]))


class PerWalkerTemplateTest(unittest.TestCase):
    """GBState.make_template allocates exactly what the live state will save."""

    def _template(self, **flags):
        nleaves, ndim = BRANCH_SHAPES["gb"]
        return GBState.make_template(
            NWALKERS,
            NTEMPS,
            num_bands=NUM_BANDS,
            band_edges=BAND_EDGES,
            nleaves_max=nleaves,
            ndim=ndim,
            **flags,
        )

    def test_flags_on_allocate_the_per_walker_arrays(self):
        names = set(
            self._template(**{flag: True for flag in _PER_WALKER_FLAGS}).storage_arrays()
        )
        for name in PER_WALKER_FAMILY:
            self.assertIn(name, names, name)

    def test_flags_off_keep_the_old_layout(self):
        names = set(self._template().storage_arrays())
        for name in PER_WALKER_FAMILY:
            self.assertNotIn(name, names, name)
        self.assertEqual(
            names, set(EXPECTED_SUB_SCHEMA["gb"]) - STATIC_DATASETS["gb"]
        )

    def test_each_flag_allocates_its_own_family(self):
        shutoff = set(SEARCH_SHUTOFF_FIELDS) | set(SEARCH_SHUTOFF_WINDOW_FIELDS)
        stage = set(SEARCH_STAGE_FIELDS) | {"band_stage"}
        cap = {"band_best_ll_w"} | {name for name, _ in CAP_CELL_PER_WALKER_FIELDS}
        for flag, own in (
            ("search_shutoff_per_walker", shutoff),
            ("search_stage_per_walker", stage),
            ("leaf_cap_per_walker", cap),
        ):
            names = set(self._template(**{flag: True}).storage_arrays())
            self.assertTrue(own <= names, f"{flag}: missing {own - names}")
            others = (shutoff | stage | cap) - own
            self.assertFalse(others & names, f"{flag}: also allocated {others & names}")

    def test_reset_kwargs_rebuild_exactly_the_live_storage(self):
        """reset_kwargs -> make_template gives the live state's datasets."""
        rng = np.random.default_rng(7)
        for per_walker in (False, True):
            live = make_gb_state(rng, per_walker).sub_states["gb"]
            rk = dict(live.reset_kwargs)
            for flag in _PER_WALKER_FLAGS:
                self.assertIs(rk[flag], per_walker, flag)
            tmpl = GBState.make_template(rk.pop("nwalkers"), rk.pop("ntemps"), **rk)
            want = live.storage_arrays()
            got = tmpl.storage_arrays()
            self.assertEqual(set(got), set(want), f"per_walker={per_walker}")
            for name, arr in want.items():
                self.assertEqual(np.shape(got[name]), np.shape(arr), name)
                self.assertEqual(np.asarray(got[name]).dtype, np.asarray(arr).dtype, name)


if __name__ == "__main__":
    unittest.main()
