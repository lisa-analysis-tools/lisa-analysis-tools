"""VGB chirp-basis store migration: rebirth at injection (and the legacy carry).

User ruling 2026-09-16: "prepare a transfer script to keep the hdf backend
except for the VGBs (and just start them at injection again)". The DEFAULT
mode of ``scripts/fstat_proposal/migrate_vgb_chirp_basis.py`` is therefore a
clean vgb REBIRTH -- the vgb chain is discarded and reseeded at catalogue
truth in the 6-column chirp basis, and EVERYTHING else in the store (the GB
chains and band state, noise / galfor, mbh / emri / sobbh, log_like, betas,
the iteration counter) must come through byte-for-byte untouched, so the run
resumes exactly where it was.

These run against a tiny synthetic store (3 stored iterations, 2 sub-backend
rungs, 3 walkers, 4 leaves) -- no data, no waveforms, no catalogue file for
the store-side cases (``migrate_store`` takes the injection rows as an
argument, so the two halves are testable independently).
"""

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lisatools.globalfit.run import check_store_branch_ndims
from lisatools.globalfit.stock.erebor.vgb import (
    VGB_SAMPLED_BASIS_CHIRP,
    VGB_SAMPLED_BASIS_DIST,
)

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "fstat_proposal" / "migrate_vgb_chirp_basis.py")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "migrate_vgb_chirp_basis_for_test", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MIG = _load_script()

NITER = 3
NTEMPS_SUB = 2
NWALKERS = 3
NLEAVES = 4
OLD_NDIM = 5
NEW_NDIM = 6


def _fake_injection(nleaves=NLEAVES):
    """Distinctive per-leaf, per-column truth (no column is a duplicate)."""
    inj = np.zeros((nleaves, NEW_NDIM))
    inj[:, 0] = np.linspace(1.0, 4.0, nleaves)          # dist kpc
    inj[:, 1] = np.linspace(0.1, 3.0, nleaves)          # phi0
    inj[:, 2] = np.linspace(-0.9, 0.9, nleaves)         # cos_iota
    inj[:, 3] = np.linspace(0.2, 2.8, nleaves)          # psi
    inj[:, 4] = np.linspace(0.20, 0.55, nleaves)        # Mc
    inj[:, 5] = np.linspace(-0.03, 0.04, nleaves)       # fdot_astro_ratio
    return inj


def _make_store(path):
    """A miniature global-fit store with a 5-column vgb branch."""
    rng = np.random.default_rng(7)
    main_shape = (NITER, 1, NWALKERS, NLEAVES, OLD_NDIM)
    sub_shape = (NITER, NTEMPS_SUB, NWALKERS, NLEAVES, OLD_NDIM)
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = NITER

        chain = g.create_group("chain")
        chain.create_dataset("vgb", data=rng.standard_normal(main_shape),
                             maxshape=(None,) + main_shape[1:])
        # control branches: must survive bit-identical
        gb_shape = (NITER, 1, NWALKERS, 7, 9)
        chain.create_dataset("gb", data=rng.standard_normal(gb_shape),
                             maxshape=(None,) + gb_shape[1:])
        psd_shape = (NITER, 1, NWALKERS, 1, 2)
        chain.create_dataset("psd", data=rng.standard_normal(psd_shape),
                             maxshape=(None,) + psd_shape[1:])

        inds = g.create_group("inds")
        inds.create_dataset(
            "vgb", data=np.ones(main_shape[:-1], dtype=bool),
            maxshape=(None,) + main_shape[1:-1])
        inds.create_dataset(
            "gb", data=np.ones(gb_shape[:-1], dtype=bool),
            maxshape=(None,) + gb_shape[1:-1])

        ndims = g.create_group("ndims")
        ndims.attrs["vgb"] = OLD_NDIM
        ndims.attrs["gb"] = 9
        ndims.attrs["psd"] = 2

        ko = g.create_group("key_order")
        ko.attrs["vgb"] = np.arange(OLD_NDIM)
        ko.attrs["gb"] = np.arange(9)

        g.create_dataset("log_like", data=rng.standard_normal((NITER, 1, NWALKERS)))
        g.create_dataset("log_prior", data=rng.standard_normal((NITER, 1, NWALKERS)))

        sb = g.create_group("sub_backend")
        vsub = sb.create_group("vgb")
        vsub.attrs["ndim"] = OLD_NDIM
        vsub.create_dataset("chain", data=rng.standard_normal(sub_shape),
                            maxshape=(None,) + sub_shape[1:])
        vsub.create_dataset("inds", data=np.ones(sub_shape[:-1], dtype=bool),
                            maxshape=(None,) + sub_shape[1:-1])
        # band bookkeeping: left alone by the migration (see the audit note
        # in the script) -- pinned here so a future change is deliberate.
        vsub.create_dataset("band_temps", data=rng.random((5, NTEMPS_SUB)))
        vsub.create_dataset("band_num_binaries", data=np.arange(5))
        vsub.create_dataset("band_proposals", data=np.arange(5) * 3)

        gsub = sb.create_group("gb")
        gsub.attrs["ndim"] = 9
        gsub.create_dataset("chain", data=rng.standard_normal(
            (NITER, NTEMPS_SUB, NWALKERS, 7, 9)))
        gsub.create_dataset("band_temps", data=rng.random((11, NTEMPS_SUB)))


def _snapshot(path, skip_vgb=True):
    """``{dataset path: bytes}`` for every dataset, plus every attr."""
    out = {}

    def _visit(name, obj):
        if skip_vgb and "vgb" in name:
            return
        if isinstance(obj, h5py.Dataset):
            out["DS:" + name] = obj[()].tobytes()
        for k, v in obj.attrs.items():
            if skip_vgb and k == "vgb":
                continue
            out[f"ATTR:{name}:{k}"] = np.asarray(v).tobytes()

    with h5py.File(path, "r") as f:
        f.visititems(_visit)
        for k, v in f["global_fit"].attrs.items():
            out[f"ATTR:global_fit:{k}"] = np.asarray(v).tobytes()
    return out


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vgbmig_")
        self.path = os.path.join(self.tmp, "store.h5")
        _make_store(self.path)
        self.inj = _fake_injection()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _migrate(self, **kwargs):
        MIG.migrate_store(
            self.path, self.inj, self.inj[:, MIG.MC_COL], **kwargs)


class RebirthAtInjectionTest(_StoreCase):
    def test_everything_else_is_bit_identical(self):
        """The whole point: keep the hdf backend except for the VGBs."""
        before = _snapshot(self.path)
        self._migrate()
        after = _snapshot(self.path)
        self.assertEqual(set(before), set(after))
        for key in before:
            self.assertEqual(before[key], after[key], key)

    def test_iteration_counter_untouched(self):
        self._migrate()
        with h5py.File(self.path, "r") as f:
            self.assertEqual(int(f["global_fit"].attrs["iteration"]), NITER)

    def test_resume_row_is_exact_catalogue_truth(self):
        """Row ``iteration - 1`` is what Backend.get_last_sample reads."""
        self._migrate()
        with h5py.File(self.path, "r") as f:
            it = int(f["global_fit"].attrs["iteration"])
            main = f["global_fit"]["chain"]["vgb"][it - 1]
            sub = f["global_fit"]["sub_backend"]["vgb"]["chain"][it - 1]
        self.assertEqual(main.shape, (1, NWALKERS, NLEAVES, NEW_NDIM))
        self.assertEqual(sub.shape, (NTEMPS_SUB, NWALKERS, NLEAVES, NEW_NDIM))
        # exact truth in EVERY rung / walker / leaf, no jitter
        np.testing.assert_array_equal(
            main, np.broadcast_to(self.inj, main.shape))
        np.testing.assert_array_equal(
            sub, np.broadcast_to(self.inj, sub.shape))

    def test_columns_land_in_the_chirp_basis_order(self):
        self._migrate()
        with h5py.File(self.path, "r") as f:
            row = f["global_fit"]["chain"]["vgb"][-1, 0, 0]  # (nleaves, 6)
        for col, name in enumerate(VGB_SAMPLED_BASIS_CHIRP):
            np.testing.assert_array_equal(
                row[:, col], self.inj[:, col],
                err_msg=f"column {col} ({name})")
        # the two columns the whole ruling is about are really there
        self.assertEqual(VGB_SAMPLED_BASIS_CHIRP.index("Mc"), MIG.MC_COL)
        self.assertEqual(
            VGB_SAMPLED_BASIS_CHIRP.index("fdot_astro_ratio"), MIG.RATIO_COL)

    def test_every_row_is_filled_and_finite(self):
        """No NaN history: whole-run reductions in the monitor stay finite."""
        self._migrate()
        with h5py.File(self.path, "r") as f:
            main = f["global_fit"]["chain"]["vgb"][:]
            sub = f["global_fit"]["sub_backend"]["vgb"]["chain"][:]
        self.assertTrue(np.all(np.isfinite(main)))
        self.assertTrue(np.all(np.isfinite(sub)))
        for row in range(NITER):
            np.testing.assert_array_equal(
                main[row], np.broadcast_to(self.inj, main[row].shape))

    def test_ndim_attrs_flipped_in_both_places(self):
        self._migrate()
        with h5py.File(self.path, "r") as f:
            g = f["global_fit"]
            self.assertEqual(int(g["ndims"].attrs["vgb"]), NEW_NDIM)
            self.assertEqual(
                int(g["sub_backend"]["vgb"].attrs["ndim"]), NEW_NDIM)
            self.assertEqual(
                list(g["key_order"].attrs["vgb"]), list(range(NEW_NDIM)))
            # untouched neighbours
            self.assertEqual(int(g["ndims"].attrs["gb"]), 9)
            self.assertEqual(int(g["sub_backend"]["gb"].attrs["ndim"]), 9)

    def test_vgb_band_bookkeeping_is_left_alone(self):
        """band_* is (num_bands, ...) -- no parameter axis, so ndim cannot
        invalidate it; band_temps is rewritten from config at every build
        (recipe.build_vgb_moves), the rest are diagnostics counters."""
        with h5py.File(self.path, "r") as f:
            vsub = f["global_fit"]["sub_backend"]["vgb"]
            before = {k: vsub[k][()] for k in vsub if k.startswith("band_")}
        self.assertTrue(before, "fixture should carry band_* datasets")
        self._migrate()
        with h5py.File(self.path, "r") as f:
            vsub = f["global_fit"]["sub_backend"]["vgb"]
            after = {k: vsub[k][()] for k in vsub if k.startswith("band_")}
        self.assertEqual(set(before), set(after))
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)

    def test_inds_stay_all_true(self):
        self._migrate()
        with h5py.File(self.path, "r") as f:
            self.assertTrue(f["global_fit"]["inds"]["vgb"][:].all())
            self.assertTrue(f["global_fit"]["sub_backend"]["vgb"]["inds"][:].all())

    def test_dead_leaf_sentinel_preserved(self):
        """A dead leaf (inds False) still gets the NaN sentinel."""
        with h5py.File(self.path, "r+") as f:
            f["global_fit"]["inds"]["vgb"][-1, 0, 0, 2] = False
        self._migrate()
        with h5py.File(self.path, "r") as f:
            main = f["global_fit"]["chain"]["vgb"][:]
        self.assertTrue(np.all(np.isnan(main[-1, 0, 0, 2])))
        self.assertTrue(np.all(np.isfinite(main[-1, 0, 0, 3])))

    def test_migrated_store_passes_the_resume_guard(self):
        """End-to-end: the run.py construction-level guard now ACCEPTS it."""
        cfg = {"vgb": NEW_NDIM, "gb": 9, "psd": 2}
        # before: the guard refuses, naming the knob
        with h5py.File(self.path, "r") as f:
            stored = {k: int(v) for k, v in f["global_fit"]["ndims"].attrs.items()}
        with self.assertRaises(ValueError) as ctx:
            check_store_branch_ndims(stored, cfg, self.path)
        self.assertIn("VGB_CHIRP_MASS_BASIS", str(ctx.exception))
        # after: accepted
        self._migrate()
        with h5py.File(self.path, "r") as f:
            stored = {k: int(v) for k, v in f["global_fit"]["ndims"].attrs.items()}
        self.assertEqual(stored["vgb"], NEW_NDIM)
        check_store_branch_ndims(stored, cfg, self.path)  # must not raise

    def test_second_run_refuses_an_already_migrated_store(self):
        self._migrate()
        with self.assertRaises(SystemExit) as ctx:
            self._migrate()
        self.assertIn("already migrated", str(ctx.exception))

    def test_leaf_count_mismatch_is_loud(self):
        with self.assertRaises(SystemExit) as ctx:
            MIG.migrate_store(
                self.path, _fake_injection(NLEAVES + 1),
                _fake_injection(NLEAVES + 1)[:, MIG.MC_COL])
        self.assertIn("leaves", str(ctx.exception))

    def test_start_factor_adds_spread_but_keeps_the_mean(self):
        """The opt-in stretch-arm escape hatch still uses the production form."""
        self._migrate(start_factor=1e-3)
        with h5py.File(self.path, "r") as f:
            main = f["global_fit"]["chain"]["vgb"][-1]
        self.assertTrue(np.all(np.isfinite(main)))
        # walkers are no longer identical ...
        self.assertFalse(np.allclose(main[0, 0], main[0, 1]))
        # ... but still within a few permille of truth on the Mc column
        rel = np.abs(main[..., MIG.MC_COL] / self.inj[:, MIG.MC_COL] - 1.0)
        self.assertLess(rel.max(), 1e-2)


class CarryHistoryTest(_StoreCase):
    """--carry-history keeps the 2026-08-14 behaviour exactly."""

    def test_old_columns_land_in_slots_0123_and_5(self):
        with h5py.File(self.path, "r") as f:
            old_main = f["global_fit"]["chain"]["vgb"][:]
        self._migrate(carry_history=True)
        with h5py.File(self.path, "r") as f:
            new_main = f["global_fit"]["chain"]["vgb"][:]
        self.assertEqual(new_main.shape[-1], NEW_NDIM)
        for j_old, j_new in enumerate(MIG.OLD_TO_NEW[:4]):
            np.testing.assert_allclose(
                new_main[..., j_new], old_main[..., j_old],
                err_msg=f"old col {j_old} -> new col {j_new}")

    def test_mc_column_is_catalogue_value_with_tiny_jitter(self):
        self._migrate(carry_history=True)
        with h5py.File(self.path, "r") as f:
            mc = f["global_fit"]["chain"]["vgb"][..., MIG.MC_COL]
        np.testing.assert_allclose(
            mc, np.broadcast_to(self.inj[:, MIG.MC_COL], mc.shape),
            rtol=1e-4)
        self.assertFalse(np.array_equal(mc[0, 0, 0], mc[0, 0, 1]))

    def test_ratio_column_is_reinitialized_not_carried(self):
        """The legacy path RESETS the ratio column (zero truth + additive)."""
        self._migrate(carry_history=True)
        with h5py.File(self.path, "r") as f:
            ratio = f["global_fit"]["chain"]["vgb"][..., MIG.RATIO_COL]
        # additive width = START_FACTOR * RATIO_INIT_WIDTH * RATIO_MAX
        width = (MIG.START_FACTOR * MIG.RATIO_INIT_WIDTH
                 * MIG.FDOT_ASTRO_RATIO_MAX)
        self.assertLess(np.abs(ratio).max(), 50.0 * max(width, 1e-12))

    def test_attrs_flipped_too(self):
        self._migrate(carry_history=True)
        with h5py.File(self.path, "r") as f:
            self.assertEqual(
                int(f["global_fit"]["ndims"].attrs["vgb"]), NEW_NDIM)
            self.assertEqual(
                int(f["global_fit"]["sub_backend"]["vgb"].attrs["ndim"]),
                NEW_NDIM)


class SidecarQuarantineTest(_StoreCase):
    """The midit checkpoint and the running backup hold 5-column vgb state."""

    def _write_sidecars(self):
        base, _ = os.path.splitext(self.path)
        made = {
            "ckpt": base + "_midit_checkpoint.pkl",
            "backup": self.path[:-3] + "_running_backup_copy.h5",
        }
        for p in made.values():
            with open(p, "wb") as fh:
                fh.write(b"stale 5-column vgb state")
        return made

    def test_sidecars_are_moved_into_the_quarantine_dir(self):
        made = self._write_sidecars()
        moved = MIG.quarantine_sidecars(self.path)
        self.assertEqual(len(moved), 2)
        qdir = os.path.join(self.tmp, MIG.QUARANTINE_DIRNAME)
        self.assertTrue(os.path.isdir(qdir))
        for p in made.values():
            self.assertFalse(os.path.exists(p), f"{p} still in place")
            self.assertTrue(
                os.path.exists(os.path.join(qdir, os.path.basename(p))))

    def test_no_sidecars_is_a_clean_no_op(self):
        self.assertEqual(MIG.quarantine_sidecars(self.path), [])

    def test_refuses_to_clobber_a_previous_quarantine(self):
        self._write_sidecars()
        MIG.quarantine_sidecars(self.path)
        self._write_sidecars()          # a second stale pair appears
        with self.assertRaises(SystemExit) as ctx:
            MIG.quarantine_sidecars(self.path)
        self.assertIn("quarantin", str(ctx.exception).lower())

    def test_store_itself_is_not_quarantined(self):
        self._write_sidecars()
        MIG.quarantine_sidecars(self.path)
        self.assertTrue(os.path.exists(self.path))


def _fake_catalogue(ratios):
    """A SELF-CONSISTENT fake VGB catalogue dict, 2 keys x 2 sources.

    Built backwards from known ``(dist, Mc, ratio, angles)`` so the test can
    assert the script recovers exactly those:

    * ``Amplitude`` is derived from (f0, Mc, dist) with the production
      :func:`gb_amp_from_dist`, so the script's catalogue-consistency check
      (which exists to catch a unit/convention change) actually passes on
      real physics rather than being bypassed;
    * ``GW22FrequencyDerivativeSourceFrame`` = ``fdot_gr * (1 + ratio)``
      with a NON-ZERO ratio, so the computed ratio column is pinned to a
      known value instead of trivially 0.

    Two sources per key exercises the "catalogue stores whole arrays under
    one id" concatenation, and the keys are deliberately out of order so the
    sorted-key leaf ordering is under test.
    """
    from lisatools.globalfit.stock.erebor.transforms import (
        McDistFdotAstroQuad,
        gb_amp_from_dist,
    )

    ratios = np.asarray(ratios, dtype=float)
    n = ratios.size
    dist_kpc = np.linspace(0.4, 2.2, n)
    mc = np.linspace(0.22, 0.48, n)
    f0_hz = np.linspace(3.2e-3, 9.7e-3, n)
    true_anom = np.linspace(0.3, 5.4, n)
    iota = np.linspace(0.4, 2.6, n)
    psi = np.linspace(0.15, 2.9, n)
    ra = np.linspace(0.2, 6.0, n)
    dec = np.linspace(-1.1, 1.1, n)

    _, _, fdot_gr, _ = McDistFdotAstroQuad()(
        dist_kpc, f0_hz, mc, np.zeros_like(dist_kpc))
    fdot = fdot_gr * (1.0 + ratios)
    amp = gb_amp_from_dist(f0_hz, mc, dist_kpc)

    half = n // 2
    sl = {"b_second": slice(half, n), "a_first": slice(0, half)}
    cat = {}
    for key, s in sl.items():
        cat[key] = {
            "Amplitude": amp[s],
            "GW22FrequencySSBFrame": f0_hz[s],
            "GW22FrequencyDerivativeSourceFrame": fdot[s],
            "TrueAnomaly": true_anom[s],
            "InclinationAngle": iota[s],
            "PolarisationAngle": psi[s],
            "RightAscension": ra[s],
            "Declination": dec[s],
            "LuminosityDistance": dist_kpc[s] * 1e-3,   # kpc -> Mpc
            "ChirpMassSSBFrame": mc[s],
        }
    # sorted(keys) == ["a_first", "b_second"] -> leaves [0:half] then [half:n]
    truth = {"dist_kpc": dist_kpc, "mc": mc, "f0_hz": f0_hz,
             "ratios": ratios, "iota": iota, "psi": psi}
    return cat, truth


class InjectionFromCatalogueTest(unittest.TestCase):
    """``vgb_injection_rows`` mirrors prepare_vgb_branch (vgb.py:1093-1151)."""

    RATIOS = np.array([0.0, 0.05, -0.02, 0.13])

    def setUp(self):
        self.cat, self.truth = _fake_catalogue(self.RATIOS)
        self._orig = MIG.load_vgb_catalogue_file
        MIG.load_vgb_catalogue_file = lambda _dir: self.cat

    def tearDown(self):
        MIG.load_vgb_catalogue_file = self._orig

    def test_six_columns_in_the_chirp_basis_order(self):
        inj = MIG.vgb_injection_rows("ignored")
        self.assertEqual(inj.shape, (self.RATIOS.size, NEW_NDIM))

    def test_dist_is_converted_mpc_to_kpc(self):
        inj = MIG.vgb_injection_rows("ignored")
        np.testing.assert_allclose(
            inj[:, VGB_SAMPLED_BASIS_CHIRP.index("dist")],
            self.truth["dist_kpc"], rtol=1e-12)

    def test_mc_is_the_catalogue_chirp_mass(self):
        inj = MIG.vgb_injection_rows("ignored")
        np.testing.assert_allclose(
            inj[:, MIG.MC_COL], self.truth["mc"], rtol=1e-12)

    def test_ratio_is_COMPUTED_not_assumed_zero(self):
        """fdot_cat / fdot_gr(d, f0, Mc) - 1, recovered to round-off."""
        inj = MIG.vgb_injection_rows("ignored")
        np.testing.assert_allclose(
            inj[:, MIG.RATIO_COL], self.truth["ratios"], atol=1e-10)
        # and it is genuinely non-trivial: a zero-filled column would fail
        self.assertGreater(np.abs(inj[:, MIG.RATIO_COL]).max(), 0.1)

    def test_angles_are_the_container_sampling_values(self):
        """phi0 / cos_iota / psi copied VERBATIM from the container rows."""
        from lisatools.globalfit.recipe import gb_catalogue_to_sampling_basis
        from lisatools.globalfit.stock.erebor.transforms import (
            make_gb_transform_container,
        )

        inj = MIG.vgb_injection_rows("ignored")
        rows = np.array([gb_catalogue_to_sampling_basis(self.cat[k])
                         for k in sorted(self.cat)])
        rows = rows.reshape(-1, rows.shape[-1])
        full = list(make_gb_transform_container(use_chirp_mass=False).input_basis)
        for name in ("phi0", "cos_iota", "psi"):
            np.testing.assert_allclose(
                inj[:, VGB_SAMPLED_BASIS_CHIRP.index(name)],
                rows[:, full.index(name)], rtol=1e-12, err_msg=name)
        # independent sanity check on one of them
        np.testing.assert_allclose(
            inj[:, VGB_SAMPLED_BASIS_CHIRP.index("cos_iota")],
            np.cos(self.truth["iota"]), atol=1e-10)

    def test_leaf_order_is_sorted_catalogue_keys(self):
        """Keys are supplied out of order; leaves must follow sorted order."""
        inj = MIG.vgb_injection_rows("ignored")
        self.assertEqual(sorted(self.cat), ["a_first", "b_second"])
        np.testing.assert_allclose(
            inj[:, MIG.MC_COL], self.truth["mc"], rtol=1e-12)

    def test_inconsistent_catalogue_amplitude_fails_loudly(self):
        """The (f0, Mc, dist) -> A guard must not be silently bypassable."""
        self.cat["a_first"]["Amplitude"] = (
            np.asarray(self.cat["a_first"]["Amplitude"]) * 1.5)
        with self.assertRaises(SystemExit) as ctx:
            MIG.vgb_injection_rows("ignored")
        self.assertIn("Amplitude", str(ctx.exception))

    def test_production_shaped_single_key_catalogue(self):
        """The REAL loader returns ONE key holding whole-column arrays.

        ``load_vgb_catalogue_file`` reads
        ``<dir>/catalogues/vgb_cat_mojito_lite_processed.hdf5`` and returns
        ``{"vgb": {column: array_over_all_55_sources}}`` -- a single entry,
        not one per source. The multi-key fixture above exercises the
        concatenation; this pins the shape production actually hits.
        """
        merged = {}
        for col in self.cat["a_first"]:
            merged[col] = np.concatenate([
                np.asarray(self.cat[k][col], dtype=float)
                for k in sorted(self.cat)
            ])
        MIG.load_vgb_catalogue_file = lambda _dir: {"vgb": merged}
        inj = MIG.vgb_injection_rows("ignored")
        self.assertEqual(inj.shape, (self.RATIOS.size, NEW_NDIM))
        np.testing.assert_allclose(
            inj[:, MIG.MC_COL], self.truth["mc"], rtol=1e-12)
        np.testing.assert_allclose(
            inj[:, MIG.RATIO_COL], self.truth["ratios"], atol=1e-10)
        np.testing.assert_allclose(
            inj[:, VGB_SAMPLED_BASIS_CHIRP.index("dist")],
            self.truth["dist_kpc"], rtol=1e-12)

    def test_leaf_chirp_masses_matches_the_injection_column(self):
        """The --carry-history helper and the reseed agree on leaf order."""
        inj = MIG.vgb_injection_rows("ignored")
        np.testing.assert_allclose(
            MIG.leaf_chirp_masses("ignored"), inj[:, MIG.MC_COL], rtol=1e-12)

    def test_end_to_end_store_reseed_uses_these_rows(self):
        """Catalogue -> injection -> the row the resume actually reads."""
        inj = MIG.vgb_injection_rows("ignored")
        tmp = tempfile.mkdtemp(prefix="vgbmig_e2e_")
        try:
            path = os.path.join(tmp, "store.h5")
            _make_store(path)
            MIG.migrate_store(path, inj, inj[:, MIG.MC_COL])
            with h5py.File(path, "r") as f:
                it = int(f["global_fit"].attrs["iteration"])
                row = f["global_fit"]["chain"]["vgb"][it - 1]
            np.testing.assert_array_equal(
                row, np.broadcast_to(inj, row.shape))
            np.testing.assert_allclose(
                row[0, 0, :, MIG.RATIO_COL], self.RATIOS, atol=1e-10)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class InjectionBasisContractTest(unittest.TestCase):
    """Column contract the reseed depends on (drift guard, no catalogue)."""

    def test_basis_constants_agree_with_the_script(self):
        self.assertEqual(MIG.NEW_NDIM, len(VGB_SAMPLED_BASIS_CHIRP))
        self.assertEqual(MIG.OLD_NDIM, len(VGB_SAMPLED_BASIS_DIST))
        self.assertEqual(
            MIG.MC_COL, VGB_SAMPLED_BASIS_CHIRP.index("Mc"))
        self.assertEqual(
            MIG.RATIO_COL, VGB_SAMPLED_BASIS_CHIRP.index("fdot_astro_ratio"))

    def test_carry_map_targets_the_ratio_slot(self):
        """Old col 4 (ratio) must map to the NEW ratio slot, not blindly 4."""
        self.assertEqual(
            VGB_SAMPLED_BASIS_DIST.index("fdot_astro_ratio"), 4)
        self.assertEqual(MIG.OLD_TO_NEW[4], MIG.RATIO_COL)
        self.assertEqual(MIG.OLD_TO_NEW, [0, 1, 2, 3, MIG.RATIO_COL])


if __name__ == "__main__":
    unittest.main()
