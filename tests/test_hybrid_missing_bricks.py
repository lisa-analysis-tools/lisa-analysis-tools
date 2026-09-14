"""Hybrid mojito data: real bricks where present, synthetic fill elsewhere.

User ask 2026-09-14 ("combination of synthetic and real (where possible)"):
the cluster has the catalogues and most bricks, but some SOBHB/EMRI L1
bricks are still in transfer. The loader learns ``allow_missing_bricks``
(record the miss, keep the catalogue parameters, keep loading);
``L1ProcessingStepWithSyntheticNoise`` learns ``synthesize_missing`` (build
the missing sources' TD streams from their CATALOGUE parameters with the
same generators, epochs, and — crucially — the run's REAL orbits, then sum
them into the data). Grid + orbits still come from the first real brick.
"""

import contextlib
import os
import types
import unittest
from unittest import mock

import h5py
import numpy as np

from lisatools.globalfit import preprocessing
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.stock.erebor import injections as inj
from lisatools.globalfit.stock.erebor.injections import (
    L1ProcessingStepWithSyntheticNoise,
    sobbh_catalogue_to_waveform_basis,
)

N_FAKE = 64
DT_FAKE = 8.0

SOBHB_KEYS = {
    "PrimaryMassSSBFrame": 60.0,
    "SecondaryMassSSBFrame": 55.0,
    "PrimarySpinCompZ": 0.1,
    "SecondarySpinCompZ": 0.2,
    "LuminosityDistance": 400.0,  # Mpc
    "InclinationAngle": 1.2,
    "GW22FrequencySSBFrame": 6e-3,
    "RightAscension": 3.1,
    "Declination": 0.2,
    "PolarisationAngle": 0.7,
    "TrueAnomaly": 1.1,
}


class _FakeOrbits:
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def _ensure_configured(self):
        return None


def _fake_l1_file():
    ts = types.SimpleNamespace(
        dt=DT_FAKE,
        fs=1.0 / DT_FAKE,
        t=lambda: np.arange(N_FAKE) * DT_FAKE,
    )
    tdis = types.SimpleNamespace(
        xyz_doppler=np.ones((N_FAKE, 3)), time_sampling=ts
    )
    return types.SimpleNamespace(tdis=tdis)


@contextlib.contextmanager
def _fake_open(path):
    yield _fake_l1_file()


class _StubL1Step(L1ProcessingStep):
    def _open(self, file_path):
        return _fake_open(file_path)


class _StubHybridStep(L1ProcessingStepWithSyntheticNoise):
    def _open(self, file_path):
        return _fake_open(file_path)


def _make_fake_mojito(tmp, sobhb_ids=(0, 1), sobhb_bricks=(0,),
                      vgb=False, vgb_brick=True):
    """A minimal mojito-shaped folder: catalogues + (some) bricks."""
    cat_dir = os.path.join(tmp, "catalogues")
    os.makedirs(cat_dir, exist_ok=True)
    n = max(sobhb_ids) + 1
    with h5py.File(os.path.join(
            cat_dir, "sobhb_cat_mojito_lite_processed_MT.hdf5"), "w") as f:
        g = f.create_group("Binaries")
        for key, base in SOBHB_KEYS.items():
            g.create_dataset(
                key, data=base * (1.0 + 0.01 * np.arange(n))
            )
    d = os.path.join(tmp, "data", "SOBHB", "L1")
    os.makedirs(d, exist_ok=True)
    for sid in sobhb_bricks:
        open(os.path.join(d, f"SOBHB_test_source{sid}_0.h5"), "wb").close()
    if vgb:
        with h5py.File(os.path.join(
                cat_dir, "vgb_cat_mojito_lite_processed.hdf5"), "w") as f:
            g = f.create_group("Binaries")
            g.create_dataset("Frequency", data=np.array([1e-3]))
        dv = os.path.join(tmp, "data", "VGB", "L1")
        os.makedirs(dv, exist_ok=True)
        if vgb_brick:
            open(os.path.join(dv, "VGB_test_source0_0.h5"), "wb").close()
    return tmp


class LoaderMissingBricksTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="hybrid_mojito_")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _step(self, **kw):
        kw.setdefault("L1_folder", self.tmp)
        kw.setdefault("source_types", ["SOBHB"])
        kw.setdefault("source_ids", {"SOBHB": [0, 1]})
        kw.setdefault("orbits_class", _FakeOrbits)
        kw.setdefault("verbose", False)
        return _StubL1Step(**kw)

    def test_default_still_raises_on_missing_brick(self):
        _make_fake_mojito(self.tmp, sobhb_bricks=(0,))
        with self.assertRaises(FileNotFoundError):
            self._step()

    def test_missing_brick_recorded_catalogue_kept_data_from_present(self):
        _make_fake_mojito(self.tmp, sobhb_bricks=(0,))
        step = self._step(allow_missing_bricks=True)
        self.assertEqual(step.missing_source_bricks, [("SOBHB", 1)])
        # catalogue parameters loaded for BOTH ids (the synthetic fill needs
        # the missing one's truth)
        self.assertIn(0, step.catalogue["SOBHB"])
        self.assertIn(1, step.catalogue["SOBHB"])
        self.assertAlmostEqual(
            float(step.catalogue["SOBHB"][1]["PrimaryMassSSBFrame"]),
            SOBHB_KEYS["PrimaryMassSSBFrame"] * 1.01,
        )
        # the data stream is exactly the ONE present brick
        np.testing.assert_allclose(step.data, np.ones((3, N_FAKE)))
        # grid + orbits came from the first real brick
        self.assertIsInstance(step.orbits, _FakeOrbits)

    def test_no_brick_at_all_is_a_clear_error(self):
        _make_fake_mojito(self.tmp, sobhb_bricks=())
        with self.assertRaises(RuntimeError):
            self._step(allow_missing_bricks=True)


class HybridSynthesisTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="hybrid_mojito_")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_sobhb_synthesized_from_catalogue_with_real_orbits(self):
        _make_fake_mojito(self.tmp, sobhb_bricks=(0,))
        marker = 7.5
        seen = {}

        def fake_builder(**kwargs):
            seen.update(kwargs)
            N = kwargs["target_N"]
            z = np.zeros((3, N))
            return z, np.full((3, N), marker), z

        with mock.patch.object(
            inj, "build_synthetic_source_streams", side_effect=fake_builder
        ):
            step = _StubHybridStep(
                L1_folder=self.tmp,
                source_ids={"SOBHB": [0, 1]},
                orbits_class=_FakeOrbits,
                verbose=False,
                add_instrument_noise=False,
                synthesize_missing=True,
            )

        # the missing source's row is EXACTLY the branch-side conversion of
        # its catalogue entry
        expected = sobbh_catalogue_to_waveform_basis(
            step.catalogue["SOBHB"][1]
        )
        np.testing.assert_allclose(
            np.atleast_2d(seen["sobbh_injections"])[0], expected
        )
        self.assertEqual(np.atleast_2d(seen["emri_injections"]).shape[0], 0)
        self.assertEqual(np.atleast_2d(seen["mbh_injections"]).shape[0], 0)
        # the run's REAL orbits are threaded into the generators
        self.assertIs(seen["orbits"], step.orbits)
        # mojito epoch for f_low
        from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME

        self.assertEqual(
            seen["sobbh_reference_time"], MOJITO_REFERENCE_TIME
        )
        # grid = the loader's grid
        self.assertEqual(seen["dt"], DT_FAKE)
        self.assertEqual(seen["t_start"], float(step.times[0]))
        # and the synthesized stream landed in the data (brick0 ones +
        # marker)
        np.testing.assert_allclose(
            step.data, np.ones((3, N_FAKE)) + marker
        )
        # provenance recorded
        self.assertEqual(step.synthesized_sources, [("SOBHB", 1)])

    def test_nothing_missing_never_calls_the_builder(self):
        _make_fake_mojito(self.tmp, sobhb_bricks=(0, 1))
        with mock.patch.object(
            inj, "build_synthetic_source_streams"
        ) as spy:
            step = _StubHybridStep(
                L1_folder=self.tmp,
                source_ids={"SOBHB": [0, 1]},
                orbits_class=_FakeOrbits,
                verbose=False,
                add_instrument_noise=False,
                synthesize_missing=True,
            )
        spy.assert_not_called()
        self.assertEqual(step.synthesized_sources, [])
        np.testing.assert_allclose(step.data, 2.0 * np.ones((3, N_FAKE)))

    def test_missing_vgb_brick_is_refused(self):
        # only MBHB/EMRI/SOBHB can be synthesized here; a missing GB/VGB
        # brick must stay a loud failure even with the knob on
        _make_fake_mojito(
            self.tmp, sobhb_bricks=(0, 1), vgb=True, vgb_brick=False
        )
        with self.assertRaises(ValueError):
            _StubHybridStep(
                L1_folder=self.tmp,
                source_ids={"SOBHB": [0, 1], "VGB": [0]},
                orbits_class=_FakeOrbits,
                verbose=False,
                add_instrument_noise=False,
                synthesize_missing=True,
            )


class BuilderOrbitsThreadingTest(unittest.TestCase):
    def test_builder_accepts_and_threads_orbits(self):
        import inspect

        sig = inspect.signature(inj.build_synthetic_source_streams)
        self.assertIn("orbits", sig.parameters)

        sentinel = object()
        captured = {}

        def fake_gen_factory(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        class _FakeWrap:
            def __init__(self, *a, **k):
                pass

            def raw_td(self, *params):
                return np.zeros((3, 8))

        import lisatools.sources.sobbh.response as sobbh_resp
        from lisatools.globalfit.stock.erebor import wrappers as wrap_mod

        with mock.patch.object(
            wrap_mod, "get_sobbh_tdionfly_gen", side_effect=fake_gen_factory
        ), mock.patch.object(
            sobbh_resp, "get_sobbh_tdionfly_gen",
            side_effect=fake_gen_factory,
        ), mock.patch.object(
            wrap_mod, "SOBBHTDIonFlyWaveWrap", _FakeWrap
        ):
            inj.build_synthetic_source_streams(
                Tobs=64.0, dt=8.0, t_start=0.0, target_N=8, nchannels=3,
                force_backend="cpu",
                emri_injections=np.zeros((0, 14)),
                sobbh_injections=np.ones((1, 11)),
                mbh_injections=np.zeros((0, 11)),
                orbits=sentinel,
            )
        self.assertIs(captured.get("orbits"), sentinel)


class VariantKnobTest(unittest.TestCase):
    def test_default_off_and_explicit_kwarg(self):
        from lisatools.globalfit.stock.erebor.variants.all_sources import (
            AllSourcesGeneralSettings,
        )

        self.assertFalse(
            AllSourcesGeneralSettings().synthesize_missing_bricks
        )
        self.assertTrue(
            AllSourcesGeneralSettings(
                synthesize_missing_bricks=True
            ).synthesize_missing_bricks
        )

    def test_env_and_kwargs_threading_in_a_fresh_process(self):
        # Stock env knobs resolve at IMPORT (the variant template is built
        # at module load; every submit script exports before launching), so
        # the env case must run in a subprocess with the env pre-set.
        import subprocess
        import sys

        code = (
            "from lisatools.globalfit.stock import erebor\n"
            "fit = erebor.all_sources(nwalkers=4)\n"
            "assert fit.general.synthesize_missing_bricks is True\n"
            "fit.set_default_processor(fit.general)\n"
            "kw = fit.general.processor_init_kwargs\n"
            "assert kw['synthesize_missing'] is True, kw\n"
            "for k in ('tdi_chan', 'tdi_gen_str', 'synth_force_backend'):\n"
            "    assert k in kw, k\n"
            "assert sorted(kw['mbh_phenom_kwargs']) == ["
            "'buffer_time', 'higher_modes', 'max_freq', 'min_freq',"
            "'phenom_tol', 'response_order', 'start_freq',"
            "'waveform_duration'], kw['mbh_phenom_kwargs']\n"
            "import copy, pickle\n"
            "pickle.dumps(copy.deepcopy(kw))\n"
            "print('HYBRID_KWARGS_OK')\n"
        )
        env = dict(os.environ)
        env["SYNTHESIZE_MISSING_BRICKS"] = "1"
        env.setdefault("OMP_NUM_THREADS", "1")
        out = subprocess.run(
            [sys.executable, "-c", code], env=env,
            capture_output=True, text=True, timeout=600,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn("HYBRID_KWARGS_OK", out.stdout)


if __name__ == "__main__":
    unittest.main()


class CombinedStreamTest(unittest.TestCase):
    """COMBINED data (user ruling 2026-09-14): the all_sources mojito path
    must honor ``general.source_types`` when it lists COMBINED -- the 6mo
    first launch died on an EMRI brick lookup because the variant derived
    its own NOISE+classes list and the env SOURCE_TYPES never reached the
    loader."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="combined_mojito_")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fit(self, **gs_over):
        from lisatools.globalfit.stock import erebor

        fit = erebor.all_sources(nwalkers=4)
        fit.general.mojito_source_ids = {"SOBHB": [0]}
        for k, v in gs_over.items():
            setattr(fit.general, k, v)
        return fit

    def test_variant_combined_passes_source_types_through(self):
        fit = self._fit(
            source_types=("COMBINED", "GB", "VGB", "SOBHB"),
            add_instrument_noise="mojito",
        )
        fit.set_default_processor(fit.general)
        kw = fit.general.processor_init_kwargs
        self.assertEqual(kw["source_types"],
                         ["COMBINED", "GB", "VGB", "SOBHB"])

    def test_variant_without_combined_omits_the_override(self):
        fit = self._fit()
        fit.set_default_processor(fit.general)
        self.assertNotIn("source_types",
                         fit.general.processor_init_kwargs)

    def test_processor_combined_refuses_double_counting(self):
        # synthetic noise / foreground would sum ON TOP of a stream that
        # already contains them -- refused before any file access.
        for bad in (dict(add_instrument_noise="synthetic"),
                    dict(add_instrument_noise=True),
                    dict(add_galactic_foreground=True)):
            with self.assertRaises(ValueError, msg=bad):
                _StubHybridStep(
                    L1_folder=os.path.join(self.tmp, "nonexistent"),
                    source_ids={"SOBHB": [0]},
                    source_types=["COMBINED", "SOBHB"],
                    orbits_class=_FakeOrbits,
                    verbose=False,
                    **bad,
                )

    def test_processor_combined_reads_stream_not_bricks(self):
        _make_fake_mojito(self.tmp, sobhb_ids=(0, 1), sobhb_bricks=(0,))
        d = os.path.join(self.tmp, "data", "COMBINED", "L1")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "mojito_light_test_L1_0_0.h5"), "wb").close()
        # SOBHB brick id 1 does NOT exist -- under COMBINED that must not
        # matter (catalogues only; the waveform is already in the stream).
        step = _StubHybridStep(
            L1_folder=self.tmp,
            source_ids={"SOBHB": [0, 1]},
            source_types=["COMBINED", "SOBHB"],
            orbits_class=_FakeOrbits,
            verbose=False,
            add_instrument_noise="mojito",
        )
        # data is EXACTLY the combined stream (a summed id-0 brick would
        # read 2.0 here), orbits came from the combined file, no brick
        # was recorded missing, both catalogue rows loaded.
        np.testing.assert_allclose(step.data, np.ones((3, N_FAKE)))
        self.assertIn("COMBINED", step.orbits.path)
        self.assertEqual(list(step.missing_source_bricks), [])
        self.assertIn(0, step.catalogue["SOBHB"])
        self.assertIn(1, step.catalogue["SOBHB"])
