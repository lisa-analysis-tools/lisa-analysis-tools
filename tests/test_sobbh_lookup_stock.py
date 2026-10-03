# tests/test_sobbh_lookup_stock.py
"""SOBBH_LIKELIHOOD=lookup: settings knobs, cfg plumbing, comp construction and the dispatch."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wdm_lookup_toy import build_tiny_table  # noqa: E402

#: Every fake general_info stays alive for the whole module: the stock wave-wrap cache is keyed on
#: ``id(general_info)`` (a production object lives for the process), so a freed fixture whose id
#: is reused by the next test would hand that test the previous test's cached comp.
_KEEP_ALIVE = []


def _general_info(domain):
    from lisatools.detector import EqualArmlengthOrbits

    gi = SimpleNamespace(
        gpus=None,
        orbits=EqualArmlengthOrbits(force_backend="cpu"),
        gpu_orbits=None,
        domain_settings=domain,
        force_backend="cpu",
        data_t0=0.0,
    )
    _KEEP_ALIVE.append(gi)
    return gi


def _cfg(path, likelihood="lookup"):
    return dict(
        nchannels=3,
        tdi_gen_str="2nd generation",
        tdi_chan="XYZ",
        sobbh_reference_time=None,
        sobbh_n_grid=64,
        sobbh_buffer_time=5000.0,
        sobbh_likelihood=likelihood,
        sobbh_lookup_table_path=path,
        sobbh_lookup_eval_dt=600.0,
        sobbh_lookup_num_m_layers=2,
        sobbh_lookup_interp="spline",
        sobbh_lookup_row_batch=32,
        sobbh_lookup_kernel="auto",
        sobbh_nt_sub=32,
        sobbh_chirp_fdot_max=None,
        sobbh_sweep_max_layers=3.5,
        sobbh_n_sparse=256,
        sobbh_n_pad=4,
        sobbh_m_band_half_width=1,
        sobbh_fill_m_band_half_width=8,
    )


class LookupSettingsTest(unittest.TestCase):
    def test_fields_and_env_knobs(self):
        from lisatools.globalfit.stock.erebor.source_runtime import SourceSOBBHSettings

        # the hard defaults: evaluated with the knobs ABSENT from the environment (a cluster
        # shell that exported SOBBH_LOOKUP_TABLE_PATH for a run would otherwise leak in here)
        with mock.patch.dict(os.environ):
            for key in (
                "SOBBH_LIKELIHOOD",
                "SOBBH_LOOKUP_TABLE_PATH",
                "SOBBH_LOOKUP_EVAL_DT",
                "SOBBH_LOOKUP_NUM_M_LAYERS",
                "SOBBH_LOOKUP_INTERP",
                "SOBBH_LOOKUP_ROW_BATCH",
                "SOBBH_LOOKUP_KERNEL",
            ):
                os.environ.pop(key, None)
            s = SourceSOBBHSettings()
        self.assertEqual(s.likelihood, "lookup")  # the stock default (2026-10-03)
        self.assertEqual(s.lookup_table_path, "")
        self.assertEqual(s.lookup_num_m_layers, 2)
        self.assertEqual(s.lookup_eval_dt, 43200.0)  # 12 h: one response point per 12 pixels
        # the table's uniform cubic B-spline: the fused lookup kernels' semantics, and equal
        # or better than Keys cubic on the 6-month gate (sources 3 / 5: mm 2.6e-7 / 7e-8 vs
        # 4.0e-7 / 1.5e-7)
        self.assertEqual(s.lookup_interp, "spline")
        self.assertEqual(s.lookup_row_batch, 32)
        self.assertEqual(s.lookup_kernel, "auto")
        env = {
            "SOBBH_LIKELIHOOD": "lookup",
            "SOBBH_LOOKUP_TABLE_PATH": "/x/y.h5",
            "SOBBH_LOOKUP_EVAL_DT": "300",
            "SOBBH_LOOKUP_NUM_M_LAYERS": "3",
            "SOBBH_LOOKUP_INTERP": "linear",
            "SOBBH_LOOKUP_ROW_BATCH": "8",
            "SOBBH_LOOKUP_KERNEL": "python",
        }
        with mock.patch.dict(os.environ, env):
            s2 = SourceSOBBHSettings()
        self.assertEqual(
            (
                s2.likelihood,
                s2.lookup_table_path,
                s2.lookup_eval_dt,
                s2.lookup_num_m_layers,
                s2.lookup_interp,
                s2.lookup_row_batch,
                s2.lookup_kernel,
            ),
            ("lookup", "/x/y.h5", 300.0, 3, "linear", 8, "python"),
        )

    def test_cfg_carries_lookup_knobs(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        s = sr.SourceSOBBHSettings(
            likelihood="lookup", lookup_table_path="/t.h5", lookup_eval_dt=120.0
        )
        gs = SimpleNamespace(
            tdi_chan="XYZ",
            tdi_gen_str="2nd generation",
            nchannels=3,
            data_mode="synthetic",
            sobbh_reference_time=None,
            mbh_waveform_t0=0.0,
            min_freq=1e-4,
            max_freq=2.5e-2,
        )
        mbh = SimpleNamespace(
            use_tdionfly=False,
            tdionfly_margin=0.0,
            waveform_duration=0.0,
            higher_modes=False,
            phenom_tol=0.0,
            start_freq=0.0,
            response_order=40,
            buffer_time=0.0,
        )
        emri = SimpleNamespace(response_order=40)
        # the MBH scoring-path resolution (resolve_mbh_batched_cfg) reads the full MBH block
        # and the run-domain spec; neither is this test's subject -> stub it out
        with (
            mock.patch.object(sr, "apply_emri_mode_selection_threshold", return_value=1e-3),
            mock.patch.object(
                sr, "resolve_mbh_batched_cfg", return_value={"mbh_waveform_duration": 0.0}
            ),
        ):
            cfg = sr.source_signal_cfg(gs, mbh, s, emri, domain_settings=None)
        self.assertEqual(cfg["sobbh_likelihood"], "lookup")
        self.assertEqual(cfg["sobbh_lookup_table_path"], "/t.h5")
        self.assertEqual(cfg["sobbh_lookup_eval_dt"], 120.0)
        self.assertEqual(cfg["sobbh_lookup_num_m_layers"], 2)
        self.assertEqual(cfg["sobbh_lookup_interp"], "spline")
        self.assertEqual(cfg["sobbh_lookup_row_batch"], 32)
        self.assertEqual(cfg["sobbh_lookup_kernel"], "auto")


class LookupCompBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wdm, cls.table = build_tiny_table(cls.tmp.name)
        cls.path = cls.table.store_path

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_no_path_and_no_run_folder_raises(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        gi = _general_info(self.wdm)  # no file_store_dir: nowhere to keep a built table
        with self.assertRaises(ValueError) as cm:
            sr.get_sobbh_lookup_comp(gi, _cfg(""))
        msg = str(cm.exception)
        self.assertIn("SOBBH_LOOKUP_TABLE_PATH", msg)
        self.assertIn("file_store_dir", msg)

    def test_builds_lookup_comp_and_caches_it(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr
        from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations

        gi = _general_info(self.wdm)
        comp = sr.get_sobbh_fast_comp(gi, _cfg(self.path))
        self.assertIsInstance(comp.primary, SOBBHLookupComputations)  # behind the device router
        self.assertIs(sr.get_sobbh_fast_comp(gi, _cfg(self.path)), comp)
        self.assertEqual(comp.direct.num_m_layers, 2)
        self.assertEqual(comp.direct.tof.eval_dt, 600.0)
        self.assertEqual(comp.d_d, 0.0)
        self.assertEqual(comp.kernel, "auto")

    def test_kernel_knob_reaches_the_comp(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        cfg = dict(_cfg(self.path), sobbh_lookup_kernel="python")
        comp = sr.get_sobbh_lookup_comp(_general_info(self.wdm), cfg)
        self.assertEqual(comp.kernel, "python")
        self.assertFalse(comp.uses_kernel)

    def test_dispatch_names(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        self.assertEqual(sr.SOBBH_FAST_LIKELIHOODS, ("chunked", "lookup"))
        with self.assertRaises(ValueError):
            sr.get_sobbh_fast_comp(_general_info(self.wdm), _cfg(self.path, likelihood="full"))


class LookupTableResolutionTest(unittest.TestCase):
    """The table during a run, the EMRI way: SOBBH_LOOKUP_TABLE_PATH when set (any sampling
    with the run's layer duration), else the run folder's canonical n_ref table (the EMRI recipe,
    one file shared with EMRI_LIKELIHOOD=direct), built and saved there when missing."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wdm, cls.table = build_tiny_table(cls.tmp.name)  # Nf=64, dt=56.25: layer 3600 s
        cls.path = cls.table.store_path

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run_folder(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        return d.name

    def _canonical(self, folder):
        from lisatools.wdm_lookup_store import lookup_table_name

        return os.path.join(folder, lookup_table_name(int(self.wdm.Nf), float(self.wdm.data_dt)))

    def _stub_build(self):
        """A build that copies the tiny table into place (the store's own tests cover the real
        build); records every call."""
        import shutil

        calls = []

        def build(path, *, Nf, dt, force_backend="cpu", recipe=None, verbose=False):
            calls.append((path, int(Nf), float(dt)))
            shutil.copyfile(self.path, path)
            return path

        return calls, build

    def test_run_folder_canonical_table_is_found(self):
        import shutil

        from lisatools.globalfit.stock.erebor import source_runtime as sr
        from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations

        folder = self._run_folder()
        canonical = self._canonical(folder)
        shutil.copyfile(self.path, canonical)
        gi = _general_info(self.wdm)
        gi.file_store_dir = folder
        calls, build = self._stub_build()
        with mock.patch("lisatools.wdm_lookup_store.build_lookup_table", build):
            path, status = sr.resolve_sobbh_lookup_table(gi, _cfg(""))
            comp = sr.get_sobbh_lookup_comp(gi, _cfg(""))
        self.assertEqual((path, status), (canonical, "found"))
        self.assertEqual(calls, [])
        self.assertIsInstance(comp, SOBBHLookupComputations)

    def test_missing_canonical_table_is_built_and_saved_in_the_run_folder(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        folder = self._run_folder()
        canonical = self._canonical(folder)
        gi = _general_info(self.wdm)
        gi.file_store_dir = folder
        calls, build = self._stub_build()
        with mock.patch("lisatools.wdm_lookup_store.build_lookup_table", build):
            path, status = sr.resolve_sobbh_lookup_table(gi, _cfg(""))
            self.assertEqual((path, status), (canonical, "built"))
            self.assertEqual(calls, [(canonical, int(self.wdm.Nf), float(self.wdm.data_dt))])
            self.assertTrue(os.path.exists(canonical))
            # a restart (another general_info) finds it: no second build
            gi2 = _general_info(self.wdm)
            gi2.file_store_dir = folder
            self.assertEqual(sr.resolve_sobbh_lookup_table(gi2, _cfg("")), (canonical, "found"))
            self.assertEqual(len(calls), 1)
            sr.get_sobbh_lookup_comp(gi2, _cfg(""))

    def test_explicit_path_wins_over_the_run_folder(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        folder = self._run_folder()
        gi = _general_info(self.wdm)
        gi.file_store_dir = folder
        calls, build = self._stub_build()
        with mock.patch("lisatools.wdm_lookup_store.build_lookup_table", build):
            path, status = sr.resolve_sobbh_lookup_table(gi, _cfg(self.path))
            comp = sr.get_sobbh_lookup_comp(gi, _cfg(self.path))
        self.assertEqual((path, status), (os.path.abspath(self.path), "explicit"))
        self.assertEqual(calls, [])
        self.assertFalse(os.path.exists(self._canonical(folder)))
        self.assertEqual(comp.direct.ev.layer_dt, float(self.wdm.layer_dt))

    def test_explicit_path_must_exist(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        gi = _general_info(self.wdm)
        with self.assertRaises(ValueError) as cm:
            sr.resolve_sobbh_lookup_table(gi, _cfg(os.path.join(self.tmp.name, "nope.h5")))
        self.assertIn("SOBBH_LOOKUP_TABLE_PATH", str(cm.exception))

    def test_explicit_table_is_checked_against_the_run_layer_duration(self):
        from lisatools.domains import WDMSettings
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        # another sampling with the SAME layer duration (Nf * dt = 3600 s) is served
        same_layer = WDMSettings(128, 64, 28.125, force_backend="cpu")
        self.assertAlmostEqual(float(same_layer.layer_dt), float(self.wdm.layer_dt))
        sr.get_sobbh_lookup_comp(_general_info(same_layer), _cfg(self.path))
        # a different layer duration is refused, naming the mismatch
        other_layer = WDMSettings(64, 128, 50.0, force_backend="cpu")
        with self.assertRaises(ValueError) as cm:
            sr.get_sobbh_lookup_comp(_general_info(other_layer), _cfg(self.path))
        self.assertIn("layer", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
