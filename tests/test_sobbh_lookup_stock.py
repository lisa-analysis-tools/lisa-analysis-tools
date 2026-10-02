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
        sobbh_lookup_interp="cubic",
        sobbh_lookup_row_batch=32,
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
            ):
                os.environ.pop(key, None)
            s = SourceSOBBHSettings(likelihood="lookup")
        self.assertEqual(s.likelihood, "lookup")
        self.assertEqual(s.lookup_table_path, "")
        self.assertEqual(s.lookup_num_m_layers, 2)
        self.assertEqual(s.lookup_eval_dt, 600.0)
        self.assertEqual(s.lookup_interp, "cubic")
        self.assertEqual(s.lookup_row_batch, 32)
        env = {
            "SOBBH_LIKELIHOOD": "lookup",
            "SOBBH_LOOKUP_TABLE_PATH": "/x/y.h5",
            "SOBBH_LOOKUP_EVAL_DT": "300",
            "SOBBH_LOOKUP_NUM_M_LAYERS": "3",
            "SOBBH_LOOKUP_INTERP": "linear",
            "SOBBH_LOOKUP_ROW_BATCH": "8",
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
            ),
            ("lookup", "/x/y.h5", 300.0, 3, "linear", 8),
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
        self.assertEqual(cfg["sobbh_lookup_interp"], "cubic")
        self.assertEqual(cfg["sobbh_lookup_row_batch"], 32)


class LookupCompBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wdm, cls.table = build_tiny_table(cls.tmp.name)
        cls.path = cls.table.store_path

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_missing_table_raises_with_builder_hint(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        with self.assertRaises(ValueError) as cm:
            sr.get_sobbh_lookup_comp(_general_info(self.wdm), _cfg(""))
        msg = str(cm.exception)
        self.assertIn("SOBBH_LOOKUP_TABLE_PATH", msg)
        self.assertIn("build_wdm_lookup_gpu", msg)

    def test_builds_lookup_comp_and_caches_it(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr
        from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations

        gi = _general_info(self.wdm)
        comp = sr.get_sobbh_fast_comp(gi, _cfg(self.path))
        self.assertIsInstance(comp, SOBBHLookupComputations)
        self.assertIs(sr.get_sobbh_fast_comp(gi, _cfg(self.path)), comp)
        self.assertEqual(comp.direct.num_m_layers, 2)
        self.assertEqual(comp.direct.tof.eval_dt, 600.0)
        self.assertEqual(comp.d_d, 0.0)

    def test_multi_gpu_run_is_refused(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        gi = _general_info(self.wdm)
        gi.gpus = [0, 1]
        with self.assertRaises(ValueError) as cm:
            sr.get_sobbh_lookup_comp(gi, _cfg(self.path))
        self.assertIn("single-device", str(cm.exception))

    def test_dispatch_names(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        self.assertEqual(sr.SOBBH_FAST_LIKELIHOODS, ("chunked", "lookup"))
        with self.assertRaises(ValueError):
            sr.get_sobbh_fast_comp(_general_info(self.wdm), _cfg(self.path, likelihood="full"))


if __name__ == "__main__":
    unittest.main()
