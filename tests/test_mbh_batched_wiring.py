"""Knobs, cfg resolution and builder selection for MBH_LIKELIHOOD=batched."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np


class MBHBatchedKnobsTest(unittest.TestCase):
    def test_defaults_leave_the_stock_path(self):
        from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("MBH_LIKELIHOOD", "MBH_BATCH_MAX_SIZE", "MBH_RESPONSE_ORDER",
                      "MBH_WINDOW_BEFORE_DAYS", "MBH_WINDOW_AFTER_DAYS",
                      "MBH_WINDOW_PAD_DAYS", "MBH_WINDOW_MARGIN_DAYS"):
                os.environ.pop(k, None)
            s = SourceMBHSettings()
        self.assertEqual(s.likelihood, "full")
        self.assertEqual(s.batch_max_size, 16)
        self.assertEqual(s.response_order, 8)
        self.assertEqual((s.window_before_days, s.window_after_days), (90.0, 10.0))
        self.assertEqual((s.window_pad_days, s.window_margin_days), (4.0, 1.0))

    def test_env_knobs(self):
        from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

        with mock.patch.dict(os.environ, {"MBH_LIKELIHOOD": "batched", "MBH_BATCH_MAX_SIZE": "8",
                                          "MBH_RESPONSE_ORDER": "30", "MBH_WINDOW_BEFORE_DAYS": "60"}):
            s = SourceMBHSettings()
        self.assertEqual((s.likelihood, s.batch_max_size, s.response_order, s.window_before_days),
                         ("batched", 8, 30, 60.0))


class ResolveBatchedCfgTest(unittest.TestCase):
    def _mbh(self, **kw):
        from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MBH_WAVEFORM_DURATION", None)
            s = SourceMBHSettings()
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    def test_full_path_passes_through(self):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_mbh_batched_cfg

        cfg = resolve_mbh_batched_cfg(self._mbh(likelihood="full"))
        self.assertEqual(cfg["mbh_likelihood"], "full")
        self.assertEqual(cfg["mbh_waveform_duration"], self._mbh().waveform_duration)

    def test_batched_pins_duration_to_the_window(self):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_mbh_batched_cfg

        cfg = resolve_mbh_batched_cfg(self._mbh(likelihood="batched"))
        self.assertEqual(cfg["mbh_waveform_duration"], 90 * 86400.0)
        self.assertEqual(cfg["mbh_window_before"], 90 * 86400.0)
        self.assertEqual(cfg["mbh_window_after"], 10 * 86400.0)
        self.assertEqual(cfg["mbh_window_pad"], 4 * 86400.0)
        self.assertEqual(cfg["mbh_window_margin"], 86400.0)
        self.assertEqual(cfg["mbh_batch_max_size"], 16)

    def test_duration_conflict_raises(self):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_mbh_batched_cfg

        with mock.patch.dict(os.environ, {"MBH_WAVEFORM_DURATION": "2592000"}):
            with self.assertRaises(ValueError):
                resolve_mbh_batched_cfg(self._mbh(likelihood="batched", waveform_duration=2592000.0))

    def test_tdionfly_conflict_raises(self):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_mbh_batched_cfg

        with self.assertRaises(ValueError):
            resolve_mbh_batched_cfg(self._mbh(likelihood="batched", use_tdionfly=True))

    def test_unknown_value_raises(self):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_mbh_batched_cfg

        with self.assertRaises(ValueError):
            resolve_mbh_batched_cfg(self._mbh(likelihood="fast"))

    def test_pad_shorter_than_buffer_time_raises(self):
        # The response zeroes the first ``buffer_time`` of the lattice head;
        # the discarded pad must cover it (controller ruling, Task 3).
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_mbh_batched_cfg

        mbh = self._mbh(likelihood="batched")
        mbh.window_pad_days = 0.5 * mbh.buffer_time / 86400.0
        with self.assertRaises(ValueError):
            resolve_mbh_batched_cfg(mbh)
        # the same pad is fine on the full path (knob unused there)
        mbh.likelihood = "full"
        resolve_mbh_batched_cfg(mbh)
        # and a pad exactly equal to buffer_time is accepted
        mbh.likelihood = "batched"
        mbh.window_pad_days = mbh.buffer_time / 86400.0
        resolve_mbh_batched_cfg(mbh)

    def test_cfg_carries_the_batched_keys(self):
        # source_signal_cfg must route the resolved values (and the pinned
        # duration) into the plain-value cfg the getters/builders read.
        from lisatools.globalfit.stock.erebor.source_runtime import (
            SourceEMRISettings,
            SourceSOBBHSettings,
            source_signal_cfg,
        )

        class _GS:
            tdi_chan = "XYZ"
            tdi_gen_str = "2nd generation"
            nchannels = 3
            data_mode = "synthetic"
            sobbh_reference_time = 0.0
            mbh_waveform_t0 = 0.0
            min_freq = 1e-4
            max_freq = 2.5e-2

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SOBBH_LIKELIHOOD", None)
            sobbh, emri = SourceSOBBHSettings(), SourceEMRISettings()
        cfg = source_signal_cfg(_GS(), self._mbh(likelihood="batched"), sobbh, emri)
        self.assertEqual(cfg["mbh_likelihood"], "batched")
        self.assertEqual(cfg["mbh_window_after"], 10 * 86400.0)
        self.assertEqual(cfg["mbh_phenom_kwargs"]["waveform_duration"], 90 * 86400.0)
        self.assertEqual(cfg["mbh_phenom_kwargs"]["response_order"], 8)


class InjectionDurationTest(unittest.TestCase):
    """Injections and templates agree on the inspiral length (fix round 1)."""

    _NINETY_DAYS = 90 * 86400.0

    def _env(self, **extra):
        env = {"MBH_LIKELIHOOD": "batched", **extra}
        patch = mock.patch.dict(os.environ, env, clear=False)
        patch.start()
        self.addCleanup(patch.stop)
        for k in ("MBH_WAVEFORM_DURATION", "USE_TDIONFLY"):
            os.environ.pop(k, None)

    def test_helper(self):
        from lisatools.globalfit.stock.erebor.source_runtime import (
            SourceMBHSettings,
            mbh_injection_duration,
        )

        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("MBH_WAVEFORM_DURATION", "MBH_LIKELIHOOD", "USE_TDIONFLY"):
                os.environ.pop(k, None)
            s = SourceMBHSettings()
            self.assertEqual(mbh_injection_duration(s), s.waveform_duration)  # full: raw
            s.likelihood = "batched"
            self.assertEqual(mbh_injection_duration(s), self._NINETY_DAYS)

    def test_programmatic_override_warns(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        self._env()
        s = sr.SourceMBHSettings()
        with self.assertNoLogs(sr.logger, level="WARNING"):
            sr.resolve_mbh_batched_cfg(s)  # class default: silent
        s.waveform_duration = self._NINETY_DAYS
        with self.assertNoLogs(sr.logger, level="WARNING"):
            sr.resolve_mbh_batched_cfg(s)  # equal to the window: silent
        s.waveform_duration = 45 * 86400.0
        with self.assertLogs(sr.logger, level="WARNING") as cm:
            cfg = sr.resolve_mbh_batched_cfg(s)
        self.assertEqual(cfg["mbh_waveform_duration"], self._NINETY_DAYS)
        self.assertIn("overridden by the window", cm.output[0])

    def test_all_sources_synthetic_site(self):
        from lisatools.globalfit.stock import erebor

        self._env()
        gs = erebor.get_stock("all_sources", data_mode="synthetic").make_general_settings()
        specs = {cls.__name__: kw for cls, kw in gs.processor_init_kwargs["processor_specs"]}
        pk = specs["SyntheticDataProcessor"]["mbh_phenom_kwargs"]
        self.assertEqual(pk["waveform_duration"], self._NINETY_DAYS)

    def test_all_sources_mojito_synthesize_site(self):
        from lisatools.globalfit.stock import erebor

        self._env()
        gs = erebor.get_stock("all_sources").make_general_settings()
        self.assertEqual(
            gs.processor_init_kwargs["mbh_phenom_kwargs"]["waveform_duration"], self._NINETY_DAYS
        )

    def test_full_year_combined_synthetic_site(self):
        from lisatools.globalfit.stock import erebor

        self._env(MBHB_IDS="0", EMRI_IDS="1", SOBHB_IDS="2")
        gs = erebor.get_stock("full_year_combined", data_mode="synthetic").make_general_settings()
        self.assertEqual(
            gs.processor_init_kwargs["mbh_phenom_kwargs"]["waveform_duration"], self._NINETY_DAYS
        )


class BuilderTest(unittest.TestCase):
    def test_builder_class_attrs(self):
        from lisatools.globalfit.moves import MBHBatchedLikeMove
        from lisatools.globalfit.recipe import MBHBatchedMoveBuilder, MBHMoveBuilder

        self.assertTrue(issubclass(MBHBatchedMoveBuilder, MBHMoveBuilder))
        self.assertIs(MBHBatchedMoveBuilder.move_class, MBHBatchedLikeMove)
        self.assertFalse(MBHBatchedMoveBuilder.use_dcga)

    def test_snap_helper(self):
        from lisatools.globalfit.stock.erebor.source_runtime import snap_waveform_t0_to_lattice

        # offset 0.327664 s = 0.13 samples -> NEAREST lattice point is k = 0
        t0, snap = snap_waveform_t0_to_lattice(97729089.327664, 97729089.0, 2.5)
        self.assertAlmostEqual(t0, 97729089.0, places=6)
        self.assertAlmostEqual(snap, -0.327664, places=6)
        self.assertAlmostEqual((t0 - 97729089.0) / 2.5, round((t0 - 97729089.0) / 2.5), places=9)
        # offset 1.5 s = 0.6 samples -> k = 1, snapped 2.5 s after data_t0
        t0c, snapc = snap_waveform_t0_to_lattice(97729090.5, 97729089.0, 2.5)
        self.assertAlmostEqual(t0c, 97729091.5, places=6)
        self.assertAlmostEqual(snapc, 1.0, places=6)
        t0b, snapb = snap_waveform_t0_to_lattice(100.0, 0.0, 2.5)
        self.assertEqual((t0b, snapb), (100.0, 0.0))


class WindowedGetterTest(unittest.TestCase):
    """``get_mbh_windowed_gen`` with the heavy classes stubbed (no orbits / phentax)."""

    def setUp(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        def _drop():
            for k in [k for k in sr._WAVE_WRAP_CACHE if k and k[0] == "mbh_windowed"]:
                del sr._WAVE_WRAP_CACHE[k]

        _drop()
        self.addCleanup(_drop)

    def _gi(self, domain_settings):
        from types import SimpleNamespace

        return SimpleNamespace(
            gpus=None, orbits=SimpleNamespace(xp=np), gpu_orbits=None,
            domain_settings=domain_settings, data_td_settings=object(),
            data_t0=97729089.0, dt=2.5, window_alpha=0.0, force_backend="cpu",
        )

    def _cfg(self):
        return dict(
            nchannels=3, tdi_gen_str="2nd generation", tdi_chan="XYZ",
            mbh_waveform_t0=97729090.5,
            mbh_phenom_kwargs=dict(
                waveform_duration=90 * 86400.0, higher_modes=(21, 33, 44), phenom_tol=1e-12,
                start_freq=7e-5, response_order=8, buffer_time=15000.0,
                min_freq=1e-4, max_freq=2.5e-2,
            ),
        )

    def test_adapter_carries_snapped_epoch_and_snap(self):
        from lisatools.domains import WDMSettings
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        wdm = mock.MagicMock(spec=WDMSettings)
        gi = self._gi(wdm)
        with mock.patch("lisatools.sources.bbh.gridaligned.WindowedGridAlignedMBHWaveform") as W, \
                mock.patch("lisatools.sources.batching.MBHWindowedWDMSignalGen") as A:
            A.side_effect = lambda gen, *a, **k: mock.MagicMock(spec=["wave_gen"], wave_gen=gen)
            adapter = sr.get_mbh_windowed_gen(gi, self._cfg())
            again = sr.get_mbh_windowed_gen(gi, self._cfg())
        self.assertIs(again, adapter)
        self.assertEqual(W.call_count, 1)
        kw = W.call_args.kwargs
        # offset 1.5 s = 0.6 samples -> nearest lattice point k = 1
        self.assertAlmostEqual(kw["waveform_t0"], 97729091.5, places=6)
        self.assertAlmostEqual(adapter.waveform_t0, 97729091.5, places=6)
        self.assertAlmostEqual(adapter.t_plunge_snap, 1.0, places=6)
        self.assertEqual(kw["Tobs"], 90 * 86400.0)
        self.assertEqual(kw["order"], 8)
        self.assertEqual(kw["buffer_time"], 15000.0)
        self.assertIs(kw["output_domain_settings"], wdm)
        # the absolute time of WDM layer 0 is the DATA START, never the
        # settings' t0 (0 from the stock factory, or data_t0 once a GB comp
        # build has mutated it -- build-order dependent)
        self.assertEqual(A.call_args.kwargs["t0_abs"], 97729089.0)

    def test_non_wdm_domain_raises(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        with self.assertRaises(ValueError):
            sr.get_mbh_windowed_gen(self._gi(object()), self._cfg())


class SnappedStockGenTest(unittest.TestCase):
    """MBH_LIKELIHOOD=batched: the STOCK generator (engine residual rebuilds
    and the move's cross-check) gets the same lattice-snapped epoch as the
    windowed one, with t_plunge shifted by the snap -- the same absolute
    merger. Measured 2026-09-30 (mojito id 17, SNR 1420): against the
    UNSNAPPED stock the near-truth rows differ by up to 0.59 nats (> the
    0.5-nat check tolerance); against the snapped stock by 1.5e-3."""

    GI = WindowedGetterTest._gi
    CFG = WindowedGetterTest._cfg

    def setUp(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        def _drop():
            for k in [k for k in sr._WAVE_WRAP_CACHE if k and k[0] == "mbh_snapped_stock"]:
                del sr._WAVE_WRAP_CACHE[k]

        _drop()
        self.addCleanup(_drop)

    def _call(self, mode):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        class _Gen:
            def __init__(self, **kw):
                self.waveform_t0 = kw["waveform_t0"]
                self.seen = []
                self.order = 8

            def get_signals_for_residuals(self, *args, **kw):
                self.seen.append(("gsr", args, kw))
                return "tmpl"

            def compute_tdi_channels(self, *args, **kw):
                self.seen.append(("tdi", args, kw))
                return "tdi"

        built = {}

        def fake_getter(**kw):
            key = kw["waveform_t0"]
            if key not in built:
                built[key] = _Gen(**kw)
            return built[key]

        cfg = self.CFG()
        cfg["mbh_likelihood"] = mode
        gi = self.GI(object())
        gi.Tobs = 120 * 86400.0
        with mock.patch.object(sr, "get_mbh_phenom_wave_gen", side_effect=fake_getter):
            gen = sr.get_mbh_phenom_gen(gi, cfg)
            again = sr.get_mbh_phenom_gen(gi, cfg)
        return gen, again, built

    def test_batched_stock_runs_on_the_snapped_epoch(self):
        gen, again, built = self._call("batched")
        self.assertIs(again, gen)
        (t0_built,) = built
        # offset 1.5 s = 0.6 samples -> nearest lattice point: snap +1.0 s
        self.assertAlmostEqual(t0_built, 97729091.5, places=6)
        self.assertAlmostEqual(gen.t_plunge_snap, 1.0, places=9)
        self.assertAlmostEqual(gen.waveform_t0, 97729090.5, places=6)   # rows' epoch
        self.assertEqual(gen.order, 8)                                    # forwards
        inner = built[t0_built]
        row = np.arange(11, dtype=float) + 100.0
        self.assertEqual(gen.get_signals_for_residuals(*row), "tmpl")
        args = inner.seen[-1][1]
        self.assertEqual(args[10], row[10] - 1.0)
        np.testing.assert_array_equal(args[:10], row[:10])
        # the absolute merger is unchanged
        self.assertAlmostEqual(t0_built + args[10], 97729090.5 + row[10], places=6)
        gen.compute_tdi_channels(*row[:8], ra=1.0, dec=0.2, merger_time=np.array([5.0, 6.0]))
        np.testing.assert_array_equal(inner.seen[-1][2]["merger_time"], [4.0, 5.0])

    def test_full_path_keeps_the_unsnapped_stock(self):
        gen, _, built = self._call("full")
        (t0_built,) = built
        self.assertAlmostEqual(t0_built, 97729090.5, places=6)
        self.assertIs(gen, built[t0_built])


class RuntimeSelectionTest(unittest.TestCase):
    """``build_mbh_move_runtime`` routes ``mbh_likelihood=batched`` to the new builder."""

    def _cfg(self, mode):
        return dict(
            mbh_likelihood=mode, mbh_use_tdionfly=False, mbh_batch_max_size=4,
            mbh_window_before=1.0, mbh_window_after=2.0, mbh_window_pad=3.0,
            mbh_window_margin=4.0,
        )

    def test_batched_selects_the_batched_builder(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        curr = mock.MagicMock()
        built = {}

        class _FakeBuilder:
            def __init__(self, **kw):
                built.update(kw)

            def build(self, *a):
                return [], ["the-move"]

        with mock.patch.object(sr, "MBHBatchedMoveBuilder", _FakeBuilder), \
                mock.patch.object(sr, "get_mbh_phenom_gen") as slow_getter, \
                mock.patch.object(sr, "build_mbh_moves_phenom") as stock:
            move = sr.build_mbh_move_runtime(curr, None, None, None, self._cfg("batched"))
        self.assertEqual(move, "the-move")
        stock.assert_not_called()
        self.assertEqual(
            (built["batch_max_size"], built["window_before"], built["window_after"],
             built["window_pad"], built["window_margin"]),
            (4, 1.0, 2.0, 3.0, 4.0),
        )
        self.assertIsInstance(built["batched_gen"], sr.DeviceLocalWaveGen)
        self.assertIs(built["batched_gen"]._getter, sr.get_mbh_windowed_gen)
        # wave_gen = the SLOW stock generator's residual method, late-bound per device
        self.assertEqual(built["wave_gen"].__name__, "get_signals_for_residuals")
        self.assertIs(built["wave_gen"].__self__, slow_getter.return_value)

    def test_full_keeps_the_stock_builder(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        curr = mock.MagicMock()
        with mock.patch.object(sr, "MBHBatchedMoveBuilder") as batched, \
                mock.patch.object(sr, "build_mbh_moves_phenom",
                                  return_value=(None, "stock-move")):
            move = sr.build_mbh_move_runtime(curr, None, None, None, self._cfg("full"))
        self.assertEqual(move, "stock-move")
        batched.assert_not_called()


class MergerWindowEdgeTest(unittest.TestCase):
    """Merger-window filter + ``t_plunge`` prior (user 2026-09-30: keep an MBH
    whose merger lands up to ~7 d after the data end; admitted => in prior)."""

    DAY = 86400.0
    DATA_T0 = 1000.0 * 86400.0  # absolute data start
    WF_T0 = 990.0 * 86400.0     # mbh_waveform_t0 (t_plunge epoch)
    TOBS = 120.0 * 86400.0

    def test_default_buffer_is_seven_days(self):
        from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MBH_MERGER_TIME_BUFFER", None)
            s = SourceMBHSettings()
        self.assertEqual(s.mbh_merger_time_buffer, 7 * 86400.0)

    def _prepare(self, merger_days_after_start):
        """Run ``prepare_mbh_branch`` on a stub catalogue; merger times are
        given in days after the DATA start. Returns (mbh, admitted t_plunge)."""
        from types import SimpleNamespace

        from lisatools.globalfit.stock.erebor import source_runtime as sr

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MBH_MERGER_TIME_BUFFER", None)
            mbh = sr.SourceMBHSettings()
        mbh.initialize_kwargs = {}
        mbh.inner_moves = []
        t_rel0 = self.DATA_T0 - self.WF_T0
        rows = {
            i: np.r_[np.zeros(10), t_rel0 + d * self.DAY]
            for i, d in enumerate(merger_days_after_start)
        }
        gsetup = SimpleNamespace(data_t0=self.DATA_T0, Tobs=self.TOBS)
        gs = SimpleNamespace(
            n_injections={"MBHB": len(rows)}, data_mode="mojito",
            mbh_waveform_t0=self.WF_T0,
        )
        with mock.patch.object(sr, "source_catalogue", return_value=rows), \
                mock.patch.object(sr, "mbh_catalogue_to_sampling_basis",
                                  side_effect=lambda r: r):
            out = sr.prepare_mbh_branch(mbh, gsetup, gs)
        return out, (np.asarray(out.injection)[:, -1] - t_rel0) / self.DAY

    def test_filter_keeps_mergers_up_to_seven_days_past_the_end(self):
        # 3 d before the end, 1 d / 6 d after the end kept; 8 d after dropped.
        mbh, kept_days = self._prepare([3.0, 117.0, 121.0, 126.0, 128.0])
        np.testing.assert_allclose(kept_days, [3.0, 117.0, 121.0, 126.0])
        self.assertEqual((mbh.nleaves_min, mbh.nleaves_max), (4, 4))

    def test_filter_drops_mergers_before_the_data_start(self):
        # a merger before the data start leaves no inspiral in the data and
        # would sit below the prior's lower edge; one AT the start is kept
        mbh, kept_days = self._prepare([-1.0, 0.0, 3.0])
        np.testing.assert_allclose(kept_days, [0.0, 3.0])
        self.assertEqual((mbh.nleaves_min, mbh.nleaves_max), (2, 2))

    def test_admitted_sources_are_inside_the_t_plunge_prior(self):
        mbh, _ = self._prepare([0.0, 3.0, 117.0, 121.0, 126.0])
        t_plunge = np.asarray(mbh.injection)[:, -1]
        dist = mbh.priors["mbh"].priors_in["t_plunge"]
        lp = np.asarray(dist.logpdf(t_plunge))
        self.assertTrue(np.all(np.isfinite(lp)), lp)
        # upper edge = data end + buffer + t_plunge_pad (headroom for the
        # start-walker scatter of a merger just under end + buffer)
        t_rel0 = self.DATA_T0 - self.WF_T0
        self.assertAlmostEqual(
            float(dist.maximum),
            t_rel0 + self.TOBS + 7 * self.DAY + float(mbh.t_plunge_pad),
        )
        self.assertGreater(float(mbh.t_plunge_pad), 0.0)
        self.assertAlmostEqual(float(dist.minimum), t_rel0)


if __name__ == "__main__":
    unittest.main()
