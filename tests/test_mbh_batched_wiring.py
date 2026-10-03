"""Knobs, cfg resolution and builder selection for the MBH scoring path.

Default since 2026-09-30: ``MBH_LIKELIHOOD=auto`` -> the batched windowed
path on every WDM, legacy-response run without a conflicting
``MBH_WAVEFORM_DURATION``; ``full`` otherwise, with one INFO line naming why."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

_SR_LOGGER = "lisatools.globalfit.stock.erebor.source_runtime"


def _wdm_spec():
    """The stock run-domain spec: a ``WDMSettings.make_factory`` factory."""
    from lisatools.domains import WDMSettings

    return WDMSettings.make_factory(Nf=32, Nt=128)


def _fd_spec():
    from lisatools.domains import FDSettings

    return FDSettings.make_factory()


class MBHBatchedKnobsTest(unittest.TestCase):
    def test_defaults(self):
        # 2026-09-30: likelihood "full" -> "auto" (the batched path is the
        # default wherever the run allows it); batch 16 -> 8 (H100 ruling).
        from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("MBH_LIKELIHOOD", "MBH_BATCH_MAX_SIZE", "MBH_RESPONSE_ORDER",
                      "MBH_WINDOW_BEFORE_DAYS", "MBH_WINDOW_AFTER_DAYS",
                      "MBH_WINDOW_PAD_DAYS", "MBH_WINDOW_MARGIN_DAYS"):
                os.environ.pop(k, None)
            s = SourceMBHSettings()
        self.assertEqual(s.likelihood, "auto")
        self.assertEqual(s.batch_max_size, 8)
        self.assertEqual(s.response_order, 8)
        self.assertEqual((s.window_before_days, s.window_after_days), (90.0, 10.0))
        self.assertEqual((s.window_pad_days, s.window_margin_days), (4.0, 1.0))

    def test_move_ctor_batch_default_matches_the_knob(self):
        import inspect

        from lisatools.globalfit.moves import MBHBatchedLikeMove

        p = inspect.signature(MBHBatchedLikeMove.__init__).parameters["batch_max_size"]
        self.assertEqual(p.default, 8)

    def test_settings_survive_deepcopy_and_pickle(self):
        import copy
        import pickle

        from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

        s = SourceMBHSettings()
        t = pickle.loads(pickle.dumps(copy.deepcopy(s)))
        self.assertEqual((t.likelihood, t.batch_max_size), (s.likelihood, s.batch_max_size))

    def test_env_knobs(self):
        from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

        with mock.patch.dict(os.environ, {"MBH_LIKELIHOOD": "batched", "MBH_BATCH_MAX_SIZE": "8",
                                          "MBH_RESPONSE_ORDER": "30", "MBH_WINDOW_BEFORE_DAYS": "60"}):
            s = SourceMBHSettings()
        self.assertEqual((s.likelihood, s.batch_max_size, s.response_order, s.window_before_days),
                         ("batched", 8, 30, 60.0))


def _mbh(**kw):
    """SourceMBHSettings with no MBH env knobs leaking in, then ``kw`` set."""
    from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

    with mock.patch.dict(os.environ, {}, clear=False):
        for k in ("MBH_WAVEFORM_DURATION", "MBH_LIKELIHOOD", "USE_TDIONFLY",
                  "MBH_BATCH_MAX_SIZE", "MBH_WINDOW_DECIMATE"):
            os.environ.pop(k, None)
        s = SourceMBHSettings()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class _NoDurationEnv:
    """Mixin: MBH_WAVEFORM_DURATION unset for the whole test."""

    def setUp(self):
        patch = mock.patch.dict(os.environ, {}, clear=False)
        patch.start()
        self.addCleanup(patch.stop)
        os.environ.pop("MBH_WAVEFORM_DURATION", None)


class ResolveBatchedCfgTest(_NoDurationEnv, unittest.TestCase):
    _mbh = staticmethod(_mbh)

    def _resolve(self, mbh, domain=None):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_mbh_batched_cfg

        return resolve_mbh_batched_cfg(
            mbh, domain_settings=_wdm_spec() if domain is None else domain
        )

    def test_full_path_passes_through(self):
        cfg = self._resolve(self._mbh(likelihood="full"))
        self.assertEqual(cfg["mbh_likelihood"], "full")
        self.assertEqual(cfg["mbh_waveform_duration"], self._mbh().waveform_duration)

    def test_full_ignores_every_batched_blocker(self):
        # explicit full: no raise, no log, raw duration -- even on FD + tdionfly
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        with self.assertNoLogs(sr.logger, level="INFO"):
            cfg = self._resolve(self._mbh(likelihood="full", use_tdionfly=True), _fd_spec())
        self.assertEqual(cfg["mbh_likelihood"], "full")

    def test_batched_pins_duration_to_the_window(self):
        cfg = self._resolve(self._mbh(likelihood="batched"))
        self.assertEqual(cfg["mbh_waveform_duration"], 90 * 86400.0)
        self.assertEqual(cfg["mbh_window_before"], 90 * 86400.0)
        self.assertEqual(cfg["mbh_window_after"], 10 * 86400.0)
        self.assertEqual(cfg["mbh_window_pad"], 4 * 86400.0)
        self.assertEqual(cfg["mbh_window_margin"], 86400.0)
        self.assertEqual(cfg["mbh_batch_max_size"], 8)
        self.assertEqual(cfg["mbh_window_decimate"], 1)

    def test_window_decimate_reaches_the_batched_cfg_only(self):
        """MBH_WINDOW_DECIMATE (``window_decimate``) rides the batched cfg; the full
        path has nothing to decimate (1); a factor < 1 is refused."""
        cfg = self._resolve(self._mbh(likelihood="batched", window_decimate=2))
        self.assertEqual(cfg["mbh_window_decimate"], 2)
        cfg = self._resolve(self._mbh(likelihood="full", window_decimate=2))
        self.assertEqual(cfg["mbh_window_decimate"], 1)
        with self.assertRaisesRegex(ValueError, "MBH_WINDOW_DECIMATE"):
            self._resolve(self._mbh(likelihood="batched", window_decimate=0))
        with mock.patch.dict(os.environ, {"MBH_WINDOW_DECIMATE": "4"}):
            from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

            self.assertEqual(SourceMBHSettings().window_decimate, 4)

    def test_domain_spec_is_required(self):
        # every site must say which domain it resolves for (injections and
        # moves would otherwise be free to disagree)
        from lisatools.globalfit.stock.erebor.source_runtime import (
            mbh_injection_duration,
            resolve_mbh_batched_cfg,
        )

        with self.assertRaises(TypeError):
            resolve_mbh_batched_cfg(self._mbh())
        with self.assertRaises(TypeError):
            mbh_injection_duration(self._mbh())

    # -- explicit batched keeps raising on every conflict ----------------------
    def test_duration_conflict_raises(self):
        with mock.patch.dict(os.environ, {"MBH_WAVEFORM_DURATION": "2592000"}):
            with self.assertRaisesRegex(ValueError, "MBH_WAVEFORM_DURATION=2592000"):
                self._resolve(self._mbh(likelihood="batched", waveform_duration=2592000.0))

    def test_tdionfly_conflict_raises(self):
        with self.assertRaisesRegex(ValueError, "USE_TDIONFLY"):
            self._resolve(self._mbh(likelihood="batched", use_tdionfly=True))

    def test_non_wdm_domain_raises_for_explicit_batched(self):
        from lisatools.domains import FDSettings

        with self.assertRaisesRegex(ValueError, "FDSettings, not WDM"):
            self._resolve(self._mbh(likelihood="batched"), _fd_spec())
        with self.assertRaisesRegex(ValueError, "FDSettings, not WDM"):
            self._resolve(
                self._mbh(likelihood="batched"),
                FDSettings(N=33, df=1.0 / 640.0, force_backend="cpu"),
            )

    def test_explicit_batched_takes_an_unidentifiable_domain_at_its_word(self):
        # a hand-written factory: the windowed getter checks the BUILT domain
        def custom_factory(times, dt, force_backend):  # pragma: no cover - never called
            raise AssertionError

        cfg = self._resolve(self._mbh(likelihood="batched"), custom_factory)
        self.assertEqual(cfg["mbh_likelihood"], "batched")

    def test_unknown_value_raises(self):
        for bad in ("fast", "Batched", "", "chunked"):
            with self.assertRaisesRegex(ValueError, "auto"):
                self._resolve(self._mbh(likelihood=bad))

    def test_pad_shorter_than_buffer_time_raises(self):
        # The response zeroes the first ``buffer_time`` of the lattice head;
        # the discarded pad must cover it (controller ruling, Task 3).
        mbh = self._mbh(likelihood="batched")
        mbh.window_pad_days = 0.5 * mbh.buffer_time / 86400.0
        with self.assertRaises(ValueError):
            self._resolve(mbh)
        # the same pad is fine on the full path (knob unused there)
        mbh.likelihood = "full"
        self._resolve(mbh)
        # and a pad exactly equal to buffer_time is accepted
        mbh.likelihood = "batched"
        mbh.window_pad_days = mbh.buffer_time / 86400.0
        self._resolve(mbh)


class AutoResolutionTest(_NoDurationEnv, unittest.TestCase):
    """``auto`` (the default): batched when the run can use it, else full with
    exactly ONE INFO line naming the reason; never "auto" in the cfg."""

    def _resolve_logged(self, mbh, domain):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        with self.assertLogs(sr.logger, level="INFO") as cm:
            cfg = sr.resolve_mbh_batched_cfg(mbh, domain_settings=domain)
        self.assertEqual(len(cm.output), 1, cm.output)
        self.assertIn("MBH_LIKELIHOOD=auto -> full", cm.output[0])
        return cfg, cm.output[0]

    def test_auto_is_batched_on_a_wdm_run(self):
        from lisatools.domains import WDMSettings
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        for domain in (_wdm_spec(), WDMSettings(32, 128, 10.0, force_backend="cpu")):
            with self.assertNoLogs(sr.logger, level="INFO"):
                cfg = sr.resolve_mbh_batched_cfg(_mbh(), domain_settings=domain)
            self.assertEqual(cfg["mbh_likelihood"], "batched")
            self.assertEqual(cfg["mbh_waveform_duration"], 90 * 86400.0)

    def test_auto_full_for_tdionfly(self):
        cfg, msg = self._resolve_logged(_mbh(use_tdionfly=True), _wdm_spec())
        self.assertEqual(cfg["mbh_likelihood"], "full")
        self.assertEqual(cfg["mbh_waveform_duration"], _mbh().waveform_duration)
        self.assertIn("USE_TDIONFLY=1", msg)

    def test_auto_full_on_a_non_wdm_domain(self):
        from lisatools.domains import STFTSettings

        for domain, name in ((_fd_spec(), "FDSettings"),
                             (STFTSettings.make_factory(big_dt=1000.0), "STFTSettings")):
            cfg, msg = self._resolve_logged(_mbh(), domain)
            self.assertEqual(cfg["mbh_likelihood"], "full")
            self.assertEqual(cfg["mbh_waveform_duration"], _mbh().waveform_duration)
            self.assertIn(f"the run domain is {name}, not WDM", msg)

    def test_auto_full_on_an_unidentifiable_domain(self):
        def custom_factory(times, dt, force_backend):  # pragma: no cover - never called
            raise AssertionError

        for domain in (custom_factory, None):
            cfg, msg = self._resolve_logged(_mbh(), domain)
            self.assertEqual(cfg["mbh_likelihood"], "full")
            self.assertIn("cannot be identified before the build", msg)

    def test_auto_full_on_an_explicit_conflicting_duration(self):
        for raw in ("2592000", "none"):
            with mock.patch.dict(os.environ, {"MBH_WAVEFORM_DURATION": raw}):
                mbh = _mbh_from_env()
                cfg, msg = self._resolve_logged(mbh, _wdm_spec())
            self.assertEqual(cfg["mbh_likelihood"], "full")
            self.assertEqual(cfg["mbh_waveform_duration"], mbh.waveform_duration)   # honoured
            self.assertIn(f"MBH_WAVEFORM_DURATION={raw} disagrees", msg)

    def test_auto_batched_when_the_env_duration_equals_the_window(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        with mock.patch.dict(os.environ, {"MBH_WAVEFORM_DURATION": str(90 * 86400.0)}):
            mbh = _mbh_from_env()
            with self.assertNoLogs(sr.logger, level="INFO"):
                cfg = sr.resolve_mbh_batched_cfg(mbh, domain_settings=_wdm_spec())
        self.assertEqual(cfg["mbh_likelihood"], "batched")

    def test_every_reason_is_named_on_the_one_line(self):
        _, msg = self._resolve_logged(_mbh(use_tdionfly=True), _fd_spec())
        self.assertIn("USE_TDIONFLY=1", msg)
        self.assertIn("FDSettings, not WDM", msg)


def _mbh_from_env(**kw):
    """SourceMBHSettings resolved under the CURRENT env (only the MBH_LIKELIHOOD
    / USE_TDIONFLY knobs cleared), so MBH_WAVEFORM_DURATION reaches the field."""
    from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

    saved = {k: os.environ.pop(k) for k in ("MBH_LIKELIHOOD", "USE_TDIONFLY") if k in os.environ}
    try:
        s = SourceMBHSettings()
    finally:
        os.environ.update(saved)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class RunDomainSpecTest(unittest.TestCase):
    def test_make_factories_declare_their_class(self):
        from lisatools.domains import FDSettings, STFTSettings, WDMSettings

        self.assertIs(WDMSettings.make_factory(Nf=4, Nt=4).domain_settings_class, WDMSettings)
        self.assertIs(FDSettings.make_factory().domain_settings_class, FDSettings)
        self.assertIs(STFTSettings.make_factory(big_dt=10.0).domain_settings_class, STFTSettings)

    def test_class_of_a_spec(self):
        from lisatools.domains import FDSettings, WDMSettings
        from lisatools.globalfit.stock.erebor.source_runtime import run_domain_settings_class

        self.assertIs(run_domain_settings_class(_wdm_spec()), WDMSettings)
        self.assertIs(run_domain_settings_class(_fd_spec()), FDSettings)
        self.assertIs(
            run_domain_settings_class(WDMSettings(32, 128, 10.0, force_backend="cpu")), WDMSettings
        )
        self.assertIsNone(run_domain_settings_class(lambda times, dt, force_backend: None))
        self.assertIsNone(run_domain_settings_class(None))
        self.assertIsNone(run_domain_settings_class(mock.MagicMock()))   # attr is not a type

    def test_built_general_info_reports_the_configured_spec(self):
        # GeneralSetup keeps the resolved general settings as ``.settings``
        # (the factory the injection sites saw) and its own domain_settings
        # is the BUILT instance: the spec wins, the instance is the fallback.
        from types import SimpleNamespace

        from lisatools.globalfit.stock.erebor.source_runtime import run_domain_spec

        spec, inst = object(), object()
        gi = SimpleNamespace(settings=SimpleNamespace(domain_settings=spec), domain_settings=inst)
        self.assertIs(run_domain_spec(gi), spec)
        self.assertIs(run_domain_spec(SimpleNamespace(domain_settings=inst)), inst)
        gi.settings.domain_settings = None
        self.assertIs(run_domain_spec(gi), inst)


class _GS:
    tdi_chan = "XYZ"
    tdi_gen_str = "2nd generation"
    nchannels = 3
    data_mode = "synthetic"
    sobbh_reference_time = 0.0
    mbh_waveform_t0 = 0.0
    min_freq = 1e-4
    max_freq = 2.5e-2


def _sobbh_emri():
    from lisatools.globalfit.stock.erebor.source_runtime import (
        SourceEMRISettings,
        SourceSOBBHSettings,
    )

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SOBBH_LIKELIHOOD", None)
        return SourceSOBBHSettings(), SourceEMRISettings()


class SourceSignalCfgTest(_NoDurationEnv, unittest.TestCase):
    def test_cfg_carries_the_batched_keys(self):
        # source_signal_cfg must route the resolved values (and the pinned
        # duration) into the plain-value cfg the getters/builders read.
        from lisatools.globalfit.stock.erebor.source_runtime import source_signal_cfg

        sobbh, emri = _sobbh_emri()
        cfg = source_signal_cfg(
            _GS(), _mbh(likelihood="batched"), sobbh, emri, domain_settings=_wdm_spec()
        )
        self.assertEqual(cfg["mbh_likelihood"], "batched")
        self.assertEqual(cfg["mbh_window_after"], 10 * 86400.0)
        self.assertEqual(cfg["mbh_phenom_kwargs"]["waveform_duration"], 90 * 86400.0)
        self.assertEqual(cfg["mbh_phenom_kwargs"]["response_order"], 8)

    def test_cfg_resolves_auto_with_the_given_domain(self):
        # the cfg every builder/getter reads carries the RESOLVED mode only
        from lisatools.globalfit.stock.erebor.source_runtime import source_signal_cfg

        sobbh, emri = _sobbh_emri()
        cfg = source_signal_cfg(_GS(), _mbh(), sobbh, emri, domain_settings=_wdm_spec())
        self.assertEqual(cfg["mbh_likelihood"], "batched")
        self.assertEqual(cfg["mbh_phenom_kwargs"]["waveform_duration"], 90 * 86400.0)
        cfg = source_signal_cfg(_GS(), _mbh(), sobbh, emri, domain_settings=_fd_spec())
        self.assertEqual(cfg["mbh_likelihood"], "full")
        self.assertEqual(cfg["mbh_phenom_kwargs"]["waveform_duration"], _mbh().waveform_duration)


class InjectionDurationTest(unittest.TestCase):
    """Injections and templates agree on the inspiral length (fix round 1),
    under the RESOLVED mode: the default (auto) is batched on the stock WDM
    runs, full when a blocker is present."""

    _NINETY_DAYS = 90 * 86400.0

    def _env(self, **extra):
        patch = mock.patch.dict(os.environ, {}, clear=False)
        patch.start()
        self.addCleanup(patch.stop)
        for k in ("MBH_WAVEFORM_DURATION", "USE_TDIONFLY", "MBH_LIKELIHOOD"):
            os.environ.pop(k, None)
        os.environ.update(extra)

    def test_helper(self):
        from lisatools.domains import WDMSettings
        from lisatools.globalfit.stock.erebor.source_runtime import mbh_injection_duration

        self._env()
        s = _mbh()
        raw = s.waveform_duration
        wdm, fd = _wdm_spec(), _fd_spec()
        self.assertEqual(s.likelihood, "auto")
        self.assertEqual(mbh_injection_duration(s, domain_settings=wdm), self._NINETY_DAYS)
        inst = WDMSettings(32, 128, 10.0, force_backend="cpu")
        self.assertEqual(mbh_injection_duration(s, domain_settings=inst), self._NINETY_DAYS)
        self.assertEqual(mbh_injection_duration(s, domain_settings=fd), raw)   # auto -> full
        s.use_tdionfly = True
        self.assertEqual(mbh_injection_duration(s, domain_settings=wdm), raw)  # auto -> full
        s.use_tdionfly = False
        s.likelihood = "full"
        self.assertEqual(mbh_injection_duration(s, domain_settings=wdm), raw)
        s.likelihood = "batched"
        self.assertEqual(mbh_injection_duration(s, domain_settings=wdm), self._NINETY_DAYS)

    def test_programmatic_override_warns(self):
        # explicit batched AND auto (resolving to batched): a programmatic,
        # non-default duration is overridden by the window, with a warning
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        self._env()
        for mode in ("batched", "auto"):
            s = _mbh(likelihood=mode)
            with self.assertNoLogs(sr.logger, level="WARNING"):
                sr.resolve_mbh_batched_cfg(s, domain_settings=_wdm_spec())  # class default
            s.waveform_duration = self._NINETY_DAYS
            with self.assertNoLogs(sr.logger, level="WARNING"):
                sr.resolve_mbh_batched_cfg(s, domain_settings=_wdm_spec())  # == window
            s.waveform_duration = 45 * 86400.0
            with self.assertLogs(sr.logger, level="WARNING") as cm:
                cfg = sr.resolve_mbh_batched_cfg(s, domain_settings=_wdm_spec())
            self.assertEqual(cfg["mbh_waveform_duration"], self._NINETY_DAYS)
            self.assertIn("overridden by the window", cm.output[0])

    def _all_sources_synthetic_pk(self, **fit_kw):
        from lisatools.globalfit.stock import erebor

        fit = erebor.get_stock("all_sources", data_mode="synthetic", **fit_kw)
        return fit, self._synthetic_pk(fit.make_general_settings())

    @staticmethod
    def _synthetic_pk(gs):
        specs = {cls.__name__: kw for cls, kw in gs.processor_init_kwargs["processor_specs"]}
        return gs, specs["SyntheticDataProcessor"]["mbh_phenom_kwargs"]

    def test_all_sources_synthetic_site(self):
        self._env()
        _, (_, pk) = self._all_sources_synthetic_pk()
        self.assertEqual(pk["waveform_duration"], self._NINETY_DAYS)   # default -> batched

    def test_all_sources_synthetic_site_follows_auto_fallback(self):
        # USE_TDIONFLY=1 / a non-WDM domain: auto -> full at the injection too
        from lisatools.globalfit.stock import erebor

        self._env(USE_TDIONFLY="1")
        fit, (_, pk) = self._all_sources_synthetic_pk()
        self.assertEqual(pk["waveform_duration"], fit.mbh.waveform_duration)
        self._env()
        fit = erebor.get_stock("all_sources", data_mode="synthetic")
        fit.general.domain_settings = _fd_spec()
        _, pk = self._synthetic_pk(fit.make_general_settings())
        self.assertEqual(pk["waveform_duration"], fit.mbh.waveform_duration)

    def test_all_sources_mojito_synthesize_site(self):
        from lisatools.globalfit.stock import erebor

        self._env()
        gs = erebor.get_stock("all_sources").make_general_settings()
        self.assertEqual(
            gs.processor_init_kwargs["mbh_phenom_kwargs"]["waveform_duration"], self._NINETY_DAYS
        )
        self._env(USE_TDIONFLY="1")
        fit = erebor.get_stock("all_sources")
        gs = fit.make_general_settings()
        self.assertEqual(
            gs.processor_init_kwargs["mbh_phenom_kwargs"]["waveform_duration"],
            fit.mbh.waveform_duration,
        )

    def test_full_year_combined_synthetic_site(self):
        from lisatools.globalfit.stock import erebor

        self._env(MBHB_IDS="0", EMRI_IDS="1", SOBHB_IDS="2")
        gs = erebor.get_stock("full_year_combined", data_mode="synthetic").make_general_settings()
        self.assertEqual(
            gs.processor_init_kwargs["mbh_phenom_kwargs"]["waveform_duration"], self._NINETY_DAYS
        )
        fit = erebor.get_stock("full_year_combined", data_mode="synthetic")
        fit.general.domain_settings = _fd_spec()
        gs = fit.make_general_settings()
        self.assertEqual(
            gs.processor_init_kwargs["mbh_phenom_kwargs"]["waveform_duration"],
            fit.mbh.waveform_duration,
        )


class AttachSiteAgreesWithInjectionTest(unittest.TestCase):
    """The variants' ``attach_runtime_objects`` resolve the moves' cfg from the
    SAME run-domain spec the injection site saw (``general_info.settings
    .domain_settings``), not from the built instance: with a hand-written
    (unidentifiable) factory that builds WDM, both sites say ``full``."""

    def setUp(self):
        patch = mock.patch.dict(os.environ, {}, clear=False)
        patch.start()
        self.addCleanup(patch.stop)
        for k in ("MBH_WAVEFORM_DURATION", "USE_TDIONFLY", "MBH_LIKELIHOOD"):
            os.environ.pop(k, None)

    def _attach(self, variant_cls, spec):
        from types import SimpleNamespace

        from lisatools.domains import WDMSettings

        plain = {k: getattr(_GS, k) for k in dir(_GS) if not k.startswith("_")}
        # the RESOLVED general settings (what the injection site saw, kept by
        # GeneralSetup as .settings) carry the spec; the fit-level block
        # carries None, as in a stock build
        gs = SimpleNamespace(**plain, domain_settings=spec)
        built = WDMSettings(32, 128, 10.0, force_backend="cpu")   # what the factory built
        fake = SimpleNamespace(
            branch_names=["mbh"], branches={"mbh": None},
            source_info={"mbh": SimpleNamespace(settings=_mbh(), transform=None)},
            general=SimpleNamespace(**plain, domain_settings=None),
            general_info=SimpleNamespace(settings=gs, domain_settings=built),
        )
        variant_cls.attach_runtime_objects(fake)
        return fake.source_info["mbh"].signal_gen.cfg

    def test_all_sources_and_full_year(self):
        from lisatools.globalfit.stock.erebor.variants.all_sources import AllSourcesGlobalFit
        from lisatools.globalfit.stock.erebor.variants.full_year_combined import (
            FullYearCombinedGlobalFit,
        )

        def custom_factory(times, dt, force_backend):  # pragma: no cover - never called
            raise AssertionError

        for cls in (AllSourcesGlobalFit, FullYearCombinedGlobalFit):
            self.assertEqual(self._attach(cls, _wdm_spec())["mbh_likelihood"], "batched", cls)
            self.assertEqual(self._attach(cls, custom_factory)["mbh_likelihood"], "full", cls)


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

    def test_decimated_window_runs_the_generator_on_the_coarse_lattice(self):
        """MBH_WINDOW_DECIMATE=2: the windowed generator gets TD settings at 2 dt
        (N / 2 samples from the data start), the response's sampling frequency
        1 / (2 dt), the epoch snapped onto the 5-s lattice, and the adapter's factor."""
        from types import SimpleNamespace

        from lisatools.domains import WDMSettings
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        wdm = mock.MagicMock(spec=WDMSettings)
        gi = self._gi(wdm)
        gi.data_td_settings = SimpleNamespace(N=1440 * 4320, dt=2.5, t0=97729089.0)
        cfg = self._cfg()
        cfg["mbh_window_decimate"] = 2
        with mock.patch("lisatools.sources.bbh.gridaligned.WindowedGridAlignedMBHWaveform") as W, \
                mock.patch("lisatools.sources.batching.MBHWindowedWDMSignalGen") as A:
            A.side_effect = lambda gen, *a, **k: mock.MagicMock(spec=["wave_gen"], wave_gen=gen)
            adapter = sr.get_mbh_windowed_gen(gi, cfg)
            undecimated = sr.get_mbh_windowed_gen(gi, self._cfg())
        self.assertIsNot(undecimated, adapter)        # the decimation keys the cache
        kw = W.call_args_list[0].kwargs
        td = kw["data_td_settings"]
        self.assertEqual((td.N, td.dt, td.t0), (1440 * 4320 // 2, 5.0, 97729089.0))
        self.assertEqual(kw["sampling_frequency"], 0.2)
        # offset 1.5 s = 0.3 coarse samples -> k = 0: snapped ONTO the data start
        self.assertAlmostEqual(kw["waveform_t0"], 97729089.0, places=6)
        self.assertAlmostEqual(adapter.t_plunge_snap, -1.5, places=6)
        self.assertEqual(A.call_args_list[0].kwargs["decimate"], 2)
        self.assertEqual(A.call_args_list[1].kwargs["decimate"], 1)


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

    def _call(self, mode, decimate=None):
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
        if decimate is not None:
            cfg["mbh_window_decimate"] = decimate
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

    def test_decimated_window_snaps_the_stock_onto_the_same_coarse_lattice(self):
        """The stock generator (residual rebuild + cross-check) must share the
        windowed one's epoch: with MBH_WINDOW_DECIMATE=2 both snap on 2 dt."""
        gen, _, built = self._call("batched", decimate=2)
        (t0_built,) = built
        self.assertAlmostEqual(t0_built, 97729089.0, places=6)
        self.assertAlmostEqual(gen.t_plunge_snap, -1.5, places=9)

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
