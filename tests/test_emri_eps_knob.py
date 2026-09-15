"""EMRI mode-selection threshold knob (``EMRI_EPS``) -- user ruling 2026-09-14.

The knob is spelled ``EMRI_EPS`` / ``EMRISettings.eps`` (the historical FEW
1.x name), but the kwarg it drives is FEW 2.x's ``mode_selection_threshold``
(``few/waveform/base.py:143``). Two facts make the wiring non-obvious and are
pinned here:

* a literal ``eps`` key is INERT -- FEW 2.x renamed the parameter, so ``eps``
  falls through ``_generate_waveform``'s ``**kwargs`` into the summation
  module and is silently dropped; and
* ``_generate_waveform`` has its OWN per-call default (1e-5) which it passes
  to the selector unconditionally, while the selector only falls back to its
  constructor value when the per-call argument is ``None``
  (``few/utils/modeselector.py:243-244``). The LAT constructor pin
  ``EMRI_MODE_SELECTOR_KWARGS`` therefore never takes effect, and the knob
  MUST arrive at CALL time.
"""

import inspect
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
MODE_KEY = "mode_selection_threshold"


class _Stop(Exception):
    """Sentinel raised by the recording generator stub.

    Deliberately NOT a ValueError/AssertionError so ``few_domain_guard`` lets
    it through untouched.
    """


class _RecordingGen:
    """Stand-in waveform generator that records its call kwargs."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, *params, **kwargs):
        self.kwargs = dict(kwargs)
        raise _Stop()


class KwargNamePinTest(unittest.TestCase):
    """Pin the kwarg name the wiring uses against the generator signature."""

    def test_few_generate_waveform_takes_mode_selection_threshold(self):
        from few.waveform.base import SphericalHarmonicWaveformBase

        sig = inspect.signature(SphericalHarmonicWaveformBase._generate_waveform)
        self.assertIn(MODE_KEY, sig.parameters)
        self.assertEqual(sig.parameters[MODE_KEY].default, 1e-5)
        # The FEW 1.x name is gone -- an ``eps`` key would be swallowed by
        # ``**kwargs`` with no error and no effect.
        self.assertNotIn("eps", sig.parameters)

    def test_lat_constructor_pin_uses_the_same_name(self):
        from lisatools.sources.emri.response import EMRI_MODE_SELECTOR_KWARGS

        self.assertEqual(list(EMRI_MODE_SELECTOR_KWARGS), [MODE_KEY])


class SettingsFieldTest(unittest.TestCase):
    def test_default_is_none(self):
        from lisatools.globalfit.stock.erebor.emri import EMRISettings

        env = {k: v for k, v in os.environ.items() if k != "EMRI_EPS"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(EMRISettings().eps)

    def test_explicit_kwarg(self):
        from lisatools.globalfit.stock.erebor.emri import EMRISettings

        self.assertEqual(EMRISettings(eps=1e-3).eps, 1e-3)


class EffectiveWaveformKwargsTest(unittest.TestCase):
    """``apply_emri_mode_selection_threshold`` -- the one resolution point.

    It RESOLVES only (2026-09-15 crash fix): the branch ``waveform_kwargs``
    are never stamped, because for EMRI they double as the move's LIKELIHOOD
    kwargs and ``inner_product`` has no such parameter. Delivery is the wave
    wrap's ``runtime_kwargs`` alone (see :class:`RuntimeWiringTest`).
    """

    def setUp(self):
        from lisatools.globalfit.stock.erebor.source_runtime import (
            apply_emri_mode_selection_threshold,
        )
        from lisatools.globalfit.stock.erebor.emri import EMRISettings

        self.apply = apply_emri_mode_selection_threshold
        self.Settings = EMRISettings

    def test_unset_adds_no_key(self):
        emri = self.Settings(eps=None, waveform_kwargs=dict())
        self.assertIsNone(self.apply(emri))
        self.assertEqual(emri.waveform_kwargs, {})

    def test_field_resolves_without_stamping_waveform_kwargs(self):
        emri = self.Settings(eps=1e-3, waveform_kwargs=dict())
        self.assertEqual(self.apply(emri), 1e-3)
        # The crash channel: this dict becomes ``waveform_like_kwargs``.
        self.assertEqual(emri.waveform_kwargs, {})

    def test_explicit_waveform_kwargs_wins(self):
        emri = self.Settings(eps=1e-3, waveform_kwargs={MODE_KEY: 5e-4})
        self.assertEqual(self.apply(emri), 5e-4)
        self.assertEqual(emri.waveform_kwargs, {MODE_KEY: 5e-4})

    def test_idempotent(self):
        emri = self.Settings(eps=1e-3, waveform_kwargs=dict())
        self.apply(emri)
        self.assertEqual(self.apply(emri), 1e-3)
        self.assertEqual(emri.waveform_kwargs, {})

    def test_none_waveform_kwargs_is_not_materialized(self):
        emri = self.Settings(eps=1e-3, waveform_kwargs=None)
        self.assertEqual(self.apply(emri), 1e-3)
        self.assertIsNone(emri.waveform_kwargs)


class LikeKwargsAssemblyTest(unittest.TestCase):
    """The 6mo production crash (2026-09-15) and its fix.

    ``EMRIMoveBuilder.like_kwargs_from_waveform_kwargs`` routes the branch
    ``waveform_kwargs`` into the move's ``waveform_like_kwargs``, which flow
    ``compute_like`` -> ``compute_acs_like`` ->
    ``AnalysisContainer.template_likelihood`` -> ``inner_product(**kwargs)``.
    A waveform-only key in there is a ``TypeError`` at the first likelihood
    call of the run.
    """

    def _info(self, **waveform_kwargs):
        import types

        return types.SimpleNamespace(waveform_kwargs=dict(waveform_kwargs))

    def _builders(self):
        from lisatools.globalfit.recipe import EMRIMoveBuilder, SOBBHMoveBuilder

        gen = _RecordingGen()
        return (
            EMRIMoveBuilder(wave_gen=gen),
            SOBBHMoveBuilder(wave_gen=gen),
        )

    def test_emri_like_kwargs_drop_the_waveform_only_key(self):
        emri_builder, _ = self._builders()
        info = self._info(**{MODE_KEY: 1e-3})
        like_kw = emri_builder.assemble_like_kwargs(info)
        self.assertNotIn(MODE_KEY, like_kw)
        # ... while the WAVEFORM-side kwargs keep it (and the branch dict is
        # not mutated as a side effect of assembling either one).
        self.assertEqual(
            emri_builder.assemble_gen_kwargs(info), {MODE_KEY: 1e-3}
        )
        self.assertEqual(info.waveform_kwargs, {MODE_KEY: 1e-3})

    def test_emri_like_kwargs_keep_every_other_key(self):
        emri_builder, _ = self._builders()
        info = self._info(**{MODE_KEY: 1e-3, "complex": True})
        self.assertEqual(
            emri_builder.assemble_like_kwargs(info), {"complex": True}
        )

    def test_sobbh_like_kwargs_are_an_unfiltered_copy(self):
        """SOBBH shares the flag but strips nothing -- byte-identical."""
        _, sobbh_builder = self._builders()
        payload = {MODE_KEY: 1e-3, "complex": True}
        info = self._info(**payload)
        self.assertEqual(sobbh_builder.assemble_like_kwargs(info), payload)

    def test_explicit_like_kwargs_are_stripped_too(self):
        """The invariant is about what ``inner_product`` accepts, so it holds
        whichever dict the like-kwargs came from."""
        from lisatools.globalfit.recipe import EMRIMoveBuilder

        builder = EMRIMoveBuilder(
            wave_gen=_RecordingGen(),
            waveform_like_kwargs={MODE_KEY: 1e-3, "complex": True},
        )
        self.assertEqual(
            builder.assemble_like_kwargs(self._info()), {"complex": True}
        )

    def test_base_builder_strips_nothing(self):
        from lisatools.globalfit.recipe import MBHMoveBuilder

        self.assertEqual(MBHMoveBuilder.like_kwargs_strip_keys, frozenset())


class InnerProductConsumerTest(unittest.TestCase):
    """Bind the assembled like-kwargs against the REAL consumer signatures.

    This is the crash reproduced without numerics: the production traceback is
    ``inner_product() got an unexpected keyword argument
    'mode_selection_threshold'`` from ``analysiscontainer.py`` (the ``d_d``
    term), and ``inner_product`` has no ``**kwargs`` sink to absorb it.
    """

    def _like_kwargs(self):
        import types

        from lisatools.globalfit.recipe import EMRIMoveBuilder

        info = types.SimpleNamespace(waveform_kwargs={MODE_KEY: 1e-3})
        return EMRIMoveBuilder(wave_gen=_RecordingGen()).assemble_like_kwargs(info)

    def test_inner_product_rejects_the_waveform_only_key(self):
        """Negative control: the PRE-FIX dict still raises."""
        from lisatools.diagnostic import inner_product

        sig = inspect.signature(inner_product)
        with self.assertRaises(TypeError):
            sig.bind(object(), object(), psd=object(), **{MODE_KEY: 1e-3})

    def test_assembled_like_kwargs_bind_to_the_whole_chain(self):
        from lisatools.analysiscontainer import AnalysisContainer
        from lisatools.diagnostic import inner_product

        like_kw = self._like_kwargs()
        # ``template_likelihood`` forwards its **kwargs verbatim (minus psd /
        # complex) into ``inner_product``; both have to accept them.
        inspect.signature(AnalysisContainer.template_likelihood).bind(
            object(), object(), **like_kw
        )
        inspect.signature(inner_product).bind(
            object(), object(), psd=object(), **like_kw
        )

    def test_mock_consumer_accepts_the_assembled_kwargs(self):
        """A stand-in with ``inner_product``'s exact parameter list."""

        def fake_inner_product(
            sig1, sig2, basis_settings=None, psd="LISASens", psd_args=(),
            psd_kwargs={}, normalize=False, complex=False,
        ):
            return 0.0

        self.assertEqual(
            fake_inner_product(object(), object(), **self._like_kwargs()), 0.0
        )


class CallTimeArrivalTest(unittest.TestCase):
    """The value has to reach the generator call, not just a constructor."""

    def test_wave_wrap_runtime_kwargs_reach_the_generator_call(self):
        from lisatools.sources.emri.response import EMRIWaveWrap

        gen = _RecordingGen()
        wrap = EMRIWaveWrap(
            gen, None, None, runtime_kwargs={MODE_KEY: 1e-3}, nchannels=3
        )
        with self.assertRaises(_Stop):
            wrap(1.0, 2.0)
        self.assertEqual(gen.kwargs.get(MODE_KEY), 1e-3)

    def test_per_call_kwargs_override_the_runtime_floor(self):
        from lisatools.sources.emri.response import EMRIWaveWrap

        gen = _RecordingGen()
        wrap = EMRIWaveWrap(
            gen, None, None, runtime_kwargs={MODE_KEY: 1e-3}, nchannels=3
        )
        with self.assertRaises(_Stop):
            wrap(1.0, **{MODE_KEY: 5e-4})
        self.assertEqual(gen.kwargs.get(MODE_KEY), 5e-4)


class RuntimeWiringTest(unittest.TestCase):
    """``source_signal_cfg`` -> ``get_emri_wave_wrap`` runtime kwargs."""

    def setUp(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        self.sr = sr
        sr._WAVE_WRAP_CACHE.clear()
        self.addCleanup(sr._WAVE_WRAP_CACHE.clear)

    def _cfg(self, eps):
        from lisatools.globalfit.stock.erebor.source_runtime import (
            SourceEMRISettings,
            source_signal_cfg,
        )

        emri = SourceEMRISettings(eps=eps, waveform_kwargs=dict())
        return source_signal_cfg(
            mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), emri
        )

    def test_cfg_carries_the_effective_threshold(self):
        self.assertEqual(self._cfg(1e-3)["emri_mode_selection_threshold"], 1e-3)
        self.assertIsNone(self._cfg(None)["emri_mode_selection_threshold"])

    def _wrap(self, threshold):
        import numpy as np
        import types

        sr = self.sr
        general_info = types.SimpleNamespace(
            dt=10.0,
            data_t0=0.0,
            Tobs=1000.0,
            data_td_settings=object(),
            force_backend="cpu",
        )
        cfg = dict(
            nchannels=3,
            tdi_gen_str="2nd generation",
            tdi_chan="AET",
            data_mode="synthetic",
            emri_response_order=40,
            emri_mode_selection_threshold=threshold,
        )
        with mock.patch.object(
            sr, "_wrap_device_and_orbits", return_value=(np, None, None, None)
        ), mock.patch.object(
            sr, "get_emri_response_wrapper", return_value=object()
        ), mock.patch.object(sr, "TDIConfig", return_value=object()):
            return sr.get_emri_wave_wrap(general_info, cfg)

    def test_wrap_gets_the_threshold_as_a_per_call_floor(self):
        self.assertEqual(self._wrap(1e-3).runtime_kwargs, {MODE_KEY: 1e-3})

    def test_wrap_unset_is_byte_identical_to_today(self):
        self.assertEqual(self._wrap(None).runtime_kwargs, {})


class EnvKnobSubprocessTest(unittest.TestCase):
    """Stock env knobs resolve at IMPORT (the variant template builds at
    module load), so the env cases run in a fresh interpreter."""

    CODE = (
        "from lisatools.globalfit.stock.erebor.source_runtime import (\n"
        "    SourceEMRISettings, apply_emri_mode_selection_threshold,\n"
        "    source_signal_cfg,\n"
        ")\n"
        "from lisatools.globalfit.recipe import EMRIMoveBuilder\n"
        "from unittest import mock\n"
        "import os\n"
        "want = os.environ.get('WANT_EPS')\n"
        "want = None if not want else float(want)\n"
        "emri = SourceEMRISettings(waveform_kwargs=dict())\n"
        "assert emri.eps == want, (emri.eps, want)\n"
        "eff = apply_emri_mode_selection_threshold(emri)\n"
        "assert eff == want, (eff, want)\n"
        "key = 'mode_selection_threshold'\n"
        "# the branch dict doubles as the move's LIKELIHOOD kwargs: the knob\n"
        "# must never be stamped into it (2026-09-15 inner_product crash)\n"
        "assert key not in emri.waveform_kwargs, emri.waveform_kwargs\n"
        "like_kw = EMRIMoveBuilder(wave_gen=None).assemble_like_kwargs(emri)\n"
        "assert key not in like_kw, like_kw\n"
        "cfg = source_signal_cfg(mock.MagicMock(), mock.MagicMock(),\n"
        "                        mock.MagicMock(), emri)\n"
        "assert cfg['emri_mode_selection_threshold'] == want, cfg[\n"
        "    'emri_mode_selection_threshold']\n"
        "print('EMRI_EPS_OK')\n"
    )

    def _run(self, eps_env):
        env = {k: v for k, v in os.environ.items() if k != "EMRI_EPS"}
        for var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[var] = "1"
        if eps_env is not None:
            env["EMRI_EPS"] = eps_env
            env["WANT_EPS"] = eps_env
        else:
            env.pop("WANT_EPS", None)
        out = subprocess.run(
            [sys.executable, "-c", self.CODE], env=env,
            capture_output=True, text=True, timeout=900,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn("EMRI_EPS_OK", out.stdout)

    def test_env_set(self):
        self._run("1e-3")

    def test_env_unset(self):
        self._run(None)


class SubmitScriptTest(unittest.TestCase):
    """Both 6mo campaign scripts pin the campaign value (2026-09-14 ruling)."""

    SCRIPTS = (
        "scripts/fstat_proposal/submit_gf_6mo_v8.sh",
        "scripts/fstat_proposal/submit_gf_6mo_v8_nogb.sh",
    )

    def test_scripts_export_the_campaign_value(self):
        for rel in self.SCRIPTS:
            text = (REPO / rel).read_text()
            self.assertIn("export EMRI_EPS=1e-3", text, rel)


if __name__ == "__main__":
    unittest.main()
