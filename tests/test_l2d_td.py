"""L2D time-domain generators: the ``td_waveform`` schema, the recipe-side closures and the shared TD residual.

The engine objects are stand-ins (as in ``test_l2d_recipe``); the waveform-level behaviour of
``get_td_signals`` is tested in ``test_td_signal_gens``.
"""

import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import types
import unittest
import warnings
from unittest import mock

import numpy as np

from lisatools.domains import DomainBaseArray, TDSettings, TDSignal
from lisatools.globalfit.stock.l2d import build_recipe
from lisatools.globalfit.stock.l2d.blocks import base as blocks_base
from lisatools.globalfit.stock.l2d.blocks.base import TDGenerator, resolve_td_generator
from lisatools.globalfit.stock.l2d.build_recipe import RecipeContext, attach_signal_gens
from lisatools.globalfit.stock.l2d.generators import SignalGen, WaveformHandle
from lisatools.globalfit.stock.l2d.io import RunConfig, dump_run, load_run
from lisatools.sources.bbh.waveform import PhenomTHMTDIOnFlyWaveform

from tests.test_l2d_sources import EMRI_BLOCK, MBH_BLOCK, _Sources, mbh_with_search, source_raw

T0, DT, N = 97729089.327664, 5.0, 64


class _TD(_Sources):
    def config(self, blocks, **data_changes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # source blocks without a search warn
            return RunConfig.model_validate(source_raw(blocks, **data_changes), context={"run_dir": self.run_dir})


class TDWaveformSchemaTest(_TD):
    def test_defaults_reuse_the_block_waveform(self):
        cfg = self.config({"mbh": MBH_BLOCK, "emri": EMRI_BLOCK})
        self.assertIsNone(cfg.blocks["mbh"].td_waveform)
        self.assertEqual(cfg.blocks["mbh"].td_generator(), TDGenerator(None, "get_td_signals", {}, {}))
        # the method call gets the block's runtime kwargs, as the analysis-domain call does
        self.assertEqual(
            cfg.blocks["emri"].td_generator().runtime_kwargs, EMRI_BLOCK["waveform"]["runtime_kwargs"]
        )
        self.assertFalse(cfg.data.keep_td)

    def test_method_and_runtime_kwargs(self):
        block = {**MBH_BLOCK, "td_waveform": {"method": "get_signals_for_residuals", "runtime_kwargs": {"a": 1}}}
        td = self.config({"mbh": block}).blocks["mbh"].td_generator()
        self.assertEqual((td.waveform_class, td.method, td.runtime_kwargs), (None, "get_signals_for_residuals", {"a": 1}))

    def test_separate_class(self):
        block = {**MBH_BLOCK, "td_waveform": {"class": "PhenomTHMTDIOnFlyWaveform", "kwargs": {"start_freq": 7e-5}}}
        td = self.config({"mbh": block}).blocks["mbh"].td_generator()
        self.assertEqual(
            td, TDGenerator(PhenomTHMTDIOnFlyWaveform, "get_td_signals", {"start_freq": 7e-5}, {})
        )

    def test_errors(self):
        cases = {
            "unknown method": ({"method": "nope"}, "has no method 'nope'"),
            "kwargs without a class": ({"kwargs": {"order": 20}}, "give td_waveform.class too"),
            "unknown class": ({"class": "NoSuchWaveform"}, "no class 'NoSuchWaveform'"),
            "engine-filled kwarg": (
                {"class": "PhenomTHMTDIOnFlyWaveform", "kwargs": {"orbits": 1}}, "filled in by the builder"
            ),
            "misspelled kwarg": (
                {"class": "PhenomTHMTDIOnFlyWaveform", "kwargs": {"ordr": 1}}, "does not accept ['ordr']"
            ),
        }
        for name, (spec, msg) in cases.items():
            with self.subTest(name):
                self.assertConfigError(
                    "value_error", "blocks.mbh.mbh", msg=msg, blocks={"mbh": {**MBH_BLOCK, "td_waveform": spec}}
                )

    def test_dump_reloads(self):
        block = {**MBH_BLOCK, "td_waveform": {"class": "PhenomTHMTDIOnFlyWaveform", "kwargs": {"start_freq": 7e-5}}}
        cfg = self.config({"mbh": block}, keep_td=True)
        again = load_run(dump_run(cfg, self.run_dir / "resolved" / "run.yaml"))
        self.assertEqual(again.blocks["mbh"].td_waveform, cfg.blocks["mbh"].td_waveform)
        self.assertTrue(again.data.keep_td)


class ResolveTDGeneratorTest(_TD):
    def test_separate_class_gets_the_engine_arguments_it_accepts(self):
        block = {**MBH_BLOCK, "td_waveform": {"class": "PhenomTHMTDIOnFlyWaveform", "kwargs": {"start_freq": 7e-5}}}
        cfg = self.config({"mbh": block})
        engine = {"waveform_t0": T0, "force_backend": "cpu", "not_a_constructor_argument": 1}
        with mock.patch.object(blocks_base, "engine_waveform_kwargs", lambda ctx, block: engine):
            td = resolve_td_generator(cfg.blocks["mbh"], ctx=None)
        self.assertEqual(td.init_kwargs, {"start_freq": 7e-5, "waveform_t0": T0, "force_backend": "cpu"})

    def test_own_waveform_is_left_as_is(self):
        cfg = self.config({"mbh": MBH_BLOCK})
        with mock.patch.object(blocks_base, "engine_waveform_kwargs", side_effect=AssertionError):
            self.assertEqual(resolve_td_generator(cfg.blocks["mbh"], ctx=None).init_kwargs, {})


# --- recipe side --- #

class Doubling:
    """Stands in for a TransformContainer: waveform params = 2 * sampling params."""

    def both_transforms(self, coords):
        assert coords.ndim == 2
        return 2 * coords


class FakeWave:
    """``get_td_signals(amp, k0)``: ``amp`` on 4 samples from sample ``k0`` (one TDSignal per row)."""

    xp = np

    def __init__(self, **init_kwargs):
        self.init_kwargs = init_kwargs
        self.calls = []

    def get_signals_for_residuals(self, *cols, **kwargs):
        self.calls.append(("fd", cols, kwargs))
        return DomainBaseArray([TDSignal(np.zeros((3, 1)), TDSettings(1, DT, t0=T0))] * len(cols[0]))

    def get_td_signals(self, amp, k0, td_settings=None, **kwargs):
        self.calls.append(("td", (amp, k0), kwargs))
        return DomainBaseArray([
            TDSignal(np.full((3, 4), a), TDSettings(4, DT, t0=T0 + int(k) * DT)) for a, k in zip(amp, k0)
        ])


class SignalGenTest(unittest.TestCase):
    def test_one_leaf_or_a_batch(self):
        wave = FakeWave()
        gen = SignalGen(wave, "get_td_signals", Doubling(), {"mode": 1})
        one = gen(1.5, 3.0)
        self.assertIsInstance(one, TDSignal)
        self.assertEqual((one.arr[0, 0], one.t0), (3.0, T0 + 6 * DT))  # transformed: amp 3, k0 6
        batch = gen(np.array([1.0, 2.0]), np.array([0.0, 1.0]), extra=True)
        self.assertIsInstance(batch, DomainBaseArray)
        self.assertEqual(len(batch), 2)
        (_, cols, kwargs) = wave.calls[-1]
        self.assertTrue(all(c.flags.c_contiguous for c in cols))
        self.assertEqual(kwargs, {"mode": 1, "extra": True})


def fake_general():
    data = np.arange(3 * N, dtype=float).reshape(3, N)
    return types.SimpleNamespace(
        data_td_settings=TDSettings(N, DT, t0=T0, force_backend="cpu"),
        data_processor=types.SimpleNamespace(td_signal=TDSignal(data, TDSettings(N, DT, t0=T0))),
    )


class AttachSignalGensTest(_TD):
    def run_attach(self, keep_td, td_generators):
        cfg = self.config({"mbh": mbh_with_search()}, keep_td=keep_td)
        self.wave = FakeWave()
        # a seeded branch's Settings.signal_gen (build_source registers it), over the shared waveform
        signal_gen = SignalGen(self.wave, "get_signals_for_residuals", None, {"wk": 1})
        info = types.SimpleNamespace(transform=None, waveform_kwargs={"wk": 1}, signal_gen=signal_gen)
        curr = types.SimpleNamespace(source_info={"mbh": info}, general_info=fake_general())
        # walker 0 (cold chain) has leaves 0 and 2 on; walker 1 differs and must be ignored
        inds = np.zeros((2, 2, 3), bool)
        inds[0, 0, [0, 2]] = True
        inds[0, 1, 1] = True
        coords = np.zeros((2, 2, 3, 2))
        coords[0, 0, 0] = [1.0, 2.0]
        coords[0, 0, 2] = [10.0, 50.0]
        coords[0, 1, 1] = [99.0, 7.0]
        state = types.SimpleNamespace(branches_inds={"mbh": inds}, branches_coords={"mbh": coords})
        ctx = RecipeContext(cfg, {"mbh": self.wave})
        attach_signal_gens(ctx, td_generators, curr, state)
        return ctx, curr

    def test_generators_share_the_seeded_waveform(self):
        ctx, curr = self.run_attach(False, {"mbh": TDGenerator(None, "get_td_signals", {}, {"td": 1})})
        self.assertIs(ctx.signal_gens["mbh"], curr.source_info["mbh"].signal_gen)  # the one setup_acs used
        self.assertEqual(sorted(ctx.td_signal_gens), ["mbh"])
        self.assertIsNone(ctx.td_ac)
        ctx.signal_gens["mbh"](1.0, 2.0)
        ctx.td_signal_gens["mbh"](1.0, 2.0)
        self.assertEqual([(c[0], c[2]) for c in self.wave.calls], [("fd", {"wk": 1}), ("td", {"td": 1})])

    def test_separate_td_class_is_built_from_its_init_kwargs(self):
        td = TDGenerator(FakeWave, "get_td_signals", {"start_freq": 7e-5}, {})
        made = []
        with mock.patch.object(FakeWave, "__init__", lambda s, **kw: made.append(kw) or setattr(s, "calls", [])):
            ctx, _ = self.run_attach(False, {"mbh": td})
        self.assertEqual(made[1:], [{"start_freq": 7e-5}])  # [0] is the seeded stand-in, built by run_attach
        ctx.td_signal_gens["mbh"](1.0, 2.0)
        self.assertEqual(self.wave.calls, [])  # the TD call went to the separate instance

    def test_keep_td_subtracts_walker_0_from_a_cpu_copy_of_the_data(self):
        ctx, curr = self.run_attach(True, {"mbh": TDGenerator(None, "get_td_signals", {}, {})})
        td = ctx.td_ac.data
        self.assertEqual((td.N, td.t0), (N, T0))
        self.assertIs(td.xp, np)  # the TD residual lives on the cpu
        expected = curr.general_info.data_processor.td_signal.arr.copy()
        expected[:, 2:6] -= 1.0    # leaf 0: amp 1 from sample 2
        expected[:, 50:54] -= 10.0  # leaf 2: amp 10 from sample 50
        np.testing.assert_array_equal(td.arr, expected)
        self.assertIsNot(td.arr, curr.general_info.data_processor.td_signal.arr)
        self.assertIn("mbh", ctx.td_ac.signal_gen)
        self.assertIsNone(ctx.td_ac.sens_mat)

    def test_keep_td_without_td_generators_warns(self):
        with mock.patch.object(build_recipe.logger, "warning") as warning:  # RecipeBuilder does not propagate
            ctx, _ = self.run_attach(True, {})
        self.assertIsNone(ctx.td_ac)
        (args, _), = warning.call_args_list
        self.assertIn("no seeded block has a TD generator", args[0])


if __name__ == "__main__":
    unittest.main()
