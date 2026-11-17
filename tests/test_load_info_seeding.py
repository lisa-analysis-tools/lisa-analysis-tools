"""Engine-side seeding and residual rebuild: ``GlobalFit.load_info`` and ``subtract_state_templates``.

``load_info`` runs on a stand-in ``self`` (no MPI GlobalFit, no backend file): a fresh start seeds through
the branches' search interfaces, or keeps the legacy defaults when no branch has one; a restart is not seeded.
"""

import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import copy
import logging
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from lisatools.domains import DomainBaseArray, TDSettings, TDSignal
from lisatools.globalfit import run as gf_run
from lisatools.globalfit.run import GlobalFit, subtract_state_templates
from lisatools.globalfit.searches import pad_ladder
from lisatools.globalfit.stock.l2d.generators import SignalGen, WaveformHandle

NTEMPS, NWALKERS = 3, 4
NLEAVES = {"psd": 1, "mbh": 3, "emri": 2}
NDIM = {"psd": 2, "mbh": 11, "emri": 14}


class UniformPrior:
    def __init__(self, ndim):
        self.ndim = ndim

    def rvs(self, size):
        return np.random.uniform(size=(*size, self.ndim))


class FakeInterface:
    """Switches on the leaves it is given, at the value ``fill`` (all leaves: like a "prior" search)."""

    def __init__(self, name, leaves=None, fill=7.0):
        self.name, self.leaves, self.fill = name, leaves, fill
        self.search_type = "stand-in"
        self.calls = []

    def populate_state(self, curr, state, betas, **kwargs):
        self.calls.append((betas, kwargs))
        leaves = slice(None) if self.leaves is None else self.leaves
        state.branches_coords[self.name][:, :, leaves] = self.fill
        state.branches_inds[self.name][:, :, leaves] = True


def info(search_interface=None, betas=None):
    return types.SimpleNamespace(search_interface=search_interface, betas=betas, injection=None)


class LoadInfoTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.backend_path = Path(tmp.name) / "run.h5"  # does not exist: a fresh start
        self.priors = {name: UniformPrior(NDIM[name]) for name in NLEAVES}

    def globalfit(self, infos):
        fake = types.SimpleNamespace(
            logger=logging.getLogger("test_load_info"),
            curr=types.SimpleNamespace(
                source_info=infos,
                general_info=types.SimpleNamespace(main_file_path=str(self.backend_path), past_file_for_start=None),
            ),
            engine_info=types.SimpleNamespace(
                branch_names=list(infos), nleaves_max={n: NLEAVES[n] for n in infos}, branch_states=None,
                branch_backends=None,
            ),
            ntemps=NTEMPS, nwalkers=NWALKERS,
        )
        fake.search_interfaces = types.MethodType(GlobalFit.search_interfaces, fake)
        return fake

    def load(self, infos, seed=3):
        np.random.seed(seed)
        return GlobalFit.load_info(self.globalfit(infos), self.priors)

    def test_without_interfaces_the_legacy_defaults_apply(self):
        state = self.load({"psd": info(), "mbh": info(), "emri": info()})
        for name in ("psd", "mbh", "emri"):
            self.assertTrue(state.branches_inds[name].all(), name)
        # the same prior draws, in the same order, as before
        np.random.seed(3)
        for name in ("psd", "mbh", "emri"):
            expected = self.priors[name].rvs(size=(NTEMPS, NWALKERS, NLEAVES[name]))
            np.testing.assert_array_equal(state.branches_coords[name], expected)

    def test_interfaces_seed_and_the_others_stay_off(self):
        psd = FakeInterface("psd", fill=1.5)  # all leaves, as the "prior" search
        mbh = FakeInterface("mbh", leaves=[0, 2])
        infos = {"psd": info(psd), "mbh": info(mbh, betas=[1.0, 0.5]), "emri": info()}
        state = self.load(infos)

        np.testing.assert_array_equal(state.branches_inds["mbh"][0, 0], [True, False, True])
        self.assertTrue((state.branches_coords["mbh"][:, :, [0, 2]] == 7.0).all())
        self.assertTrue(state.branches_inds["psd"].all())
        self.assertFalse(state.branches_inds["emri"].any())  # no interface: off (not the legacy default)

        (betas, kwargs), = mbh.calls
        np.testing.assert_array_equal(betas, [1.0, 0.5, 0.5])  # the branch ladder, padded to every temperature
        self.assertIs(kwargs["priors"], self.priors)
        np.testing.assert_array_equal(psd.calls[0][0], np.ones(NTEMPS))  # no ladder: no scaling

    def test_a_restart_is_not_seeded(self):
        self.backend_path.touch()
        restored = object()
        backend = mock.Mock(initialized=True, iteration=5)
        backend.get_last_sample.return_value = restored
        mbh = FakeInterface("mbh")
        with mock.patch.object(gf_run, "GFHDFBackend", return_value=backend):
            state = self.load({"mbh": info(mbh)})
        self.assertIs(state, restored)
        self.assertEqual(mbh.calls, [])


class PadLadderTest(unittest.TestCase):
    def test_pad(self):
        np.testing.assert_array_equal(pad_ladder([1.0, 0.3], 4), [1.0, 0.3, 0.3, 0.3])
        np.testing.assert_array_equal(pad_ladder(None, 2), [1.0, 1.0])


# --- residual rebuild --- #

T0, DT, N = 1000.0, 5.0, 40


class FakeACS:
    def __init__(self, nwalkers):
        self.acs = [types.SimpleNamespace(data=TDSignal(np.zeros((1, N)), TDSettings(N, DT, t0=T0))) for _ in range(nwalkers)]
        self.freed = 0

    def flatten(self):
        return np.array(self.acs, dtype=object)

    def free_gpu_memory(self):
        self.freed += 1


def span(amp, k0, n=4):
    """A template of ``n`` samples from sample ``k0`` (a source covering part of the data)."""
    return TDSignal(np.full((1, n), float(amp)), TDSettings(n, DT, t0=T0 + int(k0) * DT))


class RebuildTest(unittest.TestCase):
    def state(self):
        inds = np.zeros((2, 3, 2), bool)  # (ntemps, nwalkers, nleaves); the hot chain is ignored
        inds[0, :, 0] = True
        inds[0, 1, 1] = True
        inds[1] = True
        coords = np.zeros((2, 3, 2, 2))
        coords[0, :, 0] = [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]]  # (amp, start sample) per walker
        coords[0, 1, 1] = [10.0, 30.0]
        return types.SimpleNamespace(branches_inds={"mbh": inds}, branches_coords={"mbh": coords})

    def expected(self):
        out = np.zeros((3, N))
        out[:, 2:6] = -np.array([[1.0], [2.0], [3.0]])
        out[1, 30:34] -= 10.0
        return out

    def check(self, gen):
        acs = FakeACS(3)
        subtract_state_templates(acs, self.state(), {"mbh": gen})
        np.testing.assert_array_equal(np.array([ac.data.arr[0] for ac in acs.acs]), self.expected())
        self.assertEqual(acs.freed, 2)  # once per leaf

    def test_per_walker_calls(self):
        calls = []

        def gen(amp, k0):
            calls.append((amp, k0))
            return span(amp, k0)

        self.check(gen)
        self.assertEqual(len(calls), 4)  # one per active (walker, leaf), cold chain only

    def test_batched_generator(self):
        calls = []

        class Batched:
            supports_batch = True

            def __call__(self, amp, k0):
                calls.append(len(amp))
                return DomainBaseArray([span(a, k) for a, k in zip(amp, k0)])

        self.check(Batched())
        self.assertEqual(calls, [3, 1])  # one call per leaf, over its active walkers

    def test_signal_gen_batches_over_walkers(self):
        class Wave:
            xp = np

            def __init__(self):
                self.calls = 0

            def get_signals_for_residuals(self, amp, k0):
                self.calls += 1
                return DomainBaseArray([span(a, k) for a, k in zip(amp, k0)])

        wave = Wave()
        self.check(SignalGen(wave, "get_signals_for_residuals", None, {}))
        self.assertEqual(wave.calls, 2)


class GeneratorCopyTest(unittest.TestCase):
    def test_deepcopy_copies_an_unbuilt_handle_with_the_shared_objects(self):
        orbits = object.__new__(type("Orbits", (), {}))
        settings = {"init": {"orbits": orbits}, "general": {"orbits": orbits}}
        settings["gen"] = SignalGen(WaveformHandle(dict, settings["init"]), "keys", None, {})
        copied = copy.deepcopy(settings)  # one pass, as CurrentInfoGlobalFit does
        handle = copied["gen"].source
        self.assertIsNot(handle, settings["gen"].source)
        self.assertIs(handle.init_kwargs["orbits"], copied["general"]["orbits"])  # the engine's copy
        self.assertIsNot(handle.init_kwargs["orbits"], orbits)
        self.assertIs(copied["gen"].waveform, handle.get())  # built once, on the first use
        self.assertFalse(settings["gen"].source.built)

    def test_a_built_handle_is_not_copied(self):
        handle = WaveformHandle(dict, {})
        handle.get()
        with self.assertRaisesRegex(RuntimeError, "not copied"):
            copy.deepcopy(handle)


if __name__ == "__main__":
    unittest.main()
