"""SubmissionWriter: metadata is written before, and independently of, the sample post-processing."""

import glob
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.postprocessing import (
    PARAMETER_INFO_REGISTRY,
    OccupancyDetectionCriteria,
    RunMetadata,
    SourceMetadata,
    StochasticMetadata,
    SubmissionWriter,
)

NSAMP = 50


def _source_metadata(source_type):
    return SourceMetadata(
        source_type=source_type,
        frequency_ranges=[(1e-4, 1e-1)],
        waveform_model="test",
        waveform_model_code_link="",
        waveform_model_config={},
    )


def _fake_samples():
    rng = np.random.default_rng(0)
    out = {}
    for branch, nleaves in [("mbh", 2), ("gb", 3), ("psd", 1)]:
        ndim = len(PARAMETER_INFO_REGISTRY[branch])
        out[branch] = rng.uniform(1.0, 2.0, size=(NSAMP, nleaves, ndim))
    inds = {b: np.ones(s.shape[:2], dtype=bool) for b, s in out.items()}
    return out, {b: s.copy() for b, s in out.items()}, inds, np.zeros(NSAMP), np.zeros(NSAMP)


def _make_writer(parent, run_metadata=None):
    """Build a SubmissionWriter without an HDF backend or a full `curr`."""
    writer = object.__new__(SubmissionWriter)
    writer.backend = SimpleNamespace(branch_names=["mbh", "gb", "psd"])
    writer._curr = SimpleNamespace(
        source_metadata={
            "mbh": _source_metadata("MBHB"),
            "gb": _source_metadata("GB"),
            "psd": StochasticMetadata(model_config={}, frequency_ranges=[(1e-4, 1e-1)]),
        },
        general_info=SimpleNamespace(sensitivity_backend=None),
    )
    writer._ess_submission = NSAMP
    writer.detection_criteria = OccupancyDetectionCriteria()
    writer.run_metadata = run_metadata or RunMetadata(
        global_fit_version="test_v1",
        submission_parent_folder=parent,
        submission_timestamp="2026-01-01T000000",
        found_source_types_list=["GB", "MBHB", "NOISE"],
        searched_source_types_list=["GB", "MBHB", "NOISE"],
    )
    writer.save_data_and_residuals = lambda acs: None
    writer.process_samples = lambda ess, return_inds: _fake_samples()
    return writer


def _ar1(nsteps, nwalkers, tau, ncols, seed=0):
    rng = np.random.default_rng(seed)
    phi = (tau - 1.0) / (tau + 1.0)
    x = np.empty((nsteps, nwalkers, ncols))
    x[0] = rng.normal(size=(nwalkers, ncols))
    for i in range(1, nsteps):
        x[i] = phi * x[i - 1] + np.sqrt(1.0 - phi**2) * rng.normal(size=(nwalkers, ncols))
    return x


class KeptStageTest(unittest.TestCase):
    """Postprocessing reads the chain from the start of the last recipe step: no burn-in or tuning samples."""

    NWALKERS, NTEMPS, NDIM = 16, 2, 3
    BURN_IN, KEPT = 400, 1200  # iterations

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "backend.h5")

    def write_backend(self, steps=(("burn-in", 0), ("pe", 400)), started=("burn-in", "pe")):
        """A run file whose cold chain drifts down from 50 in the burn-in and has ACT 4 afterwards."""
        import h5py

        from lisatools.globalfit import recipe as gf_recipe
        from lisatools.globalfit.hdfbackend import GFHDFBackend

        nsteps = self.BURN_IN + self.KEPT
        backend = GFHDFBackend(self.path)
        backend.reset(self.NWALKERS, [self.NDIM], ntemps=self.NTEMPS, nleaves_max=[1], branch_names=["mbh"])
        backend.grow(nsteps, None)
        cold = _ar1(nsteps, self.NWALKERS, 4.0, self.NDIM)
        cold[: self.BURN_IN] += np.linspace(50.0, 0.0, self.BURN_IN)[:, None, None]
        with h5py.File(self.path, "a") as f:
            group = f[backend.name]
            chain = np.zeros((nsteps, self.NTEMPS, self.NWALKERS, 1, self.NDIM))
            chain[:, 0, :, 0] = cold
            # on file: [step, sampler, temperature, walker, ...]
            group["chain"]["mbh"][:nsteps, 0] = chain
            group["inds"]["mbh"][:nsteps] = True
            group["log_like"][:nsteps, 0] = np.arange(nsteps, dtype=float)[:, None, None]  # the iteration, to check
            group.attrs["iteration"] = nsteps
        if steps:
            recipe = gf_recipe.Recipe()
            for name, _start in steps:
                recipe.add_recipe_component(gf_recipe.PERecipeStep(moves=[]), name=name)
            backend.add_recipe(recipe)
            for name, start in steps:
                if name in started:
                    backend.recipe_step_start(name, start)
        return cold

    def consumer(self):
        from lisatools.globalfit.postprocessing import BackendConsumer

        return BackendConsumer(backend=self.path)  # by path: the run's group is found ("global_fit")

    def test_only_the_last_step_is_kept(self):
        cold = self.write_backend()
        consumer = self.consumer()
        self.assertEqual(consumer.kept_stage_start(), self.BURN_IN)
        consumer.store_cold_chains()
        np.testing.assert_array_equal(consumer.cold_chains["mbh"][:, 0, :, 0], cold[self.BURN_IN:])
        # the ACT is that of the kept stage (4), not of a chain with the burn-in's drift in it
        act = consumer.compute_auto_correlation_time()["mbh"]
        self.assertEqual(act.shape, (self.NDIM,))
        np.testing.assert_allclose(act, 4.0, atol=1.0)
        consumer.store_independent_samples(ess=10**6)
        thinned = consumer.thinned_chains["mbh"]
        self.assertEqual(thinned.shape[0], self.NWALKERS * len(range(0, self.KEPT, 4)))  # thinned by 4
        self.assertLess(np.abs(thinned).max(), 10.0)  # no burn-in sample (they sit up to 50 away)
        self.assertEqual(consumer._thinned_log_likelihood.min(), float(self.BURN_IN))

    def test_discard_counts_from_the_kept_stage(self):
        self.write_backend()
        consumer = self.consumer()
        consumer.store_independent_samples(discard=200, ess=10**6)
        self.assertEqual(consumer._thinned_log_likelihood.min(), float(self.BURN_IN + 200))

    def test_whole_chain_without_recorded_steps(self):
        for label, kwargs in [
            ("no recipe", dict(steps=())),
            ("a file from before the starts were stored", dict(started=())),
            ("the last step has not started", dict(started=("burn-in",))),
        ]:
            with self.subTest(label):
                if os.path.exists(self.path):
                    os.remove(self.path)
                self.write_backend(**kwargs)
                consumer = self.consumer()
                self.assertEqual(consumer.kept_stage_start(), 0)
                consumer.store_cold_chains()
                self.assertEqual(consumer.cold_chains["mbh"].shape[0], self.BURN_IN + self.KEPT)

    def test_an_undefined_act_thins_by_one(self):
        cold = self.write_backend()
        import h5py

        with h5py.File(self.path, "a") as f:  # one walker never moves in one parameter
            f["global_fit"]["chain"]["mbh"][:, 0, 0, 3, 0, 1] = 7.0
        consumer = self.consumer()
        with self.assertLogs("lisatools.globalfit.postprocessing", level="WARNING") as logs:
            act = consumer.compute_auto_correlation_time()["mbh"]
        self.assertIn("NaN values found in ACT", "\n".join(logs.output))
        self.assertEqual(act[1], 1.0)


class SubmissionWriterDecouplingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.parent = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _folder(self, writer):
        return writer.submission_folder

    def _metadata_json(self, writer, name):
        path = os.path.join(
            self._folder(writer), f"test_Erebor_v1_{name}_2026-01-01T000000.json"
        )
        with open(path) as f:
            return json.load(f)

    def _posterior_files(self, writer, name):
        return glob.glob(os.path.join(self._folder(writer), "*", f"*_{name}_posteriors_*.h5"))

    def test_happy_path(self):
        writer = _make_writer(self.parent)
        # GB clustering needs a GPU; replace it with the identity
        writer._prepare_gb_samples = lambda acs, orig, samples, inds: (samples, inds)
        writer.write_submission(acs=None)

        for name in ["MBHB", "GB", "noise"]:
            self.assertEqual(len(self._posterior_files(writer, name)), 1)
        self.assertEqual(len(self._metadata_json(writer, "MBHB")["posterior_files"]), 2)
        self.assertEqual(len(self._metadata_json(writer, "GB")["posterior_files"]), 3)
        with open(os.path.join(self._folder(writer), "global_metadata.json")) as f:
            self.assertEqual(json.load(f)["found_source_types_list"], ["GB", "MBHB", "NOISE"])
        for name in ["MBHB", "GB", "noise"]:
            self.assertNotIn("pending", self._metadata_json(writer, name)["comment"])

    def test_a_class_among_the_waveform_kwargs_does_not_cost_the_metadata(self):
        # few writes the trajectory class into inspiral_kwargs["func"] of the kwargs it is built with; whatever
        # ends up in the config, the h5 attributes and the json file are written
        import h5py

        writer = _make_writer(self.parent)
        writer._prepare_gb_samples = lambda acs, orig, samples, inds: (samples, inds)
        config = {
            "waveform_kwargs": {"inspiral_kwargs": {"func": SimpleNamespace, "err": np.float64(1e-8)}},
            "grid": np.arange(3), "order": 25,
        }
        writer._curr.source_metadata["mbh"].waveform_model_config = config
        writer.write_submission(acs=None)

        expected = {
            "waveform_kwargs": {"inspiral_kwargs": {"func": "types.SimpleNamespace", "err": 1e-8}},
            "grid": [0, 1, 2], "order": 25,
        }
        self.assertEqual(self._metadata_json(writer, "MBHB")["waveform_model_config"], expected)
        (h5_path,) = glob.glob(os.path.join(self._folder(writer), "*_MBHB_2026*.h5"))
        with h5py.File(h5_path, "r") as f:
            self.assertEqual(json.loads(f.attrs["waveform_model_config"]), expected)
        self.assertEqual(len(self._posterior_files(writer, "MBHB")), 1)
        self.assertIs(config["waveform_kwargs"]["inspiral_kwargs"]["func"], SimpleNamespace)  # the config is left

    def _found(self, writer):
        with open(os.path.join(self._folder(writer), "global_metadata.json")) as f:
            return json.load(f)["found_source_types_list"]

    def _offline_writer(self, writer):
        """A new writer on the existing submission, as built by scripts/postprocess_submission.py."""
        run_metadata = RunMetadata.from_submission(
            os.path.join(self._folder(writer), "global_metadata.json")
        )
        return _make_writer(self.parent, run_metadata=run_metadata)

    def test_deferred_branch_is_written_later(self):
        writer = _make_writer(self.parent)
        writer.write_submission(acs=None, defer=["gb"])

        self.assertEqual(self._posterior_files(writer, "GB"), [])
        self.assertIn("posteriors pending", self._metadata_json(writer, "GB")["comment"])
        self.assertEqual(self._found(writer), ["MBHB", "NOISE"])
        mbh_files = self._posterior_files(writer, "MBHB")
        mbh_mtime = os.path.getmtime(mbh_files[0])
        mbh_metadata = self._metadata_json(writer, "MBHB")

        offline = self._offline_writer(writer)
        offline._prepare_gb_samples = lambda acs, orig, samples, inds: (samples, inds)
        offline.write_posteriors(acs=None, branches=["gb"])

        self.assertEqual(offline.submission_folder, writer.submission_folder)
        self.assertEqual(len(self._posterior_files(writer, "GB")), 1)
        gb_metadata = self._metadata_json(writer, "GB")
        self.assertEqual(len(gb_metadata["posterior_files"]), 3)
        self.assertNotIn("pending", gb_metadata["comment"])
        self.assertEqual(self._found(writer), ["GB", "MBHB", "NOISE"])
        # the branches that were already written are left untouched
        self.assertEqual(self._posterior_files(writer, "MBHB"), mbh_files)
        self.assertEqual(os.path.getmtime(mbh_files[0]), mbh_mtime)
        self.assertEqual(self._metadata_json(writer, "MBHB"), mbh_metadata)

    def test_deferring_psd_defers_the_stochastic_component(self):
        writer = _make_writer(self.parent)
        writer._prepare_gb_samples = lambda acs, orig, samples, inds: (samples, inds)
        writer.write_submission(acs=None, defer=["psd"])

        self.assertEqual(self._posterior_files(writer, "noise"), [])
        self.assertIn("posteriors pending", self._metadata_json(writer, "noise")["comment"])
        self.assertEqual(self._found(writer), ["GB", "MBHB"])

    def test_failed_rerun_drops_the_source_type(self):
        writer = _make_writer(self.parent)
        writer._prepare_gb_samples = lambda acs, orig, samples, inds: (samples, inds)
        writer.write_submission(acs=None)
        self.assertEqual(self._found(writer), ["GB", "MBHB", "NOISE"])

        def _boom(*args, **kwargs):
            raise ValueError("new clustering exploded")

        offline = self._offline_writer(writer)
        offline._prepare_gb_samples = _boom
        with self.assertRaisesRegex(RuntimeError, "gb: postprocessing failed"):
            offline.write_posteriors(acs=None, branches=["gb"])

        self.assertEqual(self._found(writer), ["MBHB", "NOISE"])
        self.assertIn("new clustering exploded", self._metadata_json(writer, "GB")["comment"])

    def test_branch_failure_is_isolated(self):
        writer = _make_writer(self.parent)

        def _boom(*args, **kwargs):
            raise ValueError("clustering exploded")

        writer._prepare_gb_samples = _boom

        with self.assertRaisesRegex(RuntimeError, "gb: postprocessing failed"):
            writer.write_submission(acs=None)

        folder = self._folder(writer)
        with open(os.path.join(folder, "global_metadata.json")) as f:
            run_md = json.load(f)
        self.assertEqual(run_md["found_source_types_list"], ["MBHB", "NOISE"])

        gb_md = self._metadata_json(writer, "GB")
        self.assertIn("clustering exploded", gb_md["comment"])
        self.assertEqual(gb_md["posterior_files"], [])

        self.assertEqual(len(self._posterior_files(writer, "MBHB")), 1)
        self.assertEqual(len(self._posterior_files(writer, "noise")), 1)

    def test_sample_processing_failure_keeps_metadata(self):
        writer = _make_writer(self.parent)

        def _boom(ess, return_inds):
            raise FloatingPointError("ACT is NaN")

        writer.process_samples = _boom

        with self.assertRaisesRegex(RuntimeError, "3 unit"):
            writer.write_submission(acs=None)

        folder = self._folder(writer)
        with open(os.path.join(folder, "global_metadata.json")) as f:
            self.assertEqual(json.load(f)["found_source_types_list"], [])
        for name in ["MBHB", "GB", "noise"]:
            self.assertIn("ACT is NaN", self._metadata_json(writer, name)["comment"])
        self.assertEqual(glob.glob(os.path.join(folder, "*", "*_posteriors_*.h5")), [])

    def test_unregistered_branch_does_not_block_the_others(self):
        # 'sobh' has a source-type name but no parameter registry / source metadata entry
        writer = _make_writer(self.parent)
        writer.backend.branch_names = ["mbh", "gb", "psd", "sobh"]
        writer._prepare_gb_samples = lambda acs, orig, samples, inds: (samples, inds)
        saved = []
        writer.save_data_and_residuals = lambda acs: saved.append(True)

        with self.assertRaisesRegex(RuntimeError, "sobh: postprocessing failed"):
            writer.write_submission(acs=None)

        self.assertEqual(saved, [True])
        for name in ["MBHB", "GB", "noise"]:
            self.assertEqual(len(self._posterior_files(writer, name)), 1)


if __name__ == "__main__":
    unittest.main()
