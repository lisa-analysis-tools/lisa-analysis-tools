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
