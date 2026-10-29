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


def _make_writer(parent):
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
    writer.run_metadata = RunMetadata(
        global_fit_version="test_v1",
        submission_parent_folder=parent,
        submission_timestamp="2026-01-01T000000",
        found_source_types_list=["GB", "MBHB", "NOISE"],
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


if __name__ == "__main__":
    unittest.main()
