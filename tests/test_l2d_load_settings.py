"""Tests for ``lisatools.globalfit.stock.l2d.load_settings`` (RunConfig -> GeneralSettings / GeneralSetup)."""

# Pre-existing env shims (see tests/test_quintic_response.py): the installed
# scienceplots raises AttributeError at import, and eryn.utils.plot uses a bare
# ``typing`` name without importing it. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import dataclasses
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np

from lisatools.detector import L1Orbits
from lisatools.domains import FDSettings, STFTSettings
from lisatools.globalfit.engine import GeneralSettings, GeneralSetup
from lisatools.globalfit.preprocessing import BaseProcessingStep, L1ProcessingStep
from lisatools.globalfit.stock.l2d import load_settings
from lisatools.globalfit.stock.l2d.io import RunConfig
from lisatools.globalfit.stock.l2d.load_settings import load_general_setup, make_general_settings
from lisatools.sensitivity import CompositeSensitivityBackend, XYZSensitivityBackend
from lisatools.utils.constants import YRSID_SI

# YAML key -> GeneralSettings field, per pass-through section. This is the contract
# load_settings.make_general_settings implements; the coverage test below fails when a
# YAML field is added without being mapped there.
MAPPED = {
    "run": {
        "name": "base_file_name",
        "output_dir": "file_store_dir",
        "submission_dir": "submission_parent_folder",
        "seed": "random_seed",
        "num_iterations": "num_iterations",
        "backup_iter": "backup_iter",
    },
    "sampler": {"nwalkers": "nwalkers", "ntemps": "ntemps"},
    "compute": {"gpus": "gpus", "backend": "gpu_backend"},
    "window": {
        "type": "window_type",
        "taper_duration": "window_taper_duration",
        "normalize": "normalize_window",
    },
    "metadata": {
        "codename": "global_fit_codename",
        "version": "global_fit_version",
        "contact": "global_fit_contact",
        "code_link": "global_fit_code_link",
        "input_data_link": "input_data_link",
        "input_reference": "input_reference",
        "noise_model": "noise_model",
        "noise_model_code_link": "noise_model_code_link",
        "comment": "comment",
    },
}
# YAML fields of those sections that deliberately don't go into GeneralSettings.
NOT_GENERAL_SETTINGS = {"compute": {"head_rank", "main_rank"}}  # -> RankInfo
# Values that are converted on the way (checked by dedicated tests below).
CONVERTED = {("run", "output_dir"), ("run", "submission_dir"), ("window", "taper_duration")}


def run_1_raw(**changes):
    """A ``run_1``-shaped config as a dict; every metadata value distinct to catch swaps."""
    raw = dict(
        schema_version=1,
        run=dict(
            name="full_run_1_fd", output_dir="out", submission_dir="results",
            seed=1434768955, num_iterations=500, backup_iter=5,
        ),
        sampler=dict(nwalkers=20, ntemps=5),
        compute=dict(gpus=[0], backend="cuda12x"),
        data=dict(
            path="/data/mojito_light_2p5s/",
            source_types=["gb", "mbhb", "noise"],
            source_ids={"mbhb": [18, 5, 16]},
            orbits={"type": "l1", "frame": "icrs", "armlength": 2493162305.42235},
            Tobs="9 months",
            dt=5.0,
            highpass={"cutoff": 1e-5},
            lowpass={"cutoff": 1e-1},
            downsample={"window": ["kaiser", 31.0]},
            trim={"duration": 0.02, "is_percent": True},
        ),
        domain=dict(type="fd", min_freq=1e-4, max_freq=2.9e-2),
        window=dict(type="tukey", taper_duration="auto", normalize=True),
        sensitivity={
            "class": "XYZSensitivityBackend",
            "kwargs": {
                "tdi_generation": 2,
                "mask_percentage": 0.02,
                "average_transfer_functions": True,
                "galactic_grid_kwargs": {"R_d": 2.18, "z_d": 0.48},
            },
            "fixed_psd_kwargs": {"psd_params": [15e-12, 3e-15]},
        },
        metadata=dict(
            codename="erebor", version="CDL1run1_v4", contact="contact@x", code_link="code-link",
            input_data_link="data-link", input_reference="mojito light", noise_model="parametric",
            noise_model_code_link="noise-link", comment="a comment",
        ),
    )
    for section, value in changes.items():
        raw[section] = value
    return raw


class _Config(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run_dir = Path(tmp.name)

    def config(self, **changes):
        return RunConfig.model_validate(run_1_raw(**changes), context={"run_dir": self.run_dir})

    def settings(self, **changes):
        cfg = self.config(**changes)
        return cfg, make_general_settings(cfg)


class MappingTest(_Config):
    def test_every_yaml_field_is_mapped(self):
        cfg = self.config()
        for section, mapping in MAPPED.items():
            with self.subTest(section=section):
                fields = set(type(getattr(cfg, section)).model_fields)
                expected = set(mapping) | NOT_GENERAL_SETTINGS.get(section, set())
                self.assertEqual(
                    fields, expected,
                    f"{section}: YAML fields changed; map new ones in "
                    "load_settings.make_general_settings and update MAPPED here.",
                )

    def test_targets_are_general_settings_fields(self):
        general = {f.name for f in dataclasses.fields(GeneralSettings)}
        for section, mapping in MAPPED.items():
            with self.subTest(section=section):
                self.assertLessEqual(set(mapping.values()), general)

    def test_values_reach_their_targets(self):
        cfg, settings = self.settings()
        for section, mapping in MAPPED.items():
            for key, target in mapping.items():
                if (section, key) in CONVERTED:
                    continue
                with self.subTest(field=f"{section}.{key}"):
                    self.assertEqual(getattr(settings, target), getattr(getattr(cfg, section), key))

    def test_data_fields(self):
        _, settings = self.settings()
        self.assertAlmostEqual(settings.Tobs, 0.75 * YRSID_SI, delta=1e-6)
        self.assertEqual(settings.dt, 5.0)
        self.assertIs(settings.data_processor_class, L1ProcessingStep)


class ConversionTest(_Config):
    def test_file_store_dir_is_a_str_with_trailing_separator(self):
        _, settings = self.settings()
        self.assertIsInstance(settings.file_store_dir, str)
        self.assertEqual(settings.file_store_dir, str(self.run_dir / "out") + os.sep)
        # what the engine builds from it by concatenation
        self.assertEqual(
            Path(settings.file_store_dir + settings.base_file_name + "_artifacts/"),
            self.run_dir / "out" / "full_run_1_fd_artifacts",
        )

    def test_submission_folder(self):
        _, settings = self.settings()
        self.assertEqual(settings.submission_parent_folder, str(self.run_dir / "results"))
        _, settings = self.settings(run=dict(name="r", output_dir="out"))
        self.assertIsNone(settings.submission_parent_folder)

    def test_auto_taper_is_one_over_min_freq(self):
        _, settings = self.settings()
        self.assertAlmostEqual(settings.window_taper_duration, 1.0 / 1e-4)

    def test_explicit_taper_is_kept(self):
        _, settings = self.settings(window=dict(type="tukey", taper_duration="1 day"))
        self.assertEqual(settings.window_taper_duration, 86_400.0)

    def test_sensitivity(self):
        cfg, settings = self.settings()
        self.assertIs(settings.sensitivity_backend_class, XYZSensitivityBackend)
        self.assertEqual(settings.sensitivity_init_kwargs, cfg.sensitivity.kwargs)
        self.assertEqual(settings.fixed_psd_kwargs, cfg.sensitivity.fixed_psd_kwargs)

    def test_config_is_not_mutated_through_the_settings(self):
        # GeneralSetup does galactic_grid_kwargs.setdefault("t0", ...) on its (shallow) copy.
        cfg, settings = self.settings()
        before = deepcopy(cfg.sensitivity.kwargs)
        settings.sensitivity_init_kwargs["galactic_grid_kwargs"]["t0"] = 123.0
        settings.fixed_psd_kwargs["psd_params"].append(0.0)
        self.assertEqual(cfg.sensitivity.kwargs, before)
        self.assertEqual(cfg.sensitivity.fixed_psd_kwargs, {"psd_params": [15e-12, 3e-15]})


class PreprocessTest(_Config):
    STEPS = ("highpass_kwargs", "lowpass_kwargs", "downsample_kwargs", "trim_kwargs")

    def test_present_steps(self):
        _, settings = self.settings()
        pre = settings.preprocess_kwargs
        self.assertEqual(pre["highpass_kwargs"], {"cutoff": 1e-5, "order": 2, "zero_phase": True})
        self.assertEqual(pre["lowpass_kwargs"], {"cutoff": 1e-1, "order": 2, "zero_phase": True})
        self.assertEqual(pre["downsample_kwargs"], {"target_fs": 1 / 5.0, "window": ("kaiser", 31.0)})
        self.assertIsInstance(pre["downsample_kwargs"]["window"], tuple)
        self.assertEqual(
            pre["trim_kwargs"], {"duration": 0.02, "is_percent": True, "trimming_type": "from_each_end"}
        )
        self.assertAlmostEqual(pre["Tobs"], 0.75 * YRSID_SI, delta=1e-6)

    def test_absent_steps_are_an_explicit_none(self):
        # Explicit None overrides GeneralSetup's default high-pass and 200 h trim.
        data = {k: v for k, v in run_1_raw()["data"].items()
                if k not in ("highpass", "lowpass", "downsample", "trim")}
        _, settings = self.settings(data=data)
        for step in self.STEPS:
            with self.subTest(step=step):
                self.assertIn(step, settings.preprocess_kwargs)
                self.assertIsNone(settings.preprocess_kwargs[step])


class ProcessorTest(_Config):
    def test_init_kwargs(self):
        _, settings = self.settings()
        init = settings.processor_init_kwargs
        self.assertEqual(init["L1_folder"], "/data/mojito_light_2p5s")
        self.assertEqual(init["source_types"], ["gb", "mbhb", "noise"])
        self.assertEqual(init["source_ids"], {"mbhb": [18, 5, 16]})
        self.assertIs(init["orbits_class"], L1Orbits)
        self.assertEqual(
            init["orbits_kwargs"],
            {"frame": "icrs", "armlength": 2493162305.42235, "force_backend": "cuda12x"},
        )

    def test_defaults(self):
        data = dict(run_1_raw()["data"], orbits={"type": "l1"})
        data.pop("source_ids")
        _, settings = self.settings(data=data, compute=dict(backend="cpu"))
        init = settings.processor_init_kwargs
        self.assertEqual(init["source_ids"], {})  # the loader calls .get() on it
        self.assertEqual(init["orbits_kwargs"], {"frame": "icrs", "force_backend": "cpu"})


class DomainTest(_Config):
    times = np.arange(int(10 * 86400 / 5.0)) * 5.0

    def build(self, settings):
        return settings.domain_settings(times=self.times, dt=5.0, force_backend="cpu")

    def test_fd(self):
        _, settings = self.settings()
        domain = self.build(settings)
        self.assertIsInstance(domain, FDSettings)
        self.assertEqual((domain.min_freq_input, domain.max_freq_input), (1e-4, 2.9e-2))

    def test_stft(self):
        _, settings = self.settings(
            domain=dict(type="stft", min_freq=1e-4, max_freq=2.9e-2, segment_duration="1 day")
        )
        domain = self.build(settings)
        self.assertIsInstance(domain, STFTSettings)
        self.assertEqual(domain.dt, 86_400.0)
        self.assertEqual((domain.min_freq_input, domain.max_freq_input), (1e-4, 2.9e-2))


class FakeL1(BaseProcessingStep):
    """Stands in for L1ProcessingStep: same constructor kwargs, 2 days of white noise, no files.

    Longer than the test's Tobs (1 day), like a real L1 file: process() cuts the data to
    Tobs with trim(keep_from_start), which raises if Tobs >= the data length.
    """

    received = None

    def __init__(self, L1_folder, source_types, source_ids, orbits_class, orbits_kwargs):
        FakeL1.received = dict(
            L1_folder=L1_folder, source_types=source_types, source_ids=source_ids,
            orbits_class=orbits_class, orbits_kwargs=orbits_kwargs,
        )
        dt = 5.0
        times = np.arange(int(2 * 86_400 / dt)) * dt
        data = np.random.default_rng(1).normal(size=(3, times.size)) * 1e-20
        BaseProcessingStep.__init__(self, times, data, 1.0 / dt, verbose=False)
        self.orbits = None  # GeneralSetup falls back to EqualArmlengthOrbits
        self.catalogue = {}


class GeneralSetupEndToEndTest(_Config):
    """A real GeneralSetup on CPU; only the mojito file I/O is faked."""

    def test_load_general_setup(self):
        data = {k: v for k, v in run_1_raw()["data"].items()
                if k not in ("highpass", "lowpass", "downsample", "trim")}
        data["Tobs"] = "1 day"
        cfg = self.config(
            data=data,
            compute=dict(backend="cpu"),
            window=dict(type="tukey", taper_duration="auto"),
            sensitivity={"class": "CompositeSensitivityBackend"},
        )
        with mock.patch.dict(load_settings.LOADERS, {"mojito_l1": FakeL1}):
            setup = load_general_setup(cfg)

        self.assertIsInstance(setup, GeneralSetup)
        self.assertEqual(FakeL1.received["L1_folder"], "/data/mojito_light_2p5s")
        self.assertEqual(FakeL1.received["orbits_kwargs"]["force_backend"], "cpu")
        self.assertIsInstance(setup.domain_settings, FDSettings)
        self.assertEqual(setup.start_freq, 1e-4)
        # Absent preprocessing stayed off (no default 200 h trim): the data are only cut to Tobs.
        self.assertEqual(setup.Tobs, 86_400.0)
        self.assertEqual(setup.data_dt, 5.0)
        self.assertAlmostEqual(setup.window_alpha, 1e4 / 86_400.0)
        self.assertIsInstance(setup.sensitivity_backend, CompositeSensitivityBackend)
        self.assertEqual(setup.force_backend, "cpu")
        self.assertTrue(Path(setup.artifacts_file_dir).is_dir())
        self.assertEqual(Path(setup.artifacts_file_dir).parent, self.run_dir / "out")


if __name__ == "__main__":
    unittest.main()
