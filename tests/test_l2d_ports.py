"""The settings files ported to run folders (``gf_dev/interface/<name>/``) load, build and dry-run.

The folders keep the cluster's paths, GPUs and full year of data. These tests run them on CPU:
the data is ``FakeL1`` (one day of white noise), ``compute`` is CPU, and the preprocessing (written
for the real files) is dropped. Everything else is the folder as written. ``few`` is not always
installed, so ``EMRITDIWaveform`` is the stand-in from ``test_l2d_sources`` (looked up, never
constructed); the priors are the real ereborpriors classes.

The ``REFERENCE_*`` values are transcribed from ``mojito_input/emri_mbh_psd_settings.py``, which
cannot be imported here (it sets a CUDA device and starts flow trainers).
"""

# Pre-existing env shims (see tests/test_quintic_response.py): the installed
# scienceplots raises AttributeError at import, and eryn.utils.plot uses a bare
# ``typing`` name without importing it. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import contextlib
import io
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import eryn.flows.torch  # noqa: F401 -- before any sys.modules patch: torch cannot be imported twice
import numpy as np
import yaml
from eryn.moves.tempering import make_ladder

from lisatools.globalfit.stock.l2d import load_settings
from lisatools.globalfit.stock.l2d.io import load_run
from lisatools.globalfit.stock.l2d.load_settings import build_curr_info
from lisatools.utils.constants import YRSID_SI

from tests.test_l2d_load_settings import FakeL1
from tests.test_l2d_run import fake_mpi4py, load_script
from tests.test_l2d_sources import EMRI_WAVEFORM_MODULE, make_fake_emri_waveform_module

INTERFACE = Path(__file__).resolve().parents[1] / "gf_dev" / "interface"
EMRI_MBH_PSD = INTERFACE / "emri_mbh_psd"

REFERENCE_MBH_WAVEFORM = dict(
    waveform_kwargs=dict(
        higher_modes=[21, 33, 44], include_negative_modes=True, t_low_fit=True, coarse_grain=False,
        atol=1e-12, rtol=1e-12,
    ),
    tdi_generation="2nd generation", tdi_channels="XYZ", order=30,
    Tobs=1.0 / 12.0 * YRSID_SI, start_freq=7e-5, use_reference_time=True, buffer_time=15_000, fft_batch_size=2,
)
REFERENCE_EMRI_WAVEFORM = dict(
    waveform_kwargs=dict(
        sum_kwargs={"pad_output": True},
        inspiral_kwargs={"DENSE_STEPPING": 0, "buffer_length": int(1e2), "err": 1e-8},
        amplitude_kwargs={},
    ),
    tdi_generation="2nd generation", tdi_channels="XYZ", order=30,
    buffer_time=15_000, fft_batch_size=1, run_async=True,
)
# setup_recipe's move kwargs, less what the recipe fills in
REFERENCE_PE_MOVES = {
    "psd": ("MultiGPUPSDMove", dict(permute_every=100, live_dangerously=True, run_async=True, run_threaded=True)),
    "mbh": ("TDMBHSpecialMove", dict(
        permute_every=100, pad_out_of_prior=True, run_async=True, run_threaded=True, randomize_split=True,
    )),
    "emri": ("EMRISpecialMove", dict(
        permute_every=25, pad_out_of_prior=True, run_async=True, run_threaded=True, randomize_split=True,
        batch_size_per_gpu=5,
    )),
}
REFERENCE_INNER_MOVES = {
    "mbh": [
        ("StretchMove", 0.35), ("DEMove", 0.3), ("SkyMove", 0.05), ("SkyMove", 0.05), ("SkyMove", 0.05),
        ("ConditionalFlowMove", 0.2),
    ],
    "emri": [("StretchMove", 0.4), ("DEMove", 0.4), ("ConditionalFlowMove", 0.2)],
}
REFERENCE_FLOW_TRAINER = dict(  # flow_move.executor_init_kwargs, less the paths and the device
    epochs_per_round=150, min_train_samples=1500, refit_transform_every=1, torch_num_threads=2, plot_corner=True,
    fit_kwargs=dict(
        batch_size=1024, lr=1e-3, lr_annealing=True, optimizer="adamw", patience=30, validation_fraction=0.15,
        val_split="temporal", train_noise=0.0,
    ),
    seed=103209,
)
REFERENCE_MAX_BUFFER_SAMPLES = {"mbh": 6000, "emri": 4000}


def on_cpu(folder: Path, root: Path) -> Path:
    """A copy of ``folder`` that runs here: CPU, FakeL1's day of data, no preprocessing, local output."""
    run = root / folder.name
    shutil.copytree(folder, run, ignore=shutil.ignore_patterns("out"))
    raw = yaml.safe_load((run / "global.yaml").read_text())
    for step in ("highpass", "lowpass", "downsample", "trim"):
        raw["data"].pop(step, None)
    raw["data"].update(path="/data/nowhere/", Tobs="1 day")
    raw["compute"] = {"backend": "cpu"}
    raw["run"]["output_dir"] = "out"
    (run / "global.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return run


class _Port(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for patcher in (
            mock.patch.dict(_sys.modules, {EMRI_WAVEFORM_MODULE: make_fake_emri_waveform_module()}),
            mock.patch.dict(load_settings.LOADERS, {"mojito_l1": FakeL1}),
            mock.patch.dict(_sys.modules, fake_mpi4py(rank=0)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)


class EmriMbhPsdTest(_Port):
    def test_folder_loads_as_written_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # e.g. a source block without search
            cfg = load_run(EMRI_MBH_PSD)
        self.assertEqual(sorted(cfg.blocks), ["emri", "mbh", "psd"])
        self.assertEqual((cfg.sampler.nwalkers, cfg.sampler.ntemps), (24, 4))
        self.assertNotIn(cfg.compute.flow_gpu, cfg.compute.gpus)  # which GPUs is per run, not the port's
        self.assertEqual(cfg.data.source_ids, {"emri": [0, 3], "mbhb": [18, 5, 16, 7, 2, 12]})
        self.assertEqual(cfg.data.Tobs, YRSID_SI)
        self.assertEqual(cfg.recipe.class_name, "StockRecipe")

    def test_dry_run_on_cpu(self):
        stdout = io.StringIO()
        with warnings.catch_warnings(), contextlib.redirect_stdout(stdout):
            warnings.simplefilter("ignore")
            code = load_script().main(["--config", str(on_cpu(EMRI_MBH_PSD, self.root)), "--dry-run"])
        self.assertEqual(code, 0)
        out = stdout.getvalue()
        name = load_run(EMRI_MBH_PSD).run.name  # per run, not the port's
        for line in (
            f"dry run: {name} built, nothing sampled",
            "emri: 2 leaves x 12 params, ladder 1, search catalog, flow ZukoFlow on 12 params, pe_move EMRISpecialMove",
            "mbh: 6 leaves x 11 params, ladder 4, search catalog, flow ZukoFlow on 11 params, pe_move TDMBHSpecialMove",
            "psd: 1 leaves x 2 params, ladder 4, no search, pe_move MultiGPUPSDMove",
            "recipe: StockRecipe",
        ):
            self.assertIn(line, out)

    def test_built_settings_match_the_settings_file(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = load_run(on_cpu(EMRI_MBH_PSD, self.root))
            curr = build_curr_info(cfg)
        info = curr.source_info

        np.testing.assert_allclose(info["mbh"].betas, 1 / 1.2 ** np.arange(4))
        np.testing.assert_allclose(info["emri"].betas, [1.0])
        np.testing.assert_allclose(info["psd"].betas, make_ladder(2, ntemps=4, Tmax=1e6))
        self.assertEqual({n: info[n].num_prop_repeats for n in info}, {"emri": 50, "mbh": 50, "psd": 500})
        self.assertEqual((info["mbh"].nleaves_max, info["mbh"].nleaves_min), (6, 6))
        self.assertEqual((info["emri"].nleaves_max, info["emri"].nleaves_min), (2, 2))

        for name, reference in (("mbh", REFERENCE_MBH_WAVEFORM), ("emri", REFERENCE_EMRI_WAVEFORM)):
            with self.subTest(waveform=name):
                given = {k: v for k, v in info[name].initialize_kwargs.items() if k in reference}
                self.assertEqual(given, reference)
                self.assertEqual(info[name].initialize_kwargs["waveform_t0"], 97729089.327664)
        self.assertEqual(info["mbh"].waveform_kwargs, {})
        self.assertEqual(info["emri"].waveform_kwargs, {"mode_selection_threshold": 1e-4})

        for name, reference in REFERENCE_INNER_MOVES.items():
            with self.subTest(inner_moves=name):
                self.assertEqual([(type(m).__name__, w) for m, w in info[name].inner_moves], reference)
        self.assertEqual([m.which for m, _ in info["mbh"].inner_moves[2:5]], ["lat", "long", "both"])

        for name, nleaves, ndim in (("mbh", 6, 11), ("emri", 2, 12)):
            with self.subTest(flow=name):
                flow_move = info[name].inner_moves[-1][0]
                flow = flow_move.flow
                self.assertEqual((flow.dims, flow.conditioning.nleaves_max, flow.seed), (ndim, nleaves, 103209))
                self.assertEqual(
                    (flow.flow_class, flow.periodic_aliases), ("NSF", 1)
                )
                self.assertTrue(flow.data_transform.periodic_in_cholesky)
                self.assertFalse(flow.data_transform.shared)
                periodic = {n: p for n, p in info[name].periodic[name].items()}
                basis = info[name].priors[name].get_input_basis()
                self.assertEqual(flow.data_transform.periodic, {basis.index(n): (0.0, p) for n, p in periodic.items()})
                trainer = {k: v for k, v in flow_move.executor_init_kwargs.items()
                           if k not in ("save_path", "diagnostics_dir", "worker_device", "max_buffer_samples")}
                self.assertEqual(trainer, REFERENCE_FLOW_TRAINER)
                self.assertEqual(flow_move.executor_init_kwargs["max_buffer_samples"], REFERENCE_MAX_BUFFER_SAMPLES[name])

        for name, (class_name, kwargs) in REFERENCE_PE_MOVES.items():
            with self.subTest(pe_move=name):
                self.assertEqual((cfg.blocks[name].pe_move.class_name, cfg.blocks[name].pe_move.kwargs), (class_name, kwargs))

        mbh_search, emri_search = cfg.blocks["mbh"].search, cfg.blocks["emri"].search
        self.assertEqual(mbh_search.search_type, "catalog")
        self.assertEqual(mbh_search.spread, [1e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1])
        self.assertEqual((emri_search.search_type, emri_search.spread), ("catalog", 1e-6))

        g = curr.general_info
        self.assertEqual((g.start_freq, g.end_freq, g.random_seed), (1e-4, 2.9e-2, 103209))
        self.assertEqual(g.window_taper_duration, 1 / 1e-4)

    def test_flow_on_a_parameter_subset_builds_a_flow_gaussian_move(self):
        from types import SimpleNamespace

        from eryn.moves import ConditionalFlowMove, FlowGaussianMove

        from lisatools.globalfit.stock.l2d.blocks.base import BuildContext, make_flow_move

        run = on_cpu(EMRI_MBH_PSD, self.root)
        mbh_yaml = run / "blocks" / "mbh.yaml"
        raw = yaml.safe_load(mbh_yaml.read_text())
        params = ["logM", "logq", "s1z", "s2z", "d_L", "phi_ref", "t_c"]
        raw["flow"].update(parameters=params, gaussian={"adapt_cov": True, "mode": "am"})
        mbh_yaml.write_text(yaml.safe_dump(raw, sort_keys=False))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = load_run(run)

        # the flow move alone (the full build needs the data backend): a basis holding the flow's names
        basis = ["logM", "logq", "s1z", "s2z", "d_L", "t_c", "phi_ref", "cos_i", "lam", "sin_beta", "psi"]
        info = {"basis": basis, "periodic": {"mbh": {"phi_ref": 2 * np.pi, "lam": 2 * np.pi}}}
        ctx = BuildContext(cfg=cfg, general=SimpleNamespace(artifacts_file_dir=str(self.root)))
        move = make_flow_move(cfg.blocks["mbh"], ctx, info, nleaves=6)

        self.assertIsInstance(move, FlowGaussianMove)
        np.testing.assert_array_equal(move.flow_params, np.isin(basis, params))
        self.assertEqual(move.flow.dims, len(params))
        flow_names = [n for n in basis if n in params]  # the flow's columns are in basis order
        self.assertEqual(move.flow.data_transform.periodic, {flow_names.index("phi_ref"): (0.0, 2 * np.pi)})
        self.assertTrue(move.adapt_cov)
        self.assertEqual(move.flow_prob, 0.5)  # random choice of half by default
        self.assertIsNone(move.gibbs_sampling_setup)
        self.assertIn("worker_device", move.executor_init_kwargs)

        # the flow on every parameter stays a plain ConditionalFlowMove
        move = make_flow_move(cfg.blocks["emri"], ctx, {"basis": basis, "periodic": None}, nleaves=2)
        self.assertIs(type(move), ConditionalFlowMove)
        self.assertEqual(move.flow.dims, len(basis))

    def test_flow_gaussian_needs_parameters(self):
        from pydantic import ValidationError

        from lisatools.globalfit.stock.l2d.blocks.base import FlowSpec

        with self.assertRaisesRegex(ValidationError, "give parameters too"):
            FlowSpec(weight=1.0, model={"class": "ZukoFlow"}, gaussian={"adapt_cov": True})
        with self.assertRaisesRegex(ValidationError, "filled in by the builder"):
            FlowSpec(weight=1.0, parameters=["a"], model={"class": "ZukoFlow"}, gaussian={"flow_params": [True]})


if __name__ == "__main__":
    unittest.main()
