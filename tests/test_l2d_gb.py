"""Tests for the L2D gb block: schema, run-level checks and the build of its GB setup.

The gb block is a source block set up like ``mbh.yaml``: ``pe_move`` names the GB special move and its knobs.
The build runs on the CPU, on a one-day STFT grid of two-hour segments; the move itself needs the band
machinery of a full run and is not built here.
"""

# Pre-existing env shims (see tests/test_quintic_response.py): the installed
# scienceplots raises AttributeError at import, and eryn.utils.plot uses a bare
# ``typing`` name without importing it. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import yaml
from pydantic import ValidationError

from lisatools.globalfit.stock.l2d import load_settings
from lisatools.globalfit.stock.l2d.blocks.gb import GBBlock, gb_move_kwargs
from lisatools.globalfit.stock.l2d.io import RunConfig, load_run
from lisatools.globalfit.stock.l2d.load_settings import load_blocks, load_general_setup

from tests.test_l2d_blocks import cpu_raw
from tests.test_l2d_load_settings import FakeL1

RUN_FOLDERS = {
    name: Path(__file__).resolve().parents[1] / "mojito_input" / name
    for name in ("gb_psd_galfor_dffr",)
}
CANDIDATES_FILE = "gb_candidates.hdf5"
MOJITO_REFERENCE_TIME_S = 97729089.327664
# * Two candidates inside the stock prior, at the catalogue's reference epoch.
CANDIDATE_COLUMNS = dict(
    Amplitude=[1.0e-22, 3.0e-23],
    GW22FrequencySSBFrame=[2.5e-3, 7.0e-3],
    GW22FrequencyDerivativeSourceFrame=[1.0e-17, -2.0e-18],
    TrueAnomaly=[1.0, 7.0],
    InclinationAngle=[0.4, 2.0],
    PolarisationAngle=[2.0, -1.0],
    RightAscension=[4.0, 0.5],
    Declination=[-0.2, 1.1],
    TimeReferenceSSBFrame=[MOJITO_REFERENCE_TIME_S, MOJITO_REFERENCE_TIME_S],
    ID=[0, 1],
)
# ? Sampling basis: ln A, f0 [mHz], fdot, phi0, cos iota, psi, alpha, sin delta.
SPREAD = [1.0e-3, 1.0e-9, 1.0e-19, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-3, 1.0e-3]
STFT_DOMAIN = {"type": "stft", "min_freq": 1e-4, "max_freq": 2.9e-2, "segment_duration": "2 h"}
REFERENCE_PE_MOVE_KWARGS = dict(
    stretch_probability=0.0, num_stationary_fim_steps=1, run_swaps=True, use_prior_removal=False,
    phase_maximize=False, ranks_needed=0,
)


def write_candidates(path: Path) -> None:
    """A candidates file of layout version 0.2: catalogue columns at the root, the search's in a group."""
    with h5py.File(path, "w") as handle:
        for column_name, values in CANDIDATE_COLUMNS.items():
            handle.create_dataset(column_name, data=np.asarray(values))
        handle.create_dataset("Search/SignalToNoiseRatio", data=[12.0, 8.0])


def gb_raw(block_changes=None, **data_changes):
    """A CPU run with a gb block on a one-day STFT grid."""
    block = {"type": "gb", "priors": {"class": "mojitolight.GBPrior"}, **(block_changes or {})}
    raw = cpu_raw({"gb": block})
    raw["domain"] = dict(STFT_DOMAIN)
    raw["data"] = {**raw["data"], "source_types": ["noise", "gb"], **data_changes}
    return raw


def gb_search(**changes):
    return {"search_type": "catalog", "search_path": CANDIDATES_FILE, "spread": SPREAD, **changes}


class _GBConfig(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.run_dir = Path(folder.name)
        write_candidates(self.run_dir / CANDIDATES_FILE)

    def config(self, block_changes=None, **data_changes) -> RunConfig:
        with warnings.catch_warnings():
            # ? A block without a search warns that its branch starts empty.
            warnings.simplefilter("ignore", UserWarning)
            return RunConfig.model_validate(
                gb_raw(block_changes, **data_changes), context={"run_dir": self.run_dir}
            )

    def assertConfigError(self, message: str, block_changes=None, **data_changes) -> None:
        with self.assertRaises(ValidationError) as caught:
            self.config(block_changes, **data_changes)
        self.assertIn(message, str(caught.exception))


class GBSchemaTest(_GBConfig):
    def test_defaults_are_the_reference_rj_prior_move(self):
        block = self.config().blocks["gb"]
        self.assertIsInstance(block, GBBlock)
        self.assertEqual((block.nleaves, block.num_prop_repeats), (8000, 50))
        # * The reference ladder: ratio 1.2, with the hottest rung at T = 1e4.
        self.assertEqual((block.tempering.ntemps, block.tempering.ratio, block.tempering.Tmax), (None, 1.2, 1e4))
        self.assertEqual(block.waveform.class_name, "GBGPU")
        self.assertEqual(block.pe_move.class_name, "GBSpecialRJPriorMove")
        self.assertEqual(block.pe_move.kwargs, REFERENCE_PE_MOVE_KWARGS)

    def test_proposals_belong_to_the_pe_move(self):
        self.assertConfigError("takes no inner_moves", {"inner_moves": [{"class": "StretchMove", "weight": 1.0}]})

    def test_pe_move_class_and_kwargs(self):
        self.assertConfigError("is not available here", {"pe_move": {"class": "TDMBHSpecialMove"}})
        for kwargs, message in (
            ({"acs": None}, "filled in by the builder"),
            ({"stretch_probabilty": 0.5}, "does not accept"),
            ({"debug_settings": {"mem_prob": True}}, "does not accept"),
            ({"debug_settings": True}, "must be a mapping"),
        ):
            with self.subTest(kwargs=kwargs):
                self.assertConfigError(message, {"pe_move": {"class": "GBSpecialRJPriorMove", "kwargs": kwargs}})

    def test_debug_settings_mapping_becomes_the_dataclass(self):
        from lisatools.globalfit.moves.gbdebug import GBDebugSettings

        pe_move = {"class": "GBSpecialRJPriorMove", "kwargs": {"debug_settings": {"mem_probe": True}}}
        kwargs = gb_move_kwargs(self.config({"pe_move": pe_move}).blocks["gb"])
        self.assertEqual(kwargs["debug_settings"], GBDebugSettings(mem_probe=True))

    def test_waveform_and_stft_knobs(self):
        block = self.config(
            {"waveform": {"class": "GBGPU", "runtime_kwargs": {"oversample": 2}}, "stft": {"n_side_bins": 12}}
        ).blocks["gb"]
        self.assertEqual((block.waveform.runtime_kwargs, block.stft), ({"oversample": 2}, {"n_side_bins": 12}))
        for block_changes, message in (
            ({"waveform": {"class": "GBGPU", "kwargs": {"t0": 0.0}}}, "fills GBGPU's constructor arguments"),
            ({"waveform": {"class": "GBGPU", "runtime_kwargs": {"dt": 5.0}}}, "filled in by the builder"),
            ({"stft": {"side_bins": 12}}, "does not accept"),
            ({"stft": {"band_alpha": 0.1}}, "filled in by the builder"),
        ):
            with self.subTest(block_changes=block_changes):
                self.assertConfigError(message, block_changes)

    def test_search(self):
        search = self.config(
            {"search": gb_search(max_resample_tries=2000, kwargs={"evolve_to_data_start": False})}
        ).blocks["gb"].search
        self.assertEqual((search.max_resample_tries, search.kwargs), (2000, {"evolve_to_data_start": False}))
        self.assertEqual(search.search_path, self.run_dir / CANDIDATES_FILE)
        for search_spec, message in (
            ({"search_type": "catalog", "spread": SPREAD}, "needs search_path"),
            ({"search_type": "prior"}, "is not available for the 'gb' block"),
            (gb_search(kwargs={"evolve_to_data_star": False}), "does not accept"),
            (gb_search(kwargs={"spread": 1.0}), "are search fields"),
        ):
            with self.subTest(search=search_spec):
                self.assertConfigError(message, {"search": search_spec})

    def test_the_data_holds_the_galaxy(self):
        self.assertConfigError("needs 'gb' in data.source_types", source_types=["noise"])

    def test_tmax_must_stay_above_the_second_hottest_rung(self):
        ntemps = self.config().sampler.ntemps
        second_hottest = 1.2 ** (ntemps - 2)
        self.assertConfigError(
            "above the second-hottest rung's temperature", {"tempering": {"ratio": 1.2, "Tmax": second_hottest}}
        )


class GBBuildTest(_GBConfig):
    """A real CPU GeneralSetup on the STFT grid (only the mojito file I/O is faked) and the gb block's build."""

    def build(self, block_changes):
        cfg = self.config(block_changes)
        with mock.patch.dict(load_settings.LOADERS, {"mojito_l1": FakeL1}):
            general = load_general_setup(cfg)
        return cfg, load_blocks(cfg, general)["gb"]

    def test_setup_takes_the_block(self):
        from gbgpu.gbgpu import GBGPU

        from lisatools.globalfit.searches import GBSearchInterface
        from lisatools.globalfit.stock.erebor import GBSetup

        _, build = self.build({
            "num_prop_repeats": 30,
            "waveform": {"class": "GBGPU", "runtime_kwargs": {"oversample": 2}},
            "stft": {"n_side_bins": 12},
            "pe_move": {"class": "GBSpecialRJPriorMove", "kwargs": {"stretch_probability": 0.25}},
            "search": gb_search(max_resample_tries=2000, kwargs={"evolve_to_data_start": False}),
        })
        setup = build.setup
        self.assertIsInstance(setup, GBSetup)
        self.assertIs(build.waveform_class, GBGPU)
        self.assertEqual((setup.nleaves_max, setup.nleaves_min), (8000, 0))
        self.assertEqual((setup.waveform.oversample, setup.stft.n_side_bins), (2, 12))
        # * The moves group records the pe_move knobs it has fields for.
        self.assertEqual((setup.moves.num_repeat_proposals, setup.moves.stretch_probability), (30, 0.25))
        self.assertEqual(build.metadata.source_type, "gb")

        interface = setup.search_interface
        self.assertIsInstance(interface, GBSearchInterface)
        self.assertEqual((interface.evolve_to_data_start, interface.max_resample_tries), (False, 2000))
        self.assertEqual(interface.search_path, str(self.run_dir / CANDIDATES_FILE))
        # * The candidates reach the sampling basis inside the setup's prior over the proposed bands.
        seed = interface.get_catalog_values(curr=None)
        self.assertEqual(seed.shape, (2, 8))
        self.assertTrue(np.isfinite(setup.priors["gb"].logpdf(seed)).all())

    def test_ladder(self):
        cfg, build = self.build({"search": gb_search()})
        ntemps = cfg.sampler.ntemps
        # * Without tempering: the setup's reference ladder, ratio 1.2 with the hottest rung at 1e-4.
        reference_betas = 1.0 / 1.2 ** np.arange(ntemps)
        reference_betas[-1] = 1.0e-4
        np.testing.assert_allclose(build.setup.betas, reference_betas, rtol=1e-12)

        _, build = self.build({"search": gb_search(), "tempering": {"ratio": 1.5}})
        np.testing.assert_allclose(build.setup.betas, 1.0 / 1.5 ** np.arange(ntemps), rtol=1e-12)

        # * Tmax replaces only the hottest rung.
        _, build = self.build({"search": gb_search(), "tempering": {"ratio": 1.2, "Tmax": 10.0}})
        np.testing.assert_allclose(build.setup.betas[:-1], 1.0 / 1.2 ** np.arange(ntemps - 1), rtol=1e-12)
        self.assertEqual(build.setup.betas[-1], 0.1)


class GBRunFolderTest(_GBConfig):
    """The gb run folders load as written, in CPU copies with the test's own candidates file."""

    def load(self, name: str) -> RunConfig:
        from tests.test_l2d_ports import on_cpu

        run = on_cpu(RUN_FOLDERS[name], self.run_dir)
        gb_yaml = run / "blocks" / "gb.yaml"
        raw = yaml.safe_load(gb_yaml.read_text())
        raw["search"]["search_path"] = str(self.run_dir / CANDIDATES_FILE)
        gb_yaml.write_text(yaml.safe_dump(raw, sort_keys=False))
        return load_run(run)

    def test_run_folder_loads_as_written(self):
        cfg = self.load("gb_psd_galfor_dffr")
        self.assertEqual(sorted(cfg.blocks), ["galfor", "gb", "psd"])
        self.assertEqual((cfg.domain.type, cfg.domain.segment_duration), ("stft", 86400.0))
        gb_block = cfg.blocks["gb"]
        self.assertEqual(gb_block.pe_move.kwargs, REFERENCE_PE_MOVE_KWARGS)
        self.assertEqual(
            (gb_block.search.max_resample_tries, gb_block.search.kwargs), (2000, {"evolve_to_data_start": False})
        )
        self.assertIsNone(cfg.blocks["galfor"].search)  # the psd search fits it


if __name__ == "__main__":
    unittest.main()
