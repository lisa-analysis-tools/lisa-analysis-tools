"""L2D ``time_reference``: what a source block's waveforms (and so its sampled times/phases) are referenced to."""

import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import types
import unittest
import warnings

from lisatools.globalfit.stock.l2d.blocks.base import (
    BuildContext, engine_waveform_kwargs, waveform_metadata_config,
)
from lisatools.globalfit.stock.l2d.io import RunConfig

from tests.test_l2d_sources import EMRI_BLOCK, MBH_BLOCK, REFERENCE_TIME, _Sources, mbh_with_search, source_raw

DATA_T0 = REFERENCE_TIME + 720850.5  # a trimmed data start


def fake_general():
    return types.SimpleNamespace(
        data_t0=DATA_T0, data_td_settings="td", dt=5.0, gpu_orbits="gpu orbits", orbits="orbits",
        force_backend="cpu", gpu_backend="cuda12x", window_alpha=0.1, stft_dt=None, start_freq=1e-4,
        end_freq=1e-1,
    )


class _TimeReference(_Sources):
    def config(self, blocks):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # source blocks without a search warn
            return RunConfig.model_validate(source_raw(blocks), context={"run_dir": self.run_dir})


class TimeReferenceTest(_TimeReference):
    def test_default_is_the_mojito_reference_time(self):
        cfg = self.config({"mbh": MBH_BLOCK, "emri": EMRI_BLOCK})
        ctx = BuildContext(cfg, fake_general())
        for name in ("mbh", "emri"):
            self.assertEqual(cfg.blocks[name].time_reference, "mojito")
            self.assertEqual(cfg.blocks[name].waveform_t0(ctx), REFERENCE_TIME)

    def test_data_start_for_every_source_block(self):
        blocks = {
            "mbh": {**MBH_BLOCK, "time_reference": "data_start"},
            "emri": {**EMRI_BLOCK, "time_reference": "data_start"},
        }
        cfg = self.config(blocks)
        ctx = BuildContext(cfg, fake_general())
        for name in blocks:
            self.assertEqual(cfg.blocks[name].waveform_t0(ctx), DATA_T0)
            self.assertEqual(engine_waveform_kwargs(ctx, cfg.blocks[name])["waveform_t0"], DATA_T0)

    def test_unknown_value(self):
        self.assertConfigError(
            "literal_error", "blocks.mbh.mbh.time_reference",
            blocks={"mbh": {**MBH_BLOCK, "time_reference": "start"}},
        )

    def test_catalog_search_needs_the_mojito_time(self):
        self.assertConfigError(
            "value_error", "blocks.mbh.mbh", msg="needs time_reference: mojito",
            blocks={"mbh": {**mbh_with_search(), "time_reference": "data_start"}},
        )
        self.config({"mbh": mbh_with_search()})  # the default is fine

    def test_search_results_are_taken_as_they_are(self):
        search = {"search_type": "point_estimate", "search_path": ".", "spread": 1e-3}
        cfg = self.config({"mbh": {**MBH_BLOCK, "search": search, "time_reference": "data_start"}})
        self.assertEqual(cfg.blocks["mbh"].time_reference, "data_start")

    def test_metadata_names_the_anchor(self):
        init = {"waveform_t0": DATA_T0, "orbits": object(), "data_td_settings": object(), "order": 30}
        self.assertEqual(
            waveform_metadata_config(init, "data_start"),
            {"waveform_t0": DATA_T0, "order": 30, "force_backend": "cpu", "time_reference": "data_start"},
        )
        self.assertEqual(waveform_metadata_config(init, "mojito")["time_reference"], "MOJITO_REFERENCE_TIME")


if __name__ == "__main__":
    unittest.main()
