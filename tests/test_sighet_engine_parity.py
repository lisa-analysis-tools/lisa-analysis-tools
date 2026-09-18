"""GB and VGB build the SAME sig-het in-model engine from the same knobs.

Background (2026-09-17): the 6mo campaign had ``VGB_SIGHET_INMODEL=0``
("accuracy at the loudest-VGB SNRs unverified -- [GB_CELL_LL] growth in
smoke 1"), which routed every VGB information matrix through the chunked
engine at 30-45 ms/source (vgb_pe 130 s per call). The VGB wiring turned
out to pass NO ``tukey_alpha`` to ``for_band_engine`` -- ``VGBSettings``
had no ``sighet_tukey_alpha`` field -- so the engine inherited the chunked
delegate's 0.05 per-chunk stitching fraction as a WHOLE-observation taper
(~110 WDM layers per side at 6mo against a 60-layer crop; the engine only
warns). Both branches now build through ``erebor.gb.build_sighet_engine``.

These tests pin the parity structurally, without a GPU or a data build:

* every ``sighet_*`` field of the GB block exists on the VGB block;
* the ``for_band_engine`` kwargs derived from the two blocks are IDENTICAL
  under the same environment, and always carry an explicit ``tukey_alpha``;
* the shared build-time checks refuse the silently-degrading configurations
  (v5 without v4 knots / band; a taper the time crop does not exclude).
"""

import dataclasses
import os
import types
import unittest
from unittest import mock

from lisatools.globalfit.stock.erebor.gb import (
    check_sighet_build_config,
    sighet_engine_kwargs,
)
from lisatools.globalfit.stock.erebor.variants.gb_no_fg import GBNoFgGBSettings
from lisatools.globalfit.stock.erebor.vgb import VGBSettings

_SIGHET_ENV_PREFIXES = ("SIGHET_", "GB_SIGHET_", "VGB_SIGHET_")


def _sighet_env_cleared():
    """Context: every sig-het env knob removed, so defaults are the defaults."""
    drop = {k: None for k in os.environ if k.startswith(_SIGHET_ENV_PREFIXES)}
    return mock.patch.dict(os.environ, drop, clear=False) if drop else _noop()


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_comp(Nt, ind_min_t):
    return types.SimpleNamespace(
        wdm_settings=types.SimpleNamespace(Nt=Nt, ind_min_t=ind_min_t)
    )


class SigHetFieldParityTest(unittest.TestCase):
    def test_vgb_block_carries_every_gb_sighet_field(self):
        gb = {f.name for f in dataclasses.fields(GBNoFgGBSettings)
              if f.name.startswith("sighet_")}
        vgb = {f.name for f in dataclasses.fields(VGBSettings)
               if f.name.startswith("sighet_")}
        self.assertTrue(
            gb <= vgb,
            f"VGBSettings is missing GB sig-het fields: {sorted(gb - vgb)}",
        )
        # The field whose absence caused the 0.05 taper inheritance.
        self.assertIn("sighet_tukey_alpha", vgb)


class SigHetEngineKwargsTest(unittest.TestCase):
    def test_defaults_identical_and_tukey_alpha_explicit(self):
        with _sighet_env_cleared():
            gb_kw = sighet_engine_kwargs(GBNoFgGBSettings())
            vgb_kw = sighet_engine_kwargs(VGBSettings())
        self.assertEqual(gb_kw, vgb_kw)
        self.assertIn("tukey_alpha", vgb_kw)
        self.assertEqual(vgb_kw["tukey_alpha"], 0.01)
        # The production stack: v3 nodes + v4 knots + v5 on.
        self.assertEqual(vgb_kw["v3_n_nodes"], 64)
        self.assertEqual(vgb_kw["v4_knots"], 128)
        self.assertEqual(vgb_kw["v4_band"], 16)
        self.assertEqual(vgb_kw["v5"], 1)

    def test_campaign_env_reaches_both_branches_identically(self):
        env = {
            "SIGHET_TUKEY_ALPHA": "0.02",
            "SIGHET_NT_LAYER": "120",
            "SIGHET_N_CP": "256",
            "SIGHET_N_SPARSE_FD": "512",
            "SIGHET_MAX_R": "0",
        }
        with _sighet_env_cleared(), mock.patch.dict(os.environ, env):
            gb_kw = sighet_engine_kwargs(GBNoFgGBSettings())
            vgb_kw = sighet_engine_kwargs(VGBSettings())
        self.assertEqual(gb_kw, vgb_kw)
        self.assertEqual(vgb_kw["tukey_alpha"], 0.02)
        self.assertEqual(vgb_kw["nt_layer"], 120)
        self.assertEqual(vgb_kw["n_cp_build"], 256)
        self.assertEqual(vgb_kw["n_sparse_fd"], 512)

    def test_v5_off_drops_the_kwarg(self):
        with _sighet_env_cleared(), mock.patch.dict(os.environ, {"SIGHET_V5": "0"}):
            kw = sighet_engine_kwargs(VGBSettings())
        self.assertNotIn("v5", kw)


class SigHetBuildConfigCheckTest(unittest.TestCase):
    """The 6mo campaign numbers: Nt ~ 4380 layers, EDGE_CROP_WAVELETS=60."""

    def test_pinned_alpha_passes_the_edge_exclusion_invariant(self):
        with _sighet_env_cleared():
            info = VGBSettings()
        # alpha 0.01 -> taper 22 layers + 8 margin = 30 <= 60.
        check_sighet_build_config(info, _fake_comp(4380, 60), branch="vgb")

    def test_inherited_delegate_alpha_is_refused(self):
        with _sighet_env_cleared(), mock.patch.dict(
            os.environ, {"SIGHET_TUKEY_ALPHA": "0.05"}
        ):
            info = VGBSettings()
        # alpha 0.05 -> taper 110 layers + 8 = 118 > 60: the configuration
        # the VGB branch silently ran under before the shared builder.
        with self.assertRaises(ValueError) as cm:
            check_sighet_build_config(info, _fake_comp(4380, 60), branch="vgb")
        msg = str(cm.exception)
        self.assertIn("[vgb]", msg)
        self.assertIn("110", msg)
        self.assertIn("EDGE_CROP_WAVELETS", msg)

    def test_no_wdm_domain_skips_the_taper_check(self):
        with _sighet_env_cleared(), mock.patch.dict(
            os.environ, {"SIGHET_TUKEY_ALPHA": "0.05"}
        ):
            info = GBNoFgGBSettings()
        check_sighet_build_config(
            info, types.SimpleNamespace(wdm_settings=None), branch="gb")

    def test_v5_requires_v4_knots(self):
        with _sighet_env_cleared(), mock.patch.dict(
            os.environ, {"SIGHET_V5": "1", "SIGHET_V4_KNOTS": "0"}
        ):
            info = GBNoFgGBSettings()
        with self.assertRaises(ValueError) as cm:
            check_sighet_build_config(info, _fake_comp(4380, 60), branch="gb")
        self.assertIn("SIGHET_V4_KNOTS", str(cm.exception))
        self.assertIn("[gb]", str(cm.exception))

    def test_v5_arena_requires_a_band(self):
        with _sighet_env_cleared(), mock.patch.dict(
            os.environ, {"SIGHET_V5": "1", "SIGHET_V4_BAND": "0"}
        ):
            info = VGBSettings()
        with self.assertRaises(ValueError) as cm:
            check_sighet_build_config(info, _fake_comp(4380, 60), branch="vgb")
        self.assertIn("SIGHET_V4_BAND", str(cm.exception))

    def test_v5_control_arm_needs_no_band(self):
        with _sighet_env_cleared(), mock.patch.dict(
            os.environ, {"SIGHET_V5": "2", "SIGHET_V4_BAND": "0"}
        ):
            info = VGBSettings()
        check_sighet_build_config(info, _fake_comp(4380, 60), branch="vgb")


if __name__ == "__main__":
    unittest.main()
