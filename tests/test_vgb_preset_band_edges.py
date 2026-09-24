"""Tests for the PRESET VGB band-edge builder + its VGBSetup wiring.

Covers :func:`lisatools.globalfit.stock.erebor.vgb.vgb_preset_band_edges`
(source-anchored windows, odd-index parity, filler feasibility, the
installed support-separation gate) and the ``VGB_BAND_EDGES_MODE=preset``
dispatch in :class:`VGBSetup.init_band_structure`. All fixtures are toy
scale (a handful of sources, small WDM grids) -- never the production
geometry.

Style follows ``tests/test_gb_band_edges_get_n.py``.
"""

import copy
import os
import pickle
import unittest

import numpy as np
from gbgpu.utils.utility import get_N

from lisatools.globalfit.moves.gbbands import check_band_support_separation
from lisatools.globalfit.stock.erebor.vgb import (
    VGBSettings,
    VGBSetup,
    vgb_preset_band_edges,
)
from lisatools.utils.constants import YRSID_SI

# 3-month-like scale (same convention as test_gb_band_edges_get_n).
TOBS = YRSID_SI / 4.0
LAYER_DF = 1.393e-4
DF = 1.0 / TOBS

# Toy source set: an ultra-close pair (supports overlap -> must share a
# window), a loose triple, and two isolated sources.
F0 = np.array([8.0e-4, 8.001e-4, 1.5e-3, 1.52e-3, 1.54e-3, 3.0e-3, 5.0e-3])
SPAN = dict(start_freq=4.0e-4, end_freq=6.0e-3)


def build(f0=F0, **kw):
    kw.setdefault("start_freq", SPAN["start_freq"])
    kw.setdefault("end_freq", SPAN["end_freq"])
    kw.setdefault("width_cap_layers", 8.0)
    return vgb_preset_band_edges(f0, TOBS, LAYER_DF, **kw)


class PresetBandEdgesTest(unittest.TestCase):
    def test_deterministic_and_order_invariant(self):
        e1 = build()
        e2 = build()
        e3 = build(f0=F0[::-1].copy())
        np.testing.assert_array_equal(e1, e2)
        np.testing.assert_array_equal(e1, e3)

    def test_shape_and_source_placement(self):
        edges, meta = build(return_meta=True)
        self.assertTrue(np.all(np.diff(edges) > 0))
        nb = len(edges) - 1
        band_of = np.searchsorted(edges, F0, side="right") - 1
        # every source interior and inside a window band
        self.assertTrue(np.all((band_of >= 1) & (band_of <= nb - 2)))
        self.assertTrue(
            set(band_of.tolist()) <= set(meta["window_band_inds"].tolist())
        )
        # first/last bands are empty guards
        self.assertNotIn(0, band_of)
        self.assertNotIn(nb - 1, band_of)

    def test_windows_all_odd_one_stride2_unit(self):
        edges, meta = build(return_meta=True)
        winds = meta["window_band_inds"]
        self.assertTrue(np.all(winds % 2 == 1))
        # -> exactly one populated unit under band_index % 2
        self.assertEqual(len(np.unique(winds % 2)), 1)

    def test_passes_installed_separation_gate(self):
        edges = build()  # validate=True already enforces; check explicitly
        out = check_band_support_separation(
            edges, TOBS, 2, enforce=True, context="preset test"
        )
        self.assertTrue(out["passes"])

    def test_overlapping_supports_share_a_window(self):
        edges = build()
        band_of = np.searchsorted(edges, F0[:2], side="right") - 1
        # the 8.0e-4 / 8.001e-4 pair (gap << get_N/Tobs) must share a band
        self.assertEqual(band_of[0], band_of[1])

    def test_width_cap_bounds_bands_without_must_merge(self):
        edges, meta = build(width_cap_layers=4.0, return_meta=True)
        widths = np.diff(edges)
        # toy clusters are all far narrower than the cap, so no band may
        # exceed the cap (plus one straddle layer of slack in span terms)
        self.assertLessEqual(meta["max_band_span_layers"], 4 + 1)
        self.assertTrue(np.all(widths <= 4.0 * LAYER_DF * (1 + 1e-9)))

    def test_every_band_wide_enough_for_stride2(self):
        # local rule behind the gate: each band's width must cover the
        # edge half-supports of the adjacent same-unit pair
        edges = build()
        s = np.array([
            float(get_N(1e-30, f, TOBS, oversample=4).item()) * DF
            for f in edges[1:]
        ])
        widths = np.diff(edges)
        for b in range(len(widths) - 2):
            self.assertGreaterEqual(
                widths[b + 1], (s[b] + s[b + 1]) * (1 - 1e-9)
            )

    def test_guard_room_errors_are_loud(self):
        with self.assertRaises(ValueError):
            build(start_freq=F0.min())  # no room for the leading guard
        with self.assertRaises(ValueError):
            build(end_freq=F0.max())  # no room for the trailing guard

    def test_meta_covers_all_sources(self):
        _, meta = build(return_meta=True)
        covered = np.concatenate(meta["window_sources"])
        self.assertEqual(sorted(covered.tolist()), list(range(len(F0))))


class PresetSettingsWiringTest(unittest.TestCase):
    def test_settings_fields_and_env(self):
        s = VGBSettings()
        self.assertEqual(s.band_edges_mode, "uniform")  # default unchanged
        self.assertEqual(s.preset_width_cap_layers, 8.0)
        self.assertEqual(s.preset_support_margin, 1.0)
        env = {
            "VGB_BAND_EDGES_MODE": "preset",
            "VGB_PRESET_WIDTH_CAP_LAYERS": "12",
            "VGB_PRESET_SUPPORT_MARGIN": "2.0",
        }
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            s2 = VGBSettings()
            self.assertEqual(s2.band_edges_mode, "preset")
            self.assertEqual(s2.preset_width_cap_layers, 12.0)
            self.assertEqual(s2.preset_support_margin, 2.0)
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_settings_pickle_deepcopy(self):
        s = VGBSettings(band_edges_mode="preset")
        s2 = pickle.loads(pickle.dumps(copy.deepcopy(s)))
        self.assertEqual(s2.band_edges_mode, "preset")

    def _preset_setup_instance(self):
        """Minimal VGBSetup carrier for _init_preset_band_structure.

        Built without running the full Setup __init__ (which needs the
        whole sampling stack); only the attributes the band-structure
        method reads are populated -- toy WDM grid, tiny source table.
        """
        from lisatools.domains import WDMSettings

        inst = VGBSetup.__new__(VGBSetup)
        inst.domain_settings = WDMSettings(
            Nf=64, Nt=1024, dt=15.0, min_freq=2e-4, max_freq=8e-3
        )
        inst.Tobs = 64 * 1024 * 15.0
        inst.oversample = 4
        inst.band_layers = 1
        inst.band_unit_stride = 2
        inst.band_edges_mode = "preset"
        inst.preset_width_cap_layers = 8.0
        inst.preset_support_margin = 1.0
        inst.sample_distance = False
        # fixed basis [f0 (mHz), alpha, sin_delta]
        f0_mhz = np.array([1.0, 1.02, 3.0, 5.0])
        inst.fixed_params = np.column_stack(
            [f0_mhz, np.zeros(4), np.zeros(4)]
        )
        inst.start_freq = 5e-4
        inst.end_freq = 6e-3
        inst.fdot_lims = None

        class _Log:
            def info(self, *a, **k):
                pass

        inst.logger = _Log()
        return inst

    def test_setup_preset_mode_builds_band_structure(self):
        inst = self._preset_setup_instance()
        VGBSetup.init_band_structure(inst)
        self.assertTrue(np.all(np.diff(inst.band_edges) > 0))
        self.assertEqual(inst.num_sub_bands, len(inst.band_edges) - 1)
        self.assertEqual(len(inst.band_N_vals), inst.num_sub_bands)
        f0_hz = np.asarray(inst.fixed_params)[:, 0] * 1e-3
        band_of = np.searchsorted(inst.band_edges, f0_hz, side="right") - 1
        self.assertTrue(np.all(band_of % 2 == 1))
        self.assertTrue(
            (inst.f0_lims[0] < f0_hz.min())
            and (inst.f0_lims[1] > f0_hz.max())
        )
        # preset pins the stride-2 design when VGB_BAND_UNIT_STRIDE unset
        self.assertEqual(inst.band_unit_stride, 2)
        # fdot_lims derived (not explicitly set -> filled in)
        self.assertEqual(len(inst.fdot_lims), 2)

    def test_setup_preset_mode_preserves_explicit_fdot_lims(self):
        inst = self._preset_setup_instance()
        inst.fdot_lims = [-1e-14, 1e-14]
        VGBSetup.init_band_structure(inst)
        self.assertEqual(inst.fdot_lims, [-1e-14, 1e-14])

    def test_setup_preset_mode_requires_fixed_params(self):
        inst = self._preset_setup_instance()
        inst.fixed_params = None
        with self.assertRaises(ValueError):
            VGBSetup.init_band_structure(inst)

    def test_setup_preset_mode_requires_wdm(self):
        from lisatools.domains import FDSettings

        inst = self._preset_setup_instance()
        inst.domain_settings = FDSettings(N=4096, df=1.0 / inst.Tobs)
        with self.assertRaises(NotImplementedError):
            VGBSetup.init_band_structure(inst)


if __name__ == "__main__":
    unittest.main()
