"""WDM sub-box templates: settings slicing, sub-box add/subtract, and
container slicing to a template's box (MBH batched windowed likelihood,
2026-09-29). CPU, toy grid."""
from __future__ import annotations

import copy
import pickle
import unittest

import numpy as np

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix

NF, NT, DT = 32, 128, 10.0
N = NF * NT


def _chirp(lo_layer, hi_layer):
    """Three-channel chirp EXACTLY zero outside layers [lo_layer, hi_layer)."""
    from scipy.signal.windows import tukey

    t = np.arange(N) * DT
    x = np.zeros(N)
    lo, hi = lo_layer * NF, hi_layer * NF
    tt = t[lo:hi]
    x[lo:hi] = tukey(hi - lo, alpha=0.3) * np.sin(
        2 * np.pi * (4e-3 * tt + 0.5 * 1.6e-2 * (tt - tt[0]) ** 2 / (tt[-1] - tt[0]))
    )
    return np.stack([x, 0.5 * x, 0.25 * x])


def _wdm(**kw):
    return WDMSettings(NF, NT, DT, force_backend="cpu", **kw)


class WDMSettingsGetSliceTest(unittest.TestCase):
    def test_slice_narrows_box_and_round_trips(self):
        s = _wdm(min_freq=2e-3, max_freq=2e-2)
        sub = s.get_slice((slice(2, 6), slice(40, 80)))
        self.assertEqual((sub.ind_min_f, sub.ind_max_f), (s.ind_min_f + 2, s.ind_min_f + 5))
        self.assertEqual((sub.ind_min_t, sub.ind_max_t), (40, 79))
        self.assertTrue(sub.eq_without_inds(s))
        # every WDMSignal rebuilds its settings from (args, kwargs): the box must survive
        rebuilt = WDMSettings(*sub.args, **sub.kwargs)
        self.assertEqual(rebuilt, sub)
        again = pickle.loads(pickle.dumps(copy.deepcopy(sub)))
        self.assertEqual(again, sub)

    def test_slice_from_zero_is_allowed(self):
        sub = _wdm().get_slice((slice(0, NF), slice(0, 10)))
        self.assertEqual((sub.ind_min_f, sub.ind_min_t, sub.ind_max_t), (0, 0, 9))

    def test_bad_index_raises(self):
        s = _wdm()
        with self.assertRaises(ValueError):
            s.get_slice((slice(0, 4, 2), slice(0, 4)))
        with self.assertRaises(ValueError):
            s.get_slice((slice(3, 3), slice(0, 4)))
        with self.assertRaises(ValueError):
            s.get_slice(slice(0, 4))


class WDMSubBoxAddTest(unittest.TestCase):
    def test_add_then_subtract_sub_box_restores_residual(self):
        full = _wdm()
        rng = np.random.default_rng(1)
        data = WDMSignal(rng.normal(size=(3, NF, NT)), full)
        before = data.arr.copy()
        box = full.get_slice((slice(0, NF), slice(40, 80)))
        tmpl = WDMSignal(rng.normal(size=(3, NF, 40)), box)
        data.add_signal(tmpl, sign=+1)
        self.assertFalse(np.allclose(data.arr[..., 40:80], before[..., 40:80]))
        np.testing.assert_array_equal(data.arr[..., :40], before[..., :40])
        np.testing.assert_array_equal(data.arr[..., 80:], before[..., 80:])
        data.add_signal(tmpl, sign=-1)
        np.testing.assert_allclose(data.arr, before, rtol=0, atol=1e-12)

    def test_same_box_add_is_unchanged(self):
        full = _wdm()
        data = WDMSignal(np.zeros((3, NF, NT)), full)
        data.add_signal(WDMSignal(np.ones((3, NF, NT)), full), sign=+1)
        np.testing.assert_array_equal(data.arr, np.ones((3, NF, NT)))

    def test_box_outside_data_raises(self):
        full = _wdm(min_time=50 * NF * DT)  # data box starts at layer 50
        data = WDMSignal(np.zeros((3, NF, full.Nt_active)), full)
        box = _wdm().get_slice((slice(0, NF), slice(40, 80)))  # starts at 40 < 50
        with self.assertRaises(ValueError):
            data.add_signal(WDMSignal(np.zeros((3, NF, 40)), box))

    def test_shifted_grid_raises(self):
        data = WDMSignal(np.zeros((3, NF, NT)), _wdm())
        other = WDMSettings(NF, NT, DT, t0=12345.0, force_backend="cpu").get_slice(
            (slice(0, NF), slice(40, 80))
        )
        with self.assertRaises(ValueError):
            data.add_signal(WDMSignal(np.zeros((3, NF, 40)), other))


class SliceWDMToTemplateTest(unittest.TestCase):
    def test_sub_box_likelihood_equals_full_grid(self):
        full = _wdm(min_freq=2e-3)
        h_full = TDSignal(_chirp(45, 75), TDSettings(N, DT, force_backend="cpu")).transform(full)
        # template restricted to layers [40, 80): zero the rest on the full grid
        h_arr = np.array(h_full.arr, copy=True)
        h_arr[..., :40] = 0.0
        h_arr[..., 80:] = 0.0
        h_trunc = WDMSignal(h_arr, full)
        rng = np.random.default_rng(2)
        d = WDMSignal(h_arr + 1e-1 * rng.normal(size=h_arr.shape), full)
        ac = AnalysisContainer(d, XYZ2SensitivityMatrix(full, model="scirdv1"))
        ref_dh = complex(ac.template_inner_product(h_trunc))
        ref_ll = float(np.real(ac.template_likelihood(h_trunc)))

        box = full.get_slice((slice(0, full.Nf_active), slice(40, 80)))
        h_box = WDMSignal(h_arr[..., 40:80], box)
        d_box, t_box, s_box = ac._slice_to_template(h_box)
        self.assertEqual(d_box.arr.shape, h_box.arr.shape)
        self.assertEqual(tuple(s_box.invC.shape[-2:]), tuple(h_box.arr.shape[-2:]))
        self.assertIs(t_box, h_box)
        np.testing.assert_allclose(
            complex(ac.template_inner_product(h_box)), ref_dh, rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(
            float(np.real(ac.template_likelihood(h_box))), ref_ll, rtol=1e-12, atol=0
        )

    def test_box_outside_data_raises(self):
        full = _wdm(min_time=50 * NF * DT)
        d = WDMSignal(np.zeros((3, NF, full.Nt_active)), full)
        ac = AnalysisContainer(d, XYZ2SensitivityMatrix(full, model="scirdv1"))
        box = _wdm().get_slice((slice(0, NF), slice(40, 80)))
        with self.assertRaises(ValueError):
            ac._slice_to_template(WDMSignal(np.zeros((3, NF, 40)), box))


if __name__ == "__main__":
    unittest.main()
