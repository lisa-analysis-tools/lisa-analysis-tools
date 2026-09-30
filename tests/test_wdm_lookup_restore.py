"""WDMLookupTable is back in domains.py (restored from LAT f17d9a74).

* fdot-off ``n_ref_complex`` build == ``n_ref_only`` build (Re <-> cos, Im <-> sin);
* an out-of-support frequency raises ValueError (the historical code dropped into
  ``breakpoint()``);
* optional machine-precision regression against the committed sprint-root table
  (set ``WDM_LOOKUP_REF_H5``).

Grids are tiny (Nf=64, Nt=64, layer_dt = 3600 s like production) so this runs in
seconds on the laptop.
"""

import os
import tempfile
import unittest

import numpy as np


class WDMLookupRestoreTest(unittest.TestCase):
    def setUp(self):
        from lisatools.domains import WDMLookupTable, WDMSettings

        self.WDMLookupTable = WDMLookupTable
        self.wdm_set = WDMSettings(Nf=64, Nt=64, dt=56.25, force_backend="cpu")
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self, kind, fdot_vals, num_layers_diff=2, tag="", time_layers=16):
        norm_f, m_diffs, m_ref = self.WDMLookupTable.apply_eps_frequency(
            0.05, self.wdm_set, m_ref=20, num_layers_diff=num_layers_diff)
        return self.WDMLookupTable(
            self.wdm_set, 1, m_ref=m_ref, norm_freq_single_layer=norm_f,
            m_diffs=m_diffs, fdot_vals=fdot_vals,
            store_path=os.path.join(self.tmp.name, f"{kind}{tag}.h5"),
            batch_size_gen=8, build_kind=kind, time_layers=time_layers)

    def test_class_is_importable(self):
        self.assertTrue(hasattr(self.WDMLookupTable, "get_wdm_coeffs"))

    def test_run_fdot_property(self):
        self.assertFalse(self._build("n_ref_only", np.array([0.0])).run_fdot)

    def test_complex_build_matches_real_build_fdot_off(self):
        # Needs a long enough BUILD grid: the real build's fdot=0 sin table is an edge
        # artefact of a short build transform (max|sin|/max|cos| = 3.8e-4 at 16 build
        # layers, 2.8e-8 at 64, 4.6e-10 at 128); the complex build gives exactly 0.
        from lisatools.domains import WDMSettings

        self.wdm_set = WDMSettings(Nf=64, Nt=256, dt=56.25, force_backend="cpu")
        real = self._build("n_ref_only", np.array([0.0]), time_layers=64)
        cx = self._build("n_ref_complex", np.array([0.0]), time_layers=64)
        peak = float(np.max(np.abs(real.table_cos)))
        np.testing.assert_allclose(np.real(cx.table_cx), real.table_cos, rtol=0, atol=1e-12 * peak)
        np.testing.assert_allclose(np.imag(cx.table_cx), real.table_sin, rtol=0, atol=1e-7 * peak)

    def test_out_of_support_f_norm_raises(self):
        # num_layers_diff=0 -> m_diffs spans only [-1, 0]: the upper neighbour layer of an
        # in-band carrier falls outside the table's frequency support.
        t = self._build("n_ref_only", np.array([0.0]), num_layers_diff=0, tag="narrow")
        f = np.array([(t.m_ref + 0.3) * self.wdm_set.layer_df])
        with self.assertRaises(ValueError):
            t.get_wdm_coeffs(np.ones(1), np.zeros(1), f, np.zeros(1), np.array([t.n_ref]))

    def test_regression_against_committed_table(self):
        ref = os.environ.get("WDM_LOOKUP_REF_H5", "")
        if not ref or not os.path.exists(ref):
            self.skipTest("set WDM_LOOKUP_REF_H5=/Users/mkatz/Research/lisa_sprint_2026/"
                          "wdm_lookup_n_ref_only_NF365_NT10240_TL256.h5")
        loaded = self.WDMLookupTable.from_file(ref, force_backend="cpu")
        rebuilt = self.WDMLookupTable(
            loaded, loaded.nchannels, m_ref=loaded.m_ref,
            norm_freq_single_layer=loaded.norm_freq_single_layer, m_diffs=loaded.m_diffs,
            fdot_vals=loaded.fdot_vals, store_path=os.path.join(self.tmp.name, "re.h5"),
            batch_size_gen=20, build_kind="n_ref_only", time_layers=256)
        # Machine precision, not bit-for-bit: the table was built 2026-05-20 and the
        # FFT/BLAS stack has changed since (max |diff| 7.1e-15 on a peak of 60.4).
        peak = float(np.max(np.abs(loaded.table_cos)))
        np.testing.assert_allclose(rebuilt.table_cos, loaded.table_cos, rtol=0, atol=1e-14 * peak)
        np.testing.assert_allclose(rebuilt.table_sin, loaded.table_sin, rtol=0, atol=1e-14 * peak)


if __name__ == "__main__":
    unittest.main()
