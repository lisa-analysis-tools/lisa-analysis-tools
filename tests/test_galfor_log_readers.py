"""Reading a store's galfor basis, and converting out of it.

Both halves of this were live defects on 2026-09-18, on the first stores
written with ``GALFOR_LOG_SAMPLING=1``:

* every consumer handed the RAW stored galfor row to the foreground model.
  Under log sampling ``amp`` and ``f_1`` arrive negative, so
  ``(f / f_1) ** alpha`` is NaN -- and NaN does not raise. It surfaced as an
  all-zero SNR / all-False truth set out of ``build_truth.py`` (which still
  passed a "psd and galfor are nonzero" sanity check) and, far downstream, a
  fatal ``KeyError: 'nbins'`` that killed ``gf_monitor_gen.py`` outright.
* the first fix attempt ALSO failed, because ``noise_model_identity`` is an
  HDF5 **group carrying one attribute per key**, not a JSON string
  attribute. A reader that assumes the JSON form finds nothing, silently
  falls back to "linear", and reinterprets the store -- the exact failure
  the identity record exists to prevent.

So the reader's shape is load-bearing, not incidental, and is pinned here.
"""

import json
import os
import tempfile
import unittest

import numpy as np

from lisatools.globalfit.stock.erebor.noise import (
    GALFOR_BASIS,
    GALFOR_LOG_PARAMS,
    galfor_params_to_physical,
    read_noise_model_identity,
)

# amp, fk, alpha, f_1, f_2 -- a realistic converged row (3mo walker 0, row 21)
PHYS = np.array([3.292e-44, 6.007e-03, 0.1182, 6.595e-03, 4.771e-03])
LOGV = np.array([np.log10(PHYS[0]), np.log10(PHYS[1]), PHYS[2],
                 np.log10(PHYS[3]), np.log10(PHYS[4])])


class BasisConstantsTest(unittest.TestCase):
    def test_alpha_is_the_only_linear_column(self):
        self.assertEqual(GALFOR_BASIS, ("amp", "fk", "alpha", "f_1", "f_2"))
        self.assertEqual(
            set(GALFOR_BASIS) - set(GALFOR_LOG_PARAMS), {"alpha"})


class ToPhysicalTest(unittest.TestCase):
    def test_linear_store_is_an_identity_copy(self):
        out = galfor_params_to_physical(PHYS, False)
        np.testing.assert_allclose(out, PHYS)

    def test_it_copies_rather_than_mutating_the_caller(self):
        src = PHYS.copy()
        galfor_params_to_physical(src, False)
        np.testing.assert_allclose(src, PHYS)
        src2 = LOGV.copy()
        galfor_params_to_physical(src2, True)
        np.testing.assert_allclose(src2, LOGV)

    def test_log_store_round_trips_to_physical(self):
        np.testing.assert_allclose(
            galfor_params_to_physical(LOGV, True), PHYS, rtol=1e-12)

    def test_alpha_is_NOT_exponentiated(self):
        """The bug in miniature: 10**0.118 = 1.31, a different model."""
        out = galfor_params_to_physical(LOGV, True)
        self.assertAlmostEqual(out[GALFOR_BASIS.index("alpha")], PHYS[2], places=12)
        self.assertNotAlmostEqual(out[2], 10.0 ** LOGV[2], places=3)

    def test_the_raw_row_would_be_NEGATIVE_where_physical_is_positive(self):
        """Why NaN rather than a crash: amp and f_1 go negative."""
        self.assertLess(LOGV[0], 0.0)
        self.assertLess(LOGV[3], 0.0)
        self.assertGreater(PHYS[0], 0.0)
        self.assertGreater(PHYS[3], 0.0)

    def test_works_on_a_stacked_chain_not_just_one_row(self):
        chain = np.stack([LOGV, LOGV * 1.01, LOGV * 0.99])      # (3, 5)
        out = galfor_params_to_physical(chain, True)
        self.assertEqual(out.shape, (3, 5))
        np.testing.assert_allclose(out[0], PHYS, rtol=1e-12)

    def test_works_on_an_iteration_walker_cube(self):
        cube = np.broadcast_to(LOGV, (7, 4, 5)).copy()
        out = galfor_params_to_physical(cube, True)
        self.assertEqual(out.shape, (7, 4, 5))
        np.testing.assert_allclose(out[3, 2], PHYS, rtol=1e-12)

    def test_a_wrong_width_is_REFUSED_not_silently_broadcast(self):
        with self.assertRaisesRegex(ValueError, "5 columns"):
            galfor_params_to_physical(np.zeros(4), True)
        with self.assertRaises(ValueError):
            galfor_params_to_physical(np.zeros((3, 2)), False)


class _Store:
    """A throwaway h5 with whatever identity layout the test wants."""

    def __init__(self, layout, payload=None):
        import h5py

        payload = {"galfor_log_sampling": True,
                   "psd_log_sampling": False,
                   "wdm_psd_method": "layer_calibrated"} if payload is None \
            else payload
        fd, self.path = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        with h5py.File(self.path, "w") as f:
            g = f.create_group("global_fit")
            if layout == "group":
                grp = g.create_group("noise_model_identity")
                for k, v in payload.items():
                    grp.attrs[k] = v
            elif layout == "json_root":
                f.attrs["noise_model_identity"] = json.dumps(payload)
            elif layout == "json_gf":
                g.attrs["noise_model_identity"] = json.dumps(payload)
            elif layout == "none":
                pass

    def __enter__(self):
        return self.path

    def __exit__(self, *a):
        try:
            os.unlink(self.path)
        except OSError:
            pass


class IdentityReaderTest(unittest.TestCase):
    """The GROUP layout is what production actually writes."""

    def test_group_layout_is_read(self):
        with _Store("group") as p:
            ident = read_noise_model_identity(p)
        self.assertTrue(ident["galfor_log_sampling"])
        self.assertFalse(ident["psd_log_sampling"])
        self.assertEqual(ident["wdm_psd_method"], "layer_calibrated")

    def test_numpy_bools_come_back_as_plain_bools(self):
        """h5py hands back np.True_; `is True` checks downstream must work."""
        with _Store("group") as p:
            v = read_noise_model_identity(p)["galfor_log_sampling"]
        self.assertIsInstance(v, (bool, np.bool_))
        self.assertTrue(bool(v))

    def test_json_attribute_fallbacks_still_work(self):
        for layout in ("json_root", "json_gf"):
            with _Store(layout) as p:
                self.assertTrue(
                    read_noise_model_identity(p)["galfor_log_sampling"], layout)

    def test_a_store_with_no_identity_returns_EMPTY_not_false(self):
        """{} means 'defaults'. A caller must not read it as 'all False'
        without saying so -- that is how a basis gets reinterpreted."""
        with _Store("none") as p:
            self.assertEqual(read_noise_model_identity(p), {})

    def test_an_unreadable_path_returns_empty_rather_than_raising(self):
        self.assertEqual(read_noise_model_identity("/nonexistent/x.h5"), {})

    def test_the_end_to_end_decision_a_consumer_makes(self):
        with _Store("group") as p:
            log = bool(read_noise_model_identity(p).get(
                "galfor_log_sampling", False))
            np.testing.assert_allclose(
                galfor_params_to_physical(LOGV, log), PHYS, rtol=1e-12)
        with _Store("group", payload={"galfor_log_sampling": False}) as p:
            log = bool(read_noise_model_identity(p).get(
                "galfor_log_sampling", False))
            np.testing.assert_allclose(
                galfor_params_to_physical(PHYS, log), PHYS)


class AgreesWithTheStockTransformTest(unittest.TestCase):
    """The conversion must match the sampler's own transform container."""

    def test_matches_make_galfor_log_transform_container(self):
        from lisatools.globalfit.stock.erebor.noise import (
            make_galfor_log_transform_container)

        tc = make_galfor_log_transform_container()
        got = tc.transform_base_parameters(LOGV.copy()[None, :]).squeeze()
        np.testing.assert_allclose(
            galfor_params_to_physical(LOGV, True), got, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
