"""Wiring tests for PINNING the galactic foreground while psd and gb stay sampled.

``GALFOR_FIXED_PARAMS`` has existed for a while, but ``fixed_psd_kwargs`` was
read by exactly ONE branch of ``run.py::setup_acs`` -- the path taken when
there is no ``psd`` branch at all. And ``REMOVE_BRANCHES=psd`` is hard-guarded
to also require removing ``gb``, so the pin was unreachable in every
configuration where it would be useful.

Why it is useful (measured on the 6mo run, 2026-09-23): galfor absorbs the
galaxy the GB search has not yet resolved, and its excess IS the search's own
noise floor in that band -- 2x too high at 3.5 mHz, SNR x0.71, ~745 findable
binaries hidden. That is a closed loop: galfor is high because GB has not
found them, and GB cannot find them because galfor is high. Pinning the
foreground for a gb_search stage breaks it.

The contract these tests hold:

1. **Unset is a strict no-op.** Both auto-population sites
   (``engine.py::init_data_information``, ``stock/erebor/fit.py``) write
   ``galfor_params=None`` explicitly, so the ``run.py`` fallback resolves to
   ``None`` and the behaviour is byte-identical to before.
2. ``REMOVE_BRANCHES=galfor`` + ``GALFOR_FIXED_PARAMS`` leaves psd and gb
   SAMPLED while the foreground is pinned, in the physical linear basis with
   no transform applied.
3. The galfor branch really is gone from every stage's move list.

Construction level only -- no data build, no materialization (the
test_staged_sources_wiring pattern).
"""
import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np

SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "fstat_proposal" / "run_combined_staged.py")

# The 6mo freeze vector: the add-back measurement of the six-month confusion,
# amp modulation-corrected (/1.1031, since the run applies M(t) on top) and
# padded x1.3 because the add-back is a lower bound. See
# galfor_6mo_figs/RUNBOOK.md.
PIN = "1.530919e-44,2.134147e-03,3.442680,8.329708e-02,8.009456e-04"
PIN_VEC = [1.530919e-44, 2.134147e-03, 3.442680, 8.329708e-02, 8.009456e-04]

CLEARED = ("MBHB_IDS", "EMRI_IDS", "SOBHB_IDS", "SOURCE_TYPES", "GB_ONLY",
           "STAGE_SKIP_NOISE", "STAGE_SKIP_SOURCE_SEARCH", "STAGE_NOISE_ONLY",
           "STAGE_NOISE_VGB_PE", "COMBINED_SMOKE", "TOBS_TARGET",
           "GB_WARM_START_COMPONENTS", "REMOVE_BRANCHES",
           "VGB_CHIRP_MASS_BASIS", "GALFOR_FIXED_PARAMS", "PSD_FIXED_PARAMS")


def _build_fit():
    spec = importlib.util.spec_from_file_location(
        "run_combined_staged_for_galfor_pin_test", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_fit()


def _resolve(fixed_psd_kwargs):
    """The exact fallback run.py::setup_acs uses when there is no galfor branch."""
    return (fixed_psd_kwargs or {}).get("galfor_params")


class GalforPinWiringTest(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.copy()
        for k in CLEARED:
            os.environ.pop(k, None)
        os.environ["NWALKERS"] = "4"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_unset_is_a_no_op(self):
        """No GALFOR_FIXED_PARAMS -> the fallback resolves to None, as before."""
        fit = _build_fit()
        self.assertIn("galfor", fit.branches)
        self.assertIsNone(_resolve(fit.general.fixed_psd_kwargs))

    def test_auto_populated_kwargs_still_resolve_to_none(self):
        """The two auto-population shapes must not accidentally arm the pin.

        This is the guard that keeps the run.py fallback a no-op: if either
        site ever stopped writing an explicit ``galfor_params=None``, an
        unrelated run with the galfor branch removed would silently acquire a
        foreground.
        """
        for shape in (None, {}, {"psd_params": [15e-12, 3e-15], "galfor_params": None}):
            with self.subTest(shape=shape):
                self.assertIsNone(_resolve(shape))

    def test_pin_with_psd_and_gb_still_sampled(self):
        """The configuration the loop-breaker needs: only galfor leaves."""
        os.environ["REMOVE_BRANCHES"] = "galfor"
        os.environ["GALFOR_FIXED_PARAMS"] = PIN
        fit = _build_fit()

        self.assertNotIn("galfor", fit.branches)
        # the whole point: these two keep sampling
        self.assertIn("psd", fit.branches)
        self.assertIn("gb", fit.branches)

        pinned = _resolve(fit.general.fixed_psd_kwargs)
        self.assertIsNotNone(pinned)
        np.testing.assert_allclose(np.asarray(pinned, dtype=float), PIN_VEC, rtol=0, atol=0)

    def test_pin_is_physical_basis_untransformed(self):
        """The fixed path applies NO transform, so what goes in is what is used.

        A log-sampled galfor branch would store log10 coordinates; the pin must
        NOT be given in that basis. Asserted by value: a physical amp is ~1e-44,
        never ~-43.8.
        """
        os.environ["REMOVE_BRANCHES"] = "galfor"
        os.environ["GALFOR_FIXED_PARAMS"] = PIN
        os.environ["GALFOR_LOG_SAMPLING"] = "1"
        try:
            fit = _build_fit()
            pinned = np.asarray(_resolve(fit.general.fixed_psd_kwargs), dtype=float)
        finally:
            os.environ.pop("GALFOR_LOG_SAMPLING", None)
        self.assertGreater(pinned[0], 0.0)          # linear amp, not log10
        self.assertLess(pinned[0], 1e-40)

    def test_galfor_move_is_gone_from_every_stage(self):
        os.environ["REMOVE_BRANCHES"] = "galfor"
        os.environ["GALFOR_FIXED_PARAMS"] = PIN
        fit = _build_fit()
        for st in fit.recipe.stages:
            for mv in st.moves:
                self.assertNotEqual(
                    mv.name, "galfor_pe",
                    msg=f"galfor_pe survived in stage {st.name!r}")

    def test_bad_value_is_refused_loudly(self):
        os.environ["REMOVE_BRANCHES"] = "galfor"
        os.environ["GALFOR_FIXED_PARAMS"] = "1.5e-44,not-a-number,3.4,0.08,8e-4"
        with self.assertRaises(ValueError):
            _build_fit()


if __name__ == "__main__":
    unittest.main()
