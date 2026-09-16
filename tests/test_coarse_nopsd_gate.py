"""Coarse WDM noise runtime must SKIP, not raise, when no psd branch exists.

2026-09-16: the nogb null test (REMOVE_BRANCHES includes psd; fixed noise
params; source-only likelihood) inherits the main campaign's COARSE_* knobs
and died at launch on ``ValueError("Coarse WDM noise likelihood requires a
psd branch.")``. The coarse machinery exists only to accelerate PSD
sampling, so a psd-less composition should log and fall through to the fine
backend instead of refusing to run.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from lisatools.globalfit.run import GlobalFit

LOGGER = "lisatools.globalfit.run"


def _fit(branches, coarse_Q=4, mode="delayed_acceptance"):
    inst = object.__new__(GlobalFit)
    inst.curr = SimpleNamespace(
        general_info=SimpleNamespace(
            coarse_Q=coarse_Q,
            coarse_wdm_statistic=None,
            coarse_gpu_mode=mode,
        ),
        engine_info=SimpleNamespace(branch_names=list(branches)),
    )
    return inst


class CoarseNoPsdGateTest(unittest.TestCase):
    def test_no_psd_branch_skips_with_warning(self):
        fit = _fit(["sobbh", "mbh", "emri"])
        with self.assertLogs(LOGGER, level="WARNING") as cm:
            out = fit._prepare_coarse_wdm_runtime(state=None)
        self.assertIsNone(out)
        self.assertIn("psd", "\n".join(cm.output))

    def test_q1_still_returns_none_silently(self):
        fit = _fit(["sobbh"], coarse_Q=1)
        self.assertIsNone(fit._prepare_coarse_wdm_runtime(state=None))

    def test_psd_present_keeps_the_loud_mode_check(self):
        # With psd PRESENT and source branches at mode="off", the
        # all-source-sidecar refusal must still fire -- the new gate only
        # covers psd-LESS compositions.
        fit = _fit(["psd", "sobbh"], mode="off")
        with self.assertRaises(ValueError):
            fit._prepare_coarse_wdm_runtime(state=None)


if __name__ == "__main__":
    unittest.main()
