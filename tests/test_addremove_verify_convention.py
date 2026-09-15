"""`_verify_prev_logl` must compare MATCHING lnL conventions.

Verified 2026-09-14 (6mo campaign, job 487): the base check re-scored the
current points with a FORCED ``source_only=True`` while ``prev_logl``
carries the full noise-included lnL, so its "difference" was algebraically
the per-walker noise term -- offset ~1e8, "spread" ~2e6 = the WALKER-TO-
WALKER noise variation pooled across the (temps x walkers) grid -- and the
warning fired 8x on a healthy run (the convention-matched twin
``_verify_entry_vs_acs`` stayed silent at a 0.1-lnL gate; SOBBH already
overrides this exact check for this exact reason). Under
``{BRANCH}_CHECK_LL=strict`` the false positive would even RAISE.

The fix: recompute with the move's own ``waveform_like_kwargs`` verbatim
(the same convention ``compute_like`` applies), and gate on the WITHIN-
walker spread -- per-walker constants (any remaining convention offset,
e.g. the DCGA replica path's source-only lnL) cancel inside each walker's
column, while genuine residual/scoring drift within a walker still fires.
"""

import logging
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.addremovemove import (
    ResidualAddOneRemoveOneMove,
)

LOGGER = "lisatools.globalfit.moves.addremovemove"
NT, NW = 2, 10


def _stub(acs_like, like_kwargs=None, mode="warn"):
    import types

    s = types.SimpleNamespace(
        branch_name="mbh",
        check_ll_mode=mode,
        waveform_like_kwargs=dict(like_kwargs or {}),
        waveform_gen=object(),
        compute_acs_like=mock.Mock(return_value=np.asarray(acs_like)),
    )
    s._verify_prev_logl = (
        ResidualAddOneRemoveOneMove._verify_prev_logl.__get__(s)
    )
    return s


def _prev():
    rng = np.random.default_rng(5)
    return 1.0e8 + rng.normal(0.0, 1.0e6, (NT, NW))


class ConventionTest(unittest.TestCase):
    def test_recompute_never_forces_source_only(self):
        prev = _prev()
        s = _stub(prev.copy(), like_kwargs={"phase_marginalize": True})
        s._verify_prev_logl(prev, "coords", "idx", leaf=3)
        kwargs = s.compute_acs_like.call_args.kwargs
        self.assertNotIn("source_only", kwargs)
        self.assertTrue(kwargs.get("phase_marginalize"))

    def test_per_walker_constant_offset_is_silent(self):
        # the exact job-487 signature: a huge per-walker constant (the
        # walker-indexed noise term), identical across temperatures --
        # 1e8-scale offsets with a ~2e6 walker-to-walker spread.
        prev = _prev()
        noise_term = 1.0e8 + np.linspace(0.0, 2.0e6, NW)
        s = _stub(prev - noise_term[None, :])
        with self.assertNoLogs(LOGGER, level=logging.WARNING):
            s._verify_prev_logl(prev, "coords", "idx", leaf=0)

    def test_within_walker_drift_warns(self):
        prev = _prev()
        acs = prev.copy()
        acs[1, 4] += 7.5  # one rung of one walker drifts
        s = _stub(acs)
        with self.assertLogs(LOGGER, level=logging.WARNING) as cm:
            s._verify_prev_logl(prev, "coords", "idx", leaf=1)
        self.assertIn("within-walker", "\n".join(cm.output).lower())

    def test_within_walker_drift_raises_in_strict(self):
        prev = _prev()
        acs = prev.copy()
        acs[0, 2] -= 3.0
        s = _stub(acs, mode="strict")
        with self.assertRaises(ValueError):
            s._verify_prev_logl(prev, "coords", "idx", leaf=2)

    def test_invalid_points_are_ignored(self):
        prev = _prev()
        acs = prev.copy()
        prev[0, 0] = -1e300  # rejected/unset point must not poison it
        acs[1, 3] = np.nan
        s = _stub(acs)
        with self.assertNoLogs(LOGGER, level=logging.WARNING):
            s._verify_prev_logl(prev, "coords", "idx", leaf=4)


if __name__ == "__main__":
    unittest.main()
