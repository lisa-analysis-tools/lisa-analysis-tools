"""The GB leaf-cap gate's patience is walker-count invariant.

The default cap gate is MAX over cold walkers: a band's cap holds while
the BEST walker keeps improving it by D/2, and increments once no walker
has for ``leaf_cap_min_iters`` consecutive iterations. With W walkers the
max gets W independent chances per iteration, so the same iteration count
is much weaker evidence of a plateau at small W and the cap ratchets
FASTER the fewer walkers run -- backwards, since fewer walkers search
less.

Measured on the 4-walker production run (2026-09-18, iterations 35 to
68): summed cap 1232 -> 2373 while the cold chain held 420 -> 1066
leaves, leaving 1307 cap slots unused, with 202 bands already at cap >= 4
and a max of 7 while the median band still held one source. The knob was
only ever tuned at 10 and 24 walkers.

``_leaf_cap_patience`` scales the configured value by
``ref_walkers / W``, one-sided, so runs at or above the reference count
are bit-identical and only small-walker runs become more patient.
"""

import os
import unittest
from unittest import mock

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialStretchMove


def _move(min_iters=3, name="rj_fstat_search"):
    mv = GBSpecialStretchMove.__new__(GBSpecialStretchMove)
    mv.leaf_cap_min_iters = int(min_iters)
    mv.name = name
    return mv


def _env(**kw):
    return mock.patch.dict(os.environ, {k: str(v) for k, v in kw.items()})


class LeafCapPatienceTest(unittest.TestCase):
    def test_default_is_OFF_so_nothing_changes_silently(self):
        """This gate drives RJ births: it must be opted into, not inherited.

        The campaign script sets the reference count alongside its other
        cap knobs; the gate's own mechanics tests drive it at 1-2 walkers
        and must not pick up a policy layer from the environment.
        """
        with mock.patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop("GB_LEAF_CAP_MIN_ITERS_REF_WALKERS", None)
            self.assertEqual(_move(3)._leaf_cap_patience(1), 3)
            self.assertEqual(_move(3)._leaf_cap_patience(4), 3)

    def test_reference_walker_count_is_unchanged(self):
        with _env(GB_LEAF_CAP_MIN_ITERS_REF_WALKERS=10):
            self.assertEqual(_move(3)._leaf_cap_patience(10), 3)

    def test_more_walkers_than_reference_is_unchanged(self):
        """One-sided: a 24-walker run must stay bit-identical."""
        with _env(GB_LEAF_CAP_MIN_ITERS_REF_WALKERS=10):
            self.assertEqual(_move(3)._leaf_cap_patience(24), 3)
            self.assertEqual(_move(5)._leaf_cap_patience(24), 5)

    def test_four_walkers_get_the_tuned_evidence(self):
        """3 iterations x 10 walkers = 30 walker-iterations; at 4 -> 8."""
        with _env(GB_LEAF_CAP_MIN_ITERS_REF_WALKERS=10):
            self.assertEqual(_move(3)._leaf_cap_patience(4), 8)

    def test_one_walker(self):
        with _env(GB_LEAF_CAP_MIN_ITERS_REF_WALKERS=10):
            self.assertEqual(_move(3)._leaf_cap_patience(1), 30)

    def test_walker_iterations_are_held_at_or_above_the_reference(self):
        with _env(GB_LEAF_CAP_MIN_ITERS_REF_WALKERS=10):
            for w in (1, 2, 3, 4, 5, 8, 9):
                got = _move(3)._leaf_cap_patience(w)
                self.assertGreaterEqual(
                    got * w, 3 * 10,
                    f"W={w}: {got} iterations buys {got * w} walker-iterations, "
                    f"short of the reference 30")

    def test_scaling_disabled_by_zero(self):
        with _env(GB_LEAF_CAP_MIN_ITERS_REF_WALKERS=0):
            self.assertEqual(_move(3)._leaf_cap_patience(4), 3)
            self.assertEqual(_move(3)._leaf_cap_patience(1), 3)

    def test_reference_is_configurable(self):
        with _env(GB_LEAF_CAP_MIN_ITERS_REF_WALKERS=4):
            self.assertEqual(_move(3)._leaf_cap_patience(4), 3)
            self.assertEqual(_move(3)._leaf_cap_patience(2), 6)

    def test_degenerate_walker_counts_do_not_raise(self):
        with _env(GB_LEAF_CAP_MIN_ITERS_REF_WALKERS=10):
            self.assertEqual(_move(3)._leaf_cap_patience(0), 3)
            self.assertEqual(_move(3)._leaf_cap_patience(-1), 3)

    def test_logs_once_and_names_the_walker_count(self):
        mv = _move(3)
        with _env(GB_LEAF_CAP_MIN_ITERS_REF_WALKERS=10):
            with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch", level="INFO"
            ) as cm:
                mv._leaf_cap_patience(4)
            msg = "\n".join(cm.output)
            self.assertIn("GB_CAP_PATIENCE", msg)
            self.assertIn("3 -> 8", msg)
            self.assertIn("4 cold walker", msg)
            # second call is silent
            mv._leaf_cap_patience(4)
            self.assertEqual(len(cm.output), 1)


if __name__ == "__main__":
    unittest.main()
