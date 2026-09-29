"""A DAMAGED recipe group must not read as a missing recipe step.

2026-09-29, 3-month job 666. The run resumed, loaded every chain, every
sub-backend and reached ``initial log likelihood`` -- then died at
startup in ``add_recipe``:

    File ".../hdfbackend.py", line 970, in add_recipe
        recipe_step_group = recipe_group[key]
    KeyError: 'Unable to synchronously open object (message not aligned)'

``message not aligned`` is HDF5 reporting a MALFORMED OBJECT HEADER --
file damage -- but h5py surfaces it as ``KeyError``, which reads as "that
recipe step is not in the store" and sends the reader hunting for a
recipe bug that does not exist. The ``assert key in recipe_group`` on the
line above had already PASSED, because that tests the link, while opening
the group reads the header.

Why this group and not the chains: its attrs (``status``, ``order num``,
``completed_iteration``, ``start_iteration``) are rewritten on every stage
transition, and an attr write rewrites the object header. A job killed or
MPI-aborted in that window leaves the header torn. Everything else in
that store was readable.

So the fix is diagnostic, not behavioural: say DAMAGE, and name the
recovery point.
"""
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from lisatools.globalfit.hdfbackend import GFHDFBackend


class _TornGroup:
    """Link resolves, object does not -- a torn HDF5 object header."""

    def __init__(self, keys, exc):
        self._keys, self._exc = set(keys), exc

    def __contains__(self, key):
        return key in self._keys          # the link is intact

    def __getitem__(self, key):
        raise self._exc                   # the header is not


class _HealthyGroup(dict):
    pass


def _backend(group):
    be = GFHDFBackend.__new__(GFHDFBackend)
    be.name = "global_fit"
    be.filename = "/runs/gf_prod_3mo_v9_2gpu/gf_prod_3mo_testing.h5"

    @contextmanager
    def _open(mode="r"):
        yield {"global_fit": {"recipe": group}}

    be.open = _open
    return be


def _recipe(*names):
    return SimpleNamespace(recipe=[{"name": n} for n in names])


class AddRecipeOnADamagedStoreTest(unittest.TestCase):
    def setUp(self):
        self._p = mock.patch.object(
            GFHDFBackend, "has_recipe", property(lambda self: True))
        self._p.start()
        self.addCleanup(self._p.stop)

    def test_the_production_KeyError_becomes_a_DAMAGE_report(self):
        """THE REGRESSION: job 666's exact exception."""
        exc = KeyError(
            "Unable to synchronously open object (message not aligned)")
        be = _backend(_TornGroup(["noise_search", "gb_search_1"], exc))
        with self.assertRaises(RuntimeError) as cm:
            be.add_recipe(_recipe("noise_search", "gb_search_1"))
        msg = str(cm.exception)
        self.assertIn("CANNOT BE READ", msg)
        self.assertIn("DAMAGE", msg)
        self.assertIn("noise_search", msg)
        # the recovery point must be NAMED, not implied
        self.assertIn("gf_prod_3mo_testing_running_backup_copy.h5", msg)
        # and the original is kept for forensics
        self.assertIsInstance(cm.exception.__cause__, KeyError)

    def test_an_OSError_from_the_same_damage_is_caught_too(self):
        """h5py raises OSError rather than KeyError for some torn reads."""
        be = _backend(_TornGroup(["noise_search"], OSError("truncated file")))
        with self.assertRaises(RuntimeError) as cm:
            be.add_recipe(_recipe("noise_search"))
        self.assertIn("DAMAGE", str(cm.exception))

    def test_a_GENUINELY_missing_step_still_trips_the_assert(self):
        """Do not swallow the real 'this store predates that stage' case:
        a missing LINK is a different problem with a different fix."""
        steps = _HealthyGroup({
            "noise_search": SimpleNamespace(
                attrs={"status": True, "order num": 1}),
        })                                    # no gb_search_9 LINK at all
        be = _backend(steps)
        with self.assertRaises(AssertionError):
            be.add_recipe(_recipe("noise_search", "gb_search_9"))

    def test_a_healthy_store_is_untouched(self):
        steps = _HealthyGroup({
            "noise_search": SimpleNamespace(
                attrs={"status": True, "order num": 1}),
            "gb_search_1": SimpleNamespace(
                attrs={"status": False, "order num": 2}),
        })
        be = _backend(steps)
        rec = _recipe("noise_search", "gb_search_1")
        be.add_recipe(rec)
        self.assertEqual([s["status"] for s in rec.recipe], [True, False])


if __name__ == "__main__":
    unittest.main()
