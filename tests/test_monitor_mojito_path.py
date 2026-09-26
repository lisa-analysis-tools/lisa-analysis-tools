"""One mojito path for the monitor page, and a truth set built on the
LATEST sample.

Two user rulings from 2026-09-26, both about defaults that were silently
wrong rather than loudly broken:

  * ``MOJITO_INFO_PATH`` -- "the one path you need for the mojito data [is]
    the folder that contains catalogues/ and data/". The page used to need
    two variables for that one directory, resolved by different chains,
    and the second was never exported on the cluster: a page built there
    lost the residual-spectrum and data/template/residual panels and said
    nothing.
  * ``build_truth.py --iteration`` defaults to the newest usable sample
    instead of a hardcoded 78. On the 6mo v9 store, ``log_like`` has
    capacity 2003 with FOUR filled rows, so 78 read a zero row -- an
    all-zero PSD, a NaN sensitivity, and a silently all-False ``det``.

``gf_monitor_gen.py`` cannot be imported (it is a standalone generator
that ``raise SystemExit(0)`` on import), so its half is pinned at the
source level. ``build_truth.py`` imports fine and gets real behaviour
tests against a synthetic store.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

# Both moved into the package 2026-09-26 (scripts/diagnostics/*.py are now
# thin shims). Read the REAL files, not the shims, or every source-level
# assertion below passes vacuously against a 30-line forwarder.
from lisatools.globalfit import monitor as _mon  # noqa: E402
from lisatools.globalfit.monitor import build_truth as bt  # noqa: E402


def _monitor_src():
    return Path(_mon.generator_path()).read_text()


class MojitoInfoPathTest(unittest.TestCase):
    """The single-knob contract, pinned at the source level."""

    def test_the_knob_exists_and_is_resolved_once(self):
        src = _monitor_src()
        self.assertIn('os.environ.get("MOJITO_INFO_PATH")', src)
        self.assertEqual(src.count("MOJITO_INFO_PATH = "), 1)

    def test_it_is_defined_before_BOTH_consumers(self):
        """Module-level script: a helper defined after its use is a
        NameError at render time, not an import error a test would see."""
        src = _monitor_src()
        define = src.index("MOJITO_INFO_PATH = _resolve_mojito_info_path()")
        self.assertLess(define, src.index("SOMS_INJ, SA_INJ = psd_truth_levels"))
        self.assertLess(define, src.index("def _resolve_mojito_cat_dir"))

    def test_the_psd_lookup_prefers_it_over_the_legacy_var(self):
        src = _monitor_src()
        i = src.index("SOMS_INJ, SA_INJ = psd_truth_levels")
        self.assertIn(
            'MOJITO_INFO_PATH or os.environ.get("MOJITO_DATA_PATH")',
            src[i:i + 700])

    def test_the_catalogue_lookup_prefers_it_over_the_legacy_vars(self):
        """It must win over MOJITO_CAT, which is the whole point -- the two
        used to disagree about the same directory."""
        src = _monitor_src()
        i = src.index("def _resolve_mojito_cat_dir")
        body = src[i:i + 900]
        self.assertLess(body.index("MOJITO_INFO_PATH"),
                        body.index('os.environ.get("MOJITO_CAT")'))

    def test_it_validates_BOTH_subdirectories(self):
        """'the folder that contains catalogues/ and data/' is the
        contract; accepting a path with only one silently half-works."""
        src = _monitor_src()
        self.assertIn('_MOJITO_SUBDIRS = ("catalogues", "data")', src)

    def test_a_bad_path_reaches_the_page_not_just_the_log(self):
        """MISSING is what the page renders. A warning that only goes to
        stderr is lost the moment the operator closes the terminal."""
        src = _monitor_src()
        i = src.index("def _resolve_mojito_info_path")
        self.assertIn("MISSING.append", src[i:i + 2000])

    def test_the_silent_analytic_fallback_is_now_announced(self):
        """psd_truth_levels returns the round injection and says nothing;
        with neither variable set the page has to say it instead."""
        src = _monitor_src()
        i = src.index("SOMS_INJ, SA_INJ = psd_truth_levels")
        self.assertIn("MISSING.append", src[i:i + 1200])

    def test_the_launcher_exports_it(self):
        sh = (Path(__file__).resolve().parents[1] / "scripts"
              / "fstat_proposal" / "submit_gf_6mo_v9_4gpu.sh").read_text()
        self.assertIn("export MOJITO_INFO_PATH=/shared/data/mojito_cache", sh)
        # Same directory as the path the sampler itself loads bricks from;
        # if these ever diverge the page is describing a different dataset.
        self.assertIn("export MOJITO_DATA_PATH=/shared/data/mojito_cache", sh)


def _make_store(path, capacity=50, filled=6, it_attr=None,
                torn_noise_rows=0):
    """A store shaped like the real one: preallocated, partly filled."""
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        ll = np.zeros((capacity, 1, 1, 4))
        ll[:filled] = -1.0e5
        g.create_dataset("log_like", data=ll)
        c = g.create_group("chain")
        psd = np.zeros((capacity, 1, 1, 4, 1, 2))
        gal = np.zeros((capacity, 1, 1, 4, 1, 5))
        good = filled - torn_noise_rows
        psd[:good] = 1.5e-11
        gal[:good] = 1.0e-44
        c.create_dataset("psd", data=psd)
        c.create_dataset("galfor", data=gal)
        if it_attr is not None:
            g.attrs["iteration"] = it_attr
    return path


class LatestIterationTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def _store(self, name, **kw):
        return _make_store(os.path.join(self.d, name), **kw)

    def test_it_returns_the_last_FILLED_row_not_the_capacity(self):
        """The bug the old default had: capacity 2003, 4 real rows."""
        self.assertEqual(bt.latest_iteration(
            self._store("a.h5", capacity=2003, filled=4)), 3)

    def test_a_rewound_store_ignores_the_stale_rows_past_the_attr(self):
        """reset_recipe_stage moves the attr back and leaves the discarded
        trajectory in place until the next grow() truncates it."""
        self.assertEqual(bt.latest_iteration(
            self._store("b.h5", filled=20, it_attr=12)), 11)

    def test_an_attr_AHEAD_of_the_filled_rows_does_not_win(self):
        """Only a LOWER attr means a rewind; a higher one means the row is
        mid-write, and indexing it would read zeros."""
        self.assertEqual(bt.latest_iteration(
            self._store("c.h5", filled=6, it_attr=99)), 5)

    def test_a_torn_last_row_steps_back_to_the_last_written_noise(self):
        """log_like lands before chain/psd within one ~20 s save step."""
        self.assertEqual(bt.latest_iteration(
            self._store("d.h5", filled=10, torn_noise_rows=2)), 7)

    def test_an_empty_store_raises_rather_than_guessing(self):
        with self.assertRaises(RuntimeError):
            bt.latest_iteration(self._store("e.h5", filled=0))

    def test_noise_never_written_raises_rather_than_returning_zeros(self):
        """An all-zero PSD does not crash downstream -- it yields a NaN
        sensitivity and an all-False det, i.e. an empty truth set that
        still passes a 'file exists' check. Fail here instead."""
        with self.assertRaises(RuntimeError):
            bt.latest_iteration(
                self._store("f.h5", filled=8, torn_noise_rows=8))


class BuildTruthIterationDefaultTest(unittest.TestCase):
    """The CLI contract around the new default."""

    @staticmethod
    def _iteration_for(argv, env_value=None):
        """What ``--iteration`` resolves to, through the REAL parser.

        ``make_parser()`` is called inside the patched environment because
        the default is evaluated at ``add_argument`` time; ``main()`` is
        never called, since it would run a multi-minute build.
        """
        env = {} if env_value is None else {"ITERATION": env_value}
        with mock.patch.dict(os.environ, env, clear=False):
            if env_value is None:
                os.environ.pop("ITERATION", None)
            return bt.make_parser().parse_args(
                ["store.h5"] + list(argv)).iteration

    def test_the_hardcoded_78_is_gone(self):
        src = (Path(bt.__file__)).read_text()
        self.assertNotIn('"--iteration", type=int, default=78', src)

    def test_None_means_latest_and_the_RESOLVED_value_is_stamped(self):
        """A truth set stamped iteration=None cannot be audited against
        the store it came from."""
        src = (Path(bt.__file__)).read_text()
        self.assertIn("it_used = (latest_iteration(a.store) if a.iteration "
                      "is None", src)
        self.assertIn("iteration=np.array(it_used)", src)
        self.assertNotIn("iteration=np.array(a.iteration)", src)

    def test_fitted_noise_is_called_with_the_resolved_iteration(self):
        src = (Path(bt.__file__)).read_text()
        self.assertIn("fitted_noise(a.store, it_used)", src)

    def test_the_ITERATION_env_var_is_honoured(self):
        self.assertEqual(self._iteration_for([], env_value="42"), 42)

    def test_an_explicit_flag_beats_the_env_var(self):
        self.assertEqual(
            self._iteration_for(["--iteration", "7"], env_value="42"), 7)

    def test_no_env_and_no_flag_is_None(self):
        """None is the sentinel main() turns into latest_iteration()."""
        self.assertIsNone(self._iteration_for([]))


if __name__ == "__main__":
    unittest.main()
