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


class L1BrickLookupTest(unittest.TestCase):
    """USER REPORT 2026-09-26: "I cannot get the html builder to find the
    proper orbits. Even when I provide MOJITO_CAT in both ways it does not
    work."

    Root cause: ``find_l1_brick`` read a DIFFERENT env chain from
    everything else -- only ``MOJITO_DATA_PATH`` and ``~/.mojito_cache``.
    ``MOJITO_CAT``, the variable this script documents and the obvious one
    to reach for, resolved the CATALOGUE correctly and did nothing for the
    ORBITS, so the build fell back to the analytic ephemeris behind a
    single print line. The truth set came out stamped ``analytic`` and the
    monitor complained about it days later, naming neither knob.

    Above a few mHz that fallback is not a refinement: optimal SNR
    7.96 -> 8.69 and template overlap 0.052 -> 0.580 at 7.44-7.60 mHz.
    """

    def setUp(self):
        self.d = tempfile.mkdtemp()
        # build_truth's own nested convention, which is where the two
        # chains disagreed: MOJITO_CAT points at the inner directory.
        self.inner = os.path.join(self.d, "brickmarket", "mojito_light_v1_0_0")
        self.brick = os.path.join(self.inner, "data", "INSTRUMENT", "L1",
                                  "NOISE_731d_2.5s_L1_source0_0_X.h5")
        os.makedirs(os.path.dirname(self.brick))
        open(self.brick, "wb").close()
        self.catfile = os.path.join(self.inner, "catalogues",
                                    "wdwd_cat_mojito_lite_processed.hdf5")
        os.makedirs(os.path.dirname(self.catfile))
        open(self.catfile, "wb").close()
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for k in ("MOJITO_INFO_PATH", "MOJITO_CAT", "MOJITO_DATA_PATH",
                  "MOJITO_CACHE_DIR", "HOME"):
            os.environ.pop(k, None)
        # Neutralise the ~/.mojito_cache default so a real laptop cache
        # cannot make these pass for the wrong reason.
        os.environ["HOME"] = self.d + "/nohome"
        self.addCleanup(self.env.stop)

    def test_nothing_set_finds_nothing(self):
        """The control: without it these assertions are vacuous."""
        self.assertIsNone(bt.find_l1_brick())

    def test_MOJITO_CAT_as_a_DIRECTORY_now_works(self):
        os.environ["MOJITO_CAT"] = self.inner
        self.assertEqual(bt.find_l1_brick(), self.brick)

    def test_MOJITO_CAT_as_the_CATALOGUE_FILE_now_works(self):
        """"both ways" -- MOJITO_CAT accepts the wdwd_cat file, and the
        brick root is then two levels up, exactly as _resolve_catalogue
        treats it."""
        os.environ["MOJITO_CAT"] = self.catfile
        self.assertEqual(bt.find_l1_brick(), self.brick)

    def test_MOJITO_INFO_PATH_works(self):
        os.environ["MOJITO_INFO_PATH"] = self.inner
        self.assertEqual(bt.find_l1_brick(), self.brick)

    def test_MOJITO_DATA_PATH_still_works(self):
        """The one name that worked before must not regress."""
        os.environ["MOJITO_DATA_PATH"] = self.inner
        self.assertEqual(bt.find_l1_brick(), self.brick)

    def test_MOJITO_CACHE_DIR_works(self):
        os.environ["MOJITO_CACHE_DIR"] = self.d
        self.assertEqual(bt.find_l1_brick(), self.brick)

    def test_the_search_is_recursive_from_the_cache_root(self):
        """Pointing at the OUTER cache dir must still find it -- the
        laptop layout nests two levels deeper than the cluster's."""
        os.environ["MOJITO_CAT"] = self.d
        self.assertEqual(bt.find_l1_brick(), self.brick)

    def test_an_explicit_l1_brick_beats_every_env(self):
        os.environ["MOJITO_CAT"] = self.inner
        self.assertEqual(bt.find_l1_brick("/explicit/any_L1_file.h5"),
                         "/explicit/any_L1_file.h5")

    def test_the_CLI_exposes_the_override(self):
        ap = bt.make_parser()
        ns = ap.parse_args(["store.h5", "--l1-brick", "/x/y_L1_z.h5"])
        self.assertEqual(ns.l1_brick, "/x/y_L1_z.h5")

    def test_main_passes_the_override_through(self):
        src = Path(bt.__file__).read_text()
        self.assertIn("l1_orbits(path=a.l1_brick)", src)

    def test_the_analytic_fallback_is_loud_and_names_the_knobs(self):
        """It was one print line in a tens-of-minutes build."""
        src = Path(bt.__file__).read_text()
        i = src.index("orbits_tag = \"analytic\"")
        blk = src[i:i + 2600]
        self.assertIn("####", blk)
        for knob in ("MOJITO_INFO_PATH", "MOJITO_CAT", "MOJITO_DATA_PATH",
                     "--l1-brick"):
            self.assertIn(knob, blk)

    def test_requesting_analytic_explicitly_does_not_shout(self):
        """--analytic-orbits is a choice, not an accident."""
        src = Path(bt.__file__).read_text()
        i = src.index("orbits_tag = \"analytic\"")
        self.assertIn("if a.analytic_orbits:", src[i:i + 300])
